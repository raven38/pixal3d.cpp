#include "uv_bake.h"
#include "xatlas.h"
#include "meshoptimizer.h"
#include "Simplify.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "parallel_for.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstring>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>
#include <memory>
#include <queue>
#include <unordered_map>
#include <array>
#include <cstdarg>

// TRELLIS_DBG_POST=1: per-phase wall times of the postprocess to stderr (debug logging only).
static double post_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void post_tick(const char* tag, double& t) {
    static const bool on = std::getenv("TRELLIS_DBG_POST") != nullptr;
    if (!on) return;
    const double n = post_now();
    fprintf(stderr, "    [post] %-28s %.2fs\n", tag, n - t);
    t = n;
}

// TRELLIS_DBG_XATLAS=1: route xatlas's verbose log (and its XA_PROFILE timings when compiled in) to stderr.
static int xatlas_print_stderr(const char* fmt, ...) { va_list ap; va_start(ap, fmt); int r = vfprintf(stderr, fmt, ap); va_end(ap); return r; }

namespace trellis {

namespace {
// Trilinear sampler over the sparse voxel PBR field. Missing corner voxels
// drop out of the weighted sum (renormalized); a texel with no populated
// corner stays unwritten and is filled by seam dilation, matching the
// reference's inpaint step.
struct VoxSampler {
    std::unordered_map<uint64_t, int> map;
    const std::vector<float>* feats;
    int res;
    const TriBvh* snap;
    static uint64_t key(int x, int y, int z) {
        return ((uint64_t)(uint32_t)x << 40) | ((uint64_t)(uint32_t)y << 20) | (uint32_t)z;
    }
    explicit VoxSampler(const VoxelPbr& v) : feats(v.feats), res(v.res), snap(v.snap) {
        map.reserve(v.coords->size() * 2);
        for (size_t i = 0; i < v.coords->size(); ++i) {
            const auto& c = (*v.coords)[i];
            map[key(c[0], c[1], c[2])] = (int)i;
        }
    }
    bool trilinear(const float p[3], float out[6]) const {
        float w[3]; int b[3];
        for (int a = 0; a < 3; ++a) {
            const float gf = (p[a] + 0.5f) * res - 0.5f;
            b[a] = (int)std::floor(gf);
            w[a] = gf - b[a];
        }
        float acc[6] = {0,0,0,0,0,0}, wsum = 0.f;
        for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            const float ww = (dx ? w[0] : 1-w[0]) * (dy ? w[1] : 1-w[1]) * (dz ? w[2] : 1-w[2]);
            if (ww <= 0.f) continue;
            auto it = map.find(key(b[0]+dx, b[1]+dy, b[2]+dz));
            if (it == map.end()) continue;
            const float* f = &(*feats)[(size_t)it->second * 6];
            for (int k = 0; k < 6; ++k) acc[k] += ww * f[k];
            wsum += ww;
        }
        if (wsum <= 1e-6f) return false;
        for (int k = 0; k < 6; ++k) out[k] = acc[k] / wsum;
        return true;
    }
    bool sample(const float p[3], float out[6]) const {
        if (trilinear(p, out)) return true;
        // Decimation moves the surface off the voxel shell. Primary correction
        // (reference behavior): snap to the closest point on the original mesh
        // and resample there. Shell crawl remains as the no-BVH fallback.
        if (snap) {
            const TriBvh::Hit h = snap->closest(p, 8.0f / res);
            if (h.face >= 0 && trilinear(h.point, out)) return true;
        }
        int b[3];
        for (int a = 0; a < 3; ++a) b[a] = (int)std::floor((p[a] + 0.5f) * res - 0.5f);
        for (int r = 1; r <= 3; ++r) {
            float racc[6] = {0,0,0,0,0,0}; int hits = 0;
            for (int dz = -r; dz <= r; ++dz) for (int dy = -r; dy <= r; ++dy) for (int dx = -r; dx <= r; ++dx) {
                if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) continue;
                auto it = map.find(key(b[0]+dx, b[1]+dy, b[2]+dz));
                if (it == map.end()) continue;
                const float* f = &(*feats)[(size_t)it->second * 6];
                for (int k = 0; k < 6; ++k) racc[k] += f[k];
                ++hits;
            }
            if (hits) {
                for (int k = 0; k < 6; ++k) out[k] = racc[k] / hits;
                return true;
            }
        }
        return false;
    }
};

// Multi-source BFS dilation: every unwritten texel takes the color of its
// nearest written one. Bounded passes leave unshaded islands wherever a region
// (e.g. a filled hole's fan) has no voxel data at all; the reference handles
// the same case with unbounded texture-space inpainting.
void dilate_full(std::vector<uint8_t>& base, std::vector<uint8_t>& mr, std::vector<uint8_t>& mask, int T) {
    std::vector<int> queue;
    queue.reserve((size_t)T * T / 4);
    for (int y = 0; y < T; ++y) for (int x = 0; x < T; ++x)
        if (mask[(size_t)y*T + x]) queue.push_back(y*T + x);
    static const int DX[4] = {1, -1, 0, 0}, DY[4] = {0, 0, 1, -1};
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int t = queue[qi], x = t % T, y = t / T;
        for (int d = 0; d < 4; ++d) {
            const int xx = x + DX[d], yy = y + DY[d];
            if (xx < 0 || yy < 0 || xx >= T || yy >= T) continue;
            const size_t s = (size_t)yy*T + xx;
            if (mask[s]) continue;
            for (int c = 0; c < 4; ++c) { base[4*s+c] = base[4*(size_t)t+c]; mr[4*s+c] = mr[4*(size_t)t+c]; }
            mask[s] = 1;
            queue.push_back((int)s);
        }
    }
}

// CPU port of cv2.INPAINT_TELEA (Telea 2004, OpenCV icvTeleaInpaintFMM),
// matching the reference's texture gutter fill (postprocess.py: radius 3 for
// base color, 1 for metallic-roughness). Unknown texels are visited in
// fast-marching order (distance from the shaded boundary); each is filled
// with a weighted average over already-known pixels in a square window,
// weights = |direction · distance · level| as in OpenCV. Simplification vs
// OpenCV: the distance field outside the unknown region is 0 (OpenCV runs a
// second FMM outward; for gutter fill the difference is invisible).
void telea_inpaint(std::vector<uint8_t>& base, std::vector<uint8_t>& mr,
                   const std::vector<uint8_t>& mask, int T, int rad_base, int rad_mr) {
    constexpr uint8_t KNOWN = 0, BAND = 1, INSIDE = 2;
    const size_t N = (size_t)T * T;
    std::vector<uint8_t> flag(N);
    std::vector<float> dist(N);
    for (size_t i = 0; i < N; ++i) {
        flag[i] = mask[i] ? KNOWN : INSIDE;
        dist[i] = mask[i] ? 0.f : 1e6f;
    }
    // The march starts from KNOWN boundary pixels (as in OpenCV, where the
    // initial narrow band lies on the known side); every unknown pixel is
    // painted the moment it enters the band, so gathers never see unpainted
    // content.
    using QE = std::pair<float, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> q;
    static const int DX[4] = {1, -1, 0, 0}, DY[4] = {0, 0, 1, -1};
    std::vector<uint8_t> popped(N, 0);
    for (int y = 0; y < T; ++y)
        for (int x = 0; x < T; ++x) {
            const size_t t = (size_t)y * T + x;
            if (flag[t] != KNOWN) continue;
            for (int d = 0; d < 4; ++d) {
                const int xx = x + DX[d], yy = y + DY[d];
                if (xx < 0 || yy < 0 || xx >= T || yy >= T) continue;
                if (flag[(size_t)yy * T + xx] == INSIDE) {
                    q.emplace(0.f, (int)t);
                    break;
                }
            }
        }
    auto solve = [&](int x, int y) -> float {
        auto axial = [&](int xx, int yy) -> float {
            if (xx < 0 || yy < 0 || xx >= T || yy >= T) return 1e6f;
            const size_t s = (size_t)yy * T + xx;
            return flag[s] == INSIDE ? 1e6f : dist[s];
        };
        float best = 1e6f;
        for (int sx = -1; sx <= 1; sx += 2)
            for (int sy = -1; sy <= 1; sy += 2) {
                const float a = axial(x + sx, y), b = axial(x, y + sy);
                float t;
                if (a < 1e6f && b < 1e6f && std::fabs(a - b) < 1.f)
                    t = (a + b + std::sqrt(2.f - (a - b) * (a - b))) * 0.5f;
                else
                    t = std::min(a, b) + 1.f;
                best = std::min(best, t);
            }
        return best;
    };
    auto inpaint_px = [&](int x, int y) {
        const size_t t = (size_t)y * T + x;
        float gx, gy;
        {
            auto dv = [&](int xx, int yy) -> float {
                if (xx < 0 || yy < 0 || xx >= T || yy >= T) return dist[t];
                const size_t s = (size_t)yy * T + xx;
                return flag[s] == INSIDE ? dist[t] : dist[s];
            };
            gx = (dv(x + 1, y) - dv(x - 1, y)) * 0.5f;
            gy = (dv(x, y + 1) - dv(x, y - 1)) * 0.5f;
        }
        const int rmax = std::max(rad_base, rad_mr);
        double accb[4] = {0, 0, 0, 0}, accm[4] = {0, 0, 0, 0}, wb = 0, wm = 0;
        for (int dy = -rmax; dy <= rmax; ++dy)
            for (int dx = -rmax; dx <= rmax; ++dx) {
                const int xx = x + dx, yy = y + dy;
                if ((dx == 0 && dy == 0) || xx < 0 || yy < 0 || xx >= T || yy >= T) continue;
                const size_t s = (size_t)yy * T + xx;
                if (flag[s] == INSIDE) continue;
                const float len2 = (float)(dx * dx + dy * dy);
                float dir = (float)dx * gx + (float)dy * gy;
                if (std::fabs(dir) <= 0.01f) dir = 1e-6f;
                const float dst = 1.f / (len2 * std::sqrt(len2));
                const float lev = 1.f / (1.f + std::fabs(dist[s] - dist[t]));
                const float w = std::fabs(dir * dst * lev);
                if (std::abs(dx) <= rad_base && std::abs(dy) <= rad_base) {
                    for (int c = 0; c < 4; ++c) accb[c] += w * base[4 * s + c];
                    wb += w;
                }
                if (std::abs(dx) <= rad_mr && std::abs(dy) <= rad_mr) {
                    for (int c = 0; c < 4; ++c) accm[c] += w * mr[4 * s + c];
                    wm += w;
                }
            }
        if (wb > 0) for (int c = 0; c < 4; ++c) base[4 * t + c] = (uint8_t)std::min(255.0, accb[c] / wb + 0.5);
        if (wm > 0) for (int c = 0; c < 4; ++c) mr[4 * t + c] = (uint8_t)std::min(255.0, accm[c] / wm + 0.5);
    };
    while (!q.empty()) {
        const QE top = q.top(); q.pop();
        const size_t t = (size_t)top.second;
        if (popped[t]) continue;
        popped[t] = 1;
        flag[t] = KNOWN;
        const int x = (int)(t % T), y = (int)(t / T);
        for (int d = 0; d < 4; ++d) {
            const int xx = x + DX[d], yy = y + DY[d];
            if (xx < 0 || yy < 0 || xx >= T || yy >= T) continue;
            const size_t s = (size_t)yy * T + xx;
            if (popped[s] || flag[s] == KNOWN) continue;
            const float nd = solve(xx, yy);
            if (nd < dist[s]) dist[s] = nd;
            if (flag[s] == INSIDE) {
                inpaint_px(xx, yy);
                flag[s] = BAND;
            }
            q.emplace(dist[s], (int)s);
        }
    }
}
}  // namespace

void decimate_cluster(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      const std::vector<float>& pbr6, int grid,
                      std::vector<float>& ov, std::vector<int32_t>& of, std::vector<float>& op) {
    const bool hp = !pbr6.empty();
    auto cell = [&](int v, int a){ int c = (int)((verts[3*v+a] + 0.5f) * grid); return c < 0 ? 0 : (c >= grid ? grid-1 : c); };
    auto key = [](int x,int y,int z){ return ((uint64_t)(uint32_t)x<<40)|((uint64_t)(uint32_t)y<<20)|(uint32_t)z; };
    std::unordered_map<uint64_t,int> cmap; cmap.reserve(V);
    std::vector<int> vmap(V);
    std::vector<double> psum, csum; std::vector<int> cnt;
    for (int v = 0; v < V; ++v) {
        uint64_t k = key(cell(v,0), cell(v,1), cell(v,2));
        auto it = cmap.find(k);
        int idx;
        if (it == cmap.end()) { idx = (int)cnt.size(); cmap[k] = idx; cnt.push_back(0); psum.insert(psum.end(),{0,0,0}); if(hp) csum.insert(csum.end(),{0,0,0,0,0,0}); }
        else idx = it->second;
        vmap[v] = idx; cnt[idx]++;
        for (int a=0;a<3;++a) psum[3*idx+a] += verts[3*v+a];
        if (hp) for (int a=0;a<6;++a) csum[6*idx+a] += pbr6[6*v+a];
    }
    const int M = (int)cnt.size();
    ov.resize((size_t)M*3); if(hp) op.resize((size_t)M*6);
    for (int i=0;i<M;++i){ for(int a=0;a<3;++a) ov[3*i+a]=(float)(psum[3*i+a]/cnt[i]); if(hp) for(int a=0;a<6;++a) op[6*i+a]=(float)(csum[6*i+a]/cnt[i]); }
    of.clear(); of.reserve((size_t)F*3);
    for (int f=0;f<F;++f){ int a=vmap[faces[3*f]],b=vmap[faces[3*f+1]],c=vmap[faces[3*f+2]];
        if (a!=b && b!=c && a!=c) { of.push_back(a); of.push_back(b); of.push_back(c); } }
    printf("  decimate(grid=%d): V %d->%d, F %d->%d\n", grid, V, M, F, (int)of.size()/3);
}


// Vertex -> incident faces (CSR, counting sort in face order). The mesh cleanup
// passes below used one hash node per edge (60M nodes on a res-1024 remesh);
// an undirected edge (a,b) is incident to exactly the faces of a that also
// contain b, so edge multiplicity is a scan of a's short face list instead.
struct VertFaces {
    std::vector<int32_t> off, fac;   // faces of v: fac[off[v] .. off[v+1])
    void build(const std::vector<int32_t>& faces, int V) {
        off.assign((size_t)V + 1, 0);
        for (int32_t v : faces) off[(size_t)v + 1]++;
        for (int v = 0; v < V; ++v) off[(size_t)v + 1] += off[(size_t)v];
        fac.resize(faces.size());
        std::vector<int32_t> fill(off.begin(), off.end() - 1);
        for (size_t i = 0; i < faces.size(); ++i) fac[(size_t)fill[faces[i]]++] = (int32_t)(i / 3);
    }
    // Faces containing both a and b; `other` receives the last one that is not `self`.
    int edge_faces(const std::vector<int32_t>& faces, int a, int b, int self, int* other) const {
        int n = 0;
        for (int32_t i = off[(size_t)a]; i < off[(size_t)a + 1]; ++i) {
            const int f = fac[(size_t)i];
            const int32_t* t = &faces[3 * (size_t)f];
            if (t[0] == b || t[1] == b || t[2] == b) { ++n; if (f != self && other) *other = f; }
        }
        return n;
    }
    // Faces traversing the directed edge a->b (`fwd`) and b->a (`rev`) in their winding.
    void edge_dirs(const std::vector<int32_t>& faces, int a, int b, int* fwd, int* rev) const {
        *fwd = 0; *rev = 0;
        for (int32_t i = off[(size_t)a]; i < off[(size_t)a + 1]; ++i) {
            const int32_t* t = &faces[3 * (size_t)fac[(size_t)i]];
            for (int k = 0; k < 3; ++k) {
                if (t[k] == a && t[(k + 1) % 3] == b) ++*fwd;
                else if (t[k] == b && t[(k + 1) % 3] == a) ++*rev;
            }
        }
    }
};

static int max_index_plus1(const std::vector<int32_t>& faces, int V) {
    int m = V;
    for (int32_t v : faces) m = std::max(m, v + 1);
    return m;
}

int weld_vertices(std::vector<float>& verts, std::vector<int32_t>& faces, std::vector<float>* colors3,
                  float step) {
    const int V = (int)verts.size() / 3;
    if (colors3 && colors3->size() != verts.size()) {   // guard: per-vertex colors out of sync
        fprintf(stderr, "  weld: colors/verts size mismatch (%zu vs %zu), ignoring colors\n",
                colors3->size(), verts.size());
        colors3 = nullptr;
    }
    const float eps2 = step * step;
    // Greedy first-fit weld in vertex order: v joins the first already-kept vertex
    // within `step`, searching the 27 surrounding cells in a fixed order. The
    // candidate pairs are found up front with sorted cell keys instead of a hash
    // per cell: the key packs the cell coordinate monotonically (21 biased bits
    // per axis), so "cell + offset" is "key + constant" and every neighbouring
    // cell pair falls out of one merge pass per offset.
    constexpr int64_t BIAS = (int64_t)1 << 20;
    auto pack = [](int64_t cx, int64_t cy, int64_t cz) -> uint64_t {
        auto clampc = [](int64_t c) { return std::max<int64_t>(-BIAS + 1, std::min<int64_t>(BIAS - 2, c)); };
        return ((uint64_t)(clampc(cx) + BIAS) << 42) | ((uint64_t)(clampc(cy) + BIAS) << 21) | (uint64_t)(clampc(cz) + BIAS);
    };
    std::vector<std::pair<uint64_t, int32_t>> cells((size_t)V);
    parallel_for(V, [&](int64_t b, int64_t e) {
        for (int64_t v = b; v < e; ++v) {
            const float* p = &verts[3*v];
            cells[(size_t)v] = { pack((int64_t)std::floor(p[0] / step), (int64_t)std::floor(p[1] / step),
                                      (int64_t)std::floor(p[2] / step)), (int32_t)v };
        }
    });
    std::sort(cells.begin(), cells.end());
    // (v, search order of u's cell, u) for every pair within `step`, u < v
    struct Cand { int32_t v, order, u; bool operator<(const Cand& o) const {
        return v != o.v ? v < o.v : (order != o.order ? order < o.order : u < o.u); } };
    auto near = [&](int32_t a, int32_t b) {
        const float* pa = &verts[3*a]; const float* pb = &verts[3*b];
        const float ex = pa[0]-pb[0], ey = pa[1]-pb[1], ez = pa[2]-pb[2];
        return ex*ex + ey*ey + ez*ez <= eps2;
    };
    std::vector<std::vector<Cand>> parts(27);
    parallel_for(27, [&](int64_t ob, int64_t oe) {
        for (int64_t o = ob; o < oe; ++o) {
            const int dx = (int)(o % 3) - 1, dy = (int)((o / 3) % 3) - 1, dz = (int)(o / 9) - 1;
            std::vector<Cand>& out = parts[(size_t)o];
            if (dx == 0 && dy == 0 && dz == 0) {   // same cell: all pairs of a run
                for (size_t i = 0; i < cells.size();) {
                    size_t j = i;
                    while (j < cells.size() && cells[j].first == cells[i].first) ++j;
                    for (size_t a = i; a < j; ++a)
                        for (size_t b = a + 1; b < j; ++b)
                            if (near(cells[a].second, cells[b].second))
                                out.push_back({cells[b].second, 13, cells[a].second});
                    i = j;
                }
                continue;
            }
            // pairs (a, b) with cell(b) = cell(a) + d; emitted for v = b (u's cell is at -d from v's)
            const int64_t delta = ((int64_t)dx << 42) + ((int64_t)dy << 21) + (int64_t)dz;
            const int order = ((-dz + 1) * 3 + (-dy + 1)) * 3 + (-dx + 1);
            size_t j = 0;
            for (size_t i = 0; i < cells.size(); ++i) {
                const uint64_t want = (uint64_t)((int64_t)cells[i].first + delta);
                while (j < cells.size() && cells[j].first < want) ++j;
                for (size_t k = j; k < cells.size() && cells[k].first == want; ++k)
                    if (cells[i].second < cells[k].second && near(cells[i].second, cells[k].second))
                        out.push_back({cells[k].second, order, cells[i].second});
            }
        }
    });
    std::vector<Cand> cand;
    for (auto& p : parts) cand.insert(cand.end(), p.begin(), p.end());
    std::sort(cand.begin(), cand.end());

    std::vector<int> remap((size_t)V), kept_id((size_t)V, -1);
    std::vector<float> nv; nv.reserve(verts.size());
    std::vector<float> ncol; if (colors3) ncol.reserve(colors3->size());
    int keep = 0;
    size_t ci = 0;
    for (int v = 0; v < V; ++v) {
        int found = -1;
        for (; ci < cand.size() && cand[ci].v == v; ++ci)
            if (found < 0 && kept_id[(size_t)cand[ci].u] >= 0) found = kept_id[(size_t)cand[ci].u];
        if (found < 0) {
            found = keep++;
            kept_id[(size_t)v] = found;
            nv.insert(nv.end(), &verts[3*v], &verts[3*v] + 3);
            if (colors3) ncol.insert(ncol.end(), &(*colors3)[3*v], &(*colors3)[3*v] + 3);
        }
        remap[v] = found;
    }
    for (auto& f : faces) f = remap[f];
    const int welded = V - keep;
    verts.swap(nv);
    if (colors3) colors3->swap(ncol);
    if (welded) { printf("  weld: V %d->%d (%d near-duplicates merged, eps=%.2g)\n", V, keep, welded, step); fflush(stdout); }
    return welded;
}

void clean_mesh(int V, std::vector<int32_t>& faces) {
    // 1. drop degenerate faces (a repeated vertex -> zero area, blocks collapse)
    {
        std::vector<int32_t> kept; kept.reserve(faces.size());
        for (size_t f = 0; f + 2 < faces.size(); f += 3) {
            int a = faces[f], b = faces[f+1], c = faces[f+2];
            if (a != b && b != c && a != c) { kept.push_back(a); kept.push_back(b); kept.push_back(c); }
        }
        faces.swap(kept);
    }
    const int F = (int)(faces.size() / 3);
    if (F == 0) return;
    // 2. vertex -> incident faces; an edge's incident faces are the faces of one
    // endpoint that also contain the other.
    VertFaces vf;
    vf.build(faces, max_index_plus1(faces, V));
    // 3. BFS flood; flip faces to a consistent winding across manifold (exactly-2-face) edges.
    // Non-manifold / boundary edges are not crossed, so orientation stays locally consistent.
    std::vector<char> vis(F, 0), flip(F, 0);
    std::vector<int> st;
    for (int seed = 0; seed < F; ++seed) {
        if (vis[seed]) continue;
        vis[seed] = 1; st.push_back(seed);
        while (!st.empty()) {
            int f = st.back(); st.pop_back();
            int v[3] = { faces[3*f], faces[3*f+1], faces[3*f+2] };
            if (flip[f]) std::swap(v[1], v[2]);
            for (int j = 0; j < 3; ++j) {
                int a = v[j], b = v[(j+1)%3];
                int g = -1;
                if (vf.edge_faces(faces, a, b, f, &g) != 2 || g < 0) continue;
                if (vis[g]) continue;
                bool same = false;   // g traverses this edge the SAME way as f's corrected winding?
                for (int k = 0; k < 3; ++k)
                    if (faces[3*g+k] == a && faces[3*g+(k+1)%3] == b) { same = true; break; }
                if (same) flip[g] = 1;   // consistent orientation needs the neighbour to go b->a
                vis[g] = 1; st.push_back(g);
            }
        }
    }
    int nf = 0;
    for (int f = 0; f < F; ++f) if (flip[f]) { std::swap(faces[3*f+1], faces[3*f+2]); ++nf; }
    printf("    [clean] faces=%d, flipped %d for consistent winding\n", F, nf);
    fflush(stdout);
}

void decimate_simplify(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                       int target_faces, std::vector<float>& ov, std::vector<int32_t>& of) {
    // Progressive error ladder: a permissive one-shot error budget lets
    // meshopt_SimplifyPrune delete the whole mesh, while a strict one stalls far
    // above the target on messy voxel topology. Each rung re-simplifies the
    // previous result with a looser budget until the face target is reached; a
    // rung that collapses to (near) nothing is rejected.
    std::vector<unsigned int> idx(faces.begin(), faces.end());
    std::vector<unsigned int> out_idx(idx.size());
    float err = 0.f;
    size_t n = idx.size();
    const size_t target_idx = (size_t)target_faces * 3;
    static const float kErrLadder[] = {0.01f, 0.05f, 0.25f, 1.0f};
    for (float e : kErrLadder) {
        if (n <= target_idx * 6 / 5) break;
        float stage_err = 0.f;
        const size_t r = meshopt_simplify(out_idx.data(), idx.data(), n, verts.data(), (size_t)V,
                                          3 * sizeof(float), target_idx, e,
                                          e <= 0.01f ? meshopt_SimplifyPrune : 0, &stage_err);
        if (r < 3000 || r >= n) break;
        printf("    [quadric err<=%.2f] F %zu->%zu\n", e, n/3, r/3);
        std::copy(out_idx.begin(), out_idx.begin() + r, idx.begin());
        n = r;
        err = stage_err;
    }
    // Topology-preserving collapse bottoms out well above the target on voxel
    // meshes (its guards reject every remaining collapse regardless of error
    // budget), and position clustering shreds thin features. Finish with an
    // unguarded quadric collapse (FQMS) — the same approach the reference's
    // cumesh simplifier takes — which reaches the count while following the
    // error metric.
    if (n > target_idx * 6 / 5) {
        Simplify::vertices.clear();
        Simplify::triangles.clear();
        Simplify::vertices.resize((size_t)V);
        for (int v = 0; v < V; ++v) {
            Simplify::vertices[v].p.x = verts[3*v];
            Simplify::vertices[v].p.y = verts[3*v+1];
            Simplify::vertices[v].p.z = verts[3*v+2];
        }
        Simplify::triangles.resize(n / 3);
        for (size_t i = 0; i < n / 3; ++i) {
            for (int j = 0; j < 3; ++j) Simplify::triangles[i].v[j] = (int)idx[3*i+j];
            Simplify::triangles[i].attr = 0;
            Simplify::triangles[i].material = -1;
        }
        Simplify::simplify_mesh((int)(target_idx / 3), 7.0, false);
        printf("    [fqms] F %zu->%zu\n", n/3, Simplify::triangles.size());
        ov.resize(Simplify::vertices.size() * 3);
        for (size_t v = 0; v < Simplify::vertices.size(); ++v) {
            ov[3*v]   = (float)Simplify::vertices[v].p.x;
            ov[3*v+1] = (float)Simplify::vertices[v].p.y;
            ov[3*v+2] = (float)Simplify::vertices[v].p.z;
        }
        of.resize(Simplify::triangles.size() * 3);
        for (size_t i = 0; i < Simplify::triangles.size(); ++i)
            for (int j = 0; j < 3; ++j) of[3*i+j] = Simplify::triangles[i].v[j];
        Simplify::vertices.clear();
        Simplify::triangles.clear();
        printf("  simplify(target=%d): V %d->%d, F %d->%d\n", target_faces, V,
               (int)ov.size()/3, F, (int)of.size()/3);
        fflush(stdout);
        return;
    }
    std::copy(idx.begin(), idx.begin() + n, out_idx.begin());
    // Drop degenerate and zero-area faces: simplification at extreme ratios leaves
    // slivers that stall xatlas's chart growth.
    size_t w = 0;
    for (size_t i = 0; i + 2 < n; i += 3) {
        const unsigned int a = out_idx[i], b = out_idx[i+1], c2 = out_idx[i+2];
        if (a == b || b == c2 || a == c2) continue;
        float e1[3], e2[3], cr[3];
        for (int k = 0; k < 3; ++k) { e1[k] = verts[3*b+k]-verts[3*a+k]; e2[k] = verts[3*c2+k]-verts[3*a+k]; }
        cr[0]=e1[1]*e2[2]-e1[2]*e2[1]; cr[1]=e1[2]*e2[0]-e1[0]*e2[2]; cr[2]=e1[0]*e2[1]-e1[1]*e2[0];
        if (cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2] <= 0.f) continue;
        out_idx[w++] = a; out_idx[w++] = b; out_idx[w++] = c2;
    }
    n = w;
    std::vector<int> remap((size_t)V, -1);
    ov.clear(); of.resize(n);
    int nv = 0;
    for (size_t i = 0; i < n; ++i) {
        const unsigned int v = out_idx[i];
        if (remap[v] < 0) { remap[v] = nv++; ov.insert(ov.end(), &verts[3*v], &verts[3*v] + 3); }
        of[i] = remap[v];
    }
    printf("  simplify(target=%d): V %d->%d, F %d->%d (err %.4f)\n", target_faces, V, nv, F, (int)n/3, err);
    fflush(stdout);
}

// Drop connected components (shared-vertex face adjacency) whose face count is below
// frac*(largest component's face count). Removes the decode floaters and spurious
// ground-plane fragments that shatter our mesh into 50+ pieces while keeping legitimately
// large secondary shells (e.g. a hollow vase's inner+outer surface, ~50% each). The
// reference reconstruction is a single watertight component; these floaters are pure junk
// that spawn extra UV charts and visible specks. Compacts verts/faces in place.
int drop_small_components(std::vector<float>& verts, std::vector<int32_t>& faces, float frac) {
    const int V = (int)verts.size() / 3;
    const size_t F = faces.size() / 3;
    if (V == 0 || F == 0) return 0;
    // Lock-free union-find (link the larger root under the smaller, CAS on roots
    // only): threads union disjoint face ranges into one shared forest. The
    // partition into components -- all that is used below -- does not depend on
    // the interleaving.
    std::vector<std::atomic<int>> par((size_t)V);
    parallel_for(V, [&](int64_t b, int64_t e) { for (int64_t i = b; i < e; ++i) par[(size_t)i].store((int)i, std::memory_order_relaxed); });
    auto find = [&](int x) {
        for (;;) {
            int p = par[(size_t)x].load(std::memory_order_relaxed);
            if (p == x) return x;
            int gp = par[(size_t)p].load(std::memory_order_relaxed);
            if (gp == p) return p;
            par[(size_t)x].compare_exchange_weak(p, gp, std::memory_order_relaxed);   // path halving
            x = gp;
        }
    };
    auto uni = [&](int a, int b) {
        for (;;) {
            a = find(a); b = find(b);
            if (a == b) return;
            if (a < b) std::swap(a, b);
            int expected = a;
            if (par[(size_t)a].compare_exchange_strong(expected, b, std::memory_order_relaxed)) return;
        }
    };
    parallel_for((int64_t)F, [&](int64_t b, int64_t e) {
        for (int64_t f = b; f < e; ++f) { uni(faces[3*f], faces[3*f+1]); uni(faces[3*f+1], faces[3*f+2]); }
    });
    std::vector<int32_t> root((size_t)F);   // component root per face
    parallel_for((int64_t)F, [&](int64_t b, int64_t e) { for (int64_t f = b; f < e; ++f) root[(size_t)f] = find(faces[3*f]); });
    std::vector<int> fc((size_t)V, 0);   // faces per component, indexed by root vertex
    for (size_t f = 0; f < F; ++f) fc[(size_t)root[f]]++;
    int maxfc = 0;
    for (int i = 0; i < V; ++i) maxfc = std::max(maxfc, fc[i]);
    const int thresh = (int)(frac * maxfc);
    int dropped = 0;
    for (int i = 0; i < V; ++i) if (fc[i] > 0 && fc[i] < thresh) ++dropped;
    if (dropped == 0) return 0;
    std::vector<int32_t> kf; kf.reserve(faces.size());
    for (size_t f = 0; f < F; ++f)
        if (fc[(size_t)root[f]] >= thresh)
            for (int k = 0; k < 3; ++k) kf.push_back(faces[3*f + k]);
    std::vector<int> remap((size_t)V, -1);
    std::vector<float> nv; nv.reserve(verts.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        int v = kf[i];
        if (remap[v] < 0) { remap[v] = (int)(nv.size() / 3); nv.insert(nv.end(), &verts[3*v], &verts[3*v] + 3); }
        kf[i] = remap[v];
    }
    verts.swap(nv); faces.swap(kf);
    return dropped;
}

// Fill small boundary loops with a centroid fan (port of CuMesh clean_up.cu fill_holes).
// A boundary edge is a directed face edge (u,v) with no opposing (v,u); rim cycles are
// walked in reverse face direction (v->u keeps the new fan's winding consistent with the
// neighbouring faces). Loops with perimeter < max_perimeter gain one vertex at the mean
// of their edge midpoints (== CuMesh's segmented midpoint average) and a triangle per rim
// edge. Non-manifold rim junctions or non-closing walks leave that loop unfilled.
int fill_holes(std::vector<float>& verts, std::vector<int32_t>& faces, float max_perimeter) {
    const size_t F = faces.size() / 3;
    if (F == 0) return 0;
    const int V = max_index_plus1(faces, (int)(verts.size() / 3));
    VertFaces vf;
    vf.build(faces, V);
    // rim successor map: boundary edge (u,v) is traversed v->u, so nxt[v] = u;
    // a boundary edge belongs to exactly one face, so scanning faces visits each once
    std::vector<int> nxt((size_t)V, -1);
    for (size_t f = 0; f < F; ++f) {
        const int32_t* t = &faces[3*f];
        for (int k = 0; k < 3; ++k) {
            const int u = t[k], v = t[(k+1)%3];
            if (vf.edge_faces(faces, u, v, -1, nullptr) != 1) continue;
            if (nxt[(size_t)v] == -1) nxt[(size_t)v] = u; else nxt[(size_t)v] = INT32_MIN;   // non-manifold junction: poison
        }
    }
    int filled = 0;
    std::vector<char> visited((size_t)V, 0);
    std::vector<int> loop;
    for (int start = 0; start < V; ++start) {
        if (nxt[(size_t)start] == -1 || nxt[(size_t)start] == INT32_MIN || visited[(size_t)start]) continue;
        loop.clear();
        int cur = start; bool ok = true;
        double perim = 0.0;
        while (true) {
            loop.push_back(cur);
            const int n = nxt[(size_t)cur];
            if (n == -1 || n == INT32_MIN || (int)loop.size() > V) { ok = false; break; }
            const float* a = &verts[3*(size_t)cur]; const float* b = &verts[3*(size_t)n];
            perim += std::sqrt((double)(a[0]-b[0])*(a[0]-b[0]) + (double)(a[1]-b[1])*(a[1]-b[1]) + (double)(a[2]-b[2])*(a[2]-b[2]));
            cur = n;
            if (cur == start) break;
        }
        for (int v : loop) visited[(size_t)v] = 1;
        if (!ok || loop.size() < 3 || perim >= max_perimeter) continue;
        // centroid = mean of rim edge midpoints (uniform over edges; each vertex is in 2 edges
        // of the cycle, so this equals the mean of the loop vertices)
        double c[3] = {0, 0, 0};
        for (int v : loop) for (int k = 0; k < 3; ++k) c[k] += verts[3*(size_t)v + k];
        const int cid = (int)(verts.size() / 3);
        for (int k = 0; k < 3; ++k) verts.push_back((float)(c[k] / loop.size()));
        for (size_t i = 0; i < loop.size(); ++i) {
            const int v = loop[i], u = loop[(i+1) % loop.size()];   // rim direction v->u
            faces.push_back(v); faces.push_back(u); faces.push_back(cid);
        }
        ++filled;
    }
    return filled;
}

// Taubin (lambda/mu) shrink-free Laplacian smoothing over the shared-vertex mesh. The
// dual-contour surface carries ~1-voxel stair-step noise; a quadric simplifier sees that
// as "curvature everywhere" and produces a uniform-dense sliver mesh instead of an
// adaptive one. Low-passing it first (mu slightly stronger than lambda keeps volume) lets
// the simplifier find real flats. Uniform umbrella weights; boundary verts pinned so open
// holes don't shrink. In place.
void taubin_smooth(std::vector<float>& verts, const std::vector<int32_t>& faces,
                   int iters, float lambda, float mu) {
    const int V = (int)verts.size() / 3;
    const size_t F = faces.size() / 3;
    if (V == 0 || F == 0 || iters <= 0) return;
    // unique undirected edges + use-count (==1 => boundary)
    std::unordered_map<uint64_t,int> euse;
    euse.reserve(F * 3);
    auto ekey = [](int a, int b) -> uint64_t { if (a > b) { int t = a; a = b; b = t; } return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; };
    for (size_t f = 0; f < F; ++f) {
        const int32_t* t = &faces[3*f];
        for (int k = 0; k < 3; ++k) euse[ekey(t[k], t[(k+1)%3])]++;
    }
    // CSR adjacency (each edge contributes both directions) + boundary flags
    std::vector<int> deg((size_t)V, 0);
    std::vector<uint8_t> boundary((size_t)V, 0);
    for (auto& kv : euse) {
        int a = (int)(kv.first >> 32), b = (int)(kv.first & 0xffffffffu);
        deg[a]++; deg[b]++;
        if (kv.second == 1) { boundary[a] = boundary[b] = 1; }
    }
    std::vector<int> off((size_t)V + 1, 0);
    for (int i = 0; i < V; ++i) off[i+1] = off[i] + deg[i];
    std::vector<int> nbr((size_t)off[V]);
    std::vector<int> cur(off.begin(), off.end() - 1);
    for (auto& kv : euse) {
        int a = (int)(kv.first >> 32), b = (int)(kv.first & 0xffffffffu);
        nbr[cur[a]++] = b; nbr[cur[b]++] = a;
    }
    std::vector<float> tmp(verts.size());
    auto step = [&](float w) {
        for (int i = 0; i < V; ++i) {
            const int s = off[i], e = off[i+1];
            if (boundary[i] || e == s) { for (int k = 0; k < 3; ++k) tmp[3*i+k] = verts[3*i+k]; continue; }
            float c[3] = {0, 0, 0};
            for (int j = s; j < e; ++j) { const float* p = &verts[3*nbr[j]]; c[0]+=p[0]; c[1]+=p[1]; c[2]+=p[2]; }
            const float inv = 1.0f / (float)(e - s);
            for (int k = 0; k < 3; ++k) tmp[3*i+k] = verts[3*i+k] + w * (c[k]*inv - verts[3*i+k]);
        }
        verts.swap(tmp);
    };
    for (int it = 0; it < iters; ++it) { step(lambda); step(mu); }
}

int fill_small_holes(std::vector<int32_t>& faces, int max_loop) {
    const size_t F = faces.size() / 3;
    const int V = max_index_plus1(faces, 0);
    VertFaces vf;
    vf.build(faces, V);
    // Boundary edges traversed opposite to face winding so fan fills keep
    // orientation consistent with their neighbors. Chains pass only through
    // unambiguous boundary vertices (out- and in-degree exactly 1): at
    // non-manifold junctions a single-successor map silently cross-links
    // fragments of different holes into bogus mesh-spanning "loops".
    std::vector<int> nxt((size_t)V, -1), outd((size_t)V, 0), ind((size_t)V, 0);
    for (size_t f = 0; f < F; ++f) {
        for (int j = 0; j < 3; ++j) {
            const int a = faces[3*f+j], b = faces[3*f+(j+1)%3];
            int fwd, rev;
            vf.edge_dirs(faces, a, b, &fwd, &rev);
            if (fwd == 1 && rev == 0) { nxt[(size_t)b] = a; outd[(size_t)b]++; ind[(size_t)a]++; }
        }
    }
    std::vector<char> used((size_t)V, 0);
    int filled = 0;
    size_t added = 0;
    for (int start = 0; start < V; ++start) {
        if (nxt[(size_t)start] < 0 || used[(size_t)start] || outd[(size_t)start] != 1 || ind[(size_t)start] != 1) continue;
        std::vector<int> loop = {start};
        int cur = start;
        bool cycle = false, clean = true;
        for (int steps = 0; steps <= max_loop; ++steps) {
            if (nxt[(size_t)cur] < 0 || used[(size_t)cur]) { clean = false; break; }
            cur = nxt[(size_t)cur];
            if (outd[(size_t)cur] != 1 || ind[(size_t)cur] != 1) { clean = false; break; }
            if (cur == start) { cycle = true; break; }
            loop.push_back(cur);
        }
        for (int v : loop) used[(size_t)v] = 1;
        if (!clean || !cycle || loop.size() < 3 || (int)loop.size() > max_loop) continue;
        for (size_t i = 1; i + 1 < loop.size(); ++i) {
            faces.push_back(loop[0]); faces.push_back(loop[i]); faces.push_back(loop[i+1]);
            added += 1;
        }
        ++filled;
    }
    // Second pass, winding-agnostic (the GLB material is double-sided): loops
    // whose boundary direction flips (simplification tears) never chain in the
    // directed walk above. Assemble them over undirected boundary adjacency,
    // restricted to unambiguous degree-2 vertices.
    {
        const size_t F2 = faces.size() / 3;
        vf.build(faces, V);
        std::vector<std::pair<int,int>> adj;   // both directions of every boundary edge, sorted by vertex
        for (size_t f = 0; f < F2; ++f)
            for (int j = 0; j < 3; ++j) {
                const int a = faces[3*f+j], b = faces[3*f+(j+1)%3];
                if (vf.edge_faces(faces, a, b, -1, nullptr) != 1) continue;
                adj.emplace_back(a, b); adj.emplace_back(b, a);
            }
        std::sort(adj.begin(), adj.end());
        std::vector<int32_t> aoff((size_t)V + 1, 0);
        for (const auto& e : adj) aoff[(size_t)e.first + 1]++;
        for (int v = 0; v < V; ++v) aoff[(size_t)v + 1] += aoff[(size_t)v];
        auto deg = [&](int v) { return aoff[(size_t)v + 1] - aoff[(size_t)v]; };
        auto nb  = [&](int v, int i) { return adj[(size_t)aoff[(size_t)v] + i].second; };
        std::vector<char> used2((size_t)V, 0);
        for (int start = 0; start < V; ++start) {
            if (deg(start) != 2 || used2[(size_t)start]) continue;
            std::vector<int> loop = {start};
            int prev = start, cur = nb(start, 0);
            bool cycle = false, clean = true;
            for (int steps = 0; steps <= max_loop; ++steps) {
                if (deg(cur) != 2 || used2[(size_t)cur]) { clean = false; break; }
                if (cur == start) { cycle = true; break; }
                loop.push_back(cur);
                const int nx = nb(cur, 0) == prev ? nb(cur, 1) : nb(cur, 0);
                prev = cur; cur = nx;
            }
            for (int v : loop) used2[(size_t)v] = 1;
            if (!clean || !cycle || loop.size() < 3 || (int)loop.size() > max_loop) continue;
            for (size_t i = 1; i + 1 < loop.size(); ++i) {
                faces.push_back(loop[0]); faces.push_back(loop[i]); faces.push_back(loop[i+1]);
                added += 1;
            }
            ++filled;
        }
    }
    if (filled) { printf("  fill_holes: %d boundary loops filled (+%zu faces)\n", filled, added); fflush(stdout); }
    return filled;
}

// Normal-clustered planar charts with shelf packing: deterministic O(F log F)
// unwrap whose texel density is proportional to surface area. Fallback for the
// meshes xatlas cannot handle (its chart optimization wedges unrecoverably on
// heavily-greebled geometry regardless of face count).
BakedMesh uv_chart_project(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                           const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox) {
    BakedMesh out;
    const int T = texsize;
    if (F == 0) return out;
    std::unique_ptr<VoxSampler> vs;
    if (vox && vox->ok()) vs.reset(new VoxSampler(*vox));

    std::vector<float> fnorm((size_t)F * 3);
    for (int f = 0; f < F; ++f) {
        const int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        float e1[3], e2[3];
        for (int k = 0; k < 3; ++k) { e1[k] = verts[3*b+k]-verts[3*a+k]; e2[k] = verts[3*c+k]-verts[3*a+k]; }
        float n[3] = { e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0] };
        const float l = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        for (int k = 0; k < 3; ++k) fnorm[3*f+k] = l > 1e-20f ? n[k]/l : 0.f;
    }

    std::unordered_map<uint64_t, std::array<int,2>> edge_faces;
    edge_faces.reserve((size_t)F * 3);
    auto ekey = [](int a, int b){ return ((uint64_t)(uint32_t)std::min(a,b) << 32) | (uint32_t)std::max(a,b); };
    for (int f = 0; f < F; ++f)
        for (int j = 0; j < 3; ++j) {
            auto& e = edge_faces[ekey(faces[3*f+j], faces[3*f+(j+1)%3])];
            if (e[0] == 0 && e[1] == 0) { e = {f+1, 0}; }
            else if (e[1] == 0) e[1] = f+1;
        }

    // region growing on the face graph, gated on deviation from the seed normal
    std::vector<int> fchart((size_t)F, -1);
    std::vector<std::vector<int>> charts;
    std::vector<int> stack;
    for (int f0 = 0; f0 < F; ++f0) {
        if (fchart[f0] >= 0) continue;
        const int cid = (int)charts.size();
        charts.emplace_back();
        const float* ns = &fnorm[3*f0];
        stack.push_back(f0); fchart[f0] = cid;
        while (!stack.empty()) {
            const int f = stack.back(); stack.pop_back();
            charts[cid].push_back(f);
            for (int j = 0; j < 3; ++j) {
                const auto& e = edge_faces[ekey(faces[3*f+j], faces[3*f+(j+1)%3])];
                for (int s = 0; s < 2; ++s) {
                    const int g = e[s] - 1;
                    if (g < 0 || fchart[g] >= 0) continue;
                    const float* ng = &fnorm[3*g];
                    if (ns[0]*ng[0] + ns[1]*ng[1] + ns[2]*ng[2] < 0.7f) continue;
                    fchart[g] = cid;
                    stack.push_back(g);
                }
            }
        }
    }

    // per-chart planar basis from the seed normal, projected bbox
    struct Chart { float bu[3], bv[3]; float umin, vmin, umax, vmax; float ox, oy; };
    std::vector<Chart> cs(charts.size());
    for (size_t c = 0; c < charts.size(); ++c) {
        const float* n = &fnorm[3*charts[c][0]];
        float up[3] = { 0, 0, 1 };
        if (std::fabs(n[2]) > 0.9f) { up[0] = 1; up[2] = 0; }
        float* bu = cs[c].bu; float* bv = cs[c].bv;
        bu[0] = up[1]*n[2]-up[2]*n[1]; bu[1] = up[2]*n[0]-up[0]*n[2]; bu[2] = up[0]*n[1]-up[1]*n[0];
        float l = std::sqrt(bu[0]*bu[0]+bu[1]*bu[1]+bu[2]*bu[2]);
        for (int k = 0; k < 3; ++k) bu[k] = l > 1e-20f ? bu[k]/l : (k == 0 ? 1.f : 0.f);
        bv[0] = n[1]*bu[2]-n[2]*bu[1]; bv[1] = n[2]*bu[0]-n[0]*bu[2]; bv[2] = n[0]*bu[1]-n[1]*bu[0];
        cs[c].umin = cs[c].vmin = 1e30f; cs[c].umax = cs[c].vmax = -1e30f;
        for (int f : charts[c])
            for (int j = 0; j < 3; ++j) {
                const float* p = &verts[3*faces[3*f+j]];
                const float u = p[0]*bu[0]+p[1]*bu[1]+p[2]*bu[2];
                const float v = p[0]*bv[0]+p[1]*bv[1]+p[2]*bv[2];
                cs[c].umin = std::min(cs[c].umin, u); cs[c].umax = std::max(cs[c].umax, u);
                cs[c].vmin = std::min(cs[c].vmin, v); cs[c].vmax = std::max(cs[c].vmax, v);
            }
    }

    // shelf packing: global scale found by bisection so everything fits with padding
    const int pad = 2;
    std::vector<int> order(cs.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    float total = 0;
    for (auto& c : cs) total += (c.umax-c.umin) * (c.vmax-c.vmin);
    float scale = total > 0 ? std::sqrt((float)T * T * 0.72f / total) : 1.f;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return cs[a].vmax-cs[a].vmin > cs[b].vmax-cs[b].vmin;
    });
    auto try_pack = [&](float s) -> bool {
        float x = pad, y = pad, shelf = 0;
        for (int ci : order) {
            const float w = (cs[ci].umax-cs[ci].umin)*s + 2*pad, h = (cs[ci].vmax-cs[ci].vmin)*s + 2*pad;
            if (w > T || h > T) return false;
            if (x + w > T) { x = pad; y += shelf; shelf = 0; }
            if (y + h > T) return false;
            cs[ci].ox = x + pad; cs[ci].oy = y + pad;
            x += w; shelf = std::max(shelf, h);
        }
        return true;
    };
    while (scale > 1e-3f && !try_pack(scale)) scale *= 0.92f;

    // emit per-chart vertices/uvs/faces
    std::vector<int> vstamp((size_t)V, -1), vlocal((size_t)V, -1);
    for (size_t c = 0; c < charts.size(); ++c) {
        for (int f : charts[c]) {
            int lidx[3];
            for (int j = 0; j < 3; ++j) {
                const int v = faces[3*f+j];
                if (vstamp[v] != (int)c) {
                    vstamp[v] = (int)c;
                    vlocal[v] = (int)(out.verts.size() / 3);
                    const float* p = &verts[3*v];
                    out.verts.insert(out.verts.end(), p, p + 3);
                    const float u = p[0]*cs[c].bu[0]+p[1]*cs[c].bu[1]+p[2]*cs[c].bu[2];
                    const float w = p[0]*cs[c].bv[0]+p[1]*cs[c].bv[1]+p[2]*cs[c].bv[2];
                    out.uv.push_back((cs[c].ox + (u - cs[c].umin)*scale) / T);
                    out.uv.push_back((cs[c].oy + (w - cs[c].vmin)*scale) / T);
                }
                lidx[j] = vlocal[v];
            }
            for (int j = 0; j < 3; ++j) out.faces.push_back(lidx[j]);
        }
    }

    // depth-tested raster along each chart's seed normal, then dilate
    const int Fo = (int)out.faces.size() / 3;
    out.T = T; out.base.assign((size_t)T*T*4, 0); out.mr.assign((size_t)T*T*4, 0);
    std::vector<uint8_t> mask((size_t)T*T, 0);
    std::vector<float> zbuf((size_t)T*T, -1e30f);
    std::vector<int> fc2((size_t)Fo);
    { int fo = 0; for (size_t c = 0; c < charts.size(); ++c) for (size_t i = 0; i < charts[c].size(); ++i) fc2[fo++] = (int)c; }
    auto u8 = [](float x){ x = x*255.f; return (uint8_t)(x < 0 ? 0 : (x > 255 ? 255 : x)); };
    for (int f = 0; f < Fo; ++f) {
        const int idx[3] = { out.faces[3*f], out.faces[3*f+1], out.faces[3*f+2] };
        const float* n = &fnorm[3*charts[fc2[f]][0]];
        float px[3], py[3], pz[3];
        for (int j = 0; j < 3; ++j) {
            px[j] = out.uv[2*idx[j]] * T; py[j] = out.uv[2*idx[j]+1] * T;
            const float* p = &out.verts[3*idx[j]];
            pz[j] = p[0]*n[0]+p[1]*n[1]+p[2]*n[2];
        }
        int x0=(int)std::floor(std::min({px[0],px[1],px[2]})), x1=(int)std::ceil(std::max({px[0],px[1],px[2]}));
        int y0=(int)std::floor(std::min({py[0],py[1],py[2]})), y1=(int)std::ceil(std::max({py[0],py[1],py[2]}));
        x0=std::max(0,x0); y0=std::max(0,y0); x1=std::min(T-1,x1); y1=std::min(T-1,y1);
        const float d=(py[1]-py[2])*(px[0]-px[2])+(px[2]-px[1])*(py[0]-py[2]);
        if (std::fabs(d) < 1e-9f) continue;
        for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x) {
            const float fx=x+0.5f, fy=y+0.5f;
            const float w0=((py[1]-py[2])*(fx-px[2])+(px[2]-px[1])*(fy-py[2]))/d;
            const float w1=((py[2]-py[0])*(fx-px[2])+(px[0]-px[2])*(fy-py[2]))/d;
            const float w2=1-w0-w1;
            if (w0<-0.001f||w1<-0.001f||w2<-0.001f) continue;
            const size_t t=(size_t)y*T+x;
            const float dep = w0*pz[0]+w1*pz[1]+w2*pz[2];
            if (mask[t] && dep < zbuf[t]) continue;
            float s6[6];
            bool have = false;
            if (vs) {
                float p[3];
                for (int a = 0; a < 3; ++a)
                    p[a] = w0*out.verts[3*idx[0]+a] + w1*out.verts[3*idx[1]+a] + w2*out.verts[3*idx[2]+a];
                have = vs->sample(p, s6);
            }
            if (!have) continue;
            zbuf[t] = dep;
            out.base[4*t+0]=u8(s6[0]); out.base[4*t+1]=u8(s6[1]); out.base[4*t+2]=u8(s6[2]); out.base[4*t+3]=u8(s6[5]);
            out.mr[4*t+0]=0; out.mr[4*t+1]=u8(s6[4]); out.mr[4*t+2]=u8(s6[3]); out.mr[4*t+3]=255;
            mask[t]=1;
        }
    }
    dilate_full(out.base, out.mr, mask, T);
    printf("  uv_chart_project: atlas %dx%d, charts=%zu, Vo=%zu Fo=%d (scale %.1f tex/unit)\n",
           T, T, charts.size(), out.verts.size()/3, Fo, scale);
    fflush(stdout);
    return out;
}

BakedMesh uv_bake(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                  const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox) {
    BakedMesh out;
    std::unique_ptr<VoxSampler> vs;
    if (vox && vox->ok()) vs.reset(new VoxSampler(*vox));
    const int res = (vox && vox->ok()) ? vox->res : 1024;
    double post_t = post_now();

    // --- L1 (soup only, see below): duplicate-face marking. On voxel soup,
    // coincident faces create false planar regions and z-fighting geometry. On
    // a remeshed watertight mesh, dropping one copy of a coincident pair opens
    // boundary cracks (odd edge parity) and can keep the inward-facing copy --
    // which backface-culls on the single-sided export and reads as a HOLE. The
    // reference charts duplicates as-is in its remesh branch, so we do too.
    std::vector<uint8_t> sliver((size_t)F, 0);
    int n_sliver = 0, n_dup = 0;

    // --- L2: component triage. Sub-voxel greeble components would each pay
    // xatlas's full per-component overhead (the ~100k-component spray is a
    // large share of its blowup); they bypass charting into a reserved strip.
    std::vector<int> uf((size_t)V);
    for (int v = 0; v < V; ++v) uf[v] = v;
    std::function<int(int)> find = [&uf, &find](int x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };
    for (int f = 0; f < F; ++f) {
        const int r0 = find(faces[3*f]);
        const int r1 = find(faces[3*f+1]);
        const int r2 = find(faces[3*f+2]);
        uf[r1] = r0; uf[r2] = r0;
    }
    std::unordered_map<int, int> croot;
    std::vector<int> fcomp((size_t)F);
    struct CompStat { int fc = 0; float bmin[3] = {1e30f,1e30f,1e30f}, bmax[3] = {-1e30f,-1e30f,-1e30f}; double area = 0; };
    std::vector<CompStat> cstat;
    for (int f = 0; f < F; ++f) {
        const int r = find(faces[3*f]);
        auto it = croot.emplace(r, (int)cstat.size());
        if (it.second) cstat.emplace_back();
        const int c = it.first->second;
        fcomp[f] = c;
        CompStat& st = cstat[c];
        st.fc++;
        float e1[3], e2[3];
        for (int j = 0; j < 3; ++j) {
            const float* p = &verts[3*faces[3*f+j]];
            for (int k = 0; k < 3; ++k) { st.bmin[k] = std::min(st.bmin[k], p[k]); st.bmax[k] = std::max(st.bmax[k], p[k]); }
        }
        for (int k = 0; k < 3; ++k) { e1[k] = verts[3*faces[3*f+1]+k]-verts[3*faces[3*f]+k]; e2[k] = verts[3*faces[3*f+2]+k]-verts[3*faces[3*f]+k]; }
        const float cr0 = e1[1]*e2[2]-e1[2]*e2[1], cr1 = e1[2]*e2[0]-e1[0]*e2[2], cr2 = e1[0]*e2[1]-e1[1]*e2[0];
        st.area += 0.5 * std::sqrt(std::max(0.f, cr0*cr0+cr1*cr1+cr2*cr2));
    }
    // The strip triage exists for un-remeshed voxel soup (~100k sub-voxel
    // components would each pay xatlas's per-component overhead). A remeshed
    // mesh has a few hundred components; the reference charts all of them, and
    // stripping them out both wastes atlas area and opens welded-boundary
    // cracks, so triage engages only on soup-scale component counts.
    post_tick("bake: components", post_t);
    const bool soup = cstat.size() > 2000;
    if (soup) {   // L1 duplicate-face drop (see comment above): soup-scale meshes only
        std::unordered_map<uint64_t, uint8_t> seen;
        seen.reserve((size_t)F * 2);
        for (int f = 0; f < F; ++f) {
            int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
            int s0 = std::min({a,b,c}), s2 = std::max({a,b,c}), s1 = a + b + c - s0 - s2;
            const uint64_t k = ((uint64_t)(uint32_t)s0 * 0x9E3779B97F4A7C15ull) ^
                               ((uint64_t)(uint32_t)s1 * 0xC2B2AE3D27D4EB4Full) ^
                               ((uint64_t)(uint32_t)s2 * 0x165667B19E3779F9ull);
            auto ins = seen.emplace(k, 1);
            if (!ins.second) { sliver[f] = 1; ++n_dup; continue; }
        }
    }
    std::vector<uint8_t> comp_tiny(cstat.size(), 0);
    double area_big = 0, area_tiny = 0;
    for (size_t c = 0; c < cstat.size(); ++c) {
        const CompStat& st = cstat[c];
        const float dx = st.bmax[0]-st.bmin[0], dy = st.bmax[1]-st.bmin[1], dz = st.bmax[2]-st.bmin[2];
        const float diag = std::sqrt(dx*dx + dy*dy + dz*dz);
        comp_tiny[c] = (soup && (st.fc <= 32 || diag < 3.0f / res)) ? 1 : 0;
        (comp_tiny[c] ? area_tiny : area_big) += st.area;
    }
    std::vector<int> big_faces, tiny_faces;
    big_faces.reserve((size_t)F);
    for (int f = 0; f < F; ++f) {
        if (sliver[f]) continue;
        (comp_tiny[fcomp[f]] ? tiny_faces : big_faces).push_back(f);
    }
    if (big_faces.empty()) {
        big_faces.swap(tiny_faces);
        std::fill(comp_tiny.begin(), comp_tiny.end(), 0);
        area_big = area_tiny; area_tiny = 0;
    }
    int S = 0;
    if (!tiny_faces.empty() && area_tiny > 0) {
        const double frac = area_tiny / (area_big + area_tiny);
        S = (int)std::ceil(texsize * std::sqrt(frac * 1.5));
        S = std::max(16, std::min(texsize / 4, S));
    }
    const int TX = texsize - S;
    printf("  uv_bake: %zu components (%zu tiny, %d faces -> strip S=%d), %d duplicate faces dropped\n",
           cstat.size(), cstat.size() - (size_t)std::count(comp_tiny.begin(), comp_tiny.end(), 0),
           (int)tiny_faces.size(), S, n_dup);
    (void)n_sliver;
    fflush(stdout);

    // --- coarse pre-clustering of the big faces, a CPU port of cumesh
    // compute_charts (atlas.cu:1071-1210; spec 27 §6.2): bottom-up chart
    // merging. Every face starts as its own chart; each round, adjacent chart
    // pairs are scored
    //   cost = merged_cone_half_angle + 0.1·(areaA+areaB)
    //        + 1e-4·(perimA+perimB−2·shared)²/(areaA+areaB)
    // and a pair collapses iff it is the minimum-cost edge of BOTH charts and
    // cost ≤ π/2. Cones (axis = normalized Σ face normals, half-angle = max
    // deviation) recompute from scratch each round. Each resulting cluster
    // becomes its own xatlas mesh — packing remains global across meshes.
    std::vector<std::vector<int>> clusters;
    {
        const int NB = (int)big_faces.size();
        std::vector<float> fn((size_t)NB * 3), farea((size_t)NB);
        std::vector<int> chart((size_t)NB);
        for (int i = 0; i < NB; ++i) {
            const int f = big_faces[i];
            float e1[3], e2[3];
            for (int k = 0; k < 3; ++k) { e1[k] = verts[3*faces[3*f+1]+k]-verts[3*faces[3*f]+k]; e2[k] = verts[3*faces[3*f+2]+k]-verts[3*faces[3*f]+k]; }
            float n[3] = { e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0] };
            const float l = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            farea[i] = 0.5f * l;
            for (int k = 0; k < 3; ++k) fn[3*i+k] = l > 1e-20f ? n[k]/l : 0.f;
            chart[i] = i;
        }
        // manifold face-pair adjacency with edge lengths (local indices)
        struct Adj { int a, b; float len; };
        std::vector<Adj> fadj;
        {
            std::unordered_map<uint64_t, std::array<int,2>> edge2f;
            edge2f.reserve((size_t)NB * 2);
            auto ek = [](int a, int b) { return ((uint64_t)(uint32_t)std::min(a,b) << 32) | (uint32_t)std::max(a,b); };
            for (int i = 0; i < NB; ++i) {
                const int f = big_faces[i];
                for (int j = 0; j < 3; ++j) {
                    auto& e = edge2f[ek(faces[3*f+j], faces[3*f+(j+1)%3])];
                    if (e[0] == 0) e = {i + 1, 0};
                    else if (e[1] == 0) e[1] = i + 1;
                    else e[1] = -2;             // non-manifold: no adjacency
                }
            }
            fadj.reserve(edge2f.size());
            for (const auto& kv : edge2f) {
                const int a = kv.second[0] - 1, b = kv.second[1] - 1;
                if (a < 0 || b < 0) continue;
                const int v0 = (int)(kv.first >> 32), v1 = (int)(kv.first & 0xFFFFFFFFu);
                const float dx = verts[3*v0]-verts[3*v1], dy = verts[3*v0+1]-verts[3*v1+1], dz = verts[3*v0+2]-verts[3*v1+2];
                fadj.push_back({a, b, std::sqrt(dx*dx + dy*dy + dz*dz)});
            }
        }
        const float kAreaW = 0.1f, kPerimW = 1e-4f, kMaxCost = 1.5707963f;
        std::vector<float> flen((size_t)NB * 3);   // face-edge lengths, summed per chart every round
        for (int i = 0; i < NB; ++i) {
            const int f = big_faces[i];
            for (int j = 0; j < 3; ++j) {
                const int v0 = faces[3*f+j], v1 = faces[3*f+(j+1)%3];
                const float dx = verts[3*v0]-verts[3*v1], dy = verts[3*v0+1]-verts[3*v1+1], dz = verts[3*v0+2]-verts[3*v1+2];
                flen[(size_t)3*i+j] = std::sqrt(dx*dx + dy*dy + dz*dz);
            }
        }
        std::vector<int> remap((size_t)NB);
        std::vector<std::pair<uint64_t, float>> plen;   // (chart pair, edge length), reused across rounds
        for (int round = 0; round < 256; ++round) {
            // compress chart ids
            std::fill(remap.begin(), remap.end(), -1);
            int C = 0;
            for (int i = 0; i < NB; ++i) { int& r = remap[chart[i]]; if (r < 0) r = C++; }
            for (int i = 0; i < NB; ++i) chart[i] = remap[chart[i]];
            if (C <= 1) break;
            // per-chart cone / area / perimeter
            std::vector<float> axis((size_t)C * 3, 0.f), half((size_t)C, 0.f), carea((size_t)C, 0.f), cperim((size_t)C, 0.f);
            for (int i = 0; i < NB; ++i) {
                const int c = chart[i];
                for (int k = 0; k < 3; ++k) axis[3*c+k] += fn[3*i+k];
                carea[c] += farea[i];
            }
            for (int c = 0; c < C; ++c) {
                const float l = std::sqrt(axis[3*c]*axis[3*c]+axis[3*c+1]*axis[3*c+1]+axis[3*c+2]*axis[3*c+2]);
                if (l > 1e-20f) for (int k = 0; k < 3; ++k) axis[3*c+k] /= l;
                else axis[3*c+2] = 1.f;
            }
            for (int i = 0; i < NB; ++i) {
                const int c = chart[i];
                const float d = axis[3*c]*fn[3*i]+axis[3*c+1]*fn[3*i+1]+axis[3*c+2]*fn[3*i+2];
                half[c] = std::max(half[c], std::acos(std::min(1.f, std::max(-1.f, d))));
            }
            // chart-pair adjacency (shared boundary length) + perimeters. Pair
            // lengths are gathered then stable-sorted by pair, so each pair's
            // total is summed in adjacency order exactly as a per-pair
            // accumulator would (same float rounding), without a hash table.
            plen.clear();
            std::vector<float> shared_in((size_t)C, 0.f);
            for (const auto& ad : fadj) {
                const int c0 = chart[ad.a], c1 = chart[ad.b];
                if (c0 == c1) { shared_in[c0] += 2.f * ad.len; continue; }
                const uint64_t k = ((uint64_t)(uint32_t)std::min(c0,c1) << 32) | (uint32_t)std::max(c0,c1);
                plen.emplace_back(k, ad.len);
            }
            std::stable_sort(plen.begin(), plen.end(),
                             [](const std::pair<uint64_t,float>& x, const std::pair<uint64_t,float>& y) { return x.first < y.first; });
            // perimeter = total face-edge length − interior (same-chart manifold) edge length
            for (int i = 0; i < NB; ++i) {
                const int c = chart[i];
                for (int j = 0; j < 3; ++j) cperim[c] += flen[(size_t)3*i+j];
            }
            for (int c = 0; c < C; ++c) cperim[c] -= shared_in[c];
            // pair costs; per-chart argmin (pairs come out sorted by (c0, c1))
            struct Pair { int c0, c1; float len, cost; };
            std::vector<Pair> pairs;
            pairs.reserve(plen.size());
            for (size_t i = 0; i < plen.size();) {
                float sum = 0.f;
                size_t j = i;
                for (; j < plen.size() && plen[j].first == plen[i].first; ++j) sum += plen[j].second;
                pairs.push_back({(int)(plen[i].first >> 32), (int)(plen[i].first & 0xFFFFFFFFu), sum, 0.f});
                i = j;
            }
            for (auto& p : pairs) {
                const float ca = axis[3*p.c0]*axis[3*p.c1]+axis[3*p.c0+1]*axis[3*p.c1+1]+axis[3*p.c0+2]*axis[3*p.c1+2];
                const float aa = std::acos(std::min(1.f, std::max(-1.f, ca)));
                const float lo = std::min(-half[p.c0], aa - half[p.c1]);
                const float hi = std::max(half[p.c0], aa + half[p.c1]);
                const float na = carea[p.c0] + carea[p.c1];
                const float np = cperim[p.c0] + cperim[p.c1] - 2.f * p.len;
                p.cost = (hi - lo) * 0.5f + kAreaW * na + kPerimW * (np * np / na);
            }
            std::vector<int> cmin((size_t)C, -1);
            for (size_t e = 0; e < pairs.size(); ++e)
                for (int s = 0; s < 2; ++s) {
                    const int c = s ? pairs[e].c1 : pairs[e].c0;
                    if (cmin[c] < 0 || pairs[e].cost < pairs[cmin[c]].cost) cmin[c] = (int)e;
                }
            // collapse pairs that are the argmin of both endpoints
            std::vector<int> cmap((size_t)C);
            for (int c = 0; c < C; ++c) cmap[c] = c;
            bool any = false;
            for (size_t e = 0; e < pairs.size(); ++e) {
                const Pair& p = pairs[e];
                if (p.cost > kMaxCost) continue;
                if (cmin[p.c0] != (int)e || cmin[p.c1] != (int)e) continue;
                cmap[p.c1] = p.c0;
                any = true;
            }
            if (!any) break;
            for (int i = 0; i < NB; ++i) chart[i] = cmap[chart[i]];
        }
        // gather clusters (original face ids)
        std::fill(remap.begin(), remap.end(), -1);
        int C = 0;
        for (int i = 0; i < NB; ++i) { int& r = remap[chart[i]]; if (r < 0) r = C++; }
        clusters.resize((size_t)C);
        for (int i = 0; i < NB; ++i) clusters[remap[chart[i]]].push_back(big_faces[i]);
    }

    struct ClusterMesh {
        std::vector<float> v;
        std::vector<int32_t> fidx, l2orig;
    };
    std::vector<ClusterMesh> cms(clusters.size());
    {
        std::vector<int> v2l((size_t)V, -1);
        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            ClusterMesh& cm = cms[ci];
            const auto& cl = clusters[ci];
            cm.fidx.resize(cl.size() * 3);
            for (size_t i = 0; i < cl.size(); ++i) {
                const int f = cl[i];
                for (int j = 0; j < 3; ++j) {
                    const int v = faces[3*f+j];
                    if (v2l[v] < 0) {
                        v2l[v] = (int)(cm.v.size() / 3);
                        cm.v.insert(cm.v.end(), &verts[3*v], &verts[3*v] + 3);
                        cm.l2orig.push_back(v);
                    }
                    cm.fidx[3*i+j] = v2l[v];
                }
            }
            for (int32_t ov : cm.l2orig) v2l[ov] = -1;
        }
    }
    printf("  uv_bake: %zu merge clusters\n", clusters.size());
    if (std::getenv("TRELLIS_DBG_POST")) {
        std::vector<size_t> cs; for (auto& c : clusters) cs.push_back(c.size());
        std::sort(cs.rbegin(), cs.rend());
        fprintf(stderr, "    [post] cluster sizes (top 8):"); for (size_t i = 0; i < std::min<size_t>(8, cs.size()); ++i) fprintf(stderr, " %zu", cs[i]); fprintf(stderr, "\n");
    }
    post_tick("bake: pre-clustering", post_t);
    fflush(stdout);

    // Reference add_mesh passes positions only — no normals, no custom epsilon
    // (cumesh.py:453-458).
    if (std::getenv("TRELLIS_DBG_XATLAS")) xatlas::SetPrint(xatlas_print_stderr, true);
    xatlas::Atlas* atlas = xatlas::Create();
    for (auto& cm : cms) {
        xatlas::MeshDecl md;
        md.vertexCount = (uint32_t)(cm.v.size() / 3);
        md.vertexPositionData = cm.v.data();
        md.vertexPositionStride = 3 * sizeof(float);
        md.indexCount = (uint32_t)cm.fidx.size();
        md.indexData = cm.fidx.data();
        md.indexFormat = xatlas::IndexFormat::UInt32;
        if (xatlas::AddMesh(atlas, md, (uint32_t)cms.size()) != xatlas::AddMeshError::Success) {
            xatlas::Destroy(atlas);
            return out;
        }
    }
    xatlas::ChartOptions co;            // defaults
    // Hard wall-clock timeout in a worker thread: chart computation can wedge
    // without ever invoking its progress callback, so cooperative cancellation
    // is not enough. On timeout the worker (and the atlas) are abandoned —
    // state it touches is heap-owned, and the caller falls back to the
    // deterministic chart projection. Packing runs on the main thread after:
    // it is fast, re-entrant, and needed for the resolution cap below.
    auto* done = new std::atomic<bool>(false);
    xatlas::Atlas* atl = atlas;
    post_tick("bake: xatlas AddMesh", post_t);
    std::thread worker([atl, co, done] { xatlas::ComputeCharts(atl, co); done->store(true); });
    const auto tw0 = std::chrono::steady_clock::now();
    while (!done->load() &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count() < 300.0)
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (!done->load()) {
        fprintf(stderr, "  uv_bake: xatlas charting exceeded 300s; abandoning it\n");
        worker.detach();
        return out;
    }
    worker.join();
    delete done;

    // Reference packing (spec 27 §6.3): stock PackOptions — padding 0,
    // resolution 0 (xatlas grows a single ~1024² atlas at its own density
    // estimate), blockAlign false, bilinear true — and the binding then
    // divides UVs by the final atlas width/height, so charts always span the
    // FULL [0,1]² regardless of the packer's internal resolution. Density and
    // occupancy at the bake size come from this normalization, not from
    // packing at the bake resolution; chart-to-chart bleed at 0 padding is
    // handled by the inpaint (as in the reference).
    xatlas::PackOptions po;
    post_tick("bake: xatlas ComputeCharts", post_t);
    xatlas::PackCharts(atlas, po);
    if (atlas->meshCount == 0 || atlas->width == 0) { xatlas::Destroy(atlas); return out; }

    const int W = (int)atlas->width, H = (int)atlas->height;
    const int T = texsize;
    // uscale < 1 only when a strip band is reserved (soup fallback)
    const float uscale = (float)TX / (float)T;
    const float unorm = uscale / (float)W, vnorm = uscale / (float)H;

    post_tick("bake: xatlas PackCharts", post_t);
    // --- big part: re-indexed atlas vertices per cluster mesh. Faces xatlas
    // ignored internally (degenerate/zero-area, uv(0,0), atlasIndex −1) and
    // faces of failed (Invalid) charts are KEPT — dropping them opens welded-
    // boundary cracks — but emitted point-collapsed onto one charted twin UV,
    // so they sample a single texel of a neighboring chart instead of
    // painting atlas-spanning wedges.
    std::vector<float> vp;
    std::vector<std::vector<int>> strip_groups;
    size_t n_collapsed = 0;
    for (uint32_t mi = 0; mi < atlas->meshCount; ++mi) {
        const xatlas::Mesh& m = atlas->meshes[mi];
        const ClusterMesh& cm = cms[mi];
        const int base = (int)(out.verts.size() / 3);
        const int Vo = (int)m.vertexCount;
        std::unordered_map<uint32_t, std::pair<float,float>> twin;
        twin.reserve((size_t)Vo);
        for (int i = 0; i < Vo; ++i)
            if (m.vertexArray[i].atlasIndex >= 0)
                twin.emplace(m.vertexArray[i].xref,
                             std::make_pair(m.vertexArray[i].uv[0], m.vertexArray[i].uv[1]));
        for (int i = 0; i < Vo; ++i) {
            const uint32_t xref = m.vertexArray[i].xref;
            const int orig = cm.l2orig[xref];
            out.verts.insert(out.verts.end(), &verts[3*orig], &verts[3*orig] + 3);
            float u = m.vertexArray[i].uv[0], v = m.vertexArray[i].uv[1];
            if (m.vertexArray[i].atlasIndex < 0) {
                auto it = twin.find(xref);
                if (it != twin.end()) { u = it->second.first; v = it->second.second; }
                else { u = 0.f; v = 0.f; }
            }
            out.uv.push_back(u * unorm);
            out.uv.push_back(v * vnorm);
            if (!pbr6.empty()) vp.insert(vp.end(), &pbr6[6*orig], &pbr6[6*orig] + 6);
        }
        std::vector<uint8_t> charted(m.indexCount / 3, 0);
        for (uint32_t ci = 0; ci < m.chartCount; ++ci) {
            if (m.chartArray[ci].type == xatlas::ChartType::Invalid) continue;
            for (uint32_t k = 0; k < m.chartArray[ci].faceCount; ++k)
                charted[m.chartArray[ci].faceArray[k]] = 1;
        }
        for (uint32_t fi = 0; fi < m.indexCount / 3; ++fi) {
            if (charted[fi]) {
                for (int j = 0; j < 3; ++j)
                    out.faces.push_back((int32_t)(base + m.indexArray[3*fi+j]));
                continue;
            }
            ++n_collapsed;
            float cu = 0.f, cv = 0.f;
            bool found = false;
            for (int j = 0; j < 3 && !found; ++j) {
                auto it = twin.find(m.vertexArray[m.indexArray[3*fi+j]].xref);
                if (it != twin.end()) { cu = it->second.first * unorm; cv = it->second.second * vnorm; found = true; }
            }
            for (int j = 0; j < 3; ++j) {
                const int orig = cm.l2orig[m.vertexArray[m.indexArray[3*fi+j]].xref];
                out.faces.push_back((int32_t)(out.verts.size() / 3));
                out.verts.insert(out.verts.end(), &verts[3*orig], &verts[3*orig] + 3);
                out.uv.push_back(cu);
                out.uv.push_back(cv);
                if (!pbr6.empty()) vp.insert(vp.end(), &pbr6[6*orig], &pbr6[6*orig] + 6);
            }
        }
    }
    if (n_collapsed)
        printf("  uv_bake: %zu uncharted faces kept with point-collapsed UVs\n", n_collapsed);

    post_tick("bake: reindex", post_t);
    // --- tiny components: one planar chart each, shelf-packed into the
    // reserved strip (bottom band + right band) at the big charts' density ---
    {
        std::vector<std::vector<int>> tcomp_faces(cstat.size());
        for (int f : tiny_faces) tcomp_faces[fcomp[f]].push_back(f);
        for (auto& g : tcomp_faces)
            if (!g.empty()) strip_groups.push_back(std::move(g));
    }
    if (S == 0 && !strip_groups.empty()) S = std::min(texsize / 8, 64);
    if (S > 0 && !strip_groups.empty()) {
        // big charts' effective density at the bake size (packer texels/unit
        // scaled by the packer→bake normalization)
        const float density = std::max(1.f, atlas->texelsPerUnit * (float)TX / (float)std::max(W, H));
        struct TC { int comp; float bu[3], bv[3]; float umin, vmin; int w, h; int ox = 0, oy = 0; };
        std::vector<TC> tcs;
        for (size_t c = 0; c < strip_groups.size(); ++c) {
            if (strip_groups[c].empty()) continue;
            TC tc; tc.comp = (int)c;
            double an[3] = {0, 0, 0};
            for (int f : strip_groups[c]) {
                float e1[3], e2[3];
                for (int k = 0; k < 3; ++k) { e1[k] = verts[3*faces[3*f+1]+k]-verts[3*faces[3*f]+k]; e2[k] = verts[3*faces[3*f+2]+k]-verts[3*faces[3*f]+k]; }
                an[0] += e1[1]*e2[2]-e1[2]*e2[1]; an[1] += e1[2]*e2[0]-e1[0]*e2[2]; an[2] += e1[0]*e2[1]-e1[1]*e2[0];
            }
            float n[3] = {(float)an[0], (float)an[1], (float)an[2]};
            float l = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            if (l < 1e-20f) { n[0] = 0; n[1] = 0; n[2] = 1; l = 1; }
            for (int k = 0; k < 3; ++k) n[k] /= l;
            float up[3] = {0, 0, 1};
            if (std::fabs(n[2]) > 0.9f) { up[0] = 1; up[2] = 0; }
            tc.bu[0] = up[1]*n[2]-up[2]*n[1]; tc.bu[1] = up[2]*n[0]-up[0]*n[2]; tc.bu[2] = up[0]*n[1]-up[1]*n[0];
            l = std::sqrt(tc.bu[0]*tc.bu[0]+tc.bu[1]*tc.bu[1]+tc.bu[2]*tc.bu[2]);
            for (int k = 0; k < 3; ++k) tc.bu[k] = l > 1e-20f ? tc.bu[k]/l : (k == 0 ? 1.f : 0.f);
            tc.bv[0] = n[1]*tc.bu[2]-n[2]*tc.bu[1]; tc.bv[1] = n[2]*tc.bu[0]-n[0]*tc.bu[2]; tc.bv[2] = n[0]*tc.bu[1]-n[1]*tc.bu[0];
            float umin = 1e30f, umax = -1e30f, vmin = 1e30f, vmax = -1e30f;
            for (int f : strip_groups[c])
                for (int j = 0; j < 3; ++j) {
                    const float* p = &verts[3*faces[3*f+j]];
                    const float u = p[0]*tc.bu[0]+p[1]*tc.bu[1]+p[2]*tc.bu[2];
                    const float v = p[0]*tc.bv[0]+p[1]*tc.bv[1]+p[2]*tc.bv[2];
                    umin = std::min(umin, u); umax = std::max(umax, u);
                    vmin = std::min(vmin, v); vmax = std::max(vmax, v);
                }
            tc.umin = umin; tc.vmin = vmin;
            tc.w = std::max(2, (int)std::ceil((umax-umin) * density));
            tc.h = std::max(2, (int)std::ceil((vmax-vmin) * density));
            tc.w = std::min(tc.w, S - 2); tc.h = std::min(tc.h, S - 2);
            tcs.push_back(tc);
        }
        std::sort(tcs.begin(), tcs.end(), [](const TC& a, const TC& b) { return a.h > b.h; });
        float dscale = 1.f;
        auto pack_strip = [&](float ds) -> bool {
            int x = 1, y = TX + 1, shelf = 0;
            size_t i = 0;
            for (; i < tcs.size(); ++i) {
                const int w = std::max(2, (int)(tcs[i].w * ds)), h = std::max(2, (int)(tcs[i].h * ds));
                if (x + w + 1 > T) { x = 1; y += shelf + 1; shelf = 0; }
                if (y + h + 1 > T) break;
                tcs[i].ox = x; tcs[i].oy = y;
                x += w + 1; shelf = std::max(shelf, h);
            }
            if (i == tcs.size()) return true;
            x = TX + 1; y = 1; shelf = 0;
            for (; i < tcs.size(); ++i) {
                const int w = std::max(2, (int)(tcs[i].w * ds)), h = std::max(2, (int)(tcs[i].h * ds));
                if (x + w + 1 > T) { x = TX + 1; y += shelf + 1; shelf = 0; }
                if (y + h + 1 > TX) return false;
                tcs[i].ox = x; tcs[i].oy = y;
                x += w + 1; shelf = std::max(shelf, h);
            }
            return true;
        };
        while (dscale > 0.1f && !pack_strip(dscale)) dscale *= 0.7f;
        int overflow = 0;
        for (auto& tc : tcs) if (tc.ox == 0 && tc.oy == 0) ++overflow;
        if (overflow) fprintf(stderr, "  uv_bake: %d tiny charts overflowed the strip (sharing texels)\n", overflow);
        std::vector<int> vstamp((size_t)V, -1), vlocal((size_t)V, -1);
        for (const auto& tc : tcs) {
            const float ds = dscale;
            for (int f : strip_groups[tc.comp]) {
                int lidx[3];
                for (int j = 0; j < 3; ++j) {
                    const int v = faces[3*f+j];
                    if (vstamp[v] != tc.comp) {
                        vstamp[v] = tc.comp;
                        vlocal[v] = (int)(out.verts.size() / 3);
                        const float* p = &verts[3*v];
                        out.verts.insert(out.verts.end(), p, p + 3);
                        const float u = p[0]*tc.bu[0]+p[1]*tc.bu[1]+p[2]*tc.bu[2];
                        const float w = p[0]*tc.bv[0]+p[1]*tc.bv[1]+p[2]*tc.bv[2];
                        out.uv.push_back((tc.ox + (u - tc.umin) * density * ds) / T);
                        out.uv.push_back((tc.oy + (w - tc.vmin) * density * ds) / T);
                        if (!pbr6.empty()) vp.insert(vp.end(), &pbr6[6*v], &pbr6[6*v] + 6);
                    }
                    lidx[j] = vlocal[v];
                }
                for (int j = 0; j < 3; ++j) out.faces.push_back(lidx[j]);
            }
        }
    }
    const int FoAll = (int)out.faces.size() / 3;

    post_tick("bake: tiny charts", post_t);
    // --- rasterize into the atlas ---
    out.T = T;
    out.base.assign((size_t)T * T * 4, 0);
    out.mr.assign((size_t)T * T * 4, 0);
    std::vector<uint8_t> mask((size_t)T * T, 0);
    auto u8 = [](float v){ v = v*255.f; return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); };
    // Row bands in parallel: every band walks all faces in order and rasterizes
    // its rows only, so each texel is written by one thread in the same face
    // order as a serial pass (later faces still win overlapping texels).
    parallel_for(T, [&](int64_t yb, int64_t ye) {
    for (int f = 0; f < FoAll; ++f) {
        int idx[3] = { (int)out.faces[3*f], (int)out.faces[3*f+1], (int)out.faces[3*f+2] };
        float px[3], py[3];
        for (int j = 0; j < 3; ++j) { px[j] = out.uv[2*idx[j]] * T; py[j] = out.uv[2*idx[j]+1] * T; }
        int x0 = (int)std::floor(std::min({px[0],px[1],px[2]})), x1 = (int)std::ceil(std::max({px[0],px[1],px[2]}));
        int y0 = (int)std::floor(std::min({py[0],py[1],py[2]})), y1 = (int)std::ceil(std::max({py[0],py[1],py[2]}));
        x0 = std::max(0,x0); y0 = std::max((int)yb,y0); x1 = std::min(T-1,x1); y1 = std::min((int)ye-1,y1);
        if (y0 > y1) continue;
        float d = (py[1]-py[2])*(px[0]-px[2]) + (px[2]-px[1])*(py[0]-py[2]);
        if (std::fabs(d) < 1e-9f) continue;
        for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) {
            float fx = x + 0.5f, fy = y + 0.5f;
            float w0 = ((py[1]-py[2])*(fx-px[2]) + (px[2]-px[1])*(fy-py[2]))/d;
            float w1 = ((py[2]-py[0])*(fx-px[2]) + (px[0]-px[2])*(fy-py[2]))/d;
            float w2 = 1 - w0 - w1;
            if (w0 < -0.001f || w1 < -0.001f || w2 < -0.001f) continue;
            size_t t = (size_t)y * T + x;
            float s[6];
            bool have = false;
            if (vs) {
                float p[3];
                for (int a = 0; a < 3; ++a)
                    p[a] = w0*out.verts[3*idx[0]+a] + w1*out.verts[3*idx[1]+a] + w2*out.verts[3*idx[2]+a];
                have = vs->sample(p, s);
            } else if (!pbr6.empty()) {
                for (int k = 0; k < 6; ++k)
                    s[k] = w0*vp[6*idx[0]+k]+w1*vp[6*idx[1]+k]+w2*vp[6*idx[2]+k];
                have = true;
            }
            if (have) {
                out.base[4*t+0]=u8(s[0]); out.base[4*t+1]=u8(s[1]); out.base[4*t+2]=u8(s[2]); out.base[4*t+3]=u8(s[5]);
                out.mr[4*t+0]=0; out.mr[4*t+1]=u8(s[4]); out.mr[4*t+2]=u8(s[3]); out.mr[4*t+3]=255;
                mask[t] = 1;
            }
        }
    }
    });
    post_tick("bake: rasterize", post_t);
    xatlas::Destroy(atlas);

    telea_inpaint(out.base, out.mr, mask, T, 3, 1);
    post_tick("bake: telea inpaint", post_t);
    printf("  uv_bake: atlas %dx%d (xatlas %dx%d), Vo=%zu Fo=%d\n", T, T, W, H, out.verts.size()/3, FoAll);
    fflush(stdout);
    return out;
}

BakedMesh uv_box_project(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                         const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox) {
    BakedMesh out;
    const int T = texsize;
    if (F == 0) return out;
    std::unique_ptr<VoxSampler> vs;
    if (vox && vox->ok()) vs.reset(new VoxSampler(*vox));

    // in-plane axes for each dominant axis: X->(Y,Z), Y->(X,Z), Z->(X,Y)
    static const int UAX[3] = {1, 0, 0}, VAX[3] = {2, 2, 1};

    // 1) occlusion-aware bucket assignment. A face projected along its dominant
    // normal axis may be hidden behind another surface (chin over neck, ear over
    // shoulder); with a shared texel the hidden face samples the occluder's
    // colors, which is the classic box-projection "dirt". Build min/max depth
    // maps per axis from ALL faces (backfaces occlude too), then assign each
    // face to its best axis (by |n|, sign from n) along which it is actually
    // the visible surface. Faces visible along no axis keep their dominant
    // bucket and lose the later depth-tested raster fairly.
    std::vector<float> fnorm((size_t)F * 3);
    float mean_edge = 0.f;
    for (int f = 0; f < F; ++f) {
        const int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        float e1[3], e2[3];
        for (int k = 0; k < 3; ++k) { e1[k] = verts[3*b+k]-verts[3*a+k]; e2[k] = verts[3*c+k]-verts[3*a+k]; }
        fnorm[3*f+0]=e1[1]*e2[2]-e1[2]*e2[1]; fnorm[3*f+1]=e1[2]*e2[0]-e1[0]*e2[2]; fnorm[3*f+2]=e1[0]*e2[1]-e1[1]*e2[0];
        mean_edge += std::sqrt(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]);
    }
    mean_edge /= (float)F;
    const float eps = 2.f * mean_edge;

    // global projected bbox per axis (depth maps cover the whole mesh)
    float gmin[3][2], gmax[3][2];
    for (int ax = 0; ax < 3; ++ax) { gmin[ax][0]=gmin[ax][1]=1e30f; gmax[ax][0]=gmax[ax][1]=-1e30f; }
    for (int v = 0; v < V; ++v)
        for (int ax = 0; ax < 3; ++ax) {
            const float uu = verts[3*v+UAX[ax]], vv = verts[3*v+VAX[ax]];
            gmin[ax][0]=std::min(gmin[ax][0],uu); gmax[ax][0]=std::max(gmax[ax][0],uu);
            gmin[ax][1]=std::min(gmin[ax][1],vv); gmax[ax][1]=std::max(gmax[ax][1],vv);
        }

    const int D = 1024;
    std::vector<float> dmin((size_t)3*D*D, 1e30f), dmax((size_t)3*D*D, -1e30f);
    auto dtex = [&](int ax, float uu, float vv, float& fx, float& fy) {
        fx = (uu - gmin[ax][0]) / std::max(1e-6f, gmax[ax][0]-gmin[ax][0]) * (D-1);
        fy = (vv - gmin[ax][1]) / std::max(1e-6f, gmax[ax][1]-gmin[ax][1]) * (D-1);
    };
    for (int f = 0; f < F; ++f) {
        const int i0 = faces[3*f], i1 = faces[3*f+1], i2 = faces[3*f+2];
        for (int ax = 0; ax < 3; ++ax) {
            float px[3], py[3], pd[3];
            const int vi[3] = {i0, i1, i2};
            for (int j = 0; j < 3; ++j) {
                dtex(ax, verts[3*vi[j]+UAX[ax]], verts[3*vi[j]+VAX[ax]], px[j], py[j]);
                pd[j] = verts[3*vi[j]+ax];
            }
            int x0=(int)std::floor(std::min({px[0],px[1],px[2]})), x1=(int)std::ceil(std::max({px[0],px[1],px[2]}));
            int y0=(int)std::floor(std::min({py[0],py[1],py[2]})), y1=(int)std::ceil(std::max({py[0],py[1],py[2]}));
            x0=std::max(0,x0); y0=std::max(0,y0); x1=std::min(D-1,x1); y1=std::min(D-1,y1);
            float d=(py[1]-py[2])*(px[0]-px[2])+(px[2]-px[1])*(py[0]-py[2]); if (std::fabs(d)<1e-9f) continue;
            for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x){
                float fx=x+0.5f, fy=y+0.5f;
                float w0=((py[1]-py[2])*(fx-px[2])+(px[2]-px[1])*(fy-py[2]))/d;
                float w1=((py[2]-py[0])*(fx-px[2])+(px[0]-px[2])*(fy-py[2]))/d;
                float w2=1-w0-w1;
                if (w0<-0.05f||w1<-0.05f||w2<-0.05f) continue;
                const float dep = w0*pd[0]+w1*pd[1]+w2*pd[2];
                const size_t t = (size_t)ax*D*D + (size_t)y*D + x;
                dmin[t]=std::min(dmin[t],dep); dmax[t]=std::max(dmax[t],dep);
            }
        }
    }

    auto visible = [&](int f, int ax, int dir)->bool {
        const int vi[3] = {faces[3*f], faces[3*f+1], faces[3*f+2]};
        float cu=0, cv=0, cd=0;
        int pass = 0;
        auto probe = [&](float uu, float vv, float dep)->bool {
            float fx, fy; dtex(ax, uu, vv, fx, fy);
            int x = std::min(D-1, std::max(0, (int)fx)), y = std::min(D-1, std::max(0, (int)fy));
            const size_t t = (size_t)ax*D*D + (size_t)y*D + x;
            return dir == 0 ? (dep >= dmax[t] - eps) : (dep <= dmin[t] + eps);
        };
        for (int j = 0; j < 3; ++j) {
            const float uu=verts[3*vi[j]+UAX[ax]], vv=verts[3*vi[j]+VAX[ax]], dep=verts[3*vi[j]+ax];
            cu+=uu/3; cv+=vv/3; cd+=dep/3;
            if (probe(uu, vv, dep)) ++pass;
        }
        return probe(cu, cv, cd) || pass >= 2;
    };

    // A fallback axis is usable only when the face still has a healthy normal
    // component along it — a near-edge-on face projects to a sliver that
    // rasterizes almost no texels and then samples neighboring charts' colors.
    // Faces hidden along every usable axis (scalp under a raised weapon, chest
    // behind a held prop) go to a second-layer bucket of their dominant axis:
    // stacked surfaces cannot share one projected texel, so the hidden layer
    // gets its own chart. Buckets 0..5 = visible, 6..11 = occluded layer.
    std::vector<int> fbucket((size_t)F);
    int reassigned = 0, layered = 0;
    for (int f = 0; f < F; ++f) {
        const float* n = &fnorm[3*f];
        const float nlen = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        int order[3] = {0, 1, 2};
        std::sort(order, order+3, [&](int a, int b){ return std::fabs(n[a]) > std::fabs(n[b]); });
        const int primary = order[0]*2 + (n[order[0]] >= 0 ? 0 : 1);
        fbucket[f] = 6 + primary;
        for (int k = 0; k < 3; ++k) {
            const int ax = order[k], dir = n[ax] >= 0 ? 0 : 1;
            if (k > 0 && std::fabs(n[ax]) < 0.35f * nlen) break;
            if (visible(f, ax, dir)) { fbucket[f] = ax*2 + dir; if (k > 0) ++reassigned; break; }
        }
        if (fbucket[f] >= 6) ++layered;
    }

    // 2) per-bucket: dedup vertices used, compute projected (u,v) bbox
    const int NB = 12;
    std::vector<int> vmap((size_t)V * NB, -1);            // (vertex,bucket) -> local index
    std::vector<int> bxref[12];                           // local -> original vertex index
    float umin[12], umax[12], vmin[12], vmax[12];
    for (int g = 0; g < NB; ++g) { umin[g]=vmin[g]=1e30f; umax[g]=vmax[g]=-1e30f; }
    auto proj = [&](int v, int g, float& uu, float& vv){ int ax=(g%6)/2; uu=verts[3*v+UAX[ax]]; vv=verts[3*v+VAX[ax]]; };
    auto local = [&](int v, int g)->int {
        int& li = vmap[(size_t)v*NB + g];
        if (li < 0) { li = (int)bxref[g].size(); bxref[g].push_back(v);
            float uu,vv; proj(v,g,uu,vv);
            umin[g]=std::min(umin[g],uu); umax[g]=std::max(umax[g],uu);
            vmin[g]=std::min(vmin[g],vv); vmax[g]=std::max(vmax[g],vv); }
        return li;
    };
    std::vector<std::array<int,3>> bfaces[12];
    for (int f = 0; f < F; ++f) { int g = fbucket[f];
        bfaces[g].push_back({ local(faces[3*f],g), local(faces[3*f+1],g), local(faces[3*f+2],g) }); }

    // 3) atlas layout: 12 buckets (6 visible + 6 occluded-layer) in a
    // 4 (cols) x 3 (rows) grid, aspect-preserving fit + 2px pad
    const int CW = T/4, CH = T/3, pad = 2;
    int base_local[12];                                   // output-vertex offset per bucket
    int Vo = 0; for (int g = 0; g < NB; ++g) { base_local[g] = Vo; Vo += (int)bxref[g].size(); }
    out.verts.resize((size_t)Vo*3); out.uv.resize((size_t)Vo*2);
    std::vector<float> vp(pbr6.empty()?0:(size_t)Vo*6);
    for (int g = 0; g < NB; ++g) {
        const int col = g % 4, row = g / 4;
        const float ox = (float)col*CW + pad, oy = (float)row*CH + pad;
        const float bw = std::max(1e-6f, umax[g]-umin[g]), bh = std::max(1e-6f, vmax[g]-vmin[g]);
        const float s = std::min((CW-2*pad)/bw, (CH-2*pad)/bh);   // uniform scale, no stretch
        for (int i = 0; i < (int)bxref[g].size(); ++i) {
            const int v = bxref[g][i], o = base_local[g] + i;
            for (int k = 0; k < 3; ++k) out.verts[3*o+k] = verts[3*v+k];
            float uu,vv; proj(v,g,uu,vv);
            out.uv[2*o+0] = (ox + (uu-umin[g])*s) / T;
            out.uv[2*o+1] = (oy + (vv-vmin[g])*s) / T;
            if (!pbr6.empty()) for (int k = 0; k < 6; ++k) vp[6*o+k] = pbr6[6*v+k];
        }
    }
    int Fo = 0; for (int g = 0; g < NB; ++g) Fo += (int)bfaces[g].size();
    out.faces.resize((size_t)Fo*3);
    { int fo = 0; for (int g = 0; g < NB; ++g) for (auto& t : bfaces[g]) {
        out.faces[3*fo+0]=base_local[g]+t[0]; out.faces[3*fo+1]=base_local[g]+t[1]; out.faces[3*fo+2]=base_local[g]+t[2]; ++fo; } }

    // 4) depth-tested rasterization: on residual within-bucket overlap the
    // surface nearest the projection direction owns the texel (outermost wins),
    // instead of whatever face happened to rasterize last.
    out.T = T; out.base.assign((size_t)T*T*4,0); out.mr.assign((size_t)T*T*4,0);
    std::vector<uint8_t> mask((size_t)T*T,0);
    std::vector<float> zbuf((size_t)T*T, -1e30f);
    std::vector<int> fgroup((size_t)Fo);
    { int fo = 0; for (int g = 0; g < NB; ++g) for (size_t i = 0; i < bfaces[g].size(); ++i) fgroup[fo++] = g; }
    auto u8 = [](float v){ v=v*255.f; return (uint8_t)(v<0?0:(v>255?255:v)); };
    for (int f = 0; f < Fo; ++f) {
        int idx[3] = { out.faces[3*f], out.faces[3*f+1], out.faces[3*f+2] };
        const int gax = (fgroup[f]%6)/2;
        const float gsign = (fgroup[f]%2 == 0) ? 1.f : -1.f;
        float px[3], py[3], pz[3];
        for (int j=0;j<3;++j){ px[j]=out.uv[2*idx[j]]*T; py[j]=out.uv[2*idx[j]+1]*T;
            pz[j]=gsign*out.verts[3*idx[j]+gax]; }
        int x0=(int)std::floor(std::min({px[0],px[1],px[2]})), x1=(int)std::ceil(std::max({px[0],px[1],px[2]}));
        int y0=(int)std::floor(std::min({py[0],py[1],py[2]})), y1=(int)std::ceil(std::max({py[0],py[1],py[2]}));
        x0=std::max(0,x0); y0=std::max(0,y0); x1=std::min(T-1,x1); y1=std::min(T-1,y1);
        float d=(py[1]-py[2])*(px[0]-px[2])+(px[2]-px[1])*(py[0]-py[2]); if (std::fabs(d)<1e-9f) continue;
        for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x){
            float fx=x+0.5f, fy=y+0.5f;
            float w0=((py[1]-py[2])*(fx-px[2])+(px[2]-px[1])*(fy-py[2]))/d;
            float w1=((py[2]-py[0])*(fx-px[2])+(px[0]-px[2])*(fy-py[2]))/d;
            float w2=1-w0-w1;
            if (w0<-0.001f||w1<-0.001f||w2<-0.001f) continue;
            size_t t=(size_t)y*T+x;
            const float dep = w0*pz[0]+w1*pz[1]+w2*pz[2];
            if (mask[t] && dep < zbuf[t]) continue;
            float s[6];
            bool have = false;
            if (vs) {
                float p[3];
                for (int a = 0; a < 3; ++a)
                    p[a] = w0*out.verts[3*idx[0]+a] + w1*out.verts[3*idx[1]+a] + w2*out.verts[3*idx[2]+a];
                have = vs->sample(p, s);
            } else if (!pbr6.empty()) {
                for (int k = 0; k < 6; ++k)
                    s[k] = w0*vp[6*idx[0]+k]+w1*vp[6*idx[1]+k]+w2*vp[6*idx[2]+k];
                have = true;
            }
            if (!have) continue;
            zbuf[t] = dep;
            out.base[4*t+0]=u8(s[0]); out.base[4*t+1]=u8(s[1]); out.base[4*t+2]=u8(s[2]); out.base[4*t+3]=u8(s[5]);
            out.mr[4*t+0]=0; out.mr[4*t+1]=u8(s[4]); out.mr[4*t+2]=u8(s[3]); out.mr[4*t+3]=255;
            mask[t]=1;
        }
    }
    // 5) seam dilation
    dilate_full(out.base, out.mr, mask, T);
    printf("  uv_box_project: atlas %dx%d, Vo=%d Fo=%d (6 planes, %d re-bucketed, %d occluded-layer)\n",
           T, T, Vo, Fo, reassigned, layered);
    return out;
}

} // namespace trellis
