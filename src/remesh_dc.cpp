#include "remesh_dc.h"
#include "parallel_for.h"
#include "tri_bvh.h"
#include <algorithm>
#include <atomic>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cmath>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>

// TRELLIS_DBG_POST=1: per-phase wall times to stderr (debug logging only).
static double post_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void post_tick(const char* tag, double& t) {
    static const bool on = std::getenv("TRELLIS_DBG_POST") != nullptr;
    if (!on) return;
    const double n = post_now();
    fprintf(stderr, "    [post] %-28s %.2fs\n", tag, n - t);
    t = n;
}

namespace trellis {

namespace {

inline int ctz64(uint64_t v) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward64(&i, v);
    return (int)i;
#else
    return __builtin_ctzll(v);
#endif
}

inline int popcnt64(uint64_t v) {
#ifdef _MSC_VER
    return (int)__popcnt64(v);
#else
    return __builtin_popcountll(v);
#endif
}

// Dense occupancy bitset over a grid with O(1) rank (index among the set bits,
// in linear order). Stands in for the cell -> dense-index hash maps: 1/8 byte per
// grid cell instead of a hash node per occupied cell.
struct RankBitset {
    std::vector<uint64_t> bits;
    std::vector<int32_t> rank0;   // set bits before each word
    int64_t count = 0;
    void init(int64_t nbits) { bits.assign((size_t)((nbits + 63) / 64), 0); }
    void set(int64_t i) { bits[(size_t)(i >> 6)] |= 1ull << (i & 63); }
    void finalize() {
        rank0.resize(bits.size());
        int64_t c = 0;
        for (size_t w = 0; w < bits.size(); ++w) { rank0[w] = (int32_t)c; c += popcnt64(bits[w]); }
        count = c;
    }
    int find(int64_t i) const {
        const uint64_t w = bits[(size_t)(i >> 6)];
        const int b = (int)(i & 63);
        if (!((w >> b) & 1)) return -1;
        return rank0[(size_t)(i >> 6)] + popcnt64(w & ((1ull << b) - 1));
    }
};

}  // namespace

Mesh remesh_narrow_band_dc(const float* iverts, int64_t iV, const int32_t* ifaces, int64_t iF,
                           const TriBvh& bvh, int res, int band, float project_back) {
    (void)iV;
    Mesh out;
    if (iF == 0 || bvh.empty() || res <= 0) return out;

    // Reference domain: the world cube is inflated by (res+3·band)/res so the
    // offset shell never touches the boundary; eps is the offset distance.
    const float scale = (float)(res + 3 * band) / (float)res;
    const float cell = scale / (float)res;
    const float eps = (float)band * cell;
    const float keep = 0.87f * cell;
    double post_t = post_now();

    // Candidate cells: conservative dilation of every triangle's AABB by the
    // band-plus-crossing radius, marked in a res^3 bitset. Threads OR into the one
    // shared bitset atomically (commutative, so the result is thread-count
    // independent) instead of each owning a res^3/8-byte copy that is merged after.
    const int64_t nbits = (int64_t)res * res * res;
    const size_t nwords = (size_t)((nbits + 63) / 64);
    std::vector<std::atomic<uint64_t>> cand_at(nwords);
    parallel_for((int64_t)nwords, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) cand_at[(size_t)i].store(0, std::memory_order_relaxed);
    });
    // A cell is active iff UDF(center) < (band+0.87)·cell, and its closest
    // surface point lies inside some triangle-marked cell, so the per-axis
    // index distance is < band+1.37, i.e. ≤ band+1.
    const int dil = band + 1;
    parallel_for(iF, [&](int64_t fb, int64_t fe) {
        auto setb = [&](int x, int y, int z) {
            const int64_t i = ((int64_t)x * res + y) * res + z;
            cand_at[(size_t)(i >> 6)].fetch_or(1ull << (i & 63), std::memory_order_relaxed);
        };
        for (int64_t f = fb; f < fe; ++f) {
            float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
            for (int j = 0; j < 3; ++j) {
                const float* p = &iverts[3 * ifaces[3*f+j]];
                for (int k = 0; k < 3; ++k) {
                    bmin[k] = std::min(bmin[k], p[k]);
                    bmax[k] = std::max(bmax[k], p[k]);
                }
            }
            int c0[3], c1[3];
            for (int k = 0; k < 3; ++k) {
                c0[k] = std::max(0, (int)std::floor((bmin[k] / scale + 0.5f) * res) - dil);
                c1[k] = std::min(res - 1, (int)std::floor((bmax[k] / scale + 0.5f) * res) + dil);
            }
            for (int x = c0[0]; x <= c1[0]; ++x)
                for (int y = c0[1]; y <= c1[1]; ++y)
                    for (int z = c0[2]; z <= c1[2]; ++z) setb(x, y, z);
        }
    });
    std::vector<uint64_t> cand(nwords);
    for (size_t i = 0; i < nwords; ++i) cand[i] = cand_at[i].load(std::memory_order_relaxed);
    { std::vector<std::atomic<uint64_t>>().swap(cand_at); }

    post_tick("remesh: candidate bitset", post_t);
    // Active voxels: |UDF(center) - eps| < 0.87*cell (spec 27 §4.1).
    std::vector<int> acoord;
    {
        std::vector<int64_t> cand_cells;
        for (int64_t w = 0; w < (int64_t)cand.size(); ++w) {
            uint64_t bits = cand[w];
            while (bits) {
                const int b = ctz64(bits);
                bits &= bits - 1;
                cand_cells.push_back((w << 6) | b);
            }
        }
        post_tick("remesh: cand list", post_t);
        std::vector<uint8_t> act(cand_cells.size(), 0);
        parallel_for((int64_t)cand_cells.size(), [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const int64_t c = cand_cells[i];
                const int x = (int)(c / ((int64_t)res * res)), y = (int)((c / res) % res), z = (int)(c % res);
                const float p[3] = { ((x + 0.5f) / res - 0.5f) * scale,
                                     ((y + 0.5f) / res - 0.5f) * scale,
                                     ((z + 0.5f) / res - 0.5f) * scale };
                const TriBvh::Hit h = bvh.closest(p, eps + keep);
                if (h.face < 0) continue;
                const float f = std::sqrt(h.dist2) - eps;
                if (std::fabs(f) < keep) act[i] = 1;
            }
        });
        post_tick("remesh: active UDF (par)", post_t);
        for (size_t i = 0; i < cand_cells.size(); ++i) {
            if (!act[i]) continue;
            const int64_t c = cand_cells[i];
            acoord.push_back((int)(c / ((int64_t)res * res)));
            acoord.push_back((int)((c / res) % res));
            acoord.push_back((int)(c % res));
        }
        cand.clear(); cand.shrink_to_fit();
    }
    const int64_t Na = (int64_t)acoord.size() / 3;
    if (Na < 100) {
        fprintf(stderr, "  remesh: only %lld active voxels; skipping remesh\n", (long long)Na);
        return out;
    }

    post_tick("remesh: acoord", post_t);
    // Active voxels by linear cell index; acoord is in that order, so the rank of
    // a set bit is its index into acoord.
    auto lin = [res](int64_t x, int64_t y, int64_t z) { return (x * res + y) * res + z; };
    RankBitset vox;
    vox.init((int64_t)res * res * res);
    for (int64_t i = 0; i < Na; ++i) vox.set(lin(acoord[3*i], acoord[3*i+1], acoord[3*i+2]));
    vox.finalize();

    post_tick("remesh: vox map", post_t);
    // f = UDF - eps at the grid VERTICES (corner mapping v/res, spec 27 §4.3):
    // the 8 corners of every active voxel, on the (res+1)^3 vertex grid.
    const int64_t vres = (int64_t)res + 1;
    auto vlin = [vres](int64_t x, int64_t y, int64_t z) { return (x * vres + y) * vres + z; };
    RankBitset vset;
    vset.init(vres * vres * vres);
    for (int64_t i = 0; i < Na; ++i)
        for (int dx = 0; dx < 2; ++dx) for (int dy = 0; dy < 2; ++dy) for (int dz = 0; dz < 2; ++dz)
            vset.set(vlin(acoord[3*i] + dx, acoord[3*i+1] + dy, acoord[3*i+2] + dz));
    vset.finalize();
    std::vector<int> vcoord((size_t)vset.count * 3);
    parallel_for((int64_t)vset.bits.size(), [&](int64_t wb, int64_t we) {
        for (int64_t w = wb; w < we; ++w) {
            uint64_t b = vset.bits[(size_t)w];
            int64_t r = vset.rank0[(size_t)w];
            while (b) {
                const int64_t i = (w << 6) | ctz64(b);
                b &= b - 1;
                vcoord[(size_t)r*3]   = (int)(i / (vres * vres));
                vcoord[(size_t)r*3+1] = (int)((i / vres) % vres);
                vcoord[(size_t)r*3+2] = (int)(i % vres);
                ++r;
            }
        }
    });
    post_tick("remesh: vmap", post_t);
    const int64_t Nv = (int64_t)vcoord.size() / 3;
    std::vector<float> fvert((size_t)Nv);
    parallel_for(Nv, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const float p[3] = { ((float)vcoord[3*i]   / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+1] / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+2] / res - 0.5f) * scale };
            // Crossing edges always have their far endpoint under eps+cell
            // (f changes at most one cell-length per edge), so this bound
            // never clips a value that feeds the crossing interpolation.
            const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
            fvert[i] = (h.face >= 0 ? std::sqrt(h.dist2) : eps + 2 * cell) - eps;
        }
    });
    post_tick("remesh: vertex UDF (par)", post_t);
    auto fval = [&](int x, int y, int z) -> float {
        const int idx = vset.find(vlin(x, y, z));
        return idx < 0 ? 1e9f : fvert[(size_t)idx];
    };

    // Dual vertices: plain mean of edge crossings, cell-center fallback; per
    // voxel, ownership of the 3 "far" edges records crossing direction
    // (spec 27 §4.4).
    std::vector<float> dual((size_t)Na * 3);
    std::vector<int8_t> owned((size_t)Na * 3, 0);
    parallel_for(Na, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
            double sum[3] = {0, 0, 0};
            int cnt = 0;
            for (int axis = 0; axis < 3; ++axis)
                for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v) {
                    int a0[3] = {vx, vy, vz}, a1[3];
                    a0[(axis + 1) % 3] += u;
                    a0[(axis + 2) % 3] += v;
                    a1[0] = a0[0]; a1[1] = a0[1]; a1[2] = a0[2];
                    a1[axis] += 1;
                    const float v1 = fval(a0[0], a0[1], a0[2]);
                    const float v2 = fval(a1[0], a1[1], a1[2]);
                    const bool c12 = v1 < 0 && v2 >= 0, c21 = v1 >= 0 && v2 < 0;
                    if (c12 || c21) {
                        const float t = -v1 / (v2 - v1);
                        double pt[3] = {(double)a0[0], (double)a0[1], (double)a0[2]};
                        pt[axis] += t;
                        for (int k = 0; k < 3; ++k) sum[k] += pt[k];
                        ++cnt;
                    }
                    if (u == 1 && v == 1) owned[3*i + axis] = c12 ? 1 : (c21 ? -1 : 0);
                }
            if (cnt) for (int k = 0; k < 3; ++k) dual[3*i+k] = (float)(sum[k] / cnt);
            else { dual[3*i] = vx + 0.5f; dual[3*i+1] = vy + 0.5f; dual[3*i+2] = vz + 0.5f; }
        }
    });

    post_tick("remesh: dual verts (par)", post_t);
    // Quad assembly per owned crossing edge (spec 27 §4.5); winding from the
    // crossing direction. The reference's "planar diagonal" selection is a
    // latent no-op upstream (always diagonal q0-q2 unless triangle q0q1q2 is
    // exactly degenerate) — reproduced faithfully here as split 1 always.
    static const int OFF[3][4][3] = {
        {{0,0,0}, {0,0,1}, {0,1,1}, {0,1,0}},
        {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}},
        {{0,0,0}, {0,1,0}, {1,1,0}, {1,0,0}},
    };
    std::vector<int32_t> qfaces;
    qfaces.reserve((size_t)Na * 6);
    std::vector<uint8_t> used((size_t)Na, 0);
    for (int64_t i = 0; i < Na; ++i) {
        const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
        for (int axis = 0; axis < 3; ++axis) {
            const int dir = owned[3*i + axis];
            if (!dir) continue;
            int q[4];
            bool ok = true;
            for (int k = 0; k < 4 && ok; ++k) {
                const int x = vx + OFF[axis][k][0], y = vy + OFF[axis][k][1], z = vz + OFF[axis][k][2];
                const int idx = (x < res && y < res && z < res) ? vox.find(lin(x, y, z)) : -1;
                if (idx < 0) ok = false;
                else q[k] = idx;
            }
            if (!ok) continue;
            static const int S1N[6] = {0, 1, 2, 0, 2, 3};
            static const int S1P[6] = {0, 2, 1, 0, 3, 2};
            const int* sp = dir > 0 ? S1P : S1N;
            for (int k = 0; k < 6; ++k) qfaces.push_back(q[sp[k]]);
            for (int k = 0; k < 4; ++k) used[q[k]] = 1;
        }
    }
    post_tick("remesh: quads", post_t);
    if (qfaces.empty()) return out;

    // Compact used dual vertices; map back to world coordinates.
    std::vector<int32_t> remap((size_t)Na, -1);
    int nv2 = 0;
    for (int64_t i = 0; i < Na; ++i)
        if (used[i]) {
            remap[i] = nv2++;
            out.verts.push_back((dual[3*i]   / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+1] / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+2] / res - 0.5f) * scale);
        }
    out.faces.resize(qfaces.size());
    for (size_t k = 0; k < qfaces.size(); ++k) out.faces[k] = remap[qfaces[k]];

    post_tick("remesh: compact", post_t);
    // Project the dual vertices back onto the input surface (reference:
    // remesh_project=0.9, o_voxel/postprocess.py::to_glb -> remeshing.py §8).
    // Dual contouring places each vertex at the plain MEAN of its cell's edge
    // crossings, on the eps-offset shell — so sharp features come out rounded and
    // the shell keeps its own offset noise. Lerping each vertex back toward its
    // closest point on the original surface restores those features. The BVH's
    // closest point is exactly the reference's barycentric interpolation of the
    // hit triangle, so no uvw round-trip is needed.
    if (project_back > 0.0f) {
        const int64_t NV = (int64_t)out.verts.size() / 3;
        std::atomic<int64_t> missed{0};
        parallel_for(NV, [&](int64_t b, int64_t e) {
            int64_t local = 0;
            for (int64_t i = b; i < e; ++i) {
                float* v = &out.verts[3 * i];
                const float p[3] = {v[0], v[1], v[2]};
                // Vertices sit on the eps shell, so the surface is ~eps away; the
                // same bound the field pass uses is comfortably sufficient.
                const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
                if (h.face < 0) { ++local; continue; }   // no hit in range: leave as-is
                for (int k = 0; k < 3; ++k) v[k] = p[k] - project_back * (p[k] - h.point[k]);
            }
            if (local) missed.fetch_add(local, std::memory_order_relaxed);
        });
        if (missed.load())
            printf("  remesh_dc: project_back %.2f (%lld/%lld vertices had no hit in range)\n",
                   project_back, (long long)missed.load(), (long long)NV);
    }

    printf("  remesh_dc: %lld active voxels -> V=%d F=%d (eps=%.4g, project_back=%.2f)\n",
           (long long)Na, out.V(), out.F(), eps, project_back);
    fflush(stdout);
    return out;
}

}  // namespace trellis
