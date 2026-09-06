// REAL shape-SLAT -> mesh parity test for the FlexiDualGrid shape decoder (trellis-cli's
// "[5/7] FlexiDualGrid shape decode -> mesh" stage, before any postprocess: no fill_holes,
// no weld/remesh/decimate). Unlike trellis-test-shape-dec (src/test_shape_dec.cpp), which
// only checks synthetic-random-latent feats7 against tools/ref_shape_dec.py, this test feeds
// the decoder a REAL denormalized shape SLAT produced by the PyTorch reference sampler
// (tools/ref_pixal3d_slat_sample.py's f32 run) and compares the resulting RAW mesh
// (shape_decode -> dual_grid_to_mesh, exactly as trellis_cli.cpp does it up to but not
// including fill_holes) against the reference's own decoded mesh
// (FlexiDualGridVaeDecoder(slat, return_subs=True) -> mesh_list[0], saved as f32_dec_vertices/
// f32_dec_faces -- this is also pre-postprocess: o_voxel's to_glb() is never called).
//
// Fixture (produced by tools/ref_pixal3d_slat_sample.py, see its header comment):
//   coords.npy          [N,4]   int32   (b,x,y,z) at res 32 -- SS-stage active voxels
//   f32_slat.npy         [N,32]  float32 denormalized shape SLAT (f32-torso calibration run)
//   f32_dec_vertices.npy [Nv,3]  float32 reference decoded mesh vertices, in [-0.5,0.5]^3
//   f32_dec_faces.npy    [Nf,3]  int32   reference decoded mesh triangle indices
//
// Both `coords.npy` and `f32_slat.npy` use the flat [N,C] row-major layout, which is byte-
// identical to the ggml/shape_decode "channel-major, ne0=C" convention (see
// src/test_shape_dec.cpp / tools/ref_shape_dec.py: the "transpose" there is a no-op copy of a
// contiguous [N,C] block) -- no reindexing needed for coords or the latent.
//
//   trellis-test-pixal3d-shape-decode <shape_dec.gguf> <fixture_dir> [gpu] [--res R]
//                                     [--dump-mesh PREFIX] [--ext-mesh PREFIX]
//
// Fixture layouts: slat_sample (coords.npy / f32_slat.npy, res 32 -> 512) and hr_sample
// (hr_coords.npy / f32_shape_slat.npy, res 64 -> 1024; tools/ref_pixal3d_hr_sample.py). The
// final resolution is 16x the input grid (the largest input coord tells which) unless --res
// overrides it. If the fixture has f32_tex_coords.npy (the reference's decoded voxel coords)
// the final coordinate set is compared exactly, before any mesh metric.
// --dump-mesh PREFIX writes PREFIXverts.npy / PREFIXfaces.npy / PREFIXcoords.npy (this run's
// raw mesh + final coords); --ext-mesh PREFIX scores such a dump (e.g. the browser run's,
// web/shape_decode/run_playwright.js) against the reference instead of decoding here.
#include "trellis_model.h"
#include "shape_decoder.h"
#include "dual_grid.h"
#include "npy.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using std::array; using std::string; using std::vector;

namespace trellis { extern bool g_sparse_cast_f32; }

// Minimal <i4 (int32) .npy reader -- npy.h (shared, not to be touched here) is f32-only.
// Mirrors the loader in src/test_pixal3d_slat_sample.cpp.
struct I32Array {
    std::vector<int64_t> shape;
    std::vector<int32_t> data;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};
static I32Array load_npy_i32(const string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); throw std::runtime_error("npy: bad magic " + path); }
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fclose(f); throw std::runtime_error("npy: header len " + path); }
    std::string hdr(hlen, '\0');
    if (fread(hdr.data(), 1, hlen, f) != hlen) { fclose(f); throw std::runtime_error("npy: header " + path); }
    if (hdr.find("'<i4'") == std::string::npos && hdr.find("\"<i4\"") == std::string::npos) {
        fclose(f); throw std::runtime_error("npy: only <i4 supported here: " + path + " hdr=" + hdr);
    }
    I32Array a;
    size_t p = hdr.find("'shape':"); p = hdr.find('(', p); size_t q = hdr.find(')', p);
    std::string s = hdr.substr(p + 1, q - p - 1);
    for (size_t i = 0; i < s.size();) {
        if (isdigit((unsigned char)s[i])) { int64_t v = 0; while (i < s.size() && isdigit((unsigned char)s[i])) v = v * 10 + (s[i++] - '0'); a.shape.push_back(v); }
        else ++i;
    }
    if (a.shape.empty()) a.shape.push_back(1);
    a.data.resize(a.numel());
    if ((int64_t)fread(a.data.data(), sizeof(int32_t), a.numel(), f) != a.numel()) { fclose(f); throw std::runtime_error("npy: short data " + path); }
    fclose(f);
    return a;
}

static void bounds(const vector<float>& v, float mn[3], float mx[3]) {
    mn[0]=mn[1]=mn[2]=1e30f; mx[0]=mx[1]=mx[2]=-1e30f;
    for (size_t i = 0; i + 2 < v.size(); i += 3)
        for (int a = 0; a < 3; ++a) { mn[a]=std::min(mn[a],v[i+a]); mx[a]=std::max(mx[a],v[i+a]); }
}

// Uniform spatial hash over [-0.5,0.5]^3 at ~1-voxel cell size, for symmetric nearest-vertex
// distance queries between the two (potentially differently-ordered/differently-sized) vertex
// sets. Both meshes live in the same world cube (dual_grid_to_mesh: (coord+off)/res - 0.5).
struct VoxelHash {
    float cell;
    std::unordered_map<int64_t, vector<int>> buckets;
    const float* pts; size_t n;
    VoxelHash(const float* p, size_t npts, float cellsz) : cell(cellsz), pts(p), n(npts) {
        buckets.reserve(npts * 2);
        for (size_t i = 0; i < n; ++i) buckets[key(p[3*i], p[3*i+1], p[3*i+2])].push_back((int)i);
    }
    static int64_t cellidx(float x, float cell) { return (int64_t)std::floor(x / cell); }
    static int64_t key3(int64_t ix, int64_t iy, int64_t iz) {
        return ((ix & 0x1FFFFF) << 42) | ((iy & 0x1FFFFF) << 21) | (iz & 0x1FFFFF);
    }
    int64_t key(float x, float y, float z) const {
        return key3(cellidx(x, cell), cellidx(y, cell), cellidx(z, cell));
    }
    // nearest point to (x,y,z): expands the search ring (Chebyshev shells, ring 0 = own cell)
    // until it finds >=1 candidate, does exactly one more ring past the first hit (to catch
    // points just across a cell boundary that are Euclidean-closer than same-ring ones), then
    // stops. Capped at ring 64 (~0.125 world units = ~64 voxels) -- if nothing is found by then
    // the two meshes don't occupy the same region at all locally; returns a large sentinel.
    float nearest(float x, float y, float z) const {
        const int64_t cx = cellidx(x, cell), cy = cellidx(y, cell), cz = cellidx(z, cell);
        float best = 1e30f;
        int extra = -1;   // -1 = not found yet; else rings left to search after the first hit
        for (int ring = 0; ring <= 64; ++ring) {
            bool any = false;
            for (int dx = -ring; dx <= ring; ++dx) for (int dy = -ring; dy <= ring; ++dy) for (int dz = -ring; dz <= ring; ++dz) {
                if (std::max({std::abs(dx),std::abs(dy),std::abs(dz)}) != ring) continue;   // shell only
                auto it = buckets.find(key3(cx+dx, cy+dy, cz+dz));
                if (it == buckets.end()) continue;
                any = true;
                for (int idx : it->second) {
                    float ddx=pts[3*idx]-x, ddy=pts[3*idx+1]-y, ddz=pts[3*idx+2]-z;
                    float d = std::sqrt(ddx*ddx+ddy*ddy+ddz*ddz);
                    best = std::min(best, d);
                }
            }
            if (any && extra < 0) extra = 1;
            else if (extra == 0) break;
            if (extra > 0) --extra;
        }
        return best;
    }
};

struct DistStats { double mean, p95, p99, max; };
static DistStats compute_stats(vector<float>& d) {
    std::sort(d.begin(), d.end());
    DistStats s{};
    if (d.empty()) return s;
    double sum = 0; for (double x : d) sum += x;
    s.mean = sum / d.size();
    s.p95 = d[(size_t)(0.95 * (d.size()-1))];
    s.p99 = d[(size_t)(0.99 * (d.size()-1))];
    s.max = d.back();
    return s;
}


// Exact set comparison of two voxel coordinate lists (order-free; both are duplicate-free).
static void coord_parity(const char* tag, const vector<array<int,3>>& mine, const int32_t* ref, int64_t nref) {
    auto key = [](int x, int y, int z) { return ((uint64_t)(uint32_t)x << 40) | ((uint64_t)(uint32_t)y << 20) | (uint32_t)z; };
    std::unordered_map<uint64_t, uint8_t> rs; rs.reserve((size_t)nref * 2);
    for (int64_t i = 0; i < nref; ++i) rs[key(ref[3*i], ref[3*i+1], ref[3*i+2])] = 0;
    int64_t common = 0, extra = 0;
    for (const auto& c : mine) { auto it = rs.find(key(c[0], c[1], c[2])); if (it == rs.end()) ++extra; else { ++common; it->second = 1; } }
    int64_t missing = 0; for (auto& kv : rs) if (!kv.second) ++missing;
    printf("  %s: mine=%zu ref=%lld common=%lld extra(mine only)=%lld missing(ref only)=%lld -> %s\n",
           tag, mine.size(), (long long)nref, (long long)common, (long long)extra, (long long)missing,
           (extra == 0 && missing == 0) ? "EXACT" : "DIFFERENT");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <shape_dec.gguf> <fixture_dir> [gpu] [--res R] [--dump-mesh PREFIX] [--ext-mesh PREFIX]\n", argv[0]);
        return 1;
    }
    const string gguf = argv[1], fdir = argv[2];
    int gpu = 0, res_override = 0;
    string dump_prefix, ext_prefix;
    for (int i = 3; i < argc; ++i) {
        string a = argv[i];
        if (a == "--res" && i + 1 < argc) res_override = atoi(argv[++i]);
        else if (a == "--dump-mesh" && i + 1 < argc) dump_prefix = argv[++i];
        else if (a == "--ext-mesh" && i + 1 < argc) ext_prefix = argv[++i];
        else gpu = atoi(argv[i]);
    }
    printf("fixture_dir=%s gpu=%d\n", fdir.c_str(), gpu);

    // production default: fp16 torso (no --f32, no TRELLIS_F32W) -- matches the fixture's own
    // decoder dtype note ("left at the decoder's own production dtype (use_fp16=True ...) --
    // NOT upcast").
    if (getenv("TRELLIS_F32W")) { trellis::g_sparse_cast_f32 = true; printf("(TRELLIS_F32W: f32 weight compute -- NOT production default)\n"); }

    auto exists = [](const string& p) { FILE* f = fopen(p.c_str(), "rb"); if (!f) return false; fclose(f); return true; };
    const string coords_path = exists(fdir + "/coords.npy") ? fdir + "/coords.npy" : fdir + "/hr_coords.npy";
    const string slat_path = exists(fdir + "/f32_slat.npy") ? fdir + "/f32_slat.npy" : fdir + "/f32_shape_slat.npy";
    I32Array co = load_npy_i32(coords_path);                      // [N,4] (b,x,y,z)
    npy::Array slat = npy::load(slat_path);                       // [N,32] denormalized
    npy::Array rv = npy::load(fdir + "/f32_dec_vertices.npy");    // [Nv,3]
    I32Array   rf = load_npy_i32(fdir + "/f32_dec_faces.npy");    // [Nf,3]

    const int64_t N = co.shape.empty() ? 0 : co.shape[0];
    if (slat.shape[0] != N) { fprintf(stderr, "%s rows=%lld != coords N=%lld\n", slat_path.c_str(), (long long)slat.shape[0], (long long)N); return 1; }
    const int cw = (int)co.shape[1], c0 = cw - 3;
    vector<array<int,3>> coords0(N);
    int cmax = 0;
    for (int64_t i = 0; i < N; ++i) {
        coords0[i] = { co.data[i*cw + c0], co.data[i*cw + c0 + 1], co.data[i*cw + c0 + 2] };
        cmax = std::max({ cmax, coords0[i][0], coords0[i][1], coords0[i][2] });
    }
    const int in_res = cmax < 32 ? 32 : 64;
    const int resolution = res_override > 0 ? res_override : in_res * 16;
    // [N,32] row-major == ggml [32,N] channel-major (ne0=32): direct flat copy, no transpose.
    vector<float> latent(slat.data.begin(), slat.data.begin() + (size_t)32 * N);

    printf("input: N=%lld active voxels @res%d (%s) -> decode @res%d\n", (long long)N, in_res, coords_path.c_str(), resolution);

    trellis::Mesh mesh;
    vector<array<int,3>> fcoords;     // final voxel coords (mine)
    int res = resolution;
    if (ext_prefix.empty()) {
        trellis::Model m = trellis::Model::load(gguf, gpu);
        printf("loaded %s (%zu tensors) on %s\n", m.arch.c_str(), m.tensors.size(), ggml_backend_name(m.backend));
        const auto t0 = std::chrono::steady_clock::now();
        trellis::ShapeOut so = trellis::shape_decode(m, latent, coords0, resolution);
        const double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("decoded voxels @res%d = %d  (%.1f s)\n", so.res, (int)so.coords.size(), dec_s);
        mesh = trellis::dual_grid_to_mesh(so);
        m.free();
        fcoords = std::move(so.coords); res = so.res;
        if (!dump_prefix.empty()) {
            vector<int32_t> ci(fcoords.size() * 3);
            for (size_t i = 0; i < fcoords.size(); ++i) for (int a = 0; a < 3; ++a) ci[3*i + a] = fcoords[i][a];
            npy::save(dump_prefix + "verts.npy", mesh.verts.data(), { (int64_t)mesh.V(), 3 });
            npy::save_i32(dump_prefix + "faces.npy", mesh.faces.data(), { (int64_t)mesh.F(), 3 });
            npy::save_i32(dump_prefix + "coords.npy", ci.data(), { (int64_t)fcoords.size(), 3 });
            printf("dumped %sverts.npy / faces.npy / coords.npy\n", dump_prefix.c_str());
        }
    } else {
        npy::Array ev = npy::load(ext_prefix + "verts.npy");
        I32Array ef = load_npy_i32(ext_prefix + "faces.npy");
        mesh.verts = std::move(ev.data); mesh.faces = std::move(ef.data);
        if (exists(ext_prefix + "coords.npy")) {
            I32Array ec = load_npy_i32(ext_prefix + "coords.npy");
            fcoords.resize((size_t)ec.shape[0]);
            for (size_t i = 0; i < fcoords.size(); ++i) fcoords[i] = { ec.data[3*i], ec.data[3*i+1], ec.data[3*i+2] };
        }
        printf("external mesh %s: V=%d F=%d coords=%zu (no decode here)\n", ext_prefix.c_str(), mesh.V(), mesh.F(), fcoords.size());
    }

    // ---- final voxel coordinate parity (exact), where the fixture provides the reference's ----
    bool coords_ok = true;
    if (!fcoords.empty() && exists(fdir + "/f32_tex_coords.npy")) {
        // [M,3] int32 ("f32_" names the reference run, not the dtype); the tex decoder is driven by
        // the shape decoder's subdivision masks, so its voxel coords ARE the shape decoder's.
        I32Array rc = load_npy_i32(fdir + "/f32_tex_coords.npy");
        printf("\n=== final voxel coordinate parity vs reference (f32_tex_coords) ===\n");
        coord_parity("coords", fcoords, rc.data.data(), rc.shape[0]);
        coords_ok = (int64_t)fcoords.size() == rc.shape[0];   // count equality; the set report above says the rest
    }

    printf("\n=== mesh metrics (mine vs reference, BOTH pre-postprocess: no fill_holes/weld/remesh/decimate) ===\n");
    printf("  mine:      V=%d F=%d\n", mesh.V(), mesh.F());
    printf("  reference: V=%d F=%d\n", (int)rv.shape[0], (int)rf.shape[0]);
    const double face_ratio = rf.shape[0] > 0 ? (double)mesh.F() / (double)rf.shape[0] : 0.0;
    printf("  face count ratio (mine/ref) = %.4f\n", face_ratio);

    float mn_m[3], mx_m[3], mn_r[3], mx_r[3];
    bounds(mesh.verts, mn_m, mx_m);
    bounds(rv.data, mn_r, mx_r);
    printf("  bounds mine:      x[%.5f,%.5f] y[%.5f,%.5f] z[%.5f,%.5f]\n", mn_m[0],mx_m[0],mn_m[1],mx_m[1],mn_m[2],mx_m[2]);
    printf("  bounds reference: x[%.5f,%.5f] y[%.5f,%.5f] z[%.5f,%.5f]\n", mn_r[0],mx_r[0],mn_r[1],mx_r[1],mn_r[2],mx_r[2]);

    // symmetric nearest-vertex distance, in voxel units (1 voxel = 1/res at the final res).
    const float voxel = 1.0f / (float)res;
    printf("\n  building spatial hash (cell=%.6f = 1 voxel @res%d) ...\n", voxel, res);
    VoxelHash href(rv.data.data(), rv.shape[0], voxel);
    VoxelHash hmine(mesh.verts.data(), mesh.V(), voxel);

    vector<float> d_mine_to_ref(mesh.V());
    for (int i = 0; i < mesh.V(); ++i)
        d_mine_to_ref[i] = href.nearest(mesh.verts[3*i], mesh.verts[3*i+1], mesh.verts[3*i+2]);
    vector<float> d_ref_to_mine((size_t)rv.shape[0]);
    for (int64_t i = 0; i < rv.shape[0]; ++i)
        d_ref_to_mine[i] = hmine.nearest(rv.data[3*i], rv.data[3*i+1], rv.data[3*i+2]);

    auto report = [&](const char* tag, vector<float>& d) {
        DistStats s = compute_stats(d);
        printf("  %-14s mean=%.4e (%.3f vox)  p95=%.4e (%.3f vox)  p99=%.4e (%.3f vox)  max=%.4e (%.3f vox)\n",
               tag, s.mean, s.mean/voxel, s.p95, s.p95/voxel, s.p99, s.p99/voxel, s.max, s.max/voxel);
        return s;
    };
    printf("\n  symmetric nearest-vertex distance (world units, [-0.5,0.5] cube; voxel=1/%d):\n", res);
    DistStats s1 = report("mine->ref", d_mine_to_ref);
    DistStats s2 = report("ref->mine", d_ref_to_mine);

    printf("\n=== verdict ===\n");
    const double worst_mean = std::max(s1.mean, s2.mean);
    const bool mesh_ok = worst_mean <= 1.0 * voxel && face_ratio > 0.9 && face_ratio < 1.1;
    printf("worst symmetric mean nearest-vertex distance = %.4e (%.3f voxel); face ratio = %.4f\n",
           worst_mean, worst_mean/voxel, face_ratio);
    printf("%s\n", mesh_ok ? "PASS (mesh reproduces the real-SLAT reference)"
                           : "FAIL (material divergence from the real-SLAT reference)");
    (void)coords_ok;
    return mesh_ok ? 0 : 2;
}
