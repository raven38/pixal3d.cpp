// Fixture-injected partial browser E2E for Pixal3D:
//   Shape-1024 fixture -> Texture Flow -> Shape Decode -> Texture Decode -> GLB.
//
// This intentionally starts at the validated Shape-1024 boundary while v0.5 / issue #2 remain
// open. Texture Flow is not reimplemented here: include the validated v0.6 WASM translation unit
// so run_texture_impl() and its exact sampling path are reused in the same C++ translation unit.
// JS only mounts files, calls pixal3d_partial_e2e_run(), and reads the resulting GLB from MEMFS.
//
// The integration GLB uses decoded base-color as vertex colors. The full reference PBR atlas bake
// remains a later wiring step; importantly, Texture Decode is executed and its output contributes
// to the emitted GLB. This keeps the first partial-E2E gate small enough for wasm32 while proving
// the production neural stages and GLB handoff in one shared C++ call.

#include "pixal3d_texture_wasm.cpp"

#include "shape_decoder.h"
#include "dual_grid.h"
#include "mesh_glb.h"
#include "trellis_model.h"
#include "npy.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using std::array;
using std::string;
using std::vector;

namespace {

static const float TEX_MEAN[32] = {
    3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,
    0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,
    -1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293000f,-1.925608f,-1.217717f,
    1.213905f,0.971588f,-0.023631f,0.106750f,2.021786f,0.250524f,-0.662387f,-0.768862f
};
static const float TEX_STD[32] = {
    2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,
    2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,
    2.295916f,3.234985f,3.233086f,2.260140f,2.874801f,2.810596f,3.292720f,2.674999f,
    2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.775190f
};

static float clamp01(float x) { return std::max(0.0f, std::min(1.0f, x)); }

static uint64_t coord_key(int x, int y, int z) {
    // final voxel coords are non-negative, but keep this generic and sortable.
    return ((uint64_t)(uint32_t)(x + (1 << 20)) << 42) |
           ((uint64_t)(uint32_t)(y + (1 << 20)) << 21) |
           (uint32_t)(z + (1 << 20));
}

static vector<float> base_color_vertices(const trellis::Mesh& mesh,
                                         const vector<array<int,3>>& coords,
                                         const vector<float>& pbr6,
                                         int res) {
    vector<std::pair<uint64_t, uint32_t>> lut;
    lut.reserve(coords.size());
    for (uint32_t i = 0; i < coords.size(); ++i)
        lut.emplace_back(coord_key(coords[i][0], coords[i][1], coords[i][2]), i);
    std::sort(lut.begin(), lut.end(), [](const auto& a, const auto& b){ return a.first < b.first; });

    auto lookup = [&](int x, int y, int z) -> int64_t {
        const uint64_t k = coord_key(x, y, z);
        auto it = std::lower_bound(lut.begin(), lut.end(), k,
            [](const auto& a, uint64_t key){ return a.first < key; });
        return (it != lut.end() && it->first == k) ? (int64_t)it->second : -1;
    };

    vector<float> colors((size_t)mesh.V() * 3, 0.5f);
    for (int i = 0; i < mesh.V(); ++i) {
        int q[3];
        for (int a = 0; a < 3; ++a) {
            const float v = mesh.verts[(size_t)i * 3 + a];
            q[a] = std::max(0, std::min(res - 1, (int)std::floor((v + 0.5f) * res)));
        }
        int64_t idx = lookup(q[0], q[1], q[2]);
        if (idx < 0) {
            // Dual-grid vertices can sit exactly on a boundary. Search the immediate voxel shell.
            for (int dz = -1; dz <= 1 && idx < 0; ++dz)
                for (int dy = -1; dy <= 1 && idx < 0; ++dy)
                    for (int dx = -1; dx <= 1 && idx < 0; ++dx) {
                        int x = q[0] + dx, y = q[1] + dy, z = q[2] + dz;
                        if (x >= 0 && y >= 0 && z >= 0 && x < res && y < res && z < res)
                            idx = lookup(x, y, z);
                    }
        }
        if (idx >= 0 && (size_t)(idx * 6 + 2) < pbr6.size()) {
            for (int c = 0; c < 3; ++c)
                colors[(size_t)i * 3 + c] = clamp01(0.5f * pbr6[(size_t)idx * 6 + c] + 0.5f);
        }
    }
    return colors;
}

static int run_partial_e2e_impl(const string& tex_flow_gguf,
                                const string& shape_dec_gguf,
                                const string& tex_dec_gguf,
                                const string& fixture_dir,
                                const string& concat_cond_npy,
                                const string& out_glb,
                                int resolution) {
    rep("\n=== partial E2E: Shape1024 fixture -> Texture Flow -> decoders -> GLB ===\n");

    // Stage 1: exact validated Texture Flow implementation from the v0.6 translation unit.
    int rc = run_texture_impl("", "", tex_flow_gguf, "", fixture_dir, 0, 0, concat_cond_npy);
    if (rc != 0) { rep("partial-e2e: Texture Flow failed rc=%d\n", rc); return rc; }
    const vector<float> tex_x_final = g_latent; // normalized [N,32], keep before decoder work.

    // Shared Shape-1024 token coordinates.
    vector<int32_t> co; vector<int64_t> cshape;
    if (!load_npy_i32(fixture_dir + "/hr_coords.npy", co, cshape) || cshape.size() != 2) {
        rep("partial-e2e: ERROR loading hr_coords.npy\n"); return 3;
    }
    const int64_t N = cshape[0], cw = cshape[1];
    if ((int64_t)tex_x_final.size() != N * 32) {
        rep("partial-e2e: Texture Flow output size mismatch: %zu vs N*32=%lld\n",
            tex_x_final.size(), (long long)N * 32); return 3;
    }
    vector<array<int,3>> coords0((size_t)N);
    int cmax = 0;
    for (int64_t i = 0; i < N; ++i) {
        coords0[(size_t)i] = { co[(size_t)i*cw+cw-3], co[(size_t)i*cw+cw-2], co[(size_t)i*cw+cw-1] };
        cmax = std::max({cmax, coords0[(size_t)i][0], coords0[(size_t)i][1], coords0[(size_t)i][2]});
    }
    const int inferred_res = (cmax < 32 ? 32 : 64) * 16;
    const int out_res = resolution > 0 ? resolution : inferred_res;

    // Shape decoder consumes DENORMALIZED Shape-1024 SLAT fixture. Prefer the canonical HR name.
    string shape_path = fixture_dir + "/f32_shape_slat.npy";
    if (!file_exists(shape_path)) shape_path = fixture_dir + "/f32_slat.npy";
    if (!file_exists(shape_path)) {
        rep("partial-e2e: ERROR missing f32_shape_slat.npy (or f32_slat.npy)\n"); return 3;
    }
    npy::Array shape_np = npy::load(shape_path);
    if (shape_np.numel() != N * 32) {
        rep("partial-e2e: shape SLAT size mismatch: %lld vs %lld\n",
            (long long)shape_np.numel(), (long long)N * 32); return 3;
    }
    vector<float> shape_latent(shape_np.data.begin(), shape_np.data.end());

    // Texture decoder consumes DENORMALIZED texture SLAT. Use fixture norm arrays when present;
    // otherwise the production constants used by trellis-test-pixal3d-slat-sample.
    vector<float> mean(TEX_MEAN, TEX_MEAN + 32), stdv(TEX_STD, TEX_STD + 32);
    if (file_exists(fixture_dir + "/tex_norm_mean.npy")) mean = npy::load(fixture_dir + "/tex_norm_mean.npy").data;
    if (file_exists(fixture_dir + "/tex_norm_std.npy"))  stdv = npy::load(fixture_dir + "/tex_norm_std.npy").data;
    vector<float> tex_latent((size_t)N * 32);
    for (int64_t n = 0; n < N; ++n) for (int c = 0; c < 32; ++c) {
        const float m = c < (int)mean.size() ? mean[c] : TEX_MEAN[c];
        const float s = c < (int)stdv.size() ? stdv[c] : TEX_STD[c];
        tex_latent[(size_t)n * 32 + c] = tex_x_final[(size_t)n * 32 + c] * s + m;
    }

    rep("partial-e2e: decode input N=%lld grid->res%d shape=%s\n",
        (long long)N, out_res, shape_path.c_str());

    trellis::ShapeOut so;
    {
        trellis::Model m = trellis::Model::load(shape_dec_gguf, 0);
        rep("partial-e2e: shape_dec on %s (%zu tensors)\n", ggml_backend_name(m.backend), m.tensors.size());
        so = trellis::shape_decode(m, shape_latent, coords0, out_res);
        m.free();
    }
    trellis::Mesh mesh = trellis::dual_grid_to_mesh(so);
    if (mesh.F() <= 0) { rep("partial-e2e: ERROR shape decode produced empty mesh\n"); return 4; }
    rep("partial-e2e: mesh V=%d F=%d final_voxels=%zu\n", mesh.V(), mesh.F(), so.coords.size());

    vector<float> pbr;
    {
        trellis::Model m = trellis::Model::load(tex_dec_gguf, 0);
        rep("partial-e2e: tex_dec on %s (%zu tensors)\n", ggml_backend_name(m.backend), m.tensors.size());
        pbr = trellis::tex_decode(m, tex_latent, coords0, so.subs);
        m.free();
    }
    if (pbr.size() != so.coords.size() * 6) {
        rep("partial-e2e: ERROR texture decode size %zu != %zu*6\n", pbr.size(), so.coords.size());
        return 4;
    }

    vector<float> colors = base_color_vertices(mesh, so.coords, pbr, so.res);
    if (!trellis::write_glb(out_glb.c_str(), mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), colors.data())) {
        rep("partial-e2e: ERROR write_glb(%s) failed\n", out_glb.c_str()); return 5;
    }
    FILE* f = fopen(out_glb.c_str(), "rb");
    long bytes = -1;
    if (f) { fseek(f, 0, SEEK_END); bytes = ftell(f); fclose(f); }
    rep("partial-e2e: GLB OK path=%s bytes=%ld V=%d F=%d color_source=tex_decode\n",
        out_glb.c_str(), bytes, mesh.V(), mesh.F());
    rep("partial-e2e: PBR atlas bake is intentionally not yet part of this first integration gate\n");
    return 0;
}

} // namespace

extern "C" {

PIXAL3D_EXPORT const char* pixal3d_partial_e2e_run(const char* tex_flow_gguf,
                                                   const char* shape_dec_gguf,
                                                   const char* tex_dec_gguf,
                                                   const char* fixture_dir,
                                                   const char* concat_cond_npy,
                                                   const char* out_glb,
                                                   int resolution) {
    g_report.clear();
    int rc = 1;
    try {
        rc = run_partial_e2e_impl(tex_flow_gguf, shape_dec_gguf, tex_dec_gguf,
                                  fixture_dir ? fixture_dir : "",
                                  concat_cond_npy ? concat_cond_npy : "",
                                  out_glb ? out_glb : "/out/partial_e2e.glb",
                                  resolution);
    } catch (const std::exception& e) {
        rep("partial-e2e EXCEPTION: %s\n", e.what());
        rc = 1;
    }
    rep(rc == 0 ? "PARTIAL_E2E_RESULT: OK\n" : "PARTIAL_E2E_RESULT: FAIL rc=%d\n", rc);
    return g_report.c_str();
}

} // extern "C"
