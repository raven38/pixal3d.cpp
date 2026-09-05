// Validate include/pixal3d_cond.h's SLAT-stage (shape/texture) image conditioning
// (DINOv3 + ProjGridMV average fusion + NAF high-res upsample, proj = [lr || hr])
// against PyTorch golden tensors dumped by tools/ref_pixal3d_cond_slat.py.
//
//   trellis-test-pixal3d-cond-slat <dinov3.gguf> <pixal3d_naf.gguf> <fixture_dir> [gpu] [--full]
//
// Default run: s512 stage (V=4 and V=1). --full additionally runs the s1024 stage
// (DINO tokens at S=1024, full cond, gather at sparse coords) -- its fixtures are
// ~1GB and the NAF/DINO passes at S=1024 take a few minutes on CPU/Metal.
//
// Reference: pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
// (DinoV3ProjMultiViewFeatureExtractor, "shape_512"/"shape_1024" IMAGE_COND_CONFIGS
// entries) + valeoai/NAF -- see docs/spec/30-pixal3d-cond.md sections 2-4.
#include "pixal3d_cond.h"
#include "dinov3.h"
#include "naf.h"
#include "trellis_model.h"
#include "npy.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using std::string; using std::vector;
using namespace trellis;

static constexpr int NPREFIX = 5, D = 1024;

static bool compare(const char* name, const vector<float>& mine, const npy::Array& ref, double tol = 2e-2) {
    if ((int64_t)mine.size() != ref.numel()) {
        printf("  %-24s SIZE MISMATCH mine=%zu ref=%lld\n", name, mine.size(), (long long)ref.numel());
        return false;
    }
    double maxabs = 0, sumabs = 0, refmax = 0;
    for (size_t i = 0; i < mine.size(); ++i) {
        double d = std::fabs((double)mine[i] - ref.data[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)ref.data[i]));
    }
    double rel = refmax > 0 ? maxabs / refmax : maxabs;
    bool ok = rel < tol;
    printf("  %-24s max|d|=%.4e mean|d|=%.4e rel=%.4e tol=%.1e  %s\n",
           name, maxabs, sumabs / mine.size(), rel, tol, ok ? "PASS" : "FAIL");
    return ok;
}

static bool try_load(const string& path, npy::Array& out) {
    try { out = npy::load(path); return true; } catch (...) { return false; }
}

// Same reshape pixal3d_cond.cpp's internal patch_tokens_to_chw does: dinov3_encode's
// token-major [Ntok, D] patch tokens (indices NPREFIX..Ntok-1, row-major token=h*Wp+w)
// -> channel-major [D, Hp, Wp], the fmap layout proj_grid_sample/naf_upsample expect.
static vector<float> patch_tokens_to_chw(const vector<float>& tok, int Hp, int Wp) {
    vector<float> out((size_t)D * Hp * Wp);
    for (int h = 0; h < Hp; ++h)
        for (int w = 0; w < Wp; ++w) {
            const float* src = tok.data() + (size_t)(NPREFIX + h * Wp + w) * D;
            for (int c = 0; c < D; ++c)
                out[(size_t)c * Hp * Wp + (size_t)h * Wp + w] = src[c];
        }
    return out;
}

// Extracts one half (lr: c in [0,D); hr: c in [D,2D)) of a token-major [Ntok, 2D]
// proj buffer into a plain [Ntok, D] buffer, for separate lr/hr comparison.
static vector<float> proj_half(const vector<float>& proj2048, size_t Ntok, bool hr_half) {
    vector<float> out(Ntok * (size_t)D);
    const size_t off = hr_half ? (size_t)D : 0;
    for (size_t i = 0; i < Ntok; ++i)
        std::memcpy(&out[i * D], &proj2048[i * 2 * D + off], D * sizeof(float));
    return out;
}

static bool run_s512(const Model& dinov3, const Model& naf, const string& dir,
                      const vector<Pixal3dView>& views, float mesh_scale) {
    const int S = 512, R = 32, T = 512;
    bool all_ok = true;

    npy::Array dino_tokens = npy::load(dir + "/s512_dino_tokens.npy");   // [V,1029,1024]
    npy::Array naf_hr_v0   = npy::load(dir + "/s512_naf_hr_v0.npy");     // [1,1024,512,512]
    npy::Array z_global    = npy::load(dir + "/s512_z_global.npy");      // [1,5,1024]
    npy::Array z_proj      = npy::load(dir + "/s512_z_proj.npy");        // [1,32768,2048]
    npy::Array v1_z_global = npy::load(dir + "/s512_v1_z_global.npy");
    npy::Array v1_z_proj   = npy::load(dir + "/s512_v1_z_proj.npy");

    const int V = (int)views.size();
    const int64_t Ntok = dino_tokens.shape[1], Dc = dino_tokens.shape[2];
    const int Hp = S / 16, Wp = Hp;

    // ---- stage 1: per-view DINOv3 tokens ----
    printf("\n=== s512 stage 1: per-view DINOv3 tokens ===\n");
    vector<vector<float>> toks(V);
    for (int v = 0; v < V; ++v) {
        vector<float> normed = pixal3d_imagenet_normalize(views[v].rgb_premult, S);
        toks[v] = dinov3_encode(dinov3, normed, S);
        npy::Array ref_v;
        ref_v.shape = {Ntok, Dc};
        ref_v.data.assign(dino_tokens.data.data() + (size_t)v * Ntok * Dc,
                           dino_tokens.data.data() + (size_t)(v + 1) * Ntok * Dc);
        char nm[64]; snprintf(nm, sizeof(nm), "s512_dino_tokens[%d]", v);
        all_ok &= compare(nm, toks[v], ref_v);
    }

    // ---- stage 2: NAF map of view 0 vs s512_naf_hr_v0 ----
    printf("\n=== s512 stage 2: NAF map (view 0) ===\n");
    vector<float> chw0 = patch_tokens_to_chw(toks[0], Hp, Wp);
    auto t0 = std::chrono::steady_clock::now();
    vector<float> naf0 = naf_upsample(naf, views[0].rgb_premult.data(), S, chw0.data(), D, Hp, Wp, T);
    auto t1 = std::chrono::steady_clock::now();
    printf("  naf_upsample(view0, T=%d) runtime: %.2fs\n", T,
           std::chrono::duration<double>(t1 - t0).count());
    all_ok &= compare("s512_naf_hr_v0", naf0, naf_hr_v0, 3e-3);

    // ---- stage 3: fused s512 cond (V views) vs z_global/z_proj ----
    printf("\n=== s512 stage 3: pixal3d_cond_slat V=%d ===\n", V);
    Pixal3dSlatCondParams prm{S, R, T, mesh_scale};
    auto t2 = std::chrono::steady_clock::now();
    Pixal3dCond condV = pixal3d_cond_slat(dinov3, naf, views, prm);
    auto t3 = std::chrono::steady_clock::now();
    printf("  pixal3d_cond_slat(V=%d) runtime: %.2fs\n", V,
           std::chrono::duration<double>(t3 - t2).count());
    all_ok &= compare("s512_z_global", condV.global, z_global);
    const size_t NtokR = (size_t)R * R * R;
    npy::Array z_proj_lr, z_proj_hr;
    z_proj_lr.shape = {(int64_t)NtokR, D}; z_proj_lr.data = proj_half(z_proj.data, NtokR, false);
    z_proj_hr.shape = {(int64_t)NtokR, D}; z_proj_hr.data = proj_half(z_proj.data, NtokR, true);
    all_ok &= compare("s512_z_proj_lr", proj_half(condV.proj, NtokR, false), z_proj_lr);
    all_ok &= compare("s512_z_proj_hr", proj_half(condV.proj, NtokR, true), z_proj_hr);

    // ---- stage 4: V=1 variant ----
    printf("\n=== s512 stage 4: pixal3d_cond_slat V=1 ===\n");
    vector<Pixal3dView> views1{views[0]};
    Pixal3dCond cond1 = pixal3d_cond_slat(dinov3, naf, views1, prm);
    all_ok &= compare("s512_v1_z_global", cond1.global, v1_z_global);
    npy::Array v1_z_proj_lr, v1_z_proj_hr;
    v1_z_proj_lr.shape = {(int64_t)NtokR, D}; v1_z_proj_lr.data = proj_half(v1_z_proj.data, NtokR, false);
    v1_z_proj_hr.shape = {(int64_t)NtokR, D}; v1_z_proj_hr.data = proj_half(v1_z_proj.data, NtokR, true);
    all_ok &= compare("s512_v1_z_proj_lr", proj_half(cond1.proj, NtokR, false), v1_z_proj_lr);
    all_ok &= compare("s512_v1_z_proj_hr", proj_half(cond1.proj, NtokR, true), v1_z_proj_hr);

    return all_ok;
}

static bool run_s1024(const Model& dinov3, const Model& naf, const string& dir,
                       const vector<Pixal3dView>& views_src, float mesh_scale) {
    const int S = 1024, R = 64, T = 512; // shape_1024 config: NAF target stays 512
    bool all_ok = true;

    npy::Array images1024 = npy::load(dir + "/s1024_images.npy");        // [1,V,3,1024,1024]
    npy::Array dino_tokens = npy::load(dir + "/s1024_dino_tokens.npy");  // [V,4101,1024]
    npy::Array z_global   = npy::load(dir + "/s1024_z_global.npy");      // [1,5,1024]
    npy::Array coords_a   = npy::load(dir + "/s1024_coords.npy");        // [N,4] (b,x,y,z)
    npy::Array z_proj_g   = npy::load(dir + "/s1024_z_proj_gathered.npy"); // [N,2048]

    const int V = (int)images1024.shape[1];
    const size_t plane3 = (size_t)3 * S * S;
    vector<Pixal3dView> views(V);
    for (int v = 0; v < V; ++v) {
        views[v].rgb_premult.assign(images1024.data.data() + (size_t)v * plane3,
                                     images1024.data.data() + (size_t)(v + 1) * plane3);
        views[v].fov_x = views_src[v].fov_x;
        std::memcpy(views[v].c2w, views_src[v].c2w, 16 * sizeof(float));
    }

    const int64_t Ntok = dino_tokens.shape[1], Dc = dino_tokens.shape[2];

    printf("\n=== s1024 stage 5a: per-view DINOv3 tokens (S=1024) ===\n");
    vector<vector<float>> toks(V);
    for (int v = 0; v < V; ++v) {
        vector<float> normed = pixal3d_imagenet_normalize(views[v].rgb_premult, S);
        auto t0 = std::chrono::steady_clock::now();
        toks[v] = dinov3_encode(dinov3, normed, S);
        auto t1 = std::chrono::steady_clock::now();
        printf("  dinov3_encode(view=%d, S=1024) runtime: %.2fs\n", v,
               std::chrono::duration<double>(t1 - t0).count());
        npy::Array ref_v;
        ref_v.shape = {Ntok, Dc};
        ref_v.data.assign(dino_tokens.data.data() + (size_t)v * Ntok * Dc,
                           dino_tokens.data.data() + (size_t)(v + 1) * Ntok * Dc);
        char nm[64]; snprintf(nm, sizeof(nm), "s1024_dino_tokens[%d]", v);
        all_ok &= compare(nm, toks[v], ref_v);
    }

    printf("\n=== s1024 stage 5b: full cond + gather ===\n");
    Pixal3dSlatCondParams prm{S, R, T, mesh_scale};
    auto t2 = std::chrono::steady_clock::now();
    Pixal3dCond cond = pixal3d_cond_slat(dinov3, naf, views, prm);
    auto t3 = std::chrono::steady_clock::now();
    printf("  pixal3d_cond_slat(S=1024,R=64) runtime: %.2fs\n",
           std::chrono::duration<double>(t3 - t2).count());
    all_ok &= compare("s1024_z_global", cond.global, z_global);

    const int64_t N = coords_a.shape[0];
    vector<std::array<int, 3>> coords((size_t)N);
    for (int64_t n = 0; n < N; ++n) {
        // coords_a columns are (b, x, y, z); b is always 0 here (B=1 fixture).
        coords[n] = {(int)std::lround(coords_a.data[n * 4 + 1]),
                     (int)std::lround(coords_a.data[n * 4 + 2]),
                     (int)std::lround(coords_a.data[n * 4 + 3])};
    }
    vector<float> gathered = pixal3d_gather_proj(cond.proj, R, 2 * D, coords);
    all_ok &= compare("s1024_z_proj_gathered", gathered, z_proj_g);

    return all_ok;
}

int main(int argc, char** argv) {
    vector<string> pos;
    bool full = false;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--full") full = true;
        else pos.push_back(argv[i]);
    }
    if (pos.size() < 3) {
        fprintf(stderr, "usage: %s <dinov3.gguf> <pixal3d_naf.gguf> <fixture_dir> [gpu] [--full]\n", argv[0]);
        return 1;
    }
    const string dinov3_gguf = pos[0], naf_gguf = pos[1], dir = pos[2];
    const int gpu = pos.size() > 3 ? atoi(pos[3].c_str()) : 0;

    npy::Array camera_angle_x   = npy::load(dir + "/camera_angle_x.npy");    // [1,V]
    npy::Array transform_matrix = npy::load(dir + "/transform_matrix.npy"); // [1,V,4,4]
    npy::Array mesh_scale_a     = npy::load(dir + "/mesh_scale.npy");        // [1]
    const float mesh_scale = mesh_scale_a.data[0];
    const int V = (int)camera_angle_x.shape[1];
    printf("fixture: V=%d mesh_scale=%.4f\n", V, mesh_scale);

    // shared views (fov_x + c2w); per-stage image pixels are loaded per stage.
    vector<Pixal3dView> views_meta(V);
    for (int v = 0; v < V; ++v) {
        views_meta[v].fov_x = camera_angle_x.data[v];
        std::memcpy(views_meta[v].c2w, transform_matrix.data.data() + (size_t)v * 16, 16 * sizeof(float));
    }

    Model dinov3 = Model::load(dinov3_gguf, gpu);
    printf("loaded dinov3 %s (%zu tensors)\n", dinov3.arch.c_str(), dinov3.tensors.size());
    Model naf = Model::load(naf_gguf, gpu);
    printf("loaded naf %s (%zu tensors)\n", naf.arch.c_str(), naf.tensors.size());

    bool all_ok = true;

    npy::Array images512;
    if (try_load(dir + "/s512_images.npy", images512)) {
        const int Vs = (int)images512.shape[1];
        const int S = (int)images512.shape[3];
        const size_t plane3 = (size_t)3 * S * S;
        vector<Pixal3dView> views(Vs);
        for (int v = 0; v < Vs; ++v) {
            views[v].rgb_premult.assign(images512.data.data() + (size_t)v * plane3,
                                         images512.data.data() + (size_t)(v + 1) * plane3);
            views[v].fov_x = views_meta[v].fov_x;
            std::memcpy(views[v].c2w, views_meta[v].c2w, 16 * sizeof(float));
        }
        all_ok &= run_s512(dinov3, naf, dir, views, mesh_scale);
    } else {
        printf("s512_images.npy not found under %s -- s512 stage UNVERIFIED\n", dir.c_str());
        all_ok = false;
    }

    if (full) {
        all_ok &= run_s1024(dinov3, naf, dir, views_meta, mesh_scale);
    } else {
        printf("\n(s1024 stage skipped -- pass --full to run it; its fixtures are ~1GB and DINO/NAF"
               " at S=1024 take a few minutes on CPU/Metal)\n");
    }

    dinov3.free();
    naf.free();
    printf("\n=== overall %s ===\n", all_ok ? "PASS" : "FAIL (see above)");
    return all_ok ? 0 : 1;
}
