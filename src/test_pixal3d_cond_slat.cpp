// Validate include/pixal3d_cond.h's SLAT-stage (shape/texture) image conditioning
// (DINOv3 + ProjGridMV average fusion + NAF high-res upsample, proj = [lr || hr])
// against PyTorch golden tensors dumped by tools/ref_pixal3d_cond_slat.py.
//
//   trellis-test-pixal3d-cond-slat <dinov3.gguf> <pixal3d_naf.gguf> <fixture_dir> [gpu] [--full] [--stage host|gpu|all]
//
// Default run: s512 stage (V=4 and V=1). --full additionally runs the s1024 stage
// (DINO tokens at S=1024, full cond, gather at sparse coords) -- its fixtures are
// ~1GB and the NAF/DINO passes at S=1024 take a few minutes on CPU/Metal.
// --stage gpu runs only the device-resident s512 path (pixal3d_cond_slat_gpu: one graph per
// view, DINOv3 -> NAF -> both projections -> MV average, V=4 and V=1) against the fixture and
// against the host path on the same backend, with memory/timing stats; `all` (default) runs
// both, `host` the original stages only.
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

static void print_stats(const char* tag, const Pixal3dCondStats& st) {
    printf("  [%s] V=%d weights=%.1f MB cond=%.1f MB view_alloc=%.1f MB peak=%.1f MB  total=%.0f ms  slowest view=%.0f ms\n",
           tag, st.views, st.weight_bytes / 1048576.0, st.cond_bytes / 1048576.0, st.view_alloc_bytes / 1048576.0,
           st.peak_bytes / 1048576.0, st.total_ms, st.view_ms_max);
}

static bool compare_vec(const char* name, const vector<float>& mine, const vector<float>& ref, double tol) {
    npy::Array a; a.shape = {(int64_t)ref.size()}; a.data = ref;
    return compare(name, mine, a, tol);
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

static bool run_s512_gpu(const Model& dinov3, const Model& naf, const string& dir,
                          const vector<Pixal3dView>& views, float mesh_scale, bool with_host) {
    const int S = 512, R = 32, T = 512;
    const size_t NtokR = (size_t)R * R * R;
    bool all_ok = true;
    npy::Array z_global    = npy::load(dir + "/s512_z_global.npy");      // [1,5,1024]
    npy::Array z_proj      = npy::load(dir + "/s512_z_proj.npy");        // [1,32768,2048]
    npy::Array v1_z_global = npy::load(dir + "/s512_v1_z_global.npy");
    npy::Array v1_z_proj   = npy::load(dir + "/s512_v1_z_proj.npy");
    Pixal3dSlatCondParams prm{S, R, T, mesh_scale};
    const int Vs[2] = { (int)views.size(), 1 };
    for (int vi = 0; vi < 2; ++vi) {
        const int n = Vs[vi];
        if (vi == 1 && (int)views.size() == 1) break;
        vector<Pixal3dView> vs(views.begin(), views.begin() + n);
        printf("\n=== s512 stage 5: pixal3d_cond_slat_gpu V=%d (device-resident DINOv3 -> NAF -> projections -> average) ===\n", n);
        Pixal3dCondStats st;
        Pixal3dCond g = pixal3d_cond_slat_gpu(dinov3, naf, vs, prm, &st);
        print_stats("gpu", st);
        const npy::Array& zg = (n == (int)views.size()) ? z_global : v1_z_global;
        const npy::Array& zp = (n == (int)views.size()) ? z_proj : v1_z_proj;
        char nm[96];
        snprintf(nm, sizeof nm, "gpu V=%d z_global vs ref", n); all_ok &= compare(nm, g.global, zg);
        npy::Array lr_ref, hr_ref;
        lr_ref.shape = {(int64_t)NtokR, D}; lr_ref.data = proj_half(zp.data, NtokR, false);
        hr_ref.shape = {(int64_t)NtokR, D}; hr_ref.data = proj_half(zp.data, NtokR, true);
        snprintf(nm, sizeof nm, "gpu V=%d z_proj_lr vs ref", n); all_ok &= compare(nm, proj_half(g.proj, NtokR, false), lr_ref);
        snprintf(nm, sizeof nm, "gpu V=%d z_proj_hr vs ref", n); all_ok &= compare(nm, proj_half(g.proj, NtokR, true), hr_ref);
        if (with_host) {
            auto t0 = std::chrono::steady_clock::now();
            Pixal3dCond h = pixal3d_cond_slat(dinov3, naf, vs, prm);
            printf("  pixal3d_cond_slat(V=%d, host path, same backend) runtime: %.2fs\n", n,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            snprintf(nm, sizeof nm, "gpu V=%d z_global vs host", n);  all_ok &= compare_vec(nm, g.global, h.global, 1e-3);
            snprintf(nm, sizeof nm, "gpu V=%d z_proj_lr vs host", n); all_ok &= compare_vec(nm, proj_half(g.proj, NtokR, false), proj_half(h.proj, NtokR, false), 1e-3);
            snprintf(nm, sizeof nm, "gpu V=%d z_proj_hr vs host", n); all_ok &= compare_vec(nm, proj_half(g.proj, NtokR, true), proj_half(h.proj, NtokR, true), 1e-2);
        }
    }
    return all_ok;
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

// #78/#79: the production sparse path (pixal3d_cond_slat_gpu with active coords, split-graph NAF)
// against tools/ref_pixal3d_ss_res.py's per-ss_res fixtures: LR shape-512 cond at R = ss_res over
// the SS coords (ss<R>/coords.npy, cond_global.npy, cond_proj.npy) and, where dumped, the HR
// shape-1024 cond at grid 64 over that run's HR tokens (hr_coords.npy, hr_cond_*.npy).
static vector<std::array<int, 3>> coords_xyz(const npy::ArrayI32& a) {
    const int w = (int)a.shape.back();
    vector<std::array<int, 3>> v((size_t)a.shape[0]);
    for (size_t i = 0; i < v.size(); ++i)
        for (int k = 0; k < 3; ++k) v[i][k] = a.data[i * w + (w - 3) + k];
    return v;
}

static bool run_ss_res(const Model& dinov3, const Model& naf, const string& ssdir,
                       const vector<Pixal3dView>& v512, const vector<Pixal3dView>& v1024, float mesh_scale) {
    bool all_ok = true;
    for (int R : {32, 64}) {
        const string d = ssdir + "/ss" + std::to_string(R);
        npy::Array g_ref, p_ref;
        if (!try_load(d + "/cond_global.npy", g_ref) || !try_load(d + "/cond_proj.npy", p_ref)) {
            printf("\n(ss_res=%d fixture absent under %s)\n", R, d.c_str()); continue;
        }
        const auto co = coords_xyz(npy::load_i32(d + "/coords.npy"));
        printf("\n=== ss_res=%d LR cond: S=512 R=%d T=512 sparse, %zu tokens ===\n", R, R, co.size());
        Pixal3dSlatCondParams prm{512, R, 512, mesh_scale};
        prm.naf_block_chunk = 256;                       // trellis_cli.cpp's LR setting
        Pixal3dCondStats st;
        Pixal3dCond g = pixal3d_cond_slat_gpu(dinov3, naf, v512, prm, &st, &co);
        print_stats("gpu", st);
        char nm[96];
        snprintf(nm, sizeof nm, "ss%d LR z_global", R); all_ok &= compare(nm, g.global, g_ref);
        npy::Array lr, hr; lr.shape = hr.shape = {(int64_t)co.size(), D};
        lr.data = proj_half(p_ref.data, co.size(), false); hr.data = proj_half(p_ref.data, co.size(), true);
        snprintf(nm, sizeof nm, "ss%d LR z_proj_lr", R); all_ok &= compare(nm, proj_half(g.proj, co.size(), false), lr);
        snprintf(nm, sizeof nm, "ss%d LR z_proj_hr", R); all_ok &= compare(nm, proj_half(g.proj, co.size(), true), hr);
        if (!v1024.empty() && try_load(d + "/hr_cond_global.npy", g_ref) && try_load(d + "/hr_cond_proj.npy", p_ref)) {
            const auto hc = coords_xyz(npy::load_i32(d + "/hr_coords.npy"));
            printf("\n=== ss_res=%d HR cond: S=1024 R=64 T=512 sparse, %zu tokens ===\n", R, hc.size());
            Pixal3dSlatCondParams hp{1024, 64, 512, mesh_scale};
            hp.naf_block_chunk = 1024;                   // trellis_cli.cpp's HR setting
            Pixal3dCond h = pixal3d_cond_slat_gpu(dinov3, naf, v1024, hp, &st, &hc);
            print_stats("gpu", st);
            snprintf(nm, sizeof nm, "ss%d HR z_global", R); all_ok &= compare(nm, h.global, g_ref);
            lr.shape = hr.shape = {(int64_t)hc.size(), D};
            lr.data = proj_half(p_ref.data, hc.size(), false); hr.data = proj_half(p_ref.data, hc.size(), true);
            snprintf(nm, sizeof nm, "ss%d HR z_proj_lr", R); all_ok &= compare(nm, proj_half(h.proj, hc.size(), false), lr);
            snprintf(nm, sizeof nm, "ss%d HR z_proj_hr", R); all_ok &= compare(nm, proj_half(h.proj, hc.size(), true), hr);
        }
    }
    return all_ok;
}

int main(int argc, char** argv) {
    vector<string> pos;
    bool full = false;
    string stage = "all", ss_res_dir;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--full") full = true;
        else if (string(argv[i]) == "--stage" && i + 1 < argc) stage = argv[++i];
        else if (string(argv[i]) == "--ss-res-dir" && i + 1 < argc) ss_res_dir = argv[++i];
        else pos.push_back(argv[i]);
    }
    if (pos.size() < 3 || (stage != "all" && stage != "host" && stage != "gpu")) {
        fprintf(stderr, "usage: %s <dinov3.gguf> <pixal3d_naf.gguf> <fixture_dir> [gpu] [--full] [--stage host|gpu|all]"
                        " [--ss-res-dir DIR]\n", argv[0]);
        return 1;
    }
    const bool do_host = stage != "gpu", do_gpu = stage != "host";
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
        if (!ss_res_dir.empty()) {                     // only the ss_res fixtures
            vector<Pixal3dView> v1024;
            npy::Array images1024;
            if (try_load(dir + "/s1024_images.npy", images1024)) {
                const int S1 = (int)images1024.shape[3];
                const size_t p1 = (size_t)3 * S1 * S1;
                v1024.resize(Vs);
                for (int v = 0; v < Vs; ++v) {
                    v1024[v].rgb_premult.assign(images1024.data.data() + (size_t)v * p1, images1024.data.data() + (size_t)(v + 1) * p1);
                    v1024[v].fov_x = views_meta[v].fov_x;
                    std::memcpy(v1024[v].c2w, views_meta[v].c2w, 16 * sizeof(float));
                }
            }
            all_ok &= run_ss_res(dinov3, naf, ss_res_dir, views, v1024, mesh_scale);
            dinov3.free(); naf.free();
            printf("\n=== overall %s ===\n", all_ok ? "PASS" : "FAIL (see above)");
            return all_ok ? 0 : 1;
        }
        if (do_host) all_ok &= run_s512(dinov3, naf, dir, views, mesh_scale);
        if (do_gpu) all_ok &= run_s512_gpu(dinov3, naf, dir, views, mesh_scale, /*with_host=*/ stage == "gpu");
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
