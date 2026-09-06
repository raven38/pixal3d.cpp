// Pixal3D SS and Shape-512 paths as a C ABI for the WASM/browser target (web/ss/,
// web/shape512/) -- and, under a native build, as a plain executable for debugging the same
// code. JavaScript's only jobs are to hand over files (WORKERFS mounts of the GGUFs and
// fixtures), call pixal3d_ss_run() / pixal3d_shape512_run(), and display the returned report /
// save the returned latents; every stage below runs in shared C++ on the ggml backend the
// Model picked (WebGPU in the browser):
//
//   SS:        fixture views -> pixal3d_cond_ss_gpu (DINOv3 -> projection -> MV average, per view)
//                 -> DitRunner (ProjectAttention SS flow, exact SDPA) -> sample_flow (12 Euler
//                    steps, CFG) -> final latent [8,16,16,16]
//   Shape-512: fixture views -> pixal3d_cond_slat_gpu (DINOv3 -> NAF -> lr/hr projections ->
//                    MV average, one graph per view) -> sparse DitRunner over the fixture's
//                    active voxels (ProjectAttention shape flow, exact SDPA) -> sample_flow
//                    -> final latent [N,32]
//
//   Shape decode: real sampled shape SLAT fixture -> shape_decode (from_latent -> 4 x
//                    (ConvNeXt stage + C2S) -> output_layer, sparse conv on the shared
//                    neighbor tables) -> dual_grid_to_mesh -> raw mesh + final voxel coords
//                    (web/shape_decode/; scored natively by trellis-test-pixal3d-shape-decode
//                    --ext-mesh, see docs/PIXAL3D_WEBGPU_MEMORY.md §9)
//
// The SS decoder (dense Conv3D) and the texture decoder are not run here; the latents are
// returned so the native tests can score/decode them.
#include "pixal3d_cond.h"
#include "shape_decoder.h"
#include "dual_grid.h"
#include "dinov3.h"
#include "naf.h"
#include "trellis_model.h"
#include "flow_runner.h"
#include "dit.h"
#include "trellis_args.h"
#include "npy.h"
#include "ggml-backend.h"

#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define PIXAL3D_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define PIXAL3D_EXPORT
#endif

using std::string; using std::vector;
using namespace trellis;

namespace {

string g_report;
vector<float> g_latent;      // final latent, torch [1,8,16,16,16] layout
vector<float> g_steps;       // per-step latents, same layout, concatenated
int g_n_steps = 0;

void rep(const char* fmt, ...) {
    char buf[1024];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    g_report += buf;
    fputs(buf, stdout); fflush(stdout);
}

bool file_exists(const string& p) { FILE* f = fopen(p.c_str(), "rb"); if (!f) return false; fclose(f); return true; }

struct Cmp { double maxabs, meanabs, rel, cos; };
Cmp cmp(const vector<float>& a, const vector<float>& b) {
    double maxabs = 0, sumabs = 0, refmax = 0, dot = 0, na = 0, nb = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - b[i]);
        maxabs = std::max(maxabs, d); sumabs += d; refmax = std::max(refmax, std::fabs((double)b[i]));
        dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
    }
    return { maxabs, sumabs / (n ? n : 1), refmax > 0 ? maxabs / refmax : maxabs, (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0 };
}
void rep_cmp(const char* tag, const Cmp& c) { rep("    %-5s max|d|=%.4e mean|d|=%.4e rel=%.4e cos=%.7f\n", tag, c.maxabs, c.meanabs, c.rel, c.cos); }
void rep_latent(const vector<float>& v) {
    double m = 0, mx = 0; size_t bad = 0;
    for (float x : v) { if (!std::isfinite(x)) { ++bad; continue; } m += x; mx = std::max(mx, (double)std::fabs(x)); }
    m /= v.size(); double var = 0; for (float x : v) if (std::isfinite(x)) var += (x - m) * (x - m);
    rep("    latent mean=%+.6f std=%.6f max|x|=%.4f nonfinite=%zu\n", m, std::sqrt(var / v.size()), mx, bad);
}
double ms_since(const std::chrono::steady_clock::time_point& t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int run_impl(const string& dinov3_gguf, const string& ss_flow_gguf, const string& cond_dir,
             const string& sample_dir, int n_views) {
    g_latent.clear(); g_steps.clear(); g_n_steps = 0;
    const auto t_all = std::chrono::steady_clock::now();
    g_no_fa = true;   // exact chunked SDPA: the WebGPU backend has no BF16 K/V FlashAttention (--no-fa equivalent)

    // ---- fixture views (tools/ref_pixal3d_cond_ss.py layout) ----
    npy::Array images_512 = npy::load(cond_dir + "/images_512.npy");         // [1,V,3,512,512]
    npy::Array camera_angle_x = npy::load(cond_dir + "/camera_angle_x.npy"); // [1,V]
    npy::Array transform_matrix = npy::load(cond_dir + "/transform_matrix.npy"); // [1,V,4,4]
    npy::Array mesh_scale = npy::load(cond_dir + "/mesh_scale.npy");         // [1]
    const int Vfix = (int)images_512.shape[1], S = (int)images_512.shape[3];
    const int V = (n_views > 0 && n_views < Vfix) ? n_views : Vfix;
    const int R = 16;
    const size_t plane3 = (size_t)3 * S * S;
    vector<Pixal3dView> views(V);
    for (int v = 0; v < V; ++v) {
        views[v].rgb_premult.assign(images_512.data.data() + (size_t)v * plane3, images_512.data.data() + (size_t)(v + 1) * plane3);
        views[v].fov_x = camera_angle_x.data[v];
        std::memcpy(views[v].c2w, transform_matrix.data.data() + (size_t)v * 16, 16 * sizeof(float));
    }
    images_512.data.clear(); images_512.data.shrink_to_fit();
    rep("views: V=%d (fixture V=%d) S=%d R=%d mesh_scale=%.4f\n", V, Vfix, S, R, mesh_scale.data[0]);

    // ---- stage 1: DINOv3 + projection + MV fusion, device-resident ----
    auto t0 = std::chrono::steady_clock::now();
    Model dinov3 = Model::load(dinov3_gguf, 0);
    const double dino_load_ms = ms_since(t0);
    rep("dinov3: %zu tensors on %s, %.1f MB weights, load %.0f ms\n", dinov3.tensors.size(),
        ggml_backend_name(dinov3.backend), dinov3.total_bytes() / 1048576.0, dino_load_ms);
    Pixal3dCondStats st;
    Pixal3dCond cond = pixal3d_cond_ss_gpu(dinov3, views, S, R, mesh_scale.data[0], &st);
    rep("cond: V=%d weights=%.1f MB cond=%.1f MB view_alloc=%.1f MB peak=%.1f MB total=%.0f ms slowest view=%.0f ms\n",
        st.views, st.weight_bytes / 1048576.0, st.cond_bytes / 1048576.0, st.view_alloc_bytes / 1048576.0,
        st.peak_bytes / 1048576.0, st.total_ms, st.view_ms_max);
    dinov3.free();
    if (V == Vfix && file_exists(cond_dir + "/z_global.npy")) {
        rep_cmp("z_global vs PyTorch", cmp(cond.global, npy::load(cond_dir + "/z_global.npy").data));
        rep_cmp("z_proj vs PyTorch", cmp(cond.proj, npy::load(cond_dir + "/z_proj.npy").data));
    } else if (V == 1 && file_exists(cond_dir + "/v1_z_global.npy")) {
        rep_cmp("v1_z_global vs PyTorch", cmp(cond.global, npy::load(cond_dir + "/v1_z_global.npy").data));
        rep_cmp("v1_z_proj vs PyTorch", cmp(cond.proj, npy::load(cond_dir + "/v1_z_proj.npy").data));
    }

    // ---- stage 2: SS flow sampling (tools/ref_pixal3d_ss_sample.py fixture: noise + refs) ----
    npy::Array noise = npy::load(sample_dir + "/noise.npy");   // [1,8,16,16,16]
    const int64_t Cin = noise.shape[1], L = (int64_t)R * R * R;
    const int64_t Lc = cond.n_global, Dc = 1024, Dp = cond.d_proj;
    vector<float> sample(Cin * L);
    for (int64_t c = 0; c < Cin; ++c) for (int64_t sp = 0; sp < L; ++sp) sample[c + Cin * sp] = noise.data[c * L + sp];
    // cond.global/proj are token-major [tok][d] == ggml channel-major [d + D*tok]: same bytes.
    vector<float> neg_cond(cond.global.size(), 0.0f), neg_proj(cond.proj.size(), 0.0f);

    t0 = std::chrono::steady_clock::now();
    Model mf = Model::load(ss_flow_gguf, 0);
    const double flow_load_ms = ms_since(t0);
    DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (!dit_detect_proj_attn(mf, p) || p.d_proj != (int)Dp) { rep("ERROR: not a Pixal3D proj_attn checkpoint (d_proj=%d, cond %lld)\n", p.d_proj, (long long)Dp); return 2; }
    rep("ss_flow: %zu tensors on %s, %.1f MB weights, load %.0f ms, proj_attn d_proj=%d\n", mf.tensors.size(),
        ggml_backend_name(mf.backend), mf.total_bytes() / 1048576.0, flow_load_ms, p.d_proj);
    t0 = std::chrono::steady_clock::now();
    DitRunner* run = make_dense_runner(mf, p, R, (int)Lc);
    rep("dit graph: activation buffer %.1f MB, build+alloc %.0f ms\n", run->alloc_bytes() / 1048576.0, ms_since(t0));
    int n_fwd = 0; double fwd_ms = 0;
    FlowFwdProj fwd = [&](const vector<float>& x, float ts, const float* c, const float* pj) {
        const auto tf = std::chrono::steady_clock::now();
        vector<float> r = run->forward(x, ts, c, pj);
        fwd_ms += ms_since(tf); ++n_fwd;
        return r;
    };
    SamplerParams sp;   // pipeline_mv.json sparse_structure_sampler.params
    sp.steps = 12; sp.guidance_strength = 7.5f; sp.guidance_rescale = 0.7f;
    sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 5.0f; sp.sigma_min = 1e-5f;
    vector<vector<float>> trace;
    vector<float> out = sample_flow(fwd, sample, cond.global.data(), neg_cond.data(), cond.proj.data(), neg_proj.data(), sp, &trace);
    rep("dit forwards: %d, %.1f ms each, %.1f s total\n", n_fwd, n_fwd ? fwd_ms / n_fwd : 0.0, fwd_ms / 1000.0);
    delete run; mf.free();

    // ---- per-step parity vs the PyTorch f32/bf16 references ----
    auto remap = [&](const vector<float>& mine) {
        vector<float> o((size_t)Cin * L);
        for (int64_t c = 0; c < Cin; ++c) for (int64_t sp2 = 0; sp2 < L; ++sp2) o[(size_t)c * L + sp2] = mine[c + Cin * sp2];
        return o;
    };
    rep("per-step latent parity (torch [1,%lld,16,16,16] layout):\n", (long long)Cin);
    g_n_steps = (int)trace.size();
    for (size_t k = 0; k < trace.size(); ++k) {
        vector<float> mine = remap(trace[k]);
        g_steps.insert(g_steps.end(), mine.begin(), mine.end());
        rep("  step %2zu:\n", k + 1);
        rep_latent(mine);
        string f32 = sample_dir + "/f32_x_step" + std::to_string(k + 1) + ".npy";
        string b16 = sample_dir + "/bf16_x_step" + std::to_string(k + 1) + ".npy";
        if (file_exists(f32)) rep_cmp("f32", cmp(mine, npy::load(f32).data));
        if (file_exists(b16)) rep_cmp("bf16", cmp(mine, npy::load(b16).data));
    }
    g_latent = remap(out);
    rep("  final:\n");
    rep_latent(g_latent);
    if (file_exists(sample_dir + "/f32_x_final.npy")) rep_cmp("f32", cmp(g_latent, npy::load(sample_dir + "/f32_x_final.npy").data));
    if (file_exists(sample_dir + "/bf16_x_final.npy")) rep_cmp("bf16", cmp(g_latent, npy::load(sample_dir + "/bf16_x_final.npy").data));
    rep("total %.1f s\n", ms_since(t_all) / 1000.0);
    return 0;
}

// Minimal <i4 .npy reader for the fixture's coords (npy::load is f32-only; same as
// test_pixal3d_slat_sample.cpp).
bool load_npy_i32(const string& path, vector<int32_t>& data, vector<int64_t>& shape) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return false; }
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return false; }
    string hdr(hlen, '\0');
    if (fread(hdr.data(), 1, hlen, f) != hlen) { fclose(f); return false; }
    if (hdr.find("'<i4'") == string::npos && hdr.find("\"<i4\"") == string::npos) { fclose(f); return false; }
    shape.clear();
    size_t p = hdr.find("'shape':"); p = hdr.find('(', p); size_t q = hdr.find(')', p);
    string sh = hdr.substr(p + 1, q - p - 1);
    for (size_t i = 0; i < sh.size();) {
        if (isdigit((unsigned char)sh[i])) { int64_t v = 0; while (i < sh.size() && isdigit((unsigned char)sh[i])) v = v * 10 + (sh[i++] - '0'); shape.push_back(v); }
        else ++i;
    }
    int64_t n = 1; for (auto v : shape) n *= v;
    data.resize(n);
    const bool ok = (int64_t)fread(data.data(), sizeof(int32_t), n, f) == n;
    fclose(f);
    return ok;
}

// [N, 2D] token-major proj -> one 1024-wide half ([N, D]) for separate lr/hr scoring.
vector<float> proj_half(const vector<float>& proj, size_t N, bool hr) {
    vector<float> o(N * 1024);
    for (size_t i = 0; i < N; ++i) std::memcpy(&o[i * 1024], &proj[i * 2048 + (hr ? 1024 : 0)], 1024 * sizeof(float));
    return o;
}

int run_shape512_impl(const string& dinov3_gguf, const string& naf_gguf, const string& flow_gguf,
                      const string& cond_dir, const string& sample_dir, int n_views, int own_cond) {
    g_latent.clear(); g_steps.clear(); g_n_steps = 0;
    const auto t_all = std::chrono::steady_clock::now();
    g_no_fa = true;   // exact chunked SDPA (no BF16 FlashAttention on the WebGPU backend)

    // ---- fixture views (tools/ref_pixal3d_cond_slat.py layout) ----
    npy::Array images = npy::load(cond_dir + "/s512_images.npy");             // [1,V,3,512,512]
    npy::Array camera_angle_x = npy::load(cond_dir + "/camera_angle_x.npy");  // [1,V]
    npy::Array transform_matrix = npy::load(cond_dir + "/transform_matrix.npy"); // [1,V,4,4]
    npy::Array mesh_scale = npy::load(cond_dir + "/mesh_scale.npy");          // [1]
    const int Vfix = (int)images.shape[1], S = (int)images.shape[3];
    const int V = (n_views > 0 && n_views < Vfix) ? n_views : Vfix;
    const int R = 32, Tn = 512;
    const size_t plane3 = (size_t)3 * S * S;
    vector<Pixal3dView> views(V);
    for (int v = 0; v < V; ++v) {
        views[v].rgb_premult.assign(images.data.data() + (size_t)v * plane3, images.data.data() + (size_t)(v + 1) * plane3);
        views[v].fov_x = camera_angle_x.data[v];
        std::memcpy(views[v].c2w, transform_matrix.data.data() + (size_t)v * 16, 16 * sizeof(float));
    }
    images.data.clear(); images.data.shrink_to_fit();
    rep("views: V=%d (fixture V=%d) S=%d R=%d T=%d mesh_scale=%.4f\n", V, Vfix, S, R, Tn, mesh_scale.data[0]);

    // ---- stage 1: DINOv3 -> NAF -> projections -> MV fusion, device-resident (one graph per view) ----
    auto t0 = std::chrono::steady_clock::now();
    Model dinov3 = Model::load(dinov3_gguf, 0);
    Model naf = Model::load(naf_gguf, 0);
    rep("dinov3: %zu tensors on %s, %.1f MB weights; naf: %zu tensors, %.1f MB; load %.0f ms\n",
        dinov3.tensors.size(), ggml_backend_name(dinov3.backend), dinov3.total_bytes() / 1048576.0,
        naf.tensors.size(), naf.total_bytes() / 1048576.0, ms_since(t0));
    Pixal3dCondStats st;
    Pixal3dSlatCondParams prm{S, R, Tn, mesh_scale.data[0]};
    Pixal3dCond cond = pixal3d_cond_slat_gpu(dinov3, naf, views, prm, &st);
    rep("cond: V=%d weights=%.1f MB cond=%.1f MB view_alloc=%.1f MB peak=%.1f MB total=%.0f ms slowest view=%.0f ms\n",
        st.views, st.weight_bytes / 1048576.0, st.cond_bytes / 1048576.0, st.view_alloc_bytes / 1048576.0,
        st.peak_bytes / 1048576.0, st.total_ms, st.view_ms_max);
    naf.free();
    dinov3.free();
    const size_t N3 = (size_t)R * R * R;
    const string zg_name = (V == Vfix) ? "s512_z_global.npy" : (V == 1 ? "s512_v1_z_global.npy" : "");
    const string zp_name = (V == Vfix) ? "s512_z_proj.npy" : (V == 1 ? "s512_v1_z_proj.npy" : "");
    if (!zg_name.empty() && file_exists(cond_dir + "/" + zg_name)) {
        rep_cmp("z_global vs PyTorch", cmp(cond.global, npy::load(cond_dir + "/" + zg_name).data));
        if (file_exists(cond_dir + "/" + zp_name)) {
            npy::Array zp = npy::load(cond_dir + "/" + zp_name);
            rep_cmp("z_proj_lr vs PyTorch", cmp(proj_half(cond.proj, N3, false), proj_half(zp.data, N3, false)));
            rep_cmp("z_proj_hr vs PyTorch", cmp(proj_half(cond.proj, N3, true), proj_half(zp.data, N3, true)));
        }
    }

    // ---- stage 2: shape-512 sampling over the fixture's active voxels (tools/ref_pixal3d_slat_sample.py) ----
    vector<int32_t> co; vector<int64_t> cshape;
    if (!load_npy_i32(sample_dir + "/coords.npy", co, cshape) || cshape.size() != 2) { rep("ERROR: coords.npy\n"); return 2; }
    const int64_t N = cshape[0], cw = cshape[1];
    vector<std::array<int, 3>> coords3(N);
    for (int64_t i = 0; i < N; ++i)
        coords3[i] = { co[i * cw + cw - 3], co[i * cw + cw - 2], co[i * cw + cw - 1] };
    npy::Array noise = npy::load(sample_dir + "/noise.npy");             // [N,32] row-major == ggml [32,N]
    npy::Array zg = npy::load(sample_dir + "/cond_global.npy");          // [1,5,1024]
    npy::Array zp = npy::load(sample_dir + "/cond_proj.npy");            // [N,2048]
    const int64_t Cin = noise.shape[1], Lc = zg.shape[1], Dc = zg.shape[2], Dp = zp.shape[1];
    if (zp.shape[0] != N || noise.shape[0] != N) { rep("ERROR: fixture token counts disagree\n"); return 2; }
    vector<float> sample(noise.data.begin(), noise.data.begin() + (size_t)Cin * N);
    vector<float> cond_g, proj;
    if (own_cond) {
        cond_g = cond.global;                                                      // token-major == ggml [1024,5]
        proj = pixal3d_gather_proj(cond.proj, R, cond.d_proj, coords3);            // [N,2048] at the fixture coords
        rep("sampling with the condition computed above (own cond)\n");
    } else {
        cond_g.assign(zg.data.begin(), zg.data.begin() + (size_t)Lc * Dc);
        proj.assign(zp.data.begin(), zp.data.begin() + (size_t)Dp * N);
        rep("sampling with the fixture condition (calibrated gate; own-cond parity reported above)\n");
    }
    cond.proj.clear(); cond.proj.shrink_to_fit();
    vector<float> neg_cond(cond_g.size(), 0.0f), neg_proj(proj.size(), 0.0f);

    t0 = std::chrono::steady_clock::now();
    Model mf = Model::load(flow_gguf, 0);
    const double flow_load_ms = ms_since(t0);
    DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (!dit_detect_proj_attn(mf, p) || p.d_proj != (int)Dp) { rep("ERROR: not a Pixal3D proj_attn checkpoint (d_proj=%d, fixture %lld)\n", p.d_proj, (long long)Dp); return 2; }
    rep("shape_flow: %zu tensors on %s, %.1f MB weights, load %.0f ms, proj_attn d_proj=%d, N=%lld\n", mf.tensors.size(),
        ggml_backend_name(mf.backend), mf.total_bytes() / 1048576.0, flow_load_ms, p.d_proj, (long long)N);
    t0 = std::chrono::steady_clock::now();
    DitRunner* run = make_sparse_runner(mf, p, coords3, (int)Lc);
    rep("dit graph: activation buffer %.1f MB, build+alloc %.0f ms\n", run->alloc_bytes() / 1048576.0, ms_since(t0));
    int n_fwd = 0; double fwd_ms = 0;
    FlowFwdProj fwd = [&](const vector<float>& x, float ts, const float* c, const float* pj) {
        const auto tf = std::chrono::steady_clock::now();
        vector<float> r = run->forward(x, ts, c, pj);
        fwd_ms += ms_since(tf); ++n_fwd;
        return r;
    };
    SamplerParams sp;   // trellis_cli.cpp shape_flow(): steps=12 gs=7.5 gr=0.5 gi=[0.6,1.0] rescale_t=3.0
    sp.steps = 12; sp.guidance_strength = 7.5f; sp.guidance_rescale = 0.5f;
    sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 3.0f; sp.sigma_min = 1e-5f;
    vector<vector<float>> trace;
    vector<float> out = sample_flow(fwd, sample, cond_g.data(), neg_cond.data(), proj.data(), neg_proj.data(), sp, &trace);
    rep("dit forwards: %d, %.1f ms each, %.1f s total\n", n_fwd, n_fwd ? fwd_ms / n_fwd : 0.0, fwd_ms / 1000.0);
    delete run; mf.free();

    // ---- per-step parity vs the PyTorch f32/bf16 references ([N,32] row-major == ggml bytes) ----
    rep("per-step latent parity ([N=%lld,%lld]):\n", (long long)N, (long long)Cin);
    g_n_steps = (int)trace.size();
    for (size_t k = 0; k < trace.size(); ++k) {
        g_steps.insert(g_steps.end(), trace[k].begin(), trace[k].end());
        rep("  step %2zu:\n", k + 1);
        rep_latent(trace[k]);
        string f32 = sample_dir + "/f32_x_step" + std::to_string(k + 1) + ".npy";
        string b16 = sample_dir + "/bf16_x_step" + std::to_string(k + 1) + ".npy";
        if (file_exists(f32)) rep_cmp("f32", cmp(trace[k], npy::load(f32).data));
        if (file_exists(b16)) rep_cmp("bf16", cmp(trace[k], npy::load(b16).data));
    }
    g_latent = out;
    rep("  final:\n");
    rep_latent(g_latent);
    if (file_exists(sample_dir + "/f32_x_final.npy")) rep_cmp("f32", cmp(g_latent, npy::load(sample_dir + "/f32_x_final.npy").data));
    if (file_exists(sample_dir + "/bf16_x_final.npy")) rep_cmp("bf16", cmp(g_latent, npy::load(sample_dir + "/bf16_x_final.npy").data));
    rep("total %.1f s\n", ms_since(t_all) / 1000.0);
    return 0;
}

// ---- shape decoder on a real SLAT fixture (slat_sample or hr_sample layout, as the native test) ----
int run_shape_decode_impl(const string& gguf, const string& fixture_dir, int res_arg, const string& out_dir) {
    const auto t_all = std::chrono::steady_clock::now();
    const string cp = file_exists(fixture_dir + "/coords.npy") ? fixture_dir + "/coords.npy" : fixture_dir + "/hr_coords.npy";
    const string sp = file_exists(fixture_dir + "/f32_slat.npy") ? fixture_dir + "/f32_slat.npy" : fixture_dir + "/f32_shape_slat.npy";
    vector<int32_t> co; vector<int64_t> cshape;
    if (!load_npy_i32(cp, co, cshape) || cshape.size() != 2) { rep("ERROR: %s\n", cp.c_str()); return 2; }
    npy::Array slat = npy::load(sp);
    const int64_t N = cshape[0], cw = cshape[1];
    if (slat.shape.size() != 2 || slat.shape[0] != N || slat.shape[1] != 32) { rep("ERROR: %s is not [N=%lld,32]\n", sp.c_str(), (long long)N); return 2; }
    vector<std::array<int, 3>> coords0(N);
    int cmax = 0;
    for (int64_t i = 0; i < N; ++i) {
        coords0[i] = { co[i * cw + cw - 3], co[i * cw + cw - 2], co[i * cw + cw - 1] };
        cmax = std::max({ cmax, coords0[i][0], coords0[i][1], coords0[i][2] });
    }
    const int in_res = cmax < 32 ? 32 : 64;
    const int res = res_arg > 0 ? res_arg : in_res * 16;
    vector<float> latent(slat.data.begin(), slat.data.begin() + (size_t)32 * N);   // [N,32] row-major == ggml [32,N]
    rep("input: N=%lld active voxels @res%d (%s) -> decode @res%d\n", (long long)N, in_res, cp.c_str(), res);
    setenv("TRELLIS_DBG_MEM", "1", 0);   // per-stage N -> M lines on stderr (part of the browser report)

    auto t0 = std::chrono::steady_clock::now();
    Model m = Model::load(gguf, 0);
    rep("shape_dec: %zu tensors on %s, %.1f MB weights, load %.0f ms\n", m.tensors.size(),
        ggml_backend_name(m.backend), m.total_bytes() / 1048576.0, ms_since(t0));
    t0 = std::chrono::steady_clock::now();
    ShapeOut so = shape_decode(m, latent, coords0, res);
    const double dec_ms = ms_since(t0);
    m.free();
    rep("decoded voxels @res%d = %zu  (%.1f s)\n", so.res, so.coords.size(), dec_ms / 1000.0);
    Mesh mesh = dual_grid_to_mesh(so);
    float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
    for (size_t i = 0; i + 2 < mesh.verts.size(); i += 3)
        for (int a = 0; a < 3; ++a) { mn[a] = std::min(mn[a], mesh.verts[i + a]); mx[a] = std::max(mx[a], mesh.verts[i + a]); }
    rep("mesh: V=%d F=%d bounds x[%.5f,%.5f] y[%.5f,%.5f] z[%.5f,%.5f]\n", mesh.V(), mesh.F(), mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]);

    // exact final-coordinate parity against the reference decode, when the fixture carries it
    vector<int32_t> rc; vector<int64_t> rshape;
    if (load_npy_i32(fixture_dir + "/f32_tex_coords.npy", rc, rshape) && rshape.size() == 2 && rshape[1] == 3) {
        auto key = [](int x, int y, int z) { return ((uint64_t)(uint32_t)x << 40) | ((uint64_t)(uint32_t)y << 20) | (uint32_t)z; };
        std::unordered_map<uint64_t, uint8_t> rs; rs.reserve((size_t)rshape[0] * 2);
        for (int64_t i = 0; i < rshape[0]; ++i) rs[key(rc[3 * i], rc[3 * i + 1], rc[3 * i + 2])] = 0;
        int64_t common = 0, extra = 0;
        for (const auto& c : so.coords) { auto it = rs.find(key(c[0], c[1], c[2])); if (it == rs.end()) ++extra; else { ++common; it->second = 1; } }
        int64_t missing = 0; for (auto& kv : rs) if (!kv.second) ++missing;
        rep("final coords vs reference: mine=%zu ref=%lld common=%lld extra=%lld missing=%lld -> %s\n",
            so.coords.size(), (long long)rshape[0], (long long)common, (long long)extra, (long long)missing,
            (extra == 0 && missing == 0) ? "EXACT" : "DIFFERENT");
    }
    if (!out_dir.empty()) {   // raw mesh + coords for the native scorer (--ext-mesh PREFIX)
        vector<int32_t> ci(so.coords.size() * 3);
        for (size_t i = 0; i < so.coords.size(); ++i) for (int a = 0; a < 3; ++a) ci[3 * i + a] = so.coords[i][a];
        npy::save(out_dir + "/verts.npy", mesh.verts.data(), { (int64_t)mesh.V(), 3 });
        npy::save_i32(out_dir + "/faces.npy", mesh.faces.data(), { (int64_t)mesh.F(), 3 });
        npy::save_i32(out_dir + "/coords.npy", ci.data(), { (int64_t)so.coords.size(), 3 });
        rep("wrote %s/{verts,faces,coords}.npy\n", out_dir.c_str());
    }
    rep("total %.1f s\n", ms_since(t_all) / 1000.0);
    return 0;
}

} // namespace

extern "C" {

// Runs the shape decoder on a real SLAT fixture directory (coords + f32 slat, optional
// f32_tex_coords for coordinate parity). res <= 0 derives the final resolution from the input
// grid (16x). out_dir != "" writes verts/faces/coords .npy there (MEMFS in the browser; the
// worker hands them to the page for download). Returns the report; "RESULT: OK" on success.
PIXAL3D_EXPORT const char* pixal3d_shape_decode_run(const char* shape_dec_gguf, const char* fixture_dir, int res, const char* out_dir) {
    g_report.clear();
    int rc;
    try {
        rc = run_shape_decode_impl(shape_dec_gguf, fixture_dir, res, out_dir ? out_dir : "");
    } catch (const std::exception& e) {
        rep("EXCEPTION: %s\n", e.what());
        rc = 1;
    }
    rep(rc == 0 ? "RESULT: OK\n" : "RESULT: FAIL\n");
    return g_report.c_str();
}

// Runs the SS path. Paths are inside the module's filesystem (WORKERFS mounts in the browser).
// n_views <= 0 uses every view in the fixture. Returns the report text; "RESULT: OK" on success.
PIXAL3D_EXPORT const char* pixal3d_ss_run(const char* dinov3_gguf, const char* ss_flow_gguf,
                                          const char* cond_dir, const char* sample_dir, int n_views) {
    g_report.clear();
    int rc;
    try {
        rc = run_impl(dinov3_gguf, ss_flow_gguf, cond_dir, sample_dir, n_views);
    } catch (const std::exception& e) {
        rep("EXCEPTION: %s\n", e.what());
        rc = 1;
    }
    rep(rc == 0 ? "RESULT: OK\n" : "RESULT: FAIL\n");
    return g_report.c_str();
}

// Final latent (torch [1,8,16,16,16] layout) and the per-step latents (n_steps x that) from the
// last pixal3d_ss_run.
PIXAL3D_EXPORT const float* pixal3d_ss_latent(void) { return g_latent.data(); }
PIXAL3D_EXPORT int pixal3d_ss_latent_size(void) { return (int)g_latent.size(); }
PIXAL3D_EXPORT const float* pixal3d_ss_steps(void) { return g_steps.data(); }
PIXAL3D_EXPORT int pixal3d_ss_n_steps(void) { return g_n_steps; }

// Runs the Shape-512 path (conditioning + sampling). own_cond != 0 samples with the condition
// computed in stage 1 instead of the fixture's. Latents: [N,32] row-major, via the getters below.
PIXAL3D_EXPORT const char* pixal3d_shape512_run(const char* dinov3_gguf, const char* naf_gguf, const char* flow_gguf,
                                                const char* cond_dir, const char* sample_dir, int n_views, int own_cond) {
    g_report.clear();
    int rc;
    try {
        rc = run_shape512_impl(dinov3_gguf, naf_gguf, flow_gguf, cond_dir, sample_dir, n_views, own_cond);
    } catch (const std::exception& e) {
        rep("EXCEPTION: %s\n", e.what());
        rc = 1;
    }
    rep(rc == 0 ? "RESULT: OK\n" : "RESULT: FAIL\n");
    return g_report.c_str();
}
PIXAL3D_EXPORT const float* pixal3d_shape512_latent(void) { return g_latent.data(); }
PIXAL3D_EXPORT int pixal3d_shape512_latent_size(void) { return (int)g_latent.size(); }
PIXAL3D_EXPORT const float* pixal3d_shape512_steps(void) { return g_steps.data(); }
PIXAL3D_EXPORT int pixal3d_shape512_n_steps(void) { return g_n_steps; }

} // extern "C"

#ifndef __EMSCRIPTEN__
int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "--shape-decode") == 0) {
        if (argc < 4) { fprintf(stderr, "usage: %s --shape-decode <shape_dec.gguf> <fixture_dir> [res] [out_dir]\n", argv[0]); return 1; }
        const char* r = pixal3d_shape_decode_run(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 0, argc > 5 ? argv[5] : "");
        return strstr(r, "RESULT: OK") ? 0 : 1;
    }
    if (argc >= 2 && strcmp(argv[1], "--shape512") == 0) {
        if (argc < 7) {
            fprintf(stderr, "usage: %s --shape512 <dinov3.gguf> <pixal3d_naf.gguf> <pixal3d_shape_flow_512_mv.gguf> <cond_slat_dir> <slat_sample_dir> [n_views] [own_cond] [dump_prefix]\n", argv[0]);
            return 1;
        }
        const int n_views = argc > 7 ? atoi(argv[7]) : 0;
        const int own = argc > 8 ? atoi(argv[8]) : 0;
        const char* r = pixal3d_shape512_run(argv[2], argv[3], argv[4], argv[5], argv[6], n_views, own);
        if (argc > 9 && !g_latent.empty()) {
            const int64_t n = (int64_t)g_latent.size() / 32;
            npy::save(string(argv[9]) + "x_final.npy", g_latent.data(), {n, 32});
            for (int k = 0; k < g_n_steps; ++k)
                npy::save(string(argv[9]) + "x_step" + std::to_string(k + 1) + ".npy", g_steps.data() + (size_t)k * g_latent.size(), {n, 32});
        }
        return strstr(r, "RESULT: OK") ? 0 : 1;
    }
    if (argc < 5) {
        fprintf(stderr, "usage: %s <dinov3.gguf> <pixal3d_ss_flow_mv.gguf> <cond_ss_dir> <ss_sample_dir> [n_views] [latent_out.npy]\n"
                        "       %s --shape512 <dinov3.gguf> <pixal3d_naf.gguf> <shape_flow_512.gguf> <cond_slat_dir> <slat_sample_dir> [n_views] [own_cond] [dump_prefix]\n", argv[0], argv[0]);
        return 1;
    }
    const int n_views = argc > 5 ? atoi(argv[5]) : 0;
    const char* r = pixal3d_ss_run(argv[1], argv[2], argv[3], argv[4], n_views);
    if (argc > 6 && !g_latent.empty()) npy::save(argv[6], g_latent.data(), {1, 8, 16, 16, 16});
    return strstr(r, "RESULT: OK") ? 0 : 1;
}
#endif
