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
//   Texture:   hr_sample fixture (Shape-1024's hr_coords + tex_1024 condition + tex_noise) and a
//                    normalized Shape-1024 SLAT (the fixture's or a Shape-1024 run's x_final) ->
//                    sparse DitRunner in_ch=64/out_ch=32 (state ; concat) -> sample_flow (12
//                    single forwards, guidance 1.0) -> final texture latent [N,32]
//                    (pixal3d_texture_run; docs/spec/32-texture-flow-webgpu-prep.md)
//
// The decoders are not run here (Conv3D / sparse conv have no WebGPU kernels yet); the
// latents are returned so the native tests can score/decode them.
#include "pixal3d_cond.h"
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

// Texture-1024 path (docs/spec/32-texture-flow-webgpu-prep.md): the texture SLAT flow over the
// Shape-1024 stage's token list. Inputs, all from tools/ref_pixal3d_hr_sample.py's fixture
// (sample_dir): hr_coords.npy [N,4] i32 (the LR->HR quantized, deduplicated coords the Shape-1024
// flow ran on), tex_noise.npy [N,32], tex_cond_global.npy [1,5,1024] + tex_cond_proj.npy [N,2048]
// (PyTorch's tex_1024 condition -- S=1024, R=64, NAF T=1024 -- gathered at these tokens),
// tex_sampler_params.npy (12 steps, guidance 1.0 -> one forward per step, no CFG pair) and the
// 32-channel concat half. That half is the NORMALIZED Shape-1024 SLAT (== that stage's x_final,
// see test_pixal3d_slat_sample.cpp --concat-cond): `concat_cond_npy` names the file to use (a
// Shape-1024 run's own x_final dump, or "" for the fixture's f32_tex_concat_cond.npy). The DiT is
// the same sparse ProjectAttention graph as Shape-1024 with in_ch=64 / out_ch=32; every forward
// rebuilds the [N,64] input = [state(0:32) ; concat(32:64)] like trellis_cli.cpp's tex lambda.
// own_cond != 0 additionally computes the tex_1024 condition on the device (pixal3d_cond_slat_gpu,
// S=1024/R=64/T=1024: DINOv3 at 1024^2, NAF at T=1024 -> a 4 GB [1024, 1024^2] map per view plus
// dense [64^3, 2048] accumulators) -- wired for completeness and reported against the fixture, but
// expected to exceed a 4 GiB adapter until the on-demand NAF / coords-gathered accumulation land;
// the default (own_cond = 0) samples with the fixture condition and never loads DINOv3/NAF.
int run_texture_impl(const string& dinov3_gguf, const string& naf_gguf, const string& flow_gguf,
                     const string& cond_dir, const string& sample_dir, int n_views, int own_cond,
                     const string& concat_cond_npy) {
    g_latent.clear(); g_steps.clear(); g_n_steps = 0;
    const auto t_all = std::chrono::steady_clock::now();
    g_no_fa = true;   // exact chunked SDPA (no BF16 FlashAttention on the WebGPU backend)
    const int R = 64, S = 1024, Tn = 1024;

    // ---- the token list (Shape-1024's) ----
    vector<int32_t> co; vector<int64_t> cshape;
    if (!load_npy_i32(sample_dir + "/hr_coords.npy", co, cshape) || cshape.size() != 2) { rep("ERROR: hr_coords.npy\n"); return 2; }
    const int64_t N = cshape[0], cw = cshape[1];
    vector<std::array<int, 3>> coords3(N);
    for (int64_t i = 0; i < N; ++i)
        coords3[i] = { co[i * cw + cw - 3], co[i * cw + cw - 2], co[i * cw + cw - 1] };
    rep("flow tokens: N=%lld active voxels at grid %d (hr_coords.npy, shared with Shape-1024)\n", (long long)N, R);

    // ---- fixture condition + noise + concat half ----
    npy::Array noise = npy::load(sample_dir + "/tex_noise.npy");           // [N,32] row-major == ggml [32,N]
    npy::Array zg = npy::load(sample_dir + "/tex_cond_global.npy");        // [1,5,1024]
    npy::Array zp = npy::load(sample_dir + "/tex_cond_proj.npy");          // [N,2048]
    const int64_t Cin = noise.shape[1], Lc = zg.shape[1], Dc = zg.shape[2], Dp = zp.shape[1];
    if (zp.shape[0] != N || noise.shape[0] != N) { rep("ERROR: fixture token counts disagree (noise %lld, proj %lld, coords %lld)\n", (long long)noise.shape[0], (long long)zp.shape[0], (long long)N); return 2; }
    vector<float> sample(noise.data.begin(), noise.data.begin() + (size_t)Cin * N);
    const string cc_path = concat_cond_npy.empty() ? sample_dir + "/f32_tex_concat_cond.npy" : concat_cond_npy;
    npy::Array cc = npy::load(cc_path);                                    // [N,32] normalized shape SLAT
    if (cc.shape.size() != 2 || cc.shape[0] != N || cc.shape[1] != Cin) { rep("ERROR: concat cond %s is not [N=%lld,%lld]\n", cc_path.c_str(), (long long)N, (long long)Cin); return 2; }
    vector<float> concat_cond(cc.data.begin(), cc.data.begin() + (size_t)Cin * N);
    rep("concat half: %s [%lld,%lld]\n", cc_path.c_str(), (long long)N, (long long)Cin);
    if (!concat_cond_npy.empty() && file_exists(sample_dir + "/f32_tex_concat_cond.npy"))
        rep_cmp("concat vs f32 ref", cmp(concat_cond, npy::load(sample_dir + "/f32_tex_concat_cond.npy").data));

    // ---- optional: tex_1024 conditioning on the device (S=1024, R=64, T=1024) ----
    vector<float> cond_g, proj;
    if (own_cond) {
        npy::Array images = npy::load(cond_dir + "/s1024_images.npy");            // [1,V,3,1024,1024]
        npy::Array camera_angle_x = npy::load(cond_dir + "/camera_angle_x.npy");  // [1,V]
        npy::Array transform_matrix = npy::load(cond_dir + "/transform_matrix.npy"); // [1,V,4,4]
        npy::Array mesh_scale = npy::load(cond_dir + "/mesh_scale.npy");          // [1]
        const int Vfix = (int)images.shape[1], Sfix = (int)images.shape[3];
        if (Sfix != S) { rep("ERROR: s1024_images.npy is %d^2, tex_1024 needs %d^2\n", Sfix, S); return 2; }
        const int V = (n_views > 0 && n_views < Vfix) ? n_views : Vfix;
        const size_t plane3 = (size_t)3 * S * S;
        vector<Pixal3dView> views(V);
        for (int v = 0; v < V; ++v) {
            views[v].rgb_premult.assign(images.data.data() + (size_t)v * plane3, images.data.data() + (size_t)(v + 1) * plane3);
            views[v].fov_x = camera_angle_x.data[v];
            std::memcpy(views[v].c2w, transform_matrix.data.data() + (size_t)v * 16, 16 * sizeof(float));
        }
        images.data.clear(); images.data.shrink_to_fit();
        rep("views: V=%d (fixture V=%d) S=%d R=%d T=%d mesh_scale=%.4f\n", V, Vfix, S, R, Tn, mesh_scale.data[0]);
        rep("own cond: expected device buffers -- NAF map [1024,%d] f32 = %.0f MB (x2), dense proj accumulators [%d,2048] = %.0f MB; this exceeds a 4 GiB adapter (spec 32 section 7)\n",
            Tn * Tn, (double)1024 * Tn * Tn * 4 / 1048576.0, R * R * R, (double)R * R * R * 2048 * 4 / 1048576.0);
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
        cond_g = cond.global;                                                   // token-major == ggml [1024,5]
        proj = pixal3d_gather_proj(cond.proj, R, cond.d_proj, coords3);         // [N,2048] at the tokens
        cond.proj.clear(); cond.proj.shrink_to_fit();
        if (V == Vfix) {
            rep_cmp("z_global vs PyTorch", cmp(cond_g, npy::load(sample_dir + "/tex_cond_global.npy").data));
            rep_cmp("proj_lr @tokens vs PyTorch", cmp(proj_half(proj, (size_t)N, false), proj_half(zp.data, (size_t)N, false)));
            rep_cmp("proj_hr @tokens vs PyTorch", cmp(proj_half(proj, (size_t)N, true), proj_half(zp.data, (size_t)N, true)));
        }
        rep("sampling with the condition computed above (own cond)\n");
    } else {
        cond_g.assign(zg.data.begin(), zg.data.begin() + (size_t)Lc * Dc);
        proj.assign(zp.data.begin(), zp.data.begin() + (size_t)Dp * N);
        rep("sampling with the fixture condition (tex_cond_global/tex_cond_proj; DINOv3/NAF not loaded)\n");
    }
    zp.data.clear(); zp.data.shrink_to_fit();
    vector<float> neg_cond(cond_g.size(), 0.0f), neg_proj(proj.size(), 0.0f);

    // ---- texture flow sampling ----
    auto t0 = std::chrono::steady_clock::now();
    Model mf = Model::load(flow_gguf, 0);
    const double flow_load_ms = ms_since(t0);
    DiTParams p; p.in_ch = (int)(2 * Cin); p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (!dit_detect_proj_attn(mf, p) || p.d_proj != (int)Dp) { rep("ERROR: not a Pixal3D proj_attn checkpoint (d_proj=%d, fixture %lld)\n", p.d_proj, (long long)Dp); return 2; }
    rep("tex_flow: %zu tensors on %s, %.1f MB weights, load %.0f ms, proj_attn d_proj=%d, in_ch=%d out_ch=%d, N=%lld\n", mf.tensors.size(),
        ggml_backend_name(mf.backend), mf.total_bytes() / 1048576.0, flow_load_ms, p.d_proj, p.in_ch, p.out_ch, (long long)N);
    t0 = std::chrono::steady_clock::now();
    DitRunner* run = make_sparse_runner(mf, p, coords3, (int)Lc);
    rep("dit graph: activation buffer %.1f MB, build+alloc %.0f ms\n", run->alloc_bytes() / 1048576.0, ms_since(t0));
    rep("memory (%s): weights %.1f MB | dit activations %.1f MB | cond per forward %.1f MB (+ zero neg %.1f MB) | state [N,%lld] %.2f MB | tex concat half %.2f MB + [N,%lld] input %.2f MB per forward | device-resident sum %.1f MB\n",
        ggml_backend_name(mf.backend), mf.total_bytes() / 1048576.0, run->alloc_bytes() / 1048576.0,
        (cond_g.size() + proj.size()) * 4 / 1048576.0, (neg_cond.size() + neg_proj.size()) * 4 / 1048576.0,
        (long long)Cin, sample.size() * 4 / 1048576.0, concat_cond.size() * 4 / 1048576.0, (long long)(2 * Cin), (double)2 * Cin * N * 4 / 1048576.0,
        ((double)mf.total_bytes() + (double)run->alloc_bytes() + (double)(cond_g.size() + proj.size() + neg_cond.size() + neg_proj.size() + sample.size() + 2 * Cin * N) * 4) / 1048576.0);   // summed in double: the byte total exceeds 32-bit size_t under wasm32
    int n_fwd = 0; double fwd_ms = 0;
    vector<float> x64((size_t)2 * Cin * N);
    FlowFwdProj fwd = [&](const vector<float>& x, float ts, const float* c, const float* pj) {
        const auto tf = std::chrono::steady_clock::now();
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < Cin; ++k) x64[(size_t)k + 2 * Cin * n]       = x[(size_t)k + Cin * n];
            for (int64_t k = 0; k < Cin; ++k) x64[(size_t)Cin + k + 2 * Cin * n] = concat_cond[(size_t)k + Cin * n];
        }
        vector<float> r = run->forward(x64, ts, c, pj);
        const double ms = ms_since(tf);
        fwd_ms += ms; ++n_fwd;
        rep("  forward %2d: t_scaled=%.3f %.1f ms\n", n_fwd, ts, ms);
        return r;
    };
    SamplerParams sp;   // trellis_cli.cpp tex flow: steps=12 gs=1.0 gr=0.0 gi=[0.6,0.9] rescale_t=3.0
    sp.steps = 12; sp.guidance_strength = 1.0f; sp.guidance_rescale = 0.0f;
    sp.gi0 = 0.6f; sp.gi1 = 0.9f; sp.rescale_t = 3.0f; sp.sigma_min = 1e-5f;
    if (file_exists(sample_dir + "/tex_sampler_params.npy")) {
        npy::Array spa = npy::load(sample_dir + "/tex_sampler_params.npy");   // (steps, gs, gr, gi0, gi1, rescale_t)
        if (spa.numel() >= 6) {
            sp.steps = (int)std::lround((double)spa.data[0]); sp.guidance_strength = spa.data[1]; sp.guidance_rescale = spa.data[2];
            sp.gi0 = spa.data[3]; sp.gi1 = spa.data[4]; sp.rescale_t = spa.data[5];
            rep("sampler params from tex_sampler_params.npy: steps=%d gs=%.3f gr=%.3f gi=[%.2f,%.2f] rescale_t=%.2f\n",
                sp.steps, sp.guidance_strength, sp.guidance_rescale, sp.gi0, sp.gi1, sp.rescale_t);
        }
    }
    vector<vector<float>> trace;
    vector<float> out = sample_flow(fwd, sample, cond_g.data(), neg_cond.data(), proj.data(), neg_proj.data(), sp, &trace);
    rep("dit forwards: %d, %.1f ms each, %.1f s total\n", n_fwd, n_fwd ? fwd_ms / n_fwd : 0.0, fwd_ms / 1000.0);
    const string backend_name = ggml_backend_name(mf.backend);
    delete run; mf.free();

    // ---- per-step parity vs the PyTorch f32/bf16 references ([N,32] row-major == ggml bytes) ----
    rep("per-step latent parity ([N=%lld,%lld]):\n", (long long)N, (long long)Cin);
    g_n_steps = (int)trace.size();
    for (size_t k = 0; k < trace.size(); ++k) {
        g_steps.insert(g_steps.end(), trace[k].begin(), trace[k].end());
        rep("  step %2zu:\n", k + 1);
        rep_latent(trace[k]);
        string f32 = sample_dir + "/f32_tex_x_step" + std::to_string(k + 1) + ".npy";
        string b16 = sample_dir + "/bf16_tex_x_step" + std::to_string(k + 1) + ".npy";
        if (file_exists(f32)) rep_cmp("f32", cmp(trace[k], npy::load(f32).data));
        if (file_exists(b16)) rep_cmp("bf16", cmp(trace[k], npy::load(b16).data));
    }
    g_latent = out;
    rep("  final:\n");
    rep_latent(g_latent);
    Cmp cf{}, cb{}; bool hf = false, hb = false;
    if (file_exists(sample_dir + "/f32_tex_x_final.npy")) { cf = cmp(g_latent, npy::load(sample_dir + "/f32_tex_x_final.npy").data); hf = true; rep_cmp("f32", cf); }
    if (file_exists(sample_dir + "/bf16_tex_x_final.npy")) { cb = cmp(g_latent, npy::load(sample_dir + "/bf16_tex_x_final.npy").data); hb = true; rep_cmp("bf16", cb); }
    if (hf && hb) {
        // same PASS criterion as trellis-test-pixal3d-slat-sample: rel(mine, f32) <= max(2 * rel(f32, bf16), 5e-2)
        const Cmp calib = cmp(npy::load(sample_dir + "/bf16_tex_x_final.npy").data, npy::load(sample_dir + "/f32_tex_x_final.npy").data);
        const double thr = std::max(2.0 * calib.rel, 5e-2);
        rep("  verdict: rel(mine, f32 ref) = %.4e ; threshold = max(2*rel(f32,bf16)=%.4e, 5e-2) = %.4e -> %s\n",
            cf.rel, 2.0 * calib.rel, thr, cf.rel <= thr ? "PASS" : "FAIL");
    }
    rep("summary: stage=texture backend=%s N=%lld in_ch=%d steps=%d forwards=%d fwd_ms=%.1f rel_final=%.4e cos_final=%.6f%s\n",
        backend_name.c_str(), (long long)N, p.in_ch, sp.steps, n_fwd, n_fwd ? fwd_ms / n_fwd : 0.0, hf ? cf.rel : -1.0, hf ? cf.cos : -1.0,
        concat_cond_npy.empty() ? "" : " concat_cond=external");
    rep("total %.1f s\n", ms_since(t_all) / 1000.0);
    return 0;
}

} // namespace

extern "C" {

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

// Runs the Texture-1024 path over the Shape-1024 token list: sample_dir = the hr_sample fixture
// (hr_coords, tex_noise, tex_cond_*, tex_sampler_params, f32_/bf16_tex_x_*), concat_cond_npy = the
// normalized Shape-1024 SLAT [N,32] to concatenate ("" -> sample_dir/f32_tex_concat_cond.npy; pass
// a Shape-1024 run's x_final dump to chain the stages). own_cond != 0 also computes the tex_1024
// condition on the device from cond_dir's s1024_images + cameras (see run_texture_impl) --
// dinov3_gguf/naf_gguf/cond_dir are otherwise unused and may be "". Latents: [N,32] row-major.
PIXAL3D_EXPORT const char* pixal3d_texture_run(const char* dinov3_gguf, const char* naf_gguf, const char* tex_flow_gguf,
                                               const char* cond_dir, const char* sample_dir, int n_views, int own_cond,
                                               const char* concat_cond_npy) {
    g_report.clear();
    int rc;
    try {
        rc = run_texture_impl(dinov3_gguf ? dinov3_gguf : "", naf_gguf ? naf_gguf : "", tex_flow_gguf,
                              cond_dir ? cond_dir : "", sample_dir, n_views, own_cond, concat_cond_npy ? concat_cond_npy : "");
    } catch (const std::exception& e) {
        rep("EXCEPTION: %s\n", e.what());
        rc = 1;
    }
    rep(rc == 0 ? "RESULT: OK\n" : "RESULT: FAIL\n");
    return g_report.c_str();
}
PIXAL3D_EXPORT const float* pixal3d_texture_latent(void) { return g_latent.data(); }
PIXAL3D_EXPORT int pixal3d_texture_latent_size(void) { return (int)g_latent.size(); }
PIXAL3D_EXPORT const float* pixal3d_texture_steps(void) { return g_steps.data(); }
PIXAL3D_EXPORT int pixal3d_texture_n_steps(void) { return g_n_steps; }

} // extern "C"

#ifndef __EMSCRIPTEN__
int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "--texture") == 0) {
        // --texture <tex_flow.gguf> <hr_sample_dir> [concat_cond.npy|-] [own_cond] [dinov3.gguf] [naf.gguf] [cond_slat_dir] [n_views] [dump_prefix]
        if (argc < 4) {
            fprintf(stderr, "usage: %s --texture <pixal3d_tex_flow_1024_mv.gguf> <hr_sample_dir> [concat_cond.npy|-] [own_cond] [dinov3.gguf] [pixal3d_naf.gguf] [cond_slat_dir] [n_views] [dump_prefix]\n", argv[0]);
            return 1;
        }
        const char* cc = (argc > 4 && strcmp(argv[4], "-") != 0) ? argv[4] : "";
        const int own = argc > 5 ? atoi(argv[5]) : 0;
        const char* dinov3 = argc > 6 ? argv[6] : "";
        const char* naf = argc > 7 ? argv[7] : "";
        const char* cond_dir = argc > 8 ? argv[8] : "";
        const int n_views = argc > 9 ? atoi(argv[9]) : 0;
        const char* r = pixal3d_texture_run(dinov3, naf, argv[2], cond_dir, argv[3], n_views, own, cc);
        if (argc > 10 && !g_latent.empty()) {
            const int64_t n = (int64_t)g_latent.size() / 32;
            npy::save(string(argv[10]) + "tex_x_final.npy", g_latent.data(), {n, 32});
            for (int k = 0; k < g_n_steps; ++k)
                npy::save(string(argv[10]) + "tex_x_step" + std::to_string(k + 1) + ".npy", g_steps.data() + (size_t)k * g_latent.size(), {n, 32});
        }
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
                        "       %s --shape512 <dinov3.gguf> <pixal3d_naf.gguf> <shape_flow_512.gguf> <cond_slat_dir> <slat_sample_dir> [n_views] [own_cond] [dump_prefix]\n"
                        "       %s --texture <tex_flow_1024.gguf> <hr_sample_dir> [concat_cond.npy|-] [own_cond] [dinov3.gguf] [naf.gguf] [cond_slat_dir] [n_views] [dump_prefix]\n", argv[0], argv[0], argv[0]);
        return 1;
    }
    const int n_views = argc > 5 ? atoi(argv[5]) : 0;
    const char* r = pixal3d_ss_run(argv[1], argv[2], argv[3], argv[4], n_views);
    if (argc > 6 && !g_latent.empty()) npy::save(argv[6], g_latent.data(), {1, 8, 16, 16, 16});
    return strstr(r, "RESULT: OK") ? 0 : 1;
}
#endif
