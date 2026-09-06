// Pixal3D SS path as a C ABI for the WASM/browser target (web/ss/) -- and, under a native
// build, as a plain executable for debugging the same code. JavaScript's only jobs are to
// hand over files (WORKERFS mounts of the GGUFs and fixtures), call pixal3d_ss_run(), and
// display the returned report / save the returned latent; every stage below runs in shared
// C++ on the ggml backend the Model picked (WebGPU in the browser):
//
//   fixture views -> pixal3d_cond_ss_gpu (DINOv3 -> projection -> MV average, per view)
//                 -> DitRunner (ProjectAttention SS flow, exact SDPA) -> sample_flow (12 Euler
//                    steps, CFG) -> final latent [8,16,16,16]
//
// The SS *decoder* is not run here (Conv3D has no WebGPU kernel yet); the latent is returned
// so trellis-test-pixal3d-ss-sample can decode it natively and compute the occupancy IoU.
#include "pixal3d_cond.h"
#include "dinov3.h"
#include "trellis_model.h"
#include "flow_runner.h"
#include "dit.h"
#include "trellis_args.h"
#include "npy.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
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

} // extern "C"

#ifndef __EMSCRIPTEN__
int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <dinov3.gguf> <pixal3d_ss_flow_mv.gguf> <cond_ss_dir> <ss_sample_dir> [n_views] [latent_out.npy]\n", argv[0]);
        return 1;
    }
    const int n_views = argc > 5 ? atoi(argv[5]) : 0;
    const char* r = pixal3d_ss_run(argv[1], argv[2], argv[3], argv[4], n_views);
    if (argc > 6 && !g_latent.empty()) npy::save(argv[6], g_latent.data(), {1, 8, 16, 16, 16});
    return strstr(r, "RESULT: OK") ? 0 : 1;
}
#endif
