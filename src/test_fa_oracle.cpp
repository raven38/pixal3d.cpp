// trellis-test-fa-oracle: one DiT attention (src/dit.cpp::sdpa via dit_sdpa) at the real HR token
// count, every K/V storage type and the exact (--no-fa) path, each against a float64 HOST reference
// -- on the CPU backend and on the build's GPU device. This is the op-level check behind
// docs/design/2026-09-20-metal-fa-f16-kv.md: it says (a) whether the exact SDPA path is a valid
// F32 reference for the E2E comparisons (its error against float64 must be far below the FA
// variants'), and (b) how far each FA K/V type sits from float64 on THIS backend (Metal: bf16 K/V
// accumulates the output in half and stores q as bfloat; f16 K/V accumulates in float).
//
// Inputs are synthetic (the E2E range probes are the real-input check): q, k ~ N(0, sq^2) per
// element, v ~ N(0, sv^2), two regimes -- "flat" (sq = 1: scores ~ N(0,1), the softmax spreads over
// thousands of keys, the worst case for a half accumulator) and "peaked" (sq = 3: scores ~ N(0,9),
// a few keys dominate, closer to a trained attention). The float64 reference is computed for a
// sample of queries (all keys), which is what the error statistics are taken over.
//
// Usage: trellis-test-fa-oracle [N=17612] [gpu=0] [n_query_samples=64] [--gpu-only]
//   gpu: -1 = CPU only, else CPU + the build's GPU device (--gpu-only skips the CPU backend, whose
//   exact path at N=17612 takes ~45 s per config).
// Prints per (backend, regime, config): rel_rms = rms(out - ref) / rms(ref), max|d|, bias =
// mean(out - ref) / rms(ref), nonfinite; then two verdicts. FA_ORACLE_RESULT: OK if every config
// is finite with rel_rms < 1e-2 (exit bit 1 otherwise). FA_ORACLE_EXACT_REF: OK if the exact
// path's rel_rms < 1e-5 on every backend (exit bit 2 otherwise) -- on Metal it is NOT: the F32
// mul_mat there (kernel_mul_mm_f32_f32) stages both operands as half in threadgroup memory
// (simdgroup_half8x8), so --no-fa on Metal is a half-operand / float-accumulate path, at the
// same error level as f16 K/V FlashAttention (measured 2026-09-20, M4 Max: exact 4.5e-4 vs
// FA f16 3.7e-4 in the flat regime).
#include "dit.h"
#include "trellis_args.h"
#include "npy.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

struct Config { const char* name; bool no_fa; int kv; };
static const Config kConfigs[] = {
    { "exact (--no-fa)", true,  trellis::FA_KV_BF16 },
    { "FA bf16 K/V",     false, trellis::FA_KV_BF16 },
    { "FA f16 K/V",      false, trellis::FA_KV_F16  },
    { "FA f32 K/V",      false, trellis::FA_KV_F32  },
};

struct Stats { double rel_rms = 0, max_abs = 0, bias = 0; size_t bad = 0; double secs = 0; bool ok = false; };

// out: [d_model, L] channel-major (= [L][nh][hd]); ref computed for the sampled queries only.
static Stats score(const std::vector<float>& out, const std::vector<double>& ref, const std::vector<int>& qs,
                   int hd, int nh) {
    Stats s; double se = 0, sr = 0, sd = 0, mx = 0; size_t n = 0;
    for (size_t qi = 0; qi < qs.size(); ++qi) {
        for (int h = 0; h < nh; ++h) for (int d = 0; d < hd; ++d) {
            const double o = out[(size_t)qs[qi] * nh * hd + (size_t)h * hd + d];
            const double r = ref[(qi * nh + h) * hd + d];
            if (!std::isfinite(o)) { s.bad++; continue; }
            const double e = o - r;
            se += e * e; sr += r * r; sd += e; if (std::fabs(e) > mx) mx = std::fabs(e); n++;
        }
    }
    const double rr = n ? std::sqrt(sr / n) : 1;
    s.rel_rms = n ? std::sqrt(se / n) / rr : NAN; s.max_abs = mx; s.bias = n ? (sd / n) / rr : NAN;
    for (float x : out) if (!std::isfinite(x)) { s.bad = 1; break; }
    return s;
}

static int run_on(ggml_backend_t backend, const char* label, int N, int nsamp) {
    const int hd = 128, nh = 12, dm = hd * nh;
    const float scale = 1.0f / std::sqrt((float)hd);
    int rc = 0;
    for (int regime = 0; regime < 2; ++regime) {
        const float sq = regime == 0 ? 1.0f : 3.0f, sv = 4.0f;
        std::mt19937 rng(11 + regime);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> q((size_t)hd * nh * N), k(q.size()), v(q.size());   // [hd, nh, L]
        for (auto& x : q) x = sq * nd(rng);
        for (auto& x : k) x = sq * nd(rng);
        for (auto& x : v) x = sv * nd(rng);
        // float64 reference for nsamp queries spread over the sequence
        std::vector<int> qs; for (int i = 0; i < nsamp; ++i) qs.push_back((int)((int64_t)i * (N - 1) / std::max(1, nsamp - 1)));
        std::vector<double> ref((size_t)nsamp * nh * hd, 0.0);
        std::vector<double> sc(N);
        for (size_t qi = 0; qi < qs.size(); ++qi) for (int h = 0; h < nh; ++h) {
            const float* qp = &q[((size_t)qs[qi] * nh + h) * hd];
            double m = -1e300;
            for (int j = 0; j < N; ++j) {
                const float* kp = &k[((size_t)j * nh + h) * hd];
                double d = 0; for (int t = 0; t < hd; ++t) d += (double)qp[t] * kp[t];
                sc[j] = d * scale; if (sc[j] > m) m = sc[j];
            }
            double z = 0; for (int j = 0; j < N; ++j) { sc[j] = std::exp(sc[j] - m); z += sc[j]; }
            double* o = &ref[(qi * nh + h) * hd];
            for (int j = 0; j < N; ++j) { const double p = sc[j] / z; const float* vp = &v[((size_t)j * nh + h) * hd];
                for (int t = 0; t < hd; ++t) o[t] += p * vp[t]; }
        }
        printf("[%s] regime=%s  N=%d  hd=%d nh=%d  (%d sampled queries vs float64)\n", label, regime == 0 ? "flat sq=1" : "peaked sq=3", N, hd, nh, nsamp);
        for (const Config& cfg : kConfigs) {
            trellis::g_no_fa = cfg.no_fa; trellis::g_fa_kv = cfg.kv;
            ggml_init_params ip = { ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false), nullptr, true };
            ggml_context* ctx = ggml_init(ip);
            ggml_tensor* gq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, N); ggml_set_input(gq);
            ggml_tensor* gk = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, N); ggml_set_input(gk);
            ggml_tensor* gv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, N); ggml_set_input(gv);
            ggml_tensor* out = trellis::dit_sdpa(ctx, backend, gq, gk, gv, dm); ggml_set_output(out);
            ggml_cgraph* g = ggml_new_graph_custom(ctx, 8192, false);
            ggml_build_forward_expand(g, out);
            // supports_op check (e.g. bf16 FA on a device without bfloat): report and skip, don't abort
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            bool supported = true;
            if (dev) for (int i = 0; i < ggml_graph_n_nodes(g); ++i) if (!ggml_backend_dev_supports_op(dev, ggml_graph_node(g, i))) { supported = false; break; }
            if (!supported) { printf("  %-16s unsupported on this backend (skipped)\n", cfg.name); ggml_free(ctx); continue; }
            ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(alloc, g)) { fprintf(stderr, "  %s: alloc failed\n", cfg.name); rc |= 1; ggml_free(ctx); continue; }
            Stats st;
            for (int rep = 0; rep < 2; ++rep) {   // rep 0 warms the pipelines; rep 1 is timed
                // Re-upload before EVERY compute: gallocr reuses the input buffers for later nodes of the
                // same graph, so after one compute the inputs hold garbage (flow_runner.cpp does the same).
                ggml_backend_tensor_set(gq, q.data(), 0, q.size() * 4);
                ggml_backend_tensor_set(gk, k.data(), 0, k.size() * 4);
                ggml_backend_tensor_set(gv, v.data(), 0, v.size() * 4);
                const auto t0 = std::chrono::steady_clock::now();
                if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "  %s: compute failed\n", cfg.name); rc |= 1; break; }
                st.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                st.ok = true;
            }
            if (st.ok) {
                std::vector<float> o((size_t)dm * N);
                ggml_backend_tensor_get(out, o.data(), 0, o.size() * 4);
                Stats s = score(o, ref, qs, hd, nh); s.secs = st.secs;
                if (const char* dd = std::getenv("FA_ORACLE_DUMP")) {
                    const std::string b = std::string(dd) + "/" + (regime ? "peaked_" : "flat_") + std::to_string((int)(&cfg - kConfigs));
                    npy::save(b + "_q.npy", q.data(), { N, nh, hd }); npy::save(b + "_k.npy", k.data(), { N, nh, hd });
                    npy::save(b + "_v.npy", v.data(), { N, nh, hd }); npy::save(b + "_out.npy", o.data(), { N, nh, hd });
                }
                if (std::getenv("FA_ORACLE_DEBUG")) {
                    for (size_t qi = 0; qi < std::min<size_t>(qs.size(), 2); ++qi) for (int h = 0; h < 2; ++h) {
                        printf("    q%d h%d  out:", qs[qi], h);
                        for (int d = 0; d < 4; ++d) printf(" %+.4f", o[(size_t)qs[qi] * nh * hd + (size_t)h * hd + d]);
                        printf("   ref:");
                        for (int d = 0; d < 4; ++d) printf(" %+.4f", ref[(qi * nh + h) * hd + d]);
                        printf("   v[j=0..1]:");
                        for (int j = 0; j < std::min(N, 2); ++j) printf(" %+.4f", v[((size_t)j * nh + h) * hd + 0]);
                        printf("\n");
                    }
                }
                printf("  %-16s rel_rms=%.3e  max|d|=%.3e  bias=%+.2e  nonfinite=%zu  %.3fs\n", cfg.name, s.rel_rms, s.max_abs, s.bias, s.bad, s.secs);
                if (s.bad || !(s.rel_rms < 1e-2)) rc |= 1;
                if (cfg.no_fa && !(s.rel_rms < 1e-5)) { printf("  ** exact path is not a valid F32 reference here (rel_rms >= 1e-5)\n"); rc |= 2; }
            }
            ggml_gallocr_free(alloc); ggml_free(ctx);
        }
    }
    return rc;
}

int main(int argc, char** argv) {
    std::vector<std::string> pos; bool gpu_only = false;
    for (int i = 1; i < argc; ++i) { if (std::string(argv[i]) == "--gpu-only") gpu_only = true; else pos.push_back(argv[i]); }
    const int N = pos.size() > 0 ? atoi(pos[0].c_str()) : 17612;
    const int gpu = pos.size() > 1 ? atoi(pos[1].c_str()) : 0;
    const int nsamp = pos.size() > 2 ? atoi(pos[2].c_str()) : 64;
    setvbuf(stdout, nullptr, _IONBF, 0);
    int rc = 0;
    if (!gpu_only) {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        rc |= run_on(cpu, "cpu", N, nsamp);
        ggml_backend_free(cpu);
    }
    if (gpu >= 0) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (dev) {
            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            if (b) { rc |= run_on(b, ggml_backend_dev_name(dev), N, nsamp); ggml_backend_free(b); }
            else { fprintf(stderr, "gpu init failed\n"); rc |= 1; }
        } else {
            printf("no GPU device in this build; CPU only\n");
        }
    }
    printf("FA_ORACLE_RESULT: %s\n", (rc & 1) == 0 ? "OK" : "FAIL");
    printf("FA_ORACLE_EXACT_REF: %s\n", (rc & 2) == 0 ? "OK" : "NOT-F32 (the --no-fa path is not an F32 reference on some tested backend)");
    return rc;
}
