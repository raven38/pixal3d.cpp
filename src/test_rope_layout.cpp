// trellis-test-rope-layout: checks the DiT's de-interleaved RoPE (src/dit.cpp::apply_rope,
// exposed as dit_rope) against a host reference, on the CPU backend and on the build's GPU
// device when one exists.
//
//   1. layout: out[p*half + i] == (p == 0 ? x[2i]*cos_i - x[2i+1]*sin_i
//                                        : x[2i+1]*cos_i + x[2i]*sin_i), within 1e-6 --
//      backend-dependent rounding is allowed, bit-exactness is not claimed (measured 4.8e-7).
//   2. attention invariance: for random q, k the per-(head, query, key) dot products of the
//      rotated tensors equal those of the interleaved reference rotation (the only consumer of
//      the rotated values is q.k, so a layout change must not change it beyond fp32 summation
//      order). This is a LAYOUT proof computed host-side in double precision; it does not run the
//      FlashAttention / exact-SDPA kernels, so it is not a kernel integration test.
//
// Usage: trellis-test-rope-layout [gpu]   (gpu: -1 = CPU only, default = CPU + first GPU device)
#include "dit.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int run_on(ggml_backend_t backend, const char* label) {
    const int hd = 128, nh = 12, L = 37, half = hd / 2;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(-3.14159f, 3.14159f);
    std::vector<float> q((size_t)hd * nh * L), k(q.size()), cs((size_t)half * L), sn(cs.size());
    for (auto& v : q) v = nd(rng);
    for (auto& v : k) v = nd(rng);
    for (size_t i = 0; i < cs.size(); ++i) { float a = ud(rng); cs[i] = std::cos(a); sn[i] = std::sin(a); }

    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* gq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, L);       ggml_set_input(gq);
    ggml_tensor* gk = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, L);       ggml_set_input(gk);
    ggml_tensor* gc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, half, 1, L);   ggml_set_input(gc);
    ggml_tensor* gs = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, half, 1, L);   ggml_set_input(gs);
    ggml_tensor* rq = trellis::dit_rope(ctx, gq, gc, gs);  ggml_set_output(rq);
    ggml_tensor* rk = trellis::dit_rope(ctx, gk, gc, gs);  ggml_set_output(rk);
    ggml_cgraph* g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, rq);
    ggml_build_forward_expand(g, rk);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) { fprintf(stderr, "%s: alloc failed\n", label); return 1; }
    ggml_backend_tensor_set(gq, q.data(), 0, q.size() * 4);
    ggml_backend_tensor_set(gk, k.data(), 0, k.size() * 4);
    ggml_backend_tensor_set(gc, cs.data(), 0, cs.size() * 4);
    ggml_backend_tensor_set(gs, sn.data(), 0, sn.size() * 4);
    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "%s: compute failed\n", label); return 1; }
    std::vector<float> oq(q.size()), ok(k.size());
    ggml_backend_tensor_get(rq, oq.data(), 0, oq.size() * 4);
    ggml_backend_tensor_get(rk, ok.data(), 0, ok.size() * 4);

    // Host references: interleaved (the mathematical definition) and de-interleaved (the layout).
    auto ref = [&](const std::vector<float>& x, std::vector<float>& inter, std::vector<float>& deinter) {
        inter.assign(x.size(), 0.f); deinter.assign(x.size(), 0.f);
        for (int l = 0; l < L; ++l) for (int h = 0; h < nh; ++h) for (int i = 0; i < half; ++i) {
            const size_t base = ((size_t)l * nh + h) * hd;
            const float c = cs[(size_t)l * half + i], s = sn[(size_t)l * half + i];
            const float x0 = x[base + 2 * i], x1 = x[base + 2 * i + 1];
            const float ev = x0 * c - x1 * s, od = x1 * c + x0 * s;
            inter[base + 2 * i] = ev; inter[base + 2 * i + 1] = od;
            deinter[base + i] = ev;   deinter[base + half + i] = od;
        }
    };
    std::vector<float> qi, qd, ki, kd;
    ref(q, qi, qd); ref(k, ki, kd);
    double max_layout = 0;
    for (size_t i = 0; i < oq.size(); ++i) {
        max_layout = std::max(max_layout, (double)std::fabs(oq[i] - qd[i]));
        max_layout = std::max(max_layout, (double)std::fabs(ok[i] - kd[i]));
    }
    // q.k invariance over every (head, query, key) triple.
    double max_dot = 0, max_abs_dot = 0;
    for (int h = 0; h < nh; ++h) for (int a = 0; a < L; ++a) for (int b = 0; b < L; ++b) {
        const size_t qa = ((size_t)a * nh + h) * hd, kb = ((size_t)b * nh + h) * hd;
        double d_new = 0, d_ref = 0;
        for (int i = 0; i < hd; ++i) { d_new += (double)oq[qa + i] * ok[kb + i]; d_ref += (double)qi[qa + i] * ki[kb + i]; }
        max_dot = std::max(max_dot, std::fabs(d_new - d_ref));
        max_abs_dot = std::max(max_abs_dot, std::fabs(d_ref));
    }
    const bool pass = max_layout <= 1e-6 && max_dot <= 1e-4 * std::max(1.0, max_abs_dot);
    printf("%-6s layout max|d|=%.3g  q.k max|d|=%.3g (max|q.k|=%.3g)  %s\n",
           label, max_layout, max_dot, max_abs_dot, pass ? "OK" : "FAIL");
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    const int gpu = argc > 1 ? atoi(argv[1]) : 0;
    int rc = 0;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    rc |= run_on(cpu, "cpu");
    ggml_backend_free(cpu);
    if (gpu >= 0) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (dev) {
            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            if (b) { rc |= run_on(b, ggml_backend_dev_name(dev)); ggml_backend_free(b); }
            else { fprintf(stderr, "gpu init failed\n"); rc = 1; }
        } else {
            printf("no GPU device in this build; CPU only\n");
        }
    }
    printf("ROPE_LAYOUT_RESULT: %s\n", rc == 0 ? "OK" : "FAIL");
    return rc;
}
