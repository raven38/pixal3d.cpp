// ggml WebGPU backend op regression: large-dispatch shapes (patches/ggml-webgpu/0003) and the
// batched mul_mat shapes the Pixal3D NAF attention graph uses (src/naf_gpu.cpp). Every case runs
// one small graph on the WebGPU device and on ggml-cpu with identical inputs and compares.
//
// The upstream encoders for soft_max / sum_rows / norm / get_rows / concat / pad / repeat
// dispatched one workgroup per row (or per WG_SIZE elements) on the x axis only, so any op with
// more than maxComputeWorkgroupsPerDimension (65535) rows failed WebGPU validation and the node
// was silently dropped. Patch 0003 spreads those dispatches over a 2D grid; these shapes sit
// well past the old limit (up to 4M rows / 67M elements).
//
//   ./build-webgpu/trellis-webgpu-ops [--small]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <chrono>
#include <thread>
#include <string>
#include <vector>

using T = ggml_tensor;

namespace {

float detval(size_t i) { return ((i * 2654435761u) & 1023) / 512.0f - 1.0f; }

ggml_context* mkctx(size_t n_tensors) {
    size_t meta = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(512, false) + (1 << 16);
    return ggml_init({ meta, nullptr, true });
}

struct Diff { double max_abs = 0, mean_abs = 0, rel = 0, cos = 0; };

Diff compare(const std::vector<float>& a, const std::vector<float>& ref) {
    Diff d;
    double sum_abs = 0, refmax = 0, dot = 0, na = 0, nb = 0;
    const size_t n = std::min(a.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        const double diff = std::fabs((double)a[i] - (double)ref[i]);
        d.max_abs = std::max(d.max_abs, diff);
        sum_abs += diff;
        refmax = std::max(refmax, std::fabs((double)ref[i]));
        dot += (double)a[i] * ref[i]; na += (double)a[i] * a[i]; nb += (double)ref[i] * ref[i];
    }
    d.mean_abs = n ? sum_abs / n : 0;
    d.rel = refmax > 0 ? d.max_abs / refmax : d.max_abs;
    d.cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0;
    return d;
}

// One test case: `build` creates the input tensors (flagged ggml_set_input) and returns the
// output node; `fill` uploads the inputs after allocation.
struct Case {
    const char* name;
    double tol;
    std::function<T*(ggml_context*, std::vector<T*>&)> build;
};

static int g_gpu_repeat = 0;   // --gpu-repeat K: re-run the GPU graph K more times on the same inputs, compare to the first run

bool run_case(const Case& cs, ggml_backend_t gpu, ggml_backend_t cpu) {
    std::vector<float> res[2];
    size_t out_bytes = 0;
    for (int which = 0; which < 2; ++which) {
        ggml_backend_t be = which == 0 ? gpu : cpu;
        ggml_context* c = mkctx(512);
        std::vector<T*> inputs;
        T* out = cs.build(c, inputs);
        ggml_set_output(out);
        ggml_cgraph* g = ggml_new_graph_custom(c, 512, false);
        ggml_build_forward_expand(g, out);
        if (which == 0) {
            ggml_backend_dev_t dev = ggml_backend_get_device(be);
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                T* n = ggml_graph_node(g, i);
                if (!ggml_backend_dev_supports_op(dev, n)) {
                    printf("  %-28s node %d %s unsupported on %s -> FAIL\n", cs.name, i, ggml_op_name(n->op), ggml_backend_name(be));
                    ggml_free(c);
                    return false;
                }
            }
        }
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(alloc, g)) {
            printf("  %-28s alloc failed on %s -> FAIL\n", cs.name, ggml_backend_name(be));
            ggml_free(c);
            return false;
        }
        size_t seed = 1;
        std::vector<std::vector<uint8_t>> host_inputs;   // kept for re-upload (gallocr reuses input memory)
        for (T* in : inputs) {
            const size_t ne = (size_t)ggml_nelements(in);
            std::vector<uint8_t> raw(ggml_nbytes(in));
            if (in->type == GGML_TYPE_F32) {
                float* h = (float*)raw.data();
                // not f16-representable (detval alone is k/512), so f16 input staging shows up
                for (size_t i = 0; i < ne; ++i) h[i] = detval(seed + i) * (1.0f + 0.37f * detval(seed * 3 + i));
            } else if (in->type == GGML_TYPE_F16) {
                ggml_fp16_t* h = (ggml_fp16_t*)raw.data();
                for (size_t i = 0; i < ne; ++i) h[i] = ggml_fp32_to_fp16(detval(seed + i) * 0.05f);
            } else if (in->type == GGML_TYPE_I32) {
                // index inputs: the row count to index is stored in op_params[0] by the builder
                const int32_t nrows = in->op_params[0];
                int32_t* h = (int32_t*)raw.data();
                if (in->op_params[1] == 1) { for (size_t i = 0; i < ne / 2; ++i) { h[i] = (int32_t)(2 * i); h[ne / 2 + i] = (int32_t)(2 * i + 1); } }
                else for (size_t i = 0; i < ne; ++i) h[i] = (int32_t)((i * 7919u + seed) % (size_t)nrows);
            }
            ggml_backend_tensor_set(in, raw.data(), 0, ggml_nbytes(in));
            host_inputs.push_back(std::move(raw));
            seed += ne;
        }
        if (ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) {
            printf("  %-28s compute failed on %s -> FAIL\n", cs.name, ggml_backend_name(be));
            ggml_gallocr_free(alloc); ggml_free(c);
            return false;
        }
        ggml_backend_synchronize(be);
        res[which].resize((size_t)ggml_nelements(out));
        ggml_backend_tensor_get(out, res[which].data(), 0, ggml_nbytes(out));
        out_bytes = ggml_nbytes(out);
        if (which == 0 && g_gpu_repeat > 0) {
            // determinism: the same graph on the same (still-resident) inputs must be bit-identical
            std::vector<float> again(res[0].size());
            int nbad = 0;
            for (int r = 0; r < g_gpu_repeat; ++r) {
                for (size_t k = 0; k < inputs.size(); ++k)   // inputs are freed/reused by gallocr after their last use
                    ggml_backend_tensor_set(inputs[k], host_inputs[k].data(), 0, ggml_nbytes(inputs[k]));
                if (ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) { printf("  gpu-repeat %d: compute failed\n", r); break; }
                ggml_backend_synchronize(be);
                ggml_backend_tensor_get(out, again.data(), 0, ggml_nbytes(out));
                size_t ndiff = 0; double mx = 0;
                for (size_t i = 0; i < again.size(); ++i) { const double d = std::fabs((double)again[i] - res[0][i]); if (!(d == 0)) { ++ndiff; if (std::isfinite(d)) mx = std::max(mx, d); else mx = INFINITY; } }
                if (ndiff) { ++nbad; printf("  %-28s gpu-repeat %3d: %zu of %zu elements differ from run 0 (max|d| %.3e)\n", cs.name, r + 1, ndiff, again.size(), mx); }
            }
            printf("  %-28s gpu-repeat: %d of %d re-runs differed from run 0\n", cs.name, nbad, g_gpu_repeat);
            if (nbad) { ggml_gallocr_free(alloc); ggml_free(c); return false; }
        }
        ggml_gallocr_free(alloc);
        ggml_free(c);
    }
    size_t bad = 0;
    for (float v : res[0]) if (!std::isfinite(v)) ++bad;
    Diff d = compare(res[0], res[1]);
    const bool ok = bad == 0 && d.rel < cs.tol;
    if (!ok) {
        size_t first = res[0].size(), nbad = 0;
        for (size_t i = 0; i < res[0].size(); ++i)
            if (std::fabs(res[0][i] - res[1][i]) > 1e-6) { if (first == res[0].size()) first = i; ++nbad; }
        printf("    first mismatch at %zu of %zu, %zu mismatching (%.1f%%)\n", first, res[0].size(), nbad, 100.0 * nbad / res[0].size());
    }
    printf("  %-28s out %7.1f MB  max|d|=%.3e mean|d|=%.3e rel=%.3e cos=%.7f nonfinite=%zu (tol %.0e) -> %s\n",
           cs.name, out_bytes / 1048576.0, d.max_abs, d.mean_abs, d.rel, d.cos, bad, cs.tol, ok ? "PASS" : "FAIL");
    return ok;
}

T* in_f32(ggml_context* c, std::vector<T*>& ins, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    T* t = ggml_new_tensor_4d(c, GGML_TYPE_F32, n0, n1, n2, n3);
    ggml_set_input(t); ins.push_back(t);
    return t;
}
T* in_idx(ggml_context* c, std::vector<T*>& ins, int64_t n, int32_t nrows) {
    T* t = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    t->op_params[0] = nrows;   // consumed by run_case's filler
    ggml_set_input(t); ins.push_back(t);
    return t;
}

} // namespace

int main(int argc, char** argv) {
    bool small = false; int repeat = 1; std::string only; double hold_gb = 0; int hold_s = 600;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--small") small = true;
        else if (std::string(argv[i]) == "--repeat" && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (std::string(argv[i]) == "--only" && i + 1 < argc) only = argv[++i];   // substring filter on case names
        else if (std::string(argv[i]) == "--gpu-repeat" && i + 1 < argc) g_gpu_repeat = atoi(argv[++i]);
        else if (std::string(argv[i]) == "--hold-gb" && i + 1 < argc) hold_gb = atof(argv[++i]);      // memory-pressure hog mode
        else if (std::string(argv[i]) == "--hold-seconds" && i + 1 < argc) hold_s = atoi(argv[++i]);
    }

    ggml_backend_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "WebGPU", 6) == 0) { gpu = ggml_backend_dev_init(d, nullptr); break; }
    }
    if (!gpu) { fprintf(stderr, "no WebGPU device\n"); return 1; }
    if (hold_gb > 0) {
        // Memory-pressure hog: hold hold_gb GiB of device memory (one 1 GiB buffer each, touched), then
        // sleep, so another process's WebGPU run can be observed under GPU memory pressure.
        const int n = (int)std::ceil(hold_gb);
        std::vector<ggml_context*> cs; std::vector<ggml_backend_buffer_t> bufs; std::vector<T*> ts;
        std::vector<float> fill((size_t)1 << 28, 1.0f);
        for (int i = 0; i < n; ++i) {
            ggml_context* c = mkctx(4);
            T* t = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t)1 << 28);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
            if (!buf) { fprintf(stderr, "hog: alloc of GiB #%d failed\n", i); return 1; }
            ggml_backend_tensor_set(t, fill.data(), 0, ggml_nbytes(t));
            cs.push_back(c); bufs.push_back(buf); ts.push_back(t);
        }
        ggml_backend_synchronize(gpu);
        printf("hog: holding %d GiB on %s for %d s\n", n, ggml_backend_name(gpu), hold_s);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(hold_s));
        std::vector<float> back(1 << 20);
        for (T* t : ts) ggml_backend_tensor_get(t, back.data(), 0, back.size() * 4);   // keep it live to the end
        for (size_t i = 0; i < bufs.size(); ++i) { ggml_backend_buffer_free(bufs[i]); ggml_free(cs[i]); }
        ggml_backend_free(gpu);
        return 0;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    printf("=== trellis-webgpu-ops (%s vs %s)%s ===\n", ggml_backend_name(gpu), ggml_backend_name(cpu), small ? " [--small]" : "");

    // NAF attention shapes at T=512, h=w=32: 1024 blocks of 256 pixels, 4 heads, 81 taps.
    const int64_t BLK = small ? 64 : 1024, PX = 256, HD = 4, TAP = 81;
    const int64_t T2 = small ? 262144 / 4 : 262144;   // T*T
    std::vector<Case> cases = {
        { "soft_max 1M rows", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_soft_max_ext(c, in_f32(c, ins, TAP, PX, HD, BLK), nullptr, 0.125f, 0.0f); } },
        { "sum_rows 4M rows", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_sum_rows(c, in_f32(c, ins, 16, T2 * 16)); } },
        { "norm 8 rows x 4M", 1e-4, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_norm(c, in_f32(c, ins, T2 * 16, 8), 1e-5f); } },
        { "norm 1M rows x 81", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_norm(c, in_f32(c, ins, TAP, PX, HD, BLK), 1e-5f); } },
        { "get_rows 82944 x 1024", 0.0 + 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* src = in_f32(c, ins, 1024, 1024);
              return ggml_get_rows(c, src, in_idx(c, ins, TAP * BLK, 1024)); } },
        { "get_rows 262144 x 256", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* src = in_f32(c, ins, 256, T2);
              return ggml_get_rows(c, src, in_idx(c, ins, T2, (int32_t)T2)); } },
        { "concat 4M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_concat(c, in_f32(c, ins, T2 / 16, 32, 4), in_f32(c, ins, T2 / 16, 32, 4), 1); } },
        { "concat 16M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_concat(c, in_f32(c, ins, T2 / 4, 32, 4), in_f32(c, ins, T2 / 4, 32, 4), 1); } },
        { "concat 67M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_concat(c, in_f32(c, ins, T2, 32, 4), in_f32(c, ins, T2, 32, 4), 1); } },
        { "pad 33M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_pad(c, in_f32(c, ins, 512, 512, small ? 32 : 128), 1, 1, 0, 0); } },
        { "repeat 16M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* src = in_f32(c, ins, 1, 4096);
              return ggml_repeat(c, src, ggml_new_tensor_2d(c, GGML_TYPE_F32, small ? 1024 : 4096, 4096)); } },
        { "mul_mat [64,81]x[64,256] b4096", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_mul_mat(c, in_f32(c, ins, 64, TAP, HD, BLK), in_f32(c, ins, 64, PX, HD, BLK)); } },
        { "mul_mat [81,256]x[81,256] b4096", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_mul_mat(c, in_f32(c, ins, TAP, 256, HD, BLK), in_f32(c, ins, TAP, PX, HD, BLK)); } },
        { "cont(permute) 64 MB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 256, PX, HD, BLK / 16);
              return ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); } },
        { "cont(permute) 256 MB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 256, PX, HD, BLK / 4);
              return ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); } },
        { "cont(permute) 512 MB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 256, PX, HD, BLK / 2);
              return ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); } },
        // Shape-1024 thresholds: DINOv3 @S=1024 softmax (65616 rows, just past the old 65535 limit),
        // NAF T=512/h=64 (4096 blocks of 8x8: 331776-row gathers, [.,.,4,4096] batched GEMMs).
        { "soft_max [4101,4101,16] (dino S1024)", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_soft_max_ext(c, in_f32(c, ins, 4101, small ? 1025 : 4101, 16), nullptr, 0.125f, 0.0f); } },
        { "get_rows 331776 x 1024 (naf h64)", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* src = in_f32(c, ins, 1024, 4096);
              return ggml_get_rows(c, src, in_idx(c, ins, TAP * 4 * BLK, 4096)); } },
        { "mul_mat [64,81]x[64,64] b16384", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_mul_mat(c, in_f32(c, ins, 64, TAP, HD, 4 * BLK), in_f32(c, ins, 64, 64, HD, 4 * BLK)); } },
        { "mul_mat [81,256]x[81,64] b16384", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_mul_mat(c, in_f32(c, ins, TAP, 256, HD, 4 * BLK), in_f32(c, ins, TAP, 64, HD, 4 * BLK)); } },
        { "soft_max [81,64,4,16384] (naf h64)", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_soft_max_ext(c, in_f32(c, ins, TAP, 64, HD, 4 * BLK), nullptr, 0.125f, 0.0f); } },
        // The DiT's exact SDPA at the Shape-1024 token count (dit.cpp sdpa, --no-fa): K/V shared by
        // all query chunks, 14 chunks of <=1279 queries (1 GiB score budget), CONCAT of the chunk
        // outputs along ne1 -- the first multi-chunk attention Pixal3D runs on WebGPU.
        { "sdpa chunked N=17489 (dit exact)", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              const int64_t hd = 128, nh = 12, L = small ? 4377 : 17489;
              T* q = in_f32(c, ins, hd, nh, L); T* k = in_f32(c, ins, hd, nh, L); T* v = in_f32(c, ins, hd, nh, L);
              T* q2 = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
              T* k2 = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
              T* v2 = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));
              const int64_t per_q = L * nh * 4, nq0 = std::max<int64_t>(1, (1024ll << 20) / per_q);
              const int64_t nq = nq0 >= L ? L : nq0;
              T* out = nullptr;
              for (int64_t q0 = 0; q0 < L; q0 += nq) {
                  const int64_t n = std::min(nq, L - q0);
                  T* qc = (n == L) ? q2 : ggml_cont(c, ggml_view_3d(c, q2, hd, n, nh, q2->nb[1], q2->nb[2], (size_t)q0 * q2->nb[1]));
                  T* kq = ggml_soft_max_ext(c, ggml_mul_mat(c, k2, qc), nullptr, 1.0f / std::sqrt((float)hd), 0.0f);
                  T* kqv = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, v2, kq), 0, 2, 1, 3));
                  T* o = ggml_reshape_2d(c, kqv, hd * nh, n);
                  out = out ? ggml_concat(c, out, o, 1) : o;
              }
              return out; } },
        // The DiT's other N=17489 pieces (block MLP with the 546 MB hidden, the RoPE even/odd
        // scatter, LayerNorm + modulation over [1536, N]), for repeated-run flakiness checks.
        { "dit mlp N=17489 (f16 w, gelu)", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              const int64_t L = small ? 4377 : 17489;
              T* h = in_f32(c, ins, 1536, L);
              T* w1 = ggml_new_tensor_2d(c, GGML_TYPE_F16, 1536, 8192); ggml_set_input(w1); ins.push_back(w1);
              T* w2 = ggml_new_tensor_2d(c, GGML_TYPE_F16, 8192, 1536); ggml_set_input(w2); ins.push_back(w2);
              return ggml_mul_mat(c, w2, ggml_gelu(c, ggml_mul_mat(c, w1, h))); } },
        { "mul_mat f16[1536,8192] x f32[1536,17489]", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              const int64_t L = small ? 4377 : 17489;
              T* h = in_f32(c, ins, 1536, L);
              T* w1 = ggml_new_tensor_2d(c, GGML_TYPE_F16, 1536, 8192); ggml_set_input(w1); ins.push_back(w1);
              return ggml_mul_mat(c, w1, h); } },
        { "mul_mat f16[8192,1536] x f32[8192,17489]", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              const int64_t L = small ? 4377 : 17489;
              T* h = in_f32(c, ins, 8192, L);
              T* w2 = ggml_new_tensor_2d(c, GGML_TYPE_F16, 8192, 1536); ggml_set_input(w2); ins.push_back(w2);
              return ggml_mul_mat(c, w2, h); } },
        { "mul_mat f32[1536,8192] x f32[1536,17489]", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              const int64_t L = small ? 4377 : 17489;
              T* h = in_f32(c, ins, 1536, L);
              T* w1 = in_f32(c, ins, 1536, 8192);
              return ggml_mul_mat(c, w1, h); } },
        // q/k MultiHeadRMSNorm x gamma at N=17489: ggml-webgpu fuses RMS_NORM+MUL into rms_norm_mul,
        // whose encoder dispatches one workgroup per row on x only (12 x 17489 = 209868 rows).
        { "rms_norm x gamma [128,12,17489]", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              const int64_t L = small ? 4377 : 17489;
              T* x = in_f32(c, ins, 128, 12, L);
              T* gamma = in_f32(c, ins, 128);
              return ggml_mul(c, ggml_rms_norm(c, x, 1e-12f), gamma); } },
        { "rms_norm alone [128,12,17489]", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              const int64_t L = small ? 4377 : 17489;
              return ggml_rms_norm(c, in_f32(c, ins, 128, 12, L), 1e-12f); } },
        { "dit rope scatter N=17489", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              const int64_t L = small ? 4377 : 17489, hd = 128, nh = 12, half = 64;
              T* x = in_f32(c, ins, hd, nh, L);
              T* cs = in_f32(c, ins, 1, half, 1, L); T* sn = in_f32(c, ins, 1, half, 1, L);
              T* idx = in_idx(c, ins, hd, 1);   // filled below as evens|odds
              idx->op_params[1] = 1;            // marker: rope index fill
              T* x5 = ggml_reshape_4d(c, x, 2, half, nh, L);
              T* x0 = ggml_cont(c, ggml_view_4d(c, x5, 1, half, nh, L, x5->nb[1], x5->nb[2], x5->nb[3], 0));
              T* x1 = ggml_cont(c, ggml_view_4d(c, x5, 1, half, nh, L, x5->nb[1], x5->nb[2], x5->nb[3], x5->nb[0]));
              T* ev = ggml_sub(c, ggml_mul(c, x0, cs), ggml_mul(c, x1, sn));
              T* od = ggml_add(c, ggml_mul(c, x1, cs), ggml_mul(c, x0, sn));
              T* ce = ggml_view_1d(c, idx, half, 0);
              T* co = ggml_view_1d(c, idx, half, (size_t)half * 4);
              T* out = ggml_scale(c, ggml_reshape_4d(c, x, 1, hd, nh, L), 0.0f);
              out = ggml_set_rows(c, out, ev, ce);
              out = ggml_set_rows(c, out, od, co);
              return ggml_reshape_3d(c, out, hd, nh, L); } },
        { "cont(permute) 1 GB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 256, PX, HD, BLK);
              return ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); } },
    };
    bool all = true;
    for (int r = 0; r < repeat; ++r) {
        if (repeat > 1) printf("--- repeat %d/%d ---\n", r + 1, repeat);
        for (const Case& cs : cases) if (only.empty() || std::string(cs.name).find(only) != std::string::npos) all &= run_case(cs, gpu, cpu);
    }
    printf("=== %s ===\n", all ? "ALL PASS" : "SOME FAILED");
    ggml_backend_free(gpu); ggml_backend_free(cpu);
    return all ? 0 : 1;
}
