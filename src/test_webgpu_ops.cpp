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
#include <string>
#include <vector>

using T = ggml_tensor;

namespace {

float detval(size_t i) { return ((i * 2654435761u) & 1023) / 512.0f - 1.0f; }

ggml_context* mkctx(size_t n_tensors) {
    size_t meta = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(64, false) + (1 << 16);
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

bool run_case(const Case& cs, ggml_backend_t gpu, ggml_backend_t cpu) {
    std::vector<float> res[2];
    size_t out_bytes = 0;
    for (int which = 0; which < 2; ++which) {
        ggml_backend_t be = which == 0 ? gpu : cpu;
        ggml_context* c = mkctx(64);
        std::vector<T*> inputs;
        T* out = cs.build(c, inputs);
        ggml_set_output(out);
        ggml_cgraph* g = ggml_new_graph_custom(c, 64, false);
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
        for (T* in : inputs) {
            const size_t ne = (size_t)ggml_nelements(in);
            if (in->type == GGML_TYPE_F32) {
                std::vector<float> h(ne);
                // not f16-representable (detval alone is k/512), so f16 input staging shows up
                for (size_t i = 0; i < ne; ++i) h[i] = detval(seed + i) * (1.0f + 0.37f * detval(seed * 3 + i));
                ggml_backend_tensor_set(in, h.data(), 0, ggml_nbytes(in));
            } else if (in->type == GGML_TYPE_I32) {
                // index inputs: the row count to index is stored in op_params[0] by the builder
                const int32_t nrows = in->op_params[0];
                std::vector<int32_t> h(ne);
                for (size_t i = 0; i < ne; ++i) h[i] = (int32_t)((i * 7919u + seed) % (size_t)nrows);
                ggml_backend_tensor_set(in, h.data(), 0, ggml_nbytes(in));
            }
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
    bool small = false;
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--small") small = true;

    ggml_backend_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "WebGPU", 6) == 0) { gpu = ggml_backend_dev_init(d, nullptr); break; }
    }
    if (!gpu) { fprintf(stderr, "no WebGPU device\n"); return 1; }
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
        { "cont(permute) 1 GB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 256, PX, HD, BLK);
              return ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); } },
    };
    bool all = true;
    for (const Case& cs : cases) all &= run_case(cs, gpu, cpu);
    printf("=== %s ===\n", all ? "ALL PASS" : "SOME FAILED");
    ggml_backend_free(gpu); ggml_backend_free(cpu);
    return all ? 0 : 1;
}
