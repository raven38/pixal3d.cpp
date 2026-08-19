// Exercise trellis's V-range protection around CUDA FlashAttention's internal
// BF16-to-F16 conversion. Q and K are zero, so uniform attention over a
// constant V must reproduce that finite V.
//
// Usage: trellis-test-fa-bf16-range [gpu=0] [value=70000] [raw]
// `raw` omits the protection and is expected to fail above F16_MAX.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    constexpr int64_t head_dim = 64;
    constexpr int64_t queries = 64;
    constexpr int64_t keys = 256;
    const int gpu = argc > 1 ? std::atoi(argv[1]) : 0;
    const float requested = argc > 2 ? std::strtof(argv[2], nullptr) : 70000.0f;
    const bool raw = argc > 3 && std::string(argv[3]) == "raw";
    constexpr float V_SCALE = 1.0f / 256.0f;
    const float graph_scale = raw ? 1.0f : V_SCALE;
    const float output_scale = 1.0f / graph_scale;
    const float expected = ggml_bf16_to_fp32(ggml_fp32_to_bf16(requested * graph_scale)) * output_scale;

    ggml_backend_t backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        std::fprintf(stderr, "failed to initialize CUDA device %d\n", gpu);
        return 1;
    }

    const size_t meta = ggml_tensor_overhead() * 16
        + ggml_graph_overhead_custom(64, false) + (1 << 20);
    ggml_context* ctx = ggml_init({meta, nullptr, true});
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  head_dim, queries, 1, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, head_dim, keys,    1, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  head_dim, keys,    1, 1);
    ggml_tensor* vf = ggml_cast(ctx, ggml_scale(ctx, v, graph_scale), GGML_TYPE_BF16);
    ggml_tensor* out = ggml_flash_attn_ext(
        ctx, q, k, vf, nullptr, 1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    out = ggml_scale(ctx, out, output_scale);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_tensor_memset(q, 0, 0, ggml_nbytes(q));
    ggml_backend_tensor_memset(k, 0, 0, ggml_nbytes(k));
    std::vector<float> values((size_t) head_dim * (size_t) keys, requested);
    ggml_backend_tensor_set(v, values.data(), 0, values.size() * sizeof(values[0]));

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "CUDA graph compute failed: status=%d\n", (int) status);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> output((size_t) head_dim * (size_t) queries);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(output[0]));
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    size_t nonfinite = 0;
    for (float x : output) {
        if (!std::isfinite(x)) {
            ++nonfinite;
            continue;
        }
        min_value = std::min(min_value, x);
        max_value = std::max(max_value, x);
        sum += x;
    }
    const double mean = nonfinite == output.size()
        ? std::numeric_limits<double>::quiet_NaN()
        : sum / double(output.size() - nonfinite);
    const bool ok = nonfinite == 0 && std::fabs(mean - expected) <= std::fabs(expected) * 0.002;
    std::printf("mode=%s requested=%.1f expected=%.1f output mean=%.7g min=%.7g max=%.7g nonfinite=%zu %s\n",
        raw ? "raw" : "scaled", requested, expected, mean, min_value, max_value, nonfinite, ok ? "PASS" : "FAIL");

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
