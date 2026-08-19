// CUDA-only regression for dense FA masks just above 65536x65536, where the
// final query-tile offset exceeds INT_MAX when measured in half2s.
// The old CUDA helper computed that offset in int and read before the mask;
// fattn-common.cuh must keep the mask strides and pointer arithmetic in int64_t.
//
// Usage: trellis-test-fa-mask-overflow [gpu=0] [tokens=65792]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

int main(int argc, char** argv) {
    constexpr int64_t head_dim = 64;
    const int gpu = argc > 1 ? std::atoi(argv[1]) : 0;
    const int64_t tokens = argc > 2 ? std::atoll(argv[2]) : 65792;

    if (tokens <= 0 || tokens % 256 != 0) {
        std::fprintf(stderr, "tokens must be a positive multiple of 256\n");
        return 2;
    }

    // For this pinned CUDA MMA specialization, a large query uses 64 columns
    // per tile. s31 is in half2 elements, hence tokens/2 rather than tokens.
    const uint64_t final_tile_offset = uint64_t(tokens - 64) * uint64_t(tokens / 2);
    if (final_tile_offset <= uint64_t(std::numeric_limits<int>::max())) {
        std::fprintf(stderr,
            "tokens=%lld does not cross the 32-bit mask-offset boundary (%llu <= INT_MAX)\n",
            (long long) tokens, (unsigned long long) final_tile_offset);
        return 2;
    }

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

    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  head_dim, tokens, 1, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, head_dim, tokens, 1, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, head_dim, tokens, 1, 1);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, tokens, tokens);
    ggml_tensor* out = ggml_flash_attn_ext(
        ctx, q, k, v, mask, 1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const double mask_gib = double(ggml_nbytes(mask)) / double(1ull << 30);
    std::printf("device=%d tokens=%lld mask=%.3f GiB final-half2-offset=%llu\n",
        gpu, (long long) tokens, mask_gib, (unsigned long long) final_tile_offset);
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "graph allocation failed (the default case needs about 8.1 GiB VRAM)\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_tensor_memset(q, 0, 0, ggml_nbytes(q));
    ggml_backend_tensor_memset(k, 0, 0, ggml_nbytes(k));
    ggml_backend_tensor_memset(mask, 0, 0, ggml_nbytes(mask));

    // Q and K are zero, so attention is uniform. A key-index ramp in V makes
    // every output element converge to 0.5; a bad KV_max is therefore visible
    // even if the out-of-bounds helper read happens not to fault.
    std::vector<ggml_bf16_t> values((size_t) head_dim * (size_t) tokens);
    for (int64_t t = 0; t < tokens; ++t) {
        const ggml_bf16_t value = ggml_fp32_to_bf16(float(t) / float(tokens - 1));
        std::fill_n(values.data() + (size_t) t * head_dim, (size_t) head_dim, value);
    }
    ggml_backend_tensor_set(v, values.data(), 0, values.size() * sizeof(values[0]));

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "CUDA graph compute failed: status=%d\n", (int) status);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> output((size_t) head_dim * (size_t) tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(output[0]));
    double sum = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    size_t nonfinite = 0;
    for (float x : output) {
        if (!std::isfinite(x)) {
            ++nonfinite;
            continue;
        }
        sum += x;
        min_value = std::min(min_value, x);
        max_value = std::max(max_value, x);
    }
    const double mean = nonfinite == output.size()
        ? std::numeric_limits<double>::quiet_NaN()
        : sum / double(output.size() - nonfinite);
    const bool ok = nonfinite == 0 && std::fabs(mean - 0.5) < 0.01
        && std::fabs(min_value - 0.5f) < 0.02f && std::fabs(max_value - 0.5f) < 0.02f;
    std::printf("output mean=%.7f min=%.7f max=%.7f nonfinite=%zu %s\n",
        mean, min_value, max_value, nonfinite, ok ? "PASS" : "FAIL");

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
