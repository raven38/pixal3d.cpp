// Cross-scale neighborhood attention CUDA kernel for NAF -- see include/naf_attn.h
// for the k/v indexing equivalence this relies on (avoids materializing k_up/v_up).
// One thread per query pixel (y,x); each thread loops the 4 heads, computing 81
// logits (q.k/8) over a *contiguous* 9x9 window of the low-res k/v maps, softmax,
// then a weighted sum of the 81 low-res value vectors per output channel. k_pooled
// and v are tiny (<=16 MB at the largest Pixal3D config) and stay resident in L2
// across the whole T*T query grid, so this is bandwidth-friendly despite the naive
// per-thread re-reads. Self-contained: host arrays in/out (mirrors deform_conv.cu).
#include "naf_attn.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace trellis {

// Same neighborhood-window arithmetic as naf_na_window() in naf.cpp (kept in sync;
// this file only *consumes* `start`, not the dilated index, per the equivalence
// argument in naf_attn.h).
__device__ __forceinline__ void na_window_dev(int q, int L, int d, int K,
                                               int& p, int& Lr, int& start) {
    const int r = q % d;
    p = q / d;
    Lr = (L - r + d - 1) / d; // ceil((L-r)/d)
    start = p - K / 2;
    if (start < 0) start = 0;
    const int maxstart = Lr - K;
    if (start > maxstart) start = maxstart;
    if (start < 0) start = 0;
}

__global__ void naf_attn_kernel(const float* __restrict__ q, const float* __restrict__ kp,
                                 const float* __restrict__ v, int T, int h, int w, int Cv,
                                 float scale, float* __restrict__ out) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long TT = (long)T * T;
    if (idx >= TT) return;
    const int y = (int)(idx / T), x = (int)(idx % T);
    const int K = 9, heads = 4, qk_hd = 64;
    const int v_hd = Cv / heads;
    const int dy = T / h, dx = T / w;
    const long HW = (long)h * w;

    int py, Lry, starty, px, Lrx, startx;
    na_window_dev(y, T, dy, K, py, Lry, starty);
    na_window_dev(x, T, dx, K, px, Lrx, startx);

    float qv[qk_hd];
    float logits[K * K];
    float probs[K * K];

    for (int n = 0; n < heads; ++n) {
        for (int c = 0; c < qk_hd; ++c) qv[c] = q[(long)(n * qk_hd + c) * TT + idx];

        float maxlog = -1e30f;
        for (int wy = 0; wy < K; ++wy) {
            const int ky = starty + wy;
            for (int wx = 0; wx < K; ++wx) {
                const int kx = startx + wx;
                const long kpix = (long)ky * w + kx;
                float dot = 0.f;
                for (int c = 0; c < qk_hd; ++c)
                    dot += qv[c] * kp[(long)(n * qk_hd + c) * HW + kpix];
                const float lg = dot * scale;
                logits[wy * K + wx] = lg;
                maxlog = fmaxf(maxlog, lg);
            }
        }
        float sumexp = 0.f;
        for (int i = 0; i < K * K; ++i) { probs[i] = expf(logits[i] - maxlog); sumexp += probs[i]; }
        const float invsum = 1.0f / sumexp;
        for (int i = 0; i < K * K; ++i) probs[i] *= invsum;

        float* op = out + (long)n * v_hd * TT;
        for (int c = 0; c < v_hd; ++c) {
            const float* vc = v + (long)(n * v_hd + c) * HW;
            float acc = 0.f;
            for (int wy = 0; wy < K; ++wy) {
                const int ky = starty + wy;
                for (int wx = 0; wx < K; ++wx)
                    acc += probs[wy * K + wx] * vc[(long)ky * w + (startx + wx)];
            }
            op[(long)c * TT + idx] = acc;
        }
    }
}

void naf_attn_cuda(const float* q, const float* k_pooled, const float* v,
                    int T, int h, int w, int Cv, float scale, float* out, int gpu) {
    const bool log_timing = std::getenv("TRELLIS_DBG_NAF") != nullptr;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms_since = [&](std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(now() - t0).count();
    };
    auto t0 = now();

    cudaSetDevice(gpu < 0 ? 0 : gpu);
    const long TT = (long)T * T, HW = (long)h * w;
    const size_t sq = (size_t)256 * TT * sizeof(float);
    const size_t sk = (size_t)256 * HW * sizeof(float);
    const size_t sv = (size_t)Cv * HW * sizeof(float);
    const size_t so = (size_t)Cv * TT * sizeof(float);

    float *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, sq); cudaMalloc(&dk, sk); cudaMalloc(&dv, sv); cudaMalloc(&dout, so);
    if (log_timing) fprintf(stderr, "[naf_attn]   malloc      @ %8.1f ms\n", ms_since(t0));
    cudaMemcpy(dq, q, sq, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k_pooled, sk, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, sv, cudaMemcpyHostToDevice);
    if (log_timing) fprintf(stderr, "[naf_attn]   H2D copy    @ %8.1f ms\n", ms_since(t0));

    const int threads = 256;
    const int blocks = (int)((TT + threads - 1) / threads);
    naf_attn_kernel<<<blocks, threads>>>(dq, dk, dv, T, h, w, Cv, scale, dout);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) fprintf(stderr, "naf_attn kernel: %s\n", cudaGetErrorString(e));
    if (log_timing) fprintf(stderr, "[naf_attn]   kernel      @ %8.1f ms\n", ms_since(t0));
    cudaMemcpy(out, dout, so, cudaMemcpyDeviceToHost);
    if (log_timing) fprintf(stderr, "[naf_attn]   D2H copy    @ %8.1f ms\n", ms_since(t0));

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
    if (log_timing) fprintf(stderr, "[naf_attn]   free        @ %8.1f ms\n", ms_since(t0));
}

} // namespace trellis
