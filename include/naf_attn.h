// Cross-scale neighborhood attention for NAF (docs/spec/30-pixal3d-cond.md section 4,
// reference: naf_na2d in naf.{h,cpp}) -- the "heavy part" of naf_upsample, split out
// as a custom CUDA kernel so it can index the un-upsampled low-res k/v maps directly
// instead of materializing k_up/v_up ([256,T,T] / [C,T,T], up to 4 GB f32 at T=1024).
//
// Equivalence with the CPU reference (naf_na2d applied to nearest-exact-upsampled
// k_up/v_up): nearest_exact_resize upsamples src[[h or w]] to dst[T] via
// src_idx = floor((dst_idx+0.5)*h/T). Pixal3D always uses an *exact* integer factor
// d = T/h (T in {128,512,1024}, h in {32,64}), and naf_na_window's neighbor indices
// are nyi = (start+i)*d + r with r in [0,d) -- so
//   floor((nyi+0.5)/d) = floor(start+i + (r+0.5)/d) = start+i   (since 0 < (r+0.5)/d < 1)
// exactly, with no rounding ambiguity. So "look up k_up/v_up at nyi" (T-space) is
// bit-identical to "look up k_pooled/v(lr) at start+i" (h/w-space): the CUDA kernel
// below does the latter, over a plain contiguous 9x9 window in the low-res grid (no
// dilation arithmetic needed once na_window has produced `start`).
#pragma once

namespace trellis {

// q:        [256,T,T]  channel-major (4 heads x 64), RoPE'd encoder features.
// k_pooled: [256,h,w]  channel-major, adaptive_avg_pool2d(q, (h,w)) (NOT upsampled).
// v:        [Cv,h,w]   channel-major, the lr DINOv3 feature map (NOT upsampled).
// out:      [Cv,T,T]   channel-major.
// scale = head_dim^-0.5 = 1/8 for the 64-wide q/k heads. gpu = CUDA device index.
void naf_attn_cuda(const float* q, const float* k_pooled, const float* v,
                    int T, int h, int w, int Cv, float scale, float* out, int gpu);

} // namespace trellis
