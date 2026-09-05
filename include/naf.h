// NAF (valeoai/NAF neighborhood-attention feature upsampler), as used by
// TencentARC Pixal3D for the high-resolution conditioning branch (proj = [lr, hr]).
// Ports valeoai/NAF `src/model/naf.py` + `src/layers/{attentions,convolutions,rope}.py`
// (checkpoint naf_release.pth) faithfully for a single batch element (B=1) --
// see docs/spec/30-pixal3d-cond.md section 4.
//
// CPU-only, plain loops, f32: weights are pulled to host once via tensor_to_f32
// and the whole forward pass runs as ordinary C++ arithmetic (no ggml graph).
#pragma once
#include <vector>

namespace trellis {
struct Model;

// Optional per-stage intermediates, so the test can compare stage by stage
// against the PyTorch fixture instead of only the final [C,T,T] map.
// All buffers are channel-major (torch [Cn,H,W] contiguous layout).
struct NafDebug {
    int S_prime = 0;               // image resolution actually fed to the encoder
                                    // (== S unless S > 4T, in which case it is
                                    // bilinear-downsampled to min(S,4T))
    std::vector<float> enc_cat;    // [256, S_prime, S_prime]  concat(e1, e2)
    std::vector<float> enc_pooled; // [256, T, T]              adaptive_avg_pool2d(enc_cat, (T,T))
    std::vector<float> q_rope;     // [256, T, T]              enc_pooled after RoPE (== q)
    std::vector<float> k_pooled;   // [256, h, w]               adaptive_avg_pool2d(q_rope, (h,w))
    std::vector<float> k_up;       // [256, T, T]              nearest-exact resize(k_pooled, (T,T))
    std::vector<float> v_up;       // [1024, T, T]             nearest-exact resize(lr_features, (T,T))
};

// Full NAF forward: image [3,S,S] in [0,1] (NOT ImageNet-normalized), lr_features
// [C,h,w] (DINOv3 patch map, h=w=S/16, C=1024), target resolution T.
// Returns the upsampled feature map [C,T,T] channel-major (C == the lr channel
// count, 1024 for Pixal3D).
// `naf` is the loaded GGUF whose tensor names are the PyTorch NAF module's keys
// (image_encoder.{encoder,sem_encoder}.*, image_encoder.rope.periods).
std::vector<float> naf_upsample(const Model& naf, const float* image, int S,
                                 const float* lr, int C, int h, int w, int T,
                                 NafDebug* dbg = nullptr);

// ---------------------------------------------------------------------------
// Building blocks, exposed only so src/test_naf.cpp can unit-check them in
// isolation (--selftest, hand-computable cases) before/independent of the
// PyTorch fixture. Not meant to be used outside naf.{h,cpp}/test_naf.cpp.
// ---------------------------------------------------------------------------

// GroupNorm(num_groups, eps, affine): x [C,H,W] channel-major, gamma/beta [C].
std::vector<float> naf_group_norm(const std::vector<float>& x, int C, int H, int W,
                                   int num_groups, const float* gamma, const float* beta, float eps);

// SiLU(x) = x * sigmoid(x), in place.
void naf_silu_inplace(std::vector<float>& x);

// Conv2d, reflect-padded (pad = K/2; a no-op when K==1): x [Ci,H,W] -> [Co,H,W].
// w is torch-order [Co,Ci,K,K] flattened, b is [Co].
std::vector<float> naf_conv2d(const std::vector<float>& x, int Ci, int H, int W,
                               const float* w, const float* b, int Co, int K);

// adaptive_avg_pool2d(x, (Hout,Wout)): bin i -> [floor(i*In/Out), ceil((i+1)*In/Out)).
std::vector<float> naf_adaptive_avg_pool2d(const std::vector<float>& x, int C,
                                            int Hin, int Win, int Hout, int Wout);

// PyTorch 'nearest-exact' resize: dst -> src = floor((dst+0.5)*In/Out).
std::vector<float> naf_nearest_exact_resize(const std::vector<float>& x, int C,
                                             int Hin, int Win, int Hout, int Wout);

// Bilinear resize, align_corners=False (used only when S > 4T on the way in).
std::vector<float> naf_bilinear_resize(const std::vector<float>& x, int C,
                                        int Hin, int Win, int Hout, int Wout);

// RoPE (4 heads x 64, periods[16] from image_encoder.rope.periods), applied in
// place to x [256,T,T] viewed as 4 heads of 64 contiguous channels.
void naf_rope_apply_inplace(std::vector<float>& x, int T, const std::vector<float>& periods);

// NATTEN neighborhood-window arithmetic for one axis, one query index q in
// [0,L): r = q % d, p = q / d, Lr = ceil((L-r)/d), start = clamp(p-K/2, 0, Lr-K).
// The K real neighbor indices along this axis are (start+i)*d + r, i in [0,K).
void naf_na_window(int q, int L, int d, int K, int& r, int& p, int& Lr, int& start);

// Cross-scale neighborhood attention: q,k [256,T,T] (4x64), v [Cv,T,T] (4x(Cv/4)),
// kernel 9x9, dilation (dy,dx), stride 1. out: [Cv,T,T].
void naf_na2d(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
              int T, int dy, int dx, int Cv, float scale, std::vector<float>& out);

// ---------------------------------------------------------------------------
// GPU dispatch (src/naf_gpu.cpp + src/naf_attn.cu), only compiled/linked into
// CUDA builds (TRELLIS_USE_CUDA -- see CMakeLists.txt). naf_upsample() in
// naf.cpp calls these automatically (see its dispatch at the top); not meant to
// be called directly by other code. On non-CUDA backends (Metal, Vulkan,
// CPU-only) these symbols don't exist and naf_upsample() never references them,
// so naf_upsample()'s CPU body (below) is the only code that ever runs there --
// unchanged from before this GPU port.
bool naf_gpu_available(const Model& naf, int S, int T, int h, int w);
std::vector<float> naf_upsample_gpu(const Model& naf, const float* image, int S,
                                     const float* lr, int C, int h, int w, int T,
                                     NafDebug* dbg);

} // namespace trellis
