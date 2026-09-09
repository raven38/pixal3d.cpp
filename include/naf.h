// NAF (valeoai/NAF neighborhood-attention feature upsampler), as used by
// TencentARC Pixal3D for the high-resolution conditioning branch (proj = [lr, hr]).
// Ports valeoai/NAF `src/model/naf.py` + `src/layers/{attentions,convolutions,rope}.py`
// (checkpoint naf_release.pth) faithfully for a single batch element (B=1) --
// see docs/spec/30-pixal3d-cond.md section 4.
//
// CPU-only, plain loops, f32: weights are pulled to host once via tensor_to_f32
// and the whole forward pass runs as ordinary C++ arithmetic (no ggml graph).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

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
// CUDA dispatch (src/naf_gpu.cpp + src/naf_attn.cu), only compiled/linked into
// CUDA builds (TRELLIS_USE_CUDA -- see CMakeLists.txt): ggml encoder graph +
// custom neighborhood-attention CUDA kernel. naf_upsample() in naf.cpp calls
// these automatically (see its dispatch at the top); not meant to be called
// directly by other code.
#if defined(TRELLIS_USE_CUDA)
bool naf_gpu_available(const Model& naf, int S, int T, int h, int w);
std::vector<float> naf_upsample_gpu(const Model& naf, const float* image, int S,
                                     const float* lr, int C, int h, int w, int T,
                                     NafDebug* dbg);
#endif

// ---------------------------------------------------------------------------
// All-ggml path (src/naf_gpu.cpp, every backend): the whole upsampler --
// encoder, RoPE, k pooling, the cross-scale neighborhood attention and the
// [C,T,T] map -- as one ggml graph on the Model's backend, nothing on the host
// in between. This is the WebGPU/WASM path (docs/spec/30-pixal3d-cond.md
// section 4; docs/PIXAL3D_WEBGPU_OP_GAP.md D1) and doubles as the Metal/Vulkan
// one. The attention is not a custom kernel: with an integer upsample factor
// d = T/h every pixel of a d x d block shares one clamped 9x9 window of the
// low-res k/v maps (naf_attn.h's equivalence argument), so it is 81 gathered
// keys/values per block and two batched mul_mats over [blocks x heads]:
//   logits[81, d^2] = K_win[64, 81]^T q[64, d^2]  ->  softmax  ->
//   out[C/4, d^2]   = V_win[81, C/4]^T P[81, d^2]
// The map comes out pixel-major ([C, T*T], rows = pixels) in BLOCK-MAJOR pixel
// order (naf_block_order); consumers that gather from it (pixal3d_cond_slat_gpu)
// remap their pixel indices, naf_upsample_ggml un-permutes on the host.

struct NafGgmlOpts {
    // Lower GroupNorm / reflect-pad / avg-pool to ops the backend has when it lacks
    // GGML_OP_GROUP_NORM / PAD_REFLECT_1D / POOL_2D (the ggml WebGPU backend):
    // norm over a [W*H*C/8, 8] reshape, concat of mirrored border rows/cols, and
    // sum_rows over a [k, ...] reshape. Exact re-expressions, not approximations.
    bool generic_lowering = false;
    // ggml_conv_2d_direct (f32 activations, no [K*K*Ci, W*H] im2col buffer) instead
    // of ggml_conv_2d (im2col in the weight dtype + mul_mat).
    bool direct_conv = false;
};
// Probes the backend's supports_op for the ops above and returns the opts for it.
NafGgmlOpts naf_ggml_opts_for(const Model& naf);

// Input tensors created by naf_build (all flagged ggml_set_input; the caller
// uploads them after allocation with the host tables below).
struct NafGraphInputs {
    ggml_tensor* img = nullptr;       // [S,S,3,1] f32, [0,1] premultiplied RGB (NOT ImageNet-normalized)
    ggml_tensor* rope_cos = nullptr;  // [T*T, 64] f32, naf_rope_tables
    ggml_tensor* rope_sin = nullptr;
    ggml_tensor* win_idx = nullptr;   // [81 * h*w] i32, naf_window_index
    ggml_tensor* blk_idx = nullptr;   // [T*T] i32, naf_block_order (raster index per block-major position)
    // intermediates kept as graph outputs when `debug` is set (layouts as NafDebug: [C,T,T] channel-major,
    // except k_pooled which is the [256, h*w] pixel-major k_rows)
    ggml_tensor* enc_cat = nullptr;
    ggml_tensor* enc_pooled = nullptr;
    ggml_tensor* q_rope = nullptr;
    ggml_tensor* k_rows = nullptr;
};

// Builds the NAF graph in `c` for a view at resolution S (== S', bilinear input
// downsample not supported: S <= 4T) and target T, with h,w = the low-res grid
// (T % h == 0, T % w == 0) and v_rows = the [C, h*w] pixel-major low-res
// feature map (row = y*w + x -- exactly DINOv3's patch-token order, so a view of
// the DINOv3 output at token offset 5 can be passed straight in). Returns the
// upsampled map [C, T*T] pixel-major in block-major pixel order.
ggml_tensor* naf_build(ggml_context* c, const Model& naf, int S, int T, int h, int w,
                       ggml_tensor* v_rows, NafGraphInputs& in, const NafGgmlOpts& opts,
                       bool debug = false);

// Host tables for naf_build's inputs.
// RoPE cos/sin: [T*T*64], index d*T*T + pix (pix = y*T + x), the per-pixel angle
// tables naf_rope_apply_inplace uses (periods = image_encoder.rope.periods).
void naf_rope_tables(int T, const std::vector<float>& periods, std::vector<float>& cos, std::vector<float>& sin);
// Window gather index: [81 * h*w], idx[blk*81 + wy*9 + wx] = low-res row of tap (wy,wx)
// of block blk = py*w + px (naf_na_window's clamped start, dilation dy = T/h, dx = T/w).
void naf_window_index(int T, int h, int w, std::vector<int32_t>& idx);
// Block-major pixel order: raster_of_bm[p'] = raster pixel index (y*T + x) of block-major
// position p' = blk*(dy*dx) + ry*dx + rx; bm_of_raster is its inverse.
void naf_block_order(int T, int h, int w, std::vector<int32_t>& raster_of_bm, std::vector<int32_t>& bm_of_raster);

// ---------------------------------------------------------------------------
// 分割ビルド（T=1024 で [1024, T*T] = 4 GiB になる NAF 出力を一度に作らないため）。
// naf_build と同じ演算列を、(1) encoder、(2) block 行 stripe ごとの RoPE→q/k、
// (3) block chunk ごとの neighborhood attention、の3段に割る。各段が別グラフに
// なるので gallocr のバッファも段ごとに分かれ、WebGPU の maxBufferSize
// (~4 GiB) を超える単一バッファを作らずに済む。
// 詳細と実測値: docs/PIXAL3D_WEBGPU_MEMORY.md
// ---------------------------------------------------------------------------

// (1) ImageEncoder のみ。img [S,S,3,1] -> pooled [T,T,256]（naf_build 内と同じ cat→pool）。
ggml_tensor* naf_build_encoder(ggml_context* c, const Model& naf, ggml_tensor* img,
                               int S, int T, const NafGgmlOpts& o);

// (1b) encoder の片枝だけ（sem=false: image_encoder.encoder, true: sem_encoder）。
// 返り値 [S,S,128]。cat = concat(枝0, 枝1, dim=2) なので、S==T のとき（pool が恒等）は
// 2枝を別グラフで計算して pooled バッファの前半/後半へ直接書けば、concat 用の 1 GiB と
// 片枝を保持したままもう片枝を計算するぶんのピークを削れる。
ggml_tensor* naf_build_encoder_half(ggml_context* c, const Model& naf, ggml_tensor* img,
                                    bool sem, const NafGgmlOpts& o);

// (2) pooled の block 行 stripe [brow0, brow0+nbrow) について RoPE と per-block 平均を建てる。
// pooled は [T,T,256]（(1) の出力を持つ永続テンソルでよい）。rope_cos/rope_sin はこの stripe
// 分だけを切り出した [P,64]（P = nbrow*dy*T）、blk_idx_stripe は stripe 内ローカルな
// block-major -> ローカル raster の索引 [d2*nbrow*w]。
// 出力 q_bm: [256, d2*nbrow*w] block-major、k_rows: [256, nbrow*w]。
struct NafStripeQK { ggml_tensor* q_bm = nullptr; ggml_tensor* k_rows = nullptr; };
NafStripeQK naf_build_qk_stripe(ggml_context* c, ggml_tensor* pooled, ggml_tensor* rope_cos,
                                ggml_tensor* rope_sin, ggml_tensor* blk_idx_stripe,
                                int T, int h, int w, int brow0, int nbrow);

// (3) block [blk0, blk0+nblk_chunk) の neighborhood attention。q_bm_all は [256, T*T]
// block-major、k_rows は [256, h*w]（全 block 必要 -- 窓が隣接 block に伸びるため）、
// v_rows は [C, h*w]、win_idx は [81*h*w]。返り値は [C, d2*nblk_chunk] block-major。
ggml_tensor* naf_build_attn_chunk(ggml_context* c, ggml_tensor* q_bm_all, ggml_tensor* k_rows,
                                  ggml_tensor* v_rows, ggml_tensor* win_idx,
                                  int d2, int nblk_total, int blk0, int nblk_chunk);

struct NafGgmlStats {
    size_t weight_bytes = 0;   // NAF weight buffer
    size_t alloc_bytes = 0;    // gallocr buffer for the graph (activations + temporaries)
    size_t input_bytes = 0;    // image + rope tables + index inputs
    size_t output_bytes = 0;   // the [C, T*T] map
    double total_ms = 0;
    double compute_ms = 0;
    int n_nodes = 0;
};

// True when naf_upsample() should take the ggml graph path for this Model: a GPU
// backend other than CUDA (which has its own kernel path) and a config naf_build
// supports. The CPU backend keeps the reference loops unless a caller invokes
// naf_upsample_ggml directly (trellis-test-naf --ggml).
bool naf_ggml_available(const Model& naf, int S, int T, int h, int w);

// Same contract as naf_upsample (returns [C,T,T] channel-major, raster order, plus
// the NafDebug intermediates) through one ggml graph on naf.backend. `opts`
// nullptr = naf_ggml_opts_for(naf).
std::vector<float> naf_upsample_ggml(const Model& naf, const float* image, int S,
                                      const float* lr, int C, int h, int w, int T,
                                      NafDebug* dbg = nullptr, NafGgmlStats* stats = nullptr,
                                      const NafGgmlOpts* opts = nullptr);

} // namespace trellis
