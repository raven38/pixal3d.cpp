// ggml-graph paths for naf_upsample() (docs/spec/30-pixal3d-cond.md section 4).
//
// Two entry points share the encoder graph below:
//
//  * naf_upsample_gpu (CUDA builds only, TRELLIS_USE_CUDA): the ImageEncoder
//    (Conv2d/GroupNorm/SiLU/adaptive-avg-pool) runs as a ggml graph, RoPE and the k
//    adaptive-pool reuse naf.cpp's exact CPU functions on the host, and the
//    cross-scale neighborhood attention runs as a custom CUDA kernel (naf_attn.cu)
//    that indexes the un-upsampled low-res k/v maps directly (see naf_attn.h for the
//    equivalence argument). Unchanged from the native CUDA milestone.
//
//  * naf_upsample_ggml / naf_build (every backend; the WebGPU/WASM path): the whole
//    upsampler as ONE ggml graph, nothing on the host in between -- encoder, RoPE
//    (host cos/sin tables, rotate-half via views/concat), k pooling and the attention
//    itself. The attention needs no custom kernel: with an integer upsample factor
//    d = T/h every pixel of a d x d block shares one clamped 9x9 window (naf_attn.h),
//    so gathering the 81 window keys/values per block (get_rows) turns it into two
//    batched mul_mats over [blocks x heads] plus a softmax over 81 logits:
//        logits[81, d^2] = K_win[64, 81]^T q_blk[64, d^2]     (per block, per head)
//        out[C/4, d^2]   = V_win[81, C/4]^T softmax(logits/8)
//    Everything is expressed in ops the ggml WebGPU backend has; the ops it lacks
//    (GROUP_NORM, PAD_REFLECT_1D, POOL_2D) are re-expressed exactly when
//    NafGgmlOpts::generic_lowering is set (naf_ggml_opts_for probes supports_op).
//
// Only ever called from naf.cpp's naf_upsample() dispatch and the tests.
#include "naf.h"
#include "trellis_model.h"
#include "graph_dump.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#if defined(TRELLIS_USE_CUDA)
#include "naf_attn.h"
#endif

#include <cctype>
#include <chrono>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trellis {
using GT = ggml_tensor;

namespace {

// Reflect pad by `pad` on W (ne0) then H (ne1). Native: ggml_pad_reflect_1d on each axis (transposing
// in between). Generic: the mirrored border columns/rows are views concatenated around the tensor
// (reflect: index -1 -> 1, W -> W-2), an exact index remap, pad == 1 only (all NAF convs are k<=3).
GT* reflect_pad2d(ggml_context* c, GT* x, int pad, const NafGgmlOpts& o) {
    if (pad <= 0) return x;
    if (!o.generic_lowering) {
        GT* xw  = ggml_pad_reflect_1d(c, x, pad, pad);            // pad ne0 (W)
        GT* xt  = ggml_cont(c, ggml_permute(c, xw, 1, 0, 2, 3));  // -> ne0=H, ne1=W+2p
        GT* xth = ggml_pad_reflect_1d(c, xt, pad, pad);           // pad ne0 (H)
        return ggml_cont(c, ggml_permute(c, xth, 1, 0, 2, 3));    // -> ne0=W+2p, ne1=H+2p
    }
    if (pad != 1) throw std::runtime_error("naf: generic reflect pad supports pad == 1 only");
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];
    auto col = [&](int64_t i) { return ggml_cont(c, ggml_view_3d(c, x, 1, H, C, x->nb[1], x->nb[2], (size_t)i * x->nb[0])); };
    GT* xw = ggml_concat(c, ggml_concat(c, col(1), x, 0), col(W - 2), 0);                 // [W+2, H, C]
    auto row = [&](int64_t j) { return ggml_cont(c, ggml_view_3d(c, xw, W + 2, 1, C, xw->nb[1], xw->nb[2], (size_t)j * xw->nb[1])); };
    return ggml_concat(c, ggml_concat(c, row(1), xw, 1), row(H - 2), 1);                  // [W+2, H+2, C]
}

// Conv2d + bias, reflect-padded (pad = K/2, a no-op for the K==1 convs). Tensor
// names match naf.cpp's load_conv exactly (image_encoder.{encoder,sem_encoder}.*).
GT* conv_bias(ggml_context* c, const Model& m, const std::string& name, GT* x, const NafGgmlOpts& o) {
    GT* w = m.get(name + ".weight");        // ggml ne = [K,K,Ci,Co] (torch [Co,Ci,K,K] reversed)
    const int K = (int)w->ne[0];
    GT* xin = reflect_pad2d(c, x, K / 2, o);
    GT* y = o.direct_conv ? ggml_conv_2d_direct(c, w, xin, 1, 1, 0, 0, 1, 1)
                          : ggml_conv_2d(c, w, xin, 1, 1, 0, 0, 1, 1);
    GT* b = m.get(name + ".bias");
    return ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
}

// GroupNorm(8, eps=1e-5) + affine (ggml_group_norm has no affine of its own). Generic: the 8
// groups are 8 contiguous slabs of a [W,H,C] tensor, so ggml_norm over a [W*H*C/8, 8] reshape is
// the same statistic (biased variance over the group's channels x pixels).
GT* group_norm_affine(ggml_context* c, const Model& m, const std::string& name, GT* x, const NafGgmlOpts& o) {
    GT* y;
    if (!o.generic_lowering) {
        y = ggml_group_norm(c, x, 8, 1e-5f);
    } else {
        const int64_t n = ggml_nelements(x) / 8;
        y = ggml_norm(c, ggml_reshape_2d(c, x, n, 8), 1e-5f);
        y = ggml_reshape_4d(c, y, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    }
    GT* w = m.get(name + ".weight");
    GT* b = m.get(name + ".bias");
    y = ggml_mul(c, y, ggml_reshape_4d(c, w, 1, 1, w->ne[0], 1));
    y = ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
    return y;
}

// EncBlock(x) = conv2(SiLU(GN2(conv1(SiLU(GN1(x)))))) -- no residual (naf.cpp's enc_block_forward).
GT* enc_block(ggml_context* c, const Model& m, const std::string& prefix, GT* x, const NafGgmlOpts& o) {
    GT* h = group_norm_affine(c, m, prefix + ".norm1", x, o);
    h = ggml_silu(c, h);
    h = conv_bias(c, m, prefix + ".conv1", h, o);
    h = group_norm_affine(c, m, prefix + ".norm2", h, o);
    h = ggml_silu(c, h);
    h = conv_bias(c, m, prefix + ".conv2", h, o);
    return h;
}

// branch(x) = EncBlock(EncBlock(conv0(x))) -- encoder (k=1) or sem_encoder (k=3, reflect).
GT* branch(ggml_context* c, const Model& m, const std::string& prefix, GT* img, const NafGgmlOpts& o) {
    GT* e = conv_bias(c, m, prefix + ".0", img, o);
    e = enc_block(c, m, prefix + ".1", e, o);
    e = enc_block(c, m, prefix + ".2", e, o);
    return e;
}

// AvgPool2d(k, stride k) on [W,H,C]. Generic: sum over each axis' k-blocks with sum_rows on a
// [k, ...] reshape (transpose in between), then scale by 1/k^2 -- f32 sums of k^2 values.
GT* avg_pool_k(ggml_context* c, GT* x, int k, const NafGgmlOpts& o) {
    if (k == 1) return x;
    if (!o.generic_lowering) return ggml_pool_2d(c, x, GGML_OP_POOL_AVG, k, k, k, k, 0, 0);
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];
    GT* a = ggml_sum_rows(c, ggml_reshape_2d(c, x, k, (W / k) * H * C));            // [1, W/k*H*C]
    a = ggml_cont(c, ggml_transpose(c, ggml_reshape_3d(c, a, W / k, H, C)));        // [H, W/k, C]
    GT* b = ggml_sum_rows(c, ggml_reshape_2d(c, a, k, (H / k) * (W / k) * C));      // [1, H/k*W/k*C]
    b = ggml_cont(c, ggml_transpose(c, ggml_reshape_3d(c, b, H / k, W / k, C)));    // [W/k, H/k, C]
    return ggml_scale(c, b, 1.0f / (float)(k * k));
}

// The shared encoder graph: img [Sp,Sp,3,1] -> cat [Sp,Sp,256,1] -> pooled [T,T,256,1].
void build_encoder(ggml_context* c, const Model& naf, GT* img, int Sp, int T, const NafGgmlOpts& o,
                   GT*& cat, GT*& pooled) {
    GT* e1 = branch(c, naf, "image_encoder.encoder", img, o);      // k=1 convs, reflect pad is a no-op
    GT* e2 = branch(c, naf, "image_encoder.sem_encoder", img, o);  // k=3 convs, reflect pad=1
    cat = ggml_concat(c, e1, e2, 2);                                // [Sp,Sp,256,1]
    pooled = avg_pool_k(c, cat, (Sp == T) ? 1 : Sp / T, o);
}

GT* new_input(ggml_context* c, ggml_type t, int64_t n0, int64_t n1 = 1) {
    GT* x = ggml_new_tensor_2d(c, t, n0, n1);
    ggml_set_input(x);
    return x;
}

// Builds one node of each lowered op in a scratch context and asks the device.
bool dev_supports(const Model& m, GT* node) {
    ggml_backend_dev_t dev = ggml_backend_get_device(m.backend);
    return dev == nullptr || ggml_backend_dev_supports_op(dev, node);
}

} // namespace

NafGgmlOpts naf_ggml_opts_for(const Model& naf) {
    NafGgmlOpts o;
    ggml_context* c = ggml_init({ ggml_tensor_overhead() * 32 + 4096, nullptr, true });
    GT* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, 16, 16, 16, 1);
    GT* gn = ggml_group_norm(c, x, 8, 1e-5f);
    GT* pr = ggml_pad_reflect_1d(c, x, 1, 1);
    GT* pl = ggml_pool_2d(c, x, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
    GT* k3 = ggml_new_tensor_4d(c, GGML_TYPE_F16, 3, 3, 16, 16);
    GT* cd = ggml_conv_2d_direct(c, k3, x, 1, 1, 0, 0, 1, 1);
    o.generic_lowering = !(dev_supports(naf, gn) && dev_supports(naf, pr) && dev_supports(naf, pl));
    // 診断専用: WebGPU と同じ lowering を他 backend で再現してメモリを測るためのトグル
    // (docs/PIXAL3D_WEBGPU_MEMORY.md §11)。値は計算内容を変えない厳密な再表現。
    if (const char* e = getenv("TRELLIS_DBG_NAF_GENERIC")) o.generic_lowering = (*e != '0');
    // Direct conv only where the im2col path is the worse choice (WebGPU: no [K*K*Ci, W*H] f16
    // buffer, f32 activations into the GEMM); CUDA/CPU keep the validated im2col graph.
    const bool is_webgpu = naf.backend && strncmp(ggml_backend_name(naf.backend), "WebGPU", 6) == 0;
    o.direct_conv = is_webgpu && dev_supports(naf, cd);
    ggml_free(c);
    return o;
}

void naf_rope_tables(int T, const std::vector<float>& periods, std::vector<float>& cosv, std::vector<float>& sinv) {
    if ((int)periods.size() != 16) throw std::runtime_error("naf_rope_tables: periods must have 16 entries");
    const size_t TT = (size_t)T * T;
    cosv.resize(TT * 64); sinv.resize(TT * 64);
    for (int i = 0; i < T; ++i) {
        const float u = ((float)i + 0.5f) / (float)T * 2.f - 1.f;
        for (int j = 0; j < T; ++j) {
            const float v = ((float)j + 0.5f) / (float)T * 2.f - 1.f;
            const size_t pix = (size_t)i * T + j;
            for (int k = 0; k < 16; ++k) {
                const float au = 2.f * (float)M_PI * u / periods[k];
                const float av = 2.f * (float)M_PI * v / periods[k];
                const float cu = std::cos(au), su = std::sin(au), cv = std::cos(av), sv = std::sin(av);
                // d = k | 16+k | 32+k | 48+k  (angles tiled x2 over the two 32-wide halves)
                cosv[(size_t)k * TT + pix] = cu;        sinv[(size_t)k * TT + pix] = su;
                cosv[(size_t)(16 + k) * TT + pix] = cv; sinv[(size_t)(16 + k) * TT + pix] = sv;
                cosv[(size_t)(32 + k) * TT + pix] = cu; sinv[(size_t)(32 + k) * TT + pix] = su;
                cosv[(size_t)(48 + k) * TT + pix] = cv; sinv[(size_t)(48 + k) * TT + pix] = sv;
            }
        }
    }
}

void naf_window_index(int T, int h, int w, std::vector<int32_t>& idx) {
    const int K = 9, dy = T / h, dx = T / w;
    idx.resize((size_t)81 * h * w);
    for (int py = 0; py < h; ++py) {
        int ry, ppy, Lry, starty;
        naf_na_window(py * dy, T, dy, K, ry, ppy, Lry, starty);
        for (int px = 0; px < w; ++px) {
            int rx, ppx, Lrx, startx;
            naf_na_window(px * dx, T, dx, K, rx, ppx, Lrx, startx);
            int32_t* o = idx.data() + (size_t)(py * w + px) * 81;
            for (int wy = 0; wy < K; ++wy)
                for (int wx = 0; wx < K; ++wx)
                    o[wy * K + wx] = (starty + wy) * w + (startx + wx);
        }
    }
}

void naf_block_order(int T, int h, int w, std::vector<int32_t>& raster_of_bm, std::vector<int32_t>& bm_of_raster) {
    const int dy = T / h, dx = T / w;
    const size_t TT = (size_t)T * T;
    raster_of_bm.resize(TT); bm_of_raster.resize(TT);
    for (int py = 0; py < h; ++py)
        for (int px = 0; px < w; ++px)
            for (int ry = 0; ry < dy; ++ry)
                for (int rx = 0; rx < dx; ++rx) {
                    const int32_t bm = (int32_t)(((size_t)(py * w + px) * dy + ry) * dx + rx);
                    const int32_t ras = (int32_t)((size_t)(py * dy + ry) * T + (px * dx + rx));
                    raster_of_bm[bm] = ras;
                    bm_of_raster[ras] = bm;
                }
}

GT* naf_build_encoder_half(ggml_context* c, const Model& naf, GT* img, bool sem,
                           const NafGgmlOpts& o) {
    return branch(c, naf, sem ? "image_encoder.sem_encoder" : "image_encoder.encoder", img, o);
}

GT* naf_build_encoder(ggml_context* c, const Model& naf, GT* img, int S, int T,
                      const NafGgmlOpts& o) {
    GT *cat = nullptr, *pooled = nullptr;
    build_encoder(c, naf, img, S, T, o, cat, pooled);
    return pooled;
}

NafStripeQK naf_build_qk_stripe(ggml_context* c, GT* pooled, GT* rope_cos, GT* rope_sin,
                                GT* blk_idx_stripe, int T, int h, int w, int brow0, int nbrow) {
    const int dy = T / h, dx = T / w, d2 = dy * dx, nb = nbrow * w;
    const int64_t sy = (int64_t)nbrow * dy, P = sy * (int64_t)T;
    // pooled [T,T,256] の行 [brow0*dy, +sy) を取り出す（チャネル方向は stride TT なので cont が要る）。
    GT* ps = ggml_cont(c, ggml_view_3d(c, pooled, T, sy, 256, pooled->nb[1], pooled->nb[2],
                                       (size_t)brow0 * dy * pooled->nb[1]));                  // [T, sy, 256]
    GT* x3 = ggml_reshape_3d(c, ggml_reshape_2d(c, ps, P, 256), P, 64, 4);                    // [pix, d, head]
    GT* xa = ggml_cont(c, ggml_view_3d(c, x3, P, 32, 4, x3->nb[1], x3->nb[2], 0));
    GT* xb = ggml_cont(c, ggml_view_3d(c, x3, P, 32, 4, x3->nb[1], x3->nb[2], 32 * x3->nb[1]));
    GT* rot = ggml_concat(c, ggml_neg(c, xb), xa, 1);
    GT* q_rope = ggml_add(c, ggml_mul(c, x3, rope_cos), ggml_mul(c, rot, rope_sin));          // [pix, 64, 4]
    GT* q_rows = ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, q_rope, P, 256)));         // [256, pix]
    NafStripeQK out;
    out.q_bm = ggml_get_rows(c, q_rows, blk_idx_stripe);                                      // [256, d2*nb]
    GT* kt = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, out.q_bm, 256, d2, nb), 1, 0, 2, 3));
    GT* kr = ggml_sum_rows(c, ggml_reshape_2d(c, kt, d2, (int64_t)256 * nb));
    out.k_rows = ggml_scale(c, ggml_reshape_2d(c, kr, 256, nb), 1.0f / (float)d2);            // [256, nb]
    return out;
}

GT* naf_build_attn_chunk(ggml_context* c, GT* q_bm_all, GT* k_rows, GT* v_rows, GT* win_idx,
                         int d2, int nblk_total, int blk0, int nblk_chunk) {
    const int64_t C = v_rows->ne[0];
    if (C % 4 != 0) throw std::runtime_error("naf_build_attn_chunk: C must be divisible by the 4 heads");
    const int nb = nblk_chunk;
    GT* q_bm = q_bm_all;
    GT* win = win_idx;
    if (blk0 != 0 || nb != nblk_total) {
        q_bm = ggml_view_2d(c, q_bm_all, 256, (int64_t)d2 * nb, q_bm_all->nb[1],
                            (size_t)d2 * blk0 * q_bm_all->nb[1]);
        win  = ggml_view_1d(c, win_idx, (int64_t)81 * nb, (size_t)81 * blk0 * ggml_type_size(win_idx->type));
    }
    const float scale = 1.0f / std::sqrt(64.0f);
    GT* q_blk = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, q_bm, 64, 4, d2, nb), 0, 2, 1, 3)); // [64, d2, 4, blk]
    GT* k_win = ggml_get_rows(c, k_rows, win);                                                 // [256, 81*blk]
    k_win = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, k_win, 64, 4, 81, nb), 0, 2, 1, 3));   // [64, 81, 4, blk]
    GT* logits = ggml_mul_mat(c, k_win, q_blk);                                                // [81, d2, 4, blk]
    GT* probs = ggml_soft_max_ext(c, logits, nullptr, scale, 0.0f);
    GT* v_win = ggml_get_rows(c, v_rows, win);                                                 // [C, 81*blk]
    v_win = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, v_win, C / 4, 4, 81, nb), 1, 2, 0, 3)); // [81, C/4, 4, blk]
    GT* out = ggml_mul_mat(c, v_win, probs);                                                   // [C/4, d2, 4, blk]
    GT* hr = ggml_cont(c, ggml_permute(c, out, 0, 2, 1, 3));                                   // [C/4, 4, d2, blk]
    return ggml_reshape_2d(c, hr, C, (int64_t)d2 * nb);                                        // [C, d2*blk]
}

GT* naf_build(ggml_context* c, const Model& naf, int S, int T, int h, int w,
              GT* v_rows, NafGraphInputs& in, const NafGgmlOpts& o, bool debug) {
    if (S > 4 * T) throw std::runtime_error("naf_build: S > 4T (bilinear input downsample) not supported");
    if (T % h != 0 || T % w != 0) throw std::runtime_error("naf_build: T must be a multiple of h and w");
    if (S != T && S % T != 0) throw std::runtime_error("naf_build: S must be a multiple of T");
    const int Sp = S, dy = T / h, dx = T / w, d2 = dy * dx, nblk = h * w;
    const int64_t TT = (int64_t)T * T, C = v_rows->ne[0];
    if (v_rows->ne[1] != nblk) throw std::runtime_error("naf_build: v_rows must be [C, h*w]");
    if (C % 4 != 0) throw std::runtime_error("naf_build: C must be divisible by the 4 heads");

    in.img      = ggml_new_tensor_4d(c, GGML_TYPE_F32, Sp, Sp, 3, 1); ggml_set_input(in.img);
    in.rope_cos = new_input(c, GGML_TYPE_F32, TT, 64);
    in.rope_sin = new_input(c, GGML_TYPE_F32, TT, 64);
    in.win_idx  = new_input(c, GGML_TYPE_I32, (int64_t)81 * nblk);
    in.blk_idx  = new_input(c, GGML_TYPE_I32, TT);

    // ---- encoder: cat [Sp,Sp,256] -> pooled [T,T,256] (== torch [256,T,T] channel-major) ----
    GT *cat, *pooled;
    build_encoder(c, naf, in.img, Sp, T, o, cat, pooled);

    // ---- RoPE: x[d] * cos[d] + rotate_half(x)[d] * sin[d], 4 heads x 64 ----
    GT* x3 = ggml_reshape_3d(c, pooled, TT, 64, 4);                                          // [pix, d, head]
    GT* xa = ggml_cont(c, ggml_view_3d(c, x3, TT, 32, 4, x3->nb[1], x3->nb[2], 0));
    GT* xb = ggml_cont(c, ggml_view_3d(c, x3, TT, 32, 4, x3->nb[1], x3->nb[2], 32 * x3->nb[1]));
    GT* rot = ggml_concat(c, ggml_neg(c, xb), xa, 1);                                        // [-x2 || x1]
    GT* q_rope = ggml_add(c, ggml_mul(c, x3, in.rope_cos), ggml_mul(c, rot, in.rope_sin));   // [pix, 64, 4]

    // ---- pixel-major, block-major q; k = per-block mean (== adaptive_avg_pool2d to (h,w)) ----
    GT* q_rows = ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, q_rope, TT, 256)));       // [256, pix]
    GT* q_bm = ggml_get_rows(c, q_rows, in.blk_idx);                                          // [256, pix'] block-major
    GT* kt = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, q_bm, 256, d2, nblk), 1, 0, 2, 3));      // [d2, 256, blk]
    GT* k_rows = ggml_sum_rows(c, ggml_reshape_2d(c, kt, d2, (int64_t)256 * nblk));                   // [1, 256*blk]
    k_rows = ggml_scale(c, ggml_reshape_2d(c, k_rows, 256, nblk), 1.0f / (float)d2);                 // [256, blk]

    // ---- neighborhood attention over the 81-tap window of each block ----
    // 全 block を1 chunk として naf_build_attn_chunk に委譲する（分割版と同一の演算列）。
    GT* hr = naf_build_attn_chunk(c, q_bm, k_rows, v_rows, in.win_idx, d2, nblk, 0, nblk);
    hr = ggml_reshape_2d(c, hr, C, TT);                                                        // [C, pix'] block-major

    if (debug) {
        in.enc_cat = cat; ggml_set_output(cat);
        in.enc_pooled = pooled; ggml_set_output(pooled);
        in.q_rope = q_rope; ggml_set_output(q_rope);
        in.k_rows = k_rows; ggml_set_output(k_rows);
    }
    ggml_set_output(hr);
    return hr;
}

bool naf_ggml_available(const Model& naf, int S, int T, int h, int w) {
    if (!naf.on_gpu || naf.backend == nullptr) return false;
    std::string bn = ggml_backend_name(naf.backend);
    if (bn.rfind("CUDA", 0) == 0) return false;   // naf_upsample_gpu's kernel path
    if (S > 4 * T) return false;
    if (S != T && (T == 0 || S % T != 0)) return false;
    if (h <= 0 || w <= 0 || T % h != 0 || T % w != 0) return false;
    return true;
}

std::vector<float> naf_upsample_ggml(const Model& naf, const float* image, int S,
                                      const float* lr, int C, int h, int w, int T,
                                      NafDebug* dbg, NafGgmlStats* stats, const NafGgmlOpts* opts_in) {
    const auto t0 = std::chrono::steady_clock::now();
    const NafGgmlOpts o = opts_in ? *opts_in : naf_ggml_opts_for(naf);
    const size_t TT = (size_t)T * T, HW = (size_t)h * w;
    const bool log_timing = std::getenv("TRELLIS_DBG_NAF") != nullptr;
    auto lap = [&](const char* stage) {
        if (!log_timing) return;
        fprintf(stderr, "[naf_ggml] %-16s @ %8.1f ms\n", stage,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    };

    const size_t nodes = 4096;
    size_t nmeta = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20);
    ggml_context* c = ggml_init({ nmeta, nullptr, true });

    // lr [C,h,w] channel-major -> v_rows [C, h*w] pixel-major (row = y*w + x)
    GT* v_rows = new_input(c, GGML_TYPE_F32, C, (int64_t)HW);
    NafGraphInputs in;
    GT* hr = naf_build(c, naf, S, T, h, w, v_rows, in, o, dbg != nullptr);

    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, hr);
    if (dbg) { ggml_build_forward_expand(g, in.enc_cat); ggml_build_forward_expand(g, in.enc_pooled);
               ggml_build_forward_expand(g, in.q_rope); ggml_build_forward_expand(g, in.k_rows); }
    const std::string tag = "naf_ggml_S" + std::to_string(S) + "_T" + std::to_string(T);
    trellis_graph_dump(tag.c_str(), g);
    check_graph_supported(naf.backend, g, tag.c_str());

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(naf.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error("naf_upsample_ggml: graph alloc failed");
    if (log_timing)
        fprintf(stderr, "[naf_ggml] S=%d T=%d h=%d w=%d nodes=%d generic=%d direct_conv=%d activations=%.1f MB output=%.1f MB\n",
                S, T, h, w, ggml_graph_n_nodes(g), (int)o.generic_lowering, (int)o.direct_conv,
                ggml_gallocr_get_buffer_size(alloc, 0) / 1048576.0, ggml_nbytes(hr) / 1048576.0);
    lap("graph alloc");

    std::vector<float> periods = tensor_to_f32(naf.get("image_encoder.rope.periods"));
    std::vector<float> rcos, rsin;
    naf_rope_tables(T, periods, rcos, rsin);
    std::vector<int32_t> win_idx, raster_of_bm, bm_of_raster;
    naf_window_index(T, h, w, win_idx);
    naf_block_order(T, h, w, raster_of_bm, bm_of_raster);
    std::vector<float> vh(C * HW);
    for (int ch = 0; ch < C; ++ch)
        for (size_t p = 0; p < HW; ++p) vh[p * C + ch] = lr[(size_t)ch * HW + p];

    ggml_backend_tensor_set(in.img, image, 0, (size_t)3 * S * S * sizeof(float));
    ggml_backend_tensor_set(in.rope_cos, rcos.data(), 0, rcos.size() * sizeof(float));
    ggml_backend_tensor_set(in.rope_sin, rsin.data(), 0, rsin.size() * sizeof(float));
    ggml_backend_tensor_set(in.win_idx, win_idx.data(), 0, win_idx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(in.blk_idx, raster_of_bm.data(), 0, raster_of_bm.size() * sizeof(int32_t));
    ggml_backend_tensor_set(v_rows, vh.data(), 0, vh.size() * sizeof(float));
    lap("inputs");

    const auto tc = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(naf.backend, g) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("naf_upsample_ggml: graph compute failed");
    ggml_backend_synchronize(naf.backend);
    const double compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    lap("graph compute");

    // [C, pix'] block-major pixel-major -> [C, T, T] channel-major raster (NafDebug/naf_upsample layout)
    std::vector<float> hr_bm = tensor_to_f32(hr);
    std::vector<float> out((size_t)C * TT);
    for (size_t p = 0; p < TT; ++p) {
        const float* src = hr_bm.data() + p * (size_t)C;
        const size_t ras = (size_t)raster_of_bm[p];
        for (int ch = 0; ch < C; ++ch) out[(size_t)ch * TT + ras] = src[ch];
    }
    if (dbg) {
        dbg->S_prime = S;
        dbg->enc_cat = tensor_to_f32(in.enc_cat);
        dbg->enc_pooled = tensor_to_f32(in.enc_pooled);
        dbg->q_rope = tensor_to_f32(in.q_rope);
        std::vector<float> kr = tensor_to_f32(in.k_rows);          // [256, h*w] pixel-major
        dbg->k_pooled.assign((size_t)256 * HW, 0.f);
        for (size_t p = 0; p < HW; ++p)
            for (int ch = 0; ch < 256; ++ch) dbg->k_pooled[(size_t)ch * HW + p] = kr[p * 256 + ch];
        dbg->k_up = naf_nearest_exact_resize(dbg->k_pooled, 256, h, w, T, T);
        std::vector<float> lr_vec(lr, lr + (size_t)C * HW);
        dbg->v_up = naf_nearest_exact_resize(lr_vec, C, h, w, T, T);
    }
    if (stats) {
        stats->weight_bytes = naf.buffer ? ggml_backend_buffer_get_size(naf.buffer) : 0;
        stats->alloc_bytes = ggml_gallocr_get_buffer_size(alloc, 0);
        stats->input_bytes = ggml_nbytes(in.img) + ggml_nbytes(in.rope_cos) + ggml_nbytes(in.rope_sin)
                           + ggml_nbytes(in.win_idx) + ggml_nbytes(in.blk_idx) + ggml_nbytes(v_rows);
        stats->output_bytes = ggml_nbytes(hr);
        stats->compute_ms = compute_ms;
        stats->n_nodes = ggml_graph_n_nodes(g);
        stats->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    ggml_gallocr_free(alloc);
    ggml_free(c);
    lap("readback+free");
    return out;
}

#if defined(TRELLIS_USE_CUDA)

namespace {
int cuda_device_index(const Model& m) {
    std::string bn = ggml_backend_name(m.backend); // e.g. "CUDA0"
    size_t i = 0;
    while (i < bn.size() && !std::isdigit((unsigned char)bn[i])) ++i;
    return i < bn.size() ? std::atoi(bn.c_str() + i) : 0;
}
} // namespace

// Preconditions the ggml encoder graph relies on: the S>4T host-downsample path
// (never hit by any real Pixal3D config: S in {512,1024}, T in {128,512,1024})
// stays on the CPU reference rather than being re-derived here; the two
// adaptive-avg-pools (encoder cat -> T, and q_rope -> h,w) must be exact integer
// factors (true for every Pixal3D stage config) since they're implemented as
// ggml_pool_2d(k=s=factor); and the neighborhood-attention dilations T/h, T/w must
// be integers (always true -- h,w are S/16, T is always a multiple of 16).
bool naf_gpu_available(const Model& naf, int S, int T, int h, int w) {
    if (!naf.on_gpu || naf.backend == nullptr) return false;
    std::string bn = ggml_backend_name(naf.backend);
    if (bn.rfind("CUDA", 0) != 0) return false;
    if (S > 4 * T) return false;
    if (S != T && (T == 0 || S % T != 0)) return false;
    if (h <= 0 || w <= 0 || T % h != 0 || T % w != 0) return false;
    return true;
}

std::vector<float> naf_upsample_gpu(const Model& naf, const float* image, int S,
                                     const float* lr, int C, int h, int w, int T,
                                     NafDebug* dbg) {
    const int Sp = S; // naf_gpu_available() already rejected S > 4*T
    const bool log_timing = std::getenv("TRELLIS_DBG_NAF") != nullptr;
    auto t_start = std::chrono::steady_clock::now();
    auto lap = [&](const char* stage) {
        if (!log_timing) return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        fprintf(stderr, "[naf_gpu] %-16s @ %8.1f ms\n", stage, ms);
    };

    const size_t nodes = 16384;
    size_t nmeta = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20);
    ggml_context* c = ggml_init({ nmeta, nullptr, true });

    GT* img = ggml_new_tensor_4d(c, GGML_TYPE_F32, Sp, Sp, 3, 1);
    ggml_set_input(img);

    const NafGgmlOpts o;   // native ops, im2col conv: the validated CUDA graph
    GT *cat, *pooled;
    build_encoder(c, naf, img, Sp, T, o, cat, pooled);

    ggml_set_output(cat);
    ggml_set_output(pooled);
    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, cat);
    ggml_build_forward_expand(g, pooled);
    trellis_graph_dump(("naf_encoder_S" + std::to_string(Sp) + "_T" + std::to_string(T)).c_str(), g);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(naf.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error("naf_upsample_gpu: graph alloc failed");
    lap("graph alloc");
    ggml_backend_tensor_set(img, image, 0, (size_t)3 * Sp * Sp * sizeof(float));
    if (ggml_backend_graph_compute(naf.backend, g) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("naf_upsample_gpu: graph compute failed");
    lap("graph compute");

    std::vector<float> enc_pooled = tensor_to_f32(pooled); // [256,T,T] channel-major
    if (dbg) { dbg->S_prime = Sp; dbg->enc_cat = tensor_to_f32(cat); dbg->enc_pooled = enc_pooled; }
    ggml_gallocr_free(alloc);
    ggml_free(c);
    lap("readback+free");

    // RoPE + k adaptive-pool: cheap O(T^2*C) elementwise ops, reuse the exact CPU
    // reference (see file header) instead of re-implementing them in ggml/CUDA.
    std::vector<float> periods = tensor_to_f32(naf.get("image_encoder.rope.periods"));
    naf_rope_apply_inplace(enc_pooled, T, periods); // enc_pooled is now q_rope
    lap("rope");
    if (dbg) dbg->q_rope = enc_pooled;
    std::vector<float> k_pooled = naf_adaptive_avg_pool2d(enc_pooled, 256, T, T, h, w);
    lap("k_pool");
    if (dbg) {
        dbg->k_pooled = k_pooled;
        dbg->k_up = naf_nearest_exact_resize(k_pooled, 256, h, w, T, T);
        std::vector<float> lr_vec(lr, lr + (size_t)C * h * w);
        dbg->v_up = naf_nearest_exact_resize(lr_vec, C, h, w, T, T);
        lap("dbg k_up/v_up");
    }

    std::vector<float> out((size_t)C * T * T);
    naf_attn_cuda(enc_pooled.data(), k_pooled.data(), lr, T, h, w, C, 1.0f / 8.0f,
                  out.data(), cuda_device_index(naf));
    lap("attn kernel");
    return out;
}

#endif // TRELLIS_USE_CUDA

} // namespace trellis
