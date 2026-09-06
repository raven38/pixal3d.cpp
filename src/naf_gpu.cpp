// GPU dispatch for naf_upsample() (docs/spec/30-pixal3d-cond.md section 4): the
// ImageEncoder (Conv2d/GroupNorm/SiLU/adaptive-avg-pool) runs as a ggml graph on
// the Model's backend, and the cross-scale neighborhood attention -- the heavy
// part -- runs as a custom CUDA kernel (naf_attn.cu) that indexes the un-upsampled
// low-res k/v maps directly (see naf_attn.h for the equivalence argument), instead
// of materializing k_up/v_up ([256,T,T] / [C,T,T], up to 4 GB f32 at T=1024).
//
// RoPE and the k adaptive-pool are cheap, O(T^2*C) elementwise passes (not the
// bottleneck -- the convs and the 9x9-window attention are), so this file reuses
// naf.cpp's exact, already fixture-verified CPU implementations
// (naf_rope_apply_inplace, naf_adaptive_avg_pool2d, naf_nearest_exact_resize)
// rather than re-deriving the same numerics a second time in ggml/CUDA. Only two
// pieces of new numeric code exist in this GPU path: the ggml encoder graph below,
// and the attention kernel in naf_attn.cu.
//
// Only ever called from naf.cpp's naf_upsample() dispatch, and only when compiled
// with CUDA (TRELLIS_USE_CUDA) -- see CMakeLists.txt. On any other backend (Metal,
// Vulkan, CPU-only) naf_upsample() takes the original, unmodified CPU path, so
// this file does not affect those builds at all.
#include "naf.h"
#include "naf_attn.h"
#include "trellis_model.h"
#include "graph_dump.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trellis {
using GT = ggml_tensor;

namespace {

GT* reflect_pad2d(ggml_context* c, GT* x, int pad) {
    if (pad <= 0) return x;
    GT* xw  = ggml_pad_reflect_1d(c, x, pad, pad);            // pad ne0 (W)
    GT* xt  = ggml_cont(c, ggml_permute(c, xw, 1, 0, 2, 3));  // -> ne0=H, ne1=W+2p
    GT* xth = ggml_pad_reflect_1d(c, xt, pad, pad);           // pad ne0 (H)
    return ggml_cont(c, ggml_permute(c, xth, 1, 0, 2, 3));    // -> ne0=W+2p, ne1=H+2p
}

// Conv2d + bias, reflect-padded (pad = K/2, a no-op for the K==1 convs). Tensor
// names match naf.cpp's load_conv exactly (image_encoder.{encoder,sem_encoder}.*).
GT* conv_bias(ggml_context* c, const Model& m, const std::string& name, GT* x) {
    GT* w = m.get(name + ".weight");        // ggml ne = [K,K,Ci,Co] (torch [Co,Ci,K,K] reversed)
    const int K = (int)w->ne[0];
    GT* xin = reflect_pad2d(c, x, K / 2);
    GT* y = ggml_conv_2d(c, w, xin, 1, 1, 0, 0, 1, 1);
    GT* b = m.get(name + ".bias");
    return ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
}

// GroupNorm(8, eps=1e-5) + affine (ggml_group_norm has no affine of its own).
GT* group_norm_affine(ggml_context* c, const Model& m, const std::string& name, GT* x) {
    GT* y = ggml_group_norm(c, x, 8, 1e-5f);
    GT* w = m.get(name + ".weight");
    GT* b = m.get(name + ".bias");
    y = ggml_mul(c, y, ggml_reshape_4d(c, w, 1, 1, w->ne[0], 1));
    y = ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
    return y;
}

// EncBlock(x) = conv2(SiLU(GN2(conv1(SiLU(GN1(x)))))) -- no residual (naf.cpp's enc_block_forward).
GT* enc_block(ggml_context* c, const Model& m, const std::string& prefix, GT* x) {
    GT* h = group_norm_affine(c, m, prefix + ".norm1", x);
    h = ggml_silu(c, h);
    h = conv_bias(c, m, prefix + ".conv1", h);
    h = group_norm_affine(c, m, prefix + ".norm2", h);
    h = ggml_silu(c, h);
    h = conv_bias(c, m, prefix + ".conv2", h);
    return h;
}

// branch(x) = EncBlock(EncBlock(conv0(x))) -- encoder (k=1) or sem_encoder (k=3, reflect).
GT* branch(ggml_context* c, const Model& m, const std::string& prefix, GT* img) {
    GT* e = conv_bias(c, m, prefix + ".0", img);
    e = enc_block(c, m, prefix + ".1", e);
    e = enc_block(c, m, prefix + ".2", e);
    return e;
}

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
    const int pool_k = (Sp == T) ? 1 : Sp / T;
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

    GT* e1 = branch(c, naf, "image_encoder.encoder", img);      // k=1 convs, reflect pad is a no-op
    GT* e2 = branch(c, naf, "image_encoder.sem_encoder", img);  // k=3 convs, reflect pad=1
    GT* cat = ggml_concat(c, e1, e2, 2);                         // [Sp,Sp,256,1]
    GT* pooled = (pool_k == 1) ? cat
               : ggml_pool_2d(c, cat, GGML_OP_POOL_AVG, pool_k, pool_k, pool_k, pool_k, 0, 0);

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

} // namespace trellis
