// Shared TRELLIS.2 flow-DiT graph builder (dense path, B=1).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace trellis {
struct Model;

struct DiTParams {
    int n_blocks   = 30;
    int n_heads    = 12;
    int head_dim   = 128;
    int d_model    = 1536;
    int d_mlp      = 8192;     // int(1536 * 5.3334)
    int d_cond     = 1024;
    int in_ch      = 8;
    int out_ch     = 8;
    float ln_eps       = 1e-6f;
    float final_ln_eps = 1e-5f;
    float rms_eps      = 1e-12f;
    bool  cast_f32     = false;   // cast f16 weights to f32 before matmul (precision test)

    // Pixal3D ProjectAttention (image_attn_mode: "proj"): cross-attn becomes
    // global_out = cross_attn_block(norm2(h), global_cond); proj_out = proj_linear(proj_cond);
    // h = h + (global_out + proj_out). Set via dit_detect_proj_attn(), not by hand.
    bool proj_attn = false;
    int  d_proj    = 0;
};

// Detect Pixal3D's ProjectAttention from checkpoint tensor names (no JSON parsing): keyed off
// the presence and ne[0] (= proj_in) of "blocks.0.cross_attn.proj_linear.weight". Sets
// p.proj_attn/p.d_proj and returns whether it was detected (false, params unset -> TRELLIS.2).
bool dit_detect_proj_attn(const Model& m, DiTParams& p);

// Build the dense SS-flow forward graph (B=1). All input tensors live in `gctx`
// and must be flagged ggml_set_input by the caller; weights come from `m`.
//   h0   : [in_ch, L]          patchified input (channel-major)
//   tfreq: [256]               sinusoidal timestep embedding (host-computed)
//   cond : [d_cond, Lc]        conditioning tokens (global cond)
//   cos/sin: [1, head_dim/2, 1, L]  precomputed 3D-RoPE tables
//   proj : [d_proj, L]         Pixal3D proj_cond, one token per latent position (optional;
//                              only used when p.proj_attn); ignored/unused otherwise.
//   rope_idx: I32 [head_dim]   accepted for API compatibility, unused since the RoPE rewrite
//                              (2026-09-20): q/k are rotated in a de-interleaved head_dim layout
//                              and never scattered back, so no index tensor is needed.
// Returns the [out_ch, L] velocity; `inter` (optional) collects named intermediates.
ggml_tensor* build_dit_dense(ggml_context* gctx, const Model& m, const DiTParams& p,
                             ggml_tensor* h0, ggml_tensor* tfreq, ggml_tensor* cond,
                             ggml_tensor* cos, ggml_tensor* sin,
                             std::map<std::string, ggml_tensor*>* inter = nullptr,
                             ggml_tensor* proj = nullptr, ggml_tensor* rope_idx = nullptr);

// Host contents of the `rope_idx` input: [0,2,..,head_dim-2, 1,3,..,head_dim-1]. Kept for the
// callers that still upload it; build_dit_dense ignores the tensor.
void dit_rope_index(int head_dim, std::vector<int32_t>& out);

// The DiT's 3D RoPE as a graph op, exposed for trellis-test-rope-layout. x: [head_dim, n_heads, L]
// f32; cos/sin: [1, head_dim/2, 1, L] (or any contiguous tensor of head_dim/2 * L floats laid out
// data[token*half + pair]). Returns [head_dim, n_heads, L] in the DE-INTERLEAVED layout
// out[p*half + i] = (p == 0 ? x[2i]*cos_i - x[2i+1]*sin_i : x[2i+1]*cos_i + x[2i]*sin_i).
ggml_tensor* dit_rope(ggml_context* gctx, ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin);

} // namespace trellis
