// Shared TRELLIS.2 flow-DiT graph builder (dense path, B=1).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;

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

// FlashAttention の K/V ストレージ型（--fa-kv）。AUTO は build_dit_dense が backend ごとに解決する。
// 現時点の既定は全 backend で BF16（従来どおり）。Metal の bf16 カーネルは出力を half で累積し
// set_prec を読まないので f16 の方が数値的に有利な可能性があり、docs/design/2026-09-20-metal-fa-f16-kv.md
// のゲートを通ったら Metal の AUTO だけ F16 に切り替える予定。
// テストバイナリ（args を解析しない）向けの環境フォールバック: TRELLIS_FA_KV=bf16|f16|f32、
// TRELLIS_FA_FAST=1（f16 の旧名）。
enum FaKv { FA_KV_AUTO = 0, FA_KV_BF16 = 1, FA_KV_F16 = 2, FA_KV_F32 = 3 };

// One attention as the DiT builds it (sdpa + its zero-padded key mask), exposed for
// trellis-test-fa-oracle: q, k, v are [head_dim, n_heads, L] f32; returns [d_model, L] f32. Uses the
// current g_no_fa / g_fa_kv (resolved for `backend`) exactly like build_dit_dense does.
ggml_tensor* dit_sdpa(ggml_context* gctx, struct ggml_backend* backend,
                      ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, int d_model);
extern int g_fa_kv;                                  // --fa-kv; set by trellis_run (default AUTO)
int  dit_fa_kv_parse(const char* s);                 // "auto"/"bf16"/"f16"/"f32" -> FaKv, -1 if unknown
int  dit_fa_kv_effective(struct ggml_backend* backend);  // resolves AUTO for this backend
const char* dit_fa_kv_name(int kv);

} // namespace trellis
