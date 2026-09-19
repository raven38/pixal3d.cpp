#include "dit.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace trellis {

using T = ggml_tensor;

static bool g_cast_f32 = false;   // set per build_dit_dense call

// Budget for one query chunk's [Lk, nq, nh] score tile in the exact (non-FA) SDPA path.
// Bounds the peak regardless of Lq, which is what made FA necessary in the first place.
// sdpa のクエリ分割 1 チャンクあたりのスコア行列の予算。分割数を変えても結果は同じ
// （softmax はクエリ行ごとに閉じている）なので、これは純粋にメモリと速度の調整値。
//
// ブラウザだけ既定を下げる理由: WebGPU のデバイス予算は実測 4095 MB しかなく、
// テクスチャ flow（N=17690, n_heads=12）の常駐は
//   重み 2647 MB + 活性化 + conditioning 276 MB
// なので、活性化を 1172 MB 以下に収めないと予算を超える。超えるとユニファイドメモリ上で
// Metal がメモリを往復させ続け、1 forward が 40 秒から 139〜176 秒に伸びてマシン全体が
// 巻き添えで固まる（2026-09-08 実測）。
// 実測（native Metal, N=17690）: 1024 MB -> 活性化 1888 MB / 256 MB -> 1127 MB /
// 128 MB -> 1032 MB（ここで attention 以外が支配的になり頭打ち）。128 MB なら
// 合計 3956 MB で予算内に収まる。native は速度優先で 1024 MB のまま。
#ifdef __EMSCRIPTEN__
static constexpr int64_t kAttnChunkBytes = 128ll * 1024 * 1024;
#else
static constexpr int64_t kAttnChunkBytes = 1024ll * 1024 * 1024;
#endif
bool g_no_fa = false;             // --no-fa; set by trellis_run
bool g_profile = false;           // --profile; set by trellis_run (read in flow_runner.cpp)
int  g_fa_kv = FA_KV_AUTO;        // --fa-kv; set by trellis_run (resolved per build, see below)
static int s_fa_kv_build = FA_KV_BF16;                  // effective K/V type for the graph being built
static std::map<std::string, T*>* s_inter_build = nullptr;  // build_dit_dense's `inter`, for sdpa's probes
// TRELLIS_DBG_FA_RANGE=<letters>: expose every attention's pre-cast Q / K / V (F32) as named
// intermediates so flow_runner.cpp can print their max|x| per forward (F16 headroom check). The value
// selects the operands: any string containing 'q' / 'k' / 'v' ("1" or "all" = all three). Each kept
// operand costs ~103 MiB x 30 blocks at N=17612 (self) -- "v" alone (the only operand without a static
// bound, see sdpa) is ~3 GiB, "qkv" ~12 GiB. Debug-only: never changes the graph when unset.
static const std::string s_fa_range = []() -> std::string {
    const char* e = std::getenv("TRELLIS_DBG_FA_RANGE");
    if (!e) return "";
    const std::string v(e);
    return (v == "1" || v == "all") ? "qkv" : v;
}();

int dit_fa_kv_parse(const char* v) {
    if (!v) return -1;
    const std::string a(v);
    if (a == "auto") return FA_KV_AUTO;
    if (a == "bf16") return FA_KV_BF16;
    if (a == "f16")  return FA_KV_F16;
    if (a == "f32")  return FA_KV_F32;
    return -1;
}

const char* dit_fa_kv_name(int kv) {
    switch (kv) {
        case FA_KV_AUTO: return "auto";
        case FA_KV_BF16: return "bf16";
        case FA_KV_F16:  return "f16";
        case FA_KV_F32:  return "f32";
    }
    return "?";
}

// AUTO の解決。優先順: --fa-kv > TRELLIS_FA_KV > TRELLIS_FA_FAST(=f16) > backend 既定。
// backend 既定は今のところ全 backend で BF16（Metal の切替は docs/design/2026-09-20-metal-fa-f16-kv.md の
// ゲートを通してから）。CUDA: BF16 K/V は内部で F16 に変換されるが V_SCALE で範囲内、set_prec(F32) が効く。
// Vulkan (RADV/gfx1151): F16 K/V は sparse flow で crash した記録あり（docs/spec/29-perf-profile.md）。
int dit_fa_kv_effective(ggml_backend* backend) {
    int kv = g_fa_kv;
    if (kv == FA_KV_AUTO) {
        if (const char* e = std::getenv("TRELLIS_FA_KV")) { const int p = dit_fa_kv_parse(e); if (p > 0) kv = p; }
    }
    if (kv == FA_KV_AUTO && std::getenv("TRELLIS_FA_FAST")) kv = FA_KV_F16;
    if (kv == FA_KV_AUTO) {
        (void)backend;   // 現時点では backend 依存の既定は無い（全 backend BF16）
        kv = FA_KV_BF16;
    }
    return kv;
}

static T* lin(ggml_context* c, const Model& m, const std::string& p, T* x) {
    T* w = m.get(p + ".weight");
    if (g_cast_f32 && w->type == GGML_TYPE_F16) w = ggml_cast(c, w, GGML_TYPE_F32);
    T* y = ggml_mul_mat(c, w, x);
    if (T* b = m.try_get(p + ".bias")) y = ggml_add(c, y, b);
    return y;
}

// LayerNorm over ne0. weight/bias optional (affine).
static T* layernorm(ggml_context* c, T* x, float eps, T* w = nullptr, T* b = nullptr) {
    x = ggml_norm(c, x, eps);
    if (w) x = ggml_mul(c, x, w);
    if (b) x = ggml_add(c, x, b);
    return x;
}

// MultiHeadRMSNorm: ggml_rms_norm(x) already == F.normalize(x)*sqrt(head_dim); then * gamma.
static T* rms_gamma(ggml_context* c, T* x, T* gamma, float eps) {
    x = ggml_rms_norm(c, x, eps);
    return ggml_mul(c, x, gamma);   // gamma cast to f32 by caller
}

// x: [head_dim, n_heads, L]; cos/sin: [1, head_dim/2, 1, L] (host layout data[token*half + pair]).
// Interleaved-pair rotation, computed in a DE-INTERLEAVED layout that is never re-interleaved:
// the output is [ev(0..half-1) | od(0..half-1)] along head_dim instead of [ev0,od0,ev1,od1,...].
// q and k both go through this, so their dot product -- the only consumer of the rotated
// values (sdpa) -- is unchanged up to fp32 summation order; V is not rotated.
//
// Why the layout matters (2026-09-20, --profile on M4 Max / Metal, docs/results/
// 2026-09-20-metal-flow-profile.md): the previous form built every elementwise op on
// [1, half, nh, L] tensors (ne0 = 1) plus a zero-filled [1, hd, nh, L] scratch and two
// ggml_set_rows. ggml-metal's unary/binary kernels dispatch one threadgroup per ne0-row, so
// ne0 = 1 meant 64*12*N threadgroups of one thread per op -- 13.5M at N = 17612 -- and the
// RoPE elementwise ops cost ~17 s of a 32 s Shape-1024 forward, more than FlashAttention
// (8 s) and all q8_0 GEMMs (5 s) together. Here every op has rows of `half` (64) elements:
// one cont of the pair-major permute, four MUL, one SUB, one ADD, one CONCAT (a single
// launch on every backend: ne3 = 1). `rope_idx` is accepted for API compatibility and unused.
static T* apply_rope(ggml_context* c, T* x, T* cos, T* sin, T* rope_idx) {
    (void)rope_idx;
    const int64_t hd = x->ne[0], nh = x->ne[1], L = x->ne[2];
    const int64_t half = hd / 2;
    // [2, half, nh*L] -> permute -> [half, 2, nh*L]: row p = 0 holds the even (x[2i]) elements,
    // p = 1 the odd (x[2i+1]) ones, each as a contiguous row of `half`.
    T* xp = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, x, 2, half, nh * L), 1, 0, 2, 3));
    T* x0 = ggml_view_3d(c, xp, half, nh, L, xp->nb[2], xp->nb[2] * nh, 0);           // even  [half, nh, L]
    T* x1 = ggml_view_3d(c, xp, half, nh, L, xp->nb[2], xp->nb[2] * nh, xp->nb[1]);   // odd   [half, nh, L]
    T* cs = ggml_reshape_3d(c, cos, half, 1, L);                                         // broadcast over heads
    T* sn = ggml_reshape_3d(c, sin, half, 1, L);
    T* ev = ggml_sub(c, ggml_mul(c, x0, cs), ggml_mul(c, x1, sn));   // [half, nh, L] rotated even
    T* od = ggml_add(c, ggml_mul(c, x1, cs), ggml_mul(c, x0, sn));   // [half, nh, L] rotated odd
    // [half, 1, nh*L] ++ [half, 1, nh*L] along dim 1 -> [half, 2, nh*L], which is byte-identical to
    // a contiguous [hd, nh, L] with head_dim index = p*half + i.
    T* out = ggml_concat(c, ggml_reshape_3d(c, ev, half, 1, nh * L), ggml_reshape_3d(c, od, half, 1, nh * L), 1);
    return ggml_reshape_3d(c, out, hd, nh, L);
}

// An FA padding mask [Lk_pad, Lq] (F16): 0 for real keys (< Lk_real), a large negative for the
// zero-padded tail. WITHOUT it, ggml's CUDA FlashAttention folds the (zero) padded keys into the
// softmax; on the >=1024-token HR flow that path NaNs a subset of queries (props, <1024 tokens, dodge
// it). WITH it the kernel masks/skips the padded KV tiles -> correct softmax, no NaN. Built once per
// flow (same N every block) and threaded into every attention; -30000 (not -inf) so 0*mask can't NaN.
static T* build_pad_mask(ggml_context* c, int64_t Lk_real, int64_t Lq) {
    const int64_t KQ = 256;
    const int64_t Lk_pad = ((Lk_real + KQ - 1) / KQ) * KQ;
    T* sh = ggml_arange(c, 0.5f - (float)Lk_real, (float)Lk_pad - (float)Lk_real + 0.5f, 1.0f); // [Lk_pad]
    T* col = ggml_scale(c, ggml_step(c, sh), -30000.0f);            // 0 (keep) | -30000 (mask) per key
    T* colh = ggml_cast(c, ggml_reshape_2d(c, col, Lk_pad, 1), GGML_TYPE_F16);
    // The QUERY dim must be padded to GGML_KQ_MASK_PAD too, not just the key dim: CUDA's
    // flash_attn_mask_to_KV_max walks the mask in query tiles (mask += jt*ncols1*s31) to find
    // which KV tiles a tile can skip, so an unpadded Lq makes the last tile read past the end.
    // The garbage it finds there corrupts that tile's KV_max -> wrong keys survive the softmax.
    // Every row here is identical (the mask depends only on key index), so the extra rows are free.
    constexpr int64_t KQ_MASK_PAD = 64;   // llama.cpp's GGML_KQ_MASK_PAD; not exported by ggml
    const int64_t Lq_pad = ((Lq + KQ_MASK_PAD - 1) / KQ_MASK_PAD) * KQ_MASK_PAD;
    return ggml_repeat(c, colh, ggml_new_tensor_2d(c, GGML_TYPE_F16, Lk_pad, Lq_pad)); // [Lk_pad,Lq_pad] F16
}

// DiT ブロックの MLP（fc1 -> GELU -> fc2）をトークン方向に分割する。x: [d_model, N]。
// 中間 [4*d_model, N] は N に比例する最大級の活性化で、tex flow (d_model=1536, N=17489) では
// 24 KiB/token = 410 MiB になる。MLP はトークンごとに完全に独立（fc1/fc2 は列方向に独立、
// GELU は要素ごと、トークンをまたぐ縮約が無い）なので、数学的には分割しても同値。
//
// ただし**ビット一致は保証されない**: 列数を変えると行列積のカーネル選択が変わる。Metal は
// 列数 2〜8 で専用カーネル・9 以上で matrix-matrix 経路、WebGPU は列数 1 で別経路を選ぶ。
// 末尾チャンクが小さいとこの分岐を踏む。docs/design/2026-09-09-texture-flow-mlp-chunking.md 参照。
//
// 既定は sdpa のクエリ分割と揃える（native 1024 MiB / wasm 128 MiB）。native 既定では
// N が 4 万程度まで 1 チャンクのままなので、実質ブラウザ経路のための分割である。
// 分割しても数値が変わらない最小のチャンク幅。これを下回ると行列積のカーネル選択が
// 変わり、結果がビット一致しなくなる。実測（2026-09-09, Metal, f16, tex flow N=17489、
// 分割前との max|d|）: 末尾 1 -> 6.15e-04 / 2 -> 5.81e-04 / 8 -> 8.89e-04 / 9 -> 0 /
// 2180 -> 0。Metal は列数 2〜8 に専用カーネル、9 以上で matrix-matrix 経路を選ぶ
// （thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp:2057-2078, 2164-2167）。
static constexpr int64_t kMinMlpChunkTokens = 9;

static T* mlp_chunked(ggml_context* c, const Model& m, const std::string& pre, T* x) {
    const int64_t d_model = x->ne[0], N = x->ne[1];
    // 既定は分割しない。ブラウザ（WebGPU / NOFA 経路）では分割しても活性化バッファが
    // 1 バイトも減らないことを実測で確認しているので、既定で有効にすると benefit 0 に
    // 対してグラフだけが変わる。実測（N=12083 / 17489、attn 8〜1024 MiB × MLP 32〜1024
    // MiB のすべての組で 705.1 / 1020.5 MiB から動かない）。効くのは FlashAttention 経路
    // だけで、そこでは N=17489 で 1850.9 -> 1663.1 MiB。診断・native 最適化用の opt-in と
    // して置く。docs/design/2026-09-09-texture-flow-mlp-chunking.md 参照。
    const char* env_mb  = getenv("TRELLIS_MLP_CHUNK_MB");
    const char* env_tok = getenv("TRELLIS_MLP_CHUNK_TOKENS");
    int64_t nt = N;                                             // = 分割しない
    if (env_mb) {
        const int64_t budget = std::max<int64_t>(1, atoll(env_mb)) * 1024 * 1024;
        const int64_t per_token = 4 * d_model * 4;              // 中間 [4*d_model, 1] の f32 バイト数
        nt = std::max<int64_t>(1, budget / std::max<int64_t>(per_token, 1));
        // sdpa と同じ理由でチャンク数に上限を置く（1 チャンク約 8 ノード、30 ブロックが 1 グラフ）。
        // atoll は 0 や負値も返すので clamp する（そのまま割ると 0 除算になる）。
        static const int64_t kMaxMlpChunks = []() -> int64_t {
            if (const char* e = getenv("TRELLIS_MLP_MAX_CHUNKS")) return std::max<int64_t>(1, atoll(e));
            return 256;
        }();
        if (nt * kMaxMlpChunks < N) nt = (N + kMaxMlpChunks - 1) / kMaxMlpChunks;
        if (nt < N) {                                           // チャンク幅を均す
            const int64_t k = (N + nt - 1) / nt;
            nt = (N + k - 1) / k;
            if (N >= kMinMlpChunkTokens) nt = std::max(nt, kMinMlpChunkTokens);
        }
    }
    // 検証用のみ: チャンク幅をトークン数で直接指定し、均し処理も下限も迂回する。
    // 末尾を狙って 1 / 2 / 8 / 9 トークンにしてカーネル分岐を再現するために使う
    // （＝意図的にビット不一致を作れる）。無言で挙動が変わらないよう一度だけログに出す。
    if (env_tok) {
        nt = std::max<int64_t>(1, atoll(env_tok));
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[dit] TRELLIS_MLP_CHUNK_TOKENS=%s -- 検証専用。チャンク幅の均しと "
                            "最小幅 %lld を迂回するので、結果が分割前とビット一致しなくなりうる\n",
                    env_tok, (long long)kMinMlpChunkTokens);
        }
    }
    if (nt >= N) {                                              // 1 チャンク: 分割前と同じグラフ
        T* y = lin(c, m, pre + ".0", x);
        y = ggml_gelu(c, y);                                    // GELU(approximate=tanh)
        return lin(c, m, pre + ".2", y);
    }
    T* out = nullptr;
    for (int64_t t0 = 0; t0 < N; ) {
        int64_t n = std::min(nt, N - t0);
        // 均しだけでは末尾が最小幅を割ることがある（総当たりで N=1436 / 1 MiB -> 末尾 8 など
        // 8820 組）。ceil(N/k) は既に不動点なので均しを繰り返しても消えない。残りが最小幅に
        // 満たないならこのチャンクに吸収して、どのチャンクも下限を割らないようにする。
        if (!env_tok && N - t0 - n > 0 && N - t0 - n < kMinMlpChunkTokens) n = N - t0;
        T* xc = ggml_cont(c, ggml_view_2d(c, x, d_model, n, x->nb[1], (size_t)t0 * x->nb[1]));
        T* y = lin(c, m, pre + ".0", xc);                       // [4*d_model, n]
        y = ggml_gelu(c, y);
        y = lin(c, m, pre + ".2", y);                           // [d_model, n]
        out = out ? ggml_concat(c, out, y, 1) : y;
        t0 += n;
    }
    return out;                                                 // [d_model, N]
}

// SDPA over heads. q:[hd,nh,Lq]  k,v:[hd,nh,Lk] -> [d_model, Lq].  `mask`: optional [Lk_pad,Lq] F16.
// `tag` names the attention ("blocks.<i>.self_attn" / "...cross_attn...") for the FA range probes.
static T* sdpa(ggml_context* c, T* q, T* k, T* v, int d_model, T* mask = nullptr, const std::string& tag = "") {
    const float scale = 1.0f / std::sqrt((float)q->ne[0]);
    // FlashAttention: a fused, tiled SDPA that never materialises the [Lk,Lq,nh] score
    // matrix — O(N) memory instead of O(N^2). That score buffer is exactly what OOMs the
    // 1024 cascade (~18 GB single alloc); FA keeps it small so 1024 fits the 16 GB card.
    // ggml_flash_attn_ext wants [head_dim, n_tok, n_head, n_batch] with low-prec K/V.
    // Two HR-scale gotchas, both fatal to the 1024 cascade (NaN SLAT -> decoder collapse):
    //  (1) KV-length padding. ggml's CUDA FA reads keys in tiles of FATTN_KQ_STRIDE=256.
    //      With a null mask and an unpadded last tile (Lk % 256 != 0), the tail positions
    //      are uninitialised garbage that poison the softmax -> ~NaN. The LR flow (8.5k tok)
    //      happened to dodge it; the HR flow (52.7k tok) hits it ~70%. Fix: **zero-pad K/V's
    //      key dim up to a 256 multiple** — zero keys add exp(0-rowmax)~0 (numerically exact),
    //      remove the garbage tile, AND unlock the aligned (Lk%256==0) kernel path.
    //  (2) Use **BF16** (not F16) for K/V -- on CUDA. CUDA's tensor-core FA kernels still convert
    //      BF16 K/V to F16 internally, so V is power-of-two scaled before the cast and the output
    //      is scaled back afterward. Attention is linear in V, and this keeps large HR activations
    //      inside F16's +-65504 range without changing the softmax. Q and K never need this: both
    //      go through rms_gamma (per-head RMSNorm x gamma) right before, so |q_i|, |k_i| <=
    //      sqrt(head_dim) * max|gamma| (~206 for the shape flows, tools/fa_gamma_bound.py).
    // The K/V storage type is --fa-kv (dit_fa_kv_effective): bf16 (default everywhere), f16, f32.
    // METAL IS DIFFERENT (ggml-metal.metal FA_TYPES* macros, read 2026-09-20): the Metal backend
    // never reads ggml_flash_attn_ext_get_prec, so set_prec is a no-op there, and the kernel's
    // output accumulator type follows the K/V type -- bf16 K/V: q stored as bfloat (7-bit mantissa)
    // and the running softmax-weighted V sum kept in HALF across all KV tiles (the same "F16
    // accumulation stagnates" mechanism described below for CUDA TILE); f16 K/V: q as half (10-bit)
    // and the sum in FLOAT. So on Metal f16 K/V is the numerically better choice, with V_SCALE
    // covering the only range risk. See docs/design/2026-09-20-metal-fa-f16-kv.md for the gates.
    // set_prec(GGML_PREC_F32) below is LOAD-BEARING on CUDA, and only became so in ggml 2d6d0b0c.
    // Before it, only fattn-wmma-f16.cu read ggml_flash_attn_ext_get_prec; TILE (what this
    // shape gets on gfx1151), VEC and MMA silently ignored it and accumulated VKQ in half2.
    // Summing ~15k weighted V terms in F16 stagnates -- small addends round away once the
    // running sum is large -- biasing low, growing with KV tile count, and it wrecked the
    // latent: mean -0.8048 vs the reference's -0.0353+-0.0064 (~120 sigma out, n=5 a side).
    // With the fix TILE honours prec and the bias drops to -0.00235, matching the exact path.
    // Do NOT assume a stock ggml has this. To check: GGML_FA_DEBUG=1 prints the kernel and
    // prec, then run trellis-test-shape-flow twice and compare its `output` mean-delta with
    // FA against TRELLIS_NOFA=1 (the exact-SDPA oracle). The canary is "FA tracks the oracle
    // ON THIS BOX", not any fixed number -- the absolute values are backend-specific because
    // the severity is: gfx1151/TILE reads -0.0174 broken vs -0.0024 fixed, while NVIDIA/MMA
    // reads only -0.0019 vs -0.00014 (its oracle is -0.00010). MMA already accumulated KQ in
    // FP32, so only the VKQ sum stagnated there -- same bug, ~9x milder.
    // --no-fa falls back to the exact chunked path (correct on any backend, ~2.7x slower).
    const bool no_fa = g_no_fa;
    if (!no_fa) {
        const ggml_type kv_type = s_fa_kv_build == FA_KV_F16 ? GGML_TYPE_F16
                                : s_fa_kv_build == FA_KV_F32 ? GGML_TYPE_F32 : GGML_TYPE_BF16;
        // Range probes (TRELLIS_DBG_FA_RANGE): the F32 tensors right before the cast, per attention.
        auto probe = [&](const char* what, T* t) {
            if (s_inter_build && !tag.empty() && s_fa_range.find(what[4]) != std::string::npos) {  // what = ".fa_q" etc.
                const std::string nm = tag + what;
                (*s_inter_build)[nm] = t; ggml_set_name(t, nm.c_str());
            }
            return t;
        };
        const int64_t KQ_STRIDE = 256;
        auto prep_kv = [&](T* x, const char* what) {           // -> [hd, Lk_pad, nh] kv_type
            T* p = ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3));   // [hd, Lk, nh] F32
            const int64_t pad = (KQ_STRIDE - (p->ne[1] % KQ_STRIDE)) % KQ_STRIDE;
            if (pad) p = ggml_pad(c, p, 0, (int)pad, 0, 0);        // zero-pad key dim
            probe(what, p);
            return kv_type == GGML_TYPE_F32 ? p : ggml_cast(c, p, kv_type);
        };
        T* qf = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [hd, Lq, nh]
        if (qf->type != GGML_TYPE_F32) qf = ggml_cast(c, qf, GGML_TYPE_F32);
        probe(".fa_q", qf);
        T* kf = prep_kv(k, ".fa_k");
        constexpr float V_SCALE = 1.0f / 256.0f;
        T* vf = prep_kv(ggml_scale(c, v, V_SCALE), ".fa_v");   // probe sees the SCALED V (what is cast)
        // TRELLIS_FA_NOMASK=1: drop the mask. If the output is UNCHANGED, the mask is being
        // ignored and the zero-padded keys are diluting the softmax (exp(0-rowmax) is only
        // negligible when rowmax >> 0), which shrinks every output toward zero.
        static const bool fa_nomask = std::getenv("TRELLIS_FA_NOMASK") != nullptr;
        T* out = ggml_flash_attn_ext(c, qf, kf, vf, fa_nomask ? nullptr : mask, scale, 0.0f, 0.0f);  // [hd, nh, Lq]
        // F32 accumulation for bf16 / f32 K/V, as before. f16 K/V keeps the historical
        // TRELLIS_FA_FAST meaning on CUDA / Vulkan ("f16 K/V + f16 accumulate", the shapes the Vulkan
        // coopmat shaders are specialised for; Vulkan: f32acc = prec==F32 || K is BF16). Metal ignores
        // prec entirely (see above), so there f16 K/V always accumulates in float regardless.
        if (kv_type != GGML_TYPE_F16) ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        out = ggml_scale(c, out, 1.0f / V_SCALE);
        return ggml_reshape_2d(c, out, d_model, out->ne[2]);   // [d_model, Lq]
    }
    // Exact SDPA, chunked over QUERIES. The whole reason FA exists here is the [Lk, Lq, nh]
    // score matrix -- at the HR flow that is 15104*15006*12*4 = 10.9 TB, so it cannot be
    // materialised whole. But attention is independent per query: a query range needs no halo
    // (it reads all of K/V, which stay whole), so the scores can be built a slice at a time and
    // the peak is [Lk, nq, nh] instead. K/V are the UNPADDED originals, so no pad mask is
    // needed and none of FA's tile machinery is involved -- this is the path the golden-tensor
    // test matches to 8e-5 on the output mean.
    T* q2 = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));       // [hd, Lq, nh]
    T* k2 = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));       // [hd, Lk, nh]
    T* v2 = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));       // [Lk, hd, nh]
    const int64_t hd = q2->ne[0], Lq = q2->ne[1], nh = q2->ne[2], Lk = k2->ne[1];

    int64_t budget = kAttnChunkBytes;
    if (const char* e = getenv("TRELLIS_ATTN_CHUNK_MB")) budget = std::max<int64_t>(1, atoll(e)) * 1024 * 1024;
    const int64_t per_q = Lk * nh * 4;                          // one query's score column
    int64_t nq = std::max<int64_t>(1, budget / std::max<int64_t>(per_q, 1));
    // Floor on the chunk size: each chunk adds ~8 nodes and a whole 30-block DiT with two
    // attentions per block is built as ONE graph, so an unbounded chunk count exhausts the
    // ggml context (ggml_new_object: not enough space). Hitting this cap costs memory, not
    // correctness -- query chunking is bit-exact either way (no reduction crosses queries).
    // 上限は「1 グラフに入るテンソル数」から来る制約で、正しさとは無関係（クエリ分割は
    // どちらでもビット完全）。ブラウザでは活性化バッファが WebGPU の予算に収まるかどうかが
    // 死活問題なので、flow_runner.cpp のメタデータ枠を広げたうえでここも上げてある。
    // 実測（tex flow, N=17690, n_heads=12）: 上限 32 だと 1 チャンク 553 クエリ =
    // スコア 470 MB で頭打ちになり、活性化バッファは 1316 MB より下がらなかった。
    // 戻り型は明示する。atoll() は long long、(int64_t)256 は Linux/GCC では long なので
    // 推論に任せると "inconsistent types deduced for lambda return type" で落ちる（macOS
    // clang は int64_t が long long なので通ってしまい、Linux ビルドでだけ露見する）。
    static const int64_t kMaxAttnChunks = []() -> int64_t {
        // atoll は 0 や負値も返す。そのまま割ると 0 除算になるので clamp する。
        if (const char* e = getenv("TRELLIS_ATTN_MAX_CHUNKS")) return std::max<int64_t>(1, atoll(e));
        return 256;
    }();
    if (nq * kMaxAttnChunks < Lq) nq = (Lq + kMaxAttnChunks - 1) / kMaxAttnChunks;
    if (nq >= Lq) nq = Lq;                                      // small attn: single chunk, no concat

    T* out = nullptr;
    for (int64_t q0 = 0; q0 < Lq; q0 += nq) {
        const int64_t n = std::min(nq, Lq - q0);
        T* qc = (n == Lq) ? q2 : ggml_cont(c, ggml_view_3d(c, q2, hd, n, nh,
                                q2->nb[1], q2->nb[2], (size_t)q0 * q2->nb[1]));   // [hd, n, nh]
        T* kq = ggml_mul_mat(c, k2, qc);                        // [Lk, n, nh]
        kq = ggml_soft_max_ext(c, kq, nullptr, scale, 0.0f);
        T* kqv = ggml_mul_mat(c, v2, kq);                       // [hd, n, nh]
        kqv = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3));   // [hd, nh, n]
        T* o = ggml_reshape_2d(c, kqv, d_model, n);             // [d_model, n]
        out = out ? ggml_concat(c, out, o, 1) : o;
    }
    return out;                                                 // [d_model, Lq]
}

static T* gamma32(ggml_context* c, const Model& m, const std::string& key) {
    T* g = m.get(key);
    return g->type == GGML_TYPE_F32 ? g : ggml_cast(c, g, GGML_TYPE_F32);
}

static T* self_attn(ggml_context* c, const Model& m, const std::string& pre, T* h,
                    T* cos, T* sin, const DiTParams& p, T* mask = nullptr, T* rope_idx = nullptr) {
    const int hd = p.head_dim, nh = p.n_heads;
    const int64_t L = h->ne[1];
    T* qkv = lin(c, m, pre + ".to_qkv", h);                     // [3*d_model, L]
    qkv = ggml_reshape_4d(c, qkv, hd, nh, 3, L);
    auto pick = [&](int s) {
        T* t = ggml_view_4d(c, qkv, hd, nh, 1, L, qkv->nb[1], qkv->nb[2], qkv->nb[3], s * qkv->nb[2]);
        return ggml_reshape_3d(c, ggml_cont(c, t), hd, nh, L);
    };
    T* q = pick(0); T* k = pick(1); T* v = pick(2);
    q = rms_gamma(c, q, gamma32(c, m, pre + ".q_rms_norm.gamma"), p.rms_eps);
    k = rms_gamma(c, k, gamma32(c, m, pre + ".k_rms_norm.gamma"), p.rms_eps);
    q = apply_rope(c, q, cos, sin, rope_idx);
    k = apply_rope(c, k, cos, sin, rope_idx);
    return lin(c, m, pre + ".to_out", sdpa(c, q, k, v, p.d_model, mask, pre));
}

static T* cross_attn(ggml_context* c, const Model& m, const std::string& pre, T* h, T* cond,
                     const DiTParams& p, T* mask = nullptr) {
    const int hd = p.head_dim, nh = p.n_heads;
    const int64_t L = h->ne[1], Lc = cond->ne[1];
    T* q = lin(c, m, pre + ".to_q", h);
    q = ggml_reshape_3d(c, q, hd, nh, L);
    T* kv = lin(c, m, pre + ".to_kv", cond);                    // [2*d_model, Lc]
    kv = ggml_reshape_4d(c, kv, hd, nh, 2, Lc);
    auto pick = [&](int s) {
        T* t = ggml_view_4d(c, kv, hd, nh, 1, Lc, kv->nb[1], kv->nb[2], kv->nb[3], s * kv->nb[2]);
        return ggml_reshape_3d(c, ggml_cont(c, t), hd, nh, Lc);
    };
    T* k = pick(0); T* v = pick(1);
    q = rms_gamma(c, q, gamma32(c, m, pre + ".q_rms_norm.gamma"), p.rms_eps);
    k = rms_gamma(c, k, gamma32(c, m, pre + ".k_rms_norm.gamma"), p.rms_eps);
    return lin(c, m, pre + ".to_out", sdpa(c, q, k, v, p.d_model, mask, pre));
}

// x*(1+scale)+shift, scale/shift: [d_model] broadcast over L
static T* modulate(ggml_context* c, T* x, T* scale, T* shift) {
    return ggml_add(c, ggml_add(c, x, ggml_mul(c, x, scale)), shift);
}

static T* block(ggml_context* c, const Model& m, int i, T* h, T* mod, T* cond,
                T* cos, T* sin, const DiTParams& p, std::map<std::string, T*>* inter = nullptr,
                T* self_mask = nullptr, T* cross_mask = nullptr, T* proj = nullptr,
                T* rope_idx = nullptr) {
    const std::string b = "blocks." + std::to_string(i);
    const int dm = p.d_model;
    // Block 0 keeps the historical "blk0_*" names; block 15 is exposed too as a mid-depth probe.
    auto dbg = [&](const char* n, T* t) {
        if (inter && (i == 0 || i == 15)) {
            std::string nm = i == 0 ? std::string(n) : "blk15_" + std::string(n + 5);
            (*inter)[nm] = t; ggml_set_name(t, nm.c_str());
        }
        return t;
    };
    T* mb = ggml_add(c, m.get(b + ".modulation"), mod);        // [6*d_model]
    auto ch = [&](int j) { return ggml_view_1d(c, mb, dm, (size_t)j * dm * ggml_element_size(mb)); };
    T* shift_msa = ch(0), *scale_msa = ch(1), *gate_msa = ch(2);
    T* shift_mlp = ch(3), *scale_mlp = ch(4), *gate_mlp = ch(5);

    T* hh = layernorm(c, h, p.ln_eps);
    hh = modulate(c, hh, scale_msa, shift_msa);
    hh = self_attn(c, m, b + ".self_attn", hh, cos, sin, p, self_mask, rope_idx);
    dbg("blk0_msa", hh);
    h = ggml_add(c, h, ggml_mul(c, hh, gate_msa));
    // Role markers for --profile (flow_runner.cpp). Naming does not change the graph; the
    // profiler attributes every node to the first marker at or after it in graph order, which
    // is exactly the section that built it (ggml_build_forward_expand is a post-order DFS).
    // The MLP end is already marked by build_dit_dense's keep("after_block<i>").
    ggml_set_name(h, (b + ".res_msa").c_str());

    hh = layernorm(c, h, p.ln_eps, m.get(b + ".norm2.weight"), m.get(b + ".norm2.bias"));
    // Pixal3D ProjectAttention: the ordinary cross-attn weights move one level deeper
    // (blocks.N.cross_attn.cross_attn_block.*) alongside a sibling proj_linear.
    const std::string cross_pre = p.proj_attn ? (b + ".cross_attn.cross_attn_block") : (b + ".cross_attn");
    T* global_out = cross_attn(c, m, cross_pre, hh, cond, p, cross_mask);
    dbg("blk0_global_out", global_out);
    hh = global_out;
    if (p.proj_attn && proj) {
        T* proj_out = lin(c, m, b + ".cross_attn.proj_linear", proj);
        dbg("blk0_proj_out", proj_out);
        hh = ggml_add(c, hh, proj_out);
    }
    dbg("blk0_cross_out", hh);
    dbg("blk0_cross", hh);   // kept for TRELLIS_DBG_NAN's existing name lookup
    h = ggml_add(c, h, hh);
    ggml_set_name(h, (b + ".res_cross").c_str());

    hh = layernorm(c, h, p.ln_eps);
    hh = modulate(c, hh, scale_mlp, shift_mlp);
    hh = mlp_chunked(c, m, b + ".mlp.mlp", hh);
    dbg("blk0_mlp", hh);
    h = ggml_add(c, h, ggml_mul(c, hh, gate_mlp));
    return h;
}

bool dit_detect_proj_attn(const Model& m, DiTParams& p) {
    T* w = m.try_get("blocks.0.cross_attn.proj_linear.weight");
    if (!w) { p.proj_attn = false; p.d_proj = 0; return false; }
    p.proj_attn = true;
    p.d_proj = (int)w->ne[0];
    return true;
}

ggml_tensor* dit_rope(ggml_context* gctx, ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin) {
    return apply_rope(gctx, x, cos, sin, nullptr);
}

void dit_rope_index(int head_dim, std::vector<int32_t>& out) {
    out.resize(head_dim);
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) { out[i] = 2 * i; out[half + i] = 2 * i + 1; }
}

ggml_tensor* dit_sdpa(ggml_context* c, ggml_backend* backend, T* q, T* k, T* v, int d_model) {
    s_fa_kv_build = dit_fa_kv_effective(backend);
    s_inter_build = nullptr;
    T* mask = build_pad_mask(c, k->ne[2], q->ne[2]);
    return sdpa(c, q, k, v, d_model, mask, "");
}

ggml_tensor* build_dit_dense(ggml_context* c, const Model& m, const DiTParams& p,
                             T* h0, T* tfreq, T* cond, T* cos, T* sin,
                             std::map<std::string, T*>* inter, T* proj, T* rope_idx) {
    g_cast_f32 = p.cast_f32;
    s_fa_kv_build = dit_fa_kv_effective(m.backend);
    s_inter_build = inter;
    auto keep = [&](const char* n, T* t) { if (inter) (*inter)[n] = t; ggml_set_name(t, n); return t; };

    T* h = lin(c, m, "input_layer", h0);                       // [d_model, L]
    keep("after_input_layer", h);

    T* te = lin(c, m, "t_embedder.mlp.0", tfreq);
    te = ggml_silu(c, te);
    te = lin(c, m, "t_embedder.mlp.2", te);                    // [d_model]
    T* mod = lin(c, m, "adaLN_modulation.1", ggml_silu(c, te));// [6*d_model]
    keep("t_emb_mod", mod);

    // Padding masks for the two attentions, built ONCE (token counts are fixed across blocks) and
    // shared by every block — they tell the CUDA FlashAttention to exclude the zero-padded key tiles
    // (else the >=1024-token flow NaNs a subset of queries). self: Lk=L (the latent); cross: Lk=Lc.
    T* self_mask  = build_pad_mask(c, h0->ne[1], h0->ne[1]);
    T* cross_mask = build_pad_mask(c, cond->ne[1], h0->ne[1]);
    for (int i = 0; i < p.n_blocks; ++i) {
        h = block(c, m, i, h, mod, cond, cos, sin, p, inter, self_mask, cross_mask, proj, rope_idx);
        keep(("after_block" + std::to_string(i)).c_str(), h);
    }
    h = layernorm(c, h, p.final_ln_eps);
    keep("prefinal", h);
    h = lin(c, m, "out_layer", h);                             // [out_ch, L]
    keep("output", h);
    return h;
}

} // namespace trellis
