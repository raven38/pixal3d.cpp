#include "flow_runner.h"
#include "trellis_model.h"
#include "graph_dump.h"
#include "trellis_args.h"   // g_profile / g_no_fa
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef TRELLIS_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace trellis {

static bool stdout_is_tty() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

static void timestep_embedding(float t, std::vector<float>& out) {
    out.resize(256);
    for (int j = 0; j < 128; ++j) {
        float f = std::exp(-std::log(10000.f) * j / 128.f);
        out[j] = std::cos(t * f);
        out[128 + j] = std::sin(t * f);
    }
}

// グラフを組んだ直後に、このデバイスで実際に回せるかを判定する。
//
// 2026-09-08 に踏んだ事故の再発防止: WebGPU のデバイス予算は実測 4095 MB しかなく、
// テクスチャ flow を N=17690 で回すと 重み 2647 + 活性化 1888 + cond 276 = 4818 MB で
// 超過する。超過しても確保は成功してしまい、その後ユニファイドメモリ上で Metal が
// メモリを往復させ続ける。1 forward が 40 秒から 139〜176 秒に伸び、その帯域を
// WindowServer ごと奪ってマシン全体が固まった（復旧に再起動を要した）。
// 「走らせてから固まる」を避けるため、走らせる前に落とす。
//
// 必要量は N にほぼ比例する（実測 75.7 KB/token、内訳は活性化 59.7 + cond 16.0）。
// 予算 4095 MB では N ≒ 19600 が上限。詳細は docs/PIXAL3D_WEBGPU_MEMORY.md。
void DitRunner::check_device_budget() const {
    ggml_backend_dev_t dev = ggml_backend_get_device(m_.backend);
    if (!dev) return;
    size_t dev_free_sz = 0, dev_total_sz = 0;
    ggml_backend_dev_memory(dev, &dev_free_sz, &dev_total_sz);
    // dev_total も 64 bit で持つ。wasm32 では size_t が 32 bit なので、下の上書きで
    // TRELLIS_DEVICE_BUDGET_MB=4096 がちょうど 2^32 になり 0 に落ちる。0 になると
    // 直後の early return でゲートが丸ごと無効化される（4096 超も周回して過小予算になる）。
    // 4095 MB がブラウザの実測予算なので、この境界は実運用の値そのもの。
    uint64_t dev_total = dev_total_sz;
    (void)dev_free_sz;
    // ブラウザの予算（実測 4095 MB）を native から模擬してゲート自体を検証するための上書き。
    // ブラウザを起動せずに「この N はブラウザで通るか」を native で判定できる。
    if (const char* e = getenv("TRELLIS_DEVICE_BUDGET_MB")) dev_total = (uint64_t)std::max<int64_t>(0, atoll(e)) * 1048576ull;
    if (dev_total == 0) return;                       // 報告しない backend（CPU 等）は素通り

    // cond は forward ごとに再アップロードされ、negative 側と 2 本同時に載る。
    // 合計は uint64_t で持つ。wasm32 では size_t が 32 bit なので、ちょうどこのゲートが
    // 効いてほしい 4 GiB 超で加算が周回し、超過を「収まっている」と誤判定する。
    const uint64_t cond_bytes = p_.proj_attn ? (uint64_t)p_.d_proj * N_ * 4 * 2 : 0;
    const uint64_t need = (uint64_t)m_.total_bytes() + (uint64_t)alloc_bytes_ + cond_bytes;
    const double MB = 1.0 / 1048576.0;
    const bool over = need > dev_total;
    if (over || getenv("TRELLIS_DBG_BUDGET"))
        fprintf(stderr,
                "[budget] N=%d  weights %.0f + activations %.0f + cond %.0f = %.0f MB "
                "/ device %.0f MB%s\n",
                N_, m_.total_bytes() * MB, alloc_bytes_ * MB, cond_bytes * MB, need * MB,
                dev_total * MB, over ? "  ** OVER **" : "");
    if (!over) return;
    if (getenv("TRELLIS_ALLOW_OVER_BUDGET")) {        // 意図的に踏むとき用の逃がし弁
        fprintf(stderr, "[budget] TRELLIS_ALLOW_OVER_BUDGET が設定されているので続行する\n");
        return;
    }
    char msg[640];
    snprintf(msg, sizeof msg,
             "DitRunner: this token count does not fit the device memory budget. "
             "N=%d needs %.0f MB (weights %.0f + activations %.0f + cond %.0f) "
             "but the device reports %.0f MB. Running anyway thrashes unified memory and "
             "can hang the whole machine. Options: lower TRELLIS_ATTN_CHUNK_MB (activations "
             "scale with it on the non-FlashAttention path this backend uses), reduce N, "
             "quantize the flow weights, or set TRELLIS_ALLOW_OVER_BUDGET=1 to override. "
             "TRELLIS_MLP_CHUNK_MB does NOT help here -- measured zero effect on the "
             "non-FA/WebGPU path; it only lowers the peak on the FlashAttention path.",
             N_, need * MB, m_.total_bytes() * MB, alloc_bytes_ * MB, cond_bytes * MB, dev_total * MB);
    throw std::runtime_error(msg);
}

DitRunner::DitRunner(const Model& m, const DiTParams& p, int N, int n_cond,
                     const std::vector<float>& rcos, const std::vector<float>& rsin)
    : m_(m), p_(p), N_(N), Lc_(n_cond) {
    const int half = p_.head_dim / 2;
    // Pixal3D ProjectAttention uses five global DINO tokens whose value is constant
    // for the entire flow. Opt out for parity/perf A/B or unusual callers that mutate
    // conditioning in place while keeping the same pointer.
    use_cross_kv_cache_ = p_.proj_attn && std::getenv("TRELLIS_DISABLE_CROSS_KV_CACHE") == nullptr;
#ifdef TRELLIS_USE_CUDA
    // The prepacked src1 path is intentionally narrow: CUDA + Q8_0 ProjectAttention only.
    // F16 models and every other backend keep the existing F32 input path.
    if (p_.proj_attn && std::getenv("TRELLIS_DISABLE_PROJ_Q8_CACHE") == nullptr &&
        ggml_backend_is_cuda(m_.backend)) {
        ggml_tensor* pw = m_.try_get("blocks.0.cross_attn.proj_linear.weight");
        use_proj_q8_cache_ = pw && pw->type == GGML_TYPE_Q8_0;
    }
#endif
    // sdpa のクエリ分割はチャンク 1 本につき約 8 ノードを足す。30 ブロック x 2 attention で
    // 1 グラフに載るので、チャンク数を増やすとここが先に枯れる（ggml_new_object: not enough
    // space）。枠はホスト側のメタデータだけで 1 テンソル約 368 B なので、広げても数十 MB。
    // 活性化バッファ（GPU 側、ブラウザでは 4 GB 予算）を削るほうが遥かに重要。
    size_t meta = ggml_tensor_overhead() * 131072 + ggml_graph_overhead_custom(262144, false) + (1 << 20);
    ctx_ = ggml_init({ meta, nullptr, true });
    gh0_  = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, p_.in_ch, N_);   ggml_set_input(gh0_);
    gtf_  = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 256);            ggml_set_input(gtf_);
    gcond_= ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, p_.d_cond, Lc_); ggml_set_input(gcond_);
    gcos_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, 1, half, 1, N_); ggml_set_input(gcos_);
    gsin_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, 1, half, 1, N_); ggml_set_input(gsin_);
    if (p_.proj_attn) {
        if (use_proj_q8_cache_) {
            // ggml MMQ intentionally overreads the tail of the final Y tile by up to
            // get_mmq_x_max (128) block_q8_1_mmq records. Give the backing input
            // real guard storage and expose only the first N logical columns to DiT.
            constexpr int kProjQ8GuardCols = 128;
            gproj_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_Q8_1, p_.d_proj, N_ + kProjQ8GuardCols);
            ggml_set_input(gproj_);
            gproj_view_ = ggml_view_2d(ctx_, gproj_, p_.d_proj, N_, gproj_->nb[1], 0);
        } else {
            gproj_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, p_.d_proj, N_);
            ggml_set_input(gproj_);
            gproj_view_ = gproj_;
        }
    }
    if (use_cross_kv_cache_) {
        gcross_kv_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32,
                                        2 * p_.d_model, Lc_, p_.n_blocks);
        ggml_set_input(gcross_kv_);
    }
    dbg_nan_ = std::getenv("TRELLIS_DBG_NAN") != nullptr;
    gout_ = build_dit_dense(ctx_, m_, p_, gh0_, gtf_, gcond_, gcos_, gsin_,
                            dbg_nan_ ? &inter_ : nullptr, gproj_view_, gcross_kv_);
    g_ = ggml_new_graph_custom(ctx_, 262144, false);
    ggml_build_forward_expand(g_, gout_);
    ggml_set_output(gout_);
    if (dbg_nan_) for (auto& [nm, t] : inter_) { ggml_build_forward_expand(g_, t); ggml_set_output(t); }
    const std::string tag = "dit_N" + std::to_string(N_) + "_dcond" + std::to_string(Lc_) + "_proj" + std::to_string((int)p_.proj_attn);
    trellis_graph_dump(tag.c_str(), g_);
    check_graph_supported(m_.backend, g_, tag.c_str());
    alloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m_.backend));
    if (!ggml_gallocr_alloc_graph(alloc_, g_)) throw std::runtime_error("DitRunner: alloc failed");
    alloc_bytes_ = ggml_gallocr_get_buffer_size(alloc_, 0);

    if (use_cross_kv_cache_) {
        // Separate tiny graph: [d_cond,Lc] -> concat_i to_kv_i(cond).
        // It runs only on the first use of each distinct condition (normally pos/neg).
        const size_t kv_meta = ggml_tensor_overhead() * 512
                             + ggml_graph_overhead_custom(2048, false) + (1 << 20);
        kv_ctx_ = ggml_init({ kv_meta, nullptr, true });
        if (!kv_ctx_) throw std::runtime_error("DitRunner: cross-KV context alloc failed");
        kv_cond_ = ggml_new_tensor_2d(kv_ctx_, GGML_TYPE_F32, p_.d_cond, Lc_);
        ggml_set_input(kv_cond_);
        kv_out_ = build_dit_cross_kv_cache(kv_ctx_, m_, p_, kv_cond_);
        ggml_set_output(kv_out_);
        kv_g_ = ggml_new_graph_custom(kv_ctx_, 2048, false);
        ggml_build_forward_expand(kv_g_, kv_out_);
        check_graph_supported(m_.backend, kv_g_, "dit_cross_kv_cache");
        kv_alloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m_.backend));
        if (!ggml_gallocr_alloc_graph(kv_alloc_, kv_g_))
            throw std::runtime_error("DitRunner: cross-KV graph alloc failed");
    }

    check_device_budget();
    rcos_ = rcos; rsin_ = rsin;   // keep; re-upload each forward (gallocr reuses input buffers across runs)
}

DitRunner::~DitRunner() {
    for (auto& e : proj_q8_cache_) {
        if (e.buffer) ggml_backend_buffer_free(e.buffer);
        if (e.ctx) ggml_free(e.ctx);
    }
    if (kv_alloc_) ggml_gallocr_free(kv_alloc_);
    if (kv_ctx_)   ggml_free(kv_ctx_);
    if (alloc_) ggml_gallocr_free(alloc_);
    if (ctx_)   ggml_free(ctx_);
}

ggml_tensor* DitRunner::proj_q8_for(const float* proj) {
#ifdef TRELLIS_USE_CUDA
    for (auto& e : proj_q8_cache_) if (e.key == proj) return e.packed;
    if (!use_proj_q8_cache_ || !proj) throw std::runtime_error("DitRunner: invalid proj Q8 cache request");

    // Persistent destination: logical Q8_1 backing with 128 guard columns.
    // The prepacker writes only the first N columns in MMQ-private block-major
    // layout and zeroes the tail; the extra allocation makes the final MMQ
    // tile's intentional vector overread memory-safe.
    constexpr int kProjQ8GuardCols = 128;
    ProjQ8Entry e;
    e.key = proj;
    const size_t meta = ggml_tensor_overhead() * 2 + 1024;
    e.ctx = ggml_init({ meta, nullptr, true });
    if (!e.ctx) throw std::runtime_error("DitRunner: proj Q8 cache context alloc failed");
    e.packed = ggml_new_tensor_2d(e.ctx, GGML_TYPE_Q8_1, p_.d_proj, N_ + kProjQ8GuardCols);
    e.buffer = ggml_backend_alloc_ctx_tensors(e.ctx, m_.backend);
    if (!e.buffer) {
        ggml_free(e.ctx);
        throw std::runtime_error("DitRunner: proj Q8 cache buffer alloc failed");
    }

    // Temporary F32 source exists only for the one-time prepack.
    ggml_context* src_ctx = ggml_init({ ggml_tensor_overhead() + 512, nullptr, true });
    if (!src_ctx) throw std::runtime_error("DitRunner: proj Q8 source context alloc failed");
    ggml_tensor* src = ggml_new_tensor_2d(src_ctx, GGML_TYPE_F32, p_.d_proj, N_);
    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors(src_ctx, m_.backend);
    if (!src_buf) {
        ggml_free(src_ctx);
        throw std::runtime_error("DitRunner: proj Q8 source buffer alloc failed");
    }
    ggml_backend_tensor_set(src, proj, 0, (size_t)p_.d_proj * N_ * sizeof(float));
    const bool ok = ggml_backend_cuda_mmq_prepack_q8_1(m_.backend, src, e.packed, GGML_TYPE_Q8_0);
    ggml_backend_synchronize(m_.backend);
    ggml_backend_buffer_free(src_buf);
    ggml_free(src_ctx);
    if (!ok) {
        ggml_backend_buffer_free(e.buffer);
        ggml_free(e.ctx);
        throw std::runtime_error("DitRunner: proj Q8 MMQ prepack unsupported");
    }

    if (std::getenv("TRELLIS_DBG_CACHE"))
        fprintf(stderr, "      [cache] proj MMQ-Q8 miss -> %.2f MiB (F32 %.2f MiB, N=%d)\n",
                ggml_nbytes(e.packed) / 1048576.0,
                (double)p_.d_proj * N_ * sizeof(float) / 1048576.0, N_);
    proj_q8_cache_.push_back(e);
    return proj_q8_cache_.back().packed;
#else
    (void)proj;
    throw std::runtime_error("DitRunner: proj Q8 cache requested without CUDA");
#endif
}

const std::vector<float>& DitRunner::cross_kv_for(const float* cond) {
    for (const auto& e : cross_kv_host_) if (e.key == cond) return e.data;
    if (!cond || !kv_g_ || !kv_cond_ || !kv_out_)
        throw std::runtime_error("DitRunner: invalid cross-KV cache request");

    ggml_backend_tensor_set(kv_cond_, cond, 0, (size_t)p_.d_cond * Lc_ * sizeof(float));
    if (ggml_backend_graph_compute(m_.backend, kv_g_) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("DitRunner: cross-KV precompute failed");

    CrossKvEntry e;
    e.key = cond;
    e.data = tensor_to_f32(kv_out_);
    const size_t expected = (size_t)2 * p_.d_model * Lc_ * p_.n_blocks;
    if (e.data.size() != expected)
        throw std::runtime_error("DitRunner: cross-KV cache size mismatch");
    if (std::getenv("TRELLIS_DBG_CACHE"))
        fprintf(stderr, "      [cache] cross-KV miss -> %.2f MiB (%d blocks, Lc=%d)\n",
                e.data.size() * sizeof(float) / 1048576.0, p_.n_blocks, Lc_);
    cross_kv_host_.push_back(std::move(e));
    return cross_kv_host_.back().data;
}

std::vector<float> DitRunner::forward(const std::vector<float>& xt, float t_scaled, const float* cond,
                                      const float* proj) {
    std::vector<float> tf; timestep_embedding(t_scaled, tf);
    ggml_backend_tensor_set(gh0_,  xt.data(), 0, xt.size() * 4);
    ggml_backend_tensor_set(gtf_,  tf.data(), 0, tf.size() * 4);
    const std::vector<float>* cross_kv_data = nullptr;
    if (use_cross_kv_cache_) {
        const auto& cached = cross_kv_for(cond);
        cross_kv_data = &cached;
        ggml_backend_tensor_set(gcross_kv_, cached.data(), 0, cached.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(gcond_, cond, 0, (size_t)p_.d_cond * Lc_ * 4);
    }
    ggml_backend_tensor_set(gcos_, rcos_.data(), 0, rcos_.size() * 4);   // re-upload (buffers reused across runs)
    ggml_backend_tensor_set(gsin_, rsin_.data(), 0, rsin_.size() * 4);
    if (gproj_) {
        if (!proj) throw std::runtime_error("DitRunner: proj_attn model requires a proj tensor");
        if (use_proj_q8_cache_) {
            ggml_tensor* packed = proj_q8_for(proj);
            ggml_backend_tensor_copy_async(m_.backend, m_.backend, packed, gproj_);
        } else {
            ggml_backend_tensor_set(gproj_, proj, 0, (size_t)p_.d_proj * N_ * 4);
        }
    }
    const auto tc0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(m_.backend, g_) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("DitRunner: compute failed");
    const double tc = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
    // Read the output BEFORE any profiling pass: the passes re-submit slices of the same graph
    // into the same gallocr buffers, and the fused whole-graph result is what the sampler gets.
    std::vector<float> outv = tensor_to_f32(gout_);
    if (g_profile) {
        whole_s_.push_back(tc);
        printf("      [prof] fwd #%d whole graph: %.2fs%s\n", fwd_count_, tc,
               fwd_count_ == 0 ? "  (cold: lazy pipeline compiles)" : "");
        fflush(stdout);
        // Every profiling pass replays the graph from node 0, so it must start from the same input
        // state as the whole run: gallocr reuses the input buffers inside the graph and the whole
        // run has overwritten them by now (same reason rcos_/rsin_ are re-uploaded every forward).
        auto reupload = [&]() {
            ggml_backend_tensor_set(gh0_,  xt.data(), 0, xt.size() * 4);
            ggml_backend_tensor_set(gtf_,  tf.data(), 0, tf.size() * 4);
            if (use_cross_kv_cache_) {
                ggml_backend_tensor_set(gcross_kv_, cross_kv_data->data(), 0,
                                        cross_kv_data->size() * sizeof(float));
            } else {
                ggml_backend_tensor_set(gcond_, cond, 0, (size_t)p_.d_cond * Lc_ * 4);
            }
            ggml_backend_tensor_set(gcos_, rcos_.data(), 0, rcos_.size() * 4);
            ggml_backend_tensor_set(gsin_, rsin_.data(), 0, rsin_.size() * 4);
            if (gproj_) {
                if (use_proj_q8_cache_) {
                    ggml_tensor* packed = proj_q8_for(proj);
                    ggml_backend_tensor_copy_async(m_.backend, m_.backend, packed, gproj_);
                } else {
                    ggml_backend_tensor_set(gproj_, proj, 0, (size_t)p_.d_proj * N_ * 4);
                }
            }
        };
        if (fwd_count_ == 1) profile_forward(reupload);
    }
    ++fwd_count_;
    size_t out_bad = 0; for (float x : outv) if (!std::isfinite(x)) out_bad++;
    // Dump the per-layer breakdown for the FIRST forward whose OUTPUT goes NaN (the failing low-t
    // step), not just the very first forward (which is clean) — that's where to look for the cause.
    if (dbg_nan_ && !dbg_done_ && out_bad > 0) {
        dbg_done_ = true;
        fprintf(stderr, "      [dit-nan] *** first NaN forward: t_scaled=%.2f  out_nan=%zu/%zu ***\n",
                t_scaled, out_bad, outv.size());
        const char* order[] = { "after_input_layer", "after_block0", "after_block1",
                                "blk0_msa", "blk0_cross", "blk0_mlp",
                                "after_block29", "prefinal", "output" };
        for (const char* nm : order) {
            auto it = inter_.find(nm); if (it == inter_.end()) continue;
            std::vector<float> v = tensor_to_f32(it->second);
            size_t bad = 0; double amax = 0; for (float x : v) {
                if (std::isnan(x) || std::isinf(x)) bad++;
                else if (std::fabs(x) > amax) amax = std::fabs(x); }
            fprintf(stderr, "      [dit-nan] %-18s nan/inf=%zu/%zu  max|finite|=%.2f\n",
                    nm, bad, v.size(), amax);
        }
    }
    return outv;
}

// --profile: hierarchical timing of one forward on the already-built, already-allocated graph.
// ggml exposes no per-op GPU timestamps and DitRunner submits one graph per forward, so the
// portable way to get a breakdown is to re-submit contiguous slices of that graph and time each
// (ggml_backend_graph_compute is synchronous). Two granularities, both after the real whole
// forward whose output the sampler uses:
//   1. segments = role x block (~3 per block + input/t_emb/final). A slice keeps ggml-metal's
//      intra-slice fusion and concurrency, so segment times are close to the production cost and
//      the role / block tables come from here.
//   2. single nodes, for the op-kind table only. Every submission carries a fixed submit+wait
//      cost, so light ops (VIEW/ADD/...) are overstated relative to GEMMs -- these are
//      "isolated" numbers, not production shares. The node pass runs twice; the second is
//      reported (the first compiles the non-fused pipeline variants).
// Role attribution relies on the names dit.cpp sets: "blocks.<i>.res_msa" / "blocks.<i>.res_cross"
// (residual adds) and "after_block<i>" (build_dit_dense). A node belongs to the first marker at
// or after it in graph order, which is the section that built it (ggml_build_forward_expand
// appends in post-order). Known smear: the t_embedder/adaLN nodes and block 0's first layernorm
// are first needed by block 0's modulation and land in its self_attn segment (a few small ops).
void DitRunner::profile_forward(const std::function<void()>& reupload) {
    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a) { return std::chrono::duration<double>(clock::now() - a).count(); };
    const int n = ggml_graph_n_nodes(g_);
    enum Role { R_INPUT, R_TEMB, R_MSA, R_CROSS, R_MLP, R_FINAL, R_COUNT };
    static const char* role_name[R_COUNT] = { "input_layer", "t_emb", "self_attn", "cross_attn+proj", "mlp", "final" };
    struct Seg { int first, last; int block; Role role; double s; };
    std::vector<Seg> segs;
    auto parse_block = [](const std::string& nm, const char* pre, const char* suf, int& blk) {
        const size_t lp = strlen(pre), ls = strlen(suf);
        if (nm.size() <= lp + ls || nm.compare(0, lp, pre) != 0 || nm.compare(nm.size() - ls, ls, suf) != 0) return false;
        const std::string mid = nm.substr(lp, nm.size() - lp - ls);
        if (mid.empty() || mid.find_first_not_of("0123456789") != std::string::npos) return false;
        blk = atoi(mid.c_str());
        return true;
    };
    int first = 0;
    for (int j = 0; j < n; ++j) {
        const std::string nm = ggml_get_name(ggml_graph_node(g_, j));
        int blk = -1; Role r = R_COUNT;
        if      (nm == "after_input_layer")                     r = R_INPUT;
        else if (nm == "t_emb_mod")                             r = R_TEMB;
        else if (parse_block(nm, "blocks.", ".res_msa", blk))   r = R_MSA;
        else if (parse_block(nm, "blocks.", ".res_cross", blk)) r = R_CROSS;
        else if (parse_block(nm, "after_block", "", blk))       r = R_MLP;
        else if (nm == "output")                                r = R_FINAL;
        if (r == R_COUNT) continue;
        segs.push_back({ first, j, blk, r, 0.0 });
        first = j + 1;
    }
    if (first < n) segs.push_back({ first, n - 1, -1, R_FINAL, 0.0 });   // anything after "output"
    const int n_blocks = p_.n_blocks;
    {   // marker sanity: every block needs exactly res_msa / res_cross / after_block, in that order
        std::vector<int> seen(std::max(n_blocks, 1), 0);
        bool ok = true;
        for (const Seg& sg : segs) if (sg.block >= 0 && sg.block < n_blocks) seen[sg.block] |= 1 << (sg.role - R_MSA);
        for (int b = 0; b < n_blocks; ++b) if (seen[b] != 7) ok = false;
        if (!ok) fprintf(stderr, "      [prof] warning: role markers incomplete -- per-block table is unreliable\n");
    }

    // Scratch graph sized for the largest slice; ggml_graph_add_node appends without the DFS that
    // ggml_build_forward_expand would do (which would pull every ancestor back in).
    ggml_context* pctx = ggml_init({ ggml_graph_overhead_custom((size_t)n, false) + 4096, nullptr, true });
    ggml_cgraph* gv = ggml_new_graph_custom(pctx, (size_t)n, false);
    auto run_slice = [&](int a, int b) {           // inclusive node range -> seconds
        ggml_graph_clear(gv);
        for (int j = a; j <= b; ++j) ggml_graph_add_node(gv, ggml_graph_node(g_, j));
        const auto t = clock::now();
        if (ggml_backend_graph_compute(m_.backend, gv) != GGML_STATUS_SUCCESS) {
            ggml_free(pctx);
            throw std::runtime_error("DitRunner: compute failed (profile slice " + std::to_string(a) + ".." + std::to_string(b) + ")");
        }
        return secs(t);
    };

    // 1. segment pass (replays from node 0 -> restore the inputs first)
    reupload();
    const auto t_seg0 = clock::now();
    for (Seg& sg : segs) sg.s = run_slice(sg.first, sg.last);
    const double t_seg = secs(t_seg0);
    ggml_free(pctx);
    // 2. node passes via the shared primitive (graph_dump.h): first = warm-up for the non-fused
    //    pipelines, second = reported. Two 1-pass calls so the warm-up time can still be printed.
    std::vector<double> dt;
    double t_node = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        const auto t0 = clock::now();
        dt = trellis_graph_node_times(m_.backend, g_, 1, reupload);
        t_node = secs(t0);
        if (pass == 0) printf("      [prof] node pass 1 (warm-up, includes pipeline compiles): %.2fs\n", t_node);
    }

    // ---- aggregate ----
    auto op_kind = [](const ggml_tensor* t) { return trellis_op_kind(t); };
    std::vector<double> role_s(R_COUNT, 0.0);
    std::vector<int>    role_n(R_COUNT, 0);
    std::vector<std::array<double, 3>> blk_s(std::max(n_blocks, 1), { 0.0, 0.0, 0.0 });   // msa / cross / mlp
    std::map<std::string, std::pair<double, int>> op_s, op_role_s;
    for (const Seg& sg : segs) {
        role_s[sg.role] += sg.s; role_n[sg.role] += sg.last - sg.first + 1;
        if (sg.block >= 0 && sg.block < n_blocks && sg.role >= R_MSA && sg.role <= R_MLP) blk_s[sg.block][sg.role - R_MSA] += sg.s;
        for (int j = sg.first; j <= sg.last; ++j) {
            const std::string k = op_kind(ggml_graph_node(g_, j));
            auto& e = op_s[k]; e.first += dt[j]; e.second++;
            auto& e2 = op_role_s[std::string(role_name[sg.role]) + "|" + k]; e2.first += dt[j]; e2.second++;
        }
    }

    const double MB = 1.0 / 1048576.0;
    const uint64_t cond_bytes = p_.proj_attn ? (uint64_t)p_.d_proj * N_ * 4 * 2 : 0;
    printf("      [prof] N=%d tokens  Lc=%d cond  blocks=%d  nodes=%d  |  weights %.0f MB  activations %.0f MB  cond %.0f MB  |  %s\n",
           N_, Lc_, n_blocks, n, m_.total_bytes() * MB, alloc_bytes_ * MB, cond_bytes * MB,
           g_no_fa ? "exact SDPA (--no-fa)" : "FlashAttention");
    printf("      [prof] whole forward %.2fs | segment pass %.2fs (%zu slices, x%.2f) | node pass %.2fs (%d slices, x%.2f, isolated)\n",
           whole_s_.back(), t_seg, segs.size(), t_seg / whole_s_.back(), t_node, n, t_node / whole_s_.back());
    printf("      [prof] by role (segment pass; seconds, %% of segment pass):\n");
    for (int r = 0; r < R_COUNT; ++r)
        if (role_n[r])
            printf("      [prof]   %-16s %8.2fs  %5.1f%%  (%d nodes)\n", role_name[r], role_s[r], 100.0 * role_s[r] / t_seg, role_n[r]);
    printf("      [prof] by block (segment pass; self_attn / cross_attn+proj / mlp, seconds):\n");
    for (int b = 0; b < n_blocks; ++b)
        printf("      [prof]   blk%02d  %6.3f  %6.3f  %6.3f\n", b, blk_s[b][0], blk_s[b][1], blk_s[b][2]);
    std::vector<std::pair<std::string, std::pair<double, int>>> ops(op_s.begin(), op_s.end());
    std::sort(ops.begin(), ops.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    printf("      [prof] by op (node pass; ISOLATED single-node times, light ops overstated by the per-submit cost):\n");
    for (const auto& [k, v] : ops)
        if (v.first >= 0.005 * t_node)
            printf("      [prof]   %-24s %8.2fs  %5.1f%%  (n=%d, %.2f ms/op)\n", k.c_str(), v.first, 100.0 * v.first / t_node, v.second, 1e3 * v.first / v.second);
    std::vector<std::pair<std::string, std::pair<double, int>>> opr(op_role_s.begin(), op_role_s.end());
    std::sort(opr.begin(), opr.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    printf("      [prof] by role x op (node pass, top 12):\n");
    int shown = 0;
    for (const auto& [k, v] : opr) {
        if (shown++ >= 12) break;
        printf("      [prof]   %-42s %8.2fs  %5.1f%%  (n=%d)\n", k.c_str(), v.first, 100.0 * v.first / t_node, v.second);
    }
    std::vector<int> order(n);
    for (int j = 0; j < n; ++j) order[j] = j;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return dt[a] > dt[b]; });
    printf("      [prof] top 15 nodes (node pass):\n");
    for (int q = 0; q < std::min(15, n); ++q) {
        const int j = order[q];
        const ggml_tensor* t = ggml_graph_node(g_, j);
        char shp[128];
        snprintf(shp, sizeof shp, "[%lld,%lld,%lld,%lld]", (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
        char s0[96] = "";
        if (t->src[0]) snprintf(s0, sizeof s0, " src0 %s[%lld,%lld,%lld]", ggml_type_name(t->src[0]->type),
                                (long long)t->src[0]->ne[0], (long long)t->src[0]->ne[1], (long long)t->src[0]->ne[2]);
        printf("      [prof]   #%-5d %-24s %8.3fs  %s%s  %s\n", j, op_kind(t).c_str(), dt[j], shp, s0, ggml_get_name(t));
    }
    fflush(stdout);
}

// 3D interleaved-pair RoPE cos/sin tables: data[token*half + pair].
static void fill_rope(const DiTParams& p, int N, const std::function<void(int,int&,int&,int&)>& coord,
                      std::vector<float>& rcos, std::vector<float>& rsin) {
    const int half = p.head_dim / 2, fd = half / 3;     // 64, 21
    std::vector<float> freqs(fd);
    for (int j = 0; j < fd; ++j) freqs[j] = 1.0f / std::pow(10000.f, (float)j / fd);
    rcos.assign((size_t)N * half, 0.f); rsin.assign((size_t)N * half, 0.f);
    for (int tok = 0; tok < N; ++tok) {
        int cx, cy, cz; coord(tok, cx, cy, cz);
        for (int pp = 0; pp < half; ++pp) {
            float ang = 0;
            if (pp < fd) ang = cx * freqs[pp];
            else if (pp < 2*fd) ang = cy * freqs[pp - fd];
            else if (pp < 3*fd) ang = cz * freqs[pp - 2*fd];
            rcos[(size_t)tok * half + pp] = std::cos(ang);
            rsin[(size_t)tok * half + pp] = std::sin(ang);
        }
    }
}

DitRunner* make_dense_runner(const Model& m, const DiTParams& p, int R, int n_cond) {
    std::vector<float> rcos, rsin;
    fill_rope(p, R*R*R, [R](int tok, int& cx, int& cy, int& cz) {
        cx = tok / (R*R); cy = (tok / R) % R; cz = tok % R; }, rcos, rsin);
    return new DitRunner(m, p, R*R*R, n_cond, rcos, rsin);
}

DitRunner* make_sparse_runner(const Model& m, const DiTParams& p,
                              const std::vector<std::array<int,3>>& coords, int n_cond) {
    std::vector<float> rcos, rsin;
    fill_rope(p, (int)coords.size(), [&coords](int tok, int& cx, int& cy, int& cz) {
        cx = coords[tok][0]; cy = coords[tok][1]; cz = coords[tok][2]; }, rcos, rsin);
    return new DitRunner(m, p, (int)coords.size(), n_cond, rcos, rsin);
}

std::vector<float> sample_flow(const FlowFwdProj& fwd, std::vector<float> sample,
                               const float* cond, const float* neg_cond,
                               const float* proj, const float* neg_proj,
                               const SamplerParams& sp,
                               std::vector<std::vector<float>>* trace) {
    const float sm = sp.sigma_min;
    const size_t Nst = sample.size();
    std::vector<float> ts(sp.steps + 1);
    for (int i = 0; i <= sp.steps; ++i) {
        float t = 1.0f - (float)i / sp.steps;
        ts[i] = sp.rescale_t * t / (1.0f + (sp.rescale_t - 1.0f) * t);
    }
    std::vector<float> pos, neg, pred(Nst);
    static const bool dbg_step = std::getenv("TRELLIS_DBG_STEP") != nullptr;
    static const bool no_fix   = std::getenv("TRELLIS_NOFIX") != nullptr;  // robustness guards ON by default
    const auto tflow0 = std::chrono::steady_clock::now();
    int n_fwd = 0;
    auto fstats = [](const std::vector<float>& v, size_t& bad, double& mx) {
        bad = 0; mx = 0; for (float x : v) { if (!std::isfinite(x)) bad++; else if (std::fabs(x) > mx) mx = std::fabs(x); }
    };
    // Progress. The 1024 cascade's HR pass is ~16 min of silence otherwise, which reads as
    // a hang. On a TTY redraw one line; when redirected to a log, emit a line per step (12
    // steps, so it stays readable). ETA from the mean step so far -- steps are near-uniform
    // except where the guidance interval drops a forward, so it settles after step 2.
    const bool tty = stdout_is_tty();
    auto progress = [&](int done) {
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - tflow0).count();
        char bar[21];
        const int fill = sp.steps ? done * 20 / sp.steps : 20;
        for (int k = 0; k < 20; ++k) bar[k] = k < fill ? '#' : '.';
        bar[20] = 0;
        char eta[32];
        if (done) snprintf(eta, sizeof eta, "~%.0fs left", el / done * (sp.steps - done));
        else      snprintf(eta, sizeof eta, "starting");     // no rate yet -- don't invent an ETA
        printf("%s      [flow] [%s] %2d/%d  %5.1fs  %-12s%s",
               tty ? "\r" : "", bar, done, sp.steps, el, eta, tty ? "" : "\n");
        fflush(stdout);
    };
    progress(0);
    for (int i = 0; i < sp.steps; ++i) {
        const float t = ts[i], tprev = ts[i + 1];
        const float gs = (sp.gi0 <= t && t <= sp.gi1) ? sp.guidance_strength : 1.0f;
        const float tscaled = 1000.0f * t;
        if (gs == 1.0f) {
            pred = fwd(sample, tscaled, cond, proj);
            ++n_fwd;
        } else if (gs == 0.0f) {
            pred = fwd(sample, tscaled, neg_cond, neg_proj);
            ++n_fwd;
        } else {
            pos = fwd(sample, tscaled, cond, proj);
            neg = fwd(sample, tscaled, neg_cond, neg_proj);
            n_fwd += 2;
            for (size_t k = 0; k < Nst; ++k) pred[k] = gs * pos[k] + (1 - gs) * neg[k];
            if (sp.guidance_rescale > 0.0f) {
                const float a = 1 - sm, b = sm + (1 - sm) * t;
                double mp = 0, mc = 0;
                std::vector<float> x0p(Nst), x0c(Nst);
                for (size_t k = 0; k < Nst; ++k) { x0p[k] = a*sample[k] - b*pos[k]; x0c[k] = a*sample[k] - b*pred[k]; mp += x0p[k]; mc += x0c[k]; }
                mp /= Nst; mc /= Nst;
                double vp = 0, vc = 0;
                for (size_t k = 0; k < Nst; ++k) { vp += (x0p[k]-mp)*(x0p[k]-mp); vc += (x0c[k]-mc)*(x0c[k]-mc); }
                float ratio = vc > 0 ? (float)(std::sqrt(vp/(Nst-1)) / std::sqrt(vc/(Nst-1))) : 1.0f;
                // OOD inputs (e.g. a thin figure at HR) can make vc tiny -> ratio explodes -> the
                // rescaled velocity blows the latent past representable range over the 12 steps ->
                // all-NaN SLAT. Clamp ratio to a sane band: it sits at ~1.0 for in-distribution
                // props (a no-op there), and only bites on the pathological tail. (TRELLIS_NOFIX=1
                // restores the raw behaviour for A/B.)
                if (!no_fix) { if (!std::isfinite(ratio)) ratio = 1.0f; ratio = fminf(fmaxf(ratio, 0.2f), 5.0f); }
                float gr = sp.guidance_rescale;
                for (size_t k = 0; k < Nst; ++k) { float x0r = x0c[k]*ratio; float x0 = gr*x0r + (1-gr)*x0c[k]; pred[k] = (a*sample[k] - x0) / b; }
            }
        }
        // Safety net: never integrate a non-finite velocity (one poisoned tap would spread to the
        // whole latent on the next attention). A no-op when everything is finite.
        if (!no_fix) for (size_t k = 0; k < Nst; ++k) if (!std::isfinite(pred[k])) pred[k] = 0.0f;
        for (size_t k = 0; k < Nst; ++k) sample[k] -= (t - tprev) * pred[k];
        progress(i + 1);
        if (dbg_step) { size_t pb, sb; double pm, sm2; fstats(pred, pb, pm); fstats(sample, sb, sm2);
            if (tty) printf("\n");
            fprintf(stderr, "      [flow-step %2d] t=%.3f gs=%.1f  pred[nan=%zu max=%.3g]  sample[nan=%zu max=%.3g]\n", i, t, gs, pb, pm, sb, sm2); }
        if (trace) trace->push_back(sample);
    }
    if (tty) printf("\n");
    printf("      [flow] %d steps, %d forwards, %.1fs\n", sp.steps, n_fwd,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - tflow0).count());
    fflush(stdout);
    return sample;
}

std::vector<float> sample_flow(const FlowFwd& fwd, std::vector<float> sample,
                               const float* cond, const float* neg_cond, const SamplerParams& sp,
                               std::vector<std::vector<float>>* trace) {
    FlowFwdProj f = [&fwd](const std::vector<float>& x, float t, const float* c, const float*) { return fwd(x, t, c); };
    return sample_flow(f, std::move(sample), cond, neg_cond, nullptr, nullptr, sp, trace);
}

} // namespace trellis
