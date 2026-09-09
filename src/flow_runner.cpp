#include "flow_runner.h"
#include "trellis_model.h"
#include "graph_dump.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
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
        gproj_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, p_.d_proj, N_); ggml_set_input(gproj_);
    }
    gidx_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, p_.head_dim); ggml_set_input(gidx_);
    dit_rope_index(p_.head_dim, ridx_);
    dbg_nan_ = std::getenv("TRELLIS_DBG_NAN") != nullptr;
    gout_ = build_dit_dense(ctx_, m_, p_, gh0_, gtf_, gcond_, gcos_, gsin_,
                            dbg_nan_ ? &inter_ : nullptr, gproj_, gidx_);
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
    check_device_budget();
    rcos_ = rcos; rsin_ = rsin;   // keep; re-upload each forward (gallocr reuses input buffers across runs)
}

DitRunner::~DitRunner() {
    if (alloc_) ggml_gallocr_free(alloc_);
    if (ctx_)   ggml_free(ctx_);
}

std::vector<float> DitRunner::forward(const std::vector<float>& xt, float t_scaled, const float* cond,
                                      const float* proj) {
    std::vector<float> tf; timestep_embedding(t_scaled, tf);
    ggml_backend_tensor_set(gh0_,  xt.data(), 0, xt.size() * 4);
    ggml_backend_tensor_set(gtf_,  tf.data(), 0, tf.size() * 4);
    ggml_backend_tensor_set(gcond_, cond,     0, (size_t)p_.d_cond * Lc_ * 4);
    ggml_backend_tensor_set(gcos_, rcos_.data(), 0, rcos_.size() * 4);   // re-upload (buffers reused across runs)
    ggml_backend_tensor_set(gsin_, rsin_.data(), 0, rsin_.size() * 4);
    ggml_backend_tensor_set(gidx_, ridx_.data(), 0, ridx_.size() * sizeof(int32_t));
    if (gproj_) {
        if (!proj) throw std::runtime_error("DitRunner: proj_attn model requires a proj tensor");
        ggml_backend_tensor_set(gproj_, proj, 0, (size_t)p_.d_proj * N_ * 4);
    }
    if (ggml_backend_graph_compute(m_.backend, g_) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("DitRunner: compute failed");
    std::vector<float> outv = tensor_to_f32(gout_);
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
