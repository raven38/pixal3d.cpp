// Device-resident Pixal3D SS-stage conditioning (see include/pixal3d_cond.h,
// pixal3d_cond_ss_gpu). Same math as pixal3d_cond_ss (src/pixal3d_cond.cpp), but the
// projection and the multiview average run as ggml ops on the model's backend so the DINOv3
// token map never round-trips through the host:
//
//   per view v:   img -> dinov3_build -> x [1024, Ntok]
//                 patches = x[:, 5:]  (view, [1024, Hp*Wp], token h*Wp+w == fmap plane index)
//                 z = sum_t get_rows(patches, idx_t) * w_t   (4 bilinear taps, [1024, R^3])
//                 acc_glob += x[:, :5] / V ;  acc_proj += z / V    (ggml_cpy back into the
//                                                                    persistent accumulators)
//
// The accumulators live in a small persistent backend buffer; every per-view graph is
// allocated with its own gallocr and freed before the next view.
#include "pixal3d_cond.h"
#include "dinov3.h"
#include "naf.h"
#include "proj_grid.h"
#include "trellis_model.h"
#include "graph_dump.h"
#include "trellis_args.h"   // g_profile_cond
#include "ggml.h"
#include <cstdio>
#include <functional>
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trellis {
using T = ggml_tensor;

static constexpr int NPREFIX = 5, D = 1024;

Pixal3dCond pixal3d_cond_ss_gpu(const Model& dinov3, const std::vector<Pixal3dView>& views,
                                 int S, int R, float mesh_scale, Pixal3dCondStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    Pixal3dCond out;
    out.n_global = NPREFIX;
    out.d_proj = D;
    out.global.assign((size_t)NPREFIX * D, 0.0f);
    out.proj.assign((size_t)R * R * R * D, 0.0f);

    const int V = (int)views.size();
    if (V == 0) return out;
    const int Hp = S / 16, Wp = Hp, NP = Hp * Wp;
    const int64_t N3 = (int64_t)R * R * R;

    std::vector<float> c2w_flat((size_t)V * 16);
    for (int v = 0; v < V; ++v)
        std::memcpy(&c2w_flat[(size_t)v * 16], views[v].c2w, 16 * sizeof(float));
    const float* c0 = views[0].c2w;
    const float distance0 = std::sqrt(c0[3] * c0[3] + c0[7] * c0[7] + c0[11] * c0[11]);
    std::vector<float> calc;
    mv_calc_mats(c2w_flat.data(), V, distance0, calc);

    std::vector<float> rcos, rsin;
    dinov3_rope_tables(S, rcos, rsin);

    // Persistent accumulators (zeroed), channel-major like the graph tensors they receive.
    ggml_context* pc = ggml_init({ ggml_tensor_overhead() * 4 + 256, nullptr, true });
    T* acc_glob = ggml_new_tensor_2d(pc, GGML_TYPE_F32, D, NPREFIX);
    T* acc_proj = ggml_new_tensor_2d(pc, GGML_TYPE_F32, D, N3);
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pc, dinov3.backend);
    if (!pbuf) throw std::runtime_error("pixal3d_cond_ss_gpu: accumulator alloc failed");
    ggml_backend_buffer_clear(pbuf, 0);

    Pixal3dCondStats st;
    st.views = V;
    st.weight_bytes = dinov3.buffer ? ggml_backend_buffer_get_size(dinov3.buffer) : 0;
    st.cond_bytes = ggml_backend_buffer_get_size(pbuf);

    for (int v = 0; v < V; ++v) {
        const auto tv = std::chrono::steady_clock::now();
        std::vector<float> normed = pixal3d_imagenet_normalize(views[v].rgb_premult, S);

        Camera cam{};
        cam.has_c2w = true;
        cam.mesh_scale = mesh_scale;
        cam.fov_x = views[v].fov_x;
        for (int i = 0; i < 16; ++i) cam.c2w[i] = calc[(size_t)v * 16 + i];
        std::vector<int32_t> idx[4];
        std::vector<float> w[4];
        proj_grid_bilinear_taps(Hp, Wp, R, S, cam, idx, w);

        size_t meta = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(16384, false) + (1 << 20);
        ggml_context* c = ggml_init({ meta, nullptr, true });
        Dinov3Inputs in{};
        T* x = dinov3_build(c, dinov3, S, in);                       // [D, Ntok]
        T* glob = ggml_view_2d(c, x, D, NPREFIX, x->nb[1], 0);      // [D, 5]
        T* patches = ggml_view_2d(c, x, D, NP, x->nb[1], (size_t)NPREFIX * x->nb[1]);  // [D, Hp*Wp]

        T *gidx[4], *gw[4];
        T* z = nullptr;
        for (int t = 0; t < 4; ++t) {
            gidx[t] = ggml_new_tensor_1d(c, GGML_TYPE_I32, N3);       ggml_set_input(gidx[t]);
            gw[t]   = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, N3);    ggml_set_input(gw[t]);
            T* tap = ggml_mul(c, ggml_get_rows(c, patches, gidx[t]), gw[t]);   // [D, R^3] * [1, R^3]
            z = z ? ggml_add(c, z, tap) : tap;
        }
        const float inv_v = 1.0f / (float)V;
        T* ng = ggml_add(c, acc_glob, ggml_scale(c, glob, inv_v));
        T* np = ggml_add(c, acc_proj, ggml_scale(c, z, inv_v));
        T* wg = ggml_cpy(c, ng, acc_glob);   // write the running averages back in place
        T* wp = ggml_cpy(c, np, acc_proj);

        ggml_cgraph* g = ggml_new_graph_custom(c, 16384, false);
        ggml_build_forward_expand(g, wg);
        ggml_build_forward_expand(g, wp);
        const std::string tag = "pixal3d_cond_ss_gpu_S" + std::to_string(S) + "_R" + std::to_string(R) + "_v" + std::to_string(v);
        trellis_graph_dump(tag.c_str(), g);
        check_graph_supported(dinov3.backend, g, tag.c_str());

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dinov3.backend));
        if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error("pixal3d_cond_ss_gpu: alloc failed");
        const size_t ab = ggml_gallocr_get_buffer_size(alloc, 0);
        if (ab > st.view_alloc_bytes) st.view_alloc_bytes = ab;

        ggml_backend_tensor_set(in.img, normed.data(), 0, normed.size() * 4);
        ggml_backend_tensor_set(in.cos, rcos.data(), 0, rcos.size() * 4);
        ggml_backend_tensor_set(in.sin, rsin.data(), 0, rsin.size() * 4);
        for (int t = 0; t < 4; ++t) {
            ggml_backend_tensor_set(gidx[t], idx[t].data(), 0, idx[t].size() * sizeof(int32_t));
            ggml_backend_tensor_set(gw[t], w[t].data(), 0, w[t].size() * sizeof(float));
        }
        if (ggml_backend_graph_compute(dinov3.backend, g) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("pixal3d_cond_ss_gpu: compute failed");
        ggml_backend_synchronize(dinov3.backend);
        ggml_gallocr_free(alloc);   // release this view's temporaries before the next view
        ggml_free(c);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count();
        if (ms > st.view_ms_max) st.view_ms_max = ms;
    }

    // One readback of the fused condition: ggml channel-major [D, tok] -> host token-major.
    std::vector<float> hg = tensor_to_f32(acc_glob), hp = tensor_to_f32(acc_proj);
    for (int t = 0; t < NPREFIX; ++t)
        for (int c = 0; c < D; ++c) out.global[(size_t)t * D + c] = hg[(size_t)t * D + c];
    for (int64_t k = 0; k < N3; ++k)
        for (int c = 0; c < D; ++c) out.proj[(size_t)k * D + c] = hp[(size_t)k * D + c];

    ggml_backend_buffer_free(pbuf);
    ggml_free(pc);

    // peak_bytes is tracked at each graph using the persistent buffers resident at that point.
    st.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

// ---------------------------------------------------------------------------
// 分割グラフ版の SLAT conditioning（texture 段 S=1024/R=64/naf_T=1024 用）。
//
// 単一グラフ版は NAF 出力 [1024, T*T] が T=1024 で 4 GiB になり、gallocr の1本の
// バッファが実測 11.3 GiB（M4 Max/Metal）になる。WebGPU の maxBufferSize は約 4 GiB
// なのでブラウザでは確保できない。ここでは1 view を4段のグラフに割り、
//   G1: DINOv3 -> global 累積 + patch map を永続バッファへ + lr projection tap 累積
//   G2: NAF encoder -> pooled [T,T,256] を永続バッファへ
//   G3: block 行 stripe ごとに RoPE -> q_bm / k_rows を永続バッファへ
//   G4: block chunk ごとに neighborhood attention -> その chunk に落ちる hr tap だけ累積
// とする。段ごとに gallocr バッファが分かれるので、単一バッファの最大値が下がる。
// hr の tap は chunk 外なら重み 0・索引 0 に潰すので、accumulate は chunk をまたいで正しい。
static Pixal3dCond cond_slat_gpu_chunked(const Model& dinov3, const Model& naf,
                                          const std::vector<Pixal3dView>& views,
                                          const Pixal3dSlatCondParams& prm,
                                          const std::vector<int32_t>& sel, int64_t NT,
                                          Pixal3dCondStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    const int S = prm.S, R = prm.R, Tn = prm.naf_T, V = (int)views.size();
    const int Hp = S / 16, Wp = Hp, NP = Hp * Wp;
    const int dy = Tn / Hp, dx = Tn / Wp, d2 = dy * dx, nblk = Hp * Wp;
    const int64_t TT = (int64_t)Tn * Tn;
    const bool sparse = !sel.empty();

    Pixal3dCond out;
    out.n_global = NPREFIX;
    out.d_proj = 2 * D;
    out.global.assign((size_t)NPREFIX * D, 0.0f);
    out.proj.assign((size_t)NT * 2 * D, 0.0f);

    std::vector<float> c2w_flat((size_t)V * 16);
    for (int v = 0; v < V; ++v) std::memcpy(&c2w_flat[(size_t)v * 16], views[v].c2w, 16 * sizeof(float));
    const float* c0 = views[0].c2w;
    const float distance0 = std::sqrt(c0[3] * c0[3] + c0[7] * c0[7] + c0[11] * c0[11]);
    std::vector<float> calc;
    mv_calc_mats(c2w_flat.data(), V, distance0, calc);

    std::vector<float> rcos, rsin, ncos, nsin;
    dinov3_rope_tables(S, rcos, rsin);
    naf_rope_tables(Tn, tensor_to_f32(naf.get("image_encoder.rope.periods")), ncos, nsin);
    std::vector<int32_t> win_idx, raster_of_bm, bm_of_raster;
    naf_window_index(Tn, Hp, Wp, win_idx);
    naf_block_order(Tn, Hp, Wp, raster_of_bm, bm_of_raster);
    const NafGgmlOpts nopts = naf_ggml_opts_for(naf);

    // stripe / chunk サイズ: 1グラフあたりのバッファを数百 MB に抑える。
    const int64_t stripe_pixels = 65536;
    int nbrow = (int)std::max<int64_t>(1, stripe_pixels / ((int64_t)dy * Tn));
    nbrow = std::min(nbrow, Hp);
    int chunk_blk = prm.naf_block_chunk > 0 ? prm.naf_block_chunk
                  : (int)std::max<int64_t>(1, (int64_t)256 * 1024 * 1024 / ((int64_t)d2 * D * 4));
    chunk_blk = std::min(chunk_blk, nblk);

    // 永続バッファ: 大物は1本ずつ別バッファにする（1本が maxBufferSize を超えないように）。
    auto alloc_one = [&](ggml_context*& ctx, T*& t, int64_t n0, int64_t n1) {
        ctx = ggml_init({ ggml_tensor_overhead() + 256, nullptr, true });
        t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, dinov3.backend);
        if (!b) throw std::runtime_error("pixal3d_cond_slat_gpu: persistent alloc failed");
        ggml_backend_buffer_clear(b, 0);
        return b;
    };
    ggml_context *cp = nullptr, *cq = nullptr, *cs = nullptr;
    T *P_pooled = nullptr, *Q_bm = nullptr;
    // pooled / q_bm are 1 GiB each at T=1024. Allocate them only for the graph phases
    // that need them instead of keeping both resident for the whole call.
    ggml_backend_buffer_t bp = nullptr, bq = nullptr;
    size_t resident_bytes = 0;
    cs = ggml_init({ ggml_tensor_overhead() * 8 + 256, nullptr, true });
    T* K_rows   = ggml_new_tensor_2d(cs, GGML_TYPE_F32, 256, nblk);
    T* V_rows   = ggml_new_tensor_2d(cs, GGML_TYPE_F32, D, NP);
    T* acc_glob = ggml_new_tensor_2d(cs, GGML_TYPE_F32, D, NPREFIX);
    T* acc_lr   = ggml_new_tensor_2d(cs, GGML_TYPE_F32, D, NT);
    T* acc_hr   = ggml_new_tensor_2d(cs, GGML_TYPE_F32, D, NT);
    ggml_backend_buffer_t bs = ggml_backend_alloc_ctx_tensors(cs, dinov3.backend);
    if (!bs) throw std::runtime_error("pixal3d_cond_slat_gpu: accumulator alloc failed");
    ggml_backend_buffer_clear(bs, 0);

    Pixal3dCondStats st;
    st.views = V;
    st.weight_bytes = (dinov3.buffer ? ggml_backend_buffer_get_size(dinov3.buffer) : 0)
                    + (naf.buffer ? ggml_backend_buffer_get_size(naf.buffer) : 0);
    const size_t bytes_p = (size_t)TT * 256 * 4, bytes_q = (size_t)256 * TT * 4;
    st.cond_bytes = bytes_p + bytes_q + ggml_backend_buffer_get_size(bs);
    resident_bytes = ggml_backend_buffer_get_size(bs);

    // --profile-cond: per-graph-kind exclusive laps (build+alloc / upload / compute / free) summed per
    // view, plus the host-side tap preparation. printf only; nothing here re-runs a graph (the
    // accumulator graphs are NOT idempotent -- see docs/design/2026-09-20-conditioning-profiler.md).
    struct GKind { double build = 0, upload = 0, compute = 0, free_ = 0; int n = 0; };
    enum { K_DINO, K_ENC, K_QK, K_ATTN, K_COUNT };
    static const char* kind_name[K_COUNT] = { "dino", "naf_enc", "naf_qk", "naf_attn" };
    GKind gk[K_COUNT];
    double t_host_taps = 0, t_host_chunk_prep = 0, t_persistent = 0; int attn_skipped = 0;
    auto kind_of = [](const char* tag) {
        const std::string t = tag;
        if (t.find("_naf_enc") != std::string::npos) return (int)K_ENC;
        if (t.find("_naf_qk") != std::string::npos) return (int)K_QK;
        if (t.find("_naf_attn") != std::string::npos) return (int)K_ATTN;
        return (int)K_DINO;
    };
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };

    // グラフを1本組んで走らせる小ヘルパ（入力の投入は alloc 後に upload() で行う）。
    auto run_graph = [&](ggml_context* c, const std::vector<T*>& outs, const char* tag,
                         const std::function<void()>& upload) {
        const auto ta = clk::now();
        ggml_cgraph* g = ggml_new_graph_custom(c, 16384, false);
        for (T* o : outs) ggml_build_forward_expand(g, o);
        trellis_graph_dump(tag, g);
        check_graph_supported(dinov3.backend, g, tag);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dinov3.backend));
        if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error(std::string("alloc failed: ") + tag);
        const size_t ab = ggml_gallocr_get_buffer_size(alloc, 0);
        if (ab > st.view_alloc_bytes) st.view_alloc_bytes = ab;
        if (getenv("TRELLIS_DBG_COND")) fprintf(stderr, "[cond] %-40s graph buffer %.1f MB (%d nodes)\n", tag, ab / 1048576.0, ggml_graph_n_nodes(g));
        const auto tb = clk::now();
        upload();
        const auto tc = clk::now();
        if (ggml_backend_graph_compute(dinov3.backend, g) != GGML_STATUS_SUCCESS)
            throw std::runtime_error(std::string("compute failed: ") + tag);
        ggml_backend_synchronize(dinov3.backend);
        const auto td = clk::now();
        ggml_gallocr_free(alloc);
        const auto te = clk::now();
        GKind& k = gk[kind_of(tag)];
        k.build += secs(ta, tb); k.upload += secs(tb, tc); k.compute += secs(tc, td); k.free_ += secs(td, te); k.n++;
    };

    const float inv_v = 1.0f / (float)V;
    const size_t meta = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(16384, false) + (1 << 20);

    for (int v = 0; v < V; ++v) {
        const auto tv = std::chrono::steady_clock::now();
        std::vector<float> normed = pixal3d_imagenet_normalize(views[v].rgb_premult, S);
        Camera cam{};
        cam.has_c2w = true; cam.mesh_scale = prm.mesh_scale; cam.fov_x = views[v].fov_x;
        for (int i = 0; i < 16; ++i) cam.c2w[i] = calc[(size_t)v * 16 + i];
        std::vector<int32_t> idx_lr[4], idx_hr[4];
        std::vector<float> w_lr[4], w_hr[4];
        proj_grid_bilinear_taps(Hp, Wp, R, S, cam, idx_lr, w_lr);
        proj_grid_bilinear_taps(Tn, Tn, R, S, cam, idx_hr, w_hr);
        for (int t = 0; t < 4; ++t)
            for (int32_t& i : idx_hr[t]) i = bm_of_raster[i];
        if (sparse) {
            for (int t = 0; t < 4; ++t) {
                std::vector<int32_t> il(sel.size()), ih(sel.size());
                std::vector<float> wl(sel.size()), wh(sel.size());
                for (size_t j = 0; j < sel.size(); ++j) {
                    il[j] = idx_lr[t][sel[j]]; wl[j] = w_lr[t][sel[j]];
                    ih[j] = idx_hr[t][sel[j]]; wh[j] = w_hr[t][sel[j]];
                }
                idx_lr[t].swap(il); w_lr[t].swap(wl); idx_hr[t].swap(ih); w_hr[t].swap(wh);
            }
        }
        for (GKind& k : gk) k = GKind{};
        t_host_taps = secs(tv, clk::now()); t_host_chunk_prep = 0; t_persistent = 0; attn_skipped = 0;

        // ---- G1: DINOv3 -> global 累積, patch map を V_rows へ, lr tap 累積 ----
        {
            ggml_context* c = ggml_init({ meta, nullptr, true });
            Dinov3Inputs in{};
            T* x = dinov3_build(c, dinov3, S, in);
            T* glob = ggml_view_2d(c, x, D, NPREFIX, x->nb[1], 0);
            T* patches = ggml_view_2d(c, x, D, NP, x->nb[1], (size_t)NPREFIX * x->nb[1]);
            T *gidx[4], *gw[4]; T* z_lr = nullptr;
            for (int t = 0; t < 4; ++t) {
                gidx[t] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NT);   ggml_set_input(gidx[t]);
                gw[t]   = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, NT); ggml_set_input(gw[t]);
                T* tl = ggml_mul(c, ggml_get_rows(c, patches, gidx[t]), gw[t]);
                z_lr = z_lr ? ggml_add(c, z_lr, tl) : tl;
            }
            T* wv = ggml_cpy(c, patches, V_rows);
            T* wg = ggml_cpy(c, ggml_add(c, acc_glob, ggml_scale(c, glob, inv_v)), acc_glob);
            T* wl = ggml_cpy(c, ggml_add(c, acc_lr, ggml_scale(c, z_lr, inv_v)), acc_lr);
            run_graph(c, {wv, wg, wl}, "pixal3d_cond_slat_chunked_dino", [&]() {
                ggml_backend_tensor_set(in.img, normed.data(), 0, normed.size() * 4);
                ggml_backend_tensor_set(in.cos, rcos.data(), 0, rcos.size() * 4);
                ggml_backend_tensor_set(in.sin, rsin.data(), 0, rsin.size() * 4);
                for (int t = 0; t < 4; ++t) {
                    ggml_backend_tensor_set(gidx[t], idx_lr[t].data(), 0, idx_lr[t].size() * sizeof(int32_t));
                    ggml_backend_tensor_set(gw[t], w_lr[t].data(), 0, w_lr[t].size() * sizeof(float));
                }
            });
            ggml_free(c);
        }

        // ---- G2: NAF encoder -> pooled を永続バッファへ ----
        {
            const auto tp = clk::now();
            bp = alloc_one(cp, P_pooled, TT, 256);
            resident_bytes += bytes_p;
            t_persistent += secs(tp, clk::now());
        }
        // S == T（pool が恒等）なら 2 枝を別グラフにして pooled の前半/後半へ直接書く。
        // concat 用の 1 GiB と、片枝を保持したままもう片枝を回すぶんのピークが消える。
        if (S == Tn) {
            for (int half = 0; half < 2; ++half) {
                ggml_context* c = ggml_init({ meta, nullptr, true });
                T* img = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, S, 3, 1); ggml_set_input(img);
                T* e = naf_build_encoder_half(c, naf, img, half == 1, nopts);     // [S,S,128]
                T* dst = ggml_view_2d(c, P_pooled, TT, 128, P_pooled->nb[1], (size_t)half * 128 * P_pooled->nb[1]);
                T* wp = ggml_cpy(c, ggml_reshape_2d(c, e, TT, 128), dst);
                run_graph(c, {wp}, half == 0 ? "pixal3d_cond_slat_chunked_naf_enc0"
                                             : "pixal3d_cond_slat_chunked_naf_enc1", [&]() {
                    ggml_backend_tensor_set(img, views[v].rgb_premult.data(), 0, views[v].rgb_premult.size() * 4);
                });
                ggml_free(c);
            }
        } else {
            ggml_context* c = ggml_init({ meta, nullptr, true });
            T* img = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, S, 3, 1); ggml_set_input(img);
            T* pooled = naf_build_encoder(c, naf, img, S, Tn, nopts);        // [T,T,256]
            T* wp = ggml_cpy(c, ggml_reshape_2d(c, pooled, TT, 256), P_pooled);
            run_graph(c, {wp}, "pixal3d_cond_slat_chunked_naf_enc", [&]() {
                ggml_backend_tensor_set(img, views[v].rgb_premult.data(), 0, views[v].rgb_premult.size() * 4);
            });
            ggml_free(c);
        }

        // ---- G3: block 行 stripe ごとに RoPE -> q_bm / k_rows ----
        {
            const auto tp = clk::now();
            bq = alloc_one(cq, Q_bm, 256, TT);
            resident_bytes += bytes_q;
            t_persistent += secs(tp, clk::now());
        }
        for (int b0 = 0; b0 < Hp; b0 += nbrow) {
            const int nrow = std::min(nbrow, Hp - b0);
            const int64_t sy = (int64_t)nrow * dy, Pp = sy * Tn;
            const int nb = nrow * Wp;
            // stripe ローカルの RoPE 表と block-major 索引。
            std::vector<float> cs_(64 * (size_t)Pp), sn_(64 * (size_t)Pp);
            const int64_t p0 = (int64_t)b0 * dy * Tn;
            for (int d = 0; d < 64; ++d) {
                std::memcpy(&cs_[(size_t)d * Pp], &ncos[(size_t)d * TT + p0], (size_t)Pp * 4);
                std::memcpy(&sn_[(size_t)d * Pp], &nsin[(size_t)d * TT + p0], (size_t)Pp * 4);
            }
            std::vector<int32_t> blk_local((size_t)d2 * nb);
            for (int64_t i = 0; i < (int64_t)d2 * nb; ++i)
                blk_local[i] = (int32_t)(raster_of_bm[(size_t)d2 * b0 * Wp + i] - p0);

            ggml_context* c = ggml_init({ meta, nullptr, true });
            T* rc = ggml_new_tensor_2d(c, GGML_TYPE_F32, Pp, 64); ggml_set_input(rc);
            T* rs = ggml_new_tensor_2d(c, GGML_TYPE_F32, Pp, 64); ggml_set_input(rs);
            T* bi = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t)d2 * nb); ggml_set_input(bi);
            T* pooled = ggml_reshape_3d(c, P_pooled, Tn, Tn, 256);
            NafStripeQK qk = naf_build_qk_stripe(c, pooled, rc, rs, bi, Tn, Hp, Wp, b0, nrow);
            T* wq = ggml_cpy(c, qk.q_bm, ggml_view_2d(c, Q_bm, 256, (int64_t)d2 * nb, Q_bm->nb[1],
                                                      (size_t)d2 * b0 * Wp * Q_bm->nb[1]));
            T* wk = ggml_cpy(c, qk.k_rows, ggml_view_2d(c, K_rows, 256, nb, K_rows->nb[1],
                                                        (size_t)b0 * Wp * K_rows->nb[1]));
            run_graph(c, {wq, wk}, "pixal3d_cond_slat_chunked_naf_qk", [&]() {
                ggml_backend_tensor_set(rc, cs_.data(), 0, cs_.size() * 4);
                ggml_backend_tensor_set(rs, sn_.data(), 0, sn_.size() * 4);
                ggml_backend_tensor_set(bi, blk_local.data(), 0, blk_local.size() * sizeof(int32_t));
            });
            ggml_free(c);
        }

        {
            const auto tp = clk::now();
            ggml_backend_buffer_free(bp); ggml_free(cp);
            bp = nullptr; cp = nullptr; P_pooled = nullptr;
            resident_bytes -= bytes_p;
            t_persistent += secs(tp, clk::now());
        }

        // ---- G4: block chunk ごとに attention -> その chunk の hr tap を累積 ----
        for (int blk0 = 0; blk0 < nblk; blk0 += chunk_blk) {
            const auto tp = clk::now();
            const int nb = std::min(chunk_blk, nblk - blk0);
            const int64_t lo = (int64_t)d2 * blk0, hi = lo + (int64_t)d2 * nb;
            std::vector<int32_t> ci[4]; std::vector<float> cw[4];
            bool any = false;
            for (int t = 0; t < 4; ++t) {
                ci[t].assign((size_t)NT, 0); cw[t].assign((size_t)NT, 0.0f);
                for (int64_t j = 0; j < NT; ++j) {
                    const int32_t g = idx_hr[t][(size_t)j];
                    if (g >= lo && g < hi) { ci[t][(size_t)j] = (int32_t)(g - lo); cw[t][(size_t)j] = w_hr[t][(size_t)j]; any = true; }
                }
            }
            t_host_chunk_prep += secs(tp, clk::now());
            if (!any) { attn_skipped++; continue; }   // この chunk に落ちる tap が無い

            ggml_context* c = ggml_init({ meta, nullptr, true });
            T* wi = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t)81 * nblk); ggml_set_input(wi);
            T* hr = naf_build_attn_chunk(c, Q_bm, K_rows, V_rows, wi, d2, nblk, blk0, nb);
            T *gidx[4], *gw[4]; T* z_hr = nullptr;
            for (int t = 0; t < 4; ++t) {
                gidx[t] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NT);   ggml_set_input(gidx[t]);
                gw[t]   = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, NT); ggml_set_input(gw[t]);
                T* th = ggml_mul(c, ggml_get_rows(c, hr, gidx[t]), gw[t]);
                z_hr = z_hr ? ggml_add(c, z_hr, th) : th;
            }
            T* wh = ggml_cpy(c, ggml_add(c, acc_hr, ggml_scale(c, z_hr, inv_v)), acc_hr);
            run_graph(c, {wh}, "pixal3d_cond_slat_chunked_naf_attn", [&]() {
                ggml_backend_tensor_set(wi, win_idx.data(), 0, win_idx.size() * sizeof(int32_t));
                for (int t = 0; t < 4; ++t) {
                    ggml_backend_tensor_set(gidx[t], ci[t].data(), 0, ci[t].size() * sizeof(int32_t));
                    ggml_backend_tensor_set(gw[t], cw[t].data(), 0, cw[t].size() * sizeof(float));
                }
            });
            ggml_free(c);
        }
        {
            const auto tp = clk::now();
            ggml_backend_buffer_free(bq); ggml_free(cq);
            bq = nullptr; cq = nullptr; Q_bm = nullptr;
            resident_bytes -= bytes_q;
            t_persistent += secs(tp, clk::now());
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count();
        if (ms > st.view_ms_max) st.view_ms_max = ms;
        if (g_profile_cond) {
            double acc = t_host_taps + t_host_chunk_prep + t_persistent;
            printf("      [cond-v] cond_slat_gpu S=%d R=%d T=%d view %d: host taps %.2f | chunk_prep %.2f | persistent alloc/free %.2f", S, R, Tn, v, t_host_taps, t_host_chunk_prep, t_persistent);
            for (int k = 0; k < K_COUNT; ++k) {
                const GKind& g = gk[k];
                if (!g.n) continue;
                acc += g.build + g.upload + g.compute + g.free_;
                printf(" | %s x%d: build %.2f up %.2f compute %.2f free %.2f", kind_name[k], g.n, g.build, g.upload, g.compute, g.free_);
            }
            printf("  (attn chunks skipped %d; laps %.2f of view %.2fs)\n", attn_skipped, acc, ms / 1e3);
            fflush(stdout);
        }
    }

    const auto t_rb = clk::now();
    std::vector<float> hg = tensor_to_f32(acc_glob), hl = tensor_to_f32(acc_lr), hh = tensor_to_f32(acc_hr);
    for (int t = 0; t < NPREFIX; ++t)
        for (int c = 0; c < D; ++c) out.global[(size_t)t * D + c] = hg[(size_t)t * D + c];
    for (int64_t k = 0; k < NT; ++k) {
        float* dst = &out.proj[(size_t)k * 2 * D];
        std::memcpy(dst, &hl[(size_t)k * D], D * sizeof(float));
        std::memcpy(dst + D, &hh[(size_t)k * D], D * sizeof(float));
    }
    ggml_backend_buffer_free(bs);
    ggml_free(cs);
    if (g_profile_cond) { printf("      [cond-v] cond_slat_gpu: readback + interleave + free %.2f\n", secs(t_rb, clk::now())); fflush(stdout); }

    st.peak_bytes = st.weight_bytes + st.cond_bytes + st.view_alloc_bytes;
    st.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

// #82 preflight: size (in uint64, before any size_t narrowing on wasm32) the largest single
// device tensor pixal3d_cond_slat_gpu is about to create. A tensor cannot be split across
// buffers, so this -- not aggregate VRAM -- is what a WebGPU adapter's
// maxStorageBufferBindingSize bounds. Dense R^3 projection ([R^3, 1024] per accumulator, 1 GiB at
// R=64) and the unchunked NAF map ([1024, T^2], 4 GiB at T=1024) are the known offenders; the
// sparse / chunked paths keep both proportional to the active tokens / one chunk.
static bool cond_slat_chunked(const Pixal3dSlatCondParams& prm) {
    // NAF 出力 [D, T*T] が 1 GiB を超える段（= naf_T > 512）、または明示指定があるときは
    // 分割グラフ版へ。T<=512 の既存2段は検証済みの単一グラフ経路のままにする。
    return prm.naf_block_chunk > 0 || (int64_t)prm.naf_T * prm.naf_T * D * 4 > (1LL << 30);
}

Pixal3dCondSlatPlan pixal3d_cond_slat_plan(const Pixal3dSlatCondParams& prm, int64_t tokens, bool sparse) {
    Pixal3dCondSlatPlan p;
    p.sparse = sparse; p.tokens = tokens; p.chunked = cond_slat_chunked(prm);
    const uint64_t S = (uint64_t)prm.S, T = (uint64_t)prm.naf_T, Hp = S / 16, NP = Hp * Hp, C = D;
    const uint64_t d2 = (T / Hp) * (T / Hp), ntok = NP + NPREFIX;
    p.largest_bytes = (uint64_t)D * (uint64_t)tokens * 4;          // acc_lr / acc_hr / gathered taps
    p.what = sparse ? "sparse proj accumulator" : "dense R^3 proj accumulator";
    auto consider = [&](uint64_t b, const char* w) { if (b > p.largest_bytes) { p.largest_bytes = b; p.what = w; } };
    consider(ntok * ntok * 16 * 4, "DINOv3 attention scores");      // dinov3.cpp: soft_max(k^T q) [N, N, 16]
    consider(S * S * 256 * 4, "NAF encoder concat");                // build_encoder cat [S, S, 256]
    consider(T * T * 256 * 4, "NAF pooled / q_bm");
    // naf_build_attn_chunk over nb blocks: v_win [C, 81 nb], logits/probs [81, d2, 4, nb], out [C, d2 nb].
    uint64_t nb = NP;
    if (p.chunked) {
        nb = prm.naf_block_chunk > 0 ? (uint64_t)prm.naf_block_chunk
           : std::max<uint64_t>(1, (uint64_t)256 * 1024 * 1024 / (d2 * D * 4));
        nb = std::min(nb, NP);
    }
    consider(C * 81 * nb * 4, "NAF attention value windows");
    consider(81 * d2 * 4 * nb * 4, "NAF attention scores");
    consider(C * d2 * nb * 4, p.chunked ? "NAF attention chunk" : "full NAF map (unchunked)");
    p.host_proj_bytes = (uint64_t)tokens * 2 * D * 4;
    return p;
}

// Logs the plan and refuses it before any allocation when a single tensor exceeds the backend's
// per-buffer limit (ggml_backend_get_max_size), instead of an allocator failure mid-graph.
static void cond_slat_preflight(ggml_backend_t be, const Pixal3dSlatCondParams& prm, int64_t NT,
                                bool sparse, Pixal3dCondStats& st) {
    const Pixal3dCondSlatPlan p = pixal3d_cond_slat_plan(prm, NT, sparse);
    const uint64_t limit = (uint64_t)ggml_backend_get_max_size(be);
    st.largest_planned_bytes = p.largest_bytes;
    st.backend_max_buffer = limit;
    char lim[32];
    if (limit >= ((uint64_t)1 << 62)) snprintf(lim, sizeof lim, "unbounded");
    else snprintf(lim, sizeof lim, "%.1f MiB", limit / 1048576.0);
    printf("      [cond-mem] S=%d T=%d R=%d tokens=%lld (%s, NAF %s): largest planned tensor %.1f MiB (%s),"
           " backend per-buffer limit %s, host proj %.1f MiB\n",
           prm.S, prm.naf_T, prm.R, (long long)NT, sparse ? "sparse gather" : "DENSE R^3",
           p.chunked ? "chunked" : "single graph", p.largest_bytes / 1048576.0, p.what, lim, p.host_proj_bytes / 1048576.0);
    if (p.largest_bytes > limit)
        throw std::runtime_error("pixal3d_cond_slat_gpu: planned " + std::string(p.what) + " of " +
                                 std::to_string(p.largest_bytes >> 20) + " MiB exceeds the backend per-buffer limit of " +
                                 std::to_string(limit >> 20) + " MiB (S=" + std::to_string(prm.S) + " T=" +
                                 std::to_string(prm.naf_T) + " R=" + std::to_string(prm.R) + " tokens=" +
                                 std::to_string(NT) + "); use sparse coords / a smaller naf_block_chunk");
    if (p.host_proj_bytes > (uint64_t)SIZE_MAX / 2)
        throw std::runtime_error("pixal3d_cond_slat_gpu: host proj buffer of " + std::to_string(p.host_proj_bytes >> 20) +
                                 " MiB does not fit this address space");
}

Pixal3dCond pixal3d_cond_slat_gpu(const Model& dinov3, const Model& naf,
                                   const std::vector<Pixal3dView>& views,
                                   const Pixal3dSlatCondParams& prm, Pixal3dCondStats* stats,
                                   const std::vector<std::array<int, 3>>* coords) {
    const auto t0 = std::chrono::steady_clock::now();
    const int S = prm.S, R = prm.R, Tn = prm.naf_T;
    Pixal3dCond out;
    out.n_global = NPREFIX;
    out.d_proj = 2 * D;
    out.global.assign((size_t)NPREFIX * D, 0.0f);

    const int V = (int)views.size();
    const int Hp = S / 16, Wp = Hp, NP = Hp * Wp;
    const int64_t R3 = (int64_t)R * R * R;
    // 疎な coords が来たら、その token だけを蓄積する（dense R^3 は作らない）。
    // sel[j] = R^3 グリッドでの token 番号 (x*R*R + y*R + z)。
    std::vector<int32_t> sel;
    if (coords) {
        sel.reserve(coords->size());
        for (const auto& c : *coords) {
            if (c[0] < 0 || c[0] >= R || c[1] < 0 || c[1] >= R || c[2] < 0 || c[2] >= R)
                throw std::runtime_error("pixal3d_cond_slat_gpu: coord out of the R^3 grid");
            sel.push_back((int32_t)(((int64_t)c[0] * R + c[1]) * R + c[2]));
        }
    }
    const int64_t N3 = coords ? (int64_t)sel.size() : R3;
    if (V == 0 || N3 == 0) { out.proj.assign((size_t)N3 * 2 * D, 0.0f); return out; }
    if (naf.backend == nullptr || dinov3.backend == nullptr)
        throw std::runtime_error("pixal3d_cond_slat_gpu: models must be loaded on a backend");
    const bool chunked = cond_slat_chunked(prm);
    Pixal3dCondStats pre;
    cond_slat_preflight(dinov3.backend, prm, N3, coords != nullptr, pre);
    out.proj.assign((size_t)N3 * 2 * D, 0.0f);
    if (chunked) {
        Pixal3dCond r = cond_slat_gpu_chunked(dinov3, naf, views, prm, sel, N3, stats);
        if (stats) { stats->largest_planned_bytes = pre.largest_planned_bytes; stats->backend_max_buffer = pre.backend_max_buffer; }
        return r;
    }
    if (naf.backend == nullptr || dinov3.backend == nullptr)
        throw std::runtime_error("pixal3d_cond_slat_gpu: models must be loaded on a backend");

    std::vector<float> c2w_flat((size_t)V * 16);
    for (int v = 0; v < V; ++v)
        std::memcpy(&c2w_flat[(size_t)v * 16], views[v].c2w, 16 * sizeof(float));
    const float* c0 = views[0].c2w;
    const float distance0 = std::sqrt(c0[3] * c0[3] + c0[7] * c0[7] + c0[11] * c0[11]);
    std::vector<float> calc;
    mv_calc_mats(c2w_flat.data(), V, distance0, calc);

    // View-independent host tables: DINOv3 RoPE, NAF RoPE / window / block order, NAF lowering.
    std::vector<float> rcos, rsin;
    dinov3_rope_tables(S, rcos, rsin);
    std::vector<float> ncos, nsin;
    naf_rope_tables(Tn, tensor_to_f32(naf.get("image_encoder.rope.periods")), ncos, nsin);
    std::vector<int32_t> win_idx, raster_of_bm, bm_of_raster;
    naf_window_index(Tn, Hp, Wp, win_idx);
    naf_block_order(Tn, Hp, Wp, raster_of_bm, bm_of_raster);
    const NafGgmlOpts nopts = naf_ggml_opts_for(naf);

    // Persistent accumulators (zeroed), channel-major like the graph tensors they receive.
    ggml_context* pc = ggml_init({ ggml_tensor_overhead() * 4 + 256, nullptr, true });
    T* acc_glob = ggml_new_tensor_2d(pc, GGML_TYPE_F32, D, NPREFIX);
    T* acc_lr   = ggml_new_tensor_2d(pc, GGML_TYPE_F32, D, N3);
    T* acc_hr   = ggml_new_tensor_2d(pc, GGML_TYPE_F32, D, N3);
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pc, dinov3.backend);
    if (!pbuf) throw std::runtime_error("pixal3d_cond_slat_gpu: accumulator alloc failed");
    ggml_backend_buffer_clear(pbuf, 0);

    Pixal3dCondStats st;
    st.views = V;
    st.weight_bytes = (dinov3.buffer ? ggml_backend_buffer_get_size(dinov3.buffer) : 0)
                    + (naf.buffer ? ggml_backend_buffer_get_size(naf.buffer) : 0);
    st.cond_bytes = ggml_backend_buffer_get_size(pbuf);

    for (int v = 0; v < V; ++v) {
        const auto tv = std::chrono::steady_clock::now();
        std::vector<float> normed = pixal3d_imagenet_normalize(views[v].rgb_premult, S);

        Camera cam{};
        cam.has_c2w = true;
        cam.mesh_scale = prm.mesh_scale;
        cam.fov_x = views[v].fov_x;
        for (int i = 0; i < 16; ++i) cam.c2w[i] = calc[(size_t)v * 16 + i];
        std::vector<int32_t> idx_lr[4], idx_hr[4];
        std::vector<float> w_lr[4], w_hr[4];
        proj_grid_bilinear_taps(Hp, Wp, R, S, cam, idx_lr, w_lr);
        proj_grid_bilinear_taps(Tn, Tn, R, S, cam, idx_hr, w_hr);
        for (int t = 0; t < 4; ++t)                       // NAF map rows are in block-major pixel order
            for (int32_t& i : idx_hr[t]) i = bm_of_raster[i];
        if (coords) {                                     // dense な tap 列から active token 分だけ抜く
            for (int t = 0; t < 4; ++t) {
                std::vector<int32_t> il(sel.size()), ih(sel.size());
                std::vector<float> wl(sel.size()), wh(sel.size());
                for (size_t j = 0; j < sel.size(); ++j) {
                    il[j] = idx_lr[t][sel[j]]; wl[j] = w_lr[t][sel[j]];
                    ih[j] = idx_hr[t][sel[j]]; wh[j] = w_hr[t][sel[j]];
                }
                idx_lr[t].swap(il); w_lr[t].swap(wl); idx_hr[t].swap(ih); w_hr[t].swap(wh);
            }
        }

        const auto t_taps = std::chrono::steady_clock::now();   // --profile-cond laps (host taps / build+alloc / upload / compute / free)
        size_t meta = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(16384, false) + (1 << 20);
        ggml_context* c = ggml_init({ meta, nullptr, true });
        Dinov3Inputs in{};
        T* x = dinov3_build(c, dinov3, S, in);                       // [D, Ntok]
        T* glob = ggml_view_2d(c, x, D, NPREFIX, x->nb[1], 0);      // [D, 5]
        T* patches = ggml_view_2d(c, x, D, NP, x->nb[1], (size_t)NPREFIX * x->nb[1]);  // [D, Hp*Wp] == lr fmap, pixel-major

        NafGraphInputs nin;
        T* hr = naf_build(c, naf, S, Tn, Hp, Wp, patches, nin, nopts);   // [D, T*T] block-major pixel-major

        T *gidx_lr[4], *gw_lr[4], *gidx_hr[4], *gw_hr[4];
        T* z_lr = nullptr; T* z_hr = nullptr;
        for (int t = 0; t < 4; ++t) {
            gidx_lr[t] = ggml_new_tensor_1d(c, GGML_TYPE_I32, N3);   ggml_set_input(gidx_lr[t]);
            gw_lr[t]   = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, N3); ggml_set_input(gw_lr[t]);
            gidx_hr[t] = ggml_new_tensor_1d(c, GGML_TYPE_I32, N3);   ggml_set_input(gidx_hr[t]);
            gw_hr[t]   = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, N3); ggml_set_input(gw_hr[t]);
            T* tl = ggml_mul(c, ggml_get_rows(c, patches, gidx_lr[t]), gw_lr[t]);   // [D, R^3]
            T* th = ggml_mul(c, ggml_get_rows(c, hr, gidx_hr[t]), gw_hr[t]);        // [D, R^3]
            z_lr = z_lr ? ggml_add(c, z_lr, tl) : tl;
            z_hr = z_hr ? ggml_add(c, z_hr, th) : th;
        }
        const float inv_v = 1.0f / (float)V;
        T* wg = ggml_cpy(c, ggml_add(c, acc_glob, ggml_scale(c, glob, inv_v)), acc_glob);
        T* wl = ggml_cpy(c, ggml_add(c, acc_lr, ggml_scale(c, z_lr, inv_v)), acc_lr);
        T* wh = ggml_cpy(c, ggml_add(c, acc_hr, ggml_scale(c, z_hr, inv_v)), acc_hr);

        ggml_cgraph* g = ggml_new_graph_custom(c, 16384, false);
        ggml_build_forward_expand(g, wg);
        ggml_build_forward_expand(g, wl);
        ggml_build_forward_expand(g, wh);
        const std::string tag = "pixal3d_cond_slat_gpu_S" + std::to_string(S) + "_R" + std::to_string(R)
                              + "_T" + std::to_string(Tn) + "_v" + std::to_string(v);
        trellis_graph_dump(tag.c_str(), g);
        check_graph_supported(dinov3.backend, g, tag.c_str());

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dinov3.backend));
        if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error("pixal3d_cond_slat_gpu: alloc failed");
        const size_t ab = ggml_gallocr_get_buffer_size(alloc, 0);
        if (ab > st.view_alloc_bytes) st.view_alloc_bytes = ab;
        const auto t_alloc = std::chrono::steady_clock::now();

        ggml_backend_tensor_set(in.img, normed.data(), 0, normed.size() * 4);
        ggml_backend_tensor_set(in.cos, rcos.data(), 0, rcos.size() * 4);
        ggml_backend_tensor_set(in.sin, rsin.data(), 0, rsin.size() * 4);
        ggml_backend_tensor_set(nin.img, views[v].rgb_premult.data(), 0, views[v].rgb_premult.size() * 4);
        ggml_backend_tensor_set(nin.rope_cos, ncos.data(), 0, ncos.size() * 4);
        ggml_backend_tensor_set(nin.rope_sin, nsin.data(), 0, nsin.size() * 4);
        ggml_backend_tensor_set(nin.win_idx, win_idx.data(), 0, win_idx.size() * sizeof(int32_t));
        ggml_backend_tensor_set(nin.blk_idx, raster_of_bm.data(), 0, raster_of_bm.size() * sizeof(int32_t));
        for (int t = 0; t < 4; ++t) {
            ggml_backend_tensor_set(gidx_lr[t], idx_lr[t].data(), 0, idx_lr[t].size() * sizeof(int32_t));
            ggml_backend_tensor_set(gw_lr[t], w_lr[t].data(), 0, w_lr[t].size() * sizeof(float));
            ggml_backend_tensor_set(gidx_hr[t], idx_hr[t].data(), 0, idx_hr[t].size() * sizeof(int32_t));
            ggml_backend_tensor_set(gw_hr[t], w_hr[t].data(), 0, w_hr[t].size() * sizeof(float));
        }
        const auto t_up = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(dinov3.backend, g) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("pixal3d_cond_slat_gpu: compute failed");
        ggml_backend_synchronize(dinov3.backend);
        const auto t_comp = std::chrono::steady_clock::now();
        const int n_nodes = ggml_graph_n_nodes(g);   // g lives in c: read before ggml_free
        ggml_gallocr_free(alloc);   // release this view's temporaries before the next view
        ggml_free(c);
        const auto t_end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t_end - tv).count();
        if (ms > st.view_ms_max) st.view_ms_max = ms;
        if (g_profile_cond) {
            auto sec = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
            printf("      [cond-v] cond_slat_gpu_single S=%d R=%d T=%d view %d: host taps %.2f | build+alloc %.2f | upload %.2f | compute %.2f | free %.2f  (view %.2fs, graph %.0f MB, %d nodes)\n",
                   S, R, Tn, v, sec(tv, t_taps), sec(t_taps, t_alloc), sec(t_alloc, t_up), sec(t_up, t_comp), sec(t_comp, t_end), ms / 1e3,
                   ab / 1048576.0, n_nodes);
            fflush(stdout);
        }
    }

    // One readback: ggml channel-major [D, tok] -> host token-major, proj = [lr || hr] per token.
    std::vector<float> hg = tensor_to_f32(acc_glob), hl = tensor_to_f32(acc_lr), hh = tensor_to_f32(acc_hr);
    for (int t = 0; t < NPREFIX; ++t)
        for (int c = 0; c < D; ++c) out.global[(size_t)t * D + c] = hg[(size_t)t * D + c];
    for (int64_t k = 0; k < N3; ++k) {
        float* dst = &out.proj[(size_t)k * 2 * D];
        std::memcpy(dst, &hl[(size_t)k * D], D * sizeof(float));
        std::memcpy(dst + D, &hh[(size_t)k * D], D * sizeof(float));
    }

    ggml_backend_buffer_free(pbuf);
    ggml_free(pc);

    st.largest_planned_bytes = pre.largest_planned_bytes;
    st.backend_max_buffer = pre.backend_max_buffer;
    st.peak_bytes = st.weight_bytes + st.cond_bytes + st.view_alloc_bytes;
    st.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

} // namespace trellis
