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
#include "ggml.h"
#include <functional>
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <chrono>
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

    st.peak_bytes = st.weight_bytes + st.cond_bytes + st.view_alloc_bytes;
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
    ggml_backend_buffer_t bp = alloc_one(cp, P_pooled, TT, 256);   // pooled を [T*T, 256] として持つ
    ggml_backend_buffer_t bq = alloc_one(cq, Q_bm, 256, TT);
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
    st.cond_bytes = ggml_backend_buffer_get_size(bp) + ggml_backend_buffer_get_size(bq)
                  + ggml_backend_buffer_get_size(bs);

    // グラフを1本組んで走らせる小ヘルパ（入力の投入は alloc 後に upload() で行う）。
    auto run_graph = [&](ggml_context* c, const std::vector<T*>& outs, const char* tag,
                         const std::function<void()>& upload) {
        ggml_cgraph* g = ggml_new_graph_custom(c, 16384, false);
        for (T* o : outs) ggml_build_forward_expand(g, o);
        trellis_graph_dump(tag, g);
        check_graph_supported(dinov3.backend, g, tag);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dinov3.backend));
        if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error(std::string("alloc failed: ") + tag);
        const size_t ab = ggml_gallocr_get_buffer_size(alloc, 0);
        if (ab > st.view_alloc_bytes) st.view_alloc_bytes = ab;
        if (getenv("TRELLIS_DBG_COND")) fprintf(stderr, "[cond] %-40s graph buffer %.1f MB (%d nodes)\n", tag, ab / 1048576.0, ggml_graph_n_nodes(g));
        upload();
        if (ggml_backend_graph_compute(dinov3.backend, g) != GGML_STATUS_SUCCESS)
            throw std::runtime_error(std::string("compute failed: ") + tag);
        ggml_backend_synchronize(dinov3.backend);
        ggml_gallocr_free(alloc);
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

        // ---- G4: block chunk ごとに attention -> その chunk の hr tap を累積 ----
        for (int blk0 = 0; blk0 < nblk; blk0 += chunk_blk) {
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
            if (!any) continue;   // この chunk に落ちる tap が無い

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
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count();
        if (ms > st.view_ms_max) st.view_ms_max = ms;
    }

    std::vector<float> hg = tensor_to_f32(acc_glob), hl = tensor_to_f32(acc_lr), hh = tensor_to_f32(acc_hr);
    for (int t = 0; t < NPREFIX; ++t)
        for (int c = 0; c < D; ++c) out.global[(size_t)t * D + c] = hg[(size_t)t * D + c];
    for (int64_t k = 0; k < NT; ++k) {
        float* dst = &out.proj[(size_t)k * 2 * D];
        std::memcpy(dst, &hl[(size_t)k * D], D * sizeof(float));
        std::memcpy(dst + D, &hh[(size_t)k * D], D * sizeof(float));
    }
    ggml_backend_buffer_free(bp); ggml_backend_buffer_free(bq); ggml_backend_buffer_free(bs);
    ggml_free(cp); ggml_free(cq); ggml_free(cs);

    st.peak_bytes = st.weight_bytes + st.cond_bytes + st.view_alloc_bytes;
    st.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = st;
    return out;
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
    out.proj.assign((size_t)N3 * 2 * D, 0.0f);
    if (V == 0 || N3 == 0) return out;
    if (naf.backend == nullptr || dinov3.backend == nullptr)
        throw std::runtime_error("pixal3d_cond_slat_gpu: models must be loaded on a backend");
    // NAF 出力 [D, T*T] が 1 GiB を超える段（= naf_T > 512）、または明示指定があるときは
    // 分割グラフ版へ。T<=512 の既存2段は検証済みの単一グラフ経路のままにする。
    if (prm.naf_block_chunk > 0 || (int64_t)Tn * Tn * D * 4 > (1LL << 30))
        return cond_slat_gpu_chunked(dinov3, naf, views, prm, sel, N3, stats);
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
        if (ggml_backend_graph_compute(dinov3.backend, g) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("pixal3d_cond_slat_gpu: compute failed");
        ggml_backend_synchronize(dinov3.backend);
        ggml_gallocr_free(alloc);   // release this view's temporaries before the next view
        ggml_free(c);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count();
        if (ms > st.view_ms_max) st.view_ms_max = ms;
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

    st.peak_bytes = st.weight_bytes + st.cond_bytes + st.view_alloc_bytes;
    st.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

} // namespace trellis
