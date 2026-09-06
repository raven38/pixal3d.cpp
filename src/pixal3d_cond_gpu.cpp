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
#include "proj_grid.h"
#include "trellis_model.h"
#include "graph_dump.h"
#include "ggml.h"
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

} // namespace trellis
