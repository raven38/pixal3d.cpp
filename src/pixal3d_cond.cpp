#include "pixal3d_cond.h"
#include "dinov3.h"
#include "naf.h"
#include "proj_grid.h"

#include <cmath>
#include <cstring>

namespace trellis {

static constexpr int NPREFIX = 5, D = 1024;

std::vector<float> pixal3d_imagenet_normalize(const std::vector<float>& rgb_premult, int S) {
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float sd[3]   = {0.229f, 0.224f, 0.225f};
    std::vector<float> out(rgb_premult.size());
    const size_t plane = (size_t)S * S;
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < plane; ++i)
            out[c * plane + i] = (rgb_premult[c * plane + i] - mean[c]) / sd[c];
    return out;
}

// dinov3_encode returns [Ntok, D] token-major (each token's D channels contiguous, i.e.
// torch [Ntok, D]). Convert the patch tokens (indices NPREFIX..Ntok-1, row-major
// token=h*Wp+w) into channel-major [D, Hp, Wp] for proj_grid_sample's fmap layout.
static std::vector<float> patch_tokens_to_chw(const std::vector<float>& tok, int Hp, int Wp) {
    std::vector<float> out((size_t)D * Hp * Wp);
    for (int h = 0; h < Hp; ++h) {
        for (int w = 0; w < Wp; ++w) {
            const float* src = tok.data() + (size_t)(NPREFIX + h * Wp + w) * D;
            for (int c = 0; c < D; ++c)
                out[(size_t)c * Hp * Wp + (size_t)h * Wp + w] = src[c];
        }
    }
    return out;
}

Pixal3dCond pixal3d_cond_ss(const Model& dinov3, const std::vector<Pixal3dView>& views,
                             int S, int R, float mesh_scale) {
    Pixal3dCond out;
    out.n_global = NPREFIX;
    out.d_proj = D;
    out.global.assign((size_t)NPREFIX * D, 0.0f);
    out.proj.assign((size_t)R * R * R * D, 0.0f);

    const int V = (int)views.size();
    if (V == 0) return out;
    const int Hp = S / 16, Wp = Hp;

    std::vector<float> c2w_flat((size_t)V * 16);
    for (int v = 0; v < V; ++v)
        std::memcpy(&c2w_flat[(size_t)v * 16], views[v].c2w, 16 * sizeof(float));
    const float* c0 = views[0].c2w;
    const float distance0 = std::sqrt(c0[3] * c0[3] + c0[7] * c0[7] + c0[11] * c0[11]);

    std::vector<float> calc;
    mv_calc_mats(c2w_flat.data(), V, distance0, calc);

    for (int v = 0; v < V; ++v) {
        std::vector<float> normed = pixal3d_imagenet_normalize(views[v].rgb_premult, S);
        std::vector<float> tok = dinov3_encode(dinov3, normed, S);  // [Ntok, D] token-major

        for (int t = 0; t < NPREFIX; ++t)
            for (int c = 0; c < D; ++c)
                out.global[(size_t)t * D + c] += tok[(size_t)t * D + c] / (float)V;

        std::vector<float> chw = patch_tokens_to_chw(tok, Hp, Wp);

        Camera cam{};
        cam.has_c2w = true;
        cam.mesh_scale = mesh_scale;
        cam.fov_x = views[v].fov_x;
        for (int i = 0; i < 16; ++i) cam.c2w[i] = calc[(size_t)v * 16 + i];

        std::vector<float> zv = proj_grid_sample(chw.data(), D, Hp, Wp, R, S, cam);  // [R^3 * D]
        for (size_t i = 0; i < zv.size(); ++i) out.proj[i] += zv[i] / (float)V;
    }

    return out;
}

Pixal3dCond pixal3d_cond_slat(const Model& dinov3, const Model& naf,
                               const std::vector<Pixal3dView>& views,
                               const Pixal3dSlatCondParams& prm) {
    const int S = prm.S, R = prm.R, T = prm.naf_T;
    const int Dproj = 2 * D; // [lr || hr], 1024 + 1024

    Pixal3dCond out;
    out.n_global = NPREFIX;
    out.d_proj = Dproj;
    out.global.assign((size_t)NPREFIX * D, 0.0f);
    out.proj.assign((size_t)R * R * R * Dproj, 0.0f);

    const int V = (int)views.size();
    if (V == 0) return out;
    const int Hp = S / 16, Wp = Hp;

    std::vector<float> c2w_flat((size_t)V * 16);
    for (int v = 0; v < V; ++v)
        std::memcpy(&c2w_flat[(size_t)v * 16], views[v].c2w, 16 * sizeof(float));
    const float* c0 = views[0].c2w;
    const float distance0 = std::sqrt(c0[3] * c0[3] + c0[7] * c0[7] + c0[11] * c0[11]);

    std::vector<float> calc;
    mv_calc_mats(c2w_flat.data(), V, distance0, calc);

    for (int v = 0; v < V; ++v) {
        std::vector<float> normed = pixal3d_imagenet_normalize(views[v].rgb_premult, S);
        std::vector<float> tok = dinov3_encode(dinov3, normed, S);  // [Ntok, D] token-major

        for (int t = 0; t < NPREFIX; ++t)
            for (int c = 0; c < D; ++c)
                out.global[(size_t)t * D + c] += tok[(size_t)t * D + c] / (float)V;

        std::vector<float> chw = patch_tokens_to_chw(tok, Hp, Wp);  // [D, Hp, Wp], LR fmap

        Camera cam{};
        cam.has_c2w = true;
        cam.mesh_scale = prm.mesh_scale;
        cam.fov_x = views[v].fov_x;
        for (int i = 0; i < 16; ++i) cam.c2w[i] = calc[(size_t)v * 16 + i];

        std::vector<float> lr = proj_grid_sample(chw.data(), D, Hp, Wp, R, S, cam);  // [R^3, D]

        // NAF gets the un-normalized, alpha-premultiplied image (not the
        // ImageNet-normalized `normed` used for DINOv3).
        std::vector<float> naf_map =
            naf_upsample(naf, views[v].rgb_premult.data(), S, chw.data(), D, Hp, Wp, T);  // [D, T, T]
        std::vector<float> hr = proj_grid_sample(naf_map.data(), D, T, T, R, S, cam);  // [R^3, D]
        naf_map.clear();
        naf_map.shrink_to_fit();

        const size_t Ntok = (size_t)R * R * R;
        for (size_t i = 0; i < Ntok; ++i) {
            float* dst = &out.proj[i * (size_t)Dproj];
            const float* srcL = &lr[i * (size_t)D];
            const float* srcH = &hr[i * (size_t)D];
            for (int c = 0; c < D; ++c) dst[c] += srcL[c] / (float)V;
            for (int c = 0; c < D; ++c) dst[D + c] += srcH[c] / (float)V;
        }
    }

    return out;
}

std::vector<float> pixal3d_gather_proj(const std::vector<float>& proj_dense, int R, int C,
                                        const std::vector<std::array<int, 3>>& coords) {
    std::vector<float> out((size_t)coords.size() * (size_t)C);
    for (size_t n = 0; n < coords.size(); ++n) {
        const int x = coords[n][0], y = coords[n][1], z = coords[n][2];
        const size_t tok = (size_t)x * R * R + (size_t)y * R + (size_t)z;
        std::memcpy(&out[n * (size_t)C], &proj_dense[tok * (size_t)C], (size_t)C * sizeof(float));
    }
    return out;
}

} // namespace trellis
