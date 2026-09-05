// Pixal3D SS-stage image conditioning: per-view DINOv3 features -> global (CLS+4reg,
// averaged over views) and proj (ProjGridMV-sampled patch tokens, averaged over views).
// Ports the "ss" stage of DinoV3ProjMultiViewFeatureExtractor
// (pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py) -- see
// docs/spec/30-pixal3d-cond.md sections 1-3. SLAT stages additionally need NAF
// (docs/spec/30-pixal3d-cond.md section 4) and are NOT covered here.
#pragma once
#include <vector>

namespace trellis {
struct Model;

// One posed input view. rgb_premult: torch [3,S,S] memory (== ggml [S,S,3,1]),
// values in [0,1], RGB already premultiplied by alpha (inference_mv.to_cond_tensor);
// NOT ImageNet-normalized -- pixal3d_cond_ss normalizes internally before DINOv3.
// c2w: row-major 4x4 camera-to-world (Blender/NeRF convention). View 0 must be the
// main/front view (compute_relative_calc_mat uses its distance as distance0).
struct Pixal3dView {
    std::vector<float> rgb_premult;
    float fov_x;
    float c2w[16];
};

// global: [n_global*1024] token-major (CLS, then 4 register tokens), averaged over views.
// proj:   [R^3 * d_proj] token-major (token k = ix*R*R + iy*R + iz), averaged over views.
struct Pixal3dCond {
    std::vector<float> global;
    std::vector<float> proj;
    int n_global = 5;
    int d_proj = 0;
};

// ImageNet-normalizes an alpha-premultiplied [0,1] torch [3,S,S] image into the tensor
// dinov3_encode expects (dinov3_encode itself does not normalize -- see src/dinov3.cpp /
// src/preprocess.cpp::normalize_cutout, which normalizes before calling it).
std::vector<float> pixal3d_imagenet_normalize(const std::vector<float>& rgb_premult, int S);

// Computes the Pixal3D SS-stage condition from V posed views (view 0 = main/front view).
// Per view: ImageNet-normalize -> dinov3_encode(S) -> split into CLS+4reg (global) and the
// [1024,Hp,Wp] patch map (Hp=Wp=S/16) -> proj_grid_sample(patch map, R, S,
// Camera{fov_x_v, mesh_scale, calc_mat_v}); calc_mat via mv_calc_mats(c2w, V, distance0),
// distance0 = norm(views[0].c2w[:3,3]). global and proj are averaged over views.
Pixal3dCond pixal3d_cond_ss(const Model& dinov3, const std::vector<Pixal3dView>& views,
                             int S, int R, float mesh_scale);

} // namespace trellis
