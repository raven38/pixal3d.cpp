// Pixal3D SS-stage image conditioning: per-view DINOv3 features -> global (CLS+4reg,
// averaged over views) and proj (ProjGridMV-sampled patch tokens, averaged over views).
// Ports the "ss" stage of DinoV3ProjMultiViewFeatureExtractor
// (pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py) -- see
// docs/spec/30-pixal3d-cond.md sections 1-3. Also declares the SLAT-stage
// (shape/texture) variant, which additionally fuses in NAF high-res features
// (docs/spec/30-pixal3d-cond.md section 4) -- proj = [lr || hr], 2048-wide.
#pragma once
#include <array>
#include <cstddef>
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

// Per-view / per-stage resource accounting for the device-resident conditioning path.
struct Pixal3dCondStats {
    size_t weight_bytes = 0;     // DINOv3 weight buffer (resident for the whole call)
    size_t cond_bytes = 0;       // persistent accumulators (global + proj) on the device
    size_t view_alloc_bytes = 0; // largest per-view graph buffer (DINOv3 activations + projection temporaries)
    size_t peak_bytes = 0;       // weight_bytes + cond_bytes + view_alloc_bytes (buffers are freed per view)
    double total_ms = 0;         // wall time of the whole call
    double view_ms_max = 0;      // slowest single view (graph build + alloc + compute)
    int views = 0;
};

// Device-resident variant of pixal3d_cond_ss with identical semantics: per view, one ggml graph
// runs DINOv3 -> pixel-aligned bilinear projection (proj_grid_bilinear_taps: get_rows x4 +
// weighted sum over the [1024, Hp*Wp] patch map, never leaving the device) -> running average
// into two persistent accumulators on the model's backend; the view's temporaries are released
// before the next view, so peak memory is independent of V. The fused global/proj are read back
// once at the end (same host layout as pixal3d_cond_ss). Numerics differ from the host path
// only by f32 (device) vs f64 (host) accumulation of the four bilinear taps.
Pixal3dCond pixal3d_cond_ss_gpu(const Model& dinov3, const std::vector<Pixal3dView>& views,
                                 int S, int R, float mesh_scale, Pixal3dCondStats* stats = nullptr);

// SLAT-stage (shape or texture) config: S = view image resolution (512 or 1024),
// R = ProjGrid resolution (32 or 64), naf_T = NAF target resolution (512 for the
// shape stages, 1024 for the texture stage; see docs/spec/30-pixal3d-cond.md
// section 2 table).
struct Pixal3dSlatCondParams {
    int S;
    int R;
    int naf_T;
    float mesh_scale;
};

// Computes the Pixal3D SLAT-stage (shape/texture) fused condition from V posed
// views (view 0 = main/front view). Per view: ImageNet-normalize -> dinov3_encode(S)
// -> split into CLS+4reg (global) and the [1024,Hp,Wp] patch map (Hp=Wp=S/16, same
// split as pixal3d_cond_ss) -> lr = proj_grid_sample(patch map, R, S, calc_mat_v);
// hr = proj_grid_sample(naf_upsample(naf, views[v].rgb_premult (un-normalized,
// premultiplied), S, patch map, 1024, Hp, Wp, naf_T), R, S, calc_mat_v); proj_v =
// [lr || hr] (2048-wide per token). global and proj are averaged over views,
// accumulating one view at a time (never holds all V views' buffers at once).
Pixal3dCond pixal3d_cond_slat(const Model& dinov3, const Model& naf,
                               const std::vector<Pixal3dView>& views,
                               const Pixal3dSlatCondParams& prm);

// Gathers the dense SLAT proj condition [R^3, C] (token k = x*R*R + y*R + z, as
// produced by pixal3d_cond_slat) at a sparse set of active voxel coords (x,y,z),
// producing the [N, C] layout the SLAT DiT's SparseTensor feats consume.
std::vector<float> pixal3d_gather_proj(const std::vector<float>& proj_dense, int R, int C,
                                        const std::vector<std::array<int, 3>>& coords);

} // namespace trellis
