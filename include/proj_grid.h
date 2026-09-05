// Camera-aware pixel-aligned projection grid (Pixal3D ProjGrid / ProjGridMV port).
//
// Ports trainers/flow_matching/mixins/image_conditioned_proj.py (ProjGrid,
// ProjGridMV, project_points_to_image_batch, sample_features,
// compute_relative_calc_mat) faithfully for a single batch element (B=1) —
// batch dimension is handled by the caller looping views/batches.
#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

// Camera parameters for a single view / batch element.
struct Camera {
    float fov_x;       // camera_angle_x, radians (horizontal FOV)
    float distance;    // used only when has_c2w == false (front-view default)
    float mesh_scale;
    float c2w[16];     // row-major 4x4 camera-to-world; used only when has_c2w == true
    bool  has_c2w;
};

// Grid points: linspace(-1,1,R) meshgrid (indexing='ij') @ Rot^T, reshaped to
// [R^3, 3] with token index k = ix*R*R + iy*R + iz (z fastest).
// Rot = [[1,0,0],[0,0,-1],[0,1,0]].
// out: [R^3 * 3], row-major (token, then x/y/z).
void proj_grid_points(int R, std::vector<float>& out);

// Projects the R^3 grid (scaled by cam.mesh_scale as in ProjGrid.forward:
// grid_points / mesh_scale / 2) through the camera and computes pixel and
// grid_sample-normalized coordinates.
//   pix:   [R^3 * 2]  (x_pixel, y_pixel), image_resolution S
//   norm:  [R^3 * 2]  (x_norm, y_norm) = (pix + 0.5) / S * 2 - 1
//   valid: [R^3]      1 if 0<=x_pix<S && 0<=y_pix<S && depth>0, else 0
// When cam.has_c2w is false, uses the front-view transform matrix
// F = [[1,0,0,0],[0,0,-1,-2],[0,1,0,0],[0,0,0,1]] with F[1][3] = -cam.distance.
// When cam.has_c2w is true, uses cam.c2w directly as the transform matrix
// (already the per-view calc_mat for multiview use — see mv_calc_mats).
void proj_grid_project(int R, int S, const Camera& cam,
                        std::vector<float>& pix, std::vector<float>& norm,
                        std::vector<uint8_t>& valid);

// Full ProjGrid/ProjGridMV forward for one view: project the R^3 grid through
// cam and bilinear-sample fmap (grid_sample, align_corners=False,
// padding_mode='border') at the normalized coordinates.
//   fmap: [C][H][W] channel-major (C planes of H*W floats each) — this is the
//         [C,H,W] layout torch uses internally in sample_features (caller
//         converts BHWC -> CHW before calling).
// Returns [R^3 * C] token-major (token k, then channel c) i.e. torch [R^3, C].
std::vector<float> proj_grid_sample(const float* fmap, int C, int H, int W,
                                     int R, int S, const Camera& cam);

// compute_relative_calc_mat for one batch element:
//   calc_mat_i = F' @ inv(C_0) @ C_i,  F' = front-view matrix with F'[1][3] = -distance0.
// c2w: [V*16] row-major 4x4 c2w matrices (view 0 = main view).
// calc: [V*16] output row-major 4x4 calc_mat per view (calc_mat_0 == F' exactly).
// Inversion is done in double precision.
void mv_calc_mats(const float* c2w, int V, float distance0, std::vector<float>& calc);

} // namespace trellis
