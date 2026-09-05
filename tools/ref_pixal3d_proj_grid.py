#!/usr/bin/env python3
"""Golden dump for Pixal3D ProjGrid / ProjGridMV geometry + grid_sample (Phase 0, F1).

Pure geometry (camera projection + F.grid_sample) — no model weights, runs on CPU
in f32. Mirrors the projection math in
pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:
  ProjGrid.forward, ProjGridMV.forward, project_points_to_image_batch,
  compute_relative_calc_mat.

Cases (camera params taken from assets/mv_images/example/transforms.json):
  A   single-view ProjGrid, grid_resolution=16, image_resolution=512, BHWC fmap
  A32 same but grid_resolution=32                              (r32_ prefix)
  B   BCHW (BHWC=False) high-res feature-map path, same camera (hr_ prefix)
  C   ProjGridMV multiview (V=3), calc_mat via compute_relative_calc_mat (mv_ prefix)

Deviation from the literal fixture list in the task spec: Case C's transform_matrix
/camera_angle_x/distance/calc_mat are saved with an "mv_" prefix (not bare names)
to avoid colliding with Case A's single-view files of the same base name in the
same output directory.

    PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D OUT=/mnt/hdd1/pixal3d/ref/pixal3d/proj_grid \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_proj_grid.py
"""
import os, sys, json
PIXAL3D_REPO = os.environ.get("PIXAL3D_REPO", "/mnt/hdd1/pixal3d/Pixal3D")
sys.path.insert(0, PIXAL3D_REPO)
import numpy as np
import torch

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid, ProjGridMV, project_points_to_image_batch, compute_relative_calc_mat,
)

OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/proj_grid")
os.makedirs(OUT, exist_ok=True)
VIEWS_DIR = os.environ.get("VIEWS_DIR", os.path.join(PIXAL3D_REPO, "assets/mv_images/example"))
SEED = 42
DEV = "cpu"
DT = torch.float32
C = 64  # fmap channel count used throughout

shapes = {}
def save(name, t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:24s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert not np.isnan(a).any(), f"{name} has NaNs"
    return a

with open(os.path.join(VIEWS_DIR, "transforms.json")) as f:
    meta = json.load(f)
frames = meta["frames"]
top_cax = float(meta["camera_angle_x"])
mesh_scale_val = float(meta.get("mesh_scale", 1.0))

def frame_cax(fr):
    return float(fr.get("camera_angle_x", top_cax))

def frame_tm(fr):
    return torch.tensor(fr["transform_matrix"], dtype=DT)

tm0 = frame_tm(frames[0])
dist0 = float(torch.norm(tm0[:3, 3]))
cax0 = frame_cax(frames[0])
print(f"frame0: camera_angle_x={cax0:.6f} distance={dist0:.6f} mesh_scale={mesh_scale_val:.4f}")

cax_t = torch.tensor([cax0], dtype=DT)
dist_t = torch.tensor([dist0], dtype=DT)
scale_t = torch.tensor([mesh_scale_val], dtype=DT)

# ---------------- Case A: single view, grid_resolution=16 ----------------
print("\n=== Case A (grid_resolution=16, image_resolution=512) ===")
proj16 = ProjGrid(grid_resolution=16, image_resolution=512)
g_a = torch.Generator(device=DEV).manual_seed(SEED)
fmap = torch.randn(1, 32, 32, C, generator=g_a, device=DEV, dtype=DT)

save("fmap_bhwc", fmap)
save("camera_angle_x", cax_t)
save("distance", dist_t)
save("mesh_scale", scale_t)
save("grid_points", proj16.grid_points)

# replicate ProjGrid's internal geometry manually to expose the intermediates
gp_scaled = proj16.grid_points.expand(1, -1, -1) / scale_t.unsqueeze(-1).unsqueeze(-1) / 2
tm = proj16.front_view_transform_matrix.expand(1, -1, -1).clone()
tm[:, 1, 3] = -dist_t
pts2d, depth, valid = project_points_to_image_batch(gp_scaled, tm, cax_t, proj16.image_resolution)
save("points_2d", pts2d)
save("depth", depth)
save("valid_mask", valid.to(DT))
img_norm = (pts2d + 0.5) / proj16.image_resolution * 2 - 1
save("image_points_norm", img_norm)

z_proj = proj16(fmap, cax_t, dist_t, scale_t, transform_matrix=None, BHWC=True)
save("z_proj", z_proj)

# ---------------- Case A32: grid_resolution=32 ----------------
print("\n=== Case A32 (grid_resolution=32, image_resolution=512) ===")
proj32 = ProjGrid(grid_resolution=32, image_resolution=512)
g_a32 = torch.Generator(device=DEV).manual_seed(SEED + 1)
fmap32 = torch.randn(1, 32, 32, C, generator=g_a32, device=DEV, dtype=DT)

save("r32_fmap_bhwc", fmap32)
save("r32_camera_angle_x", cax_t)
save("r32_distance", dist_t)
save("r32_mesh_scale", scale_t)
save("r32_grid_points", proj32.grid_points)

gp32_scaled = proj32.grid_points.expand(1, -1, -1) / scale_t.unsqueeze(-1).unsqueeze(-1) / 2
tm32 = proj32.front_view_transform_matrix.expand(1, -1, -1).clone()
tm32[:, 1, 3] = -dist_t
pts2d32, depth32, valid32 = project_points_to_image_batch(gp32_scaled, tm32, cax_t, proj32.image_resolution)
save("r32_points_2d", pts2d32)
save("r32_depth", depth32)
save("r32_valid_mask", valid32.to(DT))
img_norm32 = (pts2d32 + 0.5) / proj32.image_resolution * 2 - 1
save("r32_image_points_norm", img_norm32)

z_proj32 = proj32(fmap32, cax_t, dist_t, scale_t, transform_matrix=None, BHWC=True)
save("r32_z_proj", z_proj32)

# ---------------- Case B: BCHW (high-res / NAF-style) path ----------------
print("\n=== Case B (BCHW, BHWC=False, high-res 128x128) ===")
g_b = torch.Generator(device=DEV).manual_seed(SEED + 2)
hr_fmap = torch.randn(1, C, 128, 128, generator=g_b, device=DEV, dtype=DT)
save("hr_fmap_bchw", hr_fmap)
hr_z_proj = proj16(hr_fmap, cax_t, dist_t, scale_t, transform_matrix=None, BHWC=False)
save("hr_z_proj", hr_z_proj)

# ---------------- Case C: ProjGridMV multiview (V=3) ----------------
print("\n=== Case C (ProjGridMV, V=3) ===")
V = 3
proj_mv = ProjGridMV(grid_resolution=16, image_resolution=512)
mv_frames = frames[:V]
mv_tm = torch.stack([frame_tm(fr) for fr in mv_frames], dim=0)[None]      # [1,V,4,4]
mv_dist = torch.norm(mv_tm[:, :, :3, 3], dim=-1)                          # [1,V]
mv_cax = torch.tensor([[frame_cax(fr) for fr in mv_frames]], dtype=DT)    # [1,V]
save("mv_transform_matrix", mv_tm)
save("mv_camera_angle_x", mv_cax)
save("mv_distance", mv_dist)

mv_calc_mat = compute_relative_calc_mat(mv_tm, mv_dist, proj_mv.front_view_transform_matrix)
save("mv_calc_mat", mv_calc_mat)

g_c = torch.Generator(device=DEV).manual_seed(SEED + 3)
mv_fmaps = torch.randn(V, 32, 32, C, generator=g_c, device=DEV, dtype=DT)
save("mv_fmap_bhwc", mv_fmaps)

per_view = []
for v in range(V):
    z_v = proj_mv(mv_fmaps[v:v+1], mv_cax[:, v], mv_dist[:, v], scale_t, mv_calc_mat[:, v], BHWC=True)
    save(f"mv_z_proj_v{v}", z_v)
    per_view.append(z_v)
mv_z_proj_avg = torch.stack(per_view, dim=0).mean(dim=0)
save("mv_z_proj_avg", mv_z_proj_avg)

# ---------------- sanity checks ----------------
print("\n=== sanity ===")
F = proj_mv.front_view_transform_matrix.clone()
F[1, 3] = -mv_dist[0, 0]
calc0_err = (mv_calc_mat[0, 0] - F).abs().max().item()
print(f"calc_mat_0 vs F (front-view matrix)         max|d|={calc0_err:.3e}")

z_v0_single = proj16(mv_fmaps[0:1], mv_cax[:, 0], mv_dist[:, 0], scale_t, transform_matrix=None, BHWC=True)
z_v0_err = (per_view[0] - z_v0_single).abs().max().item()
print(f"mv_z_proj_v0 vs single-view ProjGrid(same fmap)  max|d|={z_v0_err:.3e}")

meta = {
    "source_commit": "f7cf384",
    "script": "tools/ref_pixal3d_proj_grid.py",
    "seed": SEED,
    "dtype": "float32",
    "shapes": shapes,
    "sanity": {
        "calc_mat_0_vs_F_max_abs_diff": calc0_err,
        "mv_z_proj_v0_vs_single_view_max_abs_diff": z_v0_err,
    },
    "tolerance_note": (
        "Pure geometry (camera projection + grid_sample), no model weights, CPU f32. "
        "Deterministic; exact bitwise reproducibility expected given the seed. "
        "calc_mat_0 must equal F and mv_z_proj_v0 must equal the single-view ProjGrid "
        "run with an identical fmap to ~1e-6 (fp32 grid_sample/linalg roundoff)."
    ),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
