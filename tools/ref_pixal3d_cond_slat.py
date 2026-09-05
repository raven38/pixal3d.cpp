#!/usr/bin/env python3
"""Golden dump for real Pixal3D SLAT-stage (shape) multiview image conditioning
(DINOv3 + NAF upsample + ProjGridMV average fusion), for the shape_512 and
shape_1024 IMAGE_COND_CONFIGS.

Runs DinoV3ProjMultiViewFeatureExtractor(**IMAGE_COND_CONFIGS["shape_512"|"shape_1024"])
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj on the real
example multiview asset (assets/mv_images/example: 4 RGBA views, already matted)
on GPU in f32.

NOTE: mirrors tools/ref_pixal3d_cond_ss.py: reimplements load_views()/
to_cond_tensor() inline instead of `import inference_mv`, because
inference_mv.py's module-level code imports Pixal3DMVImageTo3DPipeline and
o_voxel (the full postprocess/mesh pipeline), which this fixture does not need
and which may not be installed/buildable in a minimal reference-dump
environment.

    HF_HOME=/mnt/d/pixal3d/hf_home HF_HUB_OFFLINE=1 PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D \
    OUT=/mnt/hdd1/pixal3d/ref/pixal3d/cond_slat \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_cond_slat.py
"""
import os, sys, json
os.environ.setdefault("HF_HOME", "/mnt/d/pixal3d/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PIXAL3D_REPO = os.environ.get("PIXAL3D_REPO", "/mnt/hdd1/pixal3d/Pixal3D")
sys.path.insert(0, PIXAL3D_REPO)
import numpy as np
import torch
from PIL import Image

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjMultiViewFeatureExtractor, compute_relative_calc_mat,
)

OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/cond_slat")
os.makedirs(OUT, exist_ok=True)
VIEWS_DIR = os.environ.get("VIEWS_DIR", os.path.join(PIXAL3D_REPO, "assets/mv_images/example"))
DEV = os.environ.get("REF_DEV", "cuda")

# same as inference_mv.py's IMAGE_COND_CONFIGS["shape_512"] / ["shape_1024"]
SHAPE_512_CFG = {
    "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "image_size": 512,
    "grid_resolution": 32,
    "use_naf_upsample": True,
    "naf_target_size": 512,
    "multiview_fusion": "average",
}
SHAPE_1024_CFG = {
    "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "image_size": 1024,
    "grid_resolution": 64,
    "use_naf_upsample": True,
    "naf_target_size": 512,
    "multiview_fusion": "average",
}

shapes = {}
def save(name, t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:26s} {str(list(a.shape)):22s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert np.isfinite(a).all(), f"{name} has NaN/Inf"


def to_cond_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Same as inference_mv.to_cond_tensor: LANCZOS resize, alpha-premultiply."""
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    rgb = torch.tensor(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return rgb * alpha.unsqueeze(0)


def load_views(views_dir: str, image_size: int):
    """Same as inference_mv.load_views, simplified: RGBA already present (no
    rembg), single resolution."""
    with open(os.path.join(views_dir, "transforms.json")) as f:
        meta = json.load(f)
    frames = meta["frames"]

    def cax_of(fr):
        for src in (fr, meta):
            if "camera_angle_x" in src:
                return float(src["camera_angle_x"])
        raise KeyError(f"'camera_angle_x' missing for {fr.get('file_path')}")

    transform_matrix = torch.tensor([fr["transform_matrix"] for fr in frames], dtype=torch.float32)[None]  # [1,V,4,4]
    camera_angle_x = torch.tensor([cax_of(fr) for fr in frames], dtype=torch.float32)[None]                # [1,V]
    camera_distance = torch.norm(transform_matrix[:, :, :3, 3], dim=-1)                                     # [1,V]

    rgba = []
    for fr in frames:
        p = os.path.join(views_dir, fr["file_path"])
        im = Image.open(p)
        assert im.mode == "RGBA", \
            f"{p}: expected pre-matted RGBA (no rembg model wired into this fixture script)"
        rgba.append(im.convert("RGBA"))

    images = torch.stack([to_cond_tensor(im, image_size) for im in rgba], dim=0)[None]  # [1,V,3,H,W]
    mesh_scale = float(meta.get("mesh_scale", 1.0))
    view_names = [fr.get("name", os.path.splitext(fr["file_path"])[0]) for fr in frames]
    print(f"[Views] V={len(frames)} ({', '.join(view_names)}) from {views_dir} @ {image_size}")
    return images, camera_angle_x, camera_distance, transform_matrix, mesh_scale


print(f"[Views] loading {VIEWS_DIR}")
images_512, camera_angle_x, distance, transform_matrix, mesh_scale_val = load_views(VIEWS_DIR, image_size=512)
images_1024, cax_1024, dist_1024, tm_1024, mesh_scale_1024 = load_views(VIEWS_DIR, image_size=1024)
# camera/pose metadata is resolution-independent; sanity-check the two loads agree.
assert torch.allclose(camera_angle_x, cax_1024) and torch.allclose(distance, dist_1024)
assert torch.allclose(transform_matrix, tm_1024) and mesh_scale_val == mesh_scale_1024

images_512 = images_512.to(DEV)
images_1024 = images_1024.to(DEV)
camera_angle_x = camera_angle_x.to(DEV)
distance = distance.to(DEV)
transform_matrix = transform_matrix.to(DEV)
mesh_scale = torch.tensor([mesh_scale_val], dtype=torch.float32, device=DEV)
V = images_512.shape[1]
print(f"V={V} mesh_scale={mesh_scale_val}")

print("\ninputs:")
save("s512_images", images_512)
save("s1024_images", images_1024)
save("camera_angle_x", camera_angle_x)
save("distance", distance)
save("transform_matrix", transform_matrix)
save("mesh_scale", mesh_scale)

print("\nloading DinoV3ProjMultiViewFeatureExtractor (shape_512):", SHAPE_512_CFG)
model_512 = DinoV3ProjMultiViewFeatureExtractor(**SHAPE_512_CFG)
model_512.eval()
model_512.cuda()
model_512._load_naf()  # lazy-load NAF (torch.hub valeoai/NAF, naf_release.pth) up front

print("loading DinoV3ProjMultiViewFeatureExtractor (shape_1024):", SHAPE_1024_CFG)
model_1024 = DinoV3ProjMultiViewFeatureExtractor(**SHAPE_1024_CFG)
model_1024.eval()
model_1024.cuda()
model_1024._load_naf()

calc_mat = compute_relative_calc_mat(transform_matrix, distance, model_512.proj_grid.front_view_transform_matrix)
calc_mat_check = compute_relative_calc_mat(transform_matrix, distance, model_1024.proj_grid.front_view_transform_matrix)
assert torch.allclose(calc_mat, calc_mat_check), "front_view_transform_matrix differs between stages?!"
save("calc_mat", calc_mat)

NUM_REG = 4  # DINOv3 config.num_register_tokens (verified from model config.json)


def patch_tokens_spatial(dino_tokens: torch.Tensor, patch_number: int) -> torch.Tensor:
    """dino_tokens: [V, 1+num_reg+patches, D] (post-final-LN) -> [V, h, w, D] BHWC."""
    D = dino_tokens.shape[-1]
    patch = dino_tokens[:, 1 + NUM_REG:, :]
    return patch.reshape(dino_tokens.shape[0], patch_number, patch_number, D)


# ---- hook extract_features / _project_single_view to capture intermediates ----
def make_hooks(model):
    orig_extract = model.extract_features
    orig_project_single = model._project_single_view
    captured = {}
    per_view = []

    def _patched_extract(image):
        tok = orig_extract(image)
        captured["dino_tokens"] = tok.detach()
        return tok

    def _patched_project_single(*args, **kwargs):
        z = orig_project_single(*args, **kwargs)
        per_view.append(z.detach())
        return z

    model.extract_features = _patched_extract
    model._project_single_view = _patched_project_single
    return captured, per_view


captured_512, per_view_512 = make_hooks(model_512)
captured_1024, per_view_1024 = make_hooks(model_1024)


def run_forward(model, captured, per_view, images, cax, dist, tm):
    per_view.clear()
    captured.pop("dino_tokens", None)
    with torch.no_grad():
        zg, zp = model(images, cax, dist, mesh_scale, tm)
    return zg, zp, captured["dino_tokens"], list(per_view)


# ======================= shape_512 =======================
print(f"\n=== shape_512, V={V} ===")
zg512, zp512, dino512, pv512_4 = run_forward(
    model_512, captured_512, per_view_512, images_512, camera_angle_x, distance, transform_matrix)
save("s512_dino_tokens", dino512)   # [4,1029,1024]
save("s512_z_global", zg512)        # [1,5,1024]
save("s512_z_proj", zp512)          # [1,32768,2048]

print(f"\n=== shape_512, V=1 ===")
zg512_1, zp512_1, dino512_1, pv512_1 = run_forward(
    model_512, captured_512, per_view_512,
    images_512[:, :1], camera_angle_x[:, :1], distance[:, :1], transform_matrix[:, :1])
save("s512_v1_z_global", zg512_1)   # [1,5,1024]
save("s512_v1_z_proj", zp512_1)     # [1,32768,2048]

# view-0 NAF HR map at S=512 (1 GB), computed exactly as _project_single_view does
# for view 0: image_for_naf_v0 is the un-normalized alpha-premultiplied [1,3,512,512]
# image (image_for_naf's slice for view 0 == raw input, since NAF's guide is taken
# before ImageNet normalization); lr_bchw_v0 is view 0's DINO patch map BCHW.
patch_spatial_512 = patch_tokens_spatial(dino512, model_512.patch_number)  # [4,32,32,1024] BHWC
image_for_naf_v0_512 = images_512[:, 0]                                   # [1,3,512,512], un-normalized
lr_bchw_v0_512 = patch_spatial_512[0:1].permute(0, 3, 1, 2).contiguous()  # [1,1024,32,32]
with torch.no_grad():
    naf_hr_v0_512 = model_512.naf_model(image_for_naf_v0_512, lr_bchw_v0_512, model_512.naf_target_size)
save("s512_naf_hr_v0", naf_hr_v0_512)  # [1,1024,512,512]

# ======================= shape_1024 =======================
print(f"\n=== shape_1024, V={V} ===")
zg1024, zp1024, dino1024, pv1024_4 = run_forward(
    model_1024, captured_1024, per_view_1024, images_1024, camera_angle_x, distance, transform_matrix)
save("s1024_dino_tokens", dino1024)   # [4,4101,1024]
save("s1024_z_global", zg1024)        # [1,5,1024]
save("s1024_z_proj", zp1024)          # [1,262144,2048]

patch_spatial_1024 = patch_tokens_spatial(dino1024, model_1024.patch_number)  # [4,64,64,1024] BHWC
image_for_naf_v0_1024 = images_1024[:, 0]                                    # [1,3,1024,1024]
lr_bchw_v0_1024 = patch_spatial_1024[0:1].permute(0, 3, 1, 2).contiguous()   # [1,1024,64,64]
with torch.no_grad():
    naf_hr_v0_1024 = model_1024.naf_model(image_for_naf_v0_1024, lr_bchw_v0_1024, model_1024.naf_target_size)
save("s1024_naf_hr_v0", naf_hr_v0_1024)  # [1,1024,512,512]

# gathered subset at 4096 random voxel coords, matching
# Pixal3DMVImageTo3DPipeline.get_proj_cond_shape's gather:
#   z_proj_grid = z_proj.reshape(B, R, R, R, -1); z_proj_grid[b_idx, x_idx, y_idx, z_idx]
GRID_R_1024 = model_1024.grid_resolution  # 64
g = torch.Generator().manual_seed(42)
N_COORDS = 4096
xyz = torch.randint(0, GRID_R_1024, (N_COORDS, 3), generator=g).to(torch.int64)
b_col = torch.zeros((N_COORDS, 1), dtype=torch.int64)
coords = torch.cat([b_col, xyz], dim=1)  # [4096,4] (b,x,y,z), int, stored as float32
save("s1024_coords", coords.to(torch.float32))

z_proj_grid_1024 = zp1024.reshape(1, GRID_R_1024, GRID_R_1024, GRID_R_1024, -1)
coords_dev = coords.to(zp1024.device)  # coords was built on CPU (CPU generator, seed 42)
b_idx, x_idx, y_idx, z_idx = coords_dev[:, 0], coords_dev[:, 1], coords_dev[:, 2], coords_dev[:, 3]
z_proj_gathered_1024 = z_proj_grid_1024[b_idx, x_idx, y_idx, z_idx]  # [4096,2048]
save("s1024_z_proj_gathered", z_proj_gathered_1024)

# ---------------- sanity ----------------
print("\n=== sanity ===")

# (1) s512_z_proj[..., :1024] == manual mean-over-views ProjGrid sample of the DINO
#     patch maps (the LR branch, before NAF concat), using the extractor's own
#     proj_grid module and the calc_mat computed above.
manual_lr_sum = None
for v in range(V):
    fmap_v = patch_spatial_512[v:v + 1]           # [1,32,32,1024] BHWC
    cax_v = camera_angle_x[:, v]
    dist_v = distance[:, v]
    cm_v = calc_mat[:, v]
    z_v = model_512.proj_grid(fmap_v, cax_v, dist_v, mesh_scale, cm_v, BHWC=True)  # [1,32768,1024]
    manual_lr_sum = z_v if manual_lr_sum is None else manual_lr_sum + z_v
manual_lr_avg = manual_lr_sum / V
d1 = (zp512[..., :1024] - manual_lr_avg).abs().max().item()
print(f"(1) s512_z_proj[...,:1024] vs manual mean ProjGrid(DINO lr maps)  max|d|={d1:.3e}")

# (2) s512_v1_z_proj[..., 1024:] (the NAF/HR branch of the V=1 run, i.e. view 0
#     only) vs ProjGrid sampling of s512_naf_hr_v0 directly (BHWC=False), using
#     the V=1 run's own calc_mat (== front-view F by construction, calc_mat_0==F).
calc_mat_v1 = compute_relative_calc_mat(
    transform_matrix[:, :1], distance[:, :1], model_512.proj_grid.front_view_transform_matrix)
z_hr_manual = model_512.proj_grid(
    naf_hr_v0_512, camera_angle_x[:, :1].squeeze(1), distance[:, :1].squeeze(1),
    mesh_scale, calc_mat_v1.squeeze(1), BHWC=False)  # [1,32768,1024]
d2 = (zp512_1[..., 1024:] - z_hr_manual).abs().max().item()
print(f"(2) s512_v1_z_proj[...,1024:] vs ProjGrid(s512_naf_hr_v0, BHWC=False)  max|d|={d2:.3e}")

# (3) no NaN/Inf anywhere: already asserted per-tensor inside save().
print("(3) no NaN/Inf in any saved tensor: OK (checked in save())")

meta = {
    "source_commit": "f7cf384",
    "script": "tools/ref_pixal3d_cond_slat.py",
    "seed": "42 (only for s1024_coords voxel sampling; everything else is deterministic: real images + frozen weights, no RNG)",
    "dtype": "float32",
    "views_dir": VIEWS_DIR,
    "V": V,
    "num_register_tokens": NUM_REG,
    "image_cond_configs": {"shape_512": SHAPE_512_CFG, "shape_1024": SHAPE_1024_CFG},
    "naf": {
        "checkpoint": "valeoai/NAF naf_release.pth (torch.hub cache)",
        "natten_version": "0.21.0",
        "dilation_shape_512": "512/32 = 16 (grid_resolution=32, naf_target_size=512)",
        "dilation_shape_1024": "512/64 = 8 (grid_resolution=64, naf_target_size=512)",
    },
    "shapes": shapes,
    "sanity": {
        "s512_z_proj_lr_vs_manual_projgrid_max_abs_diff": d1,
        "s512_v1_z_proj_hr_vs_manual_projgrid_of_naf_hr_v0_max_abs_diff": d2,
        "no_nan_inf": True,
    },
    "tolerance_note": (
        "Real DINOv3 ViT-L/16 (camenduru/dinov3-vitl16-pretrain-lvd1689m) + NAF "
        "(valeoai/NAF, naf_release.pth) + ProjGridMV average-fusion conditioning "
        "for the SLAT shape_512/shape_1024 stages; f32, GPU. Deterministic given "
        "the asset dir and NAF checkpoint (no training-time RNG); s1024_coords is "
        "the only RNG-derived tensor (seed 42, CPU generator, for the gathered-"
        "subset sanity/test fixture). Sanity (1) and (2) reproduce the LR and HR "
        "proj branches from raw captured intermediates via the extractor's own "
        "proj_grid module, so they should match to GPU float32 roundoff."
    ),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
