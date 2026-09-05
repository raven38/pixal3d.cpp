#!/usr/bin/env python3
"""Golden dump for real Pixal3D SS-stage conditioning (DINOv3 + ProjGridMV average
fusion), Phase 0 F4.

Runs DinoV3ProjMultiViewFeatureExtractor(**IMAGE_COND_CONFIGS["ss"]) from
pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py on the real
example multiview asset (assets/mv_images/example: 4 RGBA views, all already
matted so no rembg model is needed) on GPU in f32.

NOTE: this script reimplements load_views()/to_cond_tensor() inline instead of
`import inference_mv`, because inference_mv.py's module-level code imports
Pixal3DMVImageTo3DPipeline and o_voxel (the full postprocess/mesh pipeline),
which this fixture does not need and which may not be installed/buildable in a
minimal reference-dump environment. The reimplementation mirrors inference_mv.py
exactly for the RGBA-already-present, single-resolution (512) case used here.

    HF_HOME=/mnt/d/pixal3d/hf_home HF_HUB_OFFLINE=1 PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D \
    OUT=/mnt/hdd1/pixal3d/ref/pixal3d/cond_ss \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_cond_ss.py
"""
import os, sys, json
os.environ.setdefault("HF_HOME", "/mnt/d/pixal3d/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PIXAL3D_REPO = os.environ.get("PIXAL3D_REPO", "/mnt/hdd1/pixal3d/Pixal3D")
sys.path.insert(0, PIXAL3D_REPO)
import numpy as np
import torch
from PIL import Image

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjMultiViewFeatureExtractor, compute_relative_calc_mat,
)

OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/cond_ss")
os.makedirs(OUT, exist_ok=True)
VIEWS_DIR = os.environ.get("VIEWS_DIR", os.path.join(PIXAL3D_REPO, "assets/mv_images/example"))
DEV = os.environ.get("REF_DEV", "cuda")

# same as inference_mv.py's IMAGE_COND_CONFIGS["ss"]
SS_CFG = {
    "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "image_size": 512,
    "grid_resolution": 16,
    "multiview_fusion": "average",
}

shapes = {}
def save(name, t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:22s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert not np.isnan(a).any(), f"{name} has NaNs"


def to_cond_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Same as inference_mv.to_cond_tensor: LANCZOS resize, alpha-premultiply."""
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    rgb = torch.tensor(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return rgb * alpha.unsqueeze(0)


def load_views(views_dir: str, image_size: int = 512):
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
    print(f"[Views] V={len(frames)} ({', '.join(view_names)}) from {views_dir}")
    return images, camera_angle_x, camera_distance, transform_matrix, mesh_scale


print(f"[Views] loading {VIEWS_DIR}")
images_512, camera_angle_x, distance, transform_matrix, mesh_scale_val = load_views(VIEWS_DIR, image_size=512)
images_512 = images_512.to(DEV)
camera_angle_x = camera_angle_x.to(DEV)
distance = distance.to(DEV)
transform_matrix = transform_matrix.to(DEV)
mesh_scale = torch.tensor([mesh_scale_val], dtype=torch.float32, device=DEV)
V = images_512.shape[1]
print(f"V={V} mesh_scale={mesh_scale_val}")

print("\ninputs:")
save("images_512", images_512)
save("camera_angle_x", camera_angle_x)
save("distance", distance)
save("transform_matrix", transform_matrix)
save("mesh_scale", mesh_scale)

print("\nloading DinoV3ProjMultiViewFeatureExtractor:", SS_CFG)
model = DinoV3ProjMultiViewFeatureExtractor(**SS_CFG)
model.eval()
model.cuda()

calc_mat = compute_relative_calc_mat(transform_matrix, distance, model.proj_grid.front_view_transform_matrix)
save("calc_mat", calc_mat)

# capture the raw DINOv3 tokens (post final LN, pre-projection) that forward()
# computes internally, and the per-view projected contribution before averaging.
_orig_extract = model.extract_features
_orig_project_single = model._project_single_view
_captured = {}
_per_view = []

def _patched_extract(image):
    tok = _orig_extract(image)
    _captured["dino_tokens"] = tok.detach()
    return tok

def _patched_project_single(*args, **kwargs):
    z = _orig_project_single(*args, **kwargs)
    _per_view.append(z.detach())
    return z

model.extract_features = _patched_extract
model._project_single_view = _patched_project_single


def run_forward(images, cax, dist, tm):
    _per_view.clear()
    _captured.pop("dino_tokens", None)
    with torch.no_grad():
        zg, zp = model(images, cax, dist, mesh_scale, tm)
    return zg, zp, _captured["dino_tokens"], list(_per_view)


print(f"\nrunning V={V} forward on", DEV, "...")
zg4, zp4, dino4, per_view4 = run_forward(images_512, camera_angle_x, distance, transform_matrix)
print("outputs (V=%d, fusion=average):" % V)
save("dino_tokens", dino4)   # [V,1029,1024]
save("z_global", zg4)        # [1,5,1024]
save("z_proj", zp4)          # [1,4096,1024]

print(f"\nrunning V=1 forward on", DEV, "...")
zg1, zp1, dino1, per_view1 = run_forward(images_512[:, :1], camera_angle_x[:, :1], distance[:, :1], transform_matrix[:, :1])
print("outputs (V=1 variant):")
save("v1_z_global", zg1)     # [1,5,1024]
save("v1_z_proj", zp1)       # [1,4096,1024]

# ---------------- sanity ----------------
print("\n=== sanity ===")
d_self = (zp1 - per_view1[0]).abs().max().item()
print(f"v1_z_proj vs its own per-view[0] contribution (avg of 1 == itself)  max|d|={d_self:.3e}")
d_first_view = (zp1 - per_view4[0]).abs().max().item()
print(f"v1_z_proj vs V=4-run's own view-0 contribution (first-view path)   max|d|={d_first_view:.3e}")

meta = {
    "source_commit": "f7cf384",
    "script": "tools/ref_pixal3d_cond_ss.py",
    "seed": "n/a (deterministic: no RNG, real images + frozen DINOv3 weights)",
    "dtype": "float32",
    "views_dir": VIEWS_DIR,
    "V": V,
    "image_cond_config": SS_CFG,
    "shapes": shapes,
    "sanity": {
        "v1_z_proj_vs_self_avg1_max_abs_diff": d_self,
        "v1_z_proj_vs_v4run_view0_contribution_max_abs_diff": d_first_view,
    },
    "tolerance_note": (
        "Real DINOv3 ViT-L/16 (camenduru/dinov3-vitl16-pretrain-lvd1689m) + ProjGridMV "
        "average-fusion conditioning for the SS stage; f32, GPU. No RNG involved (real "
        "images, frozen weights) so this fixture is deterministic given the asset dir. "
        "v1_z_proj should match the V=4 run's own view-0 contribution (extracted via "
        "a hook on _project_single_view) to within GPU batch-size floating-point "
        "roundoff, since frame 0 is the canonical front view and DINOv3 processes each "
        "view independently along the batch dimension."
    ),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
