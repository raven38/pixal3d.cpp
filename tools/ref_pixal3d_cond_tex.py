#!/usr/bin/env python3
"""Golden dump for the Pixal3D texture-stage (tex_1024) multiview image conditioning
(DINOv3 + NAF upsample at naf_target_size=1024 + ProjGridMV average fusion).

tools/ref_pixal3d_cond_slat.py の tex 版。違いは 2 点だけ:

  * IMAGE_COND_CONFIGS["tex_1024"]（naf_target_size が 512 でなく **1024**）を使う。
    C++ 側でメモリ blocker になっていたのはこの設定（NAF 出力 [1024, 1024^2] = 4 GiB）。
  * dense な z_proj [1, 64^3, 2048] = 2.1 GB は保存せず、実際に走らせた C++ の
    active voxel（hr_coords.npy, int32 [N,3]）で gather した [N, 2048] だけを保存する。
    ファイル名・shape は C++ 側の PIXAL3D_DUMP_FIXTURE 出力と同じなので、そのまま
    trellis-test-pixal3d-cond-tex に食わせて比較できる。

    PIXAL3D_REPO=<MV版 Pixal3D>  VIEWS_DIR=<transforms.json のあるディレクトリ> \
    COORDS=<hr_coords.npy>  OUT=<出力先> \
        python3 tools/ref_pixal3d_cond_tex.py

Reference: pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
(DinoV3ProjMultiViewFeatureExtractor, inference_mv.py の IMAGE_COND_CONFIGS["tex_1024"])
+ valeoai/NAF -- docs/spec/30-pixal3d-cond.md §2-4。
"""
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PIXAL3D_REPO = os.environ.get("PIXAL3D_REPO", "/data/pixal3d_condtex/Pixal3D")
sys.path.insert(0, PIXAL3D_REPO)
import numpy as np
import torch
from PIL import Image

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjMultiViewFeatureExtractor, compute_relative_calc_mat,
)

OUT = os.environ.get("OUT", "/data/pixal3d_condtex/out")
os.makedirs(OUT, exist_ok=True)
VIEWS_DIR = os.environ["VIEWS_DIR"]
COORDS = os.environ["COORDS"]
DEV = os.environ.get("REF_DEV", "cuda")

# inference_mv.py の IMAGE_COND_CONFIGS["tex_1024"]（multiview_fusion は MV 版の既定）
TEX_1024_CFG = {
    "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "image_size": 1024,
    "grid_resolution": 64,
    "use_naf_upsample": True,
    "naf_target_size": 1024,
    "multiview_fusion": "average",
}

shapes = {}
def save(name, t):
    a = np.ascontiguousarray(np.asarray(t.detach().to(torch.float32).cpu().numpy()
                                        if torch.is_tensor(t) else t, dtype=np.float32))
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:26s} {str(list(a.shape)):22s} mean={a.mean():.6f} std={a.std():.6f} absmax={np.abs(a).max():.6f}")
    assert np.isfinite(a).all(), f"{name} has NaN/Inf"


def to_cond_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """inference_mv.to_cond_tensor と同じ: LANCZOS resize してアルファを乗算する。"""
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    rgb = torch.tensor(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return rgb * alpha.unsqueeze(0)


def load_views(views_dir: str, image_size: int):
    """inference_mv.load_views と同じ（RGBA 前提・単一解像度）。"""
    with open(os.path.join(views_dir, "transforms.json")) as f:
        meta = json.load(f)
    frames = meta["frames"]

    def cax_of(fr):
        for src in (fr, meta):
            if "camera_angle_x" in src:
                return float(src["camera_angle_x"])
        raise KeyError(f"'camera_angle_x' missing for {fr.get('file_path')}")

    transform_matrix = torch.tensor([fr["transform_matrix"] for fr in frames], dtype=torch.float32)[None]
    camera_angle_x = torch.tensor([cax_of(fr) for fr in frames], dtype=torch.float32)[None]
    camera_distance = torch.norm(transform_matrix[:, :, :3, 3], dim=-1)

    rgba = []
    for fr in frames:
        p = os.path.join(views_dir, fr["file_path"])
        im = Image.open(p)
        assert im.mode == "RGBA", f"{p}: expected pre-matted RGBA"
        rgba.append(im.convert("RGBA"))

    images = torch.stack([to_cond_tensor(im, image_size) for im in rgba], dim=0)[None]
    mesh_scale = float(meta.get("mesh_scale", 1.0))
    names = [fr.get("name", os.path.splitext(fr["file_path"])[0]) for fr in frames]
    print(f"[Views] V={len(frames)} ({', '.join(names)}) from {views_dir} @ {image_size}")
    return images, camera_angle_x, camera_distance, transform_matrix, mesh_scale


images, camera_angle_x, distance, transform_matrix, mesh_scale_val = load_views(VIEWS_DIR, image_size=1024)
images = images.to(DEV)
camera_angle_x = camera_angle_x.to(DEV)
distance = distance.to(DEV)
transform_matrix = transform_matrix.to(DEV)
mesh_scale = torch.tensor([mesh_scale_val], dtype=torch.float32, device=DEV)
V = images.shape[1]
print(f"V={V} mesh_scale={mesh_scale_val}")

print("\ninputs:")
save("s1024_images", images)
save("camera_angle_x", camera_angle_x)
save("distance", distance)
save("transform_matrix", transform_matrix)
save("mesh_scale", mesh_scale)

print(f"\nloading DinoV3ProjMultiViewFeatureExtractor (tex_1024): {TEX_1024_CFG}")
model = DinoV3ProjMultiViewFeatureExtractor(**TEX_1024_CFG)
model.eval()
model.cuda()
model._load_naf()

calc_mat = compute_relative_calc_mat(transform_matrix, distance, model.proj_grid.front_view_transform_matrix)
save("calc_mat", calc_mat)

with torch.no_grad():
    zg, zp = model(images, camera_angle_x, distance, mesh_scale, transform_matrix)
print(f"\nz_global {tuple(zg.shape)}  z_proj {tuple(zp.shape)}")

# C++ 側の PIXAL3D_DUMP_FIXTURE と同じファイル名・同じ shape で出す
save("tex_cond_global", zg)                     # [1,5,1024]

# gather: Pixal3DMVImageTo3DPipeline.get_proj_cond_shape と同じ順序
#   z_proj_grid = z_proj.reshape(B,R,R,R,-1); z_proj_grid[b, x, y, z]
R = model.grid_resolution
co = np.load(COORDS)
assert co.ndim == 2 and co.shape[1] == 3, f"coords must be [N,3], got {co.shape}"
N = co.shape[0]
print(f"[coords] N={N} from {COORDS} dtype={co.dtype} "
      f"x[{co[:,0].min()}..{co[:,0].max()}] y[{co[:,1].min()}..{co[:,1].max()}] z[{co[:,2].min()}..{co[:,2].max()}]")
assert co.min() >= 0 and co.max() < R
idx = torch.from_numpy(co.astype(np.int64)).to(zp.device)
zpg = zp.reshape(1, R, R, R, -1)
gathered = zpg[0, idx[:, 0], idx[:, 1], idx[:, 2]]   # [N,2048]
save("tex_cond_proj", gathered)

meta = {
    "script": "tools/ref_pixal3d_cond_tex.py",
    "dtype": "float32",
    "views_dir": VIEWS_DIR,
    "coords": COORDS,
    "V": V,
    "N": int(N),
    "mesh_scale": mesh_scale_val,
    "image_cond_config": TEX_1024_CFG,
    "grid_resolution": R,
    "naf": {
        "checkpoint": "valeoai/NAF naf_release.pth (torch.hub)",
        "dilation_tex_1024": "1024/64 = 16 (grid_resolution=64, naf_target_size=1024)",
    },
    "torch": torch.__version__,
    "shapes": shapes,
}
try:
    import natten
    meta["natten"] = natten.__version__
except Exception as e:
    meta["natten"] = f"unavailable ({e})"
with open(f"{OUT}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nwrote {OUT}/meta.json")
