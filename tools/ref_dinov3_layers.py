#!/usr/bin/env python3
"""DINOv3 の層ごとの出力（先頭 5 トークン = cls + register 4 本）をダンプする。

conditioning の突き合わせで、NAF も ProjGrid も通らない z_global が既に L2 相対 5e-3
ずれていた（docs/PIXAL3D_E2E_STATUS.md §5a）。f16 でも checkpoint 差でもないことは
潰したので、DINOv3 forward のどの層でずれ始めるかを層別に見るためのもの。

C++ 側は TRELLIS_DBG_DINOV3_LAYERS=<dir> で同じ [5, 1024] を cpp_layer%02d.npy /
cpp_final.npy に吐く。全トークンは 4101 x 1024 x 4B x 25 層 = 420 MB になるので
先頭 5 本だけに絞ってある（z_global が使うのはこの 5 本）。

    PIXAL3D_REPO=<MV版 Pixal3D> VIEWS_DIR=<views> OUT=<dir> \
        python3 tools/ref_dinov3_layers.py
"""
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.environ.get("PIXAL3D_REPO", "/data/pixal3d_condtex/podsend/Pixal3D"))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

OUT = os.environ.get("OUT", "/data/pixal3d_condtex/dinov3_layers")
os.makedirs(OUT, exist_ok=True)
VIEWS_DIR = os.environ["VIEWS_DIR"]
S = int(os.environ.get("IMAGE_SIZE", "1024"))

# tex_1024 / shape_1024 と同じ backbone。NAF は使わないので natten は要らない。
model = DinoV3ProjFeatureExtractor(model_name="camenduru/dinov3-vitl16-pretrain-lvd1689m",
                                   image_size=S, grid_resolution=64)
model.eval()

meta = json.load(open(os.path.join(VIEWS_DIR, "transforms.json")))
fr = meta["frames"][0]                       # view 0 だけで十分（層別の位置特定が目的）
im = Image.open(os.path.join(VIEWS_DIR, fr["file_path"])).convert("RGBA")
im = im.resize((S, S), Image.Resampling.LANCZOS)
a = torch.tensor(np.array(im.getchannel(3))).float() / 255.0
rgb = torch.tensor(np.array(im.convert("RGB"))).permute(2, 0, 1).float() / 255.0
img = (rgb * a.unsqueeze(0))[None]           # [1,3,S,S] アルファ乗算済み・未正規化
print(f"[view] {fr['file_path']} {tuple(img.shape)}")

def save(name, t):
    x = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", x)
    print(f"  {name:16s} {list(x.shape)} mean={x.mean():+.6f} std={x.std():.6f}")

with torch.no_grad():
    x = model.transform(img)                                  # ImageNet 正規化
    x = x.to(model.model.embeddings.patch_embeddings.weight.dtype)
    h = model.model.embeddings(x, bool_masked_pos=None)
    pos = model.model.rope_embeddings(x)
    save("ref_embed", h[0, :5])                               # ブロック前
    for i, layer in enumerate(model.model.layer):
        h = layer(h, position_embeddings=pos)
        save(f"ref_layer{i:02d}", h[0, :5])
    save("ref_final", F.layer_norm(h, h.shape[-1:])[0, :5])   # アフィン無し LN
print(f"\nwrote {OUT}")
