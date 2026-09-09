#!/usr/bin/env python3
"""Per-stage PyTorch reference of the Pixal3D shape decoder (FlexiDualGridVaeDecoder ==
TRELLIS.2's SparseUnetVaeDecoder torso) on a REAL sampled shape SLAT, in the same dump layout as
the C++ decoder's TRELLIS_DBG_STAGE_DUMP (src/shape_decoder.cpp::stage_dump), so
tools/compare_sparse_stages.py can locate the first divergent stage between PyTorch, CUDA and
WebGPU/WASM instead of comparing final meshes only.

Stages (the decoder's own forward(), unrolled): from_latent -> for each resolution i: the
ConvNeXt blocks ("stage{i}_convnext", taken right before the up-block) -> the C2S up-block
("stage{i}_c2s") -> final LayerNorm + output_layer ("output"). Torso dtype is the checkpoint's
own (use_fp16=True), exactly as the fixture's f32_dec_vertices were produced.

    FIX=/mnt/hdd1/pixal3d/ref/pixal3d/hr_sample OUT=/mnt/hdd1/pixal3d/ref/pixal3d/hr_stages_torch \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_shape_dec_stages.py
Env: PIXAL3D_REPO, PIXAL3D_CKPT_SHAPE_DEC (as tools/ref_pixal3d_hr_sample.py), FIX (fixture dir
with hr_coords.npy + f32_shape_slat.npy, or coords.npy + f32_slat.npy), OUT, REF_DEV.
"""
import os, sys, json
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ.setdefault("HF_HOME", "/mnt/d/pixal3d/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
PIXAL3D_REPO = os.environ.get("PIXAL3D_REPO", "/mnt/hdd1/pixal3d/Pixal3D")
sys.path.insert(0, PIXAL3D_REPO)
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from pixal3d.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
from pixal3d.modules.sparse import SparseTensor

WEIGHTS_ROOT = os.environ.get("PIXAL3D_WEIGHTS_ROOT",
    "/mnt/d/pixal3d/hf_home/hub/models--TencentARC--Pixal3D/snapshots/b0cb2e1b794cab9aa0ac38a95d794a4d9337437f")
CKPT = os.environ.get("PIXAL3D_CKPT_SHAPE_DEC", f"{WEIGHTS_ROOT}/ckpts/shape_dec_next_dc_f16c32_fp16")
FIX = os.environ.get("FIX", "/mnt/hdd1/pixal3d/ref/pixal3d/hr_sample")
OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/hr_stages_torch")
DEV = os.environ.get("REF_DEV", "cuda")
KIND = "shape_dec"
os.makedirs(OUT, exist_ok=True)

def dump(stage, h):
    """Same files/rule as src/shape_decoder.cpp::stage_dump."""
    coords = h.coords[:, 1:].detach().cpu().numpy().astype(np.int32)   # [N,3]
    feats = h.feats.detach().to(torch.float32).cpu().numpy()            # [N,C]
    N, C = feats.shape
    np.save(f"{OUT}/{KIND}_{stage}_coords.npy", np.ascontiguousarray(coords))
    K = np.uint32(max(1, N // 4096))
    c = coords.astype(np.uint32)
    with np.errstate(over="ignore"):
        hsh = (c[:, 0] * np.uint32(73856093)) ^ (c[:, 1] * np.uint32(19349663)) ^ (c[:, 2] * np.uint32(83492791))
    sel = (hsh % K) == 0
    np.save(f"{OUT}/{KIND}_{stage}_sub_coords.npy", np.ascontiguousarray(coords[sel]))
    np.save(f"{OUT}/{KIND}_{stage}_sub_feats.npy", np.ascontiguousarray(feats[sel]))
    f64 = feats.astype(np.float64)
    st = np.array([feats.size, f64.mean(), f64.std(), np.abs(f64).max()], dtype=np.float32)
    np.save(f"{OUT}/{KIND}_{stage}_stats.npy", st)
    print(f"  [stage-dump] {KIND}_{stage}: N={N} C={C} subset={int(sel.sum())} (K={int(K)}) "
          f"mean={st[1]:.5f} std={st[2]:.5f} absmax={st[3]:.4f} dtype={h.feats.dtype}", flush=True)

cfg = json.load(open(CKPT + ".json"))["args"]
with torch.device(DEV):
    dec = FlexiDualGridVaeDecoder(**cfg)
sd = load_file(CKPT + ".safetensors", device=DEV)
missing, unexpected = dec.load_state_dict(sd, strict=False)
del sd; torch.cuda.empty_cache()
dec.eval()
print(f"decoder: missing={len(missing)} unexpected={len(unexpected)} dtype={dec.dtype} blocks={[len(b) for b in dec.blocks]}")

cp = f"{FIX}/hr_coords.npy" if os.path.exists(f"{FIX}/hr_coords.npy") else f"{FIX}/coords.npy"
sp_ = f"{FIX}/f32_shape_slat.npy" if os.path.exists(f"{FIX}/f32_shape_slat.npy") else f"{FIX}/f32_slat.npy"
coords = torch.from_numpy(np.load(cp).astype(np.int32)).to(DEV)       # [N,4] (b,x,y,z)
feats = torch.from_numpy(np.load(sp_).astype(np.float32)).to(DEV)     # [N,32] denormalized
print(f"input: {cp} {tuple(coords.shape)}  {sp_} {tuple(feats.shape)}")
x = SparseTensor(feats=feats, coords=coords)

with torch.no_grad():
    h = dec.from_latent(x)
    h = h.type(dec.dtype)
    dump("from_latent", h)
    nres = len(dec.blocks)
    for i, res in enumerate(dec.blocks):
        for j, block in enumerate(res):
            if i < nres - 1 and j == len(res) - 1:
                dump(f"stage{i}_convnext", h)
                h, sub = block(h)
                dump(f"stage{i}_c2s", h)
            else:
                h = block(h)
    h = h.type(x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = dec.output_layer(h)
    dump("output", h)
print("done:", OUT)
