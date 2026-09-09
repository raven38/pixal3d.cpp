#!/usr/bin/env python3
"""Per-stage PyTorch reference of the Pixal3D texture/PBR decoder (SparseUnetVaeDecoder,
out_channels=6, pred_subdiv=False) on a REAL sampled texture SLAT, in the same dump layout as the
C++ decoder's TRELLIS_DBG_STAGE_DUMP (src/shape_decoder.cpp::stage_dump, kind "tex_dec"), so
tools/compare_sparse_stages.py --kind tex_dec can locate the first divergent stage between
PyTorch, CUDA and WebGPU/WASM instead of comparing final PBR attributes only.

The texture decoder has no subdivision head: its C2S up-blocks are driven by the `subs` the
shape decoder predicts on the SAME hr_coords + shape SLAT (guide_subs, exactly as
tools/ref_pixal3d_hr_sample.py produced f32_tex_attrs.npy and as trellis_cli.cpp wires
shape_decode -> tex_decode). So this script first runs the shape decoder (return_subs=True) and
then unrolls the texture decoder's forward(): from_latent -> per resolution i: the ConvNeXt
blocks ("stage{i}_convnext", taken right before the up-block) -> the C2S up-block with
guide_subs[i] ("stage{i}_c2s") -> final LayerNorm + output_layer ("output", PRE *0.5+0.5, as the
C++ tex_decode() returns it). Torso dtype is the checkpoints' own (use_fp16=True).

    FIX=/mnt/hdd1/pixal3d/ref/pixal3d/hr_sample OUT=/mnt/hdd1/pixal3d/ref/pixal3d/tex_stages_torch \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_tex_dec_stages.py
Env: PIXAL3D_REPO, PIXAL3D_CKPT_SHAPE_DEC, PIXAL3D_CKPT_TEX_DEC (as tools/ref_pixal3d_hr_sample.py),
FIX (fixture dir with hr_coords.npy + f32_shape_slat.npy + f32_tex_slat.npy), OUT, REF_DEV.
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
from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from pixal3d.modules.sparse import SparseTensor

WEIGHTS_ROOT = os.environ.get("PIXAL3D_WEIGHTS_ROOT",
    "/mnt/d/pixal3d/hf_home/hub/models--TencentARC--Pixal3D/snapshots/b0cb2e1b794cab9aa0ac38a95d794a4d9337437f")
CKPT_SHAPE = os.environ.get("PIXAL3D_CKPT_SHAPE_DEC", f"{WEIGHTS_ROOT}/ckpts/shape_dec_next_dc_f16c32_fp16")
CKPT_TEX = os.environ.get("PIXAL3D_CKPT_TEX_DEC", f"{WEIGHTS_ROOT}/ckpts/tex_dec_next_dc_f16c32_fp16")
FIX = os.environ.get("FIX", "/mnt/hdd1/pixal3d/ref/pixal3d/hr_sample")
OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/tex_stages_torch")
DEV = os.environ.get("REF_DEV", "cuda")
DECODE_RES = int(os.environ.get("DECODE_RES", "1024"))
KIND = "tex_dec"
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

coords = torch.from_numpy(np.load(f"{FIX}/hr_coords.npy").astype(np.int32)).to(DEV)      # [N,4] (b,x,y,z)
shape_feats = torch.from_numpy(np.load(f"{FIX}/f32_shape_slat.npy").astype(np.float32)).to(DEV)
tex_feats = torch.from_numpy(np.load(f"{FIX}/f32_tex_slat.npy").astype(np.float32)).to(DEV)
print(f"input: hr_coords {tuple(coords.shape)}  shape slat {tuple(shape_feats.shape)}  tex slat {tuple(tex_feats.shape)}")

# ---- shape decoder: only for its predicted subdivision masks (guide_subs) ----
cfg = json.load(open(CKPT_SHAPE + ".json"))["args"]
with torch.device(DEV):
    sdec = FlexiDualGridVaeDecoder(**cfg)
sd = load_file(CKPT_SHAPE + ".safetensors", device=DEV)
sdec.load_state_dict(sd, strict=False); del sd; torch.cuda.empty_cache()
sdec.eval(); sdec.set_resolution(DECODE_RES)
with torch.no_grad():
    _mesh, subs = sdec(SparseTensor(feats=shape_feats, coords=coords), return_subs=True)
print(f"shape decoder subs: {[tuple(s.feats.shape) for s in subs]} dtypes {[str(s.feats.dtype) for s in subs]}")
for i, s in enumerate(subs):   # the masks the C++ side collects as ShapeOut.subs ([8N] uint8, octant-major per voxel)
    m = (s.feats > 0) if s.feats.dtype != torch.bool else s.feats
    np.save(f"{OUT}/{KIND}_guide_subs{i}.npy", m.detach().cpu().numpy().astype(np.uint8))
    print(f"  guide_subs[{i}]: N={m.shape[0]} on={int(m.sum())} ({100.0 * m.float().mean():.2f} %)")
del sdec, _mesh; torch.cuda.empty_cache()

# ---- texture decoder, unrolled ----
tcfg = json.load(open(CKPT_TEX + ".json"))["args"]
with torch.device(DEV):
    dec = SparseUnetVaeDecoder(**tcfg)
sd = load_file(CKPT_TEX + ".safetensors", device=DEV)
missing, unexpected = dec.load_state_dict(sd, strict=False); del sd; torch.cuda.empty_cache()
dec.eval()
print(f"tex decoder: missing={len(missing)} unexpected={len(unexpected)} dtype={dec.dtype} "
      f"pred_subdiv={dec.pred_subdiv} out_channels={dec.out_channels} blocks={[len(b) for b in dec.blocks]}")
x = SparseTensor(feats=tex_feats, coords=coords)
with torch.no_grad():
    h = dec.from_latent(x)
    h = h.type(dec.dtype)
    dump("from_latent", h)
    nres = len(dec.blocks)
    for i, res in enumerate(dec.blocks):
        for j, block in enumerate(res):
            if i < nres - 1 and j == len(res) - 1:
                dump(f"stage{i}_convnext", h)
                h = block(h, subdiv=subs[i])
                dump(f"stage{i}_c2s", h)
            else:
                h = block(h)
    h = h.type(x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = dec.output_layer(h)
    dump("output", h)
    # cross-check against the fixture's final attributes (already *0.5+0.5) when present
    fa = f"{FIX}/f32_tex_attrs.npy"
    if os.path.exists(fa):
        ref = np.load(fa); mine = (h.feats.float() * 0.5 + 0.5).cpu().numpy()
        rc = np.load(f"{FIX}/f32_tex_coords.npy")
        mc = h.coords[:, 1:].cpu().numpy()
        same_coords = rc.shape == mc.shape and np.array_equal(rc, mc)
        print(f"  vs f32_tex_attrs: Nt={ref.shape[0]} mine={mine.shape[0]} coords_identical={same_coords}"
              + (f" max|d|={np.abs(mine - ref).max():.3e}" if same_coords else ""))
print("done:", OUT)
