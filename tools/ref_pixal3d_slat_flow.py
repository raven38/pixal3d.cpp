#!/usr/bin/env python3
"""Golden-tensor dump for the Pixal3D sparse SLAT shape flow with proj conditioning
(SLatFlowModel img2shape, in=out=32), Phase 0 F3.

A single deterministic forward pass through the sparse model (not a full sampler
run, unlike the older tools/ref_slat_shape.py) — analogous in spirit to
tools/ref_pixal3d_ss_flow.py but for the sparse SparseProjectAttention path:
global_out = cross_attn_block(x, global); proj_out = proj_linear(proj.feats);
out = global_out + proj_out. This checkpoint has proj_in_channels=2048.

Random deterministic voxel coords + seeded x/cond validate the block math on GPU
(sdpa attention backend, f32 weights). Block-0 SparseProjectAttention
sub-intermediates are captured via hooks (blk0_cross_out / blk0_global_out /
blk0_proj_out), same names as ref_pixal3d_ss_flow.py.

Deviation from the older ref_slat_shape.py fixture convention: coords are saved
as [N,4] here (batch column included), per this fixture's explicit spec, rather
than [N,3] (batch column dropped) as ref_slat_shape.py / test_slat_shape.cpp do.

    ATTN_BACKEND=sdpa PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D \
    OUT=/mnt/hdd1/pixal3d/ref/pixal3d/slat_flow_proj \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_slat_flow.py
"""
import os, sys, json
os.environ["ATTN_BACKEND"] = "sdpa"          # must precede pixal3d import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PIXAL3D_REPO = os.environ.get("PIXAL3D_REPO", "/mnt/hdd1/pixal3d/Pixal3D")
sys.path.insert(0, PIXAL3D_REPO)
import numpy as np
import torch
from safetensors.torch import load_file
from pixal3d.models.structured_latent_flow import SLatFlowModel
from pixal3d.modules.sparse import SparseTensor

WEIGHTS_ROOT = os.environ.get(
    "PIXAL3D_WEIGHTS_ROOT",
    "/mnt/d/pixal3d/hf_home/hub/models--TencentARC--Pixal3D/snapshots/"
    "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
)
CKPT = os.environ.get("PIXAL3D_CKPT_SLAT_SHAPE", f"{WEIGHTS_ROOT}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv")
OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/slat_flow_proj")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("REF_DEV", "cuda")
DTYPE = torch.float32
SEED = 42
N = int(os.environ.get("N", "2000"))

shapes = {}
def save(name, t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:22s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert not np.isnan(a).any(), f"{name} has NaNs"
    return a

cfg = json.load(open(CKPT + ".json"))["args"]
print("cfg:", {k: cfg.get(k) for k in (
    "resolution", "in_channels", "out_channels", "model_channels", "num_blocks",
    "num_heads", "cond_channels", "image_attn_mode", "proj_in_channels")})

with torch.device(DEV):
    model = SLatFlowModel(**cfg)     # plain (non-elastic) forward; ckpt declares ElasticSLatFlowModel
model.convert_to(DTYPE)               # undo the bf16 torso -> f32
sd = load_file(CKPT + ".safetensors", device=DEV)
missing, unexpected = model.load_state_dict(sd, strict=False)
del sd; torch.cuda.empty_cache()
print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")
if missing:    print("  missing[:8]   :", missing[:8])
if unexpected: print("  unexpected[:8]:", unexpected[:8])
model.eval()

RES = cfg["resolution"]; Cin = cfg["in_channels"]; Dc = cfg["cond_channels"]; Dproj = cfg["proj_in_channels"]

# N distinct deterministic voxel coords in [0,RES)^3, batch 0
gg = np.random.default_rng(SEED)
seen = set(); rows = []
while len(rows) < N:
    xv, yv, zv = int(gg.integers(0, RES)), int(gg.integers(0, RES)), int(gg.integers(0, RES))
    if (xv, yv, zv) not in seen:
        seen.add((xv, yv, zv)); rows.append((0, xv, yv, zv))
coords = torch.tensor(rows, dtype=torch.int32, device=DEV)

g = torch.Generator(device=DEV).manual_seed(SEED)
x_feats = torch.randn(N, Cin, generator=g, device=DEV, dtype=DTYPE)
t = torch.tensor([float(os.environ.get("TVAL", "500"))], device=DEV, dtype=DTYPE)
cond_global = torch.randn(1, 5, Dc, generator=g, device=DEV, dtype=DTYPE) * 0.5
proj_feats = torch.randn(N, Dproj, generator=g, device=DEV, dtype=DTYPE) * 0.5

x = SparseTensor(feats=x_feats, coords=coords)
proj_sparse = SparseTensor(feats=proj_feats, coords=coords)
cond = {"global": cond_global, "proj": proj_sparse}

print(f"\nN={N} RES={RES} Cin={Cin} Dc={Dc} Dproj={Dproj}")
print("inputs:")
save("coords", coords.to(torch.float32))    # [N,4] (batch,x,y,z)
save("input_x", x_feats); save("input_t", t)
save("input_global", cond_global); save("input_proj", proj_feats)

caps = {}
def _feats(o):
    return o.feats if hasattr(o, "feats") else o
hk = lambda n: (lambda m, i, o: caps.__setitem__(n, _feats(o).detach()))
for _i, _blk in enumerate(model.blocks):
    _blk.register_forward_hook(hk(f"after_block{_i}"))
model.blocks[0].self_attn.register_forward_hook(hk("blk0_msa_out"))
model.blocks[0].mlp.register_forward_hook(hk("blk0_mlp_out"))
model.blocks[0].cross_attn.register_forward_hook(hk("blk0_cross_out"))
model.blocks[0].cross_attn.cross_attn_block.register_forward_hook(hk("blk0_global_out"))
model.blocks[0].cross_attn.proj_linear.register_forward_hook(hk("blk0_proj_out"))
model.blocks[15].self_attn.register_forward_hook(hk("blk15_msa_out"))
model.blocks[15].mlp.register_forward_hook(hk("blk15_mlp_out"))
model.blocks[15].cross_attn.register_forward_hook(hk("blk15_cross_out"))
model.blocks[15].cross_attn.cross_attn_block.register_forward_hook(hk("blk15_global_out"))
model.blocks[15].cross_attn.proj_linear.register_forward_hook(hk("blk15_proj_out"))

print("\nrunning forward on", DEV, "...")
with torch.no_grad():
    out = model(x, t, cond)

print("\nintermediates:")
for k in ("blk0_msa_out", "blk0_mlp_out", "blk0_cross_out", "blk0_global_out", "blk0_proj_out",
          "blk15_msa_out", "blk15_mlp_out", "blk15_cross_out", "blk15_global_out", "blk15_proj_out"):
    save(k, caps[k])

print("\nper-block outputs (after_block0..%d):" % (len(model.blocks) - 1))
block_absmax_report = []
for i in range(len(model.blocks)):
    name = f"after_block{i}"
    a = save(name, caps[name])
    idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)   # a is [N, C]
    tok, ch = int(idx[0]), int(idx[1])
    block_absmax_report.append((i, float(np.abs(a).max()), tok, ch))

save("output", out.feats)

print("\nper-block absmax (voxel, channel) summary:")
for i, amax, tok, ch in block_absmax_report:
    print(f"  after_block{i:<2d} absmax={amax:12.4f}  at (voxel={tok:5d}, channel={ch:4d})")

meta = {
    "source_commit": "f7cf384",
    "script": "tools/ref_pixal3d_slat_flow.py",
    "seed": SEED,
    "dtype": "float32",
    "checkpoint": os.path.basename(CKPT),
    "N": N,
    "shapes": shapes,
    "block_absmax_report": [
        {"block": i, "absmax": amax, "voxel": tok, "channel": ch}
        for i, amax, tok, ch in block_absmax_report
    ],
    "tolerance_note": (
        "Sparse SLAT shape flow with proj conditioning (SparseProjectAttention, "
        "proj_in_channels=2048); f32 weights (cast from bf16 ckpt) on GPU, native "
        "sdpa sparse attention backend (no monkeypatch needed, unlike the older "
        "ref_slat_shape.py). Single forward pass, not a sampler run. coords saved "
        "as [N,4] with the batch column included (deviates from the older "
        "ref_slat_shape.py/test_slat_shape.cpp [N,3] convention per this fixture's spec)."
    ),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
