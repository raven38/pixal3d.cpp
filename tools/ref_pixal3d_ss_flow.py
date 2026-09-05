#!/usr/bin/env python3
"""Golden-tensor dump for the Pixal3D SS-flow DiT with proj conditioning (Phase 0, F2).

Mirrors tools/ref_ss_flow.py (dense SS-flow DiT) but for Pixal3D's
SparseStructureFlowModel with image_attn_mode="proj" (ProjectAttention:
global_out = cross_attn_block(x, global); proj_out = proj_linear(proj);
out = global_out + proj_out). The SS checkpoint has no proj_in_channels set, so
ProjectAttention falls back to proj_in = ctx_channels = cond_channels = 1024.

Runs the real model (sdpa attention backend, f32 weights) on GPU with a fixed
seeded input and dumps inputs + per-stage intermediates, including block-0
ProjectAttention sub-intermediates via hooks (blk0_cross_out / blk0_global_out /
blk0_proj_out).

    ATTN_BACKEND=sdpa PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D \
    OUT=/mnt/hdd1/pixal3d/ref/pixal3d/ss_flow_proj \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_ss_flow.py
"""
import os, sys, json
os.environ["ATTN_BACKEND"] = "sdpa"          # must precede pixal3d import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PIXAL3D_REPO = os.environ.get("PIXAL3D_REPO", "/mnt/hdd1/pixal3d/Pixal3D")
sys.path.insert(0, PIXAL3D_REPO)
import numpy as np
import torch
from safetensors.torch import load_file
from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel

WEIGHTS_ROOT = os.environ.get(
    "PIXAL3D_WEIGHTS_ROOT",
    "/mnt/d/pixal3d/hf_home/hub/models--TencentARC--Pixal3D/snapshots/"
    "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
)
CKPT = os.environ.get("PIXAL3D_CKPT_SS", f"{WEIGHTS_ROOT}/ckpts/ss_flow_img_dit_1_3B_64_bf16_mv")
OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/ss_flow_proj")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("REF_DEV", "cuda")
DTYPE = torch.float32
SEED = 42

shapes = {}
def save(name, t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:22s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert not np.isnan(a).any(), f"{name} has NaNs"

cfg = json.load(open(CKPT + ".json"))["args"]
print("config:", {k: cfg.get(k) for k in (
    "resolution", "in_channels", "model_channels", "num_blocks", "num_heads",
    "cond_channels", "image_attn_mode", "proj_in_channels")})

with torch.device(DEV):
    model = SparseStructureFlowModel(**cfg)   # params allocate directly on GPU
model.convert_to(DTYPE)                        # undo the bf16 torso -> f32
sd = load_file(CKPT + ".safetensors", device=DEV)  # bf16 on GPU; copy_ casts to f32
missing, unexpected = model.load_state_dict(sd, strict=False)
del sd; torch.cuda.empty_cache()
print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")
if missing:    print("  missing[:8]   :", missing[:8])
if unexpected: print("  unexpected[:8]:", unexpected[:8])
model.eval()

# deterministic inputs
g = torch.Generator(device=DEV).manual_seed(SEED)
B, R, Cin = 1, cfg["resolution"], cfg["in_channels"]
Dc = cfg["cond_channels"]
Lg, Lp = 5, R ** 3   # z_global: CLS + 4 register tokens; z_proj: grid_resolution=16 -> 4096

x = torch.randn(B, Cin, R, R, R, generator=g, device=DEV, dtype=DTYPE)
t = torch.tensor([float(os.environ.get("TVAL", "500"))], device=DEV, dtype=DTYPE)
cond_global = torch.randn(B, Lg, Dc, generator=g, device=DEV, dtype=DTYPE) * 0.5
cond_proj = torch.randn(B, Lp, Dc, generator=g, device=DEV, dtype=DTYPE) * 0.5
if os.environ.get("ZEROCOND"):
    cond_global = torch.zeros_like(cond_global)
    cond_proj = torch.zeros_like(cond_proj)
cond = {"global": cond_global, "proj": cond_proj}

print("\ninputs:")
save("input_x", x); save("input_t", t)
save("input_global", cond_global); save("input_proj", cond_proj)
ph = model.rope_phases                          # [4096,64] complex
save("rope_cos", torch.view_as_real(ph)[..., 0])
save("rope_sin", torch.view_as_real(ph)[..., 1])

caps = {}
def _feats(o):
    return o.feats if hasattr(o, "feats") else o
hk = lambda n: (lambda m, i, o: caps.__setitem__(n, _feats(o).detach()))
hkin = lambda n: (lambda m, i: caps.__setitem__(n, i[0].detach()))
model.input_layer.register_forward_hook(hk("after_input_layer"))
model.adaLN_modulation.register_forward_hook(hk("t_emb_mod"))
model.blocks[0].register_forward_hook(hk("after_block0"))
model.blocks[1].register_forward_hook(hk("after_block1"))
model.blocks[-1].register_forward_hook(hk("after_block29"))
model.out_layer.register_forward_pre_hook(hkin("prefinal"))   # after final F.layer_norm
model.blocks[0].cross_attn.register_forward_hook(hk("blk0_cross_out"))
model.blocks[0].cross_attn.cross_attn_block.register_forward_hook(hk("blk0_global_out"))
model.blocks[0].cross_attn.proj_linear.register_forward_hook(hk("blk0_proj_out"))

print("\nrunning forward on", DEV, "...")
with torch.no_grad():
    out = model(x, t, cond)

print("\nintermediates:")
for k in ("after_input_layer", "t_emb_mod", "after_block0", "after_block1", "after_block29", "prefinal",
          "blk0_cross_out", "blk0_global_out", "blk0_proj_out"):
    save(k, caps[k])
save("output", out)

meta = {
    "source_commit": "f7cf384",
    "script": "tools/ref_pixal3d_ss_flow.py",
    "seed": SEED,
    "dtype": "float32",
    "checkpoint": os.path.basename(CKPT),
    "shapes": shapes,
    "tolerance_note": (
        "SS-flow DiT with proj conditioning; f32 weights (cast from bf16 ckpt) on GPU, "
        "sdpa attention backend. proj_in_channels unset in ckpt -> falls back to "
        "cond_channels=1024. Compare vs the GGML port at rel<2e-2 per intermediate, "
        "matching the tolerance convention in test_ss_flow.cpp."
    ),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
