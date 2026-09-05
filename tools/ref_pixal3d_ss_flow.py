#!/usr/bin/env python3
"""Golden-tensor dump for the Pixal3D SS-flow DiT with proj conditioning (Phase 0, F2).

Mirrors tools/ref_ss_flow.py (dense SS-flow DiT) but for Pixal3D's
SparseStructureFlowModel with image_attn_mode="proj" (ProjectAttention:
global_out = cross_attn_block(x, global); proj_out = proj_linear(proj);
out = global_out + proj_out). The SS checkpoint has no proj_in_channels set, so
ProjectAttention falls back to proj_in = ctx_channels = cond_channels = 1024.

Runs the real model (sdpa attention backend, f32 weights by default) on GPU with
a fixed seeded input and dumps inputs + per-stage intermediates, including
block-0/block-15 ProjectAttention sub-intermediates via hooks (blk{0,15}_cross_out
/ blk{0,15}_global_out / blk{0,15}_proj_out / blk{0,15}_msa_out / blk{0,15}_mlp_out).

REF_DTYPE=float64 switches the whole forward (weights + non-torso modules +
activations) to double precision, to establish how much PyTorch's own f32
arithmetic deviates from a higher-precision "true" value. The seed-42 inputs are
still generated in float32 exactly as in the default run (identical RNG draws),
then cast to float64 before the forward -- so the f64 run differs from the f32
run ONLY in the precision of the model's arithmetic, not in the input values.
CAVEATS (two library-level float32 constants this script does not patch around):
  1. pixal3d.modules.attention.rope.RotaryPositionEmbedder.apply_rotary_embedding
     hardcodes `x.float()` before building the complex rotation, so the
     self-attention RoPE step is computed in float32 precision even in the f64 run.
  2. TimestepEmbedder.timestep_embedding hardcodes `t[:, None].float()`, so
     t_embedder + adaLN_modulation (the timestep/modulation MLPs) run in float32
     and are only upconverted to DTYPE afterward via the model's own
     `manual_cast(t_emb, self.dtype)` call, right before the block loop.
All other arithmetic (input_layer, out_layer, the 30 transformer blocks: self/
cross attention matmuls+softmax, proj_linear, mlp, layer norms) runs fully in the
chosen dtype. Saved files are always cast to float32 .npy on write (see save()),
regardless of REF_DTYPE, so the C++ side reads them unchanged either way.

    ATTN_BACKEND=sdpa PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D \
    OUT=/mnt/hdd1/pixal3d/ref/pixal3d/ss_flow_proj \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_ss_flow.py

    REF_DTYPE=float64 OUT=/mnt/hdd1/pixal3d/ref/pixal3d/ss_flow_proj_f64 \
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
DEV = os.environ.get("REF_DEV", "cuda")
REF_DTYPE = os.environ.get("REF_DTYPE", "float32")
DTYPE = {"float32": torch.float32, "float64": torch.float64}[REF_DTYPE]
_default_out = "/mnt/hdd1/pixal3d/ref/pixal3d/" + ("ss_flow_proj_f64" if REF_DTYPE == "float64" else "ss_flow_proj")
OUT = os.environ.get("OUT", _default_out)
os.makedirs(OUT, exist_ok=True)
SEED = 42

shapes = {}
def save(name, t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:22s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert not np.isnan(a).any(), f"{name} has NaNs"
    return a

print(f"REF_DTYPE={REF_DTYPE} DEV={DEV} OUT={OUT}")
cfg = json.load(open(CKPT + ".json"))["args"]
print("config:", {k: cfg.get(k) for k in (
    "resolution", "in_channels", "model_channels", "num_blocks", "num_heads",
    "cond_channels", "image_attn_mode", "proj_in_channels")})

with torch.device(DEV):
    model = SparseStructureFlowModel(**cfg)   # params allocate directly on GPU
model.convert_to(DTYPE)                        # undo the bf16 torso -> DTYPE
# non-torso modules (not covered by convert_to, which only touches self.blocks):
# default-constructed in float32, so this is a no-op when REF_DTYPE=float32 and
# byte-identical to the pre-REF_DTYPE script behavior.
model.input_layer.to(DTYPE)
model.out_layer.to(DTYPE)
# NOTE: t_embedder/adaLN_modulation are deliberately NOT cast to DTYPE.
# TimestepEmbedder.timestep_embedding hardcodes `t[:, None].float()` when building
# the sinusoidal frequency args, so its output (and therefore t_embedder.mlp's
# input) is always float32 regardless of DTYPE -- casting t_embedder's Linear
# weights to float64 makes that mlp's matmul fail with a dtype mismatch. The
# model's own forward() already handles this: it runs t_embedder + adaLN_modulation
# in float32 and then does `t_emb = manual_cast(t_emb, self.dtype)` to upconvert
# the modulation vector to DTYPE before it reaches the blocks. So even in the f64
# run, the timestep embedding itself is computed at float32 precision (a second
# library-level constant, like the RoPE one above) -- everything downstream of
# that manual_cast (all 30 blocks, input_layer, out_layer) runs fully in DTYPE.
sd = load_file(CKPT + ".safetensors", device=DEV)  # bf16 on GPU; copy_ casts to DTYPE
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

# always generate in float32 (identical RNG draws regardless of REF_DTYPE), then
# cast to the working dtype -- so the f64 run's inputs are exactly the f32 run's
# inputs, just stored at higher precision, isolating model-arithmetic precision.
x = torch.randn(B, Cin, R, R, R, generator=g, device=DEV, dtype=torch.float32).to(DTYPE)
t = torch.tensor([float(os.environ.get("TVAL", "500"))], device=DEV, dtype=torch.float32).to(DTYPE)
cond_global = (torch.randn(B, Lg, Dc, generator=g, device=DEV, dtype=torch.float32) * 0.5).to(DTYPE)
cond_proj = (torch.randn(B, Lp, Dc, generator=g, device=DEV, dtype=torch.float32) * 0.5).to(DTYPE)
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
for _i, _blk in enumerate(model.blocks):
    _blk.register_forward_hook(hk(f"after_block{_i}"))
model.out_layer.register_forward_pre_hook(hkin("prefinal"))   # after final F.layer_norm
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
for k in ("after_input_layer", "t_emb_mod", "prefinal",
          "blk0_msa_out", "blk0_mlp_out", "blk0_cross_out", "blk0_global_out", "blk0_proj_out",
          "blk15_msa_out", "blk15_mlp_out", "blk15_cross_out", "blk15_global_out", "blk15_proj_out"):
    save(k, caps[k])

print("\nper-block outputs (after_block0..%d):" % (len(model.blocks) - 1))
block_absmax_report = []
for i in range(len(model.blocks)):
    name = f"after_block{i}"
    a = save(name, caps[name])
    idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)   # a is [1, L, C]
    tok, ch = int(idx[1]), int(idx[2])
    block_absmax_report.append((i, float(np.abs(a).max()), tok, ch))

save("output", out)

print("\nper-block absmax (token, channel) summary:")
for i, amax, tok, ch in block_absmax_report:
    print(f"  after_block{i:<2d} absmax={amax:12.4f}  at (token={tok:5d}, channel={ch:4d})")

meta = {
    "source_commit": "f7cf384",
    "script": "tools/ref_pixal3d_ss_flow.py",
    "seed": SEED,
    "dtype": "float32 (saved) / compute_dtype=" + REF_DTYPE,
    "checkpoint": os.path.basename(CKPT),
    "shapes": shapes,
    "block_absmax_report": [
        {"block": i, "absmax": amax, "token": tok, "channel": ch}
        for i, amax, tok, ch in block_absmax_report
    ],
    "tolerance_note": (
        "SS-flow DiT with proj conditioning; weights (cast from bf16 ckpt) computed on GPU "
        f"in {REF_DTYPE}, sdpa attention backend, saved as float32 .npy. "
        "proj_in_channels unset in ckpt -> falls back to cond_channels=1024. Compare vs the "
        "GGML port at rel<2e-2 per intermediate, matching the tolerance convention in "
        "test_ss_flow.cpp. REF_DTYPE=float64 run note: two library-level float32 constants "
        "are NOT patched -- RotaryPositionEmbedder.apply_rotary_embedding hardcodes x.float() "
        "(RoPE stays float32-precision), and TimestepEmbedder.timestep_embedding hardcodes "
        "t.float() (t_embedder+adaLN_modulation run in float32, upconverted to DTYPE via "
        "manual_cast right before the block loop). Everything else (input_layer, out_layer, "
        "all 30 blocks) runs fully in the chosen dtype."
    ),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
