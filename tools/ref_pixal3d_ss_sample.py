#!/usr/bin/env python3
"""End-to-end golden fixture for the Pixal3D SS-flow sampling loop (proj conditioning).

Mirrors tools/ref_pixal3d_ss_flow.py (single forward pass, style/env/save()) and the
inherited tools/ref_ss_sample.py (FlowEulerGuidanceIntervalSampler driving loop,
per-step pred_x_t dump, step_*.npy naming) but for Pixal3D's SparseStructureFlowModel
with image_attn_mode="proj" and its real sparse-structure decoder, run end-to-end
(sample -> decode -> threshold -> coords), exactly the way
Pixal3DImageTo3DPipeline.sample_sparse_structure does it (pixal3d/pipelines/
pixal3d_image_to_3d.py), with real conditioning reused from tools/ref_pixal3d_cond_ss.py
(no DINOv3 recompute) instead of random cond.

Two runs share identical noise/cond/neg_cond/sampler-params and differ ONLY in torso
compute dtype, to calibrate how much the production dtype alone moves the SS-stage
result (per docs/spec/30-pixal3d-cond.md section 5):
  - f32_*  : flow torso converted to float32 (model.convert_to(torch.float32), as
             ref_pixal3d_ss_flow.py does); SS decoder torso also upcast to float32
             (decoder.convert_to_fp32()) after loading its natively-fp16 checkpoint.
  - bf16_* : flow torso left in bfloat16 exactly as the production pipeline runs it
             (ss_flow_img_dit_1_3B_64_bf16_mv.json declares dtype="bfloat16", which
             SparseStructureFlowModel.__init__ applies via self.convert_to(self.dtype)
             automatically -- NOT touched here); SS decoder left at its checkpoint's
             own use_fp16=True default (fp16 torso, f32 input/out layers) -- i.e. the
             exact dtype Pixal3DImageTo3DPipeline.sample_sparse_structure decodes in.
In both runs x_t itself (the sampler state) stays float32 throughout: noise is
generated in float32, sample_once's Euler update is float32 arithmetic, and the
model's own forward() always manual_casts its output back to x.dtype before the
final LayerNorm + out_layer (both of which are plain float32 nn.Linear, never
touched by convert_to, which only touches self.blocks) -- so only the 30
transformer blocks' internal compute differs between runs.

    ATTN_BACKEND=sdpa HF_HOME=/mnt/d/pixal3d/hf_home HF_HUB_OFFLINE=1 \
    PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D OUT=/mnt/hdd1/pixal3d/ref/pixal3d/ss_sample \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_ss_sample.py
"""
import os, sys, json
os.environ["ATTN_BACKEND"] = "sdpa"          # must precede pixal3d import
os.environ.setdefault("HF_HOME", "/mnt/d/pixal3d/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PIXAL3D_REPO = os.environ.get("PIXAL3D_REPO", "/mnt/hdd1/pixal3d/Pixal3D")
sys.path.insert(0, PIXAL3D_REPO)
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
from pixal3d.models.sparse_structure_vae import SparseStructureDecoder
from pixal3d.pipelines.samplers import FlowEulerGuidanceIntervalSampler

WEIGHTS_ROOT = os.environ.get(
    "PIXAL3D_WEIGHTS_ROOT",
    "/mnt/d/pixal3d/hf_home/hub/models--TencentARC--Pixal3D/snapshots/"
    "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
)
CKPT_SS = os.environ.get("PIXAL3D_CKPT_SS", f"{WEIGHTS_ROOT}/ckpts/ss_flow_img_dit_1_3B_64_bf16_mv")
CKPT_DEC = os.environ.get("PIXAL3D_CKPT_SS_DEC", f"{WEIGHTS_ROOT}/ckpts/ss_dec_conv3d_16l8_fp16")
COND_DIR = os.environ.get("COND_DIR", "/mnt/hdd1/pixal3d/ref/pixal3d/cond_ss")
OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/ss_sample")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("REF_DEV", "cuda")
SEED = 42

# SS sampler params, verbatim from pipeline_mv.json's sparse_structure_sampler.params
SAMP = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
            guidance_interval=(0.6, 1.0), rescale_t=5.0)
SIGMA_MIN = 1e-5
SS_RES = 32   # hardcoded in Pixal3DImageTo3DPipeline.run(): ss_res = 32 (passed as
              # `resolution` to sample_sparse_structure -> the max_pool3d target below)

shapes = {}
def save(name, t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:22s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert not np.isnan(a).any() and not np.isinf(a).any(), f"{name} has NaN/Inf"
    return a

def save_int(name, t):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype(np.int32))
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:22s} {str(list(a.shape)):20s}")
    return a

print(f"DEV={DEV} OUT={OUT}")
ss_cfg = json.load(open(CKPT_SS + ".json"))["args"]
dec_cfg = json.load(open(CKPT_DEC + ".json"))["args"]
print("ss flow config:", ss_cfg)
print("ss decoder config:", dec_cfg)

# ---- shared noise (seed 42), saved once ----
g = torch.Generator(device=DEV).manual_seed(SEED)
B, Cin, R = 1, ss_cfg["in_channels"], ss_cfg["resolution"]
noise = torch.randn(B, Cin, R, R, R, generator=g, device=DEV, dtype=torch.float32)
save("noise", noise)

# ---- shared conditioning: reuse the real 4-view DINOv3+ProjGrid dump, no recompute ----
z_global = torch.from_numpy(np.load(f"{COND_DIR}/z_global.npy")).to(DEV, torch.float32)   # [1,5,1024]
z_proj = torch.from_numpy(np.load(f"{COND_DIR}/z_proj.npy")).to(DEV, torch.float32)       # [1,4096,1024]
cond = {"global": z_global, "proj": z_proj}
neg_cond = {"global": torch.zeros_like(z_global), "proj": torch.zeros_like(z_proj)}
save("cond_global", z_global); save("cond_proj", z_proj)
save("neg_global", neg_cond["global"]); save("neg_proj", neg_cond["proj"])

# ---- timestep schedule (deterministic, identical for both runs) ----
t_seq = np.linspace(1, 0, SAMP["steps"] + 1)
t_seq = SAMP["rescale_t"] * t_seq / (1 + (SAMP["rescale_t"] - 1) * t_seq)   # [13]
np.save(f"{OUT}/t_seq.npy", t_seq.astype(np.float32))
shapes["t_seq"] = list(t_seq.shape)
gi0, gi1 = SAMP["guidance_interval"]
guidance_applied = np.array([SAMP["guidance_strength"] if gi0 <= t_seq[i] <= gi1 else 1.0
                              for i in range(SAMP["steps"])], dtype=np.float32)   # [12]
np.save(f"{OUT}/guidance_strength.npy", guidance_applied)
shapes["guidance_strength"] = list(guidance_applied.shape)
print("t_seq:", t_seq)
print("guidance_strength applied per step:", guidance_applied)

sampler = FlowEulerGuidanceIntervalSampler(sigma_min=SIGMA_MIN)


def run_stage(prefix: str, flow_dtype, decoder_to_f32: bool):
    print(f"\n=== stage {prefix} (flow_dtype={flow_dtype}, decoder_to_f32={decoder_to_f32}) ===")
    with torch.device(DEV):
        model = SparseStructureFlowModel(**ss_cfg)   # __init__ already applies
                                                       # convert_to(cfg["dtype"]=bfloat16)
    if flow_dtype is not None:
        model.convert_to(flow_dtype)
        model.input_layer.to(flow_dtype)
        model.out_layer.to(flow_dtype)
    sd = load_file(CKPT_SS + ".safetensors", device=DEV)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd; torch.cuda.empty_cache()
    print(f"flow loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    with torch.device(DEV):
        decoder = SparseStructureDecoder(**dec_cfg)   # __init__ already applies
                                                        # convert_to_fp16() (use_fp16=True)
    sdd = load_file(CKPT_DEC + ".safetensors", device=DEV)
    dmissing, dunexpected = decoder.load_state_dict(sdd, strict=False)
    del sdd; torch.cuda.empty_cache()
    print(f"decoder loaded: missing={len(dmissing)} unexpected={len(dunexpected)}")
    if decoder_to_f32:
        decoder.convert_to_fp32()
    decoder.eval()

    with torch.no_grad():
        out = sampler.sample(model, noise.clone(), cond=cond, neg_cond=neg_cond, verbose=False, **SAMP)
    print("per-step std:", " ".join(f"{x.std():.4f}" for x in out.pred_x_t))

    x_steps = [noise]                 # x_step0 = noise
    x_steps.extend(out.pred_x_t)      # x_step1..x_step{steps} = pred_x_prev after each Euler step
    for k, xs in enumerate(x_steps):
        save(f"{prefix}_x_step{k}", xs)
    x_final = out.samples
    save(f"{prefix}_x_final", x_final)
    assert torch.equal(x_final, x_steps[-1]), "x_final should equal the last recorded step"

    with torch.no_grad():
        occ_logits = decoder(x_final)          # [1,1,64,64,64], cast back to x_final.dtype (f32)
    save(f"{prefix}_occ_logits", occ_logits)
    decoded = occ_logits > 0
    if SS_RES != decoded.shape[2]:
        ratio = decoded.shape[2] // SS_RES
        decoded = F.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    save_int(f"{prefix}_coords", coords)

    del model, decoder
    torch.cuda.empty_cache()
    return {
        "x_steps": [a.detach().to(torch.float32).cpu().numpy() for a in x_steps],
        "x_final": x_final.detach().to(torch.float32).cpu().numpy(),
        "coords": coords.detach().cpu().numpy(),
        "N": int(coords.shape[0]),
    }


res_f32 = run_stage("f32", torch.float32, decoder_to_f32=True)
res_bf16 = run_stage("bf16", None, decoder_to_f32=False)

# ---------------- sanity ----------------
print("\n=== sanity ===")
per_step_maxd = []
for k in range(len(res_f32["x_steps"])):
    d = float(np.abs(res_f32["x_steps"][k] - res_bf16["x_steps"][k]).max())
    per_step_maxd.append(d)
    print(f"  x_step{k:2d} max|d|(f32 vs bf16) = {d:.4e}")

def coord_set(a):
    return set(tuple(row) for row in a.tolist())

s32, s16 = coord_set(res_f32["coords"]), coord_set(res_bf16["coords"])
inter, union = len(s32 & s16), len(s32 | s16)
iou = inter / union if union > 0 else float("nan")
print(f"N(f32)={res_f32['N']} N(bf16)={res_bf16['N']} intersection={inter} union={union} IoU={iou:.4f}")

for lbl, res in (("f32", res_f32), ("bf16", res_bf16)):
    for k, a in enumerate(res["x_steps"]):
        assert not np.isnan(a).any() and not np.isinf(a).any(), f"{lbl} x_step{k} has NaN/Inf"

meta = {
    "source_commit": "f7cf384",
    "script": "tools/ref_pixal3d_ss_sample.py",
    "seed": SEED,
    "dtype": {
        "f32": "flow torso float32 (weights cast from bf16 ckpt); SS decoder torso "
               "converted to float32 after loading its native fp16 checkpoint "
               "(decoder.convert_to_fp32())",
        "bf16": "flow torso bfloat16 (production default, ss_flow ckpt json dtype="
                "'bfloat16', un-touched); SS decoder left at its checkpoint's own "
                "use_fp16=True default (fp16 torso, f32 input/out layers) -- exactly "
                "how Pixal3DImageTo3DPipeline.sample_sparse_structure decodes",
        "common": "x_t (sampler state) is float32 throughout both runs; the model's "
                  "forward() always manual_casts back to x.dtype before the final "
                  "LayerNorm + out_layer, so input_layer/out_layer/t_embedder/"
                  "adaLN_modulation are float32 in both runs -- only the 30 transformer "
                  "blocks' internal compute dtype differs",
    },
    "checkpoint_ss_flow": os.path.basename(CKPT_SS),
    "checkpoint_ss_decoder": os.path.basename(CKPT_DEC),
    "ss_flow_config": ss_cfg,
    "ss_decoder_config": dec_cfg,
    "cond_source": COND_DIR,
    "sampler": "FlowEulerGuidanceIntervalSampler",
    "sampler_sigma_min": SIGMA_MIN,
    "sampler_params": SAMP,
    "ss_res_maxpool_target": SS_RES,
    "shapes": shapes,
    "sanity": {
        "N_f32": res_f32["N"],
        "N_bf16": res_bf16["N"],
        "coords_intersection": inter,
        "coords_union": union,
        "coords_iou": iou,
        "per_step_max_abs_diff_f32_vs_bf16": per_step_maxd,
    },
    "tolerance_note": (
        "End-to-end SS-stage sampling fixture: real 4-view DINOv3+ProjGrid conditioning "
        "(reused from tools/ref_pixal3d_cond_ss.py, not recomputed), FlowEulerGuidanceIntervalSampler "
        "with the exact pipeline_mv.json params (steps=12, guidance_strength=7.5, "
        "guidance_rescale=0.7, guidance_interval=[0.6,1.0], rescale_t=5.0, sigma_min=1e-5), "
        "decoded with the real ss_dec_conv3d_16l8 decoder and max-pooled 64->32 exactly as "
        "Pixal3DImageTo3DPipeline.run() does (ss_res=32 hardcoded). The f32-vs-bf16 IoU/"
        "per-step deltas calibrate how much the production torso dtype alone moves the "
        "SS occupancy result -- this end-to-end coordinate-set parity, not block-level "
        "tensor parity, is the production gate per docs/spec/30-pixal3d-cond.md section 5/6."
    ),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
