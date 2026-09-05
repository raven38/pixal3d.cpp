#!/usr/bin/env python3
"""End-to-end golden fixture for the Pixal3D shape-SLAT (res-512 stage) sampling loop.

Mirrors tools/ref_pixal3d_ss_sample.py (f32 + bf16 calibration runs, per-step latent dump,
style/env/save(), meta.json) but for the shape-SLAT stage: ElasticSLatFlowModel
(`slat_flow_img2shape_dit_1_3B_512_bf16_mv`, image_attn_mode="proj", proj_in_channels=2048)
driven exactly the way Pixal3DImageTo3DPipeline.sample_shape_slat does it
(pixal3d/pipelines/pixal3d_image_to_3d.py: get_proj_cond_shape + sample_shape_slat, called
from run() as "Stage 2: Shape LR 512 (proj)" -- coords come from the SS stage, unmodified;
this is NOT the 512->1024 HR cascade, i.e. sample_shape_slat_cascade / upsample() are not
exercised here).

Inputs are reused, not recomputed:
  - SS-stage active coords: tools/ref_pixal3d_ss_sample.py's f32_coords.npy [N,4] int32
    (b,x,y,z at res 32, N=4377) -- the task spec is explicit that BOTH dtype runs share
    these same (f32-run) coords, so the two runs differ only in flow-torso dtype, not in
    sparse topology.
  - SLAT-stage projection conditioning: the cond_slat dump's s512_z_global.npy [1,5,1024]
    and s512_z_proj.npy [1,32768,2048] (dense R=32 grid, token = x*32*32 + y*32 + z, per
    docs/spec/30-pixal3d-cond.md section 1/2). get_proj_cond_shape's own step --
    z_proj.reshape(B,R,R,R,-1)[b,x,y,z] gather at the SS coords -- IS reproduced here
    (that gather is stage-specific, sparse-topology-dependent, and cheap; only the DINOv3 +
    ProjGrid feature extraction upstream of it is reused).

Two runs share identical noise/cond/neg_cond/sampler params and differ ONLY in flow-torso
compute dtype (docs/spec/30-pixal3d-cond.md section 5 calibration):
  - f32_*  : flow torso converted to float32 (model.convert_to(torch.float32), matching
             ref_pixal3d_ss_sample.py); additionally, only for this run, the SLAT is decoded
             with the real shape decoder (shape_dec_next_dc_f16c32_fp16 / FlexiDualGridVaeDecoder)
             the way Pixal3DImageTo3DPipeline.decode_shape_slat does it
             (decoder.set_resolution(512); decoder(slat, return_subs=True)), left at the
             decoder's own production dtype (use_fp16=True checkpoint default -- NOT upcast,
             unlike the SS decoder in ref_pixal3d_ss_sample.py's f32 run). Note
             decode_shape_slat is only ever invoked in production on the final HR (1024/1536)
             slat inside sample_shape_slat_cascade; applying it here to the res-512-stage
             slat is done purely for verification/documentation (grid_size=512 matches this
             stage's R=32 latent grid * the decoder's fixed 4x up-sampling chain), and is not
             something the pipeline itself does for this stage.
  - bf16_* : flow torso left in bfloat16, exactly as production
             (slat_flow_img2shape_dit_1_3B_512_bf16_mv.json declares dtype="bfloat16", which
             SLatFlowModel.__init__ applies via self.convert_to(self.dtype) automatically --
             NOT touched here). No decode step for this run.
In both runs x_t itself (the sampler state, held as SparseTensor.feats) stays float32
throughout: noise is generated in float32, sample_once's Euler update is float32
arithmetic (VarLenTensor elemwise ops on plain float32 feats), and the model's own
forward() always manual_casts its output back to x.dtype before the final LayerNorm +
out_layer (both plain float32 sp.SparseLinear, never touched by convert_to, which only
touches self.blocks) -- so only the 30 transformer blocks' internal compute differs
between runs; input_layer/out_layer/t_embedder/adaLN_modulation are float32 in both.

    ATTN_BACKEND=sdpa HF_HOME=/mnt/d/pixal3d/hf_home HF_HUB_OFFLINE=1 \
    PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D OUT=/mnt/hdd1/pixal3d/ref/pixal3d/slat_sample \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_slat_sample.py
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
from safetensors.torch import load_file
from pixal3d.models.structured_latent_flow import ElasticSLatFlowModel
from pixal3d.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines.samplers import FlowEulerGuidanceIntervalSampler

WEIGHTS_ROOT = os.environ.get(
    "PIXAL3D_WEIGHTS_ROOT",
    "/mnt/d/pixal3d/hf_home/hub/models--TencentARC--Pixal3D/snapshots/"
    "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
)
CKPT_FLOW = os.environ.get("PIXAL3D_CKPT_SLAT_512", f"{WEIGHTS_ROOT}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv")
CKPT_DEC = os.environ.get("PIXAL3D_CKPT_SHAPE_DEC", f"{WEIGHTS_ROOT}/ckpts/shape_dec_next_dc_f16c32_fp16")
SS_DIR = os.environ.get("SS_DIR", "/mnt/hdd1/pixal3d/ref/pixal3d/ss_sample")
COND_DIR = os.environ.get("COND_DIR", "/mnt/hdd1/pixal3d/ref/pixal3d/cond_slat")
OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/slat_sample")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("REF_DEV", "cuda")
SEED = 42
GRID_R = 32   # shape-512 stage proj grid resolution
DEC_RES = 512

# shape_slat_sampler params, verbatim from pipeline_mv.json
SAMP = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.5,
            guidance_interval=(0.6, 1.0), rescale_t=3.0)
SIGMA_MIN = 1e-5

shapes = {}
def save(name, a_or_t):
    if torch.is_tensor(a_or_t):
        a = np.ascontiguousarray(a_or_t.detach().to(torch.float32).cpu().numpy())
    else:
        a = np.ascontiguousarray(a_or_t)
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:22s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert not np.isnan(a).any() and not np.isinf(a).any(), f"{name} has NaN/Inf"
    return a

def save_int(name, t):
    a = np.ascontiguousarray(np.asarray(t).astype(np.int32))
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:22s} {str(list(a.shape)):20s}")
    return a

print(f"DEV={DEV} OUT={OUT}")
flow_cfg = json.load(open(CKPT_FLOW + ".json"))["args"]
dec_cfg = json.load(open(CKPT_DEC + ".json"))["args"]
print("shape slat flow config:", flow_cfg)
print("shape decoder config:", dec_cfg)
assert flow_cfg["resolution"] == GRID_R and flow_cfg["in_channels"] == 32 and flow_cfg["out_channels"] == 32
assert flow_cfg["proj_in_channels"] == 2048

# ---- reuse: SS-stage active coords (f32 run) ----
coords_np = np.load(f"{SS_DIR}/f32_coords.npy").astype(np.int32)   # [N,4] (b,x,y,z)
N = coords_np.shape[0]
coords = torch.from_numpy(coords_np).to(DEV, torch.int32)
save_int("coords", coords_np)
print(f"N = {N}")

# ---- shared noise (seed 42), saved once ----
g = torch.Generator(device=DEV).manual_seed(SEED)
Cin = flow_cfg["in_channels"]
noise_feats = torch.randn(N, Cin, generator=g, device=DEV, dtype=torch.float32)
save("noise", noise_feats)

# ---- shared conditioning: gather s512 z_proj at the SS coords, exactly as
#      Pixal3DImageTo3DPipeline.get_proj_cond_shape does (z_proj.reshape(B,R,R,R,-1)
#      [b,x,y,z]); z_global is copied through unchanged (dense, not sparse-indexed) ----
z_global_np = np.load(f"{COND_DIR}/s512_z_global.npy")             # [1,5,1024]
z_proj_dense_np = np.load(f"{COND_DIR}/s512_z_proj.npy")           # [1,32768,2048]
assert z_global_np.shape == (1, 5, 1024)
assert z_proj_dense_np.shape == (1, GRID_R ** 3, 2048)
z_global = torch.from_numpy(z_global_np).to(DEV, torch.float32)
z_proj_grid = torch.from_numpy(z_proj_dense_np).to(DEV, torch.float32).reshape(1, GRID_R, GRID_R, GRID_R, -1)
b_idx = coords[:, 0].long(); x_idx = coords[:, 1].long(); y_idx = coords[:, 2].long(); z_idx = coords[:, 3].long()
z_proj_sparse = z_proj_grid[b_idx, x_idx, y_idx, z_idx]            # [N,2048]
save("cond_global", z_global)
save("cond_proj", z_proj_sparse)

cond = {"global": z_global, "proj": SparseTensor(feats=z_proj_sparse, coords=coords)}
neg_cond = {"global": torch.zeros_like(z_global),
            "proj": SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)}

# ---- normalization (pipeline_mv.json shape_slat_normalization) ----
norm = json.load(open(f"{WEIGHTS_ROOT}/pipeline_mv.json"))["args"]["shape_slat_normalization"]
norm_mean = np.array(norm["mean"], dtype=np.float32)
norm_std = np.array(norm["std"], dtype=np.float32)
assert norm_mean.shape == (32,) and norm_std.shape == (32,)
save("norm_mean", norm_mean)
save("norm_std", norm_std)
mean_t = torch.from_numpy(norm_mean)[None].to(DEV)
std_t = torch.from_numpy(norm_std)[None].to(DEV)

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


def run_stage(prefix: str, flow_dtype, decode: bool):
    print(f"\n=== stage {prefix} (flow_dtype={flow_dtype}, decode={decode}) ===")
    with torch.device(DEV):
        model = ElasticSLatFlowModel(**flow_cfg)   # __init__ already applies
                                                     # convert_to(cfg["dtype"]=bfloat16)
    if flow_dtype is not None:
        model.convert_to(flow_dtype)
        model.input_layer.to(flow_dtype)
        model.out_layer.to(flow_dtype)
    sd = load_file(CKPT_FLOW + ".safetensors", device=DEV)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd; torch.cuda.empty_cache()
    print(f"flow loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    noise = SparseTensor(feats=noise_feats.clone(), coords=coords)
    with torch.no_grad():
        out = sampler.sample(model, noise, cond=cond, neg_cond=neg_cond, verbose=False, **SAMP)
    print("per-step std:", " ".join(f"{x.feats.std():.4f}" for x in out.pred_x_t))

    x_steps = [noise.feats]                          # x_step0 = noise
    x_steps.extend(st.feats for st in out.pred_x_t)  # x_step1..x_step{steps}
    for k, xs in enumerate(x_steps):
        save(f"{prefix}_x_step{k}", xs)
    x_final_feats = out.samples.feats
    save(f"{prefix}_x_final", x_final_feats)
    assert torch.equal(x_final_feats, x_steps[-1]), "x_final should equal the last recorded step"

    # denormalize exactly as Pixal3DImageTo3DPipeline.sample_shape_slat does: slat = slat * std + mean
    slat_feats = x_final_feats * std_t + mean_t
    save(f"{prefix}_slat", slat_feats)

    del model
    torch.cuda.empty_cache()

    dec_out = None
    if decode:
        with torch.device(DEV):
            decoder = FlexiDualGridVaeDecoder(**dec_cfg)   # __init__ already applies
                                                             # convert_to_fp16() (use_fp16=True)
        sdd = load_file(CKPT_DEC + ".safetensors", device=DEV)
        dmissing, dunexpected = decoder.load_state_dict(sdd, strict=False)
        del sdd; torch.cuda.empty_cache()
        print(f"decoder loaded: missing={len(dmissing)} unexpected={len(dunexpected)}")
        decoder.eval()
        decoder.set_resolution(DEC_RES)   # as Pixal3DImageTo3DPipeline.decode_shape_slat does
        slat_st = SparseTensor(feats=slat_feats.contiguous(), coords=coords)
        with torch.no_grad():
            # decode_shape_slat: self.models['shape_slat_decoder'](slat, return_subs=True)
            mesh_list, subs = decoder(slat_st, return_subs=True)
        mesh = mesh_list[0]
        verts = mesh.vertices.detach().to(torch.float32).cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy().astype(np.int32)
        save(f"{prefix}_dec_vertices", verts)
        save_int(f"{prefix}_dec_faces", faces)
        print(f"  decoded mesh: Nv={verts.shape[0]} Nf={faces.shape[0]}")
        print(f"  subs (intermediate sparse subdivision masks, {len(subs)} levels, NOT saved -- "
              f"not mesh/dual-grid attribute data, just boolean upsample masks): "
              + ", ".join(f"lvl{i}:{s.feats.shape[0]}" for i, s in enumerate(subs)))
        dec_out = {"n_vertices": int(verts.shape[0]), "n_faces": int(faces.shape[0])}
        del decoder
        torch.cuda.empty_cache()

    return {
        "x_steps": [a.detach().to(torch.float32).cpu().numpy() for a in x_steps],
        "x_final": x_final_feats.detach().to(torch.float32).cpu().numpy(),
        "slat": slat_feats.detach().to(torch.float32).cpu().numpy(),
        "decode": dec_out,
    }


res_f32 = run_stage("f32", torch.float32, decode=True)
res_bf16 = run_stage("bf16", None, decode=False)

# ---------------- sanity ----------------
print("\n=== sanity ===")
per_step_maxd = []
for k in range(len(res_f32["x_steps"])):
    d = float(np.abs(res_f32["x_steps"][k] - res_bf16["x_steps"][k]).max())
    per_step_maxd.append(d)
    tag = " <-- reported" if k in (1, 6, 12) else ""
    print(f"  x_step{k:2d} max|d|(f32 vs bf16) = {d:.4e}{tag}")

final_maxd = float(np.abs(res_f32["x_final"] - res_bf16["x_final"]).max())
final_maxabs_f32 = float(np.abs(res_f32["x_final"]).max())
rel = final_maxd / final_maxabs_f32 if final_maxabs_f32 > 0 else float("nan")
print(f"x_final: max|d|(f32 vs bf16) = {final_maxd:.4e}, max|f32| = {final_maxabs_f32:.4e}, rel = {rel:.4e}")

for lbl, res in (("f32", res_f32), ("bf16", res_bf16)):
    for k, a in enumerate(res["x_steps"]):
        assert not np.isnan(a).any() and not np.isinf(a).any(), f"{lbl} x_step{k} has NaN/Inf"
    assert not np.isnan(res["slat"]).any() and not np.isinf(res["slat"]).any(), f"{lbl} slat has NaN/Inf"

def stats(a):
    return {"mean": float(a.mean()), "std": float(a.std()), "absmax": float(np.abs(a).max())}

meta = {
    "source_commit": "f7cf384",
    "script": "tools/ref_pixal3d_slat_sample.py",
    "seed": SEED,
    "stage": "shape_slat_512 (LR shape SLat, single-resolution -- NOT the 512->1024 cascade)",
    "dtype": {
        "f32": "flow torso float32 (weights cast from bf16 ckpt); shape decoder (only run for "
               "this dtype) left at its checkpoint's own use_fp16=True production default "
               "(fp16 torso, f32 from_latent/output_layer) -- NOT upcast, unlike the SS-stage "
               "decoder calibration in ref_pixal3d_ss_sample.py",
        "bf16": "flow torso bfloat16 (production default, slat_flow_img2shape_dit_1_3B_512_bf16_mv "
                "ckpt json dtype='bfloat16', un-touched); no decode step for this run",
        "common": "x_t (sampler state, SparseTensor.feats) is float32 throughout both runs; the "
                  "model's forward() always manual_casts back to x.dtype before the final "
                  "LayerNorm + out_layer, so input_layer/out_layer/t_embedder/adaLN_modulation "
                  "are float32 in both runs -- only the 30 transformer blocks' internal compute "
                  "dtype differs",
    },
    "checkpoint_shape_slat_flow_512": os.path.basename(CKPT_FLOW),
    "checkpoint_shape_decoder": os.path.basename(CKPT_DEC),
    "shape_slat_flow_config": flow_cfg,
    "shape_decoder_config": dec_cfg,
    "coords_source": f"{SS_DIR}/f32_coords.npy (reused, both dtype runs share these coords)",
    "cond_source": f"{COND_DIR}/s512_z_global.npy, {COND_DIR}/s512_z_proj.npy (reused DINOv3+ProjGrid "
                   "dump; only the sparse-coord gather into z_proj_sparse is recomputed here, per "
                   "Pixal3DImageTo3DPipeline.get_proj_cond_shape)",
    "sampler": "FlowEulerGuidanceIntervalSampler",
    "sampler_sigma_min": SIGMA_MIN,
    "sampler_params": SAMP,
    "decoder_resolution": DEC_RES,
    "decoder_note": "decode_shape_slat is only invoked in production on the final HR "
                    "(1024/1536) slat, inside sample_shape_slat_cascade; here it is applied "
                    "to this res-512-stage slat directly (grid_size=512 matches the decoder's "
                    "fixed 4x up-sampling chain from the R=32 latent grid) purely for "
                    "verification, which the pipeline itself never does for this stage.",
    "shapes": shapes,
    "sanity": {
        "per_step_max_abs_diff_f32_vs_bf16": per_step_maxd,
        "step1_max_abs_diff": per_step_maxd[1],
        "step6_max_abs_diff": per_step_maxd[6],
        "step12_max_abs_diff": per_step_maxd[12],
        "x_final_stats_f32": stats(res_f32["x_final"]),
        "x_final_stats_bf16": stats(res_bf16["x_final"]),
        "x_final_max_abs_diff_f32_vs_bf16": final_maxd,
        "x_final_rel_max_abs_diff": rel,
        "slat_stats_f32": stats(res_f32["slat"]),
        "slat_stats_bf16": stats(res_bf16["slat"]),
        "decode_f32": res_f32["decode"],
    },
    "tolerance_note": (
        "End-to-end shape-SLAT (res-512) sampling fixture: real SS-stage coords and SLAT-stage "
        "projection conditioning reused (not recomputed), FlowEulerGuidanceIntervalSampler with "
        "the exact pipeline_mv.json shape_slat_sampler params (steps=12, guidance_strength=7.5, "
        "guidance_rescale=0.5, guidance_interval=[0.6,1.0], rescale_t=3.0, sigma_min=1e-5), "
        "denormalized with shape_slat_normalization mean/std exactly as sample_shape_slat does. "
        "The f32-vs-bf16 per-step / x_final deltas calibrate how much the production flow-torso "
        "dtype alone moves the SLAT sampling result -- per docs/spec/30-pixal3d-cond.md section "
        "5/6, this end-to-end sampling parity (not block-level tensor parity) is the production "
        "gate; C++ SLAT-sample tests should compare against the bf16 run in bf16/f16 compute, "
        "with f32 as the calibration ceiling."
    ),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
