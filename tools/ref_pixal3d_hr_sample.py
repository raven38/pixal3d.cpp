#!/usr/bin/env python3
"""End-to-end golden fixture for the Pixal3D HR shape-SLAT (res-1024 cascade stage)
and texture-SLAT (res-1024) sampling loops.

Mirrors tools/ref_pixal3d_slat_sample.py (f32 + bf16 calibration runs, per-step latent
dump, save()/save_int(), meta.json) but for the two remaining flow stages of the
1024_cascade pipeline: ElasticSLatFlowModel `slat_flow_img2shape_dit_1_3B_1024_bf16_mv`
(HR shape) and `slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv` (texture), driven exactly
the way Pixal3DImageTo3DPipeline.run() (pixal3d/pipelines/pixal3d_image_to_3d.py,
pipeline_type="1024_cascade", inherited verbatim by Pixal3DMVImageTo3DPipeline in
pixal3d_mv_image_to_3d.py -- only get_proj_cond_ss/get_proj_cond_shape are overridden
there for multiview) does it, for "Stage 3a: Upsample LR -> HR", "Stage 3b: Shape HR
(proj)" and "Stage 4: Texture (proj)".

LR -> HR coordinate procedure (reverse-engineered from run(), verbatim):
  1. hr_coords = shape_slat_decoder.upsample(lr_slat, upsample_times=4)
     FlexiDualGridVaeDecoder.upsample() (sc_vaes/sparse_unet_vae.py) runs
     from_latent(x) then the decoder's SparseResBlock3d stacks for resolution stages
     i=0..3 (each stage's *last* block does an actual upsample=True SparseUpsample(2),
     so 4 stages = 16x spatial upsample), and returns h.coords right before starting
     stage i=4 -- i.e. real neural-net compute (ConvNeXt3d blocks + subdivision
     prediction inside each SparseResBlock3d), not a topological guess. Since the
     res-512 LR SLAT lives on a 32^3 latent grid, 16x lands its active coords on the
     512^3 voxel grid (lr_resolution=512, hardcoded in run()). Saved as
     lr_decoded_coords.npy (raw, 0..511 range, one child voxel per active leaf --
     duplicates possible across coarse parents, NOT yet deduplicated).
  2. Quantize to the HR flow's sparse-token grid (grid_res = hr_resolution//16 = 64
     for the 1024 stage) and de-duplicate:
       quant_coords = cat([hr_coords[:,:1],
                            round((hr_coords[:,1:] + 0.5) / 512 * (64 - 1))])
       hr_coords_unique = quant_coords.unique(dim=0)
     For pipeline_type="1024_cascade" the token-budget while-loop in run() always
     breaks on its first iteration (its condition is `... or actual_hr_resolution ==
     1024`, and actual_hr_resolution starts at 1024), so max_num_tokens never actually
     reduces the HR resolution here -- that reduction path only matters for
     pipeline_type="1536_cascade". Saved as upsampled_coords.npy (pre-unique, same
     grid_res=64 scale) and hr_coords.npy (post-unique -- this is exactly the sparse
     coordinate set fed to both the HR shape flow and the texture flow, and also what
     get_proj_cond_shape gathers proj-conditioning at).
  Divergence note: Trellis2ImageTo3DPipeline.sample_shape_slat_cascade (the base-class
  cascade helper, still present and used by trellis2_image_to_3d.py's own run()) uses a
  *different* quantization formula for the same step -- `((hr_coords[:,1:] + 0.5) /
  lr_resolution * (hr_resolution // 16)).int()` (no .round(), and multiplies by
  grid_res rather than grid_res-1). Pixal3DImageTo3DPipeline.run() does NOT call that
  helper for the 1024_cascade path -- it inlines its own version with `.round().int()`
  and `(grid_res - 1)`, which is what this script reproduces exactly.

concat_cond finding (Pixal3DImageTo3DPipeline.sample_tex_slat, reproduced inline here
since we need per-step latents, not just the final sample):
  shape_slat (the just-denormalized HR shape SLAT, `hr_slat * std + mean` in run()) is
  RE-NORMALIZED with the *shape* slat_normalization stats -- `(shape_slat - mean) /
  std` -- and that renormalized 32-channel tensor is passed as `concat_cond=` straight
  through FlowEulerGuidanceIntervalSampler.sample(**kwargs) into the tex
  ElasticSLatFlowModel.forward(), which does `x = sp.sparse_cat([x, concat_cond],
  dim=-1)` (structured_latent_flow.py) -- i.e. concat_cond is concatenated AFTER the
  tex noise channels (tex noise first, 32 ch; shape SLAT normalized, 32 ch appended),
  giving the tex flow's in_channels=64 (out_channels stays 32). This is why
  `noise = shape_slat.replace(feats=torch.randn(N, in_channels - shape_slat.feats.shape[1]))`
  only allocates 32 (not 64) noise channels. Saved per-dtype-run as
  {prefix}_tex_concat_cond.npy since it's derived from that same run's own HR shape
  SLAT (f32 and bf16 cascades each produce a slightly different shape SLAT, hence a
  slightly different concat_cond, even though hr_coords.npy is shared).

Inputs reused, not recomputed:
  - res-512 shape SLAT stage: tools/ref_pixal3d_slat_sample.py's coords.npy [N,4] int32
    (SS-stage res-32 coords) and f32_slat.npy [N,32] (denormalized LR shape SLAT --
    reused as *the* LR input to the upsample() step for both the f32 and bf16 HR runs,
    exactly as the f32/bf16 runs in ref_pixal3d_slat_sample.py already share one set of
    SS coords: the topology up to hr_coords_unique is deterministic given a fixed LR
    SLAT, so both dtype runs downstream must share it to be comparable).
  - 4-view images/cameras: tools/ref_pixal3d_cond_slat.py's s1024_images.npy [1,V,3,1024,1024]
    (already alpha-premultiplied, NOT ImageNet-normalized -- DinoV3ProjMultiViewFeatureExtractor.forward
    applies self.transform internally), camera_angle_x.npy [1,V], distance.npy [1,V],
    transform_matrix.npy [1,V,4,4], mesh_scale.npy [1].
  - The shape_1024 / tex_1024 DinoV3ProjMultiViewFeatureExtractor condition models
    ARE (re)built and run here (not reused as dumped tensors) because they must be
    queried at the actual hr_coords_unique sparse coordinates, which depend on the LR
    SLAT and are stage-specific (get_proj_cond_shape's `z_proj.reshape(B,R,R,R,-1)
    [b,x,y,z]` gather), per docs/spec/30-pixal3d-cond.md section 1/2 and
    inference_mv.IMAGE_COND_CONFIGS["shape_1024"|"tex_1024"].

Two full cascades (shape stage then tex stage) share identical hr_coords / noise /
cond / neg_cond / sampler params and differ ONLY in flow-torso compute dtype, exactly
like ref_pixal3d_slat_sample.py's f32_/bf16_ split:
  - f32_ : both flow torsos converted to float32 (model.convert_to(torch.float32));
           shape decoder run at set_resolution(1024) and tex decoder run with
           guide_subs from that decode, both left at their checkpoints' own
           use_fp16=True production default (NOT upcast) -- decode is only ever done
           for this dtype, purely for verification (decode_shape_slat/decode_tex_slat
           are only invoked in production on the final HR/cascade slat, which this IS).
  - bf16_: both flow torsos left in bfloat16 (production default, the
           *_1024_bf16_mv ckpt jsons declare dtype="bfloat16", applied automatically
           by ElasticSLatFlowModel.__init__ via self.convert_to(self.dtype)); no
           decode step for this run.
  - common: x_t (sampler state, SparseTensor.feats) is float32 throughout both runs
            for both stages, for the same reason as the res-512 fixture (forward()
            manual_casts back to x.dtype before the final LayerNorm + out_layer) --
            input_layer/out_layer/t_embedder/adaLN_modulation stay float32 in both.

    ATTN_BACKEND=sdpa HF_HOME=/mnt/d/pixal3d/hf_home HF_HUB_OFFLINE=1 \
    PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D OUT=/mnt/hdd1/pixal3d/ref/pixal3d/hr_sample \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_hr_sample.py
"""
import os, sys, json, time, subprocess
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
from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjMultiViewFeatureExtractor,
)

WEIGHTS_ROOT = os.environ.get(
    "PIXAL3D_WEIGHTS_ROOT",
    "/mnt/d/pixal3d/hf_home/hub/models--TencentARC--Pixal3D/snapshots/"
    "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
)
CKPT_SHAPE_HR = os.environ.get("PIXAL3D_CKPT_SHAPE_HR", f"{WEIGHTS_ROOT}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv")
CKPT_TEX_HR = os.environ.get("PIXAL3D_CKPT_TEX_HR", f"{WEIGHTS_ROOT}/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv")
CKPT_SHAPE_DEC = os.environ.get("PIXAL3D_CKPT_SHAPE_DEC", f"{WEIGHTS_ROOT}/ckpts/shape_dec_next_dc_f16c32_fp16")
CKPT_TEX_DEC = os.environ.get("PIXAL3D_CKPT_TEX_DEC", f"{WEIGHTS_ROOT}/ckpts/tex_dec_next_dc_f16c32_fp16")
SLAT_DIR = os.environ.get("SLAT_DIR", "/mnt/hdd1/pixal3d/ref/pixal3d/slat_sample")
COND_DIR = os.environ.get("COND_DIR", "/mnt/hdd1/pixal3d/ref/pixal3d/cond_slat")
OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/hr_sample")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("REF_DEV", "cuda")
SHAPE_SEED = 42
TEX_SEED = 43
LR_RESOLUTION = 512
HR_RESOLUTION = 1024
GRID_R = HR_RESOLUTION // 16   # 64
MAX_NUM_TOKENS = 49152
DECODE_RES = 1024
GPU_WAIT_THRESHOLD_MIB = 12000
GPU_WAIT_POLL_S = 15
GPU_WAIT_TIMEOUT_S = 1800

# shape_slat_sampler / tex_slat_sampler params, verbatim from pipeline_mv.json
SHAPE_SAMP = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.5,
                  guidance_interval=(0.6, 1.0), rescale_t=3.0)
TEX_SAMP = dict(steps=12, guidance_strength=1.0, guidance_rescale=0.0,
                guidance_interval=(0.6, 0.9), rescale_t=3.0)
SIGMA_MIN = 1e-5

# inference_mv.IMAGE_COND_CONFIGS["shape_1024"] / ["tex_1024"]
SHAPE_1024_CFG = {
    "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "image_size": 1024,
    "grid_resolution": 64,
    "use_naf_upsample": True,
    "naf_target_size": 512,
    "multiview_fusion": "average",
}
TEX_1024_CFG = {
    "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "image_size": 1024,
    "grid_resolution": 64,
    "use_naf_upsample": True,
    "naf_target_size": 1024,
    "multiview_fusion": "average",
}

shapes = {}
def save(name, a_or_t):
    if torch.is_tensor(a_or_t):
        a = np.ascontiguousarray(a_or_t.detach().to(torch.float32).cpu().numpy())
    else:
        a = np.ascontiguousarray(a_or_t)
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:24s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert not np.isnan(a).any() and not np.isinf(a).any(), f"{name} has NaN/Inf"
    return a

def save_int(name, t):
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    a = np.ascontiguousarray(np.asarray(t).astype(np.int32))
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:24s} {str(list(a.shape)):20s}")
    return a

def gpu_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return max(int(x.strip()) for x in out if x.strip())
    except Exception as e:
        print(f"[gpu-wait] nvidia-smi query failed ({e}), assuming 0")
        return 0

def wait_for_vram(threshold_mib=GPU_WAIT_THRESHOLD_MIB, poll_s=GPU_WAIT_POLL_S, timeout_s=GPU_WAIT_TIMEOUT_S, tag=""):
    t0 = time.time()
    used = gpu_used_mib()
    while used >= threshold_mib:
        elapsed = time.time() - t0
        print(f"[gpu-wait{(' '+tag) if tag else ''}] used={used} MiB >= {threshold_mib} MiB, waiting ({elapsed:.0f}s elapsed)...")
        if elapsed > timeout_s:
            print(f"[gpu-wait{(' '+tag) if tag else ''}] timeout after {elapsed:.0f}s, proceeding anyway at used={used} MiB")
            return used
        time.sleep(poll_s)
        used = gpu_used_mib()
    print(f"[gpu-wait{(' '+tag) if tag else ''}] OK, used={used} MiB < {threshold_mib} MiB")
    return used

print(f"DEV={DEV} OUT={OUT}")
t_wall0 = time.time()
torch.cuda.reset_peak_memory_stats()

shape_hr_cfg = json.load(open(CKPT_SHAPE_HR + ".json"))["args"]
tex_hr_cfg = json.load(open(CKPT_TEX_HR + ".json"))["args"]
shape_dec_cfg = json.load(open(CKPT_SHAPE_DEC + ".json"))["args"]
tex_dec_cfg = json.load(open(CKPT_TEX_DEC + ".json"))["args"]
print("shape HR flow config:", shape_hr_cfg)
print("tex HR flow config:", tex_hr_cfg)
assert shape_hr_cfg["resolution"] == GRID_R and shape_hr_cfg["in_channels"] == 32 and shape_hr_cfg["out_channels"] == 32
assert tex_hr_cfg["resolution"] == GRID_R and tex_hr_cfg["in_channels"] == 64 and tex_hr_cfg["out_channels"] == 32
assert shape_hr_cfg["proj_in_channels"] == 2048 and tex_hr_cfg["proj_in_channels"] == 2048

pipeline_mv = json.load(open(f"{WEIGHTS_ROOT}/pipeline_mv.json"))["args"]
shape_norm = pipeline_mv["shape_slat_normalization"]
tex_norm = pipeline_mv["tex_slat_normalization"]
shape_mean_np = np.array(shape_norm["mean"], dtype=np.float32)
shape_std_np = np.array(shape_norm["std"], dtype=np.float32)
tex_mean_np = np.array(tex_norm["mean"], dtype=np.float32)
tex_std_np = np.array(tex_norm["std"], dtype=np.float32)
save("shape_norm_mean", shape_mean_np); save("shape_norm_std", shape_std_np)
save("tex_norm_mean", tex_mean_np); save("tex_norm_std", tex_std_np)
shape_mean_t = torch.from_numpy(shape_mean_np)[None].to(DEV)
shape_std_t = torch.from_numpy(shape_std_np)[None].to(DEV)
tex_mean_t = torch.from_numpy(tex_mean_np)[None].to(DEV)
tex_std_t = torch.from_numpy(tex_std_np)[None].to(DEV)

save("shape_sampler_params", np.array(
    [SHAPE_SAMP["steps"], SHAPE_SAMP["guidance_strength"], SHAPE_SAMP["guidance_rescale"],
     SHAPE_SAMP["guidance_interval"][0], SHAPE_SAMP["guidance_interval"][1], SHAPE_SAMP["rescale_t"]],
    dtype=np.float32))
save("tex_sampler_params", np.array(
    [TEX_SAMP["steps"], TEX_SAMP["guidance_strength"], TEX_SAMP["guidance_rescale"],
     TEX_SAMP["guidance_interval"][0], TEX_SAMP["guidance_interval"][1], TEX_SAMP["rescale_t"]],
    dtype=np.float32))

# ---- reuse: res-512 SS coords + LR shape SLAT (denormalized, f32 run) ----
coords_res32_np = np.load(f"{SLAT_DIR}/coords.npy").astype(np.int32)   # [N,4] (b,x,y,z) at res-32
lr_slat_np = np.load(f"{SLAT_DIR}/f32_slat.npy").astype(np.float32)    # [N,32] denormalized
assert coords_res32_np.shape[0] == lr_slat_np.shape[0]
N_lr = coords_res32_np.shape[0]
print(f"N_lr (res-32 SS coords) = {N_lr}")
coords_res32 = torch.from_numpy(coords_res32_np).to(DEV, torch.int32)
lr_slat_feats = torch.from_numpy(lr_slat_np).to(DEV, torch.float32)

# ---- reuse: 4-view images/cameras from cond_slat dump ----
images_1024_np = np.load(f"{COND_DIR}/s1024_images.npy")              # [1,V,3,1024,1024]
camera_angle_x_np = np.load(f"{COND_DIR}/camera_angle_x.npy")         # [1,V]
distance_np = np.load(f"{COND_DIR}/distance.npy")                     # [1,V]
transform_matrix_np = np.load(f"{COND_DIR}/transform_matrix.npy")     # [1,V,4,4]
mesh_scale_np = np.load(f"{COND_DIR}/mesh_scale.npy")                 # [1]
V = images_1024_np.shape[1]
print(f"V={V} mesh_scale={mesh_scale_np[0]}")
images_1024 = torch.from_numpy(images_1024_np).to(DEV, torch.float32)
camera_angle_x = torch.from_numpy(camera_angle_x_np).to(DEV, torch.float32)
distance = torch.from_numpy(distance_np).to(DEV, torch.float32)
transform_matrix = torch.from_numpy(transform_matrix_np).to(DEV, torch.float32)
mesh_scale = torch.from_numpy(mesh_scale_np).to(DEV, torch.float32)

# ==========================================================================
# Stage 3a: Upsample LR -> HR (Pixal3DImageTo3DPipeline.run(), verbatim)
# ==========================================================================
print("\n=== Stage 3a: upsample LR shape SLAT -> HR coords ===")
with torch.device(DEV):
    shape_decoder = FlexiDualGridVaeDecoder(**shape_dec_cfg)
sdd = load_file(CKPT_SHAPE_DEC + ".safetensors", device=DEV)
d_missing, d_unexpected = shape_decoder.load_state_dict(sdd, strict=False)
del sdd; torch.cuda.empty_cache()
print(f"shape decoder loaded: missing={len(d_missing)} unexpected={len(d_unexpected)}")
shape_decoder.eval()

lr_slat_st = SparseTensor(feats=lr_slat_feats, coords=coords_res32)
with torch.no_grad():
    lr_decoded_coords = shape_decoder.upsample(lr_slat_st, upsample_times=4)   # torch.Tensor [M,4], 0..511 range
save_int("lr_decoded_coords", lr_decoded_coords)

quant_coords = torch.cat([
    lr_decoded_coords[:, :1],
    ((lr_decoded_coords[:, 1:].float() + 0.5) / LR_RESOLUTION * (GRID_R - 1)).round().int(),
], dim=1)
save_int("upsampled_coords", quant_coords)
hr_coords_unique = quant_coords.unique(dim=0).to(torch.int32)
save_int("hr_coords", hr_coords_unique)
Nh = hr_coords_unique.shape[0]
print(f"Nh (HR token count) = {Nh}  (from M={lr_decoded_coords.shape[0]} raw upsampled coords, "
      f"{quant_coords.shape[0]} quantized, deduplicated to {Nh})")

del shape_decoder, lr_decoded_coords, quant_coords, lr_slat_st
torch.cuda.empty_cache()

# ==========================================================================
# Stage 3b / 4 conditioning: DinoV3ProjMultiViewFeatureExtractor(shape_1024 / tex_1024)
# gathered at hr_coords_unique, exactly as get_proj_cond_shape does
# (grid_resolution_override=64 is a no-op here since both configs' native
# grid_resolution is already 64).
# ==========================================================================
def build_and_run_extractor(cfg, name):
    print(f"\nloading DinoV3ProjMultiViewFeatureExtractor ({name}): {cfg}")
    model = DinoV3ProjMultiViewFeatureExtractor(**cfg)
    model.eval()
    model.to(DEV)
    model._load_naf()
    with torch.no_grad():
        z_global, z_proj = model(
            images_1024, camera_angle_x=camera_angle_x, distance=distance,
            mesh_scale=mesh_scale, transform_matrix=transform_matrix,
        )
    assert cfg["grid_resolution"] == GRID_R
    z_proj_grid = z_proj.reshape(1, GRID_R, GRID_R, GRID_R, -1)
    b_idx = hr_coords_unique[:, 0].long(); x_idx = hr_coords_unique[:, 1].long()
    y_idx = hr_coords_unique[:, 2].long(); z_idx = hr_coords_unique[:, 3].long()
    z_proj_sparse = z_proj_grid[b_idx, x_idx, y_idx, z_idx]   # [Nh,2048]
    del model
    torch.cuda.empty_cache()
    return z_global, z_proj_sparse

print("\n=== Stage 3b conditioning: shape_1024 ===")
shape_z_global, shape_z_proj_sparse = build_and_run_extractor(SHAPE_1024_CFG, "shape_1024")
save("hr_cond_global", shape_z_global)          # [1,5,1024]
save("hr_cond_proj", shape_z_proj_sparse)        # [Nh,2048]

print("\n=== Stage 4 conditioning: tex_1024 ===")
tex_z_global, tex_z_proj_sparse = build_and_run_extractor(TEX_1024_CFG, "tex_1024")
save("tex_cond_global", tex_z_global)            # [1,5,1024]
save("tex_cond_proj", tex_z_proj_sparse)          # [Nh,2048]

cond_shape = {"global": shape_z_global, "proj": SparseTensor(feats=shape_z_proj_sparse, coords=hr_coords_unique)}
neg_cond_shape = {"global": torch.zeros_like(shape_z_global),
                  "proj": SparseTensor(feats=torch.zeros_like(shape_z_proj_sparse), coords=hr_coords_unique)}
cond_tex = {"global": tex_z_global, "proj": SparseTensor(feats=tex_z_proj_sparse, coords=hr_coords_unique)}
neg_cond_tex = {"global": torch.zeros_like(tex_z_global),
                "proj": SparseTensor(feats=torch.zeros_like(tex_z_proj_sparse), coords=hr_coords_unique)}

# ---- shared noise, saved once (seed 42 shape / seed 43 tex) ----
g_shape = torch.Generator(device=DEV).manual_seed(SHAPE_SEED)
shape_noise_feats = torch.randn(Nh, shape_hr_cfg["in_channels"], generator=g_shape, device=DEV, dtype=torch.float32)
save("shape_noise", shape_noise_feats)
g_tex = torch.Generator(device=DEV).manual_seed(TEX_SEED)
tex_in_ch = tex_hr_cfg["in_channels"]
tex_noise_ch = tex_in_ch - shape_hr_cfg["out_channels"]   # 64 - 32 = 32
tex_noise_feats = torch.randn(Nh, tex_noise_ch, generator=g_tex, device=DEV, dtype=torch.float32)
save("tex_noise", tex_noise_feats)

# ---- timestep schedule (deterministic, identical for both dtype runs, per stage) ----
def t_seq_of(samp):
    t_seq = np.linspace(1, 0, samp["steps"] + 1)
    t_seq = samp["rescale_t"] * t_seq / (1 + (samp["rescale_t"] - 1) * t_seq)
    return t_seq

sampler = FlowEulerGuidanceIntervalSampler(sigma_min=SIGMA_MIN)

# ==========================================================================
# Sampling
# ==========================================================================
def run_shape_stage(prefix, flow_dtype):
    print(f"\n=== shape HR stage {prefix} (flow_dtype={flow_dtype}) ===")
    if flow_dtype == torch.float32:
        wait_for_vram(tag=f"shape-{prefix}")
    with torch.device(DEV):
        model = ElasticSLatFlowModel(**shape_hr_cfg)
    if flow_dtype is not None:
        model.convert_to(flow_dtype)
        model.input_layer.to(flow_dtype)
        model.out_layer.to(flow_dtype)
    sd = load_file(CKPT_SHAPE_HR + ".safetensors", device=DEV)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd; torch.cuda.empty_cache()
    print(f"shape HR flow loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    noise = SparseTensor(feats=shape_noise_feats.clone(), coords=hr_coords_unique)
    with torch.no_grad():
        out = sampler.sample(model, noise, cond=cond_shape, neg_cond=neg_cond_shape, verbose=False, **SHAPE_SAMP)
    print("per-step std:", " ".join(f"{x.feats.std():.4f}" for x in out.pred_x_t))

    x_steps = [noise.feats]
    x_steps.extend(st.feats for st in out.pred_x_t)
    for k, xs in enumerate(x_steps):
        save(f"{prefix}_shape_x_step{k}", xs)
    x_final_feats = out.samples.feats
    save(f"{prefix}_shape_x_final", x_final_feats)
    assert torch.equal(x_final_feats, x_steps[-1])

    slat_feats = x_final_feats * shape_std_t + shape_mean_t
    save(f"{prefix}_shape_slat", slat_feats)

    del model
    torch.cuda.empty_cache()
    return {
        "x_steps": [a.detach().to(torch.float32).cpu().numpy() for a in x_steps],
        "x_final": x_final_feats.detach().to(torch.float32).cpu().numpy(),
        "slat": slat_feats.detach().to(torch.float32).cpu().numpy(),
        "slat_feats_t": slat_feats.detach(),   # kept on GPU for the tex stage
    }


def run_tex_stage(prefix, flow_dtype, shape_slat_feats_denorm_t):
    print(f"\n=== tex HR stage {prefix} (flow_dtype={flow_dtype}) ===")
    # sample_tex_slat: re-normalize the (denormalized) HR shape SLAT with the shape
    # normalization stats -- this IS the concat_cond, verbatim.
    concat_cond_feats = (shape_slat_feats_denorm_t - shape_mean_t) / shape_std_t
    save(f"{prefix}_tex_concat_cond", concat_cond_feats)
    concat_cond_st = SparseTensor(feats=concat_cond_feats.contiguous(), coords=hr_coords_unique)

    if flow_dtype == torch.float32:
        wait_for_vram(tag=f"tex-{prefix}")
    with torch.device(DEV):
        model = ElasticSLatFlowModel(**tex_hr_cfg)
    if flow_dtype is not None:
        model.convert_to(flow_dtype)
        model.input_layer.to(flow_dtype)
        model.out_layer.to(flow_dtype)
    sd = load_file(CKPT_TEX_HR + ".safetensors", device=DEV)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd; torch.cuda.empty_cache()
    print(f"tex HR flow loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    noise = SparseTensor(feats=tex_noise_feats.clone(), coords=hr_coords_unique)
    with torch.no_grad():
        out = sampler.sample(model, noise, cond=cond_tex, neg_cond=neg_cond_tex,
                              concat_cond=concat_cond_st, verbose=False, **TEX_SAMP)
    print("per-step std:", " ".join(f"{x.feats.std():.4f}" for x in out.pred_x_t))

    x_steps = [noise.feats]
    x_steps.extend(st.feats for st in out.pred_x_t)
    for k, xs in enumerate(x_steps):
        save(f"{prefix}_tex_x_step{k}", xs)
    x_final_feats = out.samples.feats
    save(f"{prefix}_tex_x_final", x_final_feats)
    assert torch.equal(x_final_feats, x_steps[-1])

    slat_feats = x_final_feats * tex_std_t + tex_mean_t
    save(f"{prefix}_tex_slat", slat_feats)

    del model
    torch.cuda.empty_cache()
    return {
        "x_steps": [a.detach().to(torch.float32).cpu().numpy() for a in x_steps],
        "x_final": x_final_feats.detach().to(torch.float32).cpu().numpy(),
        "slat": slat_feats.detach().to(torch.float32).cpu().numpy(),
        "slat_feats_t": slat_feats.detach(),
    }


# f32 cascade (shape then tex), then bf16 cascade -- sequential, freeing between.
res_shape_f32 = run_shape_stage("f32", torch.float32)
res_tex_f32 = run_tex_stage("f32", torch.float32, res_shape_f32["slat_feats_t"])
res_shape_bf16 = run_shape_stage("bf16", None)
res_tex_bf16 = run_tex_stage("bf16", None, res_shape_bf16["slat_feats_t"])

# ==========================================================================
# Stage 5: decode (f32 run only, if cheap)
# ==========================================================================
decode_f32 = None
try:
    print("\n=== decode (f32 only) ===")
    with torch.device(DEV):
        shape_decoder = FlexiDualGridVaeDecoder(**shape_dec_cfg)
    sdd = load_file(CKPT_SHAPE_DEC + ".safetensors", device=DEV)
    shape_decoder.load_state_dict(sdd, strict=False)
    del sdd; torch.cuda.empty_cache()
    shape_decoder.eval()
    shape_decoder.set_resolution(DECODE_RES)
    shape_slat_st = SparseTensor(feats=res_shape_f32["slat_feats_t"].contiguous(), coords=hr_coords_unique)
    with torch.no_grad():
        mesh_list, subs = shape_decoder(shape_slat_st, return_subs=True)
    mesh = mesh_list[0]
    verts = mesh.vertices.detach().to(torch.float32).cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32)
    save("f32_dec_vertices", verts)
    save_int("f32_dec_faces", faces)
    print(f"  shape decode: Nv={verts.shape[0]} Nf={faces.shape[0]}")
    del shape_decoder
    torch.cuda.empty_cache()

    with torch.device(DEV):
        tex_decoder = SparseUnetVaeDecoder(**tex_dec_cfg)
    tdd = load_file(CKPT_TEX_DEC + ".safetensors", device=DEV)
    tex_decoder.load_state_dict(tdd, strict=False)
    del tdd; torch.cuda.empty_cache()
    tex_decoder.eval()
    tex_slat_st = SparseTensor(feats=res_tex_f32["slat_feats_t"].contiguous(), coords=hr_coords_unique)
    with torch.no_grad():
        voxels = tex_decoder(tex_slat_st, guide_subs=subs)
        voxels = voxels.replace(voxels.feats * 0.5 + 0.5)
    save("f32_tex_attrs", voxels.feats)                 # [Nt,6]
    save_int("f32_tex_coords", voxels.coords[:, 1:])    # [Nt,3] (batch column dropped, as decode_latent does)
    print(f"  tex decode: Nt={voxels.feats.shape[0]}")
    decode_f32 = {"n_vertices": int(verts.shape[0]), "n_faces": int(faces.shape[0]),
                  "n_tex_voxels": int(voxels.feats.shape[0])}
    del tex_decoder
    torch.cuda.empty_cache()
except Exception as e:
    print(f"[decode] SKIPPED: {type(e).__name__}: {e}")
    decode_f32 = {"skipped": True, "reason": f"{type(e).__name__}: {e}"}
    torch.cuda.empty_cache()

peak_vram_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
wall_s = time.time() - t_wall0

# ==========================================================================
# Sanity
# ==========================================================================
print("\n=== sanity ===")

def per_step_maxd(res_a, res_b):
    return [float(np.abs(res_a["x_steps"][k] - res_b["x_steps"][k]).max()) for k in range(len(res_a["x_steps"]))]

def stats(a):
    return {"mean": float(a.mean()), "std": float(a.std()), "absmax": float(np.abs(a).max())}

shape_step_maxd = per_step_maxd(res_shape_f32, res_shape_bf16)
tex_step_maxd = per_step_maxd(res_tex_f32, res_tex_bf16)
for k in range(len(shape_step_maxd)):
    tag = " <-- reported" if k in (1, 6, 12) else ""
    print(f"  shape x_step{k:2d} max|d|(f32 vs bf16) = {shape_step_maxd[k]:.4e}{tag}")
for k in range(len(tex_step_maxd)):
    tag = " <-- reported" if k in (1, 6, 12) else ""
    print(f"  tex   x_step{k:2d} max|d|(f32 vs bf16) = {tex_step_maxd[k]:.4e}{tag}")

shape_final_maxd = float(np.abs(res_shape_f32["x_final"] - res_shape_bf16["x_final"]).max())
shape_final_maxabs = float(np.abs(res_shape_f32["x_final"]).max())
shape_rel = shape_final_maxd / shape_final_maxabs if shape_final_maxabs > 0 else float("nan")
tex_final_maxd = float(np.abs(res_tex_f32["x_final"] - res_tex_bf16["x_final"]).max())
tex_final_maxabs = float(np.abs(res_tex_f32["x_final"]).max())
tex_rel = tex_final_maxd / tex_final_maxabs if tex_final_maxabs > 0 else float("nan")
print(f"shape x_final: max|d|={shape_final_maxd:.4e} max|f32|={shape_final_maxabs:.4e} rel={shape_rel:.4e}")
print(f"tex   x_final: max|d|={tex_final_maxd:.4e} max|f32|={tex_final_maxabs:.4e} rel={tex_rel:.4e}")
print(f"Nh (HR token count) = {Nh}")
print(f"wall time = {wall_s:.1f}s, peak VRAM (this process) = {peak_vram_mib:.0f} MiB")

for lbl, res in (("shape_f32", res_shape_f32), ("shape_bf16", res_shape_bf16),
                 ("tex_f32", res_tex_f32), ("tex_bf16", res_tex_bf16)):
    for k, a in enumerate(res["x_steps"]):
        assert not np.isnan(a).any() and not np.isinf(a).any(), f"{lbl} x_step{k} has NaN/Inf"
    assert not np.isnan(res["slat"]).any() and not np.isinf(res["slat"]).any(), f"{lbl} slat has NaN/Inf"

meta = {
    "source_commit": "f7cf384",
    "script": "tools/ref_pixal3d_hr_sample.py",
    "stage": "1024_cascade HR shape SLAT + texture SLAT (Pixal3DImageTo3DPipeline.run(), "
             "pipeline_type='1024_cascade': Stage 3a upsample, Stage 3b shape HR, Stage 4 texture)",
    "seed_shape": SHAPE_SEED,
    "seed_tex": TEX_SEED,
    "seed_note": (
        "run() calls torch.manual_seed(seed) once at the very start and consumes the "
        "global RNG sequentially (SS noise, LR shape noise, HR shape noise, tex noise); "
        "this fixture reuses SS/LR-shape artifacts from earlier fixture scripts instead "
        "of re-running those stages, so it cannot reproduce that exact global-RNG offset. "
        "Instead it uses independent torch.Generator(device=DEV) instances seeded 42 (HR "
        "shape noise) and 43 (tex noise), matching ref_pixal3d_slat_sample.py's convention "
        "of one explicit generator per stage. Noise is generated once and shared between "
        "the f32 and bf16 cascades (only flow-torso dtype differs), exactly like "
        "ref_pixal3d_slat_sample.py."
    ),
    "lr_to_hr_procedure": (
        "hr_coords = shape_slat_decoder.upsample(lr_slat_denorm, upsample_times=4): runs "
        "FlexiDualGridVaeDecoder.from_latent() then its SparseResBlock3d stacks for "
        "resolution stages i=0..3 (real ConvNeXt3d + subdivision-prediction compute, each "
        "stage's last block upsample=True -> SparseUpsample(2)), returning h.coords right "
        "before stage i=4 -- i.e. 4 stages x 2x = 16x spatial upsample from the 32^3 LR "
        "latent grid, landing on the 512^3 voxel grid (lr_resolution=512 hardcoded in "
        "run()). Then quantized to the HR flow's grid_res=64 sparse-token grid: "
        "quant_coords = cat([hr_coords[:,:1], round((hr_coords[:,1:]+0.5)/512*(64-1))]), "
        "hr_coords_unique = quant_coords.unique(dim=0). For pipeline_type='1024_cascade' "
        "the run()-inlined token-budget while-loop always breaks on its first iteration "
        "(hr_resolution starts at 1024, and its break condition is `... or "
        "actual_hr_resolution == 1024`), so max_num_tokens never reduces resolution here "
        "-- only for '1536_cascade'. Divergence: the base-class helper "
        "Trellis2ImageTo3DPipeline/Pixal3DImageTo3DPipeline.sample_shape_slat_cascade "
        "(used by trellis2_image_to_3d.py's own run(), NOT by Pixal3DImageTo3DPipeline.run()) "
        "uses a different formula for the same step -- `((hr_coords[:,1:]+0.5)/lr_resolution*"
        "(hr_resolution//16)).int()` (no .round(), grid_res not grid_res-1) -- this script "
        "reproduces run()'s own inlined formula, since that is what pipeline_type="
        "'1024_cascade' actually executes."
    ),
    "concat_cond_finding": (
        "sample_tex_slat re-normalizes the just-denormalized HR shape SLAT with the SHAPE "
        "(not tex) slat_normalization stats -- (shape_slat_denorm - shape_mean)/shape_std -- "
        "and passes that 32-channel tensor as concat_cond= through the sampler straight "
        "into ElasticSLatFlowModel.forward(), which does "
        "x = sp.sparse_cat([x, concat_cond], dim=-1) (structured_latent_flow.py): tex noise "
        "channels first (32), normalized shape SLAT appended (32) = tex flow's "
        "in_channels=64 (out_channels stays 32). Hence the tex noise tensor itself has only "
        "32 channels (in_channels - shape_slat.feats.shape[1]). Saved as "
        "{prefix}_tex_concat_cond.npy, per dtype run (f32 and bf16 cascades produce "
        "slightly different HR shape SLATs, hence slightly different concat_cond, even "
        "though hr_coords.npy/topology is shared)."
    ),
    "dtype": {
        "f32": "both flow torsos (shape HR + tex HR) float32; shape/tex decoders (only run "
               "for this dtype) left at their checkpoints' own use_fp16=True production "
               "default, NOT upcast",
        "bf16": "both flow torsos bfloat16 (production default, *_1024_bf16_mv ckpt jsons "
                "declare dtype='bfloat16', applied automatically); no decode step",
        "common": "x_t (sampler state) float32 throughout both runs/both stages; "
                  "input_layer/out_layer/t_embedder/adaLN_modulation stay float32 in both "
                  "(model forward() manual_casts back to x.dtype before final LayerNorm + "
                  "out_layer)",
    },
    "checkpoint_shape_flow_1024": os.path.basename(CKPT_SHAPE_HR),
    "checkpoint_tex_flow_1024": os.path.basename(CKPT_TEX_HR),
    "checkpoint_shape_decoder": os.path.basename(CKPT_SHAPE_DEC),
    "checkpoint_tex_decoder": os.path.basename(CKPT_TEX_DEC),
    "shape_flow_config": shape_hr_cfg,
    "tex_flow_config": tex_hr_cfg,
    "shape_decoder_config": shape_dec_cfg,
    "tex_decoder_config": tex_dec_cfg,
    "image_cond_configs": {"shape_1024": SHAPE_1024_CFG, "tex_1024": TEX_1024_CFG},
    "lr_slat_source": f"{SLAT_DIR}/coords.npy, {SLAT_DIR}/f32_slat.npy (reused res-512 LR shape SLAT, denormalized)",
    "views_source": f"{COND_DIR}/s1024_images.npy, camera_angle_x.npy, distance.npy, transform_matrix.npy, mesh_scale.npy (reused 4-view assets)",
    "sampler": "FlowEulerGuidanceIntervalSampler",
    "sampler_sigma_min": SIGMA_MIN,
    "shape_sampler_params": SHAPE_SAMP,
    "tex_sampler_params": TEX_SAMP,
    "max_num_tokens": MAX_NUM_TOKENS,
    "decoder_resolution": DECODE_RES,
    "shapes": shapes,
    "sanity": {
        "Nh": Nh,
        "shape_per_step_max_abs_diff_f32_vs_bf16": shape_step_maxd,
        "tex_per_step_max_abs_diff_f32_vs_bf16": tex_step_maxd,
        "shape_step1_max_abs_diff": shape_step_maxd[1],
        "shape_step6_max_abs_diff": shape_step_maxd[6],
        "shape_step12_max_abs_diff": shape_step_maxd[12],
        "tex_step1_max_abs_diff": tex_step_maxd[1],
        "tex_step6_max_abs_diff": tex_step_maxd[6],
        "tex_step12_max_abs_diff": tex_step_maxd[12],
        "shape_x_final_stats_f32": stats(res_shape_f32["x_final"]),
        "shape_x_final_stats_bf16": stats(res_shape_bf16["x_final"]),
        "shape_x_final_max_abs_diff_f32_vs_bf16": shape_final_maxd,
        "shape_x_final_rel_max_abs_diff": shape_rel,
        "tex_x_final_stats_f32": stats(res_tex_f32["x_final"]),
        "tex_x_final_stats_bf16": stats(res_tex_bf16["x_final"]),
        "tex_x_final_max_abs_diff_f32_vs_bf16": tex_final_maxd,
        "tex_x_final_rel_max_abs_diff": tex_rel,
        "decode_f32": decode_f32,
        "wall_time_s": wall_s,
        "peak_vram_mib_this_process": peak_vram_mib,
    },
    "tolerance_note": (
        "End-to-end HR shape-SLAT + texture-SLAT (1024_cascade) sampling fixture: real "
        "res-512 LR SLAT and 4-view projection conditioning reused where upstream/stage-"
        "independent, LR->HR upsample/quantize and shape_1024/tex_1024 proj-conditioning "
        "recomputed exactly as Pixal3DImageTo3DPipeline.run() + Pixal3DMVImageTo3DPipeline "
        "do it, FlowEulerGuidanceIntervalSampler with the exact pipeline_mv.json "
        "shape_slat_sampler/tex_slat_sampler params. The f32-vs-bf16 per-step / x_final "
        "deltas calibrate how much the production flow-torso dtype alone moves the HR "
        "sampling result for each stage -- per docs/spec/30-pixal3d-cond.md section 5/6, "
        "end-to-end sampling parity (not block-level tensor parity) is the production "
        "gate; C++ HR shape/tex sample tests should compare against the bf16 runs in "
        "bf16/f16 compute, with f32 as the calibration ceiling."
    ),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
