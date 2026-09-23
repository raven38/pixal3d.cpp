#!/usr/bin/env python3
"""Reference oracle for the Pixal3D sparse-structure resolution (ss_res 32 | 64) cascade (#78/#79).

Pixal3DImageTo3DPipeline.run() hardcodes `ss_res = 32` and quantizes the shape decoder's
upsampled coords with

    round((hr_coords + 0.5) / lr_resolution * (grid_res - 1)),  lr_resolution = 512

where 512 is really the upsample span ss_res * 16 (4 SparseUpsample(2) stages). There is no
upstream ss_res=64 path, so this script is the oracle: it runs the reference MODULES the way
run() does with ss_res as a parameter, and records the corrected semantics that
visualbruno/ComfyUI-Trellis2#193 documents (span = ss_res*16), keeping Pixal3D's own
round + (grid-1) convention:

  1. SS coords: sample_sparse_structure()'s decode `decoded = decoder(z_s) > 0`, max_pool3d by
     64 // ss_res when ss_res < 64, argwhere. The SS decoder logits are reused from
     tools/ref_pixal3d_ss_sample.py's f32_occ_logits.npy (the SS stage itself is independent of
     ss_res -- #193 measured it bit-identical), so 32 and 64 share one SS sample.
  2. LR projected conditioning: get_proj_cond_shape(image_cond_model_shape_512, ...) with
     grid_resolution_override = ss_res (the SS coord domain -- NOT max(coord)+1 as the GGUF
     fork does), gathered at the SS coords.
  3. LR shape flow (shape_512 MV DiT) f32 + bf16 runs, seed 42, per-step latents -- the same
     file layout as tools/ref_pixal3d_slat_sample.py, so trellis-test-pixal3d-slat-sample
     --stage shape512 consumes ss<R>/ directly.
  4. upsample(lr_slat, upsample_times=4) raw coords (span ss_res*16) and quantization at
     1024 (grid 64) with the span-aware formula, plus the 1536 cascade backoff with a
     max_num_tokens chosen so it backs off at least once. The literal-512 (bug) result is
     recorded in meta.json for the ss64 run.
  5. (HR=1) HR shape flow at 1024 (shape_1024 cond gathered at the HR coords, seed 42) f32 +
     bf16, in tools/ref_pixal3d_hr_sample.py's shape_hr file layout.

Output (OUT):
  occ_logits.npy [64,64,64]              SS decoder logits (shared)
  ss32/ ss64/                            per-ss_res fixtures (see above)
  meta.json                              counts, coord ranges, spans, peak CUDA memory per stage

    ATTN_BACKEND=sdpa HF_HOME=/mnt/d/pixal3d/hf_home HF_HUB_OFFLINE=1 \
    PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D OUT=/mnt/hdd1/pixal3d/ref/pixal3d/ss_res HR=1 \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_ss_res.py
"""
import os, sys, json, time
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
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjMultiViewFeatureExtractor,
)

WEIGHTS_ROOT = os.environ.get(
    "PIXAL3D_WEIGHTS_ROOT",
    "/mnt/d/pixal3d/hf_home/hub/models--TencentARC--Pixal3D/snapshots/"
    "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
)
CKPT_FLOW_512 = f"{WEIGHTS_ROOT}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv"
CKPT_FLOW_1024 = f"{WEIGHTS_ROOT}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv"
CKPT_SHAPE_DEC = f"{WEIGHTS_ROOT}/ckpts/shape_dec_next_dc_f16c32_fp16"
SS_DIR = os.environ.get("SS_DIR", "/mnt/hdd1/pixal3d/ref/pixal3d/ss_sample")
COND_DIR = os.environ.get("COND_DIR", "/mnt/hdd1/pixal3d/ref/pixal3d/cond_slat")
OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/ss_res")
SS_RES_LIST = [int(x) for x in os.environ.get("SS_RES", "32,64").split(",")]
RUN_HR = os.environ.get("HR", "0") == "1"
DEV = "cuda"
SEED = 42
HR_RESOLUTION = 1024
SAMP = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.5,
            guidance_interval=(0.6, 1.0), rescale_t=3.0)          # pipeline_mv.json shape sampler
SIGMA_MIN = 1e-5
SHAPE_512_CFG = {"model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m", "image_size": 512,
                 "grid_resolution": 32, "use_naf_upsample": True, "naf_target_size": 512,
                 "multiview_fusion": "average"}                  # inference_mv.IMAGE_COND_CONFIGS
SHAPE_1024_CFG = {"model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m", "image_size": 1024,
                  "grid_resolution": 64, "use_naf_upsample": True, "naf_target_size": 512,
                  "multiview_fusion": "average"}
os.makedirs(OUT, exist_ok=True)


def save(d, name, a_or_t):
    a = a_or_t.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(a_or_t) else np.asarray(a_or_t, np.float32)
    a = np.ascontiguousarray(a)
    assert np.isfinite(a).all(), f"{name} has NaN/Inf"
    np.save(f"{d}/{name}.npy", a)
    return a


def save_int(d, name, t):
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    a = np.ascontiguousarray(a.astype(np.int32))
    np.save(f"{d}/{name}.npy", a)
    return a


def peak_mib():
    return round(torch.cuda.max_memory_allocated() / 2**20, 1), round(torch.cuda.max_memory_reserved() / 2**20, 1)


def quantize(up, span, grid):
    q = torch.cat([up[:, :1], ((up[:, 1:].float() + 0.5) / span * (grid - 1)).round().int()], dim=1)
    return q.unique(dim=0).to(torch.int32)


def coord_range(c):
    return [int(c[:, 1:].min()), int(c[:, 1:].max())] if c.shape[0] else [None, None]


pipeline_mv = json.load(open(f"{WEIGHTS_ROOT}/pipeline_mv.json"))["args"]
norm = pipeline_mv["shape_slat_normalization"]
mean_np = np.array(norm["mean"], np.float32); std_np = np.array(norm["std"], np.float32)
mean_t = torch.from_numpy(mean_np)[None].to(DEV); std_t = torch.from_numpy(std_np)[None].to(DEV)
flow512_cfg = json.load(open(CKPT_FLOW_512 + ".json"))["args"]
flow1024_cfg = json.load(open(CKPT_FLOW_1024 + ".json"))["args"]
dec_cfg = json.load(open(CKPT_SHAPE_DEC + ".json"))["args"]
sampler = FlowEulerGuidanceIntervalSampler(sigma_min=SIGMA_MIN)

cams = {k: torch.from_numpy(np.load(f"{COND_DIR}/{k}.npy")).to(DEV, torch.float32)
        for k in ("camera_angle_x", "distance", "transform_matrix", "mesh_scale")}

occ = np.load(f"{SS_DIR}/f32_occ_logits.npy").astype(np.float32).reshape(64, 64, 64)
np.save(f"{OUT}/occ_logits.npy", occ)
meta = {"seed": SEED, "hr_resolution": HR_RESOLUTION, "sampler": SAMP,
        "ss_logits": {"source": f"{SS_DIR}/f32_occ_logits.npy", "mean": float(occ.mean()),
                      "std": float(occ.std()), "active@64": int((occ > 0).sum())},
        "pixal3d_repo": PIXAL3D_REPO, "weights_root": WEIGHTS_ROOT, "runs": {}}
try:
    import subprocess
    meta["pixal3d_commit"] = subprocess.run(["git", "-C", PIXAL3D_REPO, "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
except Exception:
    pass


def extractor(cfg, grid, images, coords):
    """get_proj_cond_shape with grid_resolution_override=grid, gathered at coords."""
    model = DinoV3ProjMultiViewFeatureExtractor(**cfg).eval().to(DEV)
    model._load_naf()
    if grid != model.grid_resolution:
        model.grid_resolution = grid
        model.proj_grid = model.proj_grid.__class__(
            grid_resolution=grid, image_resolution=model.proj_grid.image_resolution).to(DEV)
    with torch.no_grad():
        zg, zp = model(images, camera_angle_x=cams["camera_angle_x"], distance=cams["distance"],
                       mesh_scale=cams["mesh_scale"], transform_matrix=cams["transform_matrix"])
    dense_bytes = zp.numel() * zp.element_size()
    zp = zp.reshape(1, grid, grid, grid, -1)[coords[:, 0].long(), coords[:, 1].long(),
                                             coords[:, 2].long(), coords[:, 3].long()]
    del model; torch.cuda.empty_cache()
    return zg, zp.contiguous(), dense_bytes


def run_flow(d, prefix, cfg, ckpt, flow_dtype, noise_feats, coords, cond, neg, stem=""):
    with torch.device(DEV):
        model = ElasticSLatFlowModel(**cfg)
    if flow_dtype is not None:
        model.convert_to(flow_dtype); model.input_layer.to(flow_dtype); model.out_layer.to(flow_dtype)
    sd = load_file(ckpt + ".safetensors", device=DEV)
    model.load_state_dict(sd, strict=False); del sd
    model.eval()
    noise = SparseTensor(feats=noise_feats.clone(), coords=coords)
    with torch.no_grad():
        out = sampler.sample(model, noise, cond=cond, neg_cond=neg, verbose=False, **SAMP)
    xs = [noise.feats] + [s.feats for s in out.pred_x_t]
    for k, x in enumerate(xs):
        save(d, f"{prefix}_{stem}x_step{k}", x)
    save(d, f"{prefix}_{stem}x_final", out.samples.feats)
    slat = out.samples.feats * std_t + mean_t
    save(d, f"{prefix}_{stem}slat", slat)
    del model; torch.cuda.empty_cache()
    return slat


images512 = torch.from_numpy(np.load(f"{COND_DIR}/s512_images.npy")).to(DEV, torch.float32)
for R in SS_RES_LIST:
    assert R in (32, 64), R
    d = f"{OUT}/ss{R}"; os.makedirs(d, exist_ok=True)
    run = {"ss_res": R, "source_span": R * 16}
    print(f"\n===== ss_res={R} =====")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    decoded = torch.from_numpy(occ).to(DEV)[None, None] > 0
    if R != decoded.shape[2]:
        ratio = decoded.shape[2] // R
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    save_int(d, "ss_coords", coords); save_int(d, "coords", coords)
    N = coords.shape[0]
    run["ss"] = {"active": N, "coord_range": coord_range(coords)}
    print(f"SS coords: {N} in {coord_range(coords)}")

    zg, zp, dense_bytes = extractor(SHAPE_512_CFG, R, images512, coords)
    save(d, "cond_global", zg); save(d, "cond_proj", zp)
    run["lr_cond"] = {"grid": R, "dense_proj_bytes_reference": int(dense_bytes),
                      "sparse_proj_bytes": int(zp.numel() * 4), "peak_alloc_reserved_mib": peak_mib()}
    cond = {"global": zg, "proj": SparseTensor(feats=zp, coords=coords)}
    neg = {"global": torch.zeros_like(zg), "proj": SparseTensor(feats=torch.zeros_like(zp), coords=coords)}
    save(d, "norm_mean", mean_np); save(d, "norm_std", std_np)
    g = torch.Generator(device=DEV).manual_seed(SEED)
    noise_feats = torch.randn(N, flow512_cfg["in_channels"], generator=g, device=DEV, dtype=torch.float32)
    save(d, "noise", noise_feats)
    torch.cuda.reset_peak_memory_stats()
    lr_slat = run_flow(d, "f32", flow512_cfg, CKPT_FLOW_512, torch.float32, noise_feats, coords, cond, neg)
    run_flow(d, "bf16", flow512_cfg, CKPT_FLOW_512, None, noise_feats, coords, cond, neg)
    save(d, "lr_slat", lr_slat)
    run["lr_flow_peak_alloc_reserved_mib"] = peak_mib()
    del cond, neg, zp; torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    with torch.device(DEV):
        dec = FlexiDualGridVaeDecoder(**dec_cfg)
    sdd = load_file(CKPT_SHAPE_DEC + ".safetensors", device=DEV); dec.load_state_dict(sdd, strict=False); del sdd
    dec.eval()
    with torch.no_grad():
        up = dec.upsample(SparseTensor(feats=lr_slat.contiguous(), coords=coords), upsample_times=4)
    del dec; torch.cuda.empty_cache()
    save_int(d, "upsampled_coords", up)
    span = R * 16
    run["upsample"] = {"raw": int(up.shape[0]), "coord_range": coord_range(up), "span": span,
                       "peak_alloc_reserved_mib": peak_mib()}
    hr = quantize(up, span, HR_RESOLUTION // 16)
    save_int(d, "hr1024_coords", hr)
    bug = quantize(up, 512, HR_RESOLUTION // 16)       # the literal-512 quantizer (#193)
    run["hr1024"] = {"grid": HR_RESOLUTION // 16, "tokens": int(hr.shape[0]), "coord_range": coord_range(hr),
                     "literal512_tokens": int(bug.shape[0]), "literal512_coord_range": coord_range(bug)}
    print(f"HR@1024: {hr.shape[0]} tokens in {coord_range(hr)}  (literal /512: {bug.shape[0]} in {coord_range(bug)})")
    # 1536 cascade: max_num_tokens = the 1536 count forces exactly one -128 backoff check.
    max_tok = int(quantize(up, span, 1536 // 16).shape[0])
    res = 1536
    while True:
        q = quantize(up, span, res // 16)
        if q.shape[0] < max_tok or res == 1024:
            break
        res -= 128
    save_int(d, "hr1536_coords", q); save_int(d, "hr1536_meta", np.array([max_tok, res]))
    run["hr1536"] = {"max_num_tokens": max_tok, "selected_res": res, "tokens": int(q.shape[0])}
    print(f"1536 cascade (max_tokens={max_tok}): res {res}, {q.shape[0]} tokens")

    if RUN_HR:
        images1024 = torch.from_numpy(np.load(f"{COND_DIR}/s1024_images.npy")).to(DEV, torch.float32)
        torch.cuda.reset_peak_memory_stats()
        hg, hp, hdense = extractor(SHAPE_1024_CFG, HR_RESOLUTION // 16, images1024, hr)
        del images1024
        save_int(d, "hr_coords", hr); save(d, "hr_cond_global", hg); save(d, "hr_cond_proj", hp)
        save(d, "shape_norm_mean", mean_np); save(d, "shape_norm_std", std_np)
        save(d, "shape_sampler_params", np.array([SAMP["steps"], SAMP["guidance_strength"], SAMP["guidance_rescale"],
                                                  *SAMP["guidance_interval"], SAMP["rescale_t"]], np.float32))
        gh = torch.Generator(device=DEV).manual_seed(SEED)
        hn = torch.randn(hr.shape[0], flow1024_cfg["in_channels"], generator=gh, device=DEV, dtype=torch.float32)
        save(d, "shape_noise", hn)
        hc = {"global": hg, "proj": SparseTensor(feats=hp, coords=hr)}
        hneg = {"global": torch.zeros_like(hg), "proj": SparseTensor(feats=torch.zeros_like(hp), coords=hr)}
        run_flow(d, "f32", flow1024_cfg, CKPT_FLOW_1024, torch.float32, hn, hr, hc, hneg, stem="shape_")
        run_flow(d, "bf16", flow1024_cfg, CKPT_FLOW_1024, None, hn, hr, hc, hneg, stem="shape_")
        run["hr_flow"] = {"tokens": int(hr.shape[0]), "dense_proj_bytes_reference": int(hdense),
                          "peak_alloc_reserved_mib": peak_mib()}
        del hc, hneg, hp; torch.cuda.empty_cache()
    run["wall_s"] = round(time.time() - t0, 1)
    meta["runs"][f"ss{R}"] = run
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)

print(json.dumps(meta["runs"], indent=2))
