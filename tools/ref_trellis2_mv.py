#!/usr/bin/env python3
"""Pinned TRELLIS.2 multi-image reference/oracle for issue #59.

Two modes:

  python3 tools/ref_trellis2_mv.py --selftest
      Weightless, stdlib-only synthetic oracle. This is CI-safe and freezes the
      stochastic / multidiffusion sampler semantics used by #61.

  python3 tools/ref_trellis2_mv.py --repo /path/to/TRELLIS.2 \
      --images view0.png view1.png ... --mode stochastic \
      --pipeline-type 1024_cascade --out /tmp/trellis2-mv-ref

      Full reference dump. This REQUIRES the TRELLIS.2 checkout to be exactly
      PR #104 head REF_SHA below. It writes local .npy tensors + manifest.json;
      the large tensors are intentionally not committed to pixal3d.cpp.

The primary algorithm target is microsoft/TRELLIS.2 PR #104 at REF_SHA.
Original TRELLIS multi-image support at ORIGINAL_TRELLIS_SHA is a secondary
semantic reference only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence

REF_REPO = "dongjinlee123/TRELLIS.2"
REF_SHA = "fe7b90c376cb2ed2fb48582d1b376b56e1e16c44"
REF_BASE_SHA = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
ORIGINAL_TRELLIS_SHA = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
FORMAT_VERSION = 1
SIGMA_MIN = 1e-5

_THIS_REPO = Path(__file__).resolve().parents[1]
_SYNTHETIC_FIXTURE = _THIS_REPO / "tests/fixtures/trellis2_mv/synthetic_oracle.json"


# ---------------------------------------------------------------------------
# Weightless synthetic oracle
# ---------------------------------------------------------------------------

def _close_seq(a: Sequence[float], b: Sequence[float], atol: float = 1e-11) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


def _rescaled_t_seq(steps: int, rescale_t: float) -> list[float]:
    raw = [1.0 - i / steps for i in range(steps + 1)]
    return [rescale_t * t / (1.0 + (rescale_t - 1.0) * t) for t in raw]


def _sample_std(x: Sequence[float]) -> float:
    # torch.std default correction=1 for this 1-sample synthetic vector.
    if len(x) < 2:
        raise ValueError("synthetic oracle needs at least two features")
    mean = sum(x) / len(x)
    return math.sqrt(sum((v - mean) ** 2 for v in x) / (len(x) - 1))


def _fake_model(x: Sequence[float], t: float, cond: float) -> list[float]:
    # Deterministic fake Flow model. The t term uses 1000*t because the real
    # FlowEulerSampler passes that scaled timestep to the network.
    return [
        0.15 * x[i]
        + (i + 1) * 0.2 * cond
        + 0.0003 * (1000.0 * t) * (1.0 if i % 2 == 0 else -1.0)
        for i in range(len(x))
    ]


def _pred_to_xstart(x: Sequence[float], t: float, pred: Sequence[float]) -> list[float]:
    scale = SIGMA_MIN + (1.0 - SIGMA_MIN) * t
    return [(1.0 - SIGMA_MIN) * a - scale * b for a, b in zip(x, pred)]


def _xstart_to_pred(x: Sequence[float], t: float, x0: Sequence[float]) -> list[float]:
    scale = SIGMA_MIN + (1.0 - SIGMA_MIN) * t
    return [((1.0 - SIGMA_MIN) * a - b) / scale for a, b in zip(x, x0)]


def _cfg_reference(
    x: Sequence[float],
    t: float,
    pred_pos: Sequence[float],
    pred_neg: Sequence[float],
    guidance_strength: float,
    guidance_rescale: float,
    *,
    shortcut: bool,
) -> list[float]:
    # shortcut=True matches ClassifierFreeGuidanceSamplerMixin, used by the
    # stochastic wrapper via the original sampler. shortcut=False matches PR
    # #104 multidiffusion, which computes the direct formula even for gs=0/1.
    if shortcut and guidance_strength == 1.0:
        return list(pred_pos)
    if shortcut and guidance_strength == 0.0:
        return list(pred_neg)

    pred = [
        guidance_strength * p + (1.0 - guidance_strength) * n
        for p, n in zip(pred_pos, pred_neg)
    ]
    if guidance_rescale <= 0.0:
        return pred

    x0_pos = _pred_to_xstart(x, t, pred_pos)
    x0_cfg = _pred_to_xstart(x, t, pred)
    std_pos = _sample_std(x0_pos)
    std_cfg = _sample_std(x0_cfg)
    if std_cfg == 0.0:
        raise AssertionError("synthetic fixture accidentally produced zero CFG std")
    ratio = std_pos / std_cfg
    x0_rescaled = [v * ratio for v in x0_cfg]
    x0 = [
        guidance_rescale * r + (1.0 - guidance_rescale) * c
        for r, c in zip(x0_rescaled, x0_cfg)
    ]
    return _xstart_to_pred(x, t, x0)


def _synthetic_run(mode: str, p: dict[str, Any]) -> dict[str, Any]:
    x = [float(v) for v in p["noise"]]
    conds = [float(v) for v in p["conditions"]]
    neg_cond = float(p["negative_condition"])
    steps = int(p["steps"])
    ts = _rescaled_t_seq(steps, float(p["rescale_t"]))
    lo, hi = map(float, p["guidance_interval"])
    gs = float(p["guidance_strength"])
    gr = float(p["guidance_rescale"])

    view_schedule: list[int] = []
    pos_calls: list[int] = []
    neg_calls: list[int] = []
    trace: list[dict[str, Any]] = []

    for k, (t, t_prev) in enumerate(zip(ts, ts[1:])):
        active = lo <= t <= hi
        if mode == "stochastic":
            view = k % len(conds)
            view_schedule.append(view)
            pred_pos = _fake_model(x, t, conds[view])
            pos_calls.append(1)
            if active:
                pred_neg = _fake_model(x, t, neg_cond)
                pred = _cfg_reference(x, t, pred_pos, pred_neg, gs, gr, shortcut=True)
                neg_calls.append(0 if gs == 1.0 else 1)
            else:
                pred = pred_pos
                neg_calls.append(0)
        elif mode == "multidiffusion":
            per_view = [_fake_model(x, t, c) for c in conds]
            pred_pos = [sum(v) / len(per_view) for v in zip(*per_view)]
            pos_calls.append(len(conds))
            if active:
                # PR #104 explicitly evaluates neg once inside the interval,
                # even when guidance_strength happens to be 1.
                pred_neg = _fake_model(x, t, neg_cond)
                pred = _cfg_reference(x, t, pred_pos, pred_neg, gs, gr, shortcut=False)
                neg_calls.append(1)
            else:
                pred = pred_pos
                neg_calls.append(0)
        else:
            raise ValueError(mode)

        x = [a - (t - t_prev) * b for a, b in zip(x, pred)]
        trace.append({
            "step": k,
            "t": t,
            "t_prev": t_prev,
            "guidance_active": active,
            "pred": pred,
            "state": list(x),
        })

    return {
        "t_seq": ts,
        "final": x,
        "view_schedule": view_schedule,
        "positive_calls_per_step": pos_calls,
        "negative_calls_per_step": neg_calls,
        "trace": trace,
    }


def run_selftest() -> int:
    expected = json.loads(_SYNTHETIC_FIXTURE.read_text())
    if expected["upstream_pr104_sha"] != REF_SHA:
        raise AssertionError("fixture/reference SHA mismatch")
    p = expected["parameters"]

    st = _synthetic_run("stochastic", p)
    md = _synthetic_run("multidiffusion", p)

    assert _close_seq(st["t_seq"], expected["t_seq"])
    assert st["view_schedule"] == expected["stochastic"]["view_schedule"]
    assert st["negative_calls_per_step"] == expected["stochastic"]["negative_calls_per_step"]
    assert _close_seq(st["final"], expected["stochastic"]["final"])

    assert md["positive_calls_per_step"] == expected["multidiffusion"]["positive_calls_per_step"]
    assert md["negative_calls_per_step"] == expected["multidiffusion"]["negative_calls_per_step"]
    assert _close_seq(md["final"], expected["multidiffusion"]["final"])

    # V=1 must collapse to ordinary single-condition semantics for both modes.
    one = dict(p)
    one["conditions"] = [0.375]
    a = _synthetic_run("stochastic", one)
    b = _synthetic_run("multidiffusion", one)
    assert _close_seq(a["final"], b["final"], atol=1e-10)

    # Inclusive guidance interval boundary: t=0.9 is guided in the frozen case,
    # while t=1.0 is outside.
    assert [r["guidance_active"] for r in st["trace"]] == [False, True, True, True]

    # steps < views: stochastic still advances once per sampler step.
    short = dict(p)
    short["conditions"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    short["steps"] = 2
    assert _synthetic_run("stochastic", short)["view_schedule"] == [0, 1]

    # gs=1: stochastic's original CFG mixin skips the negative model call;
    # PR #104 multidiffusion still evaluates it inside the guidance interval.
    gs1 = dict(p)
    gs1["guidance_strength"] = 1.0
    ss = _synthetic_run("stochastic", gs1)
    mm = _synthetic_run("multidiffusion", gs1)
    assert ss["negative_calls_per_step"] == [0, 0, 0, 0]
    assert mm["negative_calls_per_step"] == [0, 1, 1, 1]

    print("TRELLIS2_MV_ORACLE_OK")
    print("  PR104:", REF_SHA)
    print("  stochastic final:", " ".join(f"{v:.12g}" for v in st["final"]))
    print("  multidiffusion final:", " ".join(f"{v:.12g}" for v in md["final"]))
    return 0


# ---------------------------------------------------------------------------
# Full pinned Python reference dump
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _require_pinned_checkout(repo: Path, allow_dirty: bool) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    if head != REF_SHA:
        raise SystemExit(
            f"TRELLIS.2 reference checkout must be PR #104 head {REF_SHA}; got {head}. "
            f"Fetch {REF_REPO} and checkout that exact commit."
        )
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty and not allow_dirty:
        raise SystemExit("TRELLIS.2 reference checkout has tracked modifications; use --allow-dirty only intentionally")
    return {"repo": str(repo), "head": head, "dirty": bool(dirty)}


class Dump:
    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.arrays: dict[str, Any] = {}

    def save_np(self, name: str, value: Any) -> None:
        import numpy as np
        import torch

        if hasattr(value, "feats") and hasattr(value, "coords"):
            raise TypeError("save_np expects tensor/array, not SparseTensor")
        if torch.is_tensor(value):
            t = value.detach().cpu()
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            value = t.numpy()
        arr = np.ascontiguousarray(value)
        path = self.out / f"{name}.npy"
        np.save(path, arr)
        self.arrays[name] = {
            "file": path.name,
            "sha256": _sha256_file(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    def save_sparse(self, prefix: str, value: Any) -> None:
        self.save_np(prefix + "_coords", value.coords)
        self.save_np(prefix + "_feats", value.feats)


def _load_images(paths: Sequence[Path]) -> list[Any]:
    from PIL import Image
    images = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGBA").copy())
    return images


def _to_device_pipeline(pipeline: Any, device: str) -> None:
    import torch
    # Pipeline.cuda() is hard-wired to cuda:0 in the pinned source; preserve an
    # explicit cuda:N chosen by the fixture command.
    pipeline.to(torch.device(device))


def _stage_schedule(steps: int, views: int) -> list[int]:
    return [i % views for i in range(steps)]


def _save_preprocessed(dump: Dump, pipeline: Any, images: Sequence[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    records = []
    processed = []
    for i, image in enumerate(images):
        out = pipeline.preprocess_image(image)
        p = dump.out / f"preprocessed_view{i}.png"
        out.save(p, format="PNG")
        processed.append(out)
        records.append({
            "view": i,
            "file": p.name,
            "sha256": _sha256_file(p),
            "size": list(out.size),
            "mode": out.mode,
        })
    return processed, records


def _cond_bank(pipeline: Any, images: Sequence[Any], resolution: int) -> Any:
    import torch
    conds = [pipeline.get_cond([im], resolution)["cond"] for im in images]
    return torch.cat(conds, dim=0)


def _sample_with_injection(
    pipeline: Any,
    sampler_name: str,
    flow_model: Any,
    noise: Any,
    cond: Any,
    neg_cond: Any,
    params: dict[str, Any],
    mode: str,
    **kwargs: Any,
) -> Any:
    steps = int(params["steps"])
    sampler = getattr(pipeline, sampler_name)
    with pipeline.inject_sampler_multi_image(sampler_name, len(cond), steps, mode=mode):
        return sampler.sample(
            flow_model,
            noise,
            cond=cond,
            neg_cond=neg_cond,
            **params,
            **kwargs,
            verbose=False,
        ).samples


def _denorm_sparse(pipeline: Any, slat: Any, which: str) -> Any:
    import torch
    norm = pipeline.shape_slat_normalization if which == "shape" else pipeline.tex_slat_normalization
    std = torch.tensor(norm["std"], device=slat.feats.device, dtype=slat.feats.dtype)[None]
    mean = torch.tensor(norm["mean"], device=slat.feats.device, dtype=slat.feats.dtype)[None]
    return slat * std + mean


def _shape_norm(pipeline: Any, shape_slat: Any) -> Any:
    import torch
    std = torch.tensor(
        pipeline.shape_slat_normalization["std"],
        device=shape_slat.feats.device,
        dtype=shape_slat.feats.dtype,
    )[None]
    mean = torch.tensor(
        pipeline.shape_slat_normalization["mean"],
        device=shape_slat.feats.device,
        dtype=shape_slat.feats.dtype,
    )[None]
    return (shape_slat - mean) / std


def _decode_summary(pipeline: Any, shape_slat: Any, tex_slat: Any, resolution: int) -> list[dict[str, Any]]:
    import numpy as np

    meshes = pipeline.decode_latent(shape_slat, tex_slat, resolution)
    out = []
    for i, mesh in enumerate(meshes):
        v = np.asarray(mesh.vertices.detach().cpu() if hasattr(mesh.vertices, "detach") else mesh.vertices)
        f = np.asarray(mesh.faces.detach().cpu() if hasattr(mesh.faces, "detach") else mesh.faces)
        c = np.asarray(mesh.coords.detach().cpu() if hasattr(mesh.coords, "detach") else mesh.coords)
        a = np.asarray(mesh.attrs.detach().cpu() if hasattr(mesh.attrs, "detach") else mesh.attrs)
        out.append({
            "index": i,
            "vertices": int(len(v)),
            "faces": int(len(f)),
            "voxel_coords": int(len(c)),
            "bbox_min": v.min(axis=0).astype(float).tolist() if len(v) else None,
            "bbox_max": v.max(axis=0).astype(float).tolist() if len(v) else None,
            "vertices_sha256": _sha256_bytes(np.ascontiguousarray(v).tobytes()),
            "faces_sha256": _sha256_bytes(np.ascontiguousarray(f).tobytes()),
            "coords_sha256": _sha256_bytes(np.ascontiguousarray(c).tobytes()),
            "attrs_sha256": _sha256_bytes(np.ascontiguousarray(a).tobytes()),
            "attrs_finite": bool(np.isfinite(a).all()),
        })
    return out


def run_full(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    pin = _require_pinned_checkout(repo, args.allow_dirty)

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(repo))

    import numpy as np
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor

    image_paths = [Path(x).resolve() for x in args.images]
    if not (1 <= len(image_paths) <= 8):
        raise SystemExit("reference generator accepts 1..8 images")
    for p in image_paths:
        if not p.is_file():
            raise SystemExit(f"missing image: {p}")

    out = Path(args.out).resolve()
    dump = Dump(out)
    input_records = [{
        "view": i,
        "path": str(p),
        "sha256": _sha256_file(p),
    } for i, p in enumerate(image_paths)]

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    _to_device_pipeline(pipeline, args.device)

    torch.manual_seed(args.seed)
    images = _load_images(image_paths)
    images, prep_records = _save_preprocessed(dump, pipeline, images)

    cond512 = _cond_bank(pipeline, images, 512)
    neg512 = torch.zeros_like(cond512[:1])
    dump.save_np("cond512", cond512)
    dump.save_np("neg512", neg512)

    cond1024 = None
    neg1024 = None
    if args.pipeline_type != "512":
        cond1024 = _cond_bank(pipeline, images, 1024)
        neg1024 = torch.zeros_like(cond1024[:1])
        dump.save_np("cond1024", cond1024)
        dump.save_np("neg1024", neg1024)

    # ---- Sparse structure flow ----
    ss_flow = pipeline.models["sparse_structure_flow_model"]
    ss_res = {"512": 32, "1024_cascade": 32, "1536_cascade": 32}[args.pipeline_type]
    ss_noise = torch.randn(
        1, ss_flow.in_channels, ss_flow.resolution, ss_flow.resolution, ss_flow.resolution,
        device=pipeline.device,
    )
    dump.save_np("ss_noise", ss_noise)
    ss_params = dict(pipeline.sparse_structure_sampler_params)
    if pipeline.low_vram:
        ss_flow.to(pipeline.device)
    z_s = _sample_with_injection(
        pipeline, "sparse_structure_sampler", ss_flow, ss_noise,
        cond512, neg512, ss_params, args.mode,
        tqdm_desc="reference SS",
    )
    if pipeline.low_vram:
        ss_flow.cpu()
    dump.save_np("ss_latent", z_s)

    decoder = pipeline.models["sparse_structure_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
    decoded = decoder(z_s) > 0
    if pipeline.low_vram:
        decoder.cpu()
    if ss_res != decoded.shape[2]:
        ratio = decoded.shape[2] // ss_res
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    dump.save_np("ss_coords", coords)

    # ---- Shape SLat ----
    shape_params = dict(pipeline.shape_slat_sampler_params)
    shape_lr = None
    shape_final = None
    final_resolution = 512
    hr_coords = None
    quant_coords = None

    flow_lr = pipeline.models["shape_slat_flow_model_512"]
    lr_noise = SparseTensor(
        feats=torch.randn(coords.shape[0], flow_lr.in_channels, device=pipeline.device),
        coords=coords,
    )
    dump.save_sparse("shape_lr_noise", lr_noise)
    if pipeline.low_vram:
        flow_lr.to(pipeline.device)
    shape_lr_norm = _sample_with_injection(
        pipeline, "shape_slat_sampler", flow_lr, lr_noise,
        cond512, neg512, shape_params, args.mode,
        tqdm_desc="reference shape LR",
    )
    if pipeline.low_vram:
        flow_lr.cpu()
    dump.save_sparse("shape_lr_norm", shape_lr_norm)
    shape_lr = _denorm_sparse(pipeline, shape_lr_norm, "shape")
    dump.save_sparse("shape_lr_denorm", shape_lr)

    if args.pipeline_type == "512":
        shape_final = shape_lr
        final_resolution = 512
        tex_cond, tex_neg = cond512, neg512
        tex_flow = pipeline.models["tex_slat_flow_model_512"]
    else:
        shape_dec = pipeline.models["shape_slat_decoder"]
        if pipeline.low_vram:
            shape_dec.to(pipeline.device)
            shape_dec.low_vram = True
        hr_coords = shape_dec.upsample(shape_lr, upsample_times=4)
        if pipeline.low_vram:
            shape_dec.cpu()
            shape_dec.low_vram = False
        dump.save_np("shape_hr_upsampled_coords", hr_coords)

        target = 1024 if args.pipeline_type == "1024_cascade" else 1536
        final_resolution = target
        while True:
            quant = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / 512 * (final_resolution // 16)).int(),
            ], dim=1)
            quant_coords = quant.unique(dim=0)
            if quant_coords.shape[0] < args.max_tokens or final_resolution == 1024:
                break
            final_resolution -= 128
        dump.save_np("shape_hr_quant_coords", quant_coords)

        flow_hr = pipeline.models["shape_slat_flow_model_1024"]
        hr_noise = SparseTensor(
            feats=torch.randn(quant_coords.shape[0], flow_hr.in_channels, device=pipeline.device),
            coords=quant_coords,
        )
        dump.save_sparse("shape_hr_noise", hr_noise)
        if pipeline.low_vram:
            flow_hr.to(pipeline.device)
        shape_hr_norm = _sample_with_injection(
            pipeline, "shape_slat_sampler", flow_hr, hr_noise,
            cond1024, neg1024, shape_params, args.mode,
            tqdm_desc="reference shape HR",
        )
        if pipeline.low_vram:
            flow_hr.cpu()
        dump.save_sparse("shape_hr_norm", shape_hr_norm)
        shape_final = _denorm_sparse(pipeline, shape_hr_norm, "shape")
        dump.save_sparse("shape_hr_denorm", shape_final)
        tex_cond, tex_neg = cond1024, neg1024
        tex_flow = pipeline.models["tex_slat_flow_model_1024"]

    # ---- Texture SLat ----
    tex_params = dict(pipeline.tex_slat_sampler_params)
    shape_for_tex = _shape_norm(pipeline, shape_final)
    tex_noise_width = tex_flow.in_channels - shape_for_tex.feats.shape[1]
    tex_noise = shape_for_tex.replace(
        feats=torch.randn(shape_for_tex.coords.shape[0], tex_noise_width, device=pipeline.device)
    )
    dump.save_sparse("tex_noise", tex_noise)
    if pipeline.low_vram:
        tex_flow.to(pipeline.device)
    tex_norm = _sample_with_injection(
        pipeline, "tex_slat_sampler", tex_flow, tex_noise,
        tex_cond, tex_neg, tex_params, args.mode,
        concat_cond=shape_for_tex,
        tqdm_desc="reference texture",
    )
    if pipeline.low_vram:
        tex_flow.cpu()
    dump.save_sparse("tex_norm", tex_norm)
    tex_final = _denorm_sparse(pipeline, tex_norm, "tex")
    dump.save_sparse("tex_denorm", tex_final)

    decode_summary = None
    if args.decode_summary:
        decode_summary = _decode_summary(pipeline, shape_final, tex_final, final_resolution)

    stage_steps = {
        "ss": int(ss_params["steps"]),
        "shape_lr": int(shape_params["steps"]),
        "shape_hr": int(shape_params["steps"]) if args.pipeline_type != "512" else None,
        "texture": int(tex_params["steps"]),
    }
    schedules = {
        k: _stage_schedule(v, len(images)) if v is not None else None
        for k, v in stage_steps.items()
    }

    manifest = {
        "format": FORMAT_VERSION,
        "reference": {
            "primary_repo": REF_REPO,
            "primary_sha": REF_SHA,
            "primary_base_sha": REF_BASE_SHA,
            "original_trellis_semantic_sha": ORIGINAL_TRELLIS_SHA,
            "checkout": pin,
        },
        "run": {
            "mode": args.mode,
            "pipeline_type": args.pipeline_type,
            "seed": args.seed,
            "device": args.device,
            "model": args.model,
            "views": len(images),
            "max_tokens": args.max_tokens,
            "final_resolution": final_resolution,
            "stage_steps": stage_steps,
            "stochastic_view_schedule": schedules if args.mode == "stochastic" else None,
        },
        "inputs": input_records,
        "preprocessed": prep_records,
        "arrays": dump.arrays,
        "decode_summary": decode_summary,
    }
    stable_payload = {
        "reference_sha": REF_SHA,
        "mode": args.mode,
        "pipeline_type": args.pipeline_type,
        "seed": args.seed,
        "input_sha256": [x["sha256"] for x in input_records],
        "preprocessed_sha256": [x["sha256"] for x in prep_records],
        "array_sha256": {k: v["sha256"] for k, v in sorted(dump.arrays.items())},
        "final_resolution": final_resolution,
    }
    manifest["fixture_hash"] = _sha256_bytes(
        json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print("TRELLIS2_MV_REFERENCE_OK")
    print("  reference:", REF_SHA)
    print("  mode:", args.mode, "views:", len(images), "pipeline:", args.pipeline_type)
    print("  final resolution:", final_resolution)
    print("  fixture hash:", manifest["fixture_hash"])
    print("  out:", out)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run stdlib-only synthetic oracle")
    ap.add_argument("--repo", default=os.environ.get("TRELLIS2_REPO", "/tmp/TRELLIS.2"))
    ap.add_argument("--model", default=os.environ.get("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B"))
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--mode", choices=["stochastic", "multidiffusion"], default="stochastic")
    ap.add_argument("--pipeline-type", choices=["512", "1024_cascade", "1536_cascade"], default="1024_cascade")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=os.environ.get("REF_DEV", "cuda:0"))
    ap.add_argument("--out", default=os.environ.get("OUT", "/tmp/trellis2-mv-reference"))
    ap.add_argument("--max-tokens", type=int, default=49152)
    ap.add_argument("--decode-summary", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if not args.selftest and not args.images:
        ap.error("--images is required unless --selftest is used")
    return args


if __name__ == "__main__":
    ns = parse_args()
    raise SystemExit(run_selftest() if ns.selftest else run_full(ns))
