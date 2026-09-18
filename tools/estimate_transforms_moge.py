#!/usr/bin/env python3
"""Estimate Pixal3D camera metadata with MoGe-2.

The single-image path mirrors TencentARC/Pixal3D inference.py: MoGe-2 estimates
normalized intrinsics, horizontal FOV is derived from fx, and camera distance
is computed from FOV plus the caller-provided mesh_scale.

The optional canonical 4-view path keeps the existing front/right/back/left
extrinsics fixed, estimates normalized fx independently for the four staged
Pixal3D crops, aggregates one shared focal length in focal space, and writes a
normal Pixal3D transforms.json. It deliberately does not estimate free MV
extrinsics.

Input images are expected to be the Pixal3D object-centric RGBA crops staged by
the wrappers. Automatic MoGe inference composites alpha onto black so the
estimator sees the same RGB as pixal3d.cpp's alpha-premultiplied conditioner.
Manual-FOV mode avoids torch/MoGe entirely.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MODEL = "Ruicheng/moge-2-vitl"
DEFAULT_MAX_FOCAL_SPREAD = 0.25
CANONICAL_FOV = 0.3490658503988659
CANONICAL_DISTANCE = 3.1192049980163574
CANONICAL_VIEW_COUNT = 4


def die(message: str) -> "NoReturn":
    print(f"estimate_transforms_moge: {message}", file=sys.stderr)
    raise SystemExit(2)


def natural_name_key(path: Path) -> tuple:
    """Match the C++ canonical rig's natural ordering (view2 before view10)."""
    parts = re.split(r"(\d+)", path.name)
    key = []
    for part in parts:
        if part.isdigit():
            normalized = part.lstrip("0") or "0"
            key.append((1, len(normalized), normalized, part))
        else:
            key.append((0, part))
    return tuple(key), path.name


def list_images(directory: Path, expected_count: int = 1) -> list[Path]:
    if not directory.is_dir():
        die(f"views directory does not exist: {directory}")
    images = sorted(
        (p for p in directory.iterdir()
         if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS),
        key=natural_name_key,
    )
    if len(images) != expected_count:
        expected = "one image" if expected_count == 1 else f"{expected_count} images"
        die(f"bundled MoGe estimator requires exactly {expected}; found {len(images)}")
    return images


def validate_scale(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        die("mesh_scale must be a finite value > 0")
    return value


def validate_fov(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0 or value >= math.pi:
        die("camera FOV must be finite and satisfy 0 < fov < pi radians")
    return value


def validate_normalized_fx(value: float) -> float:
    fx = float(value)
    if not math.isfinite(fx) or fx <= 0.0:
        raise ValueError(f"normalized fx must be finite and > 0; got {value!r}")
    return fx


def fov_from_normalized_fx(fx_normalized: float) -> float:
    fx = validate_normalized_fx(fx_normalized)
    return validate_fov(2.0 * math.atan(1.0 / (2.0 * fx)))


def shared_fov_from_focal_estimates(
    focals: list[float],
    max_relative_spread: float = DEFAULT_MAX_FOCAL_SPREAD,
) -> tuple[float, float, float]:
    """Aggregate per-view normalized fx into one shared focal length (median of all views).

    Kept as the ``median4`` aggregation. The default canonical path now uses
    ``front_back`` (see ``aggregate_canonical_focals``): on real perspective
    turntables MoGe-2 over-estimates FOV on the thin right/left views by ~+12°
    while front/back are unbiased, so the four-view median is biased upward and
    the four-view spread rejects most real inputs.
    """
    values = [validate_normalized_fx(v) for v in focals]
    if not math.isfinite(max_relative_spread) or max_relative_spread < 0.0:
        raise ValueError("max_relative_spread must be finite and >= 0")
    shared_fx = statistics.median(values)
    spread = max(abs(v / shared_fx - 1.0) for v in values)
    if spread > max_relative_spread:
        raise ValueError(
            "canonical view focal estimates disagree: "
            f"max relative deviation {spread:.3f} exceeds {max_relative_spread:.3f}"
        )
    return shared_fx, fov_from_normalized_fx(shared_fx), spread


AGGREGATIONS = ("front_back", "median4")
DEFAULT_AGGREGATION = "front_back"
DEFAULT_MAX_FRONT_BACK_DIFF = 0.25
CANONICAL_ROLES = ("front", "right", "back", "left")


def aggregate_canonical_focals(
    focals: list[float],
    aggregation: str = DEFAULT_AGGREGATION,
    max_relative_spread: float = DEFAULT_MAX_FOCAL_SPREAD,
    max_front_back_diff: float = DEFAULT_MAX_FRONT_BACK_DIFF,
) -> dict:
    """Aggregate the four canonical focal estimates and classify their consistency.

    Returns a diagnostic dict (never raises for inconsistency):
      shared_fx / fov             the aggregate, always computed
      spread_all                  max_i |fx_i / shared_fx - 1| over all four views
      spread_used                 the consistency statistic of the views the aggregate used
      accepted                    False when the used views are mutually inconsistent
      warnings                    side-view disagreement is diagnostic only
    ``front_back``: shared_fx = mean(fx_front, fx_back); used-view statistic is
    max(fx_front, fx_back) / min(fx_front, fx_back) - 1 against ``max_front_back_diff``.
    ``median4``: shared_fx = median of all four; used-view statistic is spread_all
    against ``max_relative_spread`` (the original PR #16 behaviour).
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")
    values = [validate_normalized_fx(v) for v in focals]
    if len(values) != CANONICAL_VIEW_COUNT:
        raise ValueError(f"canonical aggregation needs {CANONICAL_VIEW_COUNT} focal estimates")
    for name, limit in (("max_relative_spread", max_relative_spread), ("max_front_back_diff", max_front_back_diff)):
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    by_role = dict(zip(CANONICAL_ROLES, values))
    warnings: list[str] = []
    if aggregation == "front_back":
        shared_fx = (by_role["front"] + by_role["back"]) / 2.0
        # symmetric in front/back: ratio of the larger to the smaller estimate
        spread_used = max(by_role["front"], by_role["back"]) / min(by_role["front"], by_role["back"]) - 1.0
        accepted = spread_used <= max_front_back_diff
        if not accepted:
            warnings.append(
                "front/back focal estimates disagree: "
                f"max(fx_front, fx_back) / min(fx_front, fx_back) - 1 = {spread_used:.3f} exceeds {max_front_back_diff:.3f}"
            )
    else:
        shared_fx = statistics.median(values)
        spread_used = max(abs(v / shared_fx - 1.0) for v in values)
        accepted = spread_used <= max_relative_spread
        if not accepted:
            warnings.append(
                "canonical view focal estimates disagree: "
                f"max relative deviation {spread_used:.3f} exceeds {max_relative_spread:.3f}"
            )
    spread_all = max(abs(v / shared_fx - 1.0) for v in values)
    side_spread = max(abs(by_role[r] / shared_fx - 1.0) for r in ("right", "left"))
    if aggregation == "front_back" and side_spread > max_relative_spread:
        warnings.append(
            "right/left focal estimates deviate from the front/back aggregate by "
            f"{side_spread:.3f} (> {max_relative_spread:.3f}); side views are diagnostic only"
        )
    return {
        "aggregation": aggregation,
        "fx": by_role,
        "fov_deg": {r: math.degrees(fov_from_normalized_fx(v)) for r, v in by_role.items()},
        "shared_fx": shared_fx,
        "fov": fov_from_normalized_fx(shared_fx),
        "spread_all": spread_all,
        "spread_used": spread_used,
        "side_spread": side_spread,
        "accepted": accepted,
        "warnings": warnings,
    }


def load_focals_json(path: Path) -> list[float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        try:
            raw = [raw[r] for r in CANONICAL_ROLES]
        except KeyError as exc:
            raise ValueError(f"--focals-json object must have keys {CANONICAL_ROLES}; missing {exc}") from exc
    if not isinstance(raw, list) or len(raw) != CANONICAL_VIEW_COUNT:
        raise ValueError(f"--focals-json must hold {CANONICAL_VIEW_COUNT} normalized fx values")
    return [validate_normalized_fx(float(v)) for v in raw]


def choose_canonical_fov(agg: dict, fallback_fov: float | None) -> tuple[float, str, bool]:
    """Fallback policy: accepted aggregate -> its FOV; rejected -> the explicit fallback FOV
    (flagged) when one was given, otherwise ValueError so the caller fails closed."""
    if agg["accepted"]:
        return agg["fov"], f"moge-2-{agg['aggregation']}", False
    if fallback_fov is not None:
        return validate_fov(fallback_fov), "fallback", True
    raise ValueError(agg["warnings"][0] + " (no --fallback-fov given; refusing to write camera metadata)")


def estimator_identity(model_name: str) -> dict:
    """Best-effort estimator identity for camera_estimation.json (never raises)."""
    ident: dict = {"estimator": "MoGe-2", "model": model_name, "package": None,
                   "package_commit": None, "revision": None}
    try:
        import importlib.metadata as md
        ident["package"] = md.version("moge")
        for f in md.files("moge") or []:
            if f.name == "direct_url.json":
                ident["package_commit"] = json.loads(f.read_text()).get("vcs_info", {}).get("commit_id")
    except Exception:  # noqa: BLE001
        pass
    try:
        # the snapshot MoGe actually loaded (hf cache layout: .../snapshots/<revision>/model.pt),
        # not the Hub HEAD, so a stale cache is reported truthfully
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(repo_id=model_name, filename="model.pt")
        if isinstance(cached, str):
            parts = Path(cached).parts
            if "snapshots" in parts:
                ident["revision"] = parts[parts.index("snapshots") + 1]
    except Exception:  # noqa: BLE001
        pass
    return ident


def distance_from_fov(camera_angle_x: float, mesh_scale: float) -> float:
    # Official Pixal3D inference.py, distance_from_fov(), with extend_pixel=0.
    return 1.0 / (2.0 * mesh_scale * math.tan(camera_angle_x / 2.0))


def canonical_distance_from_fov(camera_angle_x: float) -> float:
    """Preserve the existing 20-degree canonical rig's apparent projection scale."""
    fov = validate_fov(camera_angle_x)
    return CANONICAL_DISTANCE * math.tan(CANONICAL_FOV / 2.0) / math.tan(fov / 2.0)


def load_black_composited_rgb(image_path: Path):
    """Return the image Pixal3D/MoGe should see: alpha composited onto black."""
    try:
        from PIL import Image
    except Exception as exc:
        die(f"automatic FOV estimation needs Pillow; import failed: {exc}")
    with Image.open(image_path) as raw:
        rgba = raw.convert("RGBA")
    black = Image.new("RGB", rgba.size, (0, 0, 0))
    black.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    return black


def estimate_normalized_fx_moge(
    image_paths: list[Path], model_name: str, device: str
) -> list[float]:
    try:
        import numpy as np
        import torch
        from moge.model.v2 import MoGeModel
    except Exception as exc:
        die(
            "automatic FOV estimation needs torch, numpy and MoGe-2 (moge); "
            f"import failed: {exc}"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[camera] loading MoGe-2 {model_name} on {device}", file=sys.stderr)
    model = MoGeModel.from_pretrained(model_name).to(device)
    model.eval()
    focals: list[float] = []
    for image_path in image_paths:
        rgb = load_black_composited_rgb(image_path)
        image_np = np.asarray(rgb, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
        with torch.no_grad():
            output = model.infer(image_tensor)
        intrinsics = output.get("intrinsics")
        if intrinsics is None:
            die(f"MoGe-2 output did not contain intrinsics for {image_path.name}")
        try:
            fx_normalized = validate_normalized_fx(
                float(intrinsics.squeeze().detach().cpu()[0, 0].item())
            )
        except ValueError as exc:
            die(f"MoGe-2 returned invalid intrinsics for {image_path.name}: {exc}")
        focals.append(fx_normalized)
        print(
            f"[camera] view={image_path.name} fx_normalized={fx_normalized:.9f}",
            file=sys.stderr,
        )
    return focals


def estimate_fov_moge(image_path: Path, model_name: str, device: str) -> float:
    fx_normalized = estimate_normalized_fx_moge([image_path], model_name, device)[0]
    return fov_from_normalized_fx(fx_normalized)


def make_transforms(image_name: str, fov: float, mesh_scale: float) -> dict:
    distance = distance_from_fov(fov, mesh_scale)
    return {
        "camera_angle_x": fov,
        "mesh_scale": mesh_scale,
        "frames": [
            {
                "file_path": image_name,
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, -distance],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }


def make_canonical4_transforms(image_names: list[str], fov: float, mesh_scale: float) -> dict:
    if len(image_names) != CANONICAL_VIEW_COUNT:
        raise ValueError(f"canonical rig requires {CANONICAL_VIEW_COUNT} images")
    distance = canonical_distance_from_fov(fov)
    matrices = [
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -distance], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0, distance], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, distance], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 0.0, -1.0, -distance], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    ]
    return {
        "camera_angle_x": validate_fov(fov),
        "mesh_scale": mesh_scale,
        "frames": [
            {"file_path": name, "transform_matrix": matrix}
            for name, matrix in zip(image_names, matrices)
        ],
    }


def write_camera_json(args, diag: dict) -> Path:
    """Write the diagnostic next to transforms.json (issue #15 §4: per-view fx, shared fx,
    spread, estimator identity). Experiment metadata stays out of transforms.json itself."""
    path = args.camera_json or (args.output.parent / "camera_estimation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
    print(f"[camera] wrote {path}", file=sys.stderr)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate camera metadata and write Pixal3D transforms.json"
    )
    parser.add_argument("--views-dir", required=True, type=Path)
    parser.add_argument("--mesh-scale", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manual-fov", type=float, default=None,
                        help="horizontal FOV in radians; skips MoGe-2")
    parser.add_argument(
        "--canonical-4view",
        action="store_true",
        help="treat four natural-order views as fixed front/right/back/left extrinsics and estimate one shared FOV",
    )
    parser.add_argument(
        "--max-focal-spread",
        type=float,
        default=DEFAULT_MAX_FOCAL_SPREAD,
        help="canonical-4view: rejection threshold for the four-view spread under --aggregation median4; "
             "under front_back it only sets the right/left deviation that triggers a warning (default: 0.25)",
    )
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATIONS,
        default=DEFAULT_AGGREGATION,
        help="canonical-4view shared focal: front_back = mean(fx_front, fx_back) (default; right/left are diagnostic only), "
             "median4 = median of all four views (original behaviour)",
    )
    parser.add_argument(
        "--max-front-back-diff",
        type=float,
        default=DEFAULT_MAX_FRONT_BACK_DIFF,
        help="front_back aggregation: reject the automatic estimate when |fx_front / fx_back - 1| exceeds this (default: 0.25)",
    )
    parser.add_argument(
        "--fallback-fov",
        type=float,
        default=None,
        help="horizontal FOV in radians to use (with a warning) when the automatic canonical estimate is rejected; "
             "without it the helper fails closed and writes nothing",
    )
    parser.add_argument(
        "--camera-json",
        type=Path,
        default=None,
        help="where to write the camera_estimation.json diagnostic (default: next to --output)",
    )
    parser.add_argument(
        "--focals-json",
        type=Path,
        default=None,
        help="canonical-4view: skip MoGe-2 and aggregate precomputed normalized fx from this JSON "
             "(a list of four in front/right/back/left order, or an object keyed by those roles); "
             "for offline re-aggregation and contract tests",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()
    if args.focals_json is not None and not args.canonical_4view:
        parser.error("--focals-json requires --canonical-4view")
    if args.fallback_fov is not None:
        args.fallback_fov = validate_fov(args.fallback_fov)

    mesh_scale = validate_scale(args.mesh_scale)
    expected = CANONICAL_VIEW_COUNT if args.canonical_4view else 1
    image_paths = list_images(args.views_dir, expected_count=expected)

    shared_fx = None
    spread = None
    diag: dict = {"source": None, "image_names": [p.name for p in image_paths], "mesh_scale": mesh_scale,
                  "warnings": [], "fallback_used": False, "fallback_fov": args.fallback_fov}
    if args.manual_fov is not None:
        fov = validate_fov(args.manual_fov)
        source = "manual"
    elif args.canonical_4view:
        if args.focals_json is not None:
            focals = load_focals_json(args.focals_json)
        else:
            focals = estimate_normalized_fx_moge(image_paths, args.model, args.device)
        agg = aggregate_canonical_focals(
            focals, aggregation=args.aggregation, max_relative_spread=args.max_focal_spread,
            max_front_back_diff=args.max_front_back_diff,
        )
        diag.update({k: agg[k] for k in ("aggregation", "fx", "fov_deg", "spread_all", "spread_used",
                                          "side_spread", "accepted")})
        diag["estimated_shared_fx"] = agg["shared_fx"]
        diag["estimated_fov_rad"] = agg["fov"]
        diag["estimated_fov_deg"] = math.degrees(agg["fov"])
        diag["thresholds"] = {"max_focal_spread": args.max_focal_spread, "max_front_back_diff": args.max_front_back_diff}
        diag["warnings"] = list(agg["warnings"])
        diag.update(estimator_identity(args.model))
        for w in agg["warnings"]:
            print(f"[camera] warning: {w}", file=sys.stderr)
        try:
            fov, source, fallback_used = choose_canonical_fov(agg, args.fallback_fov)
        except ValueError as exc:
            diag["source"] = "rejected"     # nothing selected; estimated_* keep the rejected aggregate
            write_camera_json(args, diag)
            die(str(exc))
        if fallback_used:
            diag["fallback_used"] = True
            print(f"[camera] warning: automatic estimate rejected; falling back to manual FOV "
                  f"{math.degrees(fov):.3f} deg", file=sys.stderr)
        else:
            shared_fx, spread = agg["shared_fx"], agg["spread_used"]
    else:
        fov = estimate_fov_moge(image_paths[0], args.model, args.device)
        source = "moge-2"
    diag["source"] = source                 # what the written transforms.json actually uses
    diag["selected_fov_rad"] = fov
    diag["selected_fov_deg"] = math.degrees(fov)

    if args.canonical_4view:
        data = make_canonical4_transforms([p.name for p in image_paths], fov, mesh_scale)
        distance = canonical_distance_from_fov(fov)
    else:
        data = make_transforms(image_paths[0].name, fov, mesh_scale)
        distance = -float(data["frames"][0]["transform_matrix"][1][3])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    diag["distance"] = distance
    diag["transforms_file"] = args.output.name
    write_camera_json(args, diag)
    extra = ""
    if shared_fx is not None and spread is not None:
        extra = f" shared_fx={shared_fx:.9f} max_relative_spread={spread:.6f}"
    print(
        f"[camera] source={source} fov={fov:.9f} rad "
        f"({math.degrees(fov):.3f} deg) distance={distance:.9f} "
        f"mesh_scale={mesh_scale:.6g}{extra}",
        file=sys.stderr,
    )
    print(f"[camera] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
