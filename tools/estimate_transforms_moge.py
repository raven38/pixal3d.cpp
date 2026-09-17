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
    focal_estimates: list[float],
    max_relative_spread: float = DEFAULT_MAX_FOCAL_SPREAD,
) -> tuple[float, float, float]:
    """Return (median normalized fx, FOV, max relative deviation).

    Canonical front/right/back/left views are assumed to share one camera model.
    Aggregate in focal space so no per-view FOV noise reaches Pixal3D. A strongly
    inconsistent set fails closed because the canonical-view assumption is then
    not trustworthy.
    """
    if not focal_estimates:
        raise ValueError("no focal estimates were provided")
    if not math.isfinite(max_relative_spread) or max_relative_spread < 0.0:
        raise ValueError("max_relative_spread must be finite and >= 0")
    values = [validate_normalized_fx(v) for v in focal_estimates]
    shared_fx = float(statistics.median(values))
    spread = max(abs(v / shared_fx - 1.0) for v in values)
    if spread > max_relative_spread:
        raise ValueError(
            "canonical view focal estimates disagree: "
            f"max relative deviation {spread:.3f} exceeds {max_relative_spread:.3f}"
        )
    return shared_fx, fov_from_normalized_fx(shared_fx), spread


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
        help="maximum relative deviation from the median normalized fx in canonical-4view mode (default: 0.25)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    mesh_scale = validate_scale(args.mesh_scale)
    expected = CANONICAL_VIEW_COUNT if args.canonical_4view else 1
    image_paths = list_images(args.views_dir, expected_count=expected)

    shared_fx = None
    spread = None
    if args.manual_fov is not None:
        fov = validate_fov(args.manual_fov)
        source = "manual"
    elif args.canonical_4view:
        focals = estimate_normalized_fx_moge(image_paths, args.model, args.device)
        try:
            shared_fx, fov, spread = shared_fov_from_focal_estimates(
                focals, max_relative_spread=args.max_focal_spread
            )
        except ValueError as exc:
            die(str(exc))
        source = "moge-2-shared"
    else:
        fov = estimate_fov_moge(image_paths[0], args.model, args.device)
        source = "moge-2"

    if args.canonical_4view:
        data = make_canonical4_transforms([p.name for p in image_paths], fov, mesh_scale)
        distance = canonical_distance_from_fov(fov)
    else:
        data = make_transforms(image_paths[0].name, fov, mesh_scale)
        distance = -float(data["frames"][0]["transform_matrix"][1][3])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
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
