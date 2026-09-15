#!/usr/bin/env python3
"""Estimate Pixal3D camera metadata for one image.

The automatic path mirrors TencentARC/Pixal3D inference.py: MoGe-2 estimates
normalized intrinsics, horizontal FOV is derived from fx, and camera distance
is computed from FOV plus the caller-provided mesh_scale.  The output is a
normal Pixal3D transforms.json so the existing camera-aware runtime stays
unchanged.

Manual-FOV mode intentionally has no torch/MoGe/Pillow dependency; it exists
both as a useful fallback and as a lightweight CI contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MODEL = "Ruicheng/moge-2-vitl"


def die(message: str) -> "NoReturn":
    print(f"estimate_transforms_moge: {message}", file=sys.stderr)
    raise SystemExit(2)


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        die(f"views directory does not exist: {directory}")
    images = sorted(
        (p for p in directory.iterdir()
         if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name.encode("utf-8"),
    )
    if len(images) != 1:
        die(f"bundled MoGe estimator requires exactly one image; found {len(images)}")
    return images


def validate_scale(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        die("mesh_scale must be a finite value > 0")
    return value


def validate_fov(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0 or value >= math.pi:
        die("camera FOV must be finite and satisfy 0 < fov < pi radians")
    return value


def distance_from_fov(camera_angle_x: float, mesh_scale: float) -> float:
    # Official Pixal3D inference.py, distance_from_fov(), with extend_pixel=0,
    # grid_point=(-1,0,0), target=(0,resolution-1). Algebraically resolution
    # cancels and the expression reduces to this closed form.
    return 1.0 / (2.0 * mesh_scale * math.tan(camera_angle_x / 2.0))


def estimate_fov_moge(image_path: Path, model_name: str, device: str) -> float:
    try:
        import numpy as np
        import torch
        from PIL import Image
        from moge.model.v2 import MoGeModel
    except Exception as exc:
        die(
            "automatic FOV estimation needs torch, numpy, Pillow and MoGe-2 "
            f"(moge); import failed: {exc}"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open(image_path).convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, image).convert("RGB")
    image_np = np.asarray(rgb, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)

    print(f"[camera] loading MoGe-2 {model_name} on {device}", file=sys.stderr)
    model = MoGeModel.from_pretrained(model_name).to(device)
    model.eval()
    with torch.no_grad():
        output = model.infer(image_tensor)

    intrinsics = output.get("intrinsics")
    if intrinsics is None:
        die("MoGe-2 output did not contain intrinsics")
    fx_normalized = float(intrinsics.squeeze().detach().cpu()[0, 0].item())
    if not math.isfinite(fx_normalized) or fx_normalized <= 0.0:
        die(f"MoGe-2 returned invalid normalized fx: {fx_normalized}")

    # Official inference.py computes fx = fx_normalized * width, then
    # 2*atan(width/(2*fx)); width cancels.
    return validate_fov(2.0 * math.atan(1.0 / (2.0 * fx_normalized)))


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate one-image camera metadata and write Pixal3D transforms.json"
    )
    parser.add_argument("--views-dir", required=True, type=Path)
    parser.add_argument("--mesh-scale", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manual-fov", type=float, default=None,
                        help="horizontal FOV in radians; skips MoGe-2")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    mesh_scale = validate_scale(args.mesh_scale)
    image_path = list_images(args.views_dir)[0]

    if args.manual_fov is not None:
        fov = validate_fov(args.manual_fov)
        source = "manual"
    else:
        fov = estimate_fov_moge(image_path, args.model, args.device)
        source = "moge-2"

    data = make_transforms(image_path.name, fov, mesh_scale)
    distance = -float(data["frames"][0]["transform_matrix"][1][3])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        f"[camera] source={source} fov={fov:.9f} rad "
        f"({math.degrees(fov):.3f} deg) distance={distance:.9f} "
        f"mesh_scale={mesh_scale:.6g}",
        file=sys.stderr,
    )
    print(f"[camera] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
