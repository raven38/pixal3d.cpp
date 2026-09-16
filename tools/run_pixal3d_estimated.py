#!/usr/bin/env python3
"""Run Pixal3D from one pre-matted image without a user-authored camera JSON.

The wrapper mirrors the official Pixal3D single-image object crop, estimates/writes
camera metadata in a private temporary views directory, and invokes trellis-cli with
the dedicated single-view flow weights.  The staged image remains RGBA so the C++
conditioner and MoGe observe the same black-composited pixels.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ESTIMATOR = HERE / "estimate_transforms_moge.py"


def fail(message: str) -> "NoReturn":
    print(f"run_pixal3d_estimated: {message}", file=sys.stderr)
    raise SystemExit(2)


def preprocess_prematted_rgba(source: Path, destination: Path) -> None:
    """Mirror Pixal3D preprocess_image's resize + alpha-bbox crop, preserving RGBA.

    Official Pixal3D composites this crop onto black before both MoGe and the Python
    conditioner.  pixal3d.cpp's --views loader instead keeps RGBA and premultiplies
    RGB by alpha; preserving this cropped alpha therefore gives the same black RGB
    to the C++ conditioner while estimate_transforms_moge.py explicitly performs
    the same black composite for MoGe.
    """
    try:
        from PIL import Image
    except Exception as exc:
        fail(f"one-image preprocessing needs Pillow; import failed: {exc}")

    try:
        with Image.open(source) as raw:
            rgba = raw.convert("RGBA")
    except Exception as exc:
        fail(f"cannot read input image {source}: {exc}")

    alpha = rgba.getchannel("A")
    amin, amax = alpha.getextrema()
    if amin == 255:
        fail("input image has no real alpha matte; provide a pre-matted RGBA image")
    if amax == 0:
        fail("input image is fully transparent")

    max_side = max(rgba.size)
    scale = min(1.0, 1024.0 / float(max_side))
    if scale < 1.0:
        rgba = rgba.resize(
            (max(1, int(rgba.width * scale)), max(1, int(rgba.height * scale))),
            Image.Resampling.LANCZOS,
        )
        alpha = rgba.getchannel("A")

    # Official pipeline.py uses alpha > 0.8*255, then a square crop around the
    # foreground bbox enlarged by 1.1. PIL getbbox returns an exclusive max, so
    # subtract one before reproducing the reference min/max coordinate math.
    mask = alpha.point(lambda a: 255 if a > 0.8 * 255 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        fail("alpha matte has no pixels above the Pixal3D foreground threshold (0.8)")
    x0, y0, x1_exclusive, y1_exclusive = bbox
    x1 = x1_exclusive - 1
    y1 = y1_exclusive - 1
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    span = max(x1 - x0, y1 - y0)
    crop_size = max(2, int(span * 1.1))
    # Keep an even square just like the reference's +/- size//2 crop.
    half = max(1, crop_size // 2)
    crop_size = 2 * half
    left = math.floor(cx - half)
    top = math.floor(cy - half)
    cropped = rgba.crop((left, top, left + crop_size, top + crop_size))

    destination.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(destination, format="PNG")
    print(
        f"[estimated] preprocessed {rgba.width}x{rgba.height} -> "
        f"{cropped.width}x{cropped.height} RGBA",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="one image -> Pixal3D crop -> estimated camera -> single-view Pixal3D weights"
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--mesh-scale", required=True, type=float)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--trellis-cli", type=Path, default=Path("./build/trellis-cli"))
    parser.add_argument("--estimator", type=Path, default=DEFAULT_ESTIMATOR)
    parser.add_argument("--fov", type=float, default=None,
                        help="manual horizontal FOV in radians; omit to run MoGe-2")
    parser.add_argument("--device", default="auto", help="MoGe device: auto/cpu/cuda/...")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--res", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--no-texture", action="store_true")
    args = parser.parse_args()

    image = args.image.resolve()
    models = args.models.resolve()
    cli = args.trellis_cli.resolve()
    estimator = args.estimator.resolve()
    output = args.output.resolve()

    if not image.is_file():
        fail(f"input image not found: {image}")
    if not models.is_dir():
        fail(f"model directory not found: {models}")
    if not cli.is_file():
        fail(f"trellis-cli not found: {cli}")
    if not estimator.is_file():
        fail(f"camera estimator not found: {estimator}")

    # This wrapper intentionally refuses the old V=1 + MV-weight behavior, which
    # was measured to break down badly.  PR #6 adds the explicit SV selector.
    help_probe = subprocess.run(
        [str(cli), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if "--pixal3d-weights" not in (help_probe.stdout or ""):
        fail("trellis-cli does not support --pixal3d-weights; use a build with PR #6 or newer")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pixal3d-estimated-") as td:
        views = Path(td)
        staged = views / "input.png"
        preprocess_prematted_rgba(image, staged)
        transforms = views / "transforms.json"

        estimate_cmd = [
            sys.executable, str(estimator),
            "--views-dir", str(views),
            "--mesh-scale", repr(args.mesh_scale),
            "--output", str(transforms),
            "--device", args.device,
        ]
        if args.fov is not None:
            estimate_cmd += ["--manual-fov", repr(args.fov)]
        print("[estimated] camera metadata", file=sys.stderr)
        subprocess.run(estimate_cmd, check=True)

        run_cmd = [
            str(cli),
            "--views", str(views),
            "--models", str(models),
            "--pixal3d-weights", "sv",
            "--seed", str(args.seed),
            "--res", str(args.res),
            "-o", str(output),
        ]
        if args.no_texture:
            run_cmd.append("--no-texture")
        print("[estimated] running single-view Pixal3D weights", file=sys.stderr)
        result = subprocess.run(run_cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print(f"[estimated] wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
