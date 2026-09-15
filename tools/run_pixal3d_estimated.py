#!/usr/bin/env python3
"""Run the existing Pixal3D camera-aware pipeline from one image, without a user JSON.

This wrapper estimates/writes camera metadata into a private temporary views
directory, then invokes trellis-cli --views on that directory.  The image must
already be pre-matted RGBA because the current Pixal3D --views loader deliberately
does not perform background removal.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ESTIMATOR = HERE / "estimate_transforms_moge.py"


def fail(message: str) -> "NoReturn":
    print(f"run_pixal3d_estimated: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="one image -> estimated camera metadata -> existing Pixal3D pipeline"
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

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pixal3d-estimated-") as td:
        views = Path(td)
        staged = views / image.name
        shutil.copy2(image, staged)
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
            "--seed", str(args.seed),
            "--res", str(args.res),
            "-o", str(output),
        ]
        if args.no_texture:
            run_cmd.append("--no-texture")
        print("[estimated] running existing Pixal3D camera-aware pipeline", file=sys.stderr)
        result = subprocess.run(run_cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print(f"[estimated] wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
