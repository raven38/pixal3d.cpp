#!/usr/bin/env python3
"""Run canonical front/right/back/left Pixal3D with one shared estimated FOV.

The four view identities are explicit inputs and their extrinsics stay fixed. Each
pre-matted RGBA image is normalized with the same Pixal3D object crop used by the
single-image wrapper, MoGe-2 estimates normalized focal length per staged crop,
and estimate_transforms_moge.py writes one shared-FOV transforms.json. The normal
Pixal3D multiview flow family then consumes those staged images + metadata.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from run_pixal3d_estimated import preprocess_prematted_rgba

HERE = Path(__file__).resolve().parent
DEFAULT_ESTIMATOR = HERE / "estimate_transforms_moge.py"
DEFAULT_MODEL = "Ruicheng/moge-2-vitl"
DEFAULT_MAX_FOCAL_SPREAD = 0.25
STAGED_NAMES = ("01_front.png", "02_right.png", "03_back.png", "04_left.png")


def fail(message: str) -> "NoReturn":
    print(f"run_pixal3d_canonical4_estimated: {message}", file=sys.stderr)
    raise SystemExit(2)


def validate_mesh_scale(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        fail("mesh_scale must be a finite value > 0")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "front/right/back/left RGBA -> matched Pixal3D crops -> shared MoGe FOV -> "
            "fixed canonical extrinsics -> Pixal3D MV weights"
        )
    )
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--back", required=True, type=Path)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--mesh-scale", required=True, type=float,
                        help="existing canonical MV mesh_scale; no default is assumed")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--trellis-cli", type=Path, default=Path("./build/trellis-cli"))
    parser.add_argument("--estimator", type=Path, default=DEFAULT_ESTIMATOR)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MoGe-2 model identifier")
    parser.add_argument("--device", default="auto", help="MoGe device: auto/cpu/cuda/...")
    parser.add_argument(
        "--max-focal-spread",
        type=float,
        default=DEFAULT_MAX_FOCAL_SPREAD,
        help="fail if any per-view normalized fx differs from the median by more than this fraction",
    )
    parser.add_argument("--fov", type=float, default=None,
                        help="manual shared horizontal FOV in radians; skips MoGe-2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--res", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--no-texture", action="store_true")
    args = parser.parse_args()

    sources = [args.front.resolve(), args.right.resolve(), args.back.resolve(), args.left.resolve()]
    for source in sources:
        if not source.is_file():
            fail(f"input image not found: {source}")

    models = args.models.resolve()
    cli = args.trellis_cli.resolve()
    estimator = args.estimator.resolve()
    output = args.output.resolve()
    mesh_scale = validate_mesh_scale(args.mesh_scale)

    if not models.is_dir():
        fail(f"model directory not found: {models}")
    if not cli.is_file():
        fail(f"trellis-cli not found: {cli}")
    if not estimator.is_file():
        fail(f"camera estimator not found: {estimator}")
    if not math.isfinite(args.max_focal_spread) or args.max_focal_spread < 0.0:
        fail("max_focal_spread must be finite and >= 0")

    help_probe = subprocess.run(
        [str(cli), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if "--pixal3d-weights" not in (help_probe.stdout or ""):
        fail("trellis-cli does not support --pixal3d-weights; use a build with PR #6 or newer")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pixal3d-canonical4-estimated-") as td:
        views = Path(td)
        for source, staged_name in zip(sources, STAGED_NAMES):
            preprocess_prematted_rgba(source, views / staged_name)

        transforms = views / "transforms.json"
        estimate_cmd = [
            sys.executable, str(estimator),
            "--views-dir", str(views),
            "--mesh-scale", repr(mesh_scale),
            "--output", str(transforms),
            "--canonical-4view",
            "--max-focal-spread", repr(args.max_focal_spread),
            "--model", args.model,
            "--device", args.device,
        ]
        if args.fov is not None:
            estimate_cmd += ["--manual-fov", repr(args.fov)]
        print(
            f"[canonical 4-view] camera metadata (mesh_scale={mesh_scale:g}, "
            f"focal_spread_limit={args.max_focal_spread:g})",
            file=sys.stderr,
        )
        subprocess.run(estimate_cmd, check=True)

        run_cmd = [
            str(cli),
            "--views", str(views),
            "--models", str(models),
            "--pixal3d-weights", "mv",
            "--seed", str(args.seed),
            "--res", str(args.res),
            "-o", str(output),
        ]
        if args.no_texture:
            run_cmd.append("--no-texture")
        print("[canonical 4-view] running multiview Pixal3D weights", file=sys.stderr)
        result = subprocess.run(run_cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print(f"[canonical 4-view] wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
