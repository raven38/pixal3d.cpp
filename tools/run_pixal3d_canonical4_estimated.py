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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from run_pixal3d_estimated import preprocess_prematted_rgba

HERE = Path(__file__).resolve().parent
DEFAULT_ESTIMATOR = HERE / "estimate_transforms_moge.py"
DEFAULT_MODEL = "Ruicheng/moge-2-vitl"
DEFAULT_MAX_FOCAL_SPREAD = 0.25
DEFAULT_MAX_FRONT_BACK_DIFF = 0.25
CANONICAL_FOV = 0.3490658503988659   # the existing fixed 20° canonical rig
AGGREGATIONS = ("front_back", "median4")
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
        help="four-view spread limit: rejects under --aggregation median4, only warns about right/left under front_back",
    )
    parser.add_argument("--fov", type=float, default=None,
                        help="manual shared horizontal FOV in radians; skips MoGe-2")
    parser.add_argument("--aggregation", choices=AGGREGATIONS, default="front_back",
                        help="shared focal from front/back mean (default) or the median of all four views")
    parser.add_argument("--max-front-back-diff", type=float, default=DEFAULT_MAX_FRONT_BACK_DIFF,
                        help="reject the automatic estimate when |fx_front / fx_back - 1| exceeds this (default: 0.25)")
    parser.add_argument("--fallback-fov", type=float, default=CANONICAL_FOV,
                        help="FOV (radians) used with a warning when the automatic estimate is rejected; "
                             "default is the fixed 20° canonical rig")
    parser.add_argument("--no-fallback", action="store_true",
                        help="fail closed instead of falling back when the automatic estimate is rejected")
    parser.add_argument("--focals-json", type=Path, default=None,
                        help="skip MoGe-2 and aggregate precomputed per-view normalized fx from this JSON (offline re-aggregation)")
    parser.add_argument("--camera-json", type=Path, default=None,
                        help="where to write camera_estimation.json (default: <output>.camera_estimation.json)")
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
    if not math.isfinite(args.max_front_back_diff) or args.max_front_back_diff < 0.0:
        fail("max_front_back_diff must be finite and >= 0")
    if not (math.isfinite(args.fallback_fov) and 0.0 < args.fallback_fov < math.pi):
        fail("fallback_fov must satisfy 0 < fov < pi radians")
    camera_json = (args.camera_json or output.with_name(output.stem + ".camera_estimation.json")).resolve()

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
            "--aggregation", args.aggregation,
            "--max-front-back-diff", repr(args.max_front_back_diff),
            "--camera-json", str(views / "camera_estimation.json"),
            "--model", args.model,
            "--device", args.device,
        ]
        if not args.no_fallback:
            estimate_cmd += ["--fallback-fov", repr(args.fallback_fov)]
        if args.fov is not None:
            estimate_cmd += ["--manual-fov", repr(args.fov)]
        if args.focals_json is not None:
            estimate_cmd += ["--focals-json", str(args.focals_json.resolve())]
        print(
            f"[canonical 4-view] camera metadata (mesh_scale={mesh_scale:g}, aggregation={args.aggregation}, "
            f"front_back_diff_limit={args.max_front_back_diff:g}, focal_spread_limit={args.max_focal_spread:g}, "
            f"fallback={'none' if args.no_fallback else f'{math.degrees(args.fallback_fov):.3f} deg'})",
            file=sys.stderr,
        )
        estimate = subprocess.run(estimate_cmd)
        staged_camera = views / "camera_estimation.json"
        failed_camera = camera_json.with_name(camera_json.stem + ".failed.json")
        if estimate.returncode != 0:
            if staged_camera.is_file():
                failed_camera.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged_camera, failed_camera)
                print(f"[canonical 4-view] camera estimate rejected; diagnostic: {failed_camera}", file=sys.stderr)
            raise SystemExit(estimate.returncode)

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
            # keep the diagnostic, but never next to a possibly stale GLB under the success name
            failed_camera.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged_camera, failed_camera)
            print(f"[canonical 4-view] trellis-cli failed; diagnostic: {failed_camera}", file=sys.stderr)
            raise SystemExit(result.returncode)
        camera_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged_camera, camera_json)

    print(f"[canonical 4-view] wrote {output} (camera diagnostic: {camera_json})", file=sys.stderr)


if __name__ == "__main__":
    main()
