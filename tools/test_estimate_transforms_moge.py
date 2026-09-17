#!/usr/bin/env python3
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools" / "estimate_transforms_moge.py"
sys.path.insert(0, str(ROOT / "tools"))
import estimate_transforms_moge as estimator

FOV = 0.3490658503988659
EXPECTED_DISTANCE = 1.0 / (2.0 * math.tan(FOV / 2.0))
CANONICAL_DISTANCE = 3.1192049980163574


def run(*args):
    return subprocess.run(
        [sys.executable, str(HELPER), *map(str, args)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        views = d / "views"
        views.mkdir()
        (views / "front.png").write_bytes(b"fixture")
        out = views / "transforms.json"
        p = run("--views-dir", views, "--mesh-scale", 1.0, "--output", out,
                "--manual-fov", FOV)
        assert p.returncode == 0, p.stderr
        data = json.loads(out.read_text())
        assert data["mesh_scale"] == 1.0
        assert abs(data["camera_angle_x"] - FOV) < 1e-12
        assert len(data["frames"]) == 1
        assert data["frames"][0]["file_path"] == "front.png"
        m = data["frames"][0]["transform_matrix"]
        assert abs(m[1][3] + EXPECTED_DISTANCE) < 1e-6
        assert m[0] == [1.0, 0.0, 0.0, 0.0]
        assert m[2] == [0.0, 1.0, 0.0, 0.0]

        (views / "second.png").write_bytes(b"fixture")
        p = run("--views-dir", views, "--mesh-scale", 1.0, "--output", out,
                "--manual-fov", FOV)
        assert p.returncode != 0
        assert "exactly one image" in p.stderr.lower()

        p = run("--views-dir", views, "--mesh-scale", 0.0, "--output", out,
                "--manual-fov", FOV)
        assert p.returncode != 0
        assert "mesh_scale" in p.stderr

        # Transparent pixels may contain arbitrary hidden RGB in an RGBA file.
        # MoGe must see the same black-composited pixels as pixal3d.cpp's alpha
        # premultiplication, not those hidden values.
        alpha_path = d / "alpha.png"
        rgba = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
        rgba.putpixel((1, 0), (0, 255, 0, 255))
        rgba.save(alpha_path)
        rgb = estimator.load_black_composited_rgb(alpha_path)
        assert rgb.getpixel((0, 0)) == (0, 0, 0)
        assert rgb.getpixel((1, 0)) == (0, 255, 0)

        # Canonical 4-view mode estimates one shared focal length in focal space,
        # not four independent downstream FOVs. Median aggregation should reject
        # a strongly inconsistent set instead of silently accepting it.
        shared_fx, shared_fov, spread = estimator.shared_fov_from_focal_estimates(
            [1.0, 1.1, 0.9, 1.0], max_relative_spread=0.15
        )
        assert abs(shared_fx - 1.0) < 1e-12
        assert abs(shared_fov - 2.0 * math.atan(0.5)) < 1e-12
        assert abs(spread - 0.1) < 1e-12
        try:
            estimator.shared_fov_from_focal_estimates(
                [1.0, 1.0, 1.0, 1.5], max_relative_spread=0.25
            )
        except ValueError as exc:
            assert "disagree" in str(exc).lower()
        else:
            raise AssertionError("inconsistent canonical focal estimates were accepted")

        # Changing FOV in the canonical rig must preserve the existing 20-degree
        # projection gauge by scaling orbit distance with tan(fov/2), while keeping
        # the front/right/back/left orientations and mesh_scale contract unchanged.
        canonical = d / "canonical"
        canonical.mkdir()
        names = ["01_front.png", "02_right.png", "03_back.png", "04_left.png"]
        for name in names:
            (canonical / name).write_bytes(b"fixture")
        canonical_out = canonical / "transforms.json"
        wider_fov = math.radians(30.0)
        p = run(
            "--views-dir", canonical,
            "--mesh-scale", 0.7,
            "--output", canonical_out,
            "--canonical-4view",
            "--manual-fov", wider_fov,
        )
        assert p.returncode == 0, p.stderr
        data = json.loads(canonical_out.read_text())
        assert data["mesh_scale"] == 0.7
        assert abs(data["camera_angle_x"] - wider_fov) < 1e-12
        assert [f["file_path"] for f in data["frames"]] == names
        distance = CANONICAL_DISTANCE * math.tan(FOV / 2.0) / math.tan(wider_fov / 2.0)
        matrices = [f["transform_matrix"] for f in data["frames"]]
        assert abs(matrices[0][1][3] + distance) < 1e-6
        assert abs(matrices[1][0][3] - distance) < 1e-6
        assert abs(matrices[2][1][3] - distance) < 1e-6
        assert abs(matrices[3][0][3] + distance) < 1e-6
        assert matrices[0][0][:3] == [1.0, 0.0, 0.0]
        assert matrices[1][1][:3] == [1.0, 0.0, 0.0]
        assert matrices[2][0][:3] == [-1.0, 0.0, 0.0]
        assert matrices[3][1][:3] == [-1.0, 0.0, 0.0]

    print("ESTIMATE_TRANSFORMS_MOGE_TEST_OK")


if __name__ == "__main__":
    main()
