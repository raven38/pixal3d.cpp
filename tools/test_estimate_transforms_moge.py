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

        # front_back aggregation (default): shared fx is the front/back mean, the
        # consistency gate looks only at front vs back, and side-view disagreement is
        # a warning rather than a rejection (MoGe-2 biases thin right/left views).
        agg = estimator.aggregate_canonical_focals([1.60, 0.90, 1.40, 0.85])
        assert agg["aggregation"] == "front_back"
        assert abs(agg["shared_fx"] - 1.50) < 1e-12
        assert abs(agg["spread_used"] - (1.60 / 1.40 - 1.0)) < 1e-12
        swapped = estimator.aggregate_canonical_focals([1.40, 0.90, 1.60, 0.85])
        assert abs(swapped["spread_used"] - agg["spread_used"]) < 1e-12 and swapped["accepted"] is True
        assert agg["accepted"] is True
        assert agg["spread_all"] > 0.25 and agg["side_spread"] > 0.25
        assert any("diagnostic only" in w for w in agg["warnings"]), agg["warnings"]
        assert set(agg["fx"]) == {"front", "right", "back", "left"}
        # median4 on the same input is the original behaviour: rejected by spread.
        agg4 = estimator.aggregate_canonical_focals([1.60, 0.90, 1.40, 0.85], aggregation="median4")
        assert agg4["accepted"] is False and abs(agg4["shared_fx"] - 1.15) < 1e-12
        # front/back themselves inconsistent -> not accepted, no exception.
        bad = estimator.aggregate_canonical_focals([1.40, 1.0, 1.0, 1.0])
        assert bad["accepted"] is False and "front/back" in bad["warnings"][0]
        # Fallback policy: accepted -> estimate; rejected -> explicit fallback FOV (flagged);
        # rejected without fallback -> ValueError (fail closed).
        fov, source, used = estimator.choose_canonical_fov(agg, FOV)
        assert source == "moge-2-front_back" and used is False and abs(fov - agg["fov"]) < 1e-12
        fov, source, used = estimator.choose_canonical_fov(bad, FOV)
        assert source == "fallback" and used is True and abs(fov - FOV) < 1e-12
        try:
            estimator.choose_canonical_fov(bad, None)
        except ValueError as exc:
            assert "front/back" in str(exc)
        else:
            raise AssertionError("rejected estimate without fallback did not fail closed")
        try:
            estimator.aggregate_canonical_focals([1.0, 1.0, 1.0, 1.0], aggregation="nope")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown aggregation was accepted")

        # Direct canonical-4view directory mode must keep the same natural filename
        # ordering contract as the C++ canonical rig (view2 before view10).
        natural = d / "natural"
        natural.mkdir()
        natural_names = ["view1.png", "view2.png", "view10.png", "view11.png"]
        for name in natural_names:
            (natural / name).write_bytes(b"fixture")
        assert [p.name for p in estimator.list_images(natural, expected_count=4)] == natural_names

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
        # camera_estimation.json is written next to transforms.json, also in manual mode,
        # with an explicit --camera-json override honoured.
        cam = json.loads((canonical / "camera_estimation.json").read_text())
        assert cam["source"] == "manual" and cam["fallback_used"] is False
        assert abs(cam["selected_fov_rad"] - wider_fov) < 1e-12 and cam["mesh_scale"] == 0.7
        assert cam["image_names"] == names
        custom = d / "diag" / "cam.json"
        p = run(
            "--views-dir", canonical, "--mesh-scale", 0.7, "--output", canonical_out,
            "--canonical-4view", "--manual-fov", wider_fov, "--camera-json", custom,
        )
        assert p.returncode == 0, p.stderr
        assert json.loads(custom.read_text())["source"] == "manual"
        # Automatic canonical path exercised end-to-end through the CLI with precomputed
        # focals (--focals-json): accepted, side-view warning only, fallback, fail-closed, median4.
        focals = d / "focals.json"
        focals.write_text(json.dumps({"front": 1.60, "right": 0.90, "back": 1.40, "left": 0.85}))
        cam_out = d / "auto" / "camera.json"
        p = run("--views-dir", canonical, "--mesh-scale", 0.7, "--output", canonical_out,
                "--canonical-4view", "--focals-json", focals, "--camera-json", cam_out)
        assert p.returncode == 0, p.stderr
        assert "diagnostic only" in p.stderr and "falling back" not in p.stderr, p.stderr
        cam = json.loads(cam_out.read_text())
        assert cam["source"] == "moge-2-front_back" and cam["accepted"] is True and cam["fallback_used"] is False
        assert abs(cam["estimated_shared_fx"] - 1.5) < 1e-12
        assert abs(cam["selected_fov_rad"] - cam["estimated_fov_rad"]) < 1e-12
        assert abs(cam["selected_fov_rad"] - 2.0 * math.atan(1.0 / 3.0)) < 1e-12
        assert cam["fx"] == {"front": 1.6, "right": 0.9, "back": 1.4, "left": 0.85}
        assert cam["spread_all"] > 0.25 and cam["thresholds"]["max_front_back_diff"] == 0.25
        data = json.loads(canonical_out.read_text())
        assert abs(data["camera_angle_x"] - cam["selected_fov_rad"]) < 1e-12
        assert abs(cam["distance"] - CANONICAL_DISTANCE * math.tan(FOV / 2.0) / math.tan(cam["selected_fov_rad"] / 2.0)) < 1e-9
        assert cam["estimator"] == "MoGe-2" and "package" in cam and "revision" in cam

        # front/back inconsistent + fallback: transforms.json uses the fallback FOV, the
        # rejected estimate stays recorded separately.
        focals.write_text(json.dumps([1.60, 1.0, 1.0, 1.0]))
        p = run("--views-dir", canonical, "--mesh-scale", 0.7, "--output", canonical_out,
                "--canonical-4view", "--focals-json", focals, "--camera-json", cam_out, "--fallback-fov", FOV)
        assert p.returncode == 0, p.stderr
        assert "falling back" in p.stderr, p.stderr
        cam = json.loads(cam_out.read_text())
        assert cam["source"] == "fallback" and cam["fallback_used"] is True and cam["accepted"] is False
        assert abs(cam["selected_fov_rad"] - FOV) < 1e-12 and abs(cam["estimated_shared_fx"] - 1.3) < 1e-12
        assert abs(json.loads(canonical_out.read_text())["camera_angle_x"] - FOV) < 1e-12
        # symmetric gate: swapping front/back gives the same decision
        focals.write_text(json.dumps([1.0, 1.0, 1.60, 1.0]))
        p = run("--views-dir", canonical, "--mesh-scale", 0.7, "--output", canonical_out,
                "--canonical-4view", "--focals-json", focals, "--camera-json", cam_out, "--fallback-fov", FOV)
        assert p.returncode == 0 and json.loads(cam_out.read_text())["fallback_used"] is True

        # same input without fallback fails closed: no transforms.json, diagnostic says rejected.
        canonical_out.unlink()
        p = run("--views-dir", canonical, "--mesh-scale", 0.7, "--output", canonical_out,
                "--canonical-4view", "--focals-json", focals, "--camera-json", cam_out)
        assert p.returncode != 0 and "front/back" in p.stderr, p.stderr
        assert not canonical_out.exists()
        assert json.loads(cam_out.read_text())["source"] == "rejected"

        # median4 keeps the original behaviour on the side-biased pattern: rejected.
        focals.write_text(json.dumps([1.60, 0.90, 1.40, 0.85]))
        p = run("--views-dir", canonical, "--mesh-scale", 0.7, "--output", canonical_out,
                "--canonical-4view", "--focals-json", focals, "--camera-json", cam_out, "--aggregation", "median4")
        assert p.returncode != 0 and "disagree" in p.stderr.lower(), p.stderr
        assert not canonical_out.exists()
        # --focals-json needs --canonical-4view; malformed content is rejected
        p = run("--views-dir", views, "--mesh-scale", 1.0, "--output", out, "--focals-json", focals)
        assert p.returncode != 0
        focals.write_text(json.dumps([1.0, 1.0]))
        p = run("--views-dir", canonical, "--mesh-scale", 0.7, "--output", canonical_out,
                "--canonical-4view", "--focals-json", focals)
        assert p.returncode != 0

        # --fallback-fov must be a valid FOV.
        p = run(
            "--views-dir", canonical, "--mesh-scale", 0.7, "--output", canonical_out,
            "--canonical-4view", "--manual-fov", wider_fov, "--fallback-fov", 4.0,
        )
        assert p.returncode != 0

    print("ESTIMATE_TRANSFORMS_MOGE_TEST_OK")


if __name__ == "__main__":
    main()
