#!/usr/bin/env python3
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools" / "estimate_transforms_moge.py"
FOV = 0.3490658503988659
EXPECTED_DISTANCE = 1.0 / (2.0 * math.tan(FOV / 2.0))


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
        (d / "front.png").write_bytes(b"fixture")
        out = d / "transforms.json"
        p = run("--views-dir", d, "--mesh-scale", 1.0, "--output", out,
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

        (d / "second.png").write_bytes(b"fixture")
        p = run("--views-dir", d, "--mesh-scale", 1.0, "--output", out,
                "--manual-fov", FOV)
        assert p.returncode != 0
        assert "exactly one image" in p.stderr.lower()

        p = run("--views-dir", d, "--mesh-scale", 0.0, "--output", out,
                "--manual-fov", FOV)
        assert p.returncode != 0
        assert "mesh_scale" in p.stderr

    print("ESTIMATE_TRANSFORMS_MOGE_TEST_OK")


if __name__ == "__main__":
    main()
