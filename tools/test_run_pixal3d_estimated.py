#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_pixal3d_estimated.py"


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        image = d / "front.png"
        image.write_bytes(b"fixture")
        models = d / "models"
        models.mkdir()
        out = d / "out.glb"

        p = subprocess.run(
            [
                sys.executable, str(RUNNER),
                "--image", str(image),
                "--models", str(models),
                "--mesh-scale", "1.0",
                "--fov", "0.3490658503988659",
                "--trellis-cli", "/usr/bin/true",
                "--output", str(out),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert p.returncode == 0, p.stderr
        assert "camera metadata" in p.stderr
        assert "existing Pixal3D camera-aware pipeline" in p.stderr

        p = subprocess.run(
            [
                sys.executable, str(RUNNER),
                "--image", str(d / "missing.png"),
                "--models", str(models),
                "--mesh-scale", "1.0",
                "--fov", "0.3",
                "--trellis-cli", "/usr/bin/true",
                "--output", str(out),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert p.returncode != 0
        assert "input image not found" in p.stderr

    print("RUN_PIXAL3D_ESTIMATED_TEST_OK")


if __name__ == "__main__":
    main()
