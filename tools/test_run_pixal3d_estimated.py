#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_pixal3d_estimated.py"


def make_fake_cli(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json, sys
from pathlib import Path
from PIL import Image

if '--help' in sys.argv:
    print('trellis-cli ... --pixal3d-weights sv|mv ...')
    raise SystemExit(0)

args = sys.argv[1:]
def value(flag):
    i = args.index(flag)
    return args[i + 1]
views = Path(value('--views'))
out = Path(value('-o'))
images = sorted(p for p in views.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
assert len(images) == 1, images
rgba = Image.open(images[0]).convert('RGBA')
a = rgba.getchannel('A')
black = Image.new('RGB', rgba.size, (0, 0, 0))
black.paste(rgba.convert('RGB'), mask=a)
out.write_text(json.dumps({
    'argv': args,
    'size': list(rgba.size),
    'alpha_extrema': list(a.getextrema()),
    'corner_rgb': list(black.getpixel((0, 0))),
    'image_name': images[0].name,
}))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        image = d / "front.png"
        # Transparent border deliberately contains hidden red RGB. The visible object is green.
        # Correct preprocessing keeps alpha for C++ but both C++ premultiplication and MoGe's
        # RGB view must see the transparent border as black.
        rgba = Image.new("RGBA", (100, 80), (255, 0, 0, 0))
        draw = ImageDraw.Draw(rgba)
        draw.rectangle((30, 20, 69, 59), fill=(0, 255, 0, 255))
        rgba.save(image)

        models = d / "models"
        models.mkdir()
        out = d / "out.glb"
        fake_cli = d / "fake_trellis_cli.py"
        make_fake_cli(fake_cli)

        p = subprocess.run(
            [
                sys.executable, str(RUNNER),
                "--image", str(image),
                "--models", str(models),
                "--mesh-scale", "1.0",
                "--fov", "0.3490658503988659",
                "--trellis-cli", str(fake_cli),
                "--output", str(out),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert p.returncode == 0, p.stderr
        assert "camera metadata" in p.stderr
        assert "single-view Pixal3D weights" in p.stderr
        payload = json.loads(out.read_text())
        assert payload["size"][0] == payload["size"][1], payload
        assert payload["size"][0] < 80, payload  # object-centric crop, not the raw 100x80 canvas
        assert payload["alpha_extrema"] == [0, 255], payload
        assert payload["corner_rgb"] == [0, 0, 0], payload
        assert payload["image_name"] == "input.png", payload
        argv = payload["argv"]
        wi = argv.index("--pixal3d-weights")
        assert argv[wi + 1] == "sv", argv

        p = subprocess.run(
            [
                sys.executable, str(RUNNER),
                "--image", str(d / "missing.png"),
                "--models", str(models),
                "--mesh-scale", "1.0",
                "--fov", "0.3",
                "--trellis-cli", str(fake_cli),
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
