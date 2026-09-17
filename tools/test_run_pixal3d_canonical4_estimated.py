#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_pixal3d_canonical4_estimated.py"
FOV = 0.3490658503988659
CANONICAL_DISTANCE = 3.1192049980163574


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
assert len(images) == 4, images
checks = []
for path in images:
    rgba = Image.open(path).convert('RGBA')
    alpha = rgba.getchannel('A')
    black = Image.new('RGB', rgba.size, (0, 0, 0))
    black.paste(rgba.convert('RGB'), mask=alpha)
    checks.append({
        'name': path.name,
        'size': list(rgba.size),
        'alpha_extrema': list(alpha.getextrema()),
        'corner_rgb': list(black.getpixel((0, 0))),
    })
tf = json.loads((views / 'transforms.json').read_text())
out.write_text(json.dumps({
    'argv': args,
    'images': checks,
    'camera_angle_x': tf['camera_angle_x'],
    'mesh_scale': tf['mesh_scale'],
    'frame_names': [f['file_path'] for f in tf['frames']],
    'front_distance': -tf['frames'][0]['transform_matrix'][1][3],
}))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def make_rgba(path: Path, visible_rgb: tuple[int, int, int]) -> None:
    rgba = Image.new("RGBA", (100, 80), (255, 0, 0, 0))
    draw = ImageDraw.Draw(rgba)
    draw.rectangle((30, 20, 69, 59), fill=(*visible_rgb, 255))
    rgba.save(path)


def run_runner(*args):
    return subprocess.run(
        [sys.executable, str(RUNNER), *map(str, args)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        source_names = ["front.png", "right.png", "back.png", "left.png"]
        sources = [d / name for name in source_names]
        for path, rgb in zip(sources, [(0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
            make_rgba(path, rgb)

        models = d / "models"
        models.mkdir()
        out = d / "out.glb"
        fake_cli = d / "fake_trellis_cli.py"
        make_fake_cli(fake_cli)

        p = run_runner(
            "--front", sources[0],
            "--right", sources[1],
            "--back", sources[2],
            "--left", sources[3],
            "--models", models,
            "--mesh-scale", "0.7",
            "--fov", FOV,
            "--trellis-cli", fake_cli,
            "--output", out,
        )
        assert p.returncode == 0, p.stderr
        assert "canonical 4-view" in p.stderr.lower()
        payload = json.loads(out.read_text())
        assert payload["mesh_scale"] == 0.7, payload
        assert abs(payload["camera_angle_x"] - FOV) < 1e-12, payload
        assert abs(payload["front_distance"] - CANONICAL_DISTANCE) < 1e-6, payload
        expected_names = ["01_front.png", "02_right.png", "03_back.png", "04_left.png"]
        assert payload["frame_names"] == expected_names, payload
        assert [x["name"] for x in payload["images"]] == expected_names, payload
        for image in payload["images"]:
            assert image["size"][0] == image["size"][1], image
            assert image["size"][0] < 80, image
            assert image["alpha_extrema"] == [0, 255], image
            assert image["corner_rgb"] == [0, 0, 0], image
        argv = payload["argv"]
        wi = argv.index("--pixal3d-weights")
        assert argv[wi + 1] == "mv", argv

        p = run_runner(
            "--front", sources[0],
            "--right", sources[1],
            "--back", sources[2],
            "--left", sources[3],
            "--models", models,
            "--mesh-scale", "0",
            "--fov", FOV,
            "--trellis-cli", fake_cli,
            "--output", d / "bad-scale.glb",
        )
        assert p.returncode != 0
        assert "mesh_scale" in p.stderr

        p = run_runner(
            "--front", d / "missing.png",
            "--right", sources[1],
            "--back", sources[2],
            "--left", sources[3],
            "--models", models,
            "--mesh-scale", "0.7",
            "--fov", FOV,
            "--trellis-cli", fake_cli,
            "--output", d / "missing.glb",
        )
        assert p.returncode != 0
        assert "input image not found" in p.stderr.lower()

    print("RUN_PIXAL3D_CANONICAL4_ESTIMATED_TEST_OK")


if __name__ == "__main__":
    main()
