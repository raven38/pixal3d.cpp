#!/usr/bin/env python3
"""pre-matted RGBA クロップの実装一致検査（設計書 D9）。

Pixal3D の前処理（縮小 → alpha bbox → 1.1 倍の正方クロップ）は、これまで
tools/run_pixal3d_estimated.py と web/app/single_view.js に二重実装されていた。
0.10.0 で共有 C++ 実装（src/image_preprocess.cpp）を正本にしたので、ここで
**Python 実装と C++ 実装が同じ入力に同じクロップを返すこと**を固定する。

  .venv/bin/python tools/test_crop_parity.py ./build-metal/trellis-test-sv-input

判定（設計書 D9）:
  - クロップの 1 辺（出力サイズ）は完全一致
  - 受理/拒否の判断が一致
  - 画素はリサンプラ差（Python=LANCZOS / C++=stb_image_resize）があるので許容差で見る:
    alpha マスク IoU >= 0.99、premultiplied RGB の MAE <= 2/255

JS 実装（canvas）はブラウザ無しでは動かないので、gauge と同じく共有フィクスチャ経由で
web/app/test_single_view.mjs 側が担保する。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image, ImageDraw

from run_pixal3d_estimated import preprocess_prematted_rgba

ALPHA_IOU_MIN = 0.99
RGB_MAE_MAX = 2.0 / 255.0


def fixtures(out_dir: Path) -> list[tuple[str, Path, bool]]:
    """(名前, パス, 受理されるべきか) を返す。設計書 D9 の 6 ケース。"""
    cases: list[tuple[str, Path, bool]] = []

    def save(name: str, img: Image.Image, accept: bool) -> None:
        path = out_dir / f"{name}.png"
        img.save(path)
        cases.append((name, path, accept))

    # 1. 長辺 > 1024（縮小が起きる）
    big = Image.new("RGBA", (2000, 1400), (0, 0, 0, 0))
    ImageDraw.Draw(big).ellipse([400, 200, 1500, 1200], fill=(200, 120, 90, 255))
    save("oversized", big, True)

    # 2. 長辺 <= 1024（縮小なし）
    small = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    ImageDraw.Draw(small).rectangle([50, 60, 240, 330], fill=(60, 160, 220, 255))
    save("small", small, True)

    # 3. bbox の幅・高さが奇数（偶数丸めの境界）
    odd = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(odd).rectangle([10, 20, 76, 98], fill=(255, 255, 255, 255))  # 67 x 79
    save("odd_bbox", odd, True)

    # 4. foreground が画像端に接する（クロップが画像外へ出る）
    edge = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(edge).rectangle([0, 0, 40, 180], fill=(30, 220, 120, 255))
    save("edge", edge, True)

    # 5. foreground 1 px（最小クロップ）
    one = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    one.putpixel((32, 32), (255, 0, 0, 255))
    save("one_pixel", one, True)

    # 6. 最小 alpha 252（250 未満の画素が無い = pre-matted でない）→ 両実装とも拒否
    faint = Image.new("RGBA", (64, 64), (10, 10, 10, 252))
    ImageDraw.Draw(faint).rectangle([16, 16, 47, 47], fill=(255, 255, 255, 255))
    save("no_real_alpha", faint, False)

    return cases


def run_cpp(binary: Path, src: Path, dst: Path) -> tuple[bool, str]:
    proc = subprocess.run([str(binary), "--crop", str(src), str(dst)],
                          capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, out


def run_python(src: Path, dst: Path) -> tuple[bool, str]:
    # preprocess_prematted_rgba は失敗時に fail() で SystemExit する。
    try:
        preprocess_prematted_rgba(src, dst)
    except SystemExit as exc:
        return False, f"rejected ({exc})"
    return True, "ok"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: test_crop_parity.py <path to trellis-test-sv-input>", file=sys.stderr)
        return 2
    binary = Path(sys.argv[1])
    if not binary.exists():
        print(f"missing C++ binary: {binary}", file=sys.stderr)
        return 2

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, src, should_accept in fixtures(tmp_path):
            cpp_dst = tmp_path / f"{name}.cpp.png"
            py_dst = tmp_path / f"{name}.py.png"
            cpp_ok, cpp_msg = run_cpp(binary, src, cpp_dst)
            py_ok, py_msg = run_python(src, py_dst)

            if cpp_ok != py_ok:
                print(f"FAIL {name}: accept/reject disagree (C++ {cpp_ok}: {cpp_msg} / Python {py_ok}: {py_msg})")
                failures += 1
                continue
            if cpp_ok != should_accept:
                print(f"FAIL {name}: expected {'accept' if should_accept else 'reject'}, got the opposite ({cpp_msg})")
                failures += 1
                continue
            if not cpp_ok:
                print(f"ok   {name}: both implementations reject it")
                continue

            a = np.asarray(Image.open(py_dst).convert("RGBA")).astype(np.int32)
            b = np.asarray(Image.open(cpp_dst).convert("RGBA")).astype(np.int32)
            if a.shape != b.shape:
                print(f"FAIL {name}: crop size differs (Python {a.shape[:2]} / C++ {b.shape[:2]}) — {cpp_msg}")
                failures += 1
                continue

            am, bm = a[..., 3] > 204, b[..., 3] > 204
            union = int((am | bm).sum())
            iou = float((am & bm).sum()) / union if union else 1.0
            pa = a[..., :3] * a[..., 3:4] / 255.0
            pb = b[..., :3] * b[..., 3:4] / 255.0
            mae = float(np.abs(pa - pb).mean()) / 255.0

            ok = iou >= ALPHA_IOU_MIN and mae <= RGB_MAE_MAX
            print(f"{'ok  ' if ok else 'FAIL'} {name}: size {a.shape[0]}, alpha IoU {iou:.6f}, premult RGB MAE {mae:.6f} ({cpp_msg})")
            if not ok:
                failures += 1

    print("\nCROP_PARITY_OK" if not failures else f"\nCROP_PARITY_FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
