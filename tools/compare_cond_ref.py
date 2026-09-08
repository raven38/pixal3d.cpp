#!/usr/bin/env python3
"""Pixal3D conditioning の PyTorch 参照と C++ 出力を突き合わせる。

判定は src/test_pixal3d_cond_slat.cpp の compare() と同じ定義:
    rel = max|d| / max|ref|   （絶対値の max|d| ではない）
既定 tol は同じく 2e-2。あわせて mean abs / L2 相対 / cosine / bit 一致も出す。

    python3 tools/compare_cond_ref.py <ref_dir> <cpp_dir_or_prefix> [--tol 2e-2]

ref_dir  : tools/ref_pixal3d_cond_tex.py の出力（tex_cond_global.npy / tex_cond_proj.npy）
cpp      : 同名の .npy を持つディレクトリ（PIXAL3D_DUMP_FIXTURE の出力）か、
           trellis-test-pixal3d-cond-tex --save-prefix P の P（P_global.npy / P_proj.npy）

proj は [N, 2*D] の [lr || hr] なので、lr / hr / 全体を分けて出す。
"""
import os, sys
import numpy as np


def load_pair(path):
    """ディレクトリ（fixture 形式）と --save-prefix の両方を受ける。"""
    if os.path.isdir(path):
        return (np.load(os.path.join(path, "tex_cond_global.npy")),
                np.load(os.path.join(path, "tex_cond_proj.npy")))
    return np.load(path + "_global.npy"), np.load(path + "_proj.npy")


def row(name, a, b, tol):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        print(f"| {name} | SIZE MISMATCH {a.size} vs {b.size} |")
        return False
    d = np.abs(a - b)
    refmax = np.abs(a).max()
    rel = d.max() / refmax if refmax > 0 else d.max()
    l2 = np.linalg.norm(a - b) / np.linalg.norm(a)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    ok = rel < tol
    print(f"| {name} | {d.max():.3e} | {d.mean():.3e} | {rel:.3e} {'PASS' if ok else 'FAIL'} | "
          f"{l2:.3e} | {cos:.10f} | {'yes' if np.array_equal(a, b) else 'no'} |")
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tol = 2e-2
    for a in sys.argv[1:]:
        if a.startswith("--tol"):
            tol = float(a.split("=", 1)[1]) if "=" in a else tol
    if len(args) < 2:
        print(__doc__)
        return 2
    rg, rp = load_pair(args[0])
    cg, cp = load_pair(args[1])
    rp = rp.reshape(rp.shape[0], -1)
    cp = cp.reshape(cp.shape[0], -1)
    D = rp.shape[1] // 2
    print(f"ref  global {rg.shape} proj {rp.shape}   finite {np.isfinite(rg).all()}/{np.isfinite(rp).all()}")
    print(f"cpp  global {cg.shape} proj {cp.shape}   finite {np.isfinite(cg).all()}/{np.isfinite(cp).all()}")
    print(f"\n| tensor | max abs | mean abs | rel (tol {tol:.0e}) | L2 rel | cosine | bit identical |")
    print("|---|---|---|---|---|---|---|")
    ok = row("global", rg, cg, tol)
    ok &= row(f"proj lr [:{D}]", rp[:, :D], cp[:, :D], tol)
    ok &= row(f"proj hr [{D}:]", rp[:, D:], cp[:, D:], tol)
    ok &= row("proj 全体", rp, cp, tol)
    print(f"\nCOND_REF: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
