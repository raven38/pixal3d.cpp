#!/usr/bin/env python3
"""TRELLIS_DUMP_SLAT の 2 つ（以上）のダンプディレクトリを flow 段ごとに比較する。

各 <dir>/<name>.bin は trellis_cli.cpp::dump_slat_env の形式:
  i32 N, i32 C, i32 has_coords, i32 coords[N*3]（has_coords のとき）, f32 feats[N*C]（channel-major）
name は ss_latent / ss_logits / ss_coords / lr_slat / hr_slat / tex_slat。

比較は「同じ voxel 集合か」を先に判定してから要素比較する（coords が違えば要素の対応が無いので
要素比較は拒み、集合の差だけ出す）。`--ref <dir>` を oracle（例: --no-fa）にすると、
各候補の |cand − ref| を並べて「どちらが oracle に近いか」を読める。

使い方:
  tools/compare_slat_dumps.py --ref <dir_oracle> <dir_a> [<dir_b> ...]
  tools/compare_slat_dumps.py <dir_a> <dir_b>          # ref 無しなら a を基準に b を比べる
"""
import argparse
import os
import sys
import numpy as np

NAMES = ["ss_latent", "ss_logits", "ss_coords", "lr_slat", "hr_slat", "tex_slat"]


def load(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=3)
        if hdr.size < 3:
            raise ValueError(f"{path}: short header")
        N, C, has_coords = (int(x) for x in hdr)
        coords = np.fromfile(f, dtype=np.int32, count=N * 3).reshape(N, 3) if has_coords else None
        feats = np.fromfile(f, dtype=np.float32, count=N * C).reshape(N, C) if C > 0 else None
    return N, C, coords, feats


def coords_key(coords):
    return set(map(tuple, coords.tolist())) if coords is not None else None


def compare(name, ref, cand):
    """ref / cand: (N, C, coords, feats)。同一 voxel 集合なら要素統計、違えば集合差を返す。"""
    Nr, Cr, cr, fr = ref
    Nc, Cc, cc, fc = cand
    if cr is not None or cc is not None:
        kr, kc = coords_key(cr), coords_key(cc)
        if kr != kc:
            inter = len(kr & kc) if kr and kc else 0
            return dict(same_set=False, N_ref=Nr, N_cand=Nc, inter=inter,
                        only_ref=len(kr - kc), only_cand=len(kc - kr))
        # 同じ集合でも並びが違うかもしれないので coords で揃える
        if fr is not None and fc is not None and not np.array_equal(cr, cc):
            order_r = np.lexsort(cr.T[::-1]); order_c = np.lexsort(cc.T[::-1])
            fr, fc = fr[order_r], fc[order_c]
    if fr is None or fc is None:
        return dict(same_set=True, N_ref=Nr, N_cand=Nc, elementwise=False)
    if fr.shape != fc.shape:
        return dict(same_set=False, N_ref=Nr, N_cand=Nc, shape_ref=fr.shape, shape_cand=fc.shape)
    d = fc.astype(np.float64) - fr.astype(np.float64)
    finite = np.isfinite(d)
    rms_ref = float(np.sqrt(np.mean(fr.astype(np.float64)[finite] ** 2))) if finite.any() else 0.0
    return dict(same_set=True, N_ref=Nr, N_cand=Nc, elementwise=True,
                max_abs=float(np.abs(d[finite]).max()) if finite.any() else float("nan"),
                rms=float(np.sqrt(np.mean(d[finite] ** 2))) if finite.any() else float("nan"),
                rel_rms=(float(np.sqrt(np.mean(d[finite] ** 2))) / rms_ref) if rms_ref > 0 else float("nan"),
                mean_delta=float(d[finite].mean()) if finite.any() else float("nan"),
                nonfinite_ref=int((~np.isfinite(fr)).sum()), nonfinite_cand=int((~np.isfinite(fc)).sum()),
                bit_exact=bool(np.array_equal(fr, fc)))


def fmt(r):
    if not r.get("same_set", False):
        if "inter" in r:
            return (f"DIFFERENT voxel set: N_ref={r['N_ref']} N_cand={r['N_cand']} inter={r['inter']} "
                    f"only_ref={r['only_ref']} only_cand={r['only_cand']}  (no elementwise comparison)")
        return f"shape mismatch ref={r.get('shape_ref')} cand={r.get('shape_cand')}"
    if not r.get("elementwise", False):
        return f"same voxel set (N={r['N_ref']}), no feats"
    return (f"same set N={r['N_ref']}  max|d|={r['max_abs']:.4g}  rms(d)={r['rms']:.4g}  "
            f"rel_rms={r['rel_rms']:.3e}  mean(d)={r['mean_delta']:+.3e}  "
            f"nonfinite ref/cand={r['nonfinite_ref']}/{r['nonfinite_cand']}" +
            ("  [bit-exact]" if r["bit_exact"] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", help="oracle dump dir (e.g. the --no-fa run)")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    ref_dir = a.ref or a.dirs[0]
    cands = a.dirs if a.ref else a.dirs[1:]
    if not cands:
        print("need at least one candidate dir"); return 2
    print(f"ref: {ref_dir}")
    for name in NAMES:
        rp = os.path.join(ref_dir, name + ".bin")
        if not os.path.exists(rp):
            continue
        ref = load(rp)
        print(f"[{name}] ref N={ref[0]} C={ref[1]}")
        for cd in cands:
            cp = os.path.join(cd, name + ".bin")
            if not os.path.exists(cp):
                print(f"    {cd}: (missing)"); continue
            print(f"    {cd}: {fmt(compare(name, ref, load(cp)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
