#!/usr/bin/env python3
"""P2-num: 新（lowering）/ 旧（--naf-native-gn）の conditioning 出力 (<arm>_<stage>_{global,proj}.npy) を比べる。
出力: max|d| / mean|d| / L2rel / cos。使い方: compare_cond.py <num_dir>"""
import sys, os, numpy as np
d = sys.argv[1]
for stage in ("tex", "hr", "lr"):
    for kind in ("global", "proj"):
        a = os.path.join(d, f"new_{stage}_{kind}.npy"); b = os.path.join(d, f"old_{stage}_{kind}.npy")
        if not (os.path.exists(a) and os.path.exists(b)): print(f"{stage} {kind}: missing"); continue
        x = np.load(a).astype(np.float64); y = np.load(b).astype(np.float64)
        if x.shape != y.shape: print(f"{stage} {kind}: shape {x.shape} vs {y.shape}"); continue
        dd = x - y; fin = np.isfinite(x).all() and np.isfinite(y).all()
        l2 = np.sqrt((dd**2).sum() / max((y**2).sum(), 1e-30)); cos = (x*y).sum() / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30)
        print(f"{stage:3s} {kind:6s} shape={x.shape} max|d|={np.abs(dd).max():.3e} mean|d|={np.abs(dd).mean():.3e} "
              f"L2rel={l2:.3e} cos={cos:.8f} refmax={np.abs(y).max():.3e} finite={fin}")
