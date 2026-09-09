#!/usr/bin/env python3
"""Compare two per-stage sparse-decoder dumps (TRELLIS_DBG_STAGE_DUMP in src/shape_decoder.cpp,
or tools/ref_pixal3d_shape_dec_stages.py for the PyTorch reference): exact coordinate-set parity
per stage, then feature parity on the coordinate-hashed row subset both dumps share (rows are
matched BY COORDINATE, so the two decoders may order their voxels differently).

    python3 tools/compare_sparse_stages.py <dir_a> <dir_b> [--kind shape_dec] [--tol 5e-3] [--coord-tol 1e-3]

Per stage: |A|, |B|, common / A-only / B-only coordinates, and on the common subset rows
max|d|, mean|d|, rel (max|d| / max|B|), cosine, and nrm = mean|d| / std(B) (whole-tensor std).
Gate (exit 1 otherwise): per stage, (A-only + B-only) / |B| <= --coord-tol and nrm <= --tol.
Why not exact coords / max-rel: every C2S up-block thresholds its subdivision logits at 0, so a
voxel whose logit sits within the torso's precision of 0 flips between implementations, and each
flipped voxel perturbs its 26 neighbours' next conv (a present vs. absent = zero row), which is
where the localized max|d| comes from. Calibration on the Shape-1024 real-SLAT fixture
(docs/PIXAL3D_WEBGPU_MEMORY.md §9): native CUDA vs the PyTorch fp16 reference flips 0.027 % of
the final coords (1277/1268 of 4.65M) with nrm 1.6e-3 at the output; the defaults (0.1 %, 5e-3)
sit at ~3x that envelope.
"""
import sys, os, glob
import numpy as np

STAGES = ["from_latent", "stage0_convnext", "stage0_c2s", "stage1_convnext", "stage1_c2s",
          "stage2_convnext", "stage2_c2s", "stage3_convnext", "stage3_c2s", "output"]

def key(c):
    c = c.astype(np.int64)
    return (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]

def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    kind = "shape_dec"; tol = 5e-3; coord_tol = 1e-3
    args = sys.argv[3:]
    for i, a in enumerate(args):
        if a == "--kind": kind = args[i + 1]
        if a == "--tol": tol = float(args[i + 1])
        if a == "--coord-tol": coord_tol = float(args[i + 1])
    ok = True
    print(f"{'stage':18s} {'|A|':>9s} {'|B|':>9s} {'common':>9s} {'A-only':>7s} {'B-only':>7s} {'flip%':>7s} | {'rows':>5s} {'max|d|':>9s} {'mean|d|':>9s} {'rel':>9s} {'nrm':>9s} {'cos':>9s}")
    for st in STAGES:
        ca = f"{a_dir}/{kind}_{st}_coords.npy"; cb = f"{b_dir}/{kind}_{st}_coords.npy"
        if not (os.path.exists(ca) and os.path.exists(cb)):
            print(f"{st:18s} (missing in {'A' if not os.path.exists(ca) else 'B'})"); continue
        A = np.load(ca); B = np.load(cb)
        ka, kb = key(A), key(B)
        sa, sb = set(ka.tolist()), set(kb.tolist())
        common = len(sa & sb); aonly = len(sa - sb); bonly = len(sb - sa)
        flip = (aonly + bonly) / max(len(B), 1)
        if flip > coord_tol: ok = False
        stb = np.load(f"{b_dir}/{kind}_{st}_stats.npy")
        # features on the shared subset, matched by coordinate
        fa = np.load(f"{a_dir}/{kind}_{st}_sub_feats.npy"); fb = np.load(f"{b_dir}/{kind}_{st}_sub_feats.npy")
        sca = key(np.load(f"{a_dir}/{kind}_{st}_sub_coords.npy")); scb = key(np.load(f"{b_dir}/{kind}_{st}_sub_coords.npy"))
        ia = {int(k): i for i, k in enumerate(sca.tolist())}
        rows = [(ia[int(k)], j) for j, k in enumerate(scb.tolist()) if int(k) in ia]
        if rows:
            ra = np.array([r[0] for r in rows]); rb = np.array([r[1] for r in rows])
            x = fa[ra].astype(np.float64); y = fb[rb].astype(np.float64)
            d = np.abs(x - y); mx = d.max(); mean = d.mean()
            rel = mx / max(np.abs(y).max(), 1e-30)
            cos = float((x * y).sum() / max(np.sqrt((x * x).sum() * (y * y).sum()), 1e-30))
            nrm = mean / max(float(stb[2]), 1e-30)
            if nrm > tol: ok = False
            print(f"{st:18s} {len(A):9d} {len(B):9d} {common:9d} {aonly:7d} {bonly:7d} {100*flip:7.3f} | {len(rows):5d} {mx:9.3e} {mean:9.3e} {rel:9.3e} {nrm:9.3e} {cos:9.6f}")
        else:
            ok = False
            print(f"{st:18s} {len(A):9d} {len(B):9d} {common:9d} {aonly:7d} {bonly:7d} {100*flip:7.3f} | no shared subset rows")
        sta = np.load(f"{a_dir}/{kind}_{st}_stats.npy")
        print(f"{'':18s}   whole-tensor stats A: n={int(sta[0])} mean={sta[1]:.5f} std={sta[2]:.5f} absmax={sta[3]:.4f} | B: n={int(stb[0])} mean={stb[1]:.5f} std={stb[2]:.5f} absmax={stb[3]:.4f}")
    print("RESULT:", "PASS" if ok else "FAIL", f"(per stage: coord flips <= {100*coord_tol:g} % and mean|d|/std <= {tol:g})")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
