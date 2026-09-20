#!/usr/bin/env python3
"""legacy RoPE / new RoPE の GLB を比較する。
V/F・bbox・重心・頂点の最近傍距離（両方向、サンプリング）と、UV/テクスチャの基本統計を出す。
使い方: python3 compare_rope_glb.py <legacy.glb> <new.glb>"""
import json
import struct
import sys

import numpy as np


def load(path):
    d = open(path, "rb").read()
    assert d[:4] == b"glTF"
    jl = struct.unpack_from("<I", d, 12)[0]
    J = json.loads(d[20:20 + jl])
    p = 20 + jl
    bl = struct.unpack_from("<I", d, p)[0]
    BIN = d[p + 8:p + 8 + bl]

    def acc(i):
        a = J["accessors"][i]
        bv = J["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt, _ = {5126: ("<f4", 4), 5125: ("<u4", 4), 5123: ("<u2", 2)}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        return np.frombuffer(BIN, dtype=dt, count=a["count"] * nc, offset=off).reshape(a["count"], nc)

    prim = J["meshes"][0]["primitives"][0]
    P = acc(prim["attributes"]["POSITION"]).astype(np.float64)
    UV = acc(prim["attributes"]["TEXCOORD_0"]).astype(np.float64)
    I = acc(prim["indices"]).reshape(-1, 3).astype(np.int64)
    img = J["images"][0]
    bv = J["bufferViews"][img["bufferView"]]
    o = bv.get("byteOffset", 0)
    tex_bytes = BIN[o:o + bv["byteLength"]]
    return P, UV, I, tex_bytes, J


def nn_dist(A, B, n=20000, seed=0):
    """A の n 点サンプルから B への最近傍距離（グリッドハッシュ、B は全点）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(A), size=min(n, len(A)), replace=False)
    Q = A[idx]
    lo = B.min(0)
    cell = max((B.max(0) - lo).max() / 256.0, 1e-9)
    keys = {}
    ij = np.floor((B - lo) / cell).astype(np.int64)
    for k, key in enumerate(map(tuple, ij)):
        keys.setdefault(key, []).append(k)
    out = np.empty(len(Q))
    qij = np.floor((Q - lo) / cell).astype(np.int64)
    for t, (q, c) in enumerate(zip(Q, qij)):
        best = np.inf
        r = 1
        while True:
            cand = []
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        cand += keys.get((c[0] + dx, c[1] + dy, c[2] + dz), [])
            if cand:
                d = np.linalg.norm(B[cand] - q, axis=1).min()
                best = min(best, d)
                if best <= (r - 1) * cell or r >= 4:
                    break
            r += 1
            if r > 6:
                break
        out[t] = best
    return out


def report(tag, P, UV, I, tex):
    mn, mx = P.min(0), P.max(0)
    print(f"{tag}: V={len(P)} F={len(I)} tex={len(tex)}B")
    print(f"{tag}: bbox min=({mn[0]:.5f},{mn[1]:.5f},{mn[2]:.5f}) max=({mx[0]:.5f},{mx[1]:.5f},{mx[2]:.5f})")
    print(f"{tag}: extents=({mx[0]-mn[0]:.5f},{mx[1]-mn[1]:.5f},{mx[2]-mn[2]:.5f}) centroid=({P.mean(0)[0]:.5f},{P.mean(0)[1]:.5f},{P.mean(0)[2]:.5f})")
    print(f"{tag}: UV in [{UV.min():.4f},{UV.max():.4f}] nonfinite verts={np.count_nonzero(~np.isfinite(P))}")


a, b = sys.argv[1], sys.argv[2]
Pa, UVa, Ia, Ta, _ = load(a)
Pb, UVb, Ib, Tb, _ = load(b)
report("legacy", Pa, UVa, Ia, Ta)
report("new   ", Pb, UVb, Ib, Tb)
ea = Pa.max(0) - Pa.min(0)
eb = Pb.max(0) - Pb.min(0)
print(f"extent rel diff = {np.abs(eb - ea) / np.maximum(ea, 1e-9)}")
d_ab = nn_dist(Pa, Pb)
d_ba = nn_dist(Pb, Pa)
diag = float(np.linalg.norm(ea))
print(f"nn legacy->new : mean {d_ab.mean():.6f} p95 {np.percentile(d_ab,95):.6f} max {d_ab.max():.6f}  ({100*d_ab.mean()/diag:.4f}% of bbox diag {diag:.4f})")
print(f"nn new->legacy : mean {d_ba.mean():.6f} p95 {np.percentile(d_ba,95):.6f} max {d_ba.max():.6f}")
print(f"chamfer-ish (mean of both) = {0.5*(d_ab.mean()+d_ba.mean()):.6f}")
