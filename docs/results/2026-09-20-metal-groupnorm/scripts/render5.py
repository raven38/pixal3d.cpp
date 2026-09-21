#!/usr/bin/env python3
"""2 つの GLB を front/back/left/right/top の 5 視点で並置レンダする（上段 A・下段 B、同一カメラ）。
render_glb_fast.py と同じ簡易ラスタ（面サブサンプル + z-buffer + UV テクスチャ）。
使い方: render5.py <A.glb> <B.glb> <out.png> [maxfaces]"""
import sys, struct, json, io
import numpy as np
from PIL import Image, ImageDraw

def load_glb(path):
    d = open(path, "rb").read()
    assert d[:4] == b"glTF"; jl = struct.unpack_from("<I", d, 12)[0]; J = json.loads(d[20:20 + jl])
    p = 20 + jl; bl = struct.unpack_from("<I", d, p)[0]; BIN = d[p + 8:p + 8 + bl]
    def acc(i):
        a = J["accessors"][i]; bv = J["bufferViews"][a["bufferView"]]; off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        ct = {5126: ("<f4", 4), 5125: ("<u4", 4), 5123: ("<u2", 2)}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        return np.frombuffer(BIN, dtype=ct[0], count=a["count"] * nc, offset=off).reshape(a["count"], nc)
    prim = J["meshes"][0]["primitives"][0]
    P = acc(prim["attributes"]["POSITION"]).astype(float)
    UV = acc(prim["attributes"]["TEXCOORD_0"]).astype(float)
    I = acc(prim["indices"]).reshape(-1, 3).astype(int)
    img = J["images"][0]; bv = J["bufferViews"][img["bufferView"]]; o = bv.get("byteOffset", 0)
    tex = np.array(Image.open(io.BytesIO(BIN[o:o + bv["byteLength"]])).convert("RGB"))
    return P, UV, I, tex

# 視点: y 軸回転角 + top は x 軸回転
VIEWS = [("front", 0, 0), ("right", 90, 0), ("back", 180, 0), ("left", 270, 0), ("top", 0, 90)]
R = 360

def render(P, UV, I, tex, ry, rx, c, s):
    TH, TW = tex.shape[:2]
    a = np.radians(ry); Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    b = np.radians(rx); Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Q = (P - c) @ Ry.T @ Rx.T
    sx = (Q[:, 0] / s * 0.9 + 0.5) * R; sy = (-Q[:, 1] / s * 0.9 + 0.5) * R; sz = Q[:, 2]
    img = np.zeros((R, R, 3), np.uint8); zb = np.full((R, R), -1e9)
    order = np.argsort(Q[I].mean(1)[:, 2])
    for fi in order:
        t = I[fi]; xs = sx[t]; ys = sy[t]; zs = sz[t]; uvs = UV[t]
        x0 = int(max(0, np.floor(xs.min()))); x1 = int(min(R - 1, np.ceil(xs.max())))
        y0 = int(max(0, np.floor(ys.min()))); y1 = int(min(R - 1, np.ceil(ys.max())))
        if x1 < x0 or y1 < y0: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(d) < 1e-9: continue
        w1 = ((gx - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (gy - ys[0])) / d
        w2 = ((xs[1] - xs[0]) * (gy - ys[0]) - (gx - xs[0]) * (ys[1] - ys[0])) / d
        w0 = 1 - w1 - w2
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any(): continue
        z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        u = w0 * uvs[0, 0] + w1 * uvs[1, 0] + w2 * uvs[2, 0]; v = w0 * uvs[0, 1] + w1 * uvs[1, 1] + w2 * uvs[2, 1]
        sub = zb[y0:y1 + 1, x0:x1 + 1]; upd = m & (z > sub)
        if not upd.any(): continue
        tu = np.clip((u * TW).astype(int), 0, TW - 1); tv = np.clip((v * TH).astype(int), 0, TH - 1)
        col = tex[tv, tu]
        sub[upd] = z[upd]; img[y0:y1 + 1, x0:x1 + 1][upd] = col[upd]
    return img

A = load_glb(sys.argv[1]); B = load_glb(sys.argv[2]); out = sys.argv[3]
maxf = int(sys.argv[4]) if len(sys.argv) > 4 else 300000
rng = np.random.default_rng(0)
# 共通カメラ: 両者の bbox の和で正規化
allP = np.vstack([A[0], B[0]]); mn, mx = allP.min(0), allP.max(0); c = (mn + mx) / 2; s = (mx - mn).max()
rows = []
for name, (P, UV, I, tex) in (("A", A), ("B", B)):
    if len(I) > maxf: I = I[rng.choice(len(I), maxf, replace=False)]
    tiles = [render(P, UV, I, tex, ry, rx, c, s) for (_, ry, rx) in VIEWS]
    rows.append(np.hstack(tiles))
grid = np.vstack(rows)
im = Image.fromarray(grid); dr = ImageDraw.Draw(im)
for k, (name, _, _) in enumerate(VIEWS):
    dr.text((k * R + 6, 4), name, fill=(255, 255, 0))
dr.text((6, 20), "A: " + sys.argv[1].split("/")[-1], fill=(0, 255, 255)); dr.text((6, R + 20), "B: " + sys.argv[2].split("/")[-1], fill=(0, 255, 255))
im.save(out); print("saved", out, "A V/F", len(A[0]), len(A[2]), "B V/F", len(B[0]), len(B[2]))
