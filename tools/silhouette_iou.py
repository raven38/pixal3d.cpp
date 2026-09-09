#!/usr/bin/env python3
"""生成した GLB を入力 multiview のカメラで投影し、入力 alpha とのシルエット IoU を出す。

    python3 tools/silhouette_iou.py <mesh.glb> <views_dir> [out_montage.png]
        [--res 256] [--min-iou 0.85] [--max-scale-error 0.05]

検収に使うとき（--min-iou / --max-scale-error のどちらかを渡すと判定モード）は、
閾値を割ると終了コード 1 を返す。生成が「完走した」だけでは正しさの証拠にならない
（2026-09-08: mesh_scale の欠落で IoU 0.107 の壊れた GLB が非空で出ていた）ので、
パイプラインの最後にこれを噛ませて数値で落とす。

views_dir は transforms.json + pre-matted RGBA。カメラは近似せず、
src/proj_grid.cpp と同じ式で投影する:

    focal_px = S/2 / tan(fov_x/2)
    p_cam    = inv(c2w) @ p_world
    x_pix    =  focal_px * xc / (-zc) + S/2
    y_pix    = -focal_px * yc / (-zc) + S/2

GLB の軸と ProjGrid world 軸の対応は、`proj_grid_points` の回転 (x0,-z0,y0) と
`apply_reference_frame` の (-x,z,y) を合成しても実測と合わなかったため、**符号付き軸置換
48 通りを総当たりして決める**（--frame で固定も可）。形が違えばどの置換でも一致しないので、
この探索で判定が甘くなることはない。選ばれた置換は出力に出る。
"""
import json, struct, sys
import numpy as np
from PIL import Image


def load_glb_positions(path):
    d = open(path, "rb").read()
    jl = struct.unpack_from("<I", d, 12)[0]
    J = json.loads(d[20:20 + jl])
    p = 20 + jl
    BIN = d[p + 8:p + 8 + struct.unpack_from("<I", d, p)[0]]
    prim = J["meshes"][0]["primitives"][0]

    def acc(i, dt, nc):
        a = J["accessors"][i]
        bv = J["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(BIN, dtype=dt, count=a["count"] * nc, offset=off).reshape(-1, nc)

    P = acc(prim["attributes"]["POSITION"], "<f4", 3).astype(np.float64)
    ia = J["accessors"][prim["indices"]]
    dt = {5125: "<u4", 5123: "<u2"}[ia["componentType"]]
    I = acc(prim["indices"], dt, 1).reshape(-1, 3).astype(np.int64)
    return P, I


def splat(P2, res, r=1):
    """頂点を画素へ落としてシルエットにする。メッシュが密（数十万頂点）なので、
    三角形ラスタライズより桁違いに速く、同じ形が出る。r は穴埋めの膨張半径。"""
    x = np.round(P2[:, 0]).astype(np.int64)
    y = np.round(P2[:, 1]).astype(np.int64)
    ok = (x >= 0) & (x < res) & (y >= 0) & (y < res)
    m = np.zeros((res, res), bool)
    m[y[ok], x[ok]] = True
    if r > 0:  # dilate -> erode（穴を閉じるが外形は変えない）
        d = m.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d |= np.roll(np.roll(m, dy, 0), dx, 1)
        e = d.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                e &= np.roll(np.roll(d, dy, 0), dx, 1)
        m = e
    return m


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = 256
    for a in sys.argv[1:]:
        if a.startswith("--res"):
            res = int(a.split("=", 1)[1])
    glb, views_dir = args[0], args[1]
    out_png = args[2] if len(args) > 2 else None

    meta = json.load(open(f"{views_dir}/transforms.json"))
    frames = meta["frames"]
    ms = float(meta.get("mesh_scale", 1.0))
    for a in sys.argv[1:]:
        if a.startswith("--mesh-scale"):
            ms = float(a.split("=", 1)[1])
            print(f"[override] mesh_scale = {ms}")

    P, I = load_glb_positions(glb)

    # 入力側のマスクとカメラを先に用意
    cams, refs = [], []
    for fr in frames:
        im = Image.open(f"{views_dir}/{fr['file_path']}").convert("RGBA")
        S = im.size[0]
        refs.append(np.array(im.getchannel(3).resize((res, res), Image.NEAREST)) > 0)
        fov = float(fr.get("camera_angle_x", meta.get("camera_angle_x")))
        cams.append((np.linalg.inv(np.array(fr["transform_matrix"], dtype=np.float64)),
                     (S / 2.0) / np.tan(fov / 2.0), S))

    def project(W, cam, res):
        w2c, focal, S = cam
        pc = W @ w2c[:3, :3].T + w2c[:3, 3]
        zc = pc[:, 2]
        xp = focal * pc[:, 0] / (-zc + 1e-8) + S / 2.0
        yp = -focal * pc[:, 1] / (-zc + 1e-8) + S / 2.0
        return splat(np.stack([xp, yp], 1) * (res / S), res)

    # 符号付き軸置換 48 通り（右手系 24 + 鏡像 24）
    import itertools
    frames_all = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            frames_all.append((perm, sg))
    fixed = None
    for a in sys.argv[1:]:
        if a.startswith("--frame"):
            fixed = eval(a.split("=", 1)[1])

    def world_of(fr_):
        perm, sg = fr_
        return np.stack([sg[i] * P[:, perm[i]] for i in range(3)], 1) / ms

    if fixed is not None:
        best = (fixed, None)
    else:
        # 粗探索: 低解像度で全 48 を評価してから本解像度で確定する
        lo = 96
        refs_lo = [np.array(Image.fromarray(r.astype(np.uint8) * 255).resize((lo, lo), Image.NEAREST)) > 127
                   for r in refs]
        scored = []
        for fr_ in frames_all:
            W = world_of(fr_)
            sc = []
            for cam, rf in zip(cams, refs_lo):
                m = project(W, cam, lo)
                u = (m | rf).sum()
                sc.append(float((m & rf).sum() / u) if u else 0.0)
            scored.append((float(np.mean(sc)), fr_))
        scored.sort(reverse=True)
        print("軸置換の総当たり（上位5件, 96px 粗探索の平均 IoU）:")
        for v, fr_ in scored[:5]:
            print(f"  perm={fr_[0]} sign={fr_[1]}  mean IoU={v:.4f}")
        best = (scored[0][1], scored[0][0])
        print()

    perm, sg = best[0]
    W = world_of(best[0])
    print(f"glb V={len(P)} F={len(I)}  mesh_scale={ms}  res={res}  frame: perm={perm} sign={sg}")
    print("| view | IoU | 生成の被覆率 | 入力の被覆率 |")
    print("|---|---|---|---|")
    ious, mine = [], []
    for fr, cam, ref in zip(frames, cams, refs):
        m = project(W, cam, res)
        u = (m | ref).sum()
        v = float((m & ref).sum() / u) if u else 0.0
        ious.append(v); mine.append(m)
        print(f"| {fr.get('name', fr['file_path'])} | {v:.4f} | {m.mean():.3f} | {ref.mean():.3f} |")
    mean_iou = float(np.mean(ious))
    print(f"\nmean silhouette IoU: {mean_iou:.4f}")

    # 投影サイズの比から mesh_scale を逆算する。world = grid/(2*mesh_scale) なので
    # 投影の大きさは 1/mesh_scale に比例する。transforms.json の mesh_scale が
    # 実際のレンダと食い違っていると、ここが 1.0 から外れる。
    def extent(m):
        ys, xs = np.nonzero(m)
        return 0.0 if ys.size == 0 else float(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1)
    ratios = [extent(m) / extent(r) for m, r in zip(mine, refs) if extent(r) > 0 and extent(m) > 0]
    if ratios:
        implied = ms * float(np.mean(ratios))
        print(f"投影サイズ比 mine/ref = {np.mean(ratios):.3f} (view ごと: "
              + ", ".join(f"{x:.3f}" for x in ratios) + ")")
        print(f"=> 入力と整合する mesh_scale の推定値: {implied:.3f}  "
              f"(transforms.json の値: {ms})")

    if out_png:
        rows = [np.concatenate([np.repeat((x[:, :, None] * 255).astype(np.uint8), 3, 2) for x in r], 1)
                for r in (refs, mine)]
        Image.fromarray(np.concatenate(rows, 0)).save(out_png)
        print(f"saved {out_png}  (上段=入力 alpha, 下段=生成 GLB を同じカメラで投影)")

    # 判定モード: 閾値を渡されたときだけ終了コードで落とす。
    min_iou = max_scale_err = None
    for a in sys.argv[1:]:
        if a.startswith("--min-iou"):
            min_iou = float(a.split("=", 1)[1]) if "=" in a else 0.85
        if a.startswith("--max-scale-error"):
            max_scale_err = float(a.split("=", 1)[1]) if "=" in a else 0.05
    if min_iou is None and max_scale_err is None:
        return 0

    ok = True
    if min_iou is not None:
        good = mean_iou >= min_iou
        ok &= good
        print(f"GATE mean_iou {mean_iou:.4f} {'>=' if good else '<'} {min_iou:.4f}  "
              f"{'PASS' if good else 'FAIL'}")
    if max_scale_err is not None and ratios:
        # 投影サイズ比が 1 からどれだけ外れているか = mesh_scale の食い違い
        err = abs(float(np.mean(ratios)) - 1.0)
        good = err <= max_scale_err
        ok &= good
        print(f"GATE scale_error {err:.4f} {'<=' if good else '>'} {max_scale_err:.4f}  "
              f"{'PASS' if good else 'FAIL'}")
    print(f"SILHOUETTE_GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
