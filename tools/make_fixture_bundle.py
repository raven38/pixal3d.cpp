#!/usr/bin/env python3
"""fixture-input full E2E（pixal3d_full_fixture_run / web/full_e2e）が読む bundle を作る。

    python3 tools/make_fixture_bundle.py <views_dir> <out_dir> [--seed 1]

出力（src/pixal3d_full_e2e_wasm.cpp の fixture_views / deterministic_noise が読む名前）:

  images_512.npy      [1,V,3,512,512]  f32  アルファ乗算済み
  s1024_images.npy    [1,V,3,1024,1024]
  camera_angle_x.npy  [1,V]
  transform_matrix.npy[1,V,4,4]
  mesh_scale.npy      [1]

画像の前処理は inference_mv.py の `to_cond_tensor` と同じ **PIL LANCZOS + アルファ乗算**。
C++ の real-input 経路（src/pixal3d_input.cpp の resize_premult）は stb_image_resize を
使っていて実装が違うので、この bundle を使うと preprocessing が PyTorch 側に揃う。
その差自体を測りたいときは --compare <pytorch_dir> で PyTorch 生成の npy と比べる。

noise（noise / shape_noise / hr_shape_noise / tex_noise）は書かない。無ければ
deterministic_noise() が seed から mt19937 で作るので、そちらの方が native と揃う。
"""
import json, os, sys
import numpy as np
from PIL import Image


def to_cond_tensor(im: Image.Image, size: int) -> np.ndarray:
    """inference_mv.to_cond_tensor と同じ: LANCZOS resize -> RGB * alpha。返り値 [3,S,S]。"""
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(im.getchannel(3), dtype=np.float32) / 255.0
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return rgb * a[None]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    views_dir, out_dir = args[0], args[1]
    compare = None
    for a in sys.argv[1:]:
        if a.startswith("--compare"):
            compare = a.split("=", 1)[1]
    os.makedirs(out_dir, exist_ok=True)

    meta = json.load(open(f"{views_dir}/transforms.json"))
    frames = meta["frames"]

    def cax(fr):
        for src in (fr, meta):
            if "camera_angle_x" in src:
                return float(src["camera_angle_x"])
        raise KeyError(f"camera_angle_x missing for {fr.get('file_path')}")

    ims = []
    for fr in frames:
        im = Image.open(os.path.join(views_dir, fr["file_path"]))
        assert im.mode == "RGBA", f"{fr['file_path']}: pre-matted RGBA が要る"
        ims.append(im.convert("RGBA"))

    out = {
        "images_512": np.stack([to_cond_tensor(im, 512) for im in ims])[None],
        "s1024_images": np.stack([to_cond_tensor(im, 1024) for im in ims])[None],
        "camera_angle_x": np.array([[cax(fr) for fr in frames]], dtype=np.float32),
        "transform_matrix": np.array([[fr["transform_matrix"] for fr in frames]], dtype=np.float32),
        "mesh_scale": np.array([float(meta.get("mesh_scale", 1.0))], dtype=np.float32),
    }
    for k, v in out.items():
        v = np.ascontiguousarray(v, dtype=np.float32)
        np.save(f"{out_dir}/{k}.npy", v)
        print(f"  {k:20s} {str(list(v.shape)):26s} mean={v.mean():.6f} std={v.std():.6f}")
        assert np.isfinite(v).all(), k
    print(f"wrote {out_dir}  (V={len(frames)} mesh_scale={out['mesh_scale'][0]})")

    if compare:
        print(f"\nPyTorch 生成物との比較 ({compare}):")
        for k in out:
            p = f"{compare}/{k}.npy"
            if not os.path.exists(p):
                print(f"  {k:20s} (無し)")
                continue
            a = np.load(p).astype(np.float64).ravel()
            b = np.asarray(out[k], dtype=np.float64).ravel()
            if a.size != b.size:
                print(f"  {k:20s} SIZE {a.size} vs {b.size}")
                continue
            d = np.abs(a - b)
            rel = d.max() / (np.abs(a).max() or 1)
            print(f"  {k:20s} max|d|={d.max():.3e} mean|d|={d.mean():.3e} "
                  f"rel={rel:.3e} bit={'yes' if np.array_equal(a, b) else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
