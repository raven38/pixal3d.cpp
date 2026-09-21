#!/usr/bin/env python3
"""B3-REF R1: 参照(PyTorch)側 stochastic mode 黒化率シード掃引。

設計: docs/design/2026-09-22-trellis2-mv-b3-ref-seed-sweep.md
背景: ~/Downloads/pixal3d-mv-hardening/docs/results/2026-09-21-trellis2-mv-b3/HANDOVER.md
      （native CUDA で V=4 stochastic tex が seed 依存で黒化、6 seed 中 2 本）

`tools/ref_trellis2_mv_pod_run_v3.py` は使わない（全12step x_t capture + 自己検証込みの
フル計装で 1 run ≈ 464〜486 秒（実測、v3 fixture manifest.json の elapsed_sec）。
23 run を回すと約3時間かかり pod 予算を超える）。本スクリプトは計装を全て外し、
黒化率の分類に必要な最終出力の統計だけを見る軽量版。v2/v3 の出力ディレクトリ
（/nfs/pixal3d_trellis2mv_ref_v2, v3）・スクリプト自体は一切変更・書き込みしない。

黒判定の定義と較正（着手前に実測、docs/design 参照）:
  - `/nfs/repos/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py::decode_tex_slat`:
    `ret = tex_slat_decoder(...) * 0.5 + 0.5` を実ソースで確認済み。よって
    `pipeline.decode_latent()` が返す `mesh.attrs` は既に [0,1] 値。
  - `docs/spec/12-ovoxel_mesh.md:62`: `base_color = clip(attrs[...,0:3]*255,0,255)`。
  - 既存 v3 fixture（seed=42）の実際に bake された `mesh.glb` テクスチャを
    `trimesh` で実測し、この voxel-space proxy と同じ値域であることを確認済み:
    4view stochastic=13.02（正常域）、4view multidiffusion=9.85（正常域下限）、
    2view stochastic=**0.0**（bake後テクスチャが全texel 0 = 完全に黒。
    brief 記載の「参照側は各1seedで全て正常」は誤りだったと判明した一次証拠）。

実行例（pod、1プロセスで全 config×seed を回す）:
    HF_HOME=/nfs/.cache/huggingface HF_HUB_OFFLINE=1 \
    /nfs/envs/miniforge3/envs/trellis2/bin/python ref_trellis2_mv_seed_sweep.py \
        --out /nfs/trellis2-mv/b3-ref-sweep

self-test（GPU/重み不要）:
    python ref_trellis2_mv_seed_sweep.py --selftest
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/nfs/repos/TRELLIS.2")
# ref_trellis2_mv_inject.py を v2 と共有する（読み出し専用の再利用。v2ファイル自体は変更しない）
sys.path.insert(0, "/nfs/pixal3d_trellis2mv_ref_v2/scripts")

PINNED_MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
DEFAULT_MODEL_PATH = (
    "/nfs/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/"
    + PINNED_MODEL_REVISION)
DEFAULT_IMAGES_4V = ("/nfs/pixal3d_trellis2mv_ref_v2/inputs/yoimiya_4view/Frame1.png,"
                      "/nfs/pixal3d_trellis2mv_ref_v2/inputs/yoimiya_4view/Frame2.png,"
                      "/nfs/pixal3d_trellis2mv_ref_v2/inputs/yoimiya_4view/Frame3.png,"
                      "/nfs/pixal3d_trellis2mv_ref_v2/inputs/yoimiya_4view/Frame4.png")
DEFAULT_IMAGES_2V = ("/nfs/pixal3d_trellis2mv_ref_v2/inputs/yoimiya_2view/Frame1.png,"
                      "/nfs/pixal3d_trellis2mv_ref_v2/inputs/yoimiya_2view/Frame4.png")

# v3 fixture（seed=42、docs/design 記載の既存run）との比較先。ハードコードの転記ミスを避けるため
# sha256値は埋め込まず、pod側でmanifest.jsonを直接読み込んで比較する（advisor指摘2の妥当性ゲート、
# bit一致の実績: docs/spec/33-trellis2-mv-reference.md「決定性の再検証」節）。
V3_MANIFEST_PATHS = {
    "4v_stochastic": "/nfs/pixal3d_trellis2mv_ref_v3/run_4img_real_stochastic_1024c/manifest.json",
    "2v_stochastic": "/nfs/pixal3d_trellis2mv_ref_v3/run_2img_real_stochastic_1024c/manifest.json",
    "4v_multidiffusion": "/nfs/pixal3d_trellis2mv_ref_v3/run_4img_real_multidiffusion_1024c/manifest.json",
}


def check_seed42_gate(cfg_name: str, ss_coords_sha256: str, tex_feats_sha256: str,
                       tex_coords_sha256: str) -> dict:
    """seed=42のrunがv3 fixtureとbit一致するかを検証する（妥当性ゲート）。
    不一致でもスイープは止めない（呼び出し側でログするだけ）。"""
    path = V3_MANIFEST_PATHS.get(cfg_name)
    if path is None or not os.path.exists(path):
        return {"checked": False, "reason": f"v3 manifest not found: {path}"}
    with open(path) as f:
        m = json.load(f)
    v3_ss_sha = m.get("coords", {}).get("sha256")
    v3_tex_feats_sha = m.get("tex_slat", {}).get("feats_sha256")
    v3_tex_coords_sha = m.get("tex_slat", {}).get("coords_sha256")
    ss_match = (v3_ss_sha == ss_coords_sha256)
    tex_feats_match = (v3_tex_feats_sha == tex_feats_sha256)
    tex_coords_match = (v3_tex_coords_sha == tex_coords_sha256)
    return {
        "checked": True,
        "ss_coords_match": ss_match,
        "tex_slat_feats_match": tex_feats_match,
        "tex_slat_coords_match": tex_coords_match,
        "all_bit_identical_to_v3": bool(ss_match and tex_feats_match and tex_coords_match),
    }


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def base_color_stats(attrs) -> dict:
    """mesh.attrs（[Mv,6], [0,1]値、decode_tex_slatで既に*0.5+0.5済み）から
    base color（RGB, ch0:3）の 0-255 スケール統計を計算する。
    docs/spec/12-ovoxel_mesh.md:62 の `base_color = clip(attrs[...,0:3]*255,0,255)` と同じ式。
    """
    import numpy as np
    a = attrs.detach().float().cpu().numpy() if hasattr(attrs, "detach") else np.asarray(attrs)
    rgb = np.clip(a[:, :3], 0.0, 1.0) * 255.0
    mean_0_255 = float(rgb.mean())
    std_0_255 = float(rgb.std())
    # saturation: 3ch とも [0,1]空間で <1/255 (ほぼ0) または >254/255 (ほぼ1) の voxel 比率
    ch01 = np.clip(a[:, :3], 0.0, 1.0)
    near0 = (ch01 < (1.0 / 255.0)).all(axis=1)
    near1 = (ch01 > (254.0 / 255.0)).all(axis=1)
    saturation_rate = float((near0 | near1).mean()) if ch01.shape[0] > 0 else float("nan")
    per_channel_mean = [float(a[:, c].mean()) for c in range(a.shape[1])]
    per_channel_std = [float(a[:, c].std()) for c in range(a.shape[1])]
    return {
        "base_color_mean_0_255": mean_0_255,
        "base_color_std_0_255": std_0_255,
        "saturation_rate": saturation_rate,
        "per_channel_mean_0_1": per_channel_mean,
        "per_channel_std_0_1": per_channel_std,
        "num_voxels": int(a.shape[0]),
    }


def classify(mean_0_255: float) -> str:
    """native 実績（HANDOVER: mean<2 が黒、10〜65 が正常）を踏襲した分類。
    中間値(2<=x<10)は二値化せず intermediate として報告する（advisor指摘、
    4view multidiffusion seed42=9.85 のような境界値の実例があるため）。
    """
    if mean_0_255 < 2.0:
        return "black"
    if mean_0_255 >= 10.0:
        return "normal"
    return "intermediate"


def fisher_exact_2x2(table) -> float:
    """2x2分割表の Fisher's exact test（両側）。scipy非依存の自前実装。
    table = [[a,b],[c,d]]。周辺和を固定した超幾何分布で、観測より生起確率が
    小さいか等しい全パターンの確率を足し上げる標準アルゴリズム（two-sided, small-p-value法）。
    """
    (a, b), (c, d) = table
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d

    def log_choose(nn, kk):
        if kk < 0 or kk > nn:
            return float("-inf")
        return math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1)

    def hyper_logp(x):
        # P(a=x | 周辺和固定)
        return (log_choose(row1, x) + log_choose(row2, col1 - x)
                - log_choose(n, col1))

    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    obs_logp = hyper_logp(a)
    total = 0.0
    for x in range(lo, hi + 1):
        lp = hyper_logp(x)
        # 観測確率よりわずかに大きいだけの浮動小数誤差を許容(相対1e-7)
        if lp <= obs_logp + 1e-7 * abs(obs_logp if obs_logp != 0 else 1.0):
            total += math.exp(lp)
    return min(1.0, total)


def selftest():
    """GPU/重み不要。base_color_stats/classify/fisher_exact_2x2 の回帰防止テスト。"""
    import numpy as np

    # --- 1. base_color_stats: 既知のtoy attrsで手計算と一致 ---
    attrs = np.array([
        [0.0, 0.0, 0.0, 0.5, 0.5, 1.0],   # 黒(RGB=0)
        [1.0, 1.0, 1.0, 0.5, 0.5, 1.0],   # 白(RGB=1)
    ], dtype=np.float32)
    stats = base_color_stats(attrs)
    assert abs(stats["base_color_mean_0_255"] - 127.5) < 1e-3, stats
    assert stats["num_voxels"] == 2
    assert abs(stats["saturation_rate"] - 1.0) < 1e-9, stats  # 2/2 ともnear0 or near1
    print("SELFTEST PASS: base_color_stats matches hand-computed mean/saturation")

    # --- 2. classify: 境界値 ---
    assert classify(0.0) == "black"
    assert classify(1.999) == "black"
    assert classify(2.0) == "intermediate"
    assert classify(9.999) == "intermediate"
    assert classify(10.0) == "normal"
    assert classify(65.0) == "normal"
    print("SELFTEST PASS: classify boundaries (black<2, normal>=10)")

    # --- 3. fisher_exact_2x2: 教科書既知値（tea-tasting、[[3,1],[1,3]]）と照合 ---
    p = fisher_exact_2x2([[3, 1], [1, 3]])
    assert abs(p - 0.4857142857142857) < 1e-6, f"got {p}"
    print("SELFTEST PASS: fisher_exact_2x2 matches known tea-tasting p=0.485714...")

    # 追加: 完全に有意な例（[[10,0],[0,10]]）は p が非常に小さい
    p2 = fisher_exact_2x2([[10, 0], [0, 10]])
    assert p2 < 1e-4, f"got {p2}"
    print("SELFTEST PASS: fisher_exact_2x2 gives small p for extreme separation")

    # 対称性: [[a,b],[c,d]] と [[b,a],[d,c]] は同じp値のはず
    p3 = fisher_exact_2x2([[2, 4], [1, 2]])
    p4 = fisher_exact_2x2([[4, 2], [2, 1]])
    assert abs(p3 - p4) < 1e-9, f"{p3} vs {p4}"
    print("SELFTEST PASS: fisher_exact_2x2 symmetric under row/col swap")


def parse_seed_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


# 既存v3 fixture（seed=42）を Mac から `kubectl exec <cpu-pod> -- python3` で実測した
# bake後 base color テクスチャ(mesh.glbに埋め込まれた実PNG/WebP)の 0-255 mean。
# raw voxel attrs proxy（本スクリプトが使う指標）がこれと同じ分類を与えるかの較正に使う
# （codexレビュー指摘: 黒1例だけの較正では判定器として不足、複数ケースで検証すべき）。
V3_BAKED_TEXTURE_MEAN_GROUND_TRUTH = {
    "4v_stochastic": 13.022363,   # trimesh実測、正常域
    "2v_stochastic": 0.0,         # trimesh実測、全texel=0（完全に黒）
    "4v_multidiffusion": 9.845867,  # trimesh実測、正常域下限に近い境界値
}


def calibrate_against_v3_fixtures(pipeline, out_dir: Path) -> dict:
    """v3 fixtureの保存済み shape_slat / tex_slat（サンプリング済み、seed=42）を
    ロードし直して `pipeline.decode_latent()` だけ実行し、raw voxel attrs proxy が
    bake後の実テクスチャ mean と同じ黒/正常分類を与えるかを検証する（サンプリング無しで
    数秒〜十数秒、フルスイープ着手前のゲート）。"""
    import numpy as np
    import torch
    from trellis2.modules.sparse import SparseTensor

    report = {}
    for cfg_name, manifest_path in V3_MANIFEST_PATHS.items():
        run_dir = Path(manifest_path).parent
        with open(manifest_path) as f:
            m = json.load(f)
        res = m.get("hr_resolution", 1024)
        shape_feats = np.load(run_dir / "shape_slat_feats.npy")
        shape_coords = np.load(run_dir / "shape_slat_coords.npy")
        tex_feats = np.load(run_dir / "tex_slat_feats.npy")
        tex_coords = np.load(run_dir / "tex_slat_coords.npy")
        shape_slat = SparseTensor(torch.from_numpy(shape_feats).cuda(),
                                   torch.from_numpy(shape_coords).cuda())
        tex_slat = SparseTensor(torch.from_numpy(tex_feats).cuda(),
                                 torch.from_numpy(tex_coords).cuda())
        meshes = pipeline.decode_latent(shape_slat, tex_slat, res)
        stats = base_color_stats(meshes[0].attrs)
        proxy_mean = stats["base_color_mean_0_255"]
        gt_mean = V3_BAKED_TEXTURE_MEAN_GROUND_TRUTH[cfg_name]
        proxy_cls = classify(proxy_mean)
        gt_cls = classify(gt_mean)
        abs_diff = abs(proxy_mean - gt_mean)
        # 分類(black/intermediate/normal)が一致しなくても、native自身の黒(<2)/正常(>=10)の
        # 間に8ポイントの無人地帯がある。proxyとbake後実測の差がこの半分(2.0)以内なら
        # 「同じ黒化度合いを別の測り方で見ている」とみなせる、二値境界(10.0)のすぐ両側に
        # 実測が割れただけの誤検知を許容する（4v_multidiffusion=境界値9.85の実例で確認済み）。
        close_enough = abs_diff <= 2.0
        agrees = bool(proxy_cls == gt_cls or close_enough)
        report[cfg_name] = {
            "proxy_base_color_mean_0_255": proxy_mean,
            "baked_texture_mean_0_255_ground_truth": gt_mean,
            "proxy_classification": proxy_cls,
            "ground_truth_classification": gt_cls,
            "abs_diff": abs_diff,
            "classification_agrees": proxy_cls == gt_cls,
            "close_enough_within_2pt": close_enough,
            "agrees": agrees,
        }
        print(f"[calibrate] {cfg_name}: proxy_mean={proxy_mean:.3f} ({proxy_cls}) "
              f"vs baked_gt={gt_mean:.3f} ({gt_cls}) abs_diff={abs_diff:.3f} agrees={agrees}",
              flush=True)
        torch.cuda.empty_cache()
    return report


def run_sweep(args):
    import numpy as np
    import torch
    from ref_trellis2_mv_inject import inject_sampler_multi_image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from PIL import Image

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "sweep_results.jsonl"

    print(f"[load] pipeline from {args.model_path} ...", flush=True)
    assert PINNED_MODEL_REVISION in args.model_path, \
        f"--model-path must point at the {PINNED_MODEL_REVISION} snapshot, got {args.model_path}"
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()

    print("[calibrate] raw voxel attrs proxy vs v3 baked texture ground truth (3 cases) ...",
          flush=True)
    calib_report = calibrate_against_v3_fixtures(pipeline, out_dir)
    with open(out_dir / "calibration_report.json", "w") as f:
        json.dump(calib_report, f, indent=2)
    if not args.skip_calibration_gate and not all(v["agrees"] for v in calib_report.values()):
        print("[calibrate] FAILED: proxy disagrees with baked-texture ground truth (classification "
              "differs AND abs_diff>2.0pt) for at least one case. Aborting sweep (proxy metric not "
              "trustworthy as-is). See calibration_report.json. Pass --skip-calibration-gate to override.",
              flush=True)
        raise SystemExit(3)
    print("[calibrate] PASSED: proxy agrees with baked-texture ground truth (same classification, "
          "or abs_diff<=2.0pt) for all 3 known cases.", flush=True)

    configs = [
        {"name": "4v_stochastic", "images": args.images_4v.split(","), "mode": "stochastic",
         "seeds": parse_seed_list(args.seeds_b3)},
        {"name": "2v_stochastic", "images": args.images_2v.split(","), "mode": "stochastic",
         "seeds": parse_seed_list(args.seeds_b1)},
        {"name": "4v_multidiffusion", "images": args.images_4v.split(","), "mode": "multidiffusion",
         "seeds": parse_seed_list(args.seeds_multidiff)},
    ]

    # codexレビュー指摘: v3は`torch.manual_seed(seed)`の**後**に`get_cond`を呼んでおり、
    # get_condがRNGを消費する場合その消費分もseedの決定論的な軌道の一部になる。cond自体を
    # seedループの外でキャッシュするとこの消費（あれば）を再現できず、seed=42のv3 bit一致ゲートが
    # 偽陰性で壊れる恐れがある。よって **前処理（RNG非依存、manual_seed呼び出し前に行われる点も
    # v3と同じ）のみキャッシュし、get_cond自体は v3 と同じ位置（manual_seed直後）で毎seed呼び直す**。
    preprocessed_cache = {}

    def get_preprocessed_images(image_paths):
        key = tuple(image_paths)
        if key not in preprocessed_cache:
            raw_images = [Image.open(p) for p in image_paths]
            preprocessed_cache[key] = [pipeline.preprocess_image(im) for im in raw_images]
        return preprocessed_cache[key]

    results = []
    with open(jsonl_path, "a") as jf:
        for cfg in configs:
            num_images = len(cfg["images"])
            images = get_preprocessed_images(cfg["images"])
            for i, seed in enumerate(cfg["seeds"]):
                t0 = time.time()
                torch.manual_seed(seed)
                cond_list_512 = [pipeline.get_cond([im], 512)["cond"] for im in images]
                stacked_512 = torch.cat(cond_list_512, dim=0)
                cond_512 = {"cond": stacked_512, "neg_cond": torch.zeros_like(stacked_512[:1])}
                cond_list_1024 = [pipeline.get_cond([im], 1024)["cond"] for im in images]
                stacked_1024 = torch.cat(cond_list_1024, dim=0)
                cond_1024 = {"cond": stacked_1024, "neg_cond": torch.zeros_like(stacked_1024[:1])}
                shape_steps = pipeline.shape_slat_sampler_params.get("steps", 12)
                tex_steps = pipeline.tex_slat_sampler_params.get("steps", 12)
                ss_steps = pipeline.sparse_structure_sampler_params.get("steps", 12)
                row = {"config": cfg["name"], "num_images": num_images, "mode": cfg["mode"],
                       "resolution": "1024_cascade", "seed": seed}
                try:
                    with inject_sampler_multi_image(pipeline.sparse_structure_sampler, num_images,
                                                     ss_steps, mode=cfg["mode"]):
                        coords = pipeline.sample_sparse_structure(cond_512, 32, num_samples=1,
                                                                   sampler_params={})
                    ss_coords_np = coords.detach().cpu().numpy()
                    row["ss_active_voxels"] = int(ss_coords_np.shape[0])
                    row["ss_coords_sha256"] = sha256_bytes(ss_coords_np.tobytes())

                    shape_model_lr = pipeline.models["shape_slat_flow_model_512"]
                    shape_model_hr = pipeline.models["shape_slat_flow_model_1024"]
                    with inject_sampler_multi_image(pipeline.shape_slat_sampler, num_images,
                                                     shape_steps, mode=cfg["mode"]):
                        shape_slat, res = pipeline.sample_shape_slat_cascade(
                            cond_512, cond_1024, shape_model_lr, shape_model_hr, 512, 1024,
                            coords, {}, max_num_tokens=49152)
                    row["shape_active_voxels"] = int(shape_slat.feats.shape[0])

                    tex_model = pipeline.models["tex_slat_flow_model_1024"]
                    with inject_sampler_multi_image(pipeline.tex_slat_sampler, num_images,
                                                     tex_steps, mode=cfg["mode"]):
                        tex_slat = pipeline.sample_tex_slat(cond_1024, tex_model, shape_slat, {})
                    tex_feats_np = tex_slat.feats.detach().cpu().numpy()
                    tex_coords_np = tex_slat.coords.detach().cpu().numpy()
                    row["tex_active_voxels"] = int(tex_feats_np.shape[0])
                    row["tex_slat_feats_sha256"] = sha256_bytes(tex_feats_np.tobytes())
                    row["tex_slat_coords_sha256"] = sha256_bytes(tex_coords_np.tobytes())
                    row["tex_slat_mean"] = float(tex_feats_np.mean())
                    row["tex_slat_std"] = float(tex_feats_np.std())
                    row["tex_slat_has_nan_or_inf"] = bool(
                        (~np.isfinite(tex_feats_np)).any())
                    if seed == 42:
                        row["v3_gate"] = check_seed42_gate(
                            cfg["name"], row["ss_coords_sha256"],
                            row["tex_slat_feats_sha256"], row["tex_slat_coords_sha256"])
                        if not row["v3_gate"].get("all_bit_identical_to_v3", False):
                            print(f"[WARN] seed42 v3 gate mismatch for {cfg['name']}: "
                                  f"{row['v3_gate']}", flush=True)

                    meshes = pipeline.decode_latent(shape_slat, tex_slat, res)
                    mesh = meshes[0]
                    stats = base_color_stats(mesh.attrs)
                    row.update(stats)
                    row["classification"] = classify(stats["base_color_mean_0_255"])
                    row["decode_ok"] = True
                except Exception as e:
                    row["decode_ok"] = False
                    row["error"] = repr(e)
                    row["classification"] = "error"

                row["elapsed_sec"] = time.time() - t0
                jf.write(json.dumps(row) + "\n")
                jf.flush()
                os.fsync(jf.fileno())
                results.append(row)
                print(f"[{cfg['name']} seed={seed}] classification={row.get('classification')} "
                      f"mean={row.get('base_color_mean_0_255')} elapsed={row['elapsed_sec']:.1f}s",
                      flush=True)
                torch.cuda.empty_cache()

    with open(out_dir / "sweep_summary.json", "w") as f:
        json.dump({"n_results": len(results), "finished_at": time.time()}, f, indent=2)
    print(f"[done] {len(results)} runs -> {jsonl_path}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="/nfs/trellis2-mv/b3-ref-sweep")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--images-4v", default=DEFAULT_IMAGES_4V)
    ap.add_argument("--images-2v", default=DEFAULT_IMAGES_2V)
    ap.add_argument("--seeds-b3", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--seeds-b1", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--seeds-multidiff", default="42,43,44")
    ap.add_argument("--skip-calibration-gate", action="store_true",
                     help="proxy較正の不一致でsweepを止めない(デバッグ用)")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    run_sweep(args)


if __name__ == "__main__":
    main()
