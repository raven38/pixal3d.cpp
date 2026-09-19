#!/usr/bin/env python3
"""flow DiT の q/k RMSNorm gamma から、FlashAttention に入る Q / K 要素の静的上界を出す。

self_attn / cross_attn の q, k は `rms_gamma`（x / rms(x) * gamma、head_dim 単位）を通ってから
RoPE（ノルム保存の回転）に入る（src/dit.cpp::self_attn / cross_attn）。head_dim = hd のとき
|x_i / rms(x)| <= sqrt(hd)（全エネルギーが 1 要素に集中したとき）なので、要素の上界は
sqrt(hd) * max|gamma|。これが F16 の最大値 65504 に対してどれだけ余裕があるかを flow ごとに印字する。
V には RMSNorm が無いので、V の範囲は実入力の計測（TRELLIS_DBG_FA_RANGE）でしか分からない。

gamma は [d_model] = [head_dim * n_heads] で、RMSNorm は head_dim 軸（ggml_rms_norm の ne0）で取られる
（x は [head_dim, n_heads, L]）。head_dim は include/dit.h の DiTParams 既定 128（全 flow 共通）。

使い方: tools/fa_gamma_bound.py [--head-dim 128] <flow.gguf> [...]
"""
import sys
import numpy as np
import gguf

F16_MAX = 65504.0


def dequant_f32(t):
    """gamma は F32/F16 のはず。念のため型を確認して float32 に落とす。"""
    a = np.asarray(t.data)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return a.reshape(-1)


def main():
    args = sys.argv[1:]
    hd = 128
    if args and args[0] == "--head-dim":
        hd = int(args[1]); args = args[2:]
    if not args:
        print(__doc__)
        return 2
    for path in args:
        r = gguf.GGUFReader(path)
        rows = []
        for t in r.tensors:
            n = t.name
            if not (n.endswith(".q_rms_norm.gamma") or n.endswith(".k_rms_norm.gamma")):
                continue
            g = dequant_f32(t)
            nz = np.abs(g[g != 0])
            rows.append((n, g.shape[0], float(np.abs(g).max()), float(np.abs(g).min()), float(np.sqrt((g * g).mean())),
                         float(nz.min()) if nz.size else float("inf")))
        if not rows:
            print(f"{path}: gamma tensors not found")
            continue
        d_model = rows[0][1]
        gmax = max(x[2] for x in rows)
        gmin = min(x[3] for x in rows)   # 0 の gamma（死んだ次元）は K 要素が厳密に 0 になるだけで F16 の問題にはならない
        bound = np.sqrt(hd) * gmax
        # 種類別（self q / self k / cross q / cross k）の max|gamma|
        gnz = min(x[5] for x in rows)
        kinds = {}
        for n, _, mx, mn, rms, _nz in rows:
            kind = ("cross" if "cross_attn" in n else "self") + ("_q" if ".q_rms_norm" in n else "_k")
            kinds.setdefault(kind, []).append(mx)
        print(f"{path}")
        print(f"  gamma tensors: {len(rows)}  d_model={d_model}  head_dim={hd}  max|gamma|={gmax:.4f}  min|gamma|={gmin:.6f}  min nonzero|gamma|={gnz:.6f}")
        for k, v in sorted(kinds.items()):
            print(f"    {k:8s} n={len(v):3d}  max|gamma|={max(v):.4f}")
        print(f"  static bound sqrt(hd)*max|gamma| = {bound:.2f}   F16 headroom = {F16_MAX / bound:.1f}x")
        # F16 の正規化数下限（6.1e-5）との比: gamma が小さいほど K の小要素が subnormal に落ちやすい
        print(f"  min nonzero|gamma| / f16_min_normal(6.1e-5) = {gnz / 6.1e-5:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
