#!/usr/bin/env python3
"""B3-REF R2: native の noise を PyTorch 参照側へ注入して同一軌道を再現する実験の下地。

設計: docs/design/2026-09-22-trellis2-mv-b3-ref-seed-sweep.md「R2: native noise 注入」節。
v2 の NoiseCapture（`tools/ref_trellis2_mv_pod_run_v2.py`、torch.randn を「保存」に拡張する
monkeypatch）を「読み出し」に拡張したもの。v2ファイル自体は変更しない（新規ファイル）。

**現状（2026-09-22時点）**: `/nfs/trellis2-mv/b3-dump/{B3_seed42,B1_seed44}/` は到着済みだが、
tex DiT が必要とする64ch concatのうち guide半分(`shape_slat_norm`, 期待shape[N,32])が
npyとして保存されていない（run.logに集約統計のみ）。これが無いとtex DiT forwardを
一度も呼べないため、このモジュールは**注入機構そのもの**（NoiseInjector, selftest）だけを
先行実装する。実際のstep比較実行は、guide dump到着後に別途 `run_r2_tex_injection()`
相当のドライバを書いて行う（本ファイルには未実装、TODO参照）。

実行例（guide dump到着後のイメージ、未実装）:
    NoiseInjector({"tex": [np.load(".../tex_noise0.npy")]})
    with injector.scope("tex"):
        tex_slat = pipeline.sample_tex_slat(cond, tex_model, shape_slat_with_native_guide, {})

self-test（GPU/重み不要）:
    python ref_trellis2_mv_noise_injector.py --selftest
"""
import hashlib
import sys


def sha256_array(a) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


class NoiseInjector:
    """v2 NoiseCapture の読み出し版。scope中の `torch.randn` 呼び出しを、
    事前に与えた配列の逐次読み出しに置き換える。

    v2 NoiseCapture の教訓（本ファイルのdocstring・brief記載）: 「注入した noise が実際に
    使われたことを sha256 でログに残す」——patchが有効化されていない/対象の呼び出しを
    拾えていない、という形の無音失敗を防ぐため、`consumed_log` に
    (scope名, index, 要求shape, 注入shape, sha256) を逐次記録する。
    shape不一致は例外を送出する（無音のブロードキャスト等で誤ったテンソルを混入させない）。
    """

    def __init__(self, injections: dict):
        """injections: {scope名: [np.ndarray, ...]} 。各scopeで呼ばれる順にpopして注入する。"""
        self.injections = {k: list(v) for k, v in injections.items()}
        self._idx = {k: 0 for k in injections}
        self._name = None
        self._orig = None
        self.consumed_log = []

    def _patched(self, *args, **kwargs):
        import torch
        import numpy as np

        if self._name is None or self._name not in self.injections:
            # 対象scope外・対象名が注入リストに無い場合は元のrandnにフォールバック
            return self._orig(*args, **kwargs)
        idx = self._idx[self._name]
        pool = self.injections[self._name]
        if idx >= len(pool):
            raise RuntimeError(
                f"NoiseInjector: scope '{self._name}' で注入配列を使い切った "
                f"(要求 idx={idx}, 保持数={len(pool)})。想定より多くtorch.randnが呼ばれている。")
        arr = pool[idx]
        expected_shape = tuple(args[0]) if args and isinstance(args[0], (tuple, list)) else \
            tuple(a for a in args if isinstance(a, int))
        if expected_shape and tuple(arr.shape) != expected_shape:
            raise RuntimeError(
                f"NoiseInjector: scope '{self._name}' idx={idx} の shape不一致 "
                f"(要求={expected_shape}, 注入配列={arr.shape})。誤った配列を注入している疑い。")
        device = kwargs.get("device")
        dtype = kwargs.get("dtype", torch.float32)
        t = torch.from_numpy(np.asarray(arr)).to(dtype=dtype)
        if device is not None:
            t = t.to(device)
        self.consumed_log.append({
            "scope": self._name, "idx": idx, "shape": list(arr.shape),
            "sha256": sha256_array(np.asarray(arr)),
        })
        self._idx[self._name] += 1
        return t

    def scope(self, name: str):
        outer = self

        class _Scope:
            def __enter__(self2):
                import torch
                outer._name = name
                outer._orig = torch.randn
                torch.randn = outer._patched
                return outer

            def __exit__(self2, *exc):
                import torch
                torch.randn = outer._orig
                outer._name = None
                return False

        return _Scope()


def selftest():
    """GPU/重み不要。NoiseInjectorの注入・shape検証・sha256ログを検証する。
    v2 NoiseCaptureのselftestと同じ罠（bound methodの`is`比較が常にFalse）を踏まえ、
    等価性(==)で復元確認する。"""
    import numpy as np
    import torch

    # --- 1. 正常系: scope内でtorch.randnが注入配列を順に返す ---
    injected = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)]
    injector = NoiseInjector({"tex": injected})
    orig_randn = torch.randn
    with injector.scope("tex"):
        t1 = torch.randn(2, 2)
        t2 = torch.randn(2, 2)
    assert torch.randn == orig_randn, "scope終了後にtorch.randnが復元されていない"
    assert torch.equal(t1, torch.from_numpy(injected[0])), "1回目の注入値が一致しない"
    assert torch.equal(t2, torch.from_numpy(injected[1])), "2回目の注入値が一致しない"
    print("SELFTEST PASS: NoiseInjector injects arrays in order within scope")

    # --- 2. scope外ではtorch.randnが元の乱数生成のまま(注入されない) ---
    torch.manual_seed(0)
    expected = torch.randn(3)
    torch.manual_seed(0)
    with injector.scope("shape"):  # "shape"は注入リストに無い名前
        out = torch.randn(3)
    assert torch.equal(out, expected), "登録の無いscope名でも元のrandnにフォールバックすべき"
    print("SELFTEST PASS: unregistered scope name falls back to real torch.randn")

    # --- 3. 使い切ったら例外 ---
    injector2 = NoiseInjector({"tex": [np.zeros((1, 1), dtype=np.float32)]})
    raised = False
    with injector2.scope("tex"):
        torch.randn(1, 1)
        try:
            torch.randn(1, 1)
        except RuntimeError as e:
            raised = True
            assert "使い切った" in str(e)
    assert raised, "注入配列を使い切っても例外が出なかった"
    print("SELFTEST PASS: exhausting injection pool raises RuntimeError")

    # --- 4. shape不一致は例外、かつ元のrandnが復元される ---
    injector3 = NoiseInjector({"tex": [np.zeros((3, 3), dtype=np.float32)]})
    raised = False
    with injector3.scope("tex"):
        try:
            torch.randn(2, 2)
        except RuntimeError as e:
            raised = True
            assert "shape不一致" in str(e)
    assert raised, "shape不一致で例外が出なかった"
    assert torch.randn == orig_randn, "例外後もtorch.randnが復元されていない"
    print("SELFTEST PASS: shape mismatch raises RuntimeError and restores torch.randn")

    # --- 5. consumed_log に sha256 が記録される（brief必須要求） ---
    injector4 = NoiseInjector({"tex": [np.array([1.0, 2.0, 3.0], dtype=np.float32)]})
    with injector4.scope("tex"):
        torch.randn(3)
    assert len(injector4.consumed_log) == 1
    entry = injector4.consumed_log[0]
    expected_sha = sha256_array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert entry["sha256"] == expected_sha, "consumed_logのsha256が実際の配列と一致しない"
    print("SELFTEST PASS: consumed_log records sha256 of the actually-injected array "
          "(brief要求: 注入したnoiseが実際に使われたことをsha256でログに残す)")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    print("R2 driver (native cond/guide dump待ち、未実装): "
          "docs/design/2026-09-22-trellis2-mv-b3-ref-seed-sweep.md 参照", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
