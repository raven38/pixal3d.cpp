# Metal flow DiT の op/block 別プロファイルと RoPE レイアウト修正 — 2026-09-20

環境: macOS 26.5 / MacBook Pro M4 Max（Mac16,6、64 GB）/ Metal、`trellis-cli --profile`
（`docs/design/2026-09-20-metal-flow-profiler.md`）。重み `pixal3d-q8_0 v1`（MV）。入力は cyclops 4 view
（`docker/linux-webgpu-gate/e2e/views` + `transforms.json`、`mesh_scale` 1.0）、seed 1、`--res 1024`、FlashAttention。
比較 baseline は同入力・同 seed の `docs/results/canonical-rig/json_run.log`（2026-09-15、`--profile` 無し）。
ログ: `<scratchpad>/prof/fa.log`（E2E）、`<scratchpad>/prof/bench_*.log`（マイクロベンチ）。

## 1. E2E のタイムライン（`--profile` 付き、1882.4 s）

| 区間 | 今回 | baseline | 備考 |
|---|---:|---:|---|
| [2/6] SS cond + flow + decode | 111.5 s（flow 104.8） | 95.8（89.7） | +プロファイルパス ~13 s |
| [3/6] LR shape SLAT（N=4438） | 165.7 s（flow 150.4） | 109.4（94.7） | +パス ~18 s、残りはドリフト（§4） |
| [3/6] HR shape SLAT（N=17612） | 825.0 s（flow 772.3） | 677.3（627.5） | +パス ~100 s、残りはドリフト |
| [4/6] shape decode | 22.2 s | 23.2 | |
| [5/6] texture cond + flow + PBR decode | 581.1 s（flow 484.3） | 573.0（478.0） | |
| [6/6] postprocess | **176.7 s** | ~191（差分） | weld+fill 4.6 / bvh 1.8 / remesh 12.6 / **decimate 118.4** / uv+bake 35.0 / GLB+PNG 3.2 / PLY 1.1 |

出力 GLB は `uv_bake: Vo=639713 Fo=950170` で baseline と同一（`--profile` はサンプラの結果を変えない。
whole forward の出力をホストへ読み出してから副次パスを回す設計）。

「4 段の flow 合計と `done in` の差 5〜6 分」の正体: 各段の flow 以外（SS cond+decode 6 s、LR cond 15 s、
HR cond/upsample 50 s、texture cond(NAF@1024)+PBR decode 95 s、shape decode 23 s）≈ 190 s と、
postprocess ≈ 180〜190 s（うち decimate_qem 18.4M 面 → 1M 面が 118 s）。未計測の謎ではない。

## 2. flow ごとの内訳（forward #1 の副次パス）

segment pass = role×block の連続スライス（スライス内の fusion / concurrency は保持）。node pass = 1 node ずつ
（isolated。submit/wait の固定費を含むので軽い op を過大評価する）。

| flow | N | whole fwd | segment / whole | self_attn | cross+proj | mlp |
|---|---:|---:|---:|---:|---:|---:|
| SS | 4096 | 4.05 s | ×1.01 | 82.5% | 4.8% | 12.6% |
| Shape-512 | 4438 | 4.51 s | ×1.42 | 81.6% | 5.5% | 12.7% |
| Shape-1024 | 17612 | 40.04 s | ×0.99 | 82.4% | 4.9% | 12.4% |
| Texture | 17612 | 29.13 s | ×1.04 | 84.7% | 4.5% | 10.8% |

block 間のばらつきは小さい（Shape-1024 の self_attn: min 0.87 / max 1.52 s、blk00 が最大でそれ以外はほぼ一様）。

Shape-1024 の node pass（合計 31.8 s、isolated）:

| op | 秒 | 割合 | 所在 |
|---|---:|---:|---|
| FLASH_ATTN_EXT (bf16 K/V) | 8.23 | 26% | self_attn 30 本、137 ms/call ≈ 1.9e12 FLOP → 約 14 TFLOPS 相当 |
| MUL | 6.69 | 21% | self_attn の RoPE（[1,64,12,N] の ne0=1 テンソル） |
| MUL_MAT (q8_0) | 5.30 | 17% | mlp 3.1 / QKV+out 1.2 / cross+proj 1.0 |
| SCALE | 3.19 | 10% | RoPE の zero-fill `[1,128,12,N]`、17.5 ms/op |
| SET_ROWS | 2.73 | 9% | RoPE の再インターリーブ |
| ADD / SUB / CONT / CPY | 2.07 / 1.58 / 1.25 / 0.28 | 16% | 大半が RoPE |

Texture も同じ形（FA 8.26 / MUL 6.67 / MUL_MAT 5.31 / SCALE 3.17 / SET_ROWS 2.57）。

**結論**: attention カーネル（FA）も q8_0 GEMM も Metal で効率的に動いており、forward の約半分（isolated で
~17 s / 31.8 s）が `src/dit.cpp::apply_rope` の elementwise に消えていた。原因は RoPE を `[1, half, nh, L]`
（ne0 = 1）で組んでいたこと。ggml-metal の unary/binary カーネルは ne0 の 1 行を 1 threadgroup で処理する
（`ggml-metal-ops.cpp` `dispatch_threadgroups(ne01, ne02, ne03, nth, 1, 1)`、nth は ne0 から決まるので 1）ため、
N=17612 で **1 op あたり 64×12×17612 = 1350 万 threadgroup × 1 thread** になる。CUDA / Vulkan では flat grid で
処理されるので目立たず、Metal だけで効く。

## 3. RoPE の修正（`src/dit.cpp::apply_rope`）

pair-major に permute+cont して `[half, 2, nh*L]` にし、even/odd を ne0=64 の行として 4 MUL + SUB + ADD で回転、
CONCAT で `[half, 2, nh*L]`（= 連続 `[hd, nh, L]`、head_dim の並びは `[ev(0..63) | od(0..63)]`）に置く。
**再インターリーブしない**: 回転後の q/k の唯一の消費者は sdpa の q·k で、head_dim 内の並びが q と k で共通なら
不変（V は回転しない）。`ggml_set_rows` / zero-fill / `rope_idx` 入力が不要になった。
検証: `trellis-test-rope-layout`（ホスト参照との一致 `max|d| = 4.8e-7`、q·k の不変性 `max|d| = 2.6e-6`
at `max|q·k| = 63.6`、CPU / Metal とも OK）。

before/after（`trellis-test-dit-bench`、GGUF を読みランダム座標・入力で DiT だけを回す。E2E ではない）:

| flow | legacy RoPE | new RoPE | 比 | 備考 |
|---|---:|---:|---:|---|
| Shape-512（N=4438、6 fwd の #1–#5 平均、ABAB） | 4.99 / 4.85 s | 2.54 / 2.47 s | **1.97×** | 安定（短時間で熱が乗らない） |
| Shape-1024（N=17612、ABAB、冷却なし） | 27.1 / 37.2 s | 24.6 / 29.1 s | 1.1〜1.3× | ドリフトに埋もれる（§4） |
| Shape-1024（各計測前に 90 s 冷却、3 fwd の #1–#2 平均、ABAB） | 23.60 / 23.92 s | 14.45 / 14.75 s | **1.63×** | 再現性あり（`bench_cool_hr.log`） |

新 RoPE 後の Shape-1024 node pass（22.5 s）: FA 11.7（52%）/ MUL_MAT 6.9（31%）/ CONT 1.7 / ADD 0.5 / MUL 0.35。
RoPE 由来の op は 1 s 未満に落ち、残りは attention と GEMM のみ。

## 4. サーマルドリフト（この Mac の計測で最大の交絡）

同じバイナリ・同じ入力で、連続負荷中に forward 時間が単調に伸びる:

- Shape-1024 legacy: 24.3 s（01:19）→ 27.1（01:25）→ 37.2（01:27）
- Shape-1024 new: 19.8 → 19.6 → 24.1 → 29.7 s（4 分間、concurrency on/off を交互に。on/off の差は無い）
- E2E 内の Shape-512 も #1 4.5 s → #4 以降 6.3〜8.2 s（baseline ログでは 4.5〜4.8 s で平坦）
- 1 分ほど休ませると 18.3 s に戻る。`pmset -g therm` には記録なし（GPU 側の throttling は載らない）

したがって E2E の HR 31〜40 s/fwd や「Strix Halo iGPU より 3.5× 遅い」は **throttled 状態の持続性能**であり、
瞬間性能（fresh 18〜23 s/fwd）とは別物。Mac の E2E を比較するときは、冷却状態と実行順を揃えるか、
本ドキュメントのようにマイクロベンチを ABAB で取る。

## 5. まだやっていないこと

- 新 RoPE での **E2E 完走と GLB 比較**（V/F・bbox・複数視点レンダ）。RoPE は q·k 不変の書き換えなので理論上
  同じ形状が出るはずだが、fp32 の加算順が変わるので bit 一致は期待しない。次の E2E 1 本で確認する。
- `--no-fa` のフル E2E（予定していたが取りやめ）。FA 本体は HR で ~14 TFLOPS 相当なので優先度は低い。
  SS（N=4096）で FA が NOFA より遅かった観測（`PIXAL3D_E2E_STATUS.md:489`）は小 N の padding/tile 起因の可能性。
- decimate_qem（118 s）と texture cond（NAF@1024、~95 s）の内訳。
- WebGPU / CUDA / Vulkan での新 RoPE の速度確認（CONCAT は全 backend にある。CUDA の concat は ne3 ごとの
  launch なので `[half, 2, nh*L]`（ne3=1）で 1 launch にしてある）。
