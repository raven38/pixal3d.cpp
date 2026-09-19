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

## 4. この Mac の計測ノイズ（最大の交絡。**対になっていない比較は信用できない**）

同じバイナリ・同じ入力・同じ N=17612 で、時刻によって 2.6× 振れる:

| 時刻 | 条件 | legacy | new |
|---|---|---:|---:|
| 01:40–01:48 | 各計測前に 90 s 冷却、ABAB | 23.60 / 23.92 s | 14.45 / 14.75 s |
| 01:24–01:30 | 連続負荷、ABAB | 27.08 / 37.15 s | 24.55 / 29.07 s |
| 02:24–02:30 | E2E 直後（hot）、背中合わせ | 55.01 s | 29.10 s |
| 02:35 | 300 s 放置後 | — | 37.16 s |

300 s 放置後の方が E2E 直後より**遅い**（37.16 vs 29.10）ので、単純なサーマルリカバリでは説明できない。
実際 `mds` / `mds_stores`（Spotlight）、Time Machine の `backupd-helper`、`CoreSuggestions`（90% CPU）、JAMF が
常時動いており、load average は 4〜17 の範囲で動いていた。**この機械では単発の絶対値に意味はない。**

一方、**短時間に交互に取った対の比は安定している**:

| 比較 | legacy | new | 比 |
|---|---:|---:|---:|
| Shape-512、ABAB ×2 | 4.99 / 4.85 s | 2.54 / 2.47 s | 1.97 / 1.96× |
| Shape-1024、冷却 ABAB ×2 | 23.60 / 23.92 s | 14.45 / 14.75 s | 1.63 / 1.62× |
| Shape-1024、hot 背中合わせ | 55.01 s | 29.10 s | 1.89× |

したがって **RoPE 修正の効果は Shape-1024 で 1.6〜1.9×、Shape-512 で約 2.0×** と読む。
E2E の段別比較（§5）は日をまたいだ**対になっていない**比較なので、段ごとの比はこのノイズを含む。

## 5. 新 RoPE の E2E 1 本（2026-09-20 01:59–02:24、`--profile` なし、rc 0、1515.9 s）

同じ入力・seed 1・同じ重み。**legacy 参照は本セッションの `fa.log` / `fa.glb`**（01:18 完了、`--profile` 付き 1882.4 s）
と 2026-09-15 の `json_run.log`（1669.8 s）。

| 段 | json_run.log (09-15) | 今回 new | 参考: fa.log（legacy, `--profile` 込み） |
|---|---:|---:|---:|
| SS flow | 89.7 s | **37.7 s** | 104.8 |
| Shape-512 flow | 94.7 s | **41.3 s** | 150.4 |
| Shape-1024 flow | 627.5 s | 630.8 s | 772.3 |
| Texture flow | 478.0 s | 373.3 s | 484.3 |
| postprocess | ~177 s | 230.6 s | 176.7 |
| 合計 | 1669.8 s | **1515.9 s** | 1882.4 |

Shape-1024 だけ改善が見えないが、これは §4 のノイズによる。今回の E2E の HR は 31.5 s/fwd で、
E2E 直後に測った hot の new（29.1 s/fwd）と一致する一方、比較対象の 09-15 baseline は別日の別状態。
**同じ瞬間に測った対**（hot: legacy 55.0 vs new 29.1）では 1.89× 出ている。

### GLB の同値性（`tools/compare_glb_pair.py`、4 視点レンダは `tools/render_glb_fast.py`）

| 指標 | legacy (fa.glb) | new (rope.glb) |
|---|---|---|
| V / F | 639 713 / 950 170 | 660 440 / 962 544（+1.3%） |
| bbox extents | 0.75647 / 0.84536 / 0.98464 | 0.75649 / 0.84535 / 0.98440（相対差 1.0e-5〜2.4e-4） |
| 重心 | (−0.00113, −0.05163, −0.00313) | (0.00074, −0.04567, −0.00300) |
| 連結成分（uv_bake） | 2（+ 3 成分を後段で除去） | **1**（floater 除去も 26 vs 68 と少ない） |
| 頂点の最近傍距離（両方向 2 万点） | — | mean 0.00165（bbox 対角の **0.11%**）、p95 0.0028、max 0.0106 |
| 非有限頂点 | 0 | 0 |
| 4 視点レンダ | `2026-09-20-rope/render_legacy_rope.png` | `2026-09-20-rope/render_new_rope.png`（単眼・鼻輪・牙・耳・台座すべて一致。斑点は面サブサンプリングのアーティファクト） |

中間量も一致: active voxels @res32 = 4438（完全一致）、HR token 17 614（legacy 17 612、+0.01%）、
remesh 18 539 476 面（legacy 18 539 204、+0.001%）。差は fp32 の加算順が SS decode と HR 量子化の
離散判定に伝播したもので、bit 一致は設計上期待していない。**品質劣化の兆候は無く、成分数はむしろ改善。**

## 6. まだやっていないこと

- decimate_qem（118〜142 s）と texture cond（NAF@1024、~95 s）の内訳。
- FA の K/V を f16 にする A/B（既存 `TRELLIS_FA_FAST`）、f16 重みの GEMM 比較。
- WebGPU / CUDA / Vulkan での新 RoPE の速度確認（CONCAT は全 backend にある。`[half, 2, nh*L]` は ne3=1 なので
  CUDA の per-ne3 launch でも 1 回）。
- 機械ノイズを排した E2E 比較（同日・同条件で legacy / new を交互に回す。1 本 25 分 × 2）。
