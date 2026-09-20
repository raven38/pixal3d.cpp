# CPU QEM decimation の高速化（issue #29）— 計測結果 2026-09-20

環境: macOS 26.5 / MacBook Pro M4 Max（12P + 4E = 16 コア、64 GB）/ Metal ビルド（GPU decimate 無し →
CPU 経路）。ブランチ `perf/qem-decimate`（origin/main d54de75 起点）。設計: `docs/design/2026-09-20-qem-decimate-cpu-perf.md`、
codex レビュー: `docs/reviews/2026-09-20_qem-decimate-cpu-perf_review.md`。

fixture: `~/data/pixal3d/fixtures/cyclops_mv1024_seed1_post.bin`（501 MB、origin/main の
`trellis-cli --views docker/linux-webgpu-gate/e2e/views -m ~/data/weights/pixal3d/gguf-q8_0 --res 1024 --seed 1 --dump-post`
の出力 = remesh + clean 済みメッシュ V 9,183,186 / F 18,372,632 + PBR 4,739,865 voxel @res1024。
baseline log `docs/results/canonical-rig/json_run.log` の `remesh postproc: ... V=9183186 F=18372632` と一致）。
すべての計測は他セッションの `build-metal/trellis*` プロセスが無いことを `ps` で確認してから取った。

## 1. `decimate_qem`（18.37M 面 → 1M 面）の壁時計とピーク RSS

`trellis-test-decimate-bench <fixture> 1000000`（`/usr/bin/time -l`）:

| 実装 | 壁時計 | ラウンド | ピーク RSS | 出力 V / F |
|---|---:|---:|---:|---|
| 変更前（origin/main、hash 順のエッジ ID） | 128.7 s | 45 | — | 478,776 / 961,394（baseline log と一致） |
| 変更前 + エッジ ID を (a,b) 昇順に固定（参照実装 `--legacy`） | 92.8 / 89.4 s | 45 | 3.90 GB | 478,472 / 960,784 |
| **新実装（CSR エッジ抽出 + 並列化、16 スレッド）** | **7.22 / 7.19 / 7.29 s（中央値 7.22 s）**、最終コミット（入口検査追加後）で 7.05 / 7.07 / 7.23 s | 45 | 3.26 GB | 478,472 / 960,784 |

- 受け入れ基準 ≤ 25 s（stretch 15 s）を満たす。**17.8×**（128.7 → 7.22 s）。
- 数字の対応: この表の壁時計は bench ハーネスの `decimate_qem()` 呼び出し全体（入力コピー・最終圧縮・変更前は
  hash map の破棄を含む）。issue の「118 s」は E2E の `[post]` lap「decimate + weld + fill + drop」
  （`--profile` 付きブランチの計測）で、対応する E2E 側の削減は §4。ピーク RSS は bench プロセス全体
  （fixture 501 MB の読み込み分を含む）で、decimate 単体のフットプリントではない。
- 新実装と参照実装（エッジ順固定の変更前）は 45 ラウンドを通して**出力が bit 一致**（`compare` → `bitwise identical: yes`、2 回確認）。
- 変更前（hash 順）との差は V で 304（0.06%）、F で 610（0.06%）。平面領域（首の切断面・地面）で
  コストが bit 一致するエッジのタイ処理が、`unordered_map` の走査順から (a,b) 昇順（CUDA 版 `cub_sort_keys` と
  同じ順）に変わったため。
- RSS の減少 −640 MB は `unordered_map`（27.6M ノード）の消滅分。変更前の壁時計 128.7 s のうちラウンド外の
  ~11 s はこの map の破棄だった。

段別（`TRELLIS_DBG_DECIM=1`、45 ラウンド合計）:

| 段 | 変更前 | 新 | 備考 |
|---|---:|---:|---|
| adj（v→f CSR） | 0.8 | 0.8 | 直列のまま（累積順の保存） |
| edges | 29.1（+ map 破棄 ~11） | 1.4 | 頂点局所抽出、2 パス、並列 |
| qem | 2.5 | 0.3 | 並列 |
| cost | 61.7 | 2.9 | 並列 |
| prop | 18.3 | 1.1 | atomic CAS min、並列 |
| collapse | 4.2 | 0.3 | 並列 |
| compact | 0.8 | 0.7 | 直列のまま |
| 合計 | 117.5 | 7.4 | |

入力の頂点次数: p50 6 / p95 8 / p99 10 / max 14（`trellis-test-decimate-bench` の `degree:` 行）。局所エッジ抽出の候補数は 2×次数 ≤ 28 なので、挿入ソート（n ≤ 32）で足り、`std::sort` フォールバックは実メッシュでは発火しない。

同規模の手続き torus（V 9.0M / F 18.0M、53 ラウンド）: 変更前 175.3 s（他セッションの E2E と同時実行中の値）。

## 2. 正しさ（T1〜T3）

- T1 `trellis-test-decimate-cpu`（torus 閉/開/目標以下/極小目標/目標 0、三角形 1 枚、F=0、非多様体辺、
  高次数 fan 200、重複面 + 孤立頂点、縮退面 + 範囲外 index、スレッド数 1/2/3/8/16）: **PASS**（全項目）。
  `-fsanitize=thread` ビルド（`clang++ -O1 -g -fsanitize=thread`、R=120）: **警告 0、PASS**。
- T2 実 fixture: 参照実装（エッジ順固定）vs 新 = **bit 一致**。
- T3 実 fixture: 変更前（hash 順）vs 新（`trellis-test-decimate-bench compare` / `dist`、面積重み 500k 点、
  `TriBvh` の正確な最近点距離）:

| 指標 | 変更前 | 新 |
|---|---|---|
| V / F | 478,776 / 961,394 | 478,472 / 960,784 |
| bbox | [-0.37832,-0.42284,-0.49277]..[0.37814,0.42252,0.49187] | [-0.37832,-0.42284,-0.49274]..[0.37814,0.42252,0.49187] |
| 開放辺 / 非多様体辺 / 成分数 | 0 / 1814 / 5 | 0 / 1806 / 5 |

| 表面距離（両方向） | mean | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| 変更前 ↔ 新 | 3.1e-6 | 1.5e-8 | 1.7e-5 | 6.2e-5 | 5.3e-4 |
| 入力（remesh 18.4M 面）→ 変更前 | 2.63e-5 | 8.4e-6 | 1.10e-4 | 1.90e-4 | 1.84e-3 |
| 入力（remesh 18.4M 面）→ 新 | 2.63e-5 | 8.4e-6 | 1.10e-4 | 1.90e-4 | 1.84e-3 |

（単位はモデル座標、bbox 対角 1.502。res 1024 のボクセル幅 ≈ 1e-3。）変更前と新の差（mean 3e-6）は
間引き誤差そのもの（mean 2.6e-5）の 1/8 で、**間引き誤差の分布は 3 桁一致**する。

## 3. GLB の A/B（T5）

同じ fixture から変更前 / 新の間引きメッシュを作り、同じ bake を通した
（`post-replay <fixture> <out.glb> --no-weld --no-fill --no-remesh --faces 1000000 --atlas 4096 --decimated <mesh.bin>`）:

| | 変更前 | 新 |
|---|---|---|
| decimate 後 → weld/fill/drop 後 F | 961,394 → 950,170 | 960,784 → 949,600 |
| `uv_bake` Vo / Fo | 639,713 / 950,170（**baseline log と一致**） | 656,916 / 949,600 |
| xatlas チャート数（`glb_metrics.py` 推定） | 19,907 | 22,499 |
| 溶接後の頂点 / 境界 / 非多様体 | 473,176 / 0 / 1804 | 472,892 / 0 / 1796 |
| texel density mean / p50 | 1.607e6 / 1.477e6 | 1.605e6 / 1.471e6 |
| GLB サイズ | 32.9 MB | 33.4 MB |

固定 5 視点（front / right / back / left / bottom、正射影 512 px、点スプラット、`scratchpad/render_splat.py`）:

| 視点 | silhouette IoU | 重なり部の mean \|ΔRGB\| | max-channel 差 > 40 の画素率 |
|---|---:|---:|---:|
| front | 0.99998 | 1.41 / 255 | 0.06% |
| right | 0.99993 | 1.60 / 255 | 0.03% |
| back | 0.99998 | 2.02 / 255 | 0.98% |
| left | 0.99993 | 1.67 / 255 | 0.12% |
| bottom | 0.99997 | 3.93 / 255 | 5.43%（首の切断面 = 平面領域の三角形分割の違いによるスプラットの粒） |

並置画像: `docs/results/2026-09-20-qem-decimate-ab-grid.jpg`（縮小版。上段 変更前、中段 新、下段 |差| ×4）。目視で構造差なし。

## 4. E2E（T6）

新 decimate を組み込んだ `trellis-cli`（origin/main + 本変更、`TRELLIS_DBG_DECIM=1`）で同入力・同 seed を GLB まで通した。
比較 baseline は `docs/results/canonical-rig/json_run.log`（origin/main 相当、`--profile` 無し、1669.8 s）。
**注意**: この E2E の flow 段は、同時に走らせた単体テスト・TSAN・再ビルド（CPU 負荷）とサーマルの影響を受けて
いるので、`done in` の差分は decimate の効果としては読めない。読めるのは postprocess 区間と decimate 自身の時間。

| 区間 | baseline（json_run.log） | 新 decimate | 差 |
|---|---:|---:|---:|
| [2/6] SS | 95.8 | 102.6 | flow 段（本変更と無関係、負荷・サーマルで変動） |
| [3/6] LR shape | 109.4 | 185.3 | 〃 |
| [3/6] HR shape | 677.3 | 784.3 | 〃 |
| [4/6] shape decode | 23.2 | 22.9 | |
| [5/6] texture | 573.0 | 463.3 | 〃 |
| **[1/6] load + [6/6] postprocess（`done in` − 各段の和）** | **191.1** | **73.3** | **−117.8 s** |
| `done in` | 1669.8 | 1631.7 | −38.1（flow 段の変動込み） |

- postprocess 内の `decimate_qem`: 45 ラウンド、ラウンド内合計 **8.36 s**（`[decim]` 行の和。25 分の GPU 負荷直後の
  サーマル状態で、アイドル bench の 7.4 s より少し遅い）。変更前の同区間は 118 s（issue #28 の `[post]` lap）。
- 出力: `decimate_qem(target=1000000): V 9183186->478472, F 18372632->960784 (thresh=1.0e-07, 16 threads)`、
  `decimated postproc: dropped 3 more comps -> F=949600`、`uv_bake: ... Vo=656916 Fo=949600` — §3 の post-replay
  経由の新 GLB と同じ数字（decimate の入力 remesh メッシュが baseline と同一 V/F だったことも log で確認）。
- GLB: `<scratchpad>/e2e/cyclops_new.glb`（33.4 MB）。

## 5. 再現手順

```sh
cmake -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --target trellis-test-decimate-cpu trellis-test-decimate-bench post-replay trellis-cli
./build-metal/trellis-test-decimate-cpu                                  # T1
F=~/data/pixal3d/fixtures/cyclops_mv1024_seed1_post.bin
TRELLIS_DBG_DECIM=1 ./build-metal/trellis-test-decimate-bench $F 1000000 new.bin          # T4（段別も出る）
./build-metal/trellis-test-decimate-bench --legacy $F 1000000 legacy.bin                  # 参照（~90 s）
./build-metal/trellis-test-decimate-bench compare legacy.bin new.bin                      # T2（exit 0 = bit 一致）
./build-metal/trellis-test-decimate-bench dist old.bin new.bin 500000                     # T3
./build-metal/post-replay $F out.glb --no-weld --no-fill --no-remesh --faces 1000000 --atlas 4096 --decimated new.bin  # T5
```
