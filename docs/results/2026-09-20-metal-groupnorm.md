# Metal の NAF GroupNorm を `ggml_norm` 再表現に切り替えた結果（issue #55、#33 follow-up (b)）— 2026-09-20

環境: macOS 26.5 / MacBook Pro M4 Max（64 GB）/ Metal（`MTL0`）。重み `pixal3d-q8_0 v1`（MV、`~/nfs/weights/pixal3d/gguf-q8_0/`）。
入力は cyclops 4 view（`docker/linux-webgpu-gate/e2e/views`）。コード: `perf/metal-groupnorm`（origin/main f147aa2 起点、実装 27fe0da、
worktree `~/Downloads/pixal3d-metal-groupnorm`、`build-metal`）。設計と受け入れ基準: `docs/design/2026-09-20-metal-groupnorm.md`
（codex レビュー `docs/reviews/2026-09-20_metal-groupnorm_design_review.md`）。raw ログ・スクリプトは
`docs/results/2026-09-20-metal-groupnorm/`（`numerics/` = N1 / N2、`ab2/` = P1 / P2 / P2-num（`ab2/pre_restart/` = 厳格ゲートの 4 run、`ab1_contaminated/` = 不採用の 1 回目）、
`e2e/` = E1、`nth/` = U1、`upstream/` = パッチ + PR 本文、`scripts/` = `gate_lib.sh` / `ab_gn2.sh` / `ab_gn2_extra.sh` / `e2e_gn.sh` / `u1_*.sh` / `n1_rerun.sh` /
`nth_patch.py` / `classify_runs.py` / `paired_gn.py` / `summarize_gn.py` / `compare_cond.py`）。

新 = 既定（Metal で `NafGgmlOpts::generic_groupnorm`）、旧 = `--naf-native-gn`（native `ggml_group_norm`、32 スレッド/group）。バイナリは同一。

## 0. 結論（先に）

**判定: 7 基準 PASS（N1 / N2 / N3 / P2 / P2-num / E1 / U1）+ P1 は逸脱付き PASS（40 %/30 s 系列の旧側 1 run が凍結範囲外）。**
P1 / P2 の主系列（6 block、`ab2/`）は設計 doc の凍結計測条件（実行ごと 90 s の静穏ゲート）ではなく 40 %/30 s のゲートで取った値なので、
codex の反証レビュー（`docs/reviews/2026-09-20_metal-groupnorm_results_review.md`、「凍結ゲートで未判定のまま全 PASS は過大」）を受けて
**凍結条件（他 trellis 無し かつ GPU < 25 % が 90 s、E1 と同じ）で 2 block × 6 config = 24 run を取り直した（`ab3_strict/`、20:29〜21:16、24/24 clean）。
結果は 40 %/30 s 系列と全 config で 0.1 s（0.5 %）以内、pair 差は 0.02 s 以内で一致し、凍結条件でも全 config が閾値を満たす**（§4 / §5 の
「凍結ゲート」行。旧 T1024 も 6.26 s で範囲内）。背景の GPU 負荷を run 単位で切り分ける手段は無い（§9.1）ので、性能値は「§9.1 の計測条件での値」。
限界と最悪ケース:

- **未検証**: PyTorch の NAF fixture はこの Mac に無い（natten が要る）。数値の参照は (1) float64 の op 単体 truth（N1）、(2) `naf.cpp` の CPU 経路
  （fixture 検証済み、double 累積 GN）（N2）、(3) 新旧の cond テンソル差（P2-num）で、**PyTorch そのものとの突き合わせは含まない**。
- **E2E の GLB 差は精度差ではない**: SS の voxel 集合は bit 一致（4438）だが、LR cond の 1e-4 級の差で HR / tex の noise 行が別サンプルになる
  （`trellis-cli` の noise 行番号結合）。E1 は「壊れていない」の確認（完走・非有限 0・V/F −1.5 %/−0.9 %・bbox ≤ 1.7e-3・5 視点目視）であって
  品質非回帰の証明ではない。数値の非回帰は N1 / N2 / P2-num が担う。
- **最悪ケース**: P1 の pair 差の最小は 3.53 s（T512）、P2 の最小は −14.14 s（tex / HR）、−3.50 s（LR）。逸脱は block 1 の旧 T1024 が 6.41 s
  （凍結範囲 6.0〜6.3 の上限 +0.11 s、原因不明、同 block の新は影響なし）と、機械判定で除外した 1 run（新 LR、XProtect、値は他と同一、補充済み）。
- **副作用**: 再表現は S=1024 で peak メモリ +513 MB（HR +8.5 %、tex +8.8 %。`ggml_norm` の 512 MiB 出力が view 越しの affine と in-place に
  ならない）。Metal の unified 64 GB では実害なし、WebGPU は元から同じ形。LR / T=1024 `--naf-ops` は不変。
- **計測条件の変更**（§9.1）: 25 %/90 s の GPU 静穏ゲートがユーザー操作中に開かず、18:28 から 40 %/30 s に緩めた（6 block 系列）。閾値は動かして
  いない。凍結条件の 2 block 系列（`ab3_strict/`）で同じ値が出ることを確認した（上記）。

主結果（Metal、M4 Max、q8_0 MV v1、cyclops 4 view、新 = `ggml_norm[n/8,8]` 既定、旧 = `--naf-native-gn`）:

| 何を | 旧 | 新 | 差 | 節 |
|---|---:|---:|---:|---|
| op 単体 `[1024,1024,128]` vs float64 max\|d\| | 2.17e-3（1/std を過大評価） | **9.5e-7** | 2000 倍正確 | §1 |
| op 単体 ms | 466 | 8〜16 | 29〜58× | §1 |
| NAF グラフ T=1024 whole graph（6 block median） | 6.27 s | **2.72 s** | −3.55 s | §4 |
| NAF encoder 出力 `enc_cat` vs CPU 参照 L2rel | 1.07e-3 | **5.25e-4** | 半減 | §2 |
| 4-view cond tex / HR / LR total（6 block median） | 31.54 / 27.69 / 7.01 s | **17.36 / 13.50 / 3.48 s** | −14.18 / −14.19 / −3.53 s | §5 |
| flow に入る proj cond 新旧 L2rel（tex / HR / LR） | — | 4.1e-5 / 3.9e-5 / 4.4e-5（global は bit 一致） | — | §6 |
| E2E cyclops MV1024 `cond_slat` 3 段合計 | 94.7 s | **65.2 s** | −29.5 s（SS voxel 4438 一致） | §7 |
| upstream `nth` パッチ（native op、参考） | 466 ms / 2.17e-3 | 16.1 ms / 1.24e-5、graph 2.78 s | 再表現に 2.6 % 差 | §8 |

#33 で「native 比 L2rel 1.2e-4、bit 一致ではない」と書いた差は、ほぼ native カーネル（32 スレッド/group の f32 逐次加算）側の誤差だった。
upstream への `nth` パッチは別提案（`upstream/`、push は GO 待ち）。本 PR は ggml を触らない。

## 1. N1: op 単体、float64 参照（`trellis-test-naf --gn-oracle`、`[1024,1024,128]` f32、8 group、eps 1e-5）

乱数入力（チャネルごとに平均 ±3・標準偏差 0.1〜5）。truth は host の double 累積（`naf.cpp::naf_group_norm` と同じ式）。
`numerics/n1_metal.log` / `n1_cpu.log`。

| backend | 構成 | max\|d\| | mean\|d\| | L2rel | group ごとの誤差（offset / slope） |
|---|---|---:|---:|---:|---|
| Metal | native `ggml_group_norm`（nth=32） | **2.17e-3** | 9.1e-5 | 1.33e-4 | offset ~1e-7、**slope +6e-5〜+2.6e-4**（1/std を過大評価 = 分散を f32 で過小評価） |
| Metal | `ggml_norm` over `[n/8, 8]`（新の既定） | **9.5e-7** | 3.1e-8 | 6.1e-8 | offset ~1e-8、slope ~1e-8（f32 出力の丸め） |
| CPU（参考） | native `ggml_group_norm` | 9.5e-7 | 3.2e-8 | 6.3e-8 | — |
| CPU（参考） | `ggml_norm` over `[n/8, 8]` | 2.17e-3 | 9.1e-5 | 1.33e-4 | — |

- **Metal では再表現の方が 2000 倍以上真値に近い**（native の誤差は各スレッドが 524k 要素を f32 で逐次加算する分散推定のバイアス）。
  #33 で「native 比 L2rel 1.2e-4、bit 一致ではない」と書いた差は、ほぼ native カーネル側の誤差だった。
- CPU backend では逆（CPU の `norm` は 16.7M 要素の行を f32 で縮約、CPU の `group_norm` は正確）。再表現の精度は backend の norm 実装次第で、
  合否は「その backend で `naf_ggml_opts_for()` が選ぶ構成」に対して判定する（Metal = 再表現、CPU = native、両方 PASS）。
- 判定: **PASS**（Metal: 選択 = 再表現、9.5e-7 ≤ 1e-3 かつ ≤ 1.5 × 2.17e-3）。
- op 時間（同じ oracle の 1 compute、`ms` 列）: Metal native 466.7 ms、再表現 8.2 ms。PR #56 レビュー（計時区間が kernel 完了まで
  含まれているか）を受けて `gn_run` の計時を `ggml_backend_synchronize` で挟んだ f1002cc で再実行（`n1_metal_sync.log` / `n1_cpu_sync.log`、
  18:44、ロック + ゲート経由）: 数値は 4 行とも bit 一致、時間は native 466.4 ms / 再表現 16.1 ms（再表現は 8〜16 ms の範囲で振れる。
  `ggml_backend_graph_compute()` は元々 `compute_async` + `synchronize`（Metal は `waitUntilCompleted`）なので計時区間は変わっていない）。

## 2. N2: NAF グラフ、view 0、CPU 参照（`--naf-ops --naf-compare-cpu`、`numerics/n2_T512.log` / `n2_T1024.log`）

同一入力（cyclops view 0、DINOv3@1024 の patch map）で、新（lowering）・旧（native）を Metal で回し、`naf.cpp` の CPU 経路
（double 累積 GroupNorm、PyTorch fixture で t128 / t512 検証済み。S=1024 で T=512 109 s、T=1024 261 s）と比べる。
**PyTorch の NAF fixture はこの Mac に無い**（natten が要る）ので、CPU 経路が真値の代替。

| T | テンソル | 新（lowering）vs CPU: max\|d\| / mean\|d\| / L2rel | 旧（native）vs CPU: max\|d\| / mean\|d\| / L2rel | 新 vs 旧 L2rel |
|---|---|---|---|---:|
| 512 | `enc_cat`（encoder 出力、GroupNorm 8 層の直後・attention 前、[256,1024,1024]） | 3.85e-2 / **6.1e-4** / **5.25e-4** | 4.08e-2 / **1.61e-3** / **1.07e-3** | — |
| 512 | 最終出力 [1024,512,512] | 7.45e-3 / 5.6e-5 / **9.17e-5** | 1.09e-2 / 6.5e-5 / **1.10e-4** | 1.19e-4 |
| 1024 | `enc_cat`（同上、S 依存なので T=512 と同一） | 3.85e-2 / 6.1e-4 / 5.25e-4 | 4.08e-2 / 1.61e-3 / 1.07e-3 | — |
| 1024 | 最終出力 [1024,1024,1024] | 9.24e-3 / 5.9e-5 / **9.73e-5** | 1.47e-2 / 6.9e-5 / **1.15e-4** | 1.22e-4 |

- encoder 出力の CPU 比 L2rel は **1.07e-3 → 5.25e-4（半減）**、mean|d| は 1.61e-3 → 6.1e-4。残る 5e-4 は f16 GEMM（im2col + mul_mat f16）と
  畳み込みの縮約順の分で、GroupNorm 由来ではない（原因の切り分けは N1）。最終出力は 1.10e-4 → 9.2e-5（T=512）、1.15e-4 → 9.7e-5（T=1024）。
- 新 vs 旧の L2rel 1.19e-4 / 1.22e-4 は #33 §4.1 / 4.2 の 1.2e-4 と一致。
- 判定: **PASS**（新 vs 旧 ≤ 3e-3、`enc_cat` と最終出力の両方で 新の CPU 比 ≤ 旧の CPU 比 × 1.5。実際は両方とも新の方が近い）。

## 3. N3: `trellis-test-naf --selftest` — PASS。

## 4. P1: NAF グラフの whole graph（`--naf-ops`、view 0、ISOLATED node pass 付き）

`ab2/{new,old}_nafops_*_HHMMSS.log`（18:31〜19:37、6 paired block × 3 config = 36 run、`scripts/ab_gn2.sh` → `paired_gn.py` / `summarize_gn.py`）。
各 run は同一バイナリで `--naf-ops --views 1`（view 0、graph 1 本 = compute の壁時計 + ISOLATED node pass）。block k の pair = 時刻順 k 番目の
new と old（ABBA で交互）。`classify_runs.py` で **38 run（36 + 補充 pair 2）中 37 clean、1 run（新 LR 18:42:59、XProtect が CPU > 50 % で 2 スナップ）を機械的に除外**
（集計コマンド: `paired_gn.py ab2 --contam ab2/contam.txt`。`--contam` を渡さないと除外されない）
（値は 0.66 s で他と同じだが規則どおり外す。補充 pair を block 7 として追加計測（`ab_gn2_extra.sh 7 nafops_LR`、19:59、両方 clean）、下表は補充込み: LR は 7 pair 中 clean 6）。

| config（S / T） | 新 whole graph: median / min / max（n） | 旧 whole graph: median / min / max（n） | pair 差 旧−新: median / min / max（clean pair 数） | 旧/新 |
|---|---|---|---|---:|
| nafops_T1024（S=1024, T=1024） | **2.72** / 2.71 / 2.72 s（6） | 6.27 / 6.27 / 6.41 s（6） | **3.55** / 3.55 / 3.70 s（6） | 2.31× |
| nafops_T512（S=1024, T=512） | **2.45** / 2.45 / 2.48 s（6） | 6.00 / 6.00 / 6.01 s（6） | **3.55** / 3.53 / 3.56 s（6） | 2.45× |
| nafops_LR（S=512, T=512） | **0.66** / 0.66 / 0.66 s（6） | 1.54 / 1.54 / 1.55 s（7） | **0.88** / 0.88 / 0.89 s（6） | 2.33× |
| 凍結ゲート（90 s / 25 %）T1024（`ab3_strict/`、2 block） | 2.71 / 2.71 / 2.72 s（2） | 6.26 / 6.26 / 6.26 s（2） | 3.54 / 3.54 / 3.55 s（2） | 2.31× |
| 凍結ゲート T512 | 2.45 / 2.44 / 2.45 s（2） | 5.99 / 5.99 / 5.99 s（2） | 3.54 / 3.54 / 3.55 s（2） | 2.45× |
| 凍結ゲート LR | 0.66 / 0.66 / 0.66 s（2） | 1.54 / 1.54 / 1.54 s（2） | 0.88 / 0.88 / 0.88 s（2） | 2.33× |

| config | 旧 `GROUP_NORM` 合計（n=8） | 新 `NORM` 合計（n=8） | op 差 | graph 差との比 |
|---|---:|---:|---:|---:|
| T1024 | 3.62〜3.66 s（453〜458 ms/op、57 %） | 0.06 s（7.9〜8.0 ms/op、2.3 %） | −3.57 s | pair 差 3.55 s の 100 % |
| T512 | 3.62〜3.63 s（453〜454 ms/op、60 %） | 0.06 s（8.0〜8.1 ms/op、2.6 %） | −3.57 s | 100 % |
| LR | 0.90 s（112 ms/op、57 %） | 0.02 s（2.0〜2.1 ms/op、2.4 %） | −0.88 s | 100 % |

- 6 block のばらつきは 1 %（T1024 の旧 6.27〜6.41、新 2.71〜2.72）。最悪 pair でも 3.53 s 以上の短縮。
- 凍結ゲート（90 s / 25 %、`ab3_strict/`、20:29〜20:52、12/12 clean、ゲート待ち 1 回のみ）の 2 block は 40 %/30 s 系列と 0.01 s 以内で一致し、
  旧 T1024 も 6.26 s で凍結範囲 6.0〜6.3 に入る。
- graph の短縮 = `GROUP_NORM` → `NORM` の op 差そのもの（ISOLATED node pass の合計と一致）。他の op（im2col GEMM、pad、pool、attention）は不変。
- `nafops_T1024` の 2.71〜2.72 s は #33 §4.3 の見込み（6.27 − 3.62 = 2.65 s）と一致。
- 判定: **PASS**（設計 doc の凍結基準: 新 T=1024 ≤ 2.9 s → median 2.72（max 2.72）、新 T=512 ≤ 2.6 s → 2.45（max 2.48）、
  GROUP_NORM/NORM 合計 ≤ 0.1 s → 0.06 s、旧 6.0〜6.3 s → median 6.27 / 6.00（block 1 の旧 T1024 だけ 6.41 で範囲を 0.11 s 超え。同 block の
  新は 2.71 で影響なし。ps には他 trellis 無し・XProtect も無し、原因は特定できない）。n は設計 doc の各 2 から issue コメントの要求どおり 6 block に増やした
  （基準値は動かしていない）。

## 5. P2: 4 view の conditioning A/B（`--mode gpu`、sparse 経路、N=17 489 / LR は `--stride 3`）

`ab2/{new,old}_{tex,hr,lr}_HHMMSS.log`（19:37〜20:13、6 paired block × 3 config = 36 run、`trellis-test-pixal3d-cond-tex … --profile-cond --mode gpu`、
cyclops 4 view、tex = S=1024 T=1024（chunked、`naf_enc x2` を計時）/ HR = S=1024 T=512 / LR = S=512 T=512 `--stride 3`（HR / LR は 1 graph/view =
`cond_slat_gpu_single` の compute を計時）。`classify_runs.py`: **36 run 全部 clean**。集計 `paired_gn.py`。

| config | 新 4-view total: median / min / max（n=6） | 旧 4-view total: median / min / max（n=6） | pair 差 旧−新: median / min / max | 旧/新 | 凍結基準 |
|---|---|---|---|---:|---|
| tex（S=1024, T=1024） | **17.36** / 17.32 / 17.41 s | 31.54 / 31.51 / 31.59 s | **−14.18** / −14.14 / −14.19 s | 1.82× | median −10 s 以上 → **PASS** |
| HR（S=1024, T=512） | **13.50** / 13.48 / 13.52 s | 27.69 / 27.66 / 27.70 s | **−14.19** / −14.14 / −14.22 s | 2.05× | median −10 s 以上 → **PASS** |
| LR（S=512, T=512） | **3.48** / 3.47 / 3.49 s | 7.01 / 6.99 / 7.03 s | **−3.53** / −3.50 / −3.56 s | 2.01× | 報告のみ（見込み 7.0 → ≈3.5 と一致） |
| 凍結ゲート（90 s / 25 %）tex（`ab3_strict/`、2 block） | 17.27 / 17.27 / 17.27 s | 31.44 / 31.42 / 31.45 s | −14.17 / −14.16 / −14.18 s | 1.82× | PASS（凍結条件） |
| 凍結ゲート HR | 13.48 / 13.46 / 13.51 s | 27.63 / 27.61 / 27.65 s | −14.14 / −14.14 / −14.15 s | 2.05× | PASS（凍結条件） |
| 凍結ゲート LR | 3.48 / 3.47 / 3.48 s | 6.98 / 6.98 / 6.99 s | −3.51 / −3.49 / −3.52 s | 2.01× | 報告のみ |

| config | NAF encoder（view あたり compute、4 view の中央値、6 run の median / min / max） | 差 / view | 4 view 換算 |
|---|---|---:|---:|
| tex `naf_enc x2` | 旧 5.89 / 5.89 / 5.89 s → 新 **2.33** / 2.33 / 2.33 s | −3.56 s | −14.2 s（total 差 −14.18 と一致） |
| HR `cond_slat_gpu_single`（DINO + NAF + taps） | 旧 6.76 / 6.75 / 6.76 s → 新 **3.20** / 3.20 / 3.20 s | −3.56 s | −14.2 s |
| LR `cond_slat_gpu_single` | 旧 1.65 / 1.65 / 1.65 s → 新 **0.77** / 0.77 / 0.78 s | −0.88 s | −3.5 s |

- view あたりの差 3.56 s（S=1024）/ 0.88 s（S=512）は P1 の graph 差（3.55 / 0.88 s）= `GROUP_NORM` 8 op の op 差と一致。DINO（0.75 s）、
  `naf_qk` / `naf_attn`（0.21 / 0.37 s）は新旧で不変。6 block のばらつきは total で 0.1 s 以下（0.3 %）。
- #33 §1 の見込み（tex 31.2 → ≈17、HR 27.8 → ≈13.6）と実測（31.5 → 17.4、27.7 → 13.5）が一致。
- 凍結ゲート（`ab3_strict/`、20:54〜21:16、12/12 clean）の 2 block は 40 %/30 s 系列より 0.05〜0.1 s（0.3 %）だけ速く（背景負荷が少ない分）、
  pair 差は 0.02 s 以内で一致。NAF encoder/view（2.33 / 5.88 s）と peak メモリ（+513 MB）も同じ。
- **副作用（メモリ）**: 再表現は S=1024 で peak を **+513 MB** 押し上げる（`[gpu]` 行: HR view_alloc 5583 → 6097 MB、peak 6030 → 6544 MB（+8.5 %）、
  tex 3332 → 3845 MB / 5847 → 6360 MB（+8.8 %）。LR と `--naf-ops` T=1024 は不変）。`ggml_norm` の出力（[1024,1024,128] f32 = 512 MiB）が
  reshape view 越しの mul/add と in-place にならず 1 本余分に生きるため。Metal（unified 64 GB）では問題にならず、WebGPU は元から
  `generic_lowering` で同じ形を通っている（`docs/PIXAL3D_WEBGPU_MEMORY.md`）。affine を in-place にする改善は別件。

## 6. P2-num: flow に入る cond テンソルの新旧差（`--save-prefix`、同じ coords）

`ab2/num/{new,old}_{tex,hr,lr}.log` + `_global.npy` / `_proj.npy`（20:14〜20:19、`--mode gpu --save-prefix`、同じ coords、`compare_cond.py` → `num/compare.txt`）。
global = DINO 由来の [5,1024]（NAF を通らない）、proj = flow に入る投影 cond [N,2048]（N = 17 489（tex / HR、`hr_coords_f32.npy`）/ 3 872（LR、`--stride 3`））。

| config | global max\|d\| | proj max\|d\|（refmax 24.3） | proj mean\|d\| | proj L2rel | cos | 非有限 | 凍結基準 |
|---|---:|---:|---:|---:|---:|---:|---|
| tex（S=1024, T=1024） | **0**（bit 一致） | 1.92e-3 | 1.9e-5 | **4.07e-5** | 1.00000000 | 0 | global bit 一致 ∧ proj L2rel ≤ 1e-3 ∧ 非有限 0 → **PASS** |
| HR（S=1024, T=512） | **0** | 2.05e-3 | 1.8e-5 | **3.92e-5** | 1.00000000 | 0 | **PASS** |
| LR（S=512, T=512） | **0** | 3.15e-3 | 1.9e-5 | **4.39e-5** | 1.00000000 | 0 | **PASS** |

- proj の新旧差 L2rel 4e-5 は N2 の NAF 出力差（1.2e-4）より小さい（投影 cond は NAF 出力の attention / tap の線形結合で、差が平均化される）。
  max|d| は refmax の 1.3e-4（LR）以下。N1 / N2 より、この差の大半は旧（native）側の誤差。
- global が bit 一致 = DINO 経路と入力前処理は新旧で同一（切り替えが NAF の GroupNorm 以外に触れていないことの確認）。

## 7. E1: E2E 2 本（cyclops MV 1024 seed 1、`--profile-cond`、旧 → 新）

`scripts/e2e_gn.sh`（`gate_lib.sh` のゲート: 他 trellis バイナリ無し + GPU Device Utilization < 25% が 90 s）。同一バイナリ（57cf916 = main d581aac を
merge 済み → **LR / HR / tex の 3 段とも device sparse 経路**）で 旧（`--naf-native-gn`、16:40:56〜17:17:25）→ 新（既定、17:18:57〜17:51:38）。
`trellis-cli --views docker/linux-webgpu-gate/e2e/views -m ~/nfs/weights/pixal3d/gguf-q8_0 --res 1024 --seed 1 --profile-cond`。
raw: `e2e/e2e_old.log` / `e2e_new.log` / `ps_snapshots.log` / `old.glb` / `new.glb` / `glb_metrics.txt` / `compare_glb_pair.txt` / `render5_old_vs_new.png`。

**条件の注記**: (a) 新側の開始直後 17:19:00〜17:20 に別セッション（`pixal3d-cond-sparse`）の cond-tex テストが 1 分だけ重なった
（同型ゲートの同時開放。相手側が中断。以後は協調ロック `/tmp/pixal3d-metal-gpu.lock` で排他）。重なったのは **SS 段の flow の先頭 1 分**で、
NAF を使う conditioning のラップ（LR cond は 17:21 以降）は無傷。SS flow が 77.0 → 99.4 s なのはこの重なり。(b) ユーザーの対話中の
GPU（Chrome / WindowServer、ゲート通過時は 0%）は両腕とも同条件とは限らない。**flow の時間と total は性能の根拠にしない**
（noise の行番号結合で HR 以降は別サンプルでもある）。使うのは `[cond] cond_slat` の 3 段と非破壊の確認だけ。

| 項目 | 旧（native GroupNorm） | 新（lowering） | 差 |
|---|---:|---:|---:|
| `active voxels @res32`（SS、NAF 非依存） | 4438 | **4438** | bit 一致（期待どおり） |
| `[cond] cond_slat S=512 R=32 T=512`（LR、device sparse） | 9.8 s | 7.6 s | −2.2 s |
| `[cond] cond_slat S=1024 T=512`（HR） | 39.7 s | 25.0 s | **−14.7 s** |
| `[cond] cond_slat S=1024 T=1024`（tex） | 45.2 s | 32.6 s | **−12.6 s** |
| **cond_slat 3 段合計** | 94.7 s | 65.2 s | **−29.5 s**（合格条件 −25 s 以上） |
| `[cond] naf lowering on MTL0` | `groupnorm=native … (--naf-native-gn)` | `groupnorm=ggml_norm[n/8,8] pad/pool=native conv=im2col` | 選択経路をログで確認 |
| decoded voxels @res1024 | 4 785 132 | 4 710 169 | 別サンプル（−1.6%） |
| GLB V / F | 666 366 / 960 030 | 656 387 / 951 638 | −1.5% / −0.9%（±5% 以内） |
| GLB bbox | [-0.378,-0.423,-0.491]〜[0.378,0.423,0.492] | [-0.378,-0.423,-0.493]〜[0.378,0.423,0.492] | extent rel diff ≤ 1.7e-3 |
| 非有限 / non-manifold edges / winding | 0 / 0 / 100% | 0 / 0 / 100% | — |
| `compare_glb_pair` nn 距離 | — | mean 0.0017、p95 0.0030、max 0.0092（bbox 対角の 0.11%） | 別サンプル間の差として小さい |
| total（参考、根拠にしない） | 2188.2 s | 1960.5 s | flow の変動込み |

5 視点レンダ（上段 旧・下段 新、`render5_old_vs_new.png`）は同じ cyclops で破綻なし（目視。V/F・bbox・nn 距離・目視は局所的な差を
通しうるので「壊れていない」の確認であって品質非回帰の証明ではない。数値の非回帰は §1 / §2 / §6）。

判定: **PASS**（(1) SS voxel 一致、(2) cond_slat 合計 −29.5 s ≥ 25 s、(3) 完走・非有限 0・V/F/bbox ±5% 以内・目視 OK）。
#33 の見込み −29 s（tex + HR、host 経路）に対し、main の device 経路で LR を含めて −29.5 s。

## 8. U1: upstream 候補 `nth` パッチ（`upstream/0001-metal-group-norm-nth.patch`、`build-metal-nth`）

パッチ = `ggml_metal_op_group_norm` の `nth` を group サイズ基準で 32 → 最大 1024 に拡張 + `kernel_group_norm_f32` の 2 回目 zero-fill 前に
`threadgroup_barrier`（2 ハンク、`scripts/nth_patch.py apply|revert`）。**barrier は安全側の追加**: mean の縮約で全 SIMD group が `buf[]` を読む箇所と
SIMD group 0 が `buf[]` を再ゼロ化する箇所の間に同期が無い（`kernel_norm_fuse_impl` は同じ位置に barrier を持つ）ことに基づく形式的な判断で、
**race の再現も Metal の同期検証もしていない**（`test-backend-ops` PASS と速度改善は barrier の必要性の証拠ではない）。submodule の作業ツリーに当てて `build-metal-nth` で `trellis-test-naf` /
`trellis-test-pixal3d-cond-tex` / ggml `test-backend-ops` だけをビルドし、直後に revert（`GGML_METAL_EMBED_LIBRARY=ON` なので shader はビルド時に
埋め込まれ、`build-metal` の走行中バイナリには影響しない。`strings` の `threadgroup_barrier` 数 78 → 79 で埋め込みを確認、submodule は clean）。
raw: `nth/gn_oracle_nth.log`、`nth/nafops_T1024_nth_native_{1,2}.log`、`nth/nafops_T1024_nth_lowering_1.log`、`nth/test_backend_ops_group_norm*_nth.log`。

| 構成（Metal、`[1024,1024,128]` f32、8 group） | op 単体 max\|d\| vs float64 | op 単体 ms | NAF グラフ T=1024 whole graph | GROUP_NORM / NORM 合計（n=8） |
|---|---:|---:|---:|---:|
| native `ggml_group_norm`、nth=32（現行 upstream） | 2.17e-3 | 466 | 6.37〜6.41 s | 3.66 s（458 ms/op、57 %） |
| native `ggml_group_norm`、**nth パッチ** | **1.24e-5** | **16.1** | **2.78 s**（2 本とも） | 0.13 s（16.1 ms/op、4.6 %） |
| `ggml_norm[n/8,8]`（本 PR の既定、パッチ無関係） | 9.5e-7 | 8〜16 | 2.71 s | 0.06 s（8.0 ms/op、2.3 %） |

- パッチだけで native op は **29 倍速く・175 倍正確**になり（group ごとの slope 誤差 +6e-5〜+2.6e-4 → +3e-7〜+1.6e-6）、whole graph は
  6.4 → 2.78 s で再表現（2.71 s）に 2.6 % 差まで迫る。再表現の方がなお僅かに速く正確（1 op 8 ms vs 16 ms、9.5e-7 vs 1.2e-5）。
- ggml 標準テスト: `test-backend-ops test -b MTL0 -o GROUP_NORM` 2/2 OK、`-o GROUP_NORM_MUL_ADD` 2/2 OK（Metal vs CPU、パッチ入り）。
- `--gn-oracle` の判定はパッチ入りでも PASS（選択 = 再表現 9.5e-7 ≤ 1.5 × 1.24e-5）。
- 位置づけ: upstream 候補として別提案（`upstream/0001-metal-group-norm-nth.patch` + `PR_BODY.md`、push は GO 待ち）。本 PR の Metal 既定は
  再表現のまま（ggml を更新しなくても効く、`docs/PIXAL3D_UPSTREAM_POLICY.md` の「ggml と Pixal3D グラフを同じコミットで触らない」に従う）。
  upstream が取り込まれた後は `generic_groupnorm` の Metal 既定を外す選択肢が生まれる（差 2.6 %、再評価は取り込み後）。

## 9. 汚染の判定

### 9.1 計測条件の変更履歴（基準は動かしていない。動かしたのはゲート = 計測の前提条件）

| 期間 | ゲート | 状態 |
|---|---|---|
| 1 回目の A/B（`ab1_contaminated/`） | 他 trellis バイナリ無しのみ（ps） | WindowServer / Chrome が GPU 90〜100 % の状態で回っていた + 別セッションの E2E と重複 → **全 run 不採用**（op 単体の値だけ参考） |
| E1 と `ab2/pre_restart/`（17:51〜18:25、4 run） | 他 trellis 無し **かつ** GPU Device Utilization < 25 % が 90 s 連続 + jitter 後再確認 | 18:06〜18:28 の 22 分でゲートが開かず（ユーザー操作中の Chrome / WindowServer で 11〜65 %）、run 4 本で停止 |
| `ab2/`（18:28〜）と U1 | 他 trellis 無し かつ GPU < **40 %** が **30 s** 連続 + jitter 後再確認、run 中の GPU 使用率を 5 s おきに `<log>.gpu` へ記録、協調ロックに公平化（待機者に 20 s 譲る） | 本節の集計対象 |
| `ab3_strict/`（20:29〜21:16、codex レビュー後） | 他 trellis 無し かつ GPU < 25 % が 90 s 連続（E1 と同じ = 凍結条件）、2 block × 6 config | 凍結条件での確認。全 config で `ab2/` と 0.5 % 以内 |

ゲート緩和の根拠: run は 15〜30 s（P1）/ 1〜2 分（P2）と短く、新旧の差は whole graph で 2.4 倍・op 単体で 57 倍なので、20〜40 % の背景負荷では順位は覆らない。
ただし絶対値（秒）は静穏時より数 % 悪い可能性がある。**run 中の GPU 使用率（`<log>.gpu`）は自分の run を含む合計**（自分の compute 中は
96〜100 %、U1 で確認）なので、他者の利用を run 単位で切り分ける手段は無い。汚染判定は (1) ps スナップショットに他 trellis が居ない（`classify_runs.py`）、
(2) ゲート通過時の GPU 使用率、(3) 6 block の paired 差のばらつき（min / median / max を全部載せ、結論は median で、最悪ケースも併記）で行う。
18:30 の時点で私は「run 中 max ≥ 50 % を汚染扱い」と書いたが、上の理由で成立しないので撤回した（自分の run で常に 100 % になる）。
**検出限界**: (a) 非 trellis の GPU 利用（Chrome / WindowServer）は run 中は検出できない。(b) `gate_lib.sh` の `others()` は自 worktree
（`pixal3d-metal-groupnorm/build-metal*`）のバイナリを除外するので、自分のスクリプト同士（ab_gn2 / u1_run / u1_tbo / n1_rerun / extra）の重なりは
ps 規則では見えない。これは全スクリプトが同じ `mkdir` ロックを取ることで直列化しており、ps スナップショット 164 本（ab2 154 + nth 10）の
`-- trellis:` 欄に trellis プロセスが 2 つ以上写った瞬間は 0 だった（機械確認）。性能の結論は「この計測条件（§9.1 の表）での値」に限定する。

### 9.2 判定

`classify_runs.py ab2/ps_snapshots.log`（規則: (a) ps に他の trellis バイナリ、(b) trellis 以外で CPU > 50 % が 2 スナップ以上、(c) 前窓 90 s に他 trellis、のいずれかで不採用）。

| 区分 | run 数 | clean | 不採用 | 不採用の内訳 |
|---|---:|---:|---:|---|
| P1 `--naf-ops`（6 block × 3 + 補充 1 pair） | 38 | 37 | 1 | 新 nafops_LR 18:42:59: XProtect（`/Library/Apple/System/Library/CoreServices/XProtect…`）が CPU > 50 % で 2/2 スナップ。値 0.66 s は他の 6 本と同じ |
| P2 4-view（6 block × 3） | 36 | 36 | 0 | — |
| 凍結ゲート確認 `ab3_strict/`（2 block × 6 config） | 24 | 24 | 0 | — |
| P2-num（timing に使わない） | 6 | — | — | ps スナップショット無し（数値のみ） |
| U1（nth） | 4 + test-backend-ops 2 | 4 | 0 | 別スクリプト（`nth/ps_snapshots.log`） |
| E1（§7） | 2 | 2（注記付き） | 0 | 新側の SS flow 先頭 1 分に別セッションの短い run が重なった（conditioning ラップは無傷、§7 の注記） |

- 他セッション（`pixal3d-cond-sparse`）とは協調ロックで交互に実行し、`run_guarded` の 25 s 衝突検知は一度も発火しなかった（driver log に `collision` 無し）。
- macOS の `suggestd` が 18:42 以降ほぼ常時 1 コア 50 % 超（74 run 中 64 run で 2/2 スナップ、両 arm 共通）。#33 と同じ扱いで**採否の基準から外し、逸脱として明記**
  する（classify の `suggestd>50%` 列）。影響の実測: suggestd 無しの 18:31〜18:41 の 4 run（新 T1024 2.71、旧 6.41、新 T512 2.48、旧 T512 6.01）と
  有りの run（2.72 / 6.27 / 2.45 / 6.00）に差が無い（run は GPU 束縛で、1 コアの CPU 負荷は乗らない）。
- 非 trellis の GPU 利用（Chrome / WindowServer）は run 単位では検出できない（§9.1）。その代わり 6 block のばらつきが P1 で ≤ 1 %、P2 で ≤ 0.3 % に収まっていることを
  「重なりがあっても結果を動かしていない」根拠にする（最悪 pair でも合格条件を満たす）。
