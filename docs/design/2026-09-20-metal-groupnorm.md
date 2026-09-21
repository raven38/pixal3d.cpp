# Metal の NAF GroupNorm を `ggml_norm` 再表現に切り替える（issue #55、#33 follow-up (b)）

日付: 2026-09-20 / 対象ブランチ: `perf/metal-groupnorm`（origin/main f147aa2 起点、worktree `~/Downloads/pixal3d-metal-groupnorm`）/
ステータス: 実装・計測中（codex 設計レビュー 1 周目 = 差し戻し → 本改訂: kill-switch を env から `--naf-native-gn` フラグへ、
N1 を backend の選択に追従する判定へ、N2 に encoder 出力の比較を追加、E1 を「段の時間 + 非破壊スモーク」に格下げして
数値の非回帰は P2-num へ移動、upstream の barrier は `kernel_norm_fuse_impl` の前例で根拠づけ。レビュー記録
`docs/reviews/2026-09-20_metal-groupnorm_design_review.md`）/ 親: #28 → #33、本 issue: #55 /
並行作業: #33 follow-up (a)（LR site の device sparse 経路、`perf/cond-sparse-device`、別セッション）。本設計が触るのは
`naf_gpu.cpp::naf_ggml_opts_for` / `naf.h` / テストだけで、(a) が触る `pixal3d_cond_gpu.cpp` / `trellis_cli.cpp` とは重ならない
（2026-09-20 時点で (a) のブランチは main と同一、未コミット差分なし）。両経路とも `naf_ggml_opts_for(naf)` を通るので、
本変更は (a) の前後どちらでも同じだけ効く。

## 1. 何を直すか（#33 §4.3 で確定済みの事実。ここでは再検討しない）

- `thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp::ggml_metal_op_group_norm` は group ごとに 1 threadgroup × **`nth = 32` 固定**
  （`ne00` に応じて増やすループはコメントアウト。upstream `llama.cpp` master も同じ、2026-09-20 に raw を取得して確認）。
  NAF image encoder（S=1024）の `[1024,1024,128]` f32（512 MB）× 8 op を 8 × 32 = 256 スレッドで 3 パス舐めるので **453 ms/op、
  3.62 s/view = NAF encoder グラフの 58 %**（T=1024 で 6.27 s/view、T=512 で 6.00 s）。
- 対処 (i): GroupNorm だけを既存の `ggml_norm` 再表現（`[W*H*C/8, 8]` reshape、`NafGgmlOpts::generic_groupnorm`、#33 で診断用に追加済み）
  に変える → **2.71 s/view（T=1024）、2.45 s（T=512）**、GROUP_NORM 8 × 8 ms。full generic（pad / pool も再表現）は T=1024 で
  2.60 s と 0.11 s 速いが、T=512 で 2.68 s と 0.23 s 遅い（`sum_rows` pool）。E2E は T=512 が 2 段（LR / HR）、T=1024 が 1 段（tex）
  なので GroupNorm だけの構成を採る。
- 対処 (ii): ggml-metal の `nth` を戻す（group_norm の dispatcher 限定で 32 → 1024 にした反証テストで 453 → 16 ms/op、
  グラフ 2.78 s）。(i) より 0.07 s/view 遅く、`thirdparty/ggml` への変更が要る。**(i) を既定にし、(ii) は upstream への提案として
  用意する**（#33 の結論どおり）。

## 2. 変更点

### 2.1 本体（`src/naf_gpu.cpp::naf_ggml_opts_for`、`include/naf.h`）

```
o.generic_lowering  = !(supports GROUP_NORM && PAD_REFLECT_1D && POOL_2D);   // 従来どおり（WebGPU）
o.generic_groupnorm = is_metal && !g_naf_native_gn;                           // 新規: ggml_backend_is_metal()、--naf-native-gn で戻す
```

- backend 判定は公開 API `ggml_backend_is_metal(naf.backend)`（`ggml-metal.h`、GUID 判定。`GGML_USE_METAL` が無いビルドでは false）。
  1 周目は `ggml_backend_name` の `"MTL"` 接頭辞判定だったが、codex の指摘（名前は実装詳細、ラッパー backend で誤判定）で置き換えた。
  CPU / Vulkan / CUDA は native のまま（CUDA は `naf_upsample_gpu` の別経路で `group_norm_f32<1024>`、Vulkan は未計測なので触らない）。
- `naf.h` の `generic_groupnorm` のコメント「Diagnostic … naf_ggml_opts_for() leaves it false」を「Metal では既定 on」に書き換える。
- **A/B 用の戻し口は CLI フラグ `--naf-native-gn`**（`trellis_args` の `naf_native_gn` → `g_naf_native_gn`、`--no-fa` と同じ
  「A/B 用に旧経路へ戻す」フラグの前例に揃える。`README.md` のフラグ表に 1 行追加）。1 周目は `TRELLIS_DBG_NAF_GENERIC=0` の
  拡張で済ませる案だったが、README の「挙動を変える env は無い」契約（L144）と矛盾する・サービス環境に残った env で
  production 経路が変わる、という codex の指摘で撤回。`TRELLIS_DBG_NAF_GENERIC` は従来どおり（`generic_lowering` だけ）。
  `trellis-test-pixal3d-cond-tex` も同名フラグで同じグローバルを立てる。

変更行数: `naf_gpu.cpp` 12 行、`naf.h` 3 行、`trellis_args.{h,cpp}` 4 行、`trellis_cli.cpp` 2 行、`README.md` 1 行。
`test_pixal3d_cond_tex.cpp` の `--naf-generic-gn` は「既定で on」になるが `--naf-compare` の native 側は明示的に false にしているので
そのまま動く。

### 2.2 op 単体の数値テスト（`trellis-test-naf --gn-oracle <naf.gguf> [gpu]`、新モード）

実形状 `[W=1024, H=1024, C=128, 1]` f32 の乱数入力（チャネルごとに平均 ±3・標準偏差 0.1〜5 の realistic な分布。seed 固定）に対し、

| 構成 | 何を計算 |
|---|---|
| (a) native | `ggml_group_norm(x, 8, 1e-5)` を対象 backend で |
| (b) lowering | `ggml_norm(reshape_2d(x, W*H*C/8, 8), 1e-5)` → reshape 戻し、同じ backend で |
| (truth) | host で double 累積（`naf.cpp::naf_group_norm` と同じ式: sum / sumsq → mean / var）。affine は掛けない（op の比較なので） |

(a)(b) それぞれ truth との max|d| / mean|d| / L2rel と、group ごとの誤差の分解（offset = 平均の誤差、slope = 1/std の誤差）を印字する。
ggml-standalone-graph-test の罠（compute のたびに入力を再アップロード、Metal は `Model::free()` してから exit）に従う。
`gpu=-1` で CPU backend の 2 構成も同じ表に出す。**合否は「その backend で `naf_ggml_opts_for()` が選ぶ構成」に対して判定する**
（再表現の精度は backend の norm 実装次第で、CPU では逆転する。1 周目の「lowering 固定」判定は CPU で必ず FAIL したので直した）。
ビルドに `nth` パッチを当てた別ディレクトリ（`build-metal-nth`、§5）でも同じバイナリを回し、(ii) の数値を upstream 提案に添える。

### 2.3 NAF グラフの数値テスト（`trellis-test-pixal3d-cond-tex --naf-ops … --naf-compare` 拡張）

既存の `--naf-compare`（同一プロセス・同一入力で native を回して出力差）に **`--naf-compare-cpu`** を足す: NAF を CPU（`Model::load(gguf, -1)`）に
もう 1 本ロードして `naf_upsample()`（`naf.cpp` の double 累積 GroupNorm を持つ検証済み CPU 経路）を回し、**native と lowering の両方**を
CPU 出力と比べる（max|d| / L2rel、サイズ一致と finite を確認してから）。「lowering vs native の差 1.2e-4」だけでは、どちらが真値に
近いかが分からないため。比較は 2 段: **encoder 出力 `enc_cat`**（GroupNorm 8 層の直後・attention の前、`NafDebug`）と最終出力。
最終出力の差には f16 GEMM・畳み込みの縮約順・attention も混ざるので「原因が GroupNorm」の証明にはならない（それは N1 の役目）。
N2 が言えるのは「lowering に変えた結果が CPU 参照から遠ざからない」こと（方向の確認）で、`enc_cat` の比較で GroupNorm を含む
encoder の段に効果を絞る。CPU の S=1024 encoder は T=512 で 110 s（実測）なので T=1024 も回す。

### 2.4 upstream 提案（(ii)、push は GO 後）

`ggml_metal_op_group_norm` の `nth` を **group サイズ `gs = ne00*ne01*ceil(ne02/ngrp)`** に基づいて 32 から倍々で
`min(1024, pipeline max)` まで上げる（コメントアウトされていた元のループは `norm` からのコピーで `ne00/4` 基準。group_norm では
1 threadgroup が `gs` 要素を担当するので基準を変える）。カーネル `kernel_group_norm_f32` は `ntg > 32` のときの SIMD group 間縮約
（`buf[32]`）を既に持つ。**ただし mean パスの `tmp = buf[tiisg]` 読み（L3220）と variance パスの `buf[tiisg] = 0` 再初期化（L3235）の
間に barrier が無い**。SIMD group 間の順序は barrier でしか保証されないので、simdgroup 0 が variance ループを抜けて zero-fill する
前に別の simdgroup が L3220 の読みを終えている保証は形式的には無い（`nth = 32` では simdgroup が 1 つなので顕在化しない）。
上流自身の新しい `kernel_norm_fuse_impl`（同ファイル L2955〜）は同じ 2 パス構成で、**2 回目の partial 書き込みの前に
`threadgroup_barrier` を置いている**（L3011）ので、提案もその前例に揃えて zero-fill の前に barrier を 1 本足す
（codex は「新たな保証にならない」としたが、それは SIMD group 間のプログラム順序を仮定した読みで、Metal のメモリモデルは
それを保証しない。観測された故障は無いので PR 本文では「形式的な欠落、`kernel_norm_fuse_impl` と同じ守り」と書く）。
`gs` の乗算は `int64_t` にキャストしてから、`nth` は smem 32 要素の制約で 1024 に cap。カーネル側の `int start/end`
（2^31 要素超で溢れる）と `ne02 % ngrp != 0` のときの除数（最後の group の実要素数 < gs）は既存の問題で、今回の範囲外
（PR 本文で触れるだけ）。
成果物: パッチ（`docs/results/2026-09-20-metal-groupnorm/upstream/0001-metal-group-norm-nth.patch`）+ PR 本文 + §2.2 の数値。
`patches/` には置かない（本リポジトリのビルドでは (i) が既定で、この op を使う経路が NAF 以外に無いので適用しても効果が無い。
`docs/GGML_FORK_DIFF.md` に「未適用の upstream 候補」として 1 行書く）。push 先（`ggml-org/llama.cpp` か `pwilkin/ggml`
`trellis-patches`）は go-queue の項目でユーザーが決める。

## 3. 受け入れ基準（着手前に凍結。動かしたら報告に明記）

**PyTorch の NAF fixture はこの Mac に無い**（natten が要り pod でしか取れない。`docs/results/2026-09-20-conditioning-profile.md`
「Caveats」）。数値の真値は (1) op 単体の double、(2) `naf.cpp` の CPU 経路（PyTorch fixture で t128 / t512 を検証済み）で代替する。

| # | 何を | 合格条件 | 根拠 |
|---|---|---|---|
| N1 | §2.2 op 単体、`[1024,1024,128]`、Metal（CPU も参考に回す） | その backend で `naf_ggml_opts_for()` が選ぶ構成の truth 比 max\|d\| ≤ 1e-3 **かつ** もう一方の max\|d\| × 1.5 以下 | 出力は単位分散なので絶対≈相対。「選んだ方が悪くない」を直接測る。**Metal で lowering が native の 1.5 倍を超えたら (i) を止めて (ii) に切り替える**（設計上の分岐点） |
| N2 | §2.3 NAF グラフ、view 0、T=1024 と T=512 | lowering vs native の L2rel ≤ 3e-3（既存 `trellis-test-naf` の out tol）、かつ **`enc_cat` と最終出力の両方で** lowering の CPU 比 L2rel ≤ native の CPU 比 L2rel × 1.5 | #33 の実測 1.2e-4 の再確認 + 方向の確認（原因の特定は N1） |
| N3 | `trellis-test-naf --selftest` | PASS | 既存の CPU GroupNorm 単体チェックの回帰 |
| P1 | `--naf-ops` whole graph、view 0、既定（新）vs `--naf-native-gn`（旧）、各 n=2、実行ごと 90 s ゲート | 新: T=1024 ≤ 2.9 s、T=512 ≤ 2.6 s、GROUP_NORM/NORM 合計 ≤ 0.1 s。旧: 6.0〜6.3 s（#33 と一致することの確認） | #33 §4.1 / 4.2 の値（2.71 / 2.45）に 0.15 s の余裕 |
| P2 | `--mode gpu` 4 view、tex（S=1024 T=1024）/ HR（S=1024 T=512）/ LR（S=512 T=512）、新 vs 旧、各 n=3 | tex と HR は中央値で **−10 s 以上**（見込み tex 31.2 → ≈17、HR 27.8 → ≈13.6）。LR は見込み（7.0 → ≈3.5、GN は 128 MB × 8 op）を報告するだけで合否には使わない | #33 §1 の device 版の値 + §4 の GN 単独利得 3.55 s/view |
| P2-num | `--mode gpu --save-prefix` で 新 / 旧 の conditioning 出力（global [5,1024] / proj [N,2048]）を tex / HR / LR の 3 構成で保存して比較 | 3 構成とも global は bit 一致（DINO 由来、NAF を通らない）、proj の L2rel ≤ 1e-3（見込み ~1e-4 = N2 と同水準）、非有限 0 | **flow に入る cond テンソルそのものの差**を測る。これが下流に伝わる唯一の差で、flow の重み量子化（q8_0、相対 ~4e-3）より小さければ「同一サンプルの数値比較」として十分。E1 の GLB 比較は noise 結合のためこの役目を果たせない（codex 指摘で分離） |
| E1 | E2E（cyclops MV 1024 seed 1、`--profile-cond`）、同一バイナリで旧（`--naf-native-gn`）→ 新の 2 本、直列、ゲート付き | (1) `active voxels @res32` が一致（SS は NAF を使わないので bit 一致が期待値。違えば別の何かを壊している）、(2) `[cond] cond_slat` 3 段の合計が **−25 s 以上**（見込み −29 s + LR 分）、(3) 非破壊スモーク: 完走・`tools/glb_metrics.py` で非有限 0・V/F/bbox が旧の ±5 %・5 視点レンダを並置して目視 | **E1 は段の時間の統合確認と「壊れていない」スモークであって、品質の非回帰の証明ではない**（noise の行番号結合で LR cond が 1e-4 変わると HR 以降は別サンプルになる。GLB の差は精度差として読まない。V/F/bbox/目視は局所的な破綻を通しうる）。数値の非回帰は N1 / N2 / P2-num で示す。HR の固定 noise 比較（`TRELLIS_DUMP_SLAT`）は main に無いので今回は使わない |
| U1 | §2.4 のパッチを当てた `build-metal-nth` で N1 と P1（T=1024）を 1 回 | N1 合格 + P1 で GROUP_NORM ≈ 16 ms/op | upstream 提案に添える数値。**GO 前に push しない** |

計測は全部 `docs/results/2026-09-20-cond-profile/scripts/ab_clean2.sh` の `gate / snapshot / run` テンプレで（実行ごと 90 s の静穏ゲート、
前・中 30 s・後の `ps` スナップショット）、採否は `classify_runs.py` で機械判定する（別セッションが同じ GPU で (a) の作業中）。
バイナリは新旧とも同一（切り替えは `--naf-native-gn` フラグのみ）。ログ・スクリプト・レンダは `docs/results/2026-09-20-metal-groupnorm/` に置く。

## 4. やらないこと

- pad / pool の再表現を Metal で有効にする（T=512 で遅い）。IM2COL / CONT は (c) の別 issue。
- Vulkan の GroupNorm（未計測。同型の 1 workgroup / group だが nth は未確認）。
- `--naf-native-gn` 以外の CLI フラグの追加（本番で GroupNorm の実装を選ぶ用途は無い）。
- `thirdparty/ggml` の submodule pin の変更、`patches/` への Metal パッチ追加。

## 5. 手順

1. 2.1 実装 → ビルド → N3。
2. 2.2 実装 → N1（Metal、CPU 参考値）。**ここで (b) > (a) × 1.5 なら停止して (ii) へ**。
3. 2.3 実装 → N2（T=1024 / T=512、CPU の所要時間次第で構成を落とす）。
4. P1 → P2 → P2-num（`ab_gn.sh`、約 1 時間、他セッションの GPU 使用と重なれば延びる）。
5. E1（2 本 × 約 30 分）。
6. `nth_patch.py` 相当で submodule 作業ツリーにパッチを当てて `build-metal-nth` を作り U1 → 作業ツリーを復元（`git -C thirdparty/ggml checkout .`）。
   パッチ + PR 本文を `docs/results/…/upstream/` に置き、go-queue に登録。
7. docs: `naf.h` コメント、`docs/PIXAL3D_WEBGPU_OP_GAP.md` / `docs/PIXAL3D_WEBGPU_MEMORY.md` §11 / `docs/spec/30-pixal3d-cond.md` §4（lowering が
   WebGPU 専用でなくなる）、`docs/results/2026-09-20-conditioning-profile.md` §4.3（follow-up へのリンク）、`docs/GGML_FORK_DIFF.md`。
8. 結果を `docs/results/2026-09-20-metal-groupnorm.md` に書き、codex の反証レビュー → PR。
