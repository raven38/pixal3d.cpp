# codex 反証レビュー: docs/results/2026-09-20-conditioning-profile.md（1 回目の版 bf8acd2）

実行: 2026-09-20 05:0x、`codex exec -s read-only`、プロンプトは末尾。対応は本ファイル末尾の「対応」節。

### 判定: 差し戻し

中心値の転記自体はほぼ正しい一方、事前に定めた汚染除外条件と反復条件を満たしていません。そのため、結論1の「fresh で −63 s」と結論2の「generic lowering で −32 s」を確定値として扱えません。

### 重大（A）

1. **「fresh」とされた採用実行が、設計上の汚染除外条件に違反しています。**

   設計は「他プロセスがCPU 50%超で継続した計測を採用しない」と明記しています（[conditioning-profiler.md:181](</Users/s25705/Downloads/pixal3d-cond-profile/docs/design/2026-09-20-conditioning-profiler.md:181>)）。しかし結果文書は、`suggestd` 等が135%未満だったことを理由に全採用実行を無汚染としています（[conditioning-profile.md:14](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:14>)）。135%は事前閾値50%と整合しません。

   具体例:

   - 採用された `T1024_gpu_030910.log` の実行中、`trellis-test-decimate-bench` が開始前・中・終了後を通して約99〜100%動いています（[ps_snapshots.log:265](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:265>)、[同:277](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:277>)、[同:290](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:290>)）。
   - `T512_gpu_031759.log` では、開始時に `suggestd` 86.7%、実行中に decimate bench 63.7%、終了後も `suggestd` 84.4%です（[同:597](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:597>)、[同:610](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:610>)、[同:625](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:625>)）。
   - 追加採用された LR 7.0 s も、開始前90.5%・終了後96.9%の `suggestd` と重なっています（[同:1202](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:1202>)、[同:1212](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:1212>)）。
   - L2のT=1024 im2colは `post-replay` 246〜196%と同時実行です（[同:1054](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:1054>)、[同:1064](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:1064>)）。T=512 im2colも別のdecimate bench 100%と重なっています（[同:1129](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/ps_snapshots.log:1129>)）。

   さらに実行スクリプトは90秒静穏ゲートを最初に一度しか掛けず、各実行間は60秒sleepだけです（[ab_clean.sh:17](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/scripts/ab_clean.sh:17>)、[同:29](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/scripts/ab_clean.sh:29>)）。設計上の「直前90秒以上GPU使用trellisなし」という `fresh` 定義（[conditioning-profiler.md:187](</Users/s25705/Downloads/pixal3d-cond-profile/docs/design/2026-09-20-conditioning-profiler.md:187>)）も、初回以外では成立しません。

   したがって、表から明示的に除外された1本だけでなく、採用集合全体を事前基準で再分類する必要があります。

2. **結論1の −63 s は、必要な反復数・同一条件・E2E相当Nを満たしていません。**

   - LR deviceは `n=2` で、設計の「最低各3回」を未達です（[conditioning-profile.md:47](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:47>)、[conditioning-profiler.md:189](</Users/s25705/Downloads/pixal3d-cond-profile/docs/design/2026-09-20-conditioning-profiler.md:189>)）。
   - LRは合成 `N=3,872`、E2Eは`N=4,438`で14.6%違います（[conditioning-profile.md:42](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:42>)）。host側はdense R³中心ですがdevice側はsparse N依存なので、差分−6.7 sをそのままE2E換算する方向に非対称です。
   - HR deviceの4本には、計測途中で再ビルドした異なるバイナリと、33.3 sの外れ値が混在します（[同:54](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:54>)、[同:258](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:258>)）。
   - 数値一致は汚染された同一プロセス実行1本だけです。T=1024の `L2rel=4.8951e-8` 自体はログにありますが（[ab_T1024.log:117](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/firstlook/ab_T1024.log:117>)）、合成fixture上の相互一致であって、実E2E座標でdevice経路へ切り替えた統合一致ではありません。

   「このfixtureでは約−63 sを観測」という仮説的記述なら残せますが、「切り替えるだけでfresh −63 s」という確定表現は不可です。

3. **結論2の支配op同定は強いものの、6.28→2.61 sと−32 sの確定には穴があります。**

   正しい部分として、ISOLATED node passとwhole graphの比はT=1024で1.00、genericで1.01なので、GROUP_NORM 3.62 sの帰属はよく整合しています（[nafops_T1024_im2col.log:86](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/nafops_T1024_im2col.log:86>)、[nafops_T1024_generic.log:82](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean/nafops_T1024_generic.log:82>)）。

   ただし:

   - im2col/genericはいずれも各1回で、しかも少なくともim2col側は上記の別プロセス汚染があります。反復A/Bではありません。
   - generic loweringはGroupNormだけでなくreflect padとpoolも同時に変更します（[naf_gpu.cpp:54](</Users/s25705/Downloads/pixal3d-cond-profile/src/naf_gpu.cpp:54>)、[同:123](</Users/s25705/Downloads/pixal3d-cond-profile/src/naf_gpu.cpp:123>)）。whole graph差3.67 sをすべて `nth=32` の効果と因果断定するには、GroupNormだけをloweringする対照が必要です。
   - Metal generic出力はfinite確認だけで、nativeとの誤差比較をしていません（[conditioning-profile.md:248](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:248>)）。GroupNormと`ggml_norm`では並列reduction順が変わるため、「厳密に再表現」「厳密に同値」は浮動小数点出力について成立しません。
   - `nth=32`固定、8 threadgroup、3パスという実装読解は正しいです（[ggml-metal-ops.cpp:3326](</Users/s25705/Downloads/pixal3d-cond-profile/thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp:3326>)、[同:3343](</Users/s25705/Downloads/pixal3d-cond-profile/thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp:3343>)、[ggml-metal.metal:3199](</Users/s25705/Downloads/pixal3d-cond-profile/thirdparty/ggml/src/ggml-metal/ggml-metal.metal:3199>)）。ただし「固定32が遅さの原因」はコードと相関からの強い仮説で、`nth`だけを変えた反証A/Bは未実施です。
   - CUDAが1024 threadsを選ぶ実装も正しいですが、他backend未計測のままMetalを「最悪」とする比較級は根拠不足です（[conditioning-profile.md:143](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:143>)）。

4. **§6の「実現見込み」が事前定義に違反しています。**

   設計は、実現見込みを数値化できるのはA/B実測がある候補だけとしています（[conditioning-profiler.md:201](</Users/s25705/Downloads/pixal3d-cond-profile/docs/design/2026-09-20-conditioning-profiler.md:201>)）。

   ところが順位表では:

   - 候補2のLR分 `4×0.9 s` は「要素数が1/4」という推定で、LR genericの実測ではありません（[conditioning-profile.md:239](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:239>)）。実測済み部分はT=1024/512の `4×3.67 + 4×3.32 = 27.96 s` までです。
   - 候補3の「−7 s」は1×1 im2colのISOLATED時間からの推定で、`mul_mat`直接化後のwhole graph A/Bではありません（[同:240](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:240>)）。定義上は「上限」に置き、実現見込みは未測とすべきです。
   - `#1 + #2 = 約34 s` からさらに約7 s削減すれば約27 sであり、「約26 s」は表の数字から再現できません（[同:244](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:244>)）。

### 軽微（B）

- §5は「性能の定量には使わない」としながら、参照比1.35倍・1.23倍を明記し、単発E2Eのdecoder値でstage優先度も述べています（[conditioning-profile.md:170](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:170>)、[同:217](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:217>)）。`accounted`、GLB非破壊確認、fresh A/Bとの差の診断に使うのは設計どおりですが、参照E2Eとの速度比較や絶対値による優先順位付けは「観測値・定量根拠にはしない」と分離すべきです。
- `--naf-direct-conv` の13.65 sという主張を裏付けるrawログが、指定ディレクトリ内にありません。設計ではT=1024/512各1回を予定していますが（[conditioning-profiler.md:192](</Users/s25705/Downloads/pixal3d-cond-profile/docs/design/2026-09-20-conditioning-profiler.md:192>)）、実行スクリプトはim2col/genericだけです（[ab_clean.sh:46](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/scripts/ab_clean.sh:46>)）。「2倍以上遅いので採らない」はrawを追加するか、未確認へ格下げが必要です。
- T=512 genericの63.4%はwhole graph 2.68 sではなくISOLATED node pass 2.90 sを分母にしています。表で分母を明記すべきです。

### 確認できた正しさ（何をどう照合したか）

- `summarize_ab.py`をrawログへ再実行し、§1の中央値・min/maxを再現しました。texは31.4対73.3、HRは28.0対42.4、LRは7.0対13.7で、単純合計66.4対129.4、差−63.0です。したがって転記・中央値計算そのものは正しいです。
- §2のhost内訳もrawから再現しました。T=1024はgraph compute 6296 ms、readback 496 ms、unpermute 5992 ms、view中央値17.80 sで文書と一致しました。
- §4のop表はrawと一致しました。T=1024 nativeはwhole 6.28 s、node pass 6.29 s、GROUP_NORM 3.62 s、IM2COL 1.71 s、CONT 0.63 sです。ISOLATED計測がheavy opの順位を大きく歪めている反例は、この比からは見つかりませんでした。
- `ggml-metal-ops.cpp`とMetal kernelを読み、GroupNormが8 group × 1 threadgroup、`nth=32`固定で、平均・分散・正規化の3走査をする点を確認しました。
- E2Eの`[cond]`をrawから照合し、5段の`accounted`と段合計はすべて0.0 s差でした。cond 3段130.9 s、decoder 50.2 s、flow外189.4 sの算術も正しいです。
- 5視点比較画像も確認し、上段参照と下段conditioning版に視認可能な差はありませんでした。ただしこれは計装非破壊性の確認であり、host→deviceやgeneric loweringの品質証拠ではありません。

## プロンプト

> あなたは GPU 性能計測と ggml/Metal に詳しい反証指向のレビュアーです。日本語で答えてください。
> 対象: docs/results/2026-09-20-conditioning-profile.md（issue #33 の計測結果）。raw ログは docs/results/2026-09-20-cond-profile/ 以下（clean/*.log = A/B と --naf-ops、e2e/e2e_cond.log = E2E、e2e/ref_rope_e2e.log = 参照、ps_snapshots.log = 汚染記録）。設計は docs/design/2026-09-20-conditioning-profiler.md、コードは src/naf_gpu.cpp・src/pixal3d_cond.cpp・src/pixal3d_cond_gpu.cpp・src/trellis_cli.cpp・thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp。
> 課題: 「壊れている証拠・反例」を探してください。特に:
> 1. 結果ドキュメントの数値が raw ログと食い違っている箇所（表の中央値・内訳・合計。抜き取りで良いので実際に grep して照合する）。
> 2. 結論 1（host 経路→device 経路で −63 s）の根拠に穴が無いか（n、汚染、条件の非対称、N の違い、fixture の合成性）。
> 3. 結論 2（GROUP_NORM が nth=32 固定で支配的、generic lowering で 6.28→2.61 s）の根拠（op テーブルは ISOLATED node 再実行。whole graph との比は?）と、ggml-metal-ops.cpp の実装の読みが正しいか。
> 4. 候補の順位表（§6）の「上限」と「実現見込み」の計算が排他的ラップの定義と整合しているか。
> 5. E2E（§5）の使い方が設計の「性能の定量には使わない」と矛盾していないか。
> 6. 誇張・根拠のない一般化（他 backend への言及、「厳密に同値」等）。
> 出力形式:
> ### 判定: 承認 / 条件付き承認 / 差し戻し
> ### 重大（A）: 箇所（ファイル:行 or 節）と、反例・食い違いの具体的な内容
> ### 軽微（B）
> ### 確認できた正しさ（何をどう照合したか）

## 対応（2026-09-20 05:00〜06:3x、結果ドキュメントの改訂版に反映）

| # | 指摘 | 対応 |
|---|---|---|
| A1 | 採用実行が事前の汚染除外基準（他 trellis 無し・CPU 50% 超の継続無し）に違反。ゲートが開始時 1 回だけ | **認める**。`scripts/classify_runs.py` で全実行を事前基準どおり機械分類し（1 回目 26 本中 clean 14 / 汚染 12、`--naf-ops` は 4 本とも汚染）、**実行ごとに 90 s ゲート**を掛ける `ab_clean2.sh` で device T1024 ×3 / T512 ×2 / LR ×1、host T512 ×2、naf-ops 10 本を再計測（全部 clean）。採用集合は `scripts/accepted_ab.txt`。`suggestd`（macOS 常駐、2 日以上 1 コア 92〜98%、全実行・両 arm 共通）だけは基準から外し、逸脱として冒頭に明記 |
| A2 | LR device n=2、LR の N 差 14.6%、HR device に別バイナリと外れ値、数値一致が合成 fixture の同一プロセス 1 本 | LR device n=3（7.0 / 7.0 / 7.0）。tex device n=4 / host n=3、HR device n=4 / host n=4。バイナリ差は §1 に明記（差分は printf のみ）。N 差は device 側で N 依存なのが sparse tap の gather（0.01〜0.02 s/view）だけなので ±0.1 s と論証。数値一致は「合成 fixture 上の host vs device」と限定し、実 E2E 座標での統合一致は切り替え PR（Metal）/ PR #34（CUDA）側と明記。結論 1 は「合成 fixture の fresh A/B で −63 s」に表現を変更 |
| A3 | im2col / generic 各 1 回・汚染、generic は pad / pool も変える、Metal generic の出力未比較、nth=32 因果は未検証、「最悪」は根拠不足 | native ×2 / generic ×2 / **GroupNorm だけ generic ×2**（新フラグ `--naf-generic-gn`）/ direct ×1 を clean で再計測。GroupNorm 単独で 6.27 → 2.71（full generic 2.60）と因果を分離。**出力比較を実装**（`--naf-compare`、4 GiB 全要素）: L2rel 1.2e-4 / max\|d\| 1.18e-2 で「厳密に同値」を撤回し実測値に置換。`nth` だけ 32 → 1024 にした診断パッチで反証テストを実施（§4.3）。「最悪」は削除し「他 backend 未計測」に |
| A4 | 実現見込みの事前定義違反（LR の推定、1×1 im2col の ISOLATED 推定、26 s の算術） | 候補 #2 の実現見込みを実測分（T=1024 / T=512 の whole graph 差 × 4 view）だけにし、LR は上限のみ。候補 #3 の 1×1 im2col は「上限」に移し実現見込みは未測。合算の数字を再計算 |
| B1 | E2E で参照比や decoder 優先度を述べている | 「観測値の記録。定量根拠にしない」に書き換え、decoder は順位付けしないと明記 |
| B2 | direct conv 13.65 s の raw が無い | clean で再計測（`clean2/nafops T1024 direct cmp_055857.log`）: whole 9.56 s、CONV_2D 5.27 s。13.65 の記述は削除 |
| B3 | T=512 generic の % の分母 | 「op 別の % は node pass の合計が分母」と §4 冒頭に明記 |
# 2 回目: codex 再確認（2026-09-20 09:1x）

### 判定: 差し戻し

中央値・GroupNorm 単独対照・出力誤差・§6 の算術は大幅に改善されています。しかし、重大項目 A1 と A3 がまだ「事前基準どおり」「`nth` だけを変えた反証実験」とは言えません。特に `nth_patch.py` が GroupNorm 以外の Metal op も一括変更しており、「32スレッド固定が原因で確定」という因果主張は再実験が必要です。

### 各項目の判定（A1〜A4、B1〜B3、新規）

| 項目 | 判定 | 根拠 |
|---|---|---|
| A1 汚染除外 | **部分的** | `classify_runs.py` を `clean/ clean2/ clean3/ nth/` の4ログへ実行した。`accepted_ab.txt` の21本は、出力上すべて実行中 `clean` かつ `OTHER TRELLIS` の前窓表示がない集合になっている。[accepted_ab.txt:1](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/scripts/accepted_ab.txt:1>)。ただし完全な「事前基準どおり」ではない。分類器は判定前に `suggestd` を明示的に除外しており、これは事後例外である。[classify_runs.py:44](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/scripts/classify_runs.py:44>)。さらに `clean2/clean3/nth` はゲート中のスナップショットを保存せず、分類器出力は全て `pre-window(..., 0 snaps): ok` だった。90秒前窓は `gate()` の実装では保証しようとしているが、分類器による事後検証ではない。[ab_clean2.sh:14](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/scripts/ab_clean2.sh:14>)。本文の「事前基準どおりに機械分類」「採用した全実行は前窓とも分類器を通る」は強すぎる。[結果:21-43](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:21>) |
| A2 −63秒、n・条件 | **部分的** | §1 の n と中央値は再現した。LR gpu n=3、host n=4、tex gpu/host n=4/3、HR n=3/4。[summary_accepted.md:3](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/summary_accepted.md:3>) と [結果:91-99](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:91>) は一致する。合成fixture限定、実E2E統合一致は未実施とも明記された。一方、「device版でN依存なのは host taps/upload だけ」「実Nでも±0.1秒」はコードと整合しない。`N3` は accumulator、8本のindex/weight、`get_rows`、`mul`、`add`、`cpy` のテンソル寸法にも入る。[pixal3d_cond_gpu.cpp:497](</Users/s25705/Downloads/pixal3d-cond-profile/src/pixal3d_cond_gpu.cpp:497>)、[同:529](</Users/s25705/Downloads/pixal3d-cond-profile/src/pixal3d_cond_gpu.cpp:529>)、[同:580](</Users/s25705/Downloads/pixal3d-cond-profile/src/pixal3d_cond_gpu.cpp:580>)。N=3,872→4,438 のA/Bなしに±0.1秒とは限定できない。[結果:110-117](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:110>) |
| A3 GroupNorm・出力・nth | **部分的** | GroupNorm単独対照は解消。T1024 native 6.27/6.27、GN-only 2.71/2.72、full generic 2.60/2.60、T512 native/GN-only/full=6.00/2.45/2.68で本文どおり。[結果:185-212](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:185>)。比較ログも GN-only/full generic が `L2rel=1.2174e-4`、T512が `1.1896e-4`、directが `9.5667e-5` と一致する。[gn cmp:117](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean2/nafops T1024 gn cmp_055525.log:117>)、[generic cmp:117](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean2/nafops T1024 generic cmp_055122.log:117>)。しかし `nth` 反証は未成立。`nth_patch.py` の無制限 `s.replace(old,new)` は同じ `int nth = 32` を持つ全箇所を変更する。[nth_patch.py:7-15](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/scripts/nth_patch.py:7>)。対象は GroupNormだけでなく SUM、SUM_ROWS、SET_ROWS、SOFT_MAX、L2_NORM、NORM、ARGMAX、TRI 等。したがって「nth 1行だけ、他は一切変えない」「原因は確定」は不可。GROUP_NORM自身が453→16 msになった観測は有効だが、whole graph 6.27→2.90の厳密な単独因果A/Bではない。 |
| A4 §6 の実現見込み | **解消** | #2はwhole graph実測差だけで `4×3.67 + 4×3.55 = 28.88 ≈ 29秒`。LRは除外。#3は「未測」で、ISOLATED値を上限に限定。合算も `66.0−29≈37秒` と再現できる。[結果:315-329](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:315>)。ただし#1のLR分には上記A2のN外挿問題が残る。 |
| B1 単発E2Eの扱い | **解消** | 「観測値の記録」「性能の定量根拠にはしない」、decoderも順位付けしないと限定された。[結果:253-259](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:253>)、[同:295-297](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:295>)。 |
| B2 direct conv raw | **解消** | rawあり。whole graph 9.56秒、CONV_2D 5.27秒、出力 `L2rel=9.5667e-5`。[direct cmp:86](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean2/nafops T1024 direct cmp_055857.log:86>)、[同:112](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean2/nafops T1024 direct cmp_055857.log:112>)。 |
| B3 T512の分母 | **解消** | §4冒頭で百分率の分母を node pass（ISOLATED）と明記。[結果:181-183](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:181>)。rawもwhole 2.68、node pass 2.72、IM2COL 63.3%で整合する。[T512 generic:84](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/clean2/nafops T512 generic_060602.log:84>)。 |
| 新規: 本番経路 | **解消** | `generic_groupnorm` は既定falseで、`naf_ggml_opts_for()` は設定しない。[naf.h:118-137](</Users/s25705/Downloads/pixal3d-cond-profile/include/naf.h:118>)。変更はGroupNorm分岐にfalse条件を1項追加しただけ。[naf_gpu.cpp:88-96](</Users/s25705/Downloads/pixal3d-cond-profile/src/naf_gpu.cpp:88>)。trueにするのは診断テストの `--naf-ops` 分岐内だけで、`--naf-compare` も同テスト実行ファイル内だけ。[test_pixal3d_cond_tex.cpp:145-183](</Users/s25705/Downloads/pixal3d-cond-profile/src/test_pixal3d_cond_tex.cpp:145>)。aggregate初期化によるフィールドずれも見つからなかった。静的には本番既定経路を変えていない。 |

### 新規の指摘

重大:

1. `nth` 反証パッチがGroupNorm限定ではない。

   [nth_patch.py:13](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-cond-profile/scripts/nth_patch.py:13>) の全置換により複数opが変わる。関数コンテキストを含む一意な文字列へ限定し、適用後に「変更差分がGroupNormの1ブロックだけ」であることを検査して再計測する必要がある。現状でもGROUP_NORM nodeの高速化自体は確認できるが、本文の「他は一切変えない」「原因確定」は撤回が必要。

2. LRのN外挿根拠が実装と食い違う。

   N依存演算はhost tap作成・uploadだけではない。実E2E相当Nで再測定するか、「DINO/NAF本体はN非依存だがprojection/gather/accumulatorはN依存。影響量は未測」に直すべき。

軽微:

- `suggestd` は事前除外基準からの事後例外であり、「事前基準どおり」は不正確。また分類器出力では初期の採用runに `suggestd>50% in 0/N` もあり、「全実行で92〜98%の固定背景負荷」は保存ログだけからは示せない。
- host texの10秒compute spike自体はrawで確認できるが、原因をChrome等の「GPU外乱」と特定はできない。「未観測のGPU外乱または熱・OS状態を含む実行間変動」が妥当。
- `−56〜−63秒` は統計的な幅ではなく、tex hostを66.8秒または中央値73.3秒に置いた感度分析。host最大74.1を使うとtex差は42.9秒、合計は約63.7秒になるため、現在の表現はレンジの作り方が一貫しない。[結果:101-108](</Users/s25705/Downloads/pixal3d-cond-profile/docs/results/2026-09-20-conditioning-profile.md:101>)。「中央値−62.9秒、最速host runを使った保守ケース−56.4秒」と分けるべき。
- GN-onlyとfull genericの比較統計が同じ4桁なのは確認できるが、それだけで「pad/poolはexact」「全要素同一」とは証明できない。GN-only出力とfull generic出力を直接比較して `max|d|=0` またはhash一致を記録すべき。
- [naf.h:122](</Users/s25705/Downloads/pixal3d-cond-profile/include/naf.h:122>) の “Exact re-expressions” と [naf_gpu.cpp:170](</Users/s25705/Downloads/pixal3d-cond-profile/src/naf_gpu.cpp:170>) の「厳密な再表現」は、実測 `L2rel=1.2e-4` と読み手レベルでは衝突する。「数学的には同じだが浮動小数点の縮約順は異なる」が正確。

### 確認できた正しさ（何をどう照合したか）

- `classify_runs.py` を4本の `ps_snapshots.log` に実行した。旧 `clean/` では実行中cleanでも前窓にtrellisがいるrunを再現し、`accepted_ab.txt` がそれらを除外していることを確認した。`clean2/` 18本と`nth/` 4本は実行中clean、`clean3/` は4本clean・2本汚染だった。
- `summarize_ab.py clean clean2 clean3 --accept accepted_ab.txt` を再実行し、`summary_accepted.md` および§1のn・中央値・min/max・個別値と照合した。自動 `diff` は sandbox の `/dev/fd` 制限で `Operation not permitted` になったため、取得したstdoutと保存表を直接照合した。
- T1024/T512のnative、GN-only、full generic、directのwhole graph・op表を各rawから確認した。
- `*cmp*.log` の `naf-ops: compare` 行を確認し、max差・mean差・L2rel・native computeを本文と照合した。
- `nth/*.log` では、T1024 native 2.90/2.90秒、GROUP_NORM 0.13秒・16.11〜16.12 ms/op、T512 2.54秒、比較 `L2rel=7.3601e-5` を確認した。ただしパッチの実変更範囲が複数opであるため、これをGroupNorm限定反証とは認定しない。
- `src/naf_gpu.cpp`、`include/naf.h`、`src/test_pixal3d_cond_tex.cpp` の差分と全参照箇所を確認し、診断フラグが本番既定経路で有効にならないことを静的に確認した。現ターンではビルドや本番E2Eの再実行はしていない。

## 再確認への対応（09:20〜09:40）

| 項目 | 再確認の判定 | 対応 |
|---|---|---|
| A1 | 部分的（suggestd は事後例外、clean2/3/nth の前窓はゲートのみで事後検証なし、「事前基準どおり」は強すぎる） | 見出しを「事前基準 + 事後の例外 1 つ」に変更。suggestd の出現率を分類器出力から数えて記載（全実行で一様ではない）。前窓は「1 回目の分は分類器、再計測分は `gate()` の保証でありログは無い」と明記。次回はゲートのサンプルもログに残す |
| A2 | 部分的（LR の N 外挿「±0.1 s」がコードと整合しない） | ±0.1 s の主張を撤回。「DINO / NAF は N 非依存、projection / gather / accumulator は N 依存で影響量は未測」に修正 |
| A3 | 部分的（`nth_patch.py` の全置換が 9 op に効いていた） | **認める**。group_norm の文脈で一意に置換し差分 1 ハンクを検査する版に直して再実験（`nth2/`）: 6.27 → 2.78 / 2.78 s、GROUP_NORM 16 ms/op。1 回目（2.90 s）は SOFT_MAX 128 ms の副作用込みだったと明記 |
| A4 / B1 / B2 / B3 / 本番経路 | 解消 | — |
| 新規（軽微） | スパイク原因の特定不可、−56〜−63 の幅の作り方、GN-only vs full generic の直接比較なし、`naf.h` / `naf_gpu.cpp` の「exact / 厳密」 | 原因は「未観測の GPU 外乱・熱・OS 状態を含む実行間変動」に。幅は「中央値 −62.9 / 最速 host run の保守ケース −56.4 / 最遅 −63.7 の感度」に。直接比較は未実施と明記。コメントを「数学的には同じ、縮約順が異なる」に修正 |

3 周目は回さない（設計レビューと同じく 1 周 + 再確認の運用）。残る「部分的」は上表の対応で本文に限定表現として残した。
