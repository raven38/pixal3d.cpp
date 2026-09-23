# TASK-STAGES-V3 設計レビュー（codex exec、2026-09-21）

`docs/design/2026-09-21-trellis2-mv-stages-v3.md` の実装前設計レビュー。
`codex exec --sandbox read-only`（model gpt-5.6-sol、reasoning effort medium）で実施。
レビュー結論を実装へ反映した節は同ファイル §0.5 参照。

## プロンプト

```
あなたはC++/GGML/PyTorch数値パリティに詳しいレビュアーです。以下の設計ドキュメントを実装前にレビューしてください。対象リポジトリは /Users/<redacted-user>/Downloads/pixal3d-mv-stages（trellis.cpp のフォーク、pixal3d.cpp）。

設計ドキュメント全文:
---
# TASK-STAGES-V3: v3 fixture による shape 全 step parity・最終 SLat・cascade 境界・#77 clamp 実測

対象 issue: trellis2-mv エピック。前任の設計 `2026-09-21-trellis2-mv-stages-shape-tex.md`（v2 fixture、
step{0,8,11}のみ）を拡張する。fixture: `~/nfs/pixal3d_trellis2mv_ref_v3/`（カタログ:
`~/Downloads/pixal3d-mv-59/docs/spec/33-trellis2-mv-reference.md` §2.4）。

## 0. 事前実測（本実装の前に numpy で確認した事実。これが無いと設計が誤る）

1. **x_t のチェイン整合は正規化空間（モデル出力そのまま）**。`shape_lr_step0_xt_feats.npy` から
   `x1_pred = x0 - dt0*cfg0` を計算し `shape_lr_step1_xt_feats.npy` と比較すると
   `rel = 5.0e-8`（float64 で再計算）。denorm 不要。
2. **`shape_lr_final_slat_feats.npy` / `shape_slat_feats.npy`（最終 SLat 系ファイル）は
   `SHAPE_STD*x+SHAPE_MEAN` を適用した後の値（非正規化空間）**。step11 の x_t と cfg から
   `x11 - dt11*cfg11` を計算し正規化空間のまま比較すると `rel=0.82`（不一致）、
   `SHAPE_STD/MEAN` を適用してから比較すると `rel=9.5e-8`（HR）/ `1.2e-7`（LR）で一致。
   `trellis_cli.cpp:973`（`lr_dn = lr_norm*SHAPE_STD+SHAPE_MEAN`）と同じ変換。
3. **quantize 式は native と fixture 側の独立検証コードで完全に一致**（`(int)((c+0.5)/512*g)`
   切り捨てキャスト、`g=hr_res/16`）。fixture 生成スクリプト
   `~/nfs/pixal3d_trellis2mv_ref_v3/scripts/ref_trellis2_mv_pod_run_v3.py:597-610` の
   `((prequant[:,1:].float()+0.5)/512*(res//16)).int()` は native
   `trellis_cli.cpp:985`/`test_trellis2_mv_shape.cpp` の `quantize_hr_backoff` と同一の式
   （torch `.int()` は 0 方向切り捨てで C++ `(int)` キャストと同じ）。
4. **`shape_dec.gguf` はこの Mac には q8_0 版しかない**（`~/nfs/weights/pixal3d/gguf-q8_0/`,
   841MB。f16 版は無し、NFS へ退避済み — `pixal3d-release-model-sets` メモリ参照）。
   fixture の参照側は元の PyTorch 重み（f32/bf16 相当）で upsample を実行しているため、
   S3-upsample を q8_0 decoder で走らせた場合、不一致が出ても「native のバグ」と
   「量子化ノイズ」を区別できない。これを結果に明記する（期待値は変えない — 一致しない
   場合は差集合サイズ・両側にしかない座標例を報告するだけで、閾値緩和はしない）。

## 1. S1+S2: 全 12 step の CFG parity + 最終 SLat を単一の trace 呼び出しで検証する

前任の `sp.steps=1` トリック（step0 だけ ts1[0]==1.0 が常に成り立つことを利用）は
`step i>0` に一般化できない（`flow_t_schedule` は steps 引数から schedule を作り直すため、
`i` step目の t 値は steps=12 の schedule 固有の値で、部分 schedule では再現できない）。

代わりに **12 step 通しで `sample_flow_multi(..., sp.steps=12, trace=&tr)` を 1 回呼ぶ**。
ReplayModel は各 step の記録済み `pos{0..V-1}`/`neg` を、production の呼び出し順序
（stochastic: 1回 pos + 条件付き1回 neg / multidiffusion: V回 pos + 条件付き1回 neg）で
そのまま返す（`build_replay_one_step` を 12 step 分チェインするだけ、CFG/rescale 自体は
複製しない）。`trace[i]`（0-indexed、i=0..11）は step i 実行後の sample。

- **x_t 検証（S2 の主ゲート）**: `trace[i]` (i=0..10) を `x_t[i+1]`（fixture、正規化空間）と
  直接比較。`trace[11]` は `SHAPE_STD*x+SHAPE_MEAN` を適用してから `shape_lr_final_slat_feats`
  （LR）/ `shape_slat_feats`（HR、cascade でない run は `shape_slat_feats` そのもの）と比較。
  閾値: `rel = max|a-b|/max(|b|,1e-12) < 1e-4`（参照側の独自 Euler 再計算が 3.4e-7 なので
  native の float32 累積を込みでも 1e-4 は十分緩い事前値。実測がこれを超えたら緩和せず
  finding として書く）。
- **CFG 逆算検証（S1 の主ゲート）**: `pred[i] = (x_t[i] - trace[i]) / dt[i]` を全 step で
  `cfg_output[i]` と比較。dt が小さい step（dt0≈0.029）は割り算で丸めが増幅されるため、
  x_t 比較より緩い副ゲート `rel < 1e-3` とする（前任の step0 実測 rel 1e-6〜1e-3 帯を踏襲）。
- **step8 の否定対照（inside 確認）**: fixture データのみを使う純算術。step8 の `cfg_output`
  と `positive_0`（stochastic）/`avg`（multidiffusion）を直接比較し `rel > 1e-2` であることを
  assert する（「outside と仮定した式 pred=pos」では説明できないほど乖離している、の実証。
  step9（outside 確定）で同じ比較をすると `rel` が生成的に極小 or 0 になることも併記して対比）。
  加えて ReplayModel の消費ログから「step8 は neg ファイルを消費、step9 は消費しない」ことを
  構造的に assert（既存 `is_neg_expected` 機構の延長）。
- clamp ON/OFF の両ゲート（production / `TRELLIS_NOFIX=1` self-re-exec）を維持。production gate
  は #77 の ratio が既知の帯（0.2 未満なら informational）に従う（S4 と同じ判定を各 step に適用）。

## 2. S3: cascade 境界の自己導出

**quantize 半分（model 不要、決定的）**: fixture の `shape_hr_coords_prequantize.npy`
（[N,4] int32、列0=batch idx 固定0）を入力に、既存 `quantize_hr_backoff`（列0を除いた
xyz のみ）を `hr_target=1024, max_tokens=49152`（`trellis_cli.cpp` の `cfg.hr_res`/
`cfg.max_tokens` 既定値）で実行し、結果に batch列0を戻して `shape_slat_coords.npy`
（=HR step0 x_t coords、fixture 側で独立確認済み）と集合として完全一致を assert する。

**upsample 半分（model 要、q8_0）**: `shape_lr_final_slat_feats.npy` を latent、
`ss_coords.npy` を coords0 として `shape_upsample(shape_dec.gguf[q8_0], latent, coords0)` を
呼び、結果座標集合を `shape_hr_coords_prequantize.npy` と比較する。**完全一致を主張しない**
（§0-4 の q8_0 confound）。一致すればそのまま assert、不一致なら `check(false, ...)` はせず
`printf("info ...")` で差集合サイズ・双方のみに存在する座標の例（最大5件）を出力し、
「q8_0 decoder のため native バグと量子化差を区別できない」と明記する。実行対象は
`run_1img_baseline_1024c` 1本のみ（GPU/Metal 負荷抑制、COMMON.md §6 のスレッド制約）。

## 3. S4: #77 クランプの shape 発火実測（production gate, 全 run 全 inside step）

`compute_step0_clamp_ratio` を一般化し、任意 step の prefix・x_t を受け取れるようにする
（関数名を `compute_clamp_ratio` に変更、step0 専用の `noise` 引数を `x_t[i]` に一般化)。
RUNS_512 / RUNS_1024C（既存6+6、v2から流用）の全 run・全 inside step（guided な step のみ、
shape は step0..8 の9点）で native ratio を計算し、fixture manifest の `ood_ratio` と比較
（rel、複製算術の妥当性確認）。判断はしない: 0.2 を下回る step の有無だけを表にして
EVIDENCE に出す。全 run で 0.2 以上（v3 doc 実測: 最小 0.246）なら
`check(ratio >= 0.2, ...)` を hard assert する方針を採用（既存 step0 実装と同じ根拠）。
0.2 未満が実際に観測されたら hard assert をやめて info に落とし、その値を doc に明記する
（事後変更として結論節に書く）。

追加で **NOFIX と production の trace が全 step bit-identical**であることを assert する
（clamp が発火していないことの直接証拠。ratio 計算の複製算術に依存しない独立証拠）。

## 4. S5（任意）

tex は `guidance_rescale=0.0` なので `apply_guidance_rescale` が呼ばれず、v2 の
step{0,8,11}比較で本質的な検証はすでに終わっている。時間があれば S2 と同じ trace 機構を
tex にも適用し全 step x_t を確認するが、優先度は低い（brief 通り任意）。

## 5. 凍結する受け入れ基準（結果 doc で対照表を作る対象）

| # | 内容 | 閾値/期待値 |
|---|---|---|
| S1 | 全12step, LR/HR, 全run: `pred[i]` vs `cfg_output[i]` | rel<1e-3 (parity gate) |
| S1 | step8 否定対照 | rel(cfg8, pos8_or_avg8) > 1e-2 |
| S2 | 全12step x_t chain (i=0..10) | rel<1e-4 |
| S2 | 最終SLat (denorm後) | rel<1e-4 |
| S3 | quantize (fixture prequantize -> native quantize) | 完全一致（集合) |
| S3 | upsample (q8_0, run_1img_baseline_1024cのみ) | best-effort、不一致は情報として報告のみ |
| S4 | 全run 全insideコップ ratio | manifest比 rel、ratio>=0.2 (実測次第で緩和せず finding化) |
| S4 | NOFIX/production trace bit-identical | 完全一致 |

## 6. 実装ファイル

`src/test_trellis2_mv_shape.cpp` のみ変更（既存 §1-§4 を維持しつつ、新しい "full trajectory"
セクションを追加。fixture root 既定を v2 から v3 に変更）。tex は変更なし（S5 見送りなら）。
各 S を個別コミットにする（S1+S2 は同一トレイルの単一実装のため1コミットにまとめる—
brief の「各項目を別コミット」に対し、実装が不可分なため統合する旨を commit message に明記）。
---

参考: 既存実装 src/test_trellis2_mv_shape.cpp（v2 fixture、step{0,8,11}のみ対応）と
include/flow_runner.h / src/flow_runner.cpp（sample_flow_multi, apply_guidance_rescale,
flow_t_schedule）はこのリポジトリ内にあります。必要なら Read してください。

レビュー観点:
1. S1+S2 の「12 step 通しで sample_flow_multi(steps=12, trace=&tr) を 1 回呼び、ReplayModel
   に fixture の全step分の pos/neg を注入する」方式は、CFG/rescale ロジックを複製せず本番
   コードをそのまま検証する設計として妥当か。見落としている罠はあるか
   （例: trace の push タイミング、guided/inside 境界でのReplayModel消費順序のズレ、
   sp.steps=12 で使う SamplerParams が本番 shape_production_params() と完全一致しているか）。
2. 最終SLatの正規化（SHAPE_STD*x+SHAPE_MEANを適用してから比較）は実測で確認済みだが、
   実装コード上でこの変換をテスト側に書くことは許容範囲か（trellis_cli.cppの複製ではなく
   同じ定数を使った素朴なアフィン変換）。
3. S3のquantize半分（model不要）とupsample半分（q8_0 decoder、ベストエフォート）の切り分けは
   妥当か。q8_0 confoundの扱い（不一致でもcheck(false)にしない）は「期待値を変えない」原則に
   抵触しないか。
4. S4のNOFIX/production trace bit-identical assertionは、clampが発火していないことの
   独立した証拠として有効か。
5. 閾値（S1: rel<1e-3, S2: rel<1e-4, step8否定対照: rel>1e-2）は妥当か、緩すぎ/厳しすぎないか。
6. その他、実装前に直すべき設計上の欠陥があれば指摘してください。

簡潔に（重要度順、箇条書き）回答してください。致命的な問題がなければ「実装に進んでよい」と
明言してください。
```

## 回答（最終メッセージ）

- **重要: S1 の CFG 逆算式を修正してください。** `pred[i]` の左辺は fixture の `x_t[i]` ではなく、sampler が実際に使用した状態、すなわち `prev = (i==0 ? initial_x_t : trace[i-1])` として `(prev-trace[i])/dt[i]` にすべきです。既存 SS テストもこの方式です（[test_trellis2_mv_ss.cpp](/Users/<redacted-user>/Downloads/pixal3d-mv-stages/src/test_trellis2_mv_ss.cpp:505)）。fixture の `x_t[i]` は独立した S2 比較に使います。

- **重要: NOFIX/production の bit-identical 比較方法が未定義です。** `no_fix` は関数内 `static const` なので、同一プロセスで `setenv` を切り替えても再評価されません。別プロセスで両 trace を生成し、全 float バイト列を一時ファイル経由で比較するなど、具体的な受け渡しが必要です。なお一致は有効な証拠ですが、clamp だけでなく非 finite pred の sanitizer も含めた「production guards 全体が no-op」の証拠です（[flow_runner.cpp](/Users/<redacted-user>/Downloads/pixal3d-mv-stages/src/flow_runner.cpp:605)）。

- **重要: S4 の clamp 条件は下限だけでは不十分です。** 実装の clamp 範囲は `[0.2, 5.0]` かつ finite です（[flow_runner.cpp](/Users/<redacted-user>/Downloads/pixal3d-mv-stages/src/flow_runner.cpp:431)）。`ratio >= 0.2` ではなく `isfinite(ratio) && 0.2 <= ratio && ratio <= 5.0` を判定してください。また manifest 比較の許容誤差が未指定なので、全 run の事前実測から固定する必要があります。

- **S1+S2 の通し replay 自体は妥当です。** trace は Euler 更新後に push されているため、`trace[i] == fixture x_t[i+1]` の対応は正しいです（[flow_runner.cpp](/Users/<redacted-user>/Downloads/pixal3d-mv-stages/src/flow_runner.cpp:606)）。ReplayModel の「全 pos→neg」という順序も現在の stochastic/multidiffusion 実装と一致します。cascade HR には本番同様、LRから継続した `stochastic_counter` を渡してください。現行 V=1/2/4 では12を割り切るため結果差は出ませんが、配線契約として必要です（[trellis_cli.cpp](/Users/<redacted-user>/Downloads/pixal3d-mv-stages/src/trellis_cli.cpp:927)）。

- **SamplerParams は現在の production と一致しています。** ただし fixture manifest の steps/guidance/rescale/interval/rescale_t とも構造的に照合し、将来の既定値変更を検出してください。各 step の coords も canonical coords と**順序込みで**一致することを確認すべきです。feats は行順に依存するため、集合一致や行数だけでは不十分です。

- **最終 SLat の test-side denorm は許容範囲です。** 素朴な `SHAPE_STD*x+SHAPE_MEAN` で fixture と比較して問題ありません。ただしこれは「sampler の正規化出力を期待定数で変換した parity」であり、CLI の denorm 実装そのものを直接 exercise するテストではない、と結果を限定してください。可能なら定数を共有ヘッダへ寄せるのが望ましいです。

- **S3 の切り分けは妥当です。** quantize は完全一致の hard gate、q8_0 upsample は明示的な diagnostic/observational check としてください。不一致を成功扱いするのではなく、S3-upsample を `INFO（parity未判定）` と記録し、S3 全体の PASS 根拠に数えないなら「期待値を変えない」原則には抵触しません。真の parity gate には同精度 decoder が別途必要です。

- **閾値:** S2 の global-scale relative error `max_abs_error / max_abs_reference < 1e-4`、S1 の `<1e-3` は既存実測を踏まえれば妥当です。式を「要素ごとの相対誤差最大」と誤実装しないよう明記してください。step8 の `>1e-2` は全 run/LR/HR の観測最小値を先に出して余裕を確認し、step9 も単なる併記ではなく `cfg≈pos/avg` を数値 assert すると境界対照が完成します。

以上の3点（CFG逆算、別プロセスtrace比較、S4の両側条件・誤差基準）を設計へ反映後、実装に進んでよいです。方式そのものに致命的な欠陥はありません。
