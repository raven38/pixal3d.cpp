# TASK-STAGES-V3: v3 fixture による shape 全step parity・最終SLat・cascade境界・#77 clamp 実測

- 対象: `src/test_trellis2_mv_shape.cpp`（`trellis-test-trellis2-mv-shape`）。v3 fixture
  `~/nfs/pixal3d_trellis2mv_ref_v3/` を既定 fixture root に変更（v2 は byte 一致でサブセット、
  既存 step0-only テストは無変更で継続 PASS）。
- 設計・codex exec レビュー: `docs/design/2026-09-21-trellis2-mv-stages-v3.md`、
  レビュー全文 `docs/design/2026-09-21-trellis2-mv-stages-v3-codex-review.md`。
- worktree: `~/Downloads/pixal3d-mv-stages`（`feat/trellis2-mv-stages`）。先頭 SHA（着手前 ff）:
  `0d4cfc5`。実装コミット: S1+S2 `935ac7a`、S3 `50277ac`、S4 `439865a`、差し戻し対応
  `456b43d`（§S4差し戻し対応節参照）。
- 実行: `TRELLIS_THREADS=4 cmake --build build --target trellis-test-trellis2-mv-shape -j4
  && TRELLIS_THREADS=4 flock /tmp/pixal3d-metal-gpu.lock -c
  './build/trellis-test-trellis2-mv-shape'`（S3-upsample のみ Metal GPU を使うため lock 下で
  実行、COMMON.md §6 のスレッド制約に合わせ `TRELLIS_THREADS=4` で再実行）。全出力:
  `full_run_log.txt`（同ディレクトリ、971行）。**真の hard assert（`check()`）の内訳**:
  `grep -c "^ok "` = 603、`grep -c "^FAIL"` = 0。**informational（`check()` を経ない printf）**:
  `grep -c "^info "` = 1（S3-upsample の q8_0 診断のみ——他の informational は全て §S4差し戻し対応
  で hard assert へ昇格済み）。プロセス exit code = 0。S5（tex 全step、任意）は優先度低のため
  見送り。

## 結論（先に）

1. **S1（全12step CFG parity）+ S2（x_t chain・最終SLat）**: shape-512 全6run・
   shape-1024cascade（LR+HR）全6run、計18 run×side で `sample_flow_multi(steps=12,
   trace=...)` を1回だけ呼ぶ full-trajectory replay を実施。native 自身の trace から
   逆算した CFG 予測は fixture `cfg_output` と全12step・全run で **max rel = 9.7e-7**
   （閾値1e-3を大幅に下回る）。x_t chain（steps1-11）+ 最終SLat（denorm後）も
   fixture と **max rel = 4.2e-7**（閾値1e-4を大幅に下回る）。
2. **step8/step9 否定対照**: step8（inside）の `cfg_output` は `positive_0`/`avg` から
   **rel 0.603〜1.00**（全18点）乖離——「outside と仮定した式」では説明できない。
   step9（outside）は全点 `rel < 1e-12`（trivial 一致）。
3. **S3-quantize（model不要）**: fixture の `shape_hr_coords_prequantize.npy` を native の
   quantize backoff ロジック（`trellis_cli.cpp:982-997` の複製、pure arithmetic）に通した
   結果は、全6 cascade run で `shape_slat_coords.npy` と**完全一致**（set一致、native
   N=fixture N、diff 0）。
4. **S3-upsample（q8_0, best-effort）**: `run_1img_baseline_1024c` 1本で実施。native
   1,385,777 候補座標 vs fixture 1,390,129（差 5,528/9,880、約0.4-0.7%）。この Mac には
   `shape_dec.gguf` の q8_0 版しか無く（f16 は NFS 退避済み）、fixture 参照側は元の
   PyTorch 重みで実行しているため、**この差が native バグか量子化ノイズかは切り分けられない**
   （設計時から明示的に「期待値を変えない」情報提供のみと位置づけ、hard failure にしない）。
5. **S4（#77 clamp 実測）**: shape LR/HR・全run・guided な9step（step0-8）、計
   **162測定点**で native ratio を実測。**最小0.2462、最大0.9575**、全点
   `isfinite && ratio in [0.2,5.0]`（clamp no-op の判定式）を満たす——**clampは1点も
   発火しない**。manifest の `ood_ratio` と native の rel は最大1.97e-7（実質一致、
   複製算術の正しさの独立確認）。追加で、production（clamp ON）と `TRELLIS_NOFIX=1`
   （clamp OFF）を**別プロセス**で実行し 12-step trace を `memcmp` した結果、
   **全18 run×side で bit-identical**（clamp・NaN sanitizer 双方が no-op であることの、
   ratio算術に依存しない独立証拠）。**判断はしない**（keep/removeはユーザー判断、
   統括が issue #77 に転記）。

## 凍結基準との対照表（基準1行=1行。証拠は `full_run_log.txt` の該当行 or 保存済みファイル）

| # | 凍結文面（design §5） | 実測 | 一致 | 証拠 |
|---|---|---|---|---|
| S1 主ゲート | 全12step, LR/HR, 全run: `pred[i]` vs `cfg_output[i]`、rel<1e-3 (parity gate) | 全18 run×side, max rel=9.7377e-07 | PASS | `full_run_log.txt` の `full-trajectory CFG reconstruction` 行（parity subprocess区間、18件） |
| S1 副対照 | step8否定対照 rel>0.1（設計改訂: 当初1e-2→事前実測後0.1に確定、design§0.5） | 全18点 rel 0.603〜1.00 | PASS | `full-trajectory` セクション各run `step8 (inside)` 行 |
| S1 trivial | step9 trivial rel<1e-12（design§0.5、当初未凍結→codexレビュー後に追加） | 全18点 rel=0.0000e+00 | PASS | 各run `step9 (outside)` 行 |
| S2 主ゲート | 全12step x_t chain (i=0..10)、rel<1e-4 | 全18 run×side、上記CFGと同一計測内でmax rel=4.1673e-07 | PASS | `full-trajectory x_t chain` 行 |
| S2 最終SLat | denorm後 (`SHAPE_STD*x+SHAPE_MEAN`) rel<1e-4 | 同上（x_t chainと同一assertに統合、design§1の通り） | PASS | 同上 |
| S3 quantize | fixture prequantize→native quantize、完全一致（集合） | 全6 run、native N=fixture N、diff 0 | PASS | `S3-quantize: native quantize(...) == fixture ... exactly` 行×6 |
| S3 upsample | q8_0、best-effort、不一致は情報として報告のみ・期待値は変えない | native 1,385,777 vs fixture 1,390,129（diff 5,528/9,880） | INFO（PASS/FAILの対象外、design通り） | `S3-upsample: ... MISMATCH` 行 + diff例5件 |
| S4 clamp条件 | `isfinite(ratio) && 0.2<=ratio<=5.0`（codexレビューで下限のみ→両側+finite に修正）を**162行each個別にhard assert**（差し戻し対応、旧実装は集計後 `if(would_fire==0) check(true,...)` という tautology で1点もhard assertしていなかった） | 162行 `check()` 全PASS、min=0.2462, max=0.9575 | PASS | `S4: ... OOD clamp guard is no-op` 行×162 |
| S4 manifest照合 | rel（許容誤差は事前未指定→実測後 1e-6 級であることを doc に明記して確定） | 全162点 max rel=1.97e-07 | PASS（参考、hard gateではない） | `summary: ... max rel vs manifest ood_ratio=1.97e-07` 行 |
| S4 bit-identical | NOFIX/production trace が別プロセスでbit-identical | 全18 run×side 一致 | PASS | `production vs TRELLIS_NOFIX=1 full-trajectory trace is bit-identical` 行×18 |

事後変更: 上記のうちS1副対照の閾値（1e-2→0.1）とS1 trivial/S4 clamp条件式の2点は、
**実装前** の codex exec レビュー + 事前 numpy 実測を経て設計ドキュメント側で確定させたもので、
実装後に閾値を緩めた事後変更ではない（design doc §0.5 に実測根拠込みで記録済み）。

## S4差し戻し対応（2026-09-22）

外部レビューで3点の指摘を受け、コードのみ修正（凍結した数値・閾値は変更なし）:

1. **`src/test_trellis2_mv_shape.cpp` の S4 集計ロジックが構造的にtautologyだった**:
   旧実装は `would_fire_count` を数えたあと `if (would_fire_count==0) check(true, ...)` と
   していたため、この分岐に入る時点で条件は既に真——`check(true,...)` は何を検証しても
   必ずPASSする空虚な assert で、1点も実質的にhard assertされていなかった（else 側は
   printfのみで、発火時も個々の行がFAILとして報告される経路が無かった）。
   **修正**: ループ内で行ごとに `check(no_fire, tag + ": OOD clamp guard is no-op (...)")`
   を呼ぶ形に変更（162行、各行が独立した合否）。missing行も `check(false, ...)` に変更。
   集計 printf は「informational aggregate、実ゲートは上のper-row check」と明記して残す。
2. **production-gate の informational 36件（`run_full_trajectory_512/1024c` の
   `hard_assert=false` 呼び出し、x_t chain + CFG reconstruction × 18 run-side）**:
   S4修正でclampが全162測定点で発火しないことが個別hard assertで確立され、かつ
   NOFIX/production trace のbit-identical証拠（S4、18 run-side）もあるため、`main()` を
   並べ替えてS4を先に実行し、両方成立した場合のみ `hard_assert=true` へ昇格するよう変更
   （`promote_production_gate = s4_ratio_ok && s4_bitid_ok`）。実測ではこの Mac の v3 fixture
   全体で両方成立したため、**36件全てhard assertへ昇格**（`[production gate, informational]`
   タグが出力に0件になったことで確認、`full_run_log.txt`）。降格せず残すべき情報は無かった
   （残っている唯一のinformationalはS3-upsampleのq8_0診断で、これは§0-4の理由により意図的に
   hard assertにしない——native バグと量子化ノイズを区別できないため）。
3. **design doc §0.5 が実在しないパス `docs/design/../codex-review` を参照していた**:
   実際の codex exec 出力（プロンプト全文＋最終回答）を
   `docs/design/2026-09-21-trellis2-mv-stages-v3-codex-review.md` として保存し、
   design doc の参照をこの実在パスへ修正した。
4. `TRELLIS_THREADS=4` で再実行（COMMON.md §6）、結果は変わらず全PASS
   （§冒頭の実行ログ参照、603 ok / 0 FAIL / informational 1件）。

## hardening への merge 後の再実行（2026-09-22、統括）

`feat/trellis2-mv-stages`（元 tip 221af49）を `feat/trellis2-mv-hardening`（B3 決着 3 コミット込み、
tip e2a9add）へ rebase し、`git merge --ff-only` で hardening に取り込んだ（rebase 後 tip **b367532**、
src/ の内容は 221af49 と差分ゼロ: `git diff --stat 221af49 HEAD -- src/` が空）。その上で同じコマンドで再実行:

```
TRELLIS_THREADS=4 cmake --build build --target trellis-test-trellis2-mv-shape trellis-test-trellis2-mv-tex -j4   # rc=0
TRELLIS_THREADS=4 flock -w 600 /tmp/pixal3d-metal-gpu.lock -c './build/trellis-test-trellis2-mv-shape'           # rc=0
TRELLIS_THREADS=4 flock -w 600 /tmp/pixal3d-metal-gpu.lock -c './build/trellis-test-trellis2-mv-tex'             # rc=0
```

| 項目 | 凍結値（rebase 前、`full_run_log.txt`） | 再実行実測（`full_run_log_hardening_b367532.txt`） | 一致 |
|---|---|---|---|
| shape: 真の check（`grep -c '^ok '`） | 603 | **603** | 一致 |
| shape: `grep -c '^FAIL'` | 0 | **0** | 一致 |
| shape: informational（`grep -c '^info '`） | 1（S3-upsample q8_0 のみ） | **1** | 一致 |
| shape: プロセス exit code | 0 | **0** | 一致 |
| shape: ログ全文 | — | ポインタアドレス（`0x…`）と時間表示を伏せた差分 **0 行** | 一致 |
| tex: `^ok ` / `^FAIL` / exit | （本 doc 対象外、参考） | **150 / 0 / 0**（`tex_run_log_hardening_b367532.txt`） | — |

Metal GPU は他セッションと共有のため `/tmp/pixal3d-metal-gpu.lock` 下で実行した。

## 既知の限界・申し送り

- S3-upsample は q8_0 decoder 限定の diagnostic。native バグの有無を判定するには f16（or 元の
  PyTorch と同精度）decoder が必要——NFS 退避済みの f16 セットを戻すか、再変換が必要（統括判断）。
- S4 の #77 判断（clampの keep/remove）はこのタスクでは行わない。issue #77 へ実測（162点、
  min 0.246・max 0.958、bit-identical証拠込み）を転記する。
- S5（tex 全step full-trajectory）は brief で任意指定・優先度低のため未実施。

Generated with Claude Code
