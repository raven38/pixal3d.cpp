# #55 Metal GroupNorm 結果ドキュメントの反証レビュー（codex gpt-5.6-luna、read-only、2026-09-20 20:20〜20:25）

対象: `docs/results/2026-09-20-metal-groupnorm.md`（5be941c 時点）、設計 doc の凍結基準、raw ログ。プロンプトは末尾。

## codex の出力（原文）

- **重大 — §9.1、scripts/ab_gn2.sh:14–18、docs/design/…:106,112–114**  
  凍結基準の静穏ゲートは 25%未満・90秒なのに、実測は 40%未満・30秒へ緩和されている。「合格条件は動かしていない」は不正確で、P1/P2 の測定前提を変更している。元条件で再計測するか、P1/P2を「条件変更下の参考値」に下げるべき。

- **重大 — §4:89–92, §9.2:231–234、scripts/paired_gn.py:5–10**  
  §4は「36 run中35 clean」と書く一方、補充後の実数は P1 38 run、37 clean。さらに `paired_gn.py` は `--contam contam.txt` を渡さない限り汚染runを除外しない。集計コマンドと表の母数を一致させ、補充前後・除外後のnを明記すべき。

- **重大 — §4:109–112、raw `ab2/old_nafops_T1024_183631.log:86–110`**  
  凍結基準の旧T1024範囲は6.0–6.3 sだが、実測に6.41 sがある。結果自身も逸脱を認めているのに、8基準すべてPASSとしている。旧側の基準を「中央値のみ」と事前定義していない限り、PASSではなく逸脱付き判定にすべき。

- **重要 — §9.1:221–244、scripts/classify_runs.py:31–52**  
  run中GPU使用率を判定に使えない理由の説明は妥当だが、6 blockのばらつきが小さいことは「背景GPU負荷が順位を変えなかった」証明ではない。特に同一worktree配下の別trellisは `selfkey` 除外で見逃し得る。検出限界と未検証範囲を明記し、性能結論を「この測定条件で」に限定すべき。

- **重要 — §0:15–27、§7:162–187**  
  PyTorch fixture不在、E2Eが別noiseサンプル、汚染run除外、E1が品質非回帰の証明でない点は先頭付近に記載されており確認済み。ただし冒頭の「8基準すべてPASS。既定にしてよい」は、直後の重大な測定条件変更・別サンプル比較より強く、過大主張になっている。「元の凍結ゲートでは未判定」を先に出すべき。

- **重要 — §8:191–209、design §2.4:81–87、upstream/PR_BODY.md:21–28**  
  U1は速度・数値・標準テストの実測を裏付けるが、追加barrierの必要性やraceを裏付けていない。`test-backend-ops` PASSやnth=32→1024の改善はbarrierの因果証明ではない。「形式的に安全側のbarrier」と表現し、最小race再現またはMetalの同期検証が得られるまで「必要」と断定しないべき。

- **軽微 — upstream/PR_BODY.md:8–12, nth/gn_oracle_nth.log:29–35**  
  PR本文の `ggml_norm` 8 msは非nthの別計測（結果§1では8〜16 ms）で、U1 rawでは10.8 ms。主張自体は測定系列内に存在するが、「U1で裏付けた値」と読める構成ではない。測定runと値の出典を明示すべき。

- **確認済み — §1 / numerics/n1_metal_sync.log:29–35, n1_cpu_sync.log:2–8**  
  Metal 2.1744e-3→9.5367e-7、466.4→16.1 ms、CPU側の選択反転を含め、表の主要値はrawと一致する。

- **確認済み — §4 / ab2 rawログ**  
  T1024 2.72/6.27 s、T512 2.45/6.00 s、LR 0.66/1.54 s、およびGroupNorm/NORM合計はrawと一致する。ただし上記のrun数・旧T1024逸脱は別問題。

- **確認済み — §5 / ab2 rawログ**  
  tex 17.36/31.54 s、HR 13.50/27.69 s、LR 3.48/7.01 s、メモリ+513 MBはrawと一致する。

- **確認済み — §6 / ab2/num/compare.txt:1–6**  
  global max差0、proj L2rel 4.069e-5/3.915e-5/4.386e-5、finite=Trueは表と一致する。

- **確認済み — §7 / e2e/e2e_old.log:148,191,275, e2e_new.log:148,189,273, e2e/compare_glb_pair.txt:1–12**  
  active voxel 4438一致、cond_slat 94.7→65.2 s、V/F・bbox・NN距離の記載はrawと一致する。5視点レンダも確認できるが、別サンプル比較なので品質非回帰の証明ではない。

- **確認済み — §8 / nth/gn_oracle_nth.log:29–35、nth/nafops_T1024_nth_native_{1,2}.log:86–114**  
  nthパッチの1.24e-5、16.1 ms、graph 2.78 s、標準テスト2/2はrawと一致する。ただしbarrierの必要性は未検証。

マージを止めるべき指摘: **Yes**。凍結ゲートを緩和したまま「8基準すべてPASS」としている点が解消されるまで止めるべきです。

## 応答（2026-09-20 20:30、結果 doc に反映済み）

| 指摘 | 対応 |
|---|---|
| 重大: 凍結ゲート（90 s）を 40 %/30 s に緩めたまま「全 PASS」 | 認める。§0 冒頭を「7 基準 PASS + P1 逸脱付き PASS、P1/P2 は凍結計測条件ではない」に書き換え。凍結条件（25 %/90 s、E1 と同じ）での 2 block × 6 config の再確認 `scripts/ab_gn3_strict.sh` → `ab3_strict/` を 20:27 に投入（GPU が 90 s 静穏になる時間帯にしか進まない。完了後に §4 / §5 へ併記）。既取得の凍結条件データ（E1、`ab2/pre_restart/` 4 run）と 40 %/30 s の値が 1 % 以内で一致することは根拠として書くが、証明ではないと明記 |
| 重大: run 数（36/35）が補充後の実数（38/37）と不一致、`--contam` 無しでは除外されない | §4 を 38/37 に修正し、集計コマンド（`paired_gn.py ab2 --contam ab2/contam.txt`）を明記 |
| 重大: 旧 T1024 の 6.41 s は凍結範囲 6.0〜6.3 の外なのに全 PASS | P1 を「逸脱付き PASS」に格下げ（§0）。旧側の基準を中央値で判定する事前定義は無かった |
| 重要: ばらつきが小さいことは背景 GPU 負荷が順位を変えなかった証明ではない、`selfkey` 除外で同一 worktree の別 trellis を見逃す | §9.1 に検出限界を明記。自 worktree のバイナリ同士はロックで直列化しており、ps スナップショット 164 本に trellis が 2 つ以上写った瞬間は 0（機械確認）。結論を「この計測条件での値」に限定 |
| 重要: §0 の「全 PASS・既定にしてよい」が過大 | 上記の書き換えで、凍結ゲート未判定を先頭に出した |
| 重要: barrier の必要性は未検証 | §8 と `upstream/PR_BODY.md` を「安全側の追加、race は再現していない、`test-backend-ops` は両ハンク込みで実行」に修正 |
| 軽微: PR 本文の `ggml_norm` 8 ms の出典 | 「8〜16 ms（U1 と同 run では 10.8 ms）」に修正 |
| 確認済み: §1 / §4 / §5 / §6 / §7 / §8 の主要値は raw と一致 | — |

「マージを止めるべき: Yes」への扱い: PR #56 は draft を解除してレビューに出すが、本文冒頭に「凍結条件での確認は進行中」を明記し、マージの判断はユーザーに委ねる。

**追記（21:20）**: 凍結条件の確認 `ab3_strict/`（2 block × 6 config = 24 run、20:29〜21:16、24/24 clean）が完了。全 config で 40 %/30 s 系列と 0.5 % 以内、
pair 差は 0.02 s 以内で一致し、旧 T1024 も 6.26 s で凍結範囲内。結果 doc §0 / §4 / §5 / §9 に併記した。これで指摘 1 は解消。

## プロンプト

あなたは反証指向のレビュアーです。日本語で答えてください。対象は pixal3d.cpp の issue #55（Metal の NAF GroupNorm を `ggml_norm` 再表現に切り替える PR #56）の結果ドキュメントです。「正しいか確認」ではなく「壊れている証拠・過大主張・基準のすり替え・数値と本文の不整合」を探してください。

読むもの（このリポジトリ内、read-only）:
- 結果: docs/results/2026-09-20-metal-groupnorm.md
- 設計と凍結基準: docs/design/2026-09-20-metal-groupnorm.md（§受け入れ基準の表 N1/N2/N3/P1/P2/P2-num/E1/U1）
- 前回のあなたの設計レビューと応答: docs/reviews/2026-09-20_metal-groupnorm_design_review.md
- raw ログ: docs/results/2026-09-20-metal-groupnorm/{numerics,ab2,e2e,nth}/ と scripts/（paired_gn.py, classify_runs.py, summarize_gn.py, compare_cond.py, gate_lib.sh, ab_gn2.sh）
- 実装: src/naf_gpu.cpp（naf_ggml_opts_for, group_norm_affine）, src/test_naf.cpp（--gn-oracle）, src/test_pixal3d_cond_tex.cpp（--naf-compare-cpu）

特に確認してほしいこと:
1. 結果 doc の各表の数値が raw ログの値と一致しているか（サンプリングで良いので、少なくとも §1・§4・§5・§7・§8 から各 3 値以上を突き合わせる）。
2. 判定が設計 doc の凍結基準に対してなされているか。基準を後から緩めた・言い換えた箇所があれば指摘（§9.1 に計測ゲートの変更履歴があるが、合格条件そのものは動かしていないつもり）。
3. §0 の結論に、最悪ケース・逸脱・未検証（PyTorch NAF fixture がこの Mac に無い、E2E の GLB 差は別サンプル、汚染 run の除外）が先頭で明記されているか。過大主張の文があれば具体的に引用して指摘。
4. 汚染判定の妥当性: classify_runs.py の規則、run 中の GPU 使用率が判定に使えない理由の記述、6 block のばらつきの扱い。
5. upstream 提案（docs/results/2026-09-20-metal-groupnorm/upstream/PR_BODY.md と 0001-metal-group-norm-nth.patch）の主張が U1 の実測で裏付けられているか。barrier の必要性の論証に穴がないか。

出力形式: 重要度順の箇条書き。各項目に file:line（または §番号）と、何が問題で、どう直すべきかを 1〜3 行で。問題が無い項目は「確認済み」と一行で。最後に「マージを止めるべき指摘」があるかを Yes/No で。
