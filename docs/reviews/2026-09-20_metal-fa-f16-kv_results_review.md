# 結果ドキュメントの反証レビュー（codex exec、2026-09-20、1 周）

対象: `docs/results/2026-09-20-metal-fa-f16-kv.md`（commit 1d61f7c 時点）。依頼は「壊れている証拠・別解釈・過剰な主張を探せ」。
判定は**差し戻し**。対応は末尾「レビュー後の対応」。

### 判定（差し戻し）

§3 の「この M4 Max 実行では `--no-fa` が half オペランドの `kernel_mul_mm_f32_f32` を通る」は確認でき、実測とも整合します。

一方、主要結論の「精度 100 倍改善」「真値から約 0.2%」「F16 overflow リスクなし」「WebGPU でも参照ノイズ以下」はデータが支えていません。また、§6.2 の noise 説明は同一カーディナリティの場合と異なる場合を混同しています。既定変更の判断材料としては差し戻しです。

### 重大な問題

1. §7–§8：「100 倍」は真値に対する精度ではない

対象: [結果 §7–§8](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/docs/results/2026-09-20-metal-fa-f16-kv.md:190>)

数値:

- bf16–no-fa: 2.95–3.78%
- f16–no-fa: 0.020–0.025%
- synthetic op の float64 比: bf16 7.5e-3、f16 3.7e-4、no-fa 4.5e-4

Q2 が示すのは「tex flow の固定 fixture で f16 が no-fa に約160倍近い」だけです。no-fa 自身が真値でないため、「既定を f16 にすれば精度が100倍改善」は成立しません。文書自身が認める「f16 と no-fa が共通誤差を持ち、両方とも真値から3%離れる」可能性は排除できていません。

さらに「違いは累積順序だけ」も誤りです。no-fa の2段目 `V × softmax` では、F32 softmax 出力も half にステージされます。一方、FA f16 の softmax scratch は float です。[FA の型定義](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/thirdparty/ggml/src/ggml-metal/ggml-metal.metal:5859>) と [no-fa の2回の mul_mat](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/src/dit.cpp:373>) から、両者は次の点でも異なります。

- materialized softmax と online softmax
- softmax 確率の half 丸めの有無
- タイル分割・正規化・累積順序

解消条件:

- 実 tex fixture の実 Q/K/V を複数 step・block から dump する。
- 同じオペランドを Metal bf16/f16/no-fa とホスト float64 に通す。
- attention entropy別に十分な query を比較する。
- full-flow の絶対精度を主張するなら、真の F32/float64 attention を使った flow 参照、または各 attention 誤差を追跡できる stepwise oracle を追加する。

それまでは「100倍」は「no-fa への距離」に限定してください。`~0.2%` 外挿は削除すべきです。

2. §3 の機構はこの実行では正しいが、適用範囲を限定すべき

対象: [結果 §3](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/docs/results/2026-09-20-metal-fa-f16-kv.md:51>)

確認結果:

- no-fa は QK と PV の両方を `ggml_mul_mat` で構成しています。
- native の 1 GiB chunk では N=17612 に対して通常 chunk の `ne11=1270`、最終 chunk も1102で、いずれも `ne11 > 8` です。
- M4 Max は `has_simdgroup_mm=true`、入力は `ggml_cont` 済みなので、[dispatch 条件](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp:2057>) により `mul_mm` が選ばれます。
- 実ログにも `kernel_mul_mm_f32_f32` の compile が記録されています。
- この M4 Max では tensor API が既定で無効です。legacy kernel の [実体化](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/thirdparty/ggml/src/ggml-metal/ggml-metal.metal:10203>) は F32 device inputをhalf threadgroup行列へ変換し、float accumulatorを使います。

したがって exact 4.5e-4、FA f16 3.7e-4、CPU exact 1.4e-7という順序は十分整合します。

ただし「Metal の `--no-fa` は常にこの経路」は広すぎます。`ne11 <= 8`、転置、simdgroup非対応、tensor API有効化では分岐が変わります。

解消条件:

- 「本実験の M4 Max、tensor API無効、native既定chunkでは」と限定する。
- 環境欄に `has_tensor=false` を追加する。
- oracle log に実際の QK/PV pipeline 名と `ne11` を保存する。

3. §5：速度の方向は支持されるが、「Textureで有意に4.5%遅い」は脆い

対象: [結果 §5](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/docs/results/2026-09-20-metal-fa-f16-kv.md:93>)

paired log-ratio と t-CI は再計算して一致しました。

- Shape: 0.9741 [0.9322, 1.0178]
- Texture: 0.9548 [0.9220, 0.9888]

ただし以下の穴があります。

- HR は6順序中5順序だけで、完全な順序均衡ではありません。
- fwd1→fwd2 はbf16/f16でも概ね2–4%、f32では3–14%増えており、warm-up後にもドリフトが残っています。
- Texture block 4 の f16=15.35 s は他の14.38–14.57 sから外れています。
- leave-one-block-outでは、TextureのCI上限が block 0除外で1.002、block 2除外で1.005となり、「95%有意」が維持されません。
- n=5ではlog-ratioの正規性を検証できません。
- 前後だけのプロセスsnapshotでは、run途中だけ重なった負荷を除外できません。

ただし5/5 blockで `bf16/f16 < 1` なので、「この測定でf16が速い証拠はない」「遅い方向は一貫」は維持できます。

解消条件:

- 欠けたBAC順序を含め最低6 block、望ましくは12 block。
- 温度・電力・GPU residencyをrun中も記録。
- randomization/permutation CIまたはbootstrap感度分析を併記。
- 「有意に4.5%遅い」を「約3–5%遅い方向。小標本CIは感度あり」へ弱める。

なお `bf16/f16=0.955` は、時間比では f16が `1/0.955−1 = 4.7%` 長い、throughputでは約4.5%低い、が正確です。

4. §6.2：noise結合の説明が2種類のずれを混同している

対象: [結果 §6.2](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/docs/results/2026-09-20-metal-fa-f16-kv.md:142>)、[noise実装](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/src/trellis_cli.cpp:393>)

保存済みdumpを再確認すると:

- bf16/f16 のSS voxel数は両方4438、共通4435。
- bf16/f16 のHR `shape_noise.npy` は先頭17614行が実際にbit-exact。
- ただし座標集合・ソート位置が違うため、同じ行のnoiseが異なるvoxelへ割り当てられる。
- no-faはSS voxel数が4440なので、LRで消費する乱数個数自体が違い、後段のRNG stateもずれます。bf16–no-faのHR noiseは行単位でも一致しません。
- bf16/f16でもHR token数が17614/17625なので、その後のtex noise開始位置はずれ、tex noiseは行単位で一致しません。

従って「集合が1 voxelでも違えば、その行以降すべて別noise」は一般には不正確です。

- カーディナリティ同一: 行noiseは同じだが、座標への対応が局所区間でずれる。
- カーディナリティ相違: RNG消費量が変わり、後続stage全体のnoise系列もずれる。

設計ゲートへの影響:

- 3aはdense SSなので有効。
- 3bのJaccard/GLBはK/V精度検定ではなく、離散化とnoise再割当を含むE2E安定性・破綻検査に降格すべき。
- `1.41 ≈ √2` はdecorrelationの証拠にはなりますが、「独立サンプル」と断定するには独立seed baselineが必要です。

解消条件:

- 説明を上記2ケースに分ける。
- stageごとに固定noiseを入力するfixture試験をshape HRにも追加。
- 3bを「判定不能」ではなく「元の因果ゲートとして無効」と明示する。

5. §0・§8：範囲とWebGPUの主張が測定範囲を超えている

対象: [結果冒頭](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/docs/results/2026-09-20-metal-fa-f16-kv.md:9>)、[三分岐結論](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/docs/results/2026-09-20-metal-fa-f16-kv.md:243>)

問題:

- 「F16 の overflow リスクは無い」はVを1入力でしか測っていない事実と矛盾します。
- Q/Kについては静的上界があるため強い主張が可能ですが、Vはcyclops 1入力でheadroom 12646倍を観測しただけです。
- Q2はtex flowのみ、かつbf16 E2E由来のconditioningを固定した試験です。shape flowや別conditioningには一般化できません。
- SSではf16/no-faとbf16/no-faの比は約6.9倍、synthetic opでは8–20倍です。「精度100倍」はtex Q2固有です。
- WebGPUは別backend・別kernelです。Metal Q2から「リスクはV量子化だけ」「参照ノイズ以下」は導けません。Q/K丸め、softmax、累積方式も変数です。
- 「速度ゲート不通過なのでholdout不要」としながら、精度目的の既定変更を選択肢に残すのは不整合です。精度理由で変更するならholdoutは必要です。

解消条件:

- 冒頭を「Q/K overflowは静的に排除。V overflowはcyclops 1入力では観測なし」に修正。
- belleなど性質の異なる2入力目でV範囲を測る。
- shape HR fixtureでもpaired試験を行う。
- WebGPUではWebGPU自身のop oracleとE2E fixtureを実施する。
- 「100倍」は「cyclops由来tex fixtureのno-fa距離」に限定する。

6. §3：64 queryサンプルからmax誤差・局所機構を強く言いすぎている

対象: [oracle実装](</Users/<redacted-user>/Downloads/pixal3d-fa-f16-kv/src/test_fa_oracle.cpp:52>)

`max|d|` は17612 query全体ではなく、等間隔に選んだ64 query、全体の約0.36%に対する最大です。「peaked queryで局所的に大きく外す」は候補としては妥当ですが、実装はattention entropyとの対応を測っていません。

解消条件:

- 表を `sampled max|d|` と改名。
- queryごとのentropy、最大softmax重み、誤差を併記。
- 少なくとも最大/最小entropy層化サンプル、可能なら全queryのstreaming oracleを追加。

### 軽微な指摘

- 「exact softmax」は数学的exactではなく、「materialized non-FA softmax」と呼ぶ方が安全です。
- `4.5e-4/op` をflowの「参照ノイズ床」と呼べません。入力分布・出力・演算回数が異なり、しかも決定的誤差です。
- §7の「1 stepあたり約×1.5」「増幅率は同じ」は端点と一致しません。端点からの幾何率は概算でbf16が約1.60–1.67倍/step、f16が約1.37–1.41倍/stepです。
- f32が約2倍遅い事実は支持されますが、「帯域が倍だから」という因果はcounter計測なしでは仮説です。
- 4視点レンダは確認しました。3腕とも致命的破綻は見えませんが、生成物が別サンプル化しているため、これは精度順位ではなく安全性確認に限られます。
- Q2の5 seedは初期noiseの反復であり、conditioning/inputの反復ではありません。実質的な入力数は1です。

### 別解釈・反証の候補

- f16とno-faは共通するhalf丸めにより近接し、両方が真値から大きく偏っている。
- bf16差の一部は「q=bfloat」ではなく、half O累積の長系列停滞が主因。q型とO累積を独立にA/Bしない限り寄与分解できない。
- Q2の大きな比は、12-step反復系が小さい初期差を非線形に増幅した結果で、単発attention精度の一般的な倍率ではない。
- Textureの速度差はdtype差ではなく、残留熱・順序・block 4の状態差を含む可能性がある。
- HR/GLB差はK/V型の直接効果より、SS threshold付近の数voxel差→noise再割当→別trajectoryという離散分岐が支配している。
- `no-fa` がFA f16より常に高精度とは限りません。synthetic flatではFA f16 3.7e-4がno-fa 4.5e-4より小さく、f32 FAは2.1e-4です。参照名ではなく各腕を独立にfloat64へ比較すべきです。


## レビュー後の対応（同日、commit で反映）

| # | 指摘 | 対応 |
|---|---|---|
| 1 | 「100 倍」は真値精度ではない、~0.2% 外挿は削除、f16/no-fa は累積順序以外も違う | §0/§7/§8 を「no-fa 経路への距離」に限定し、外挿を削除。no-fa が P も half にステージすること（materialized softmax）と FA の float scratch の違いを §7 に明記。「真値に対する flow 精度は未測定」を限界に追加。実 Q/K/V の op oracle・F32 attention の flow 参照は**未実施**（follow-up） |
| 2 | §3 の機構は本環境限定 | 環境欄に tensor API 無効を追加、§3 に dispatch 条件（ne11 = 1270 / 1102 > 8、cont、simdgroup_mm、legacy mul_mm）と適用範囲の限定を追記。oracle log への pipeline 名 / ne11 の保存は**未実施** |
| 3 | Texture「有意に 4.5% 遅い」は脆い | 「約 3–5% 遅い方向、leave-one-out で CI が 1 を跨ぐ」に弱め、block 4 の外れ値・順序 5/6・fwd1→fwd2 ドリフト・snapshot の限界を §5 に明記。6〜12 block の追加計測・bootstrap は**未実施** |
| 4 | noise 結合の 2 ケース混同 | §6.2 を「voxel 数同じ / 違う」の 2 ケースに分け、3b を「因果ゲートとして無効、安定性検査に降格」と明記。独立 seed の対照は**未実施** |
| 5 | 範囲・WebGPU・holdout の過剰主張 | §0 を「Q/K は静的に排除、V は 1 入力で観測なし」に修正。WebGPU は「示唆に留まる」に変更。ゲート 5 の行を「精度理由で変えるなら holdout 必要」に修正。「100 倍」を tex fixture 限定に |
| 6 | 64 query サンプルの max\|d\| | 「sampled max\|d\|」と明記し、entropy との対応は未測定と記載。層化サンプル・全 query oracle は**未実施** |
| 軽微 | 「exact softmax」→ materialized、増幅率 ×1.67 / ×1.41、f32 帯域は仮説、5 seed は入力 1 本 | すべて反映 |

未実施項目はいずれも「既定を f16 に変える」判断の前に必要な追加計測で、本 issue（速度ゲート不通過、既定不変）の結論は変わらない。
