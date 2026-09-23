# HANDOVER: B3(4view+stochastic res1024 テクスチャ全面黒)調査、圧縮前スナップショット

**圧縮後はまずこのファイルを読んで再開すること。** 元指示: `~/Downloads/claude-work/docs/briefs/trellis2-mv/{COMMON,TASK-B3}.md`
（TASK-B3.md は22:25訂正版: 手順6=view順ログ追加不要、手順7=7a/7b を手順5のpodに相乗り）。
設計doc: `docs/design/2026-09-21-trellis2-mv-b3-tex-black.md`（仮説・codexレビュー・advisorレビュー反映済み）。

## 現在地（要約）

> **【2026-09-22 統括追記・決着】** 本ファイルは調査途中のスナップショットで、「現在地（要約）」の
> 「真因 = tex forward の数値精度」という主張は、その後の pod v9 シード掃引（下記「00:5x 最重要」節）と
> 参照側 23 run 掃引（`docs/results/2026-09-22-trellis2-mv-b3-ref/2026-09-22-b3-ref-r1-results.md`）で
> **前提ごと組み替わっている**。最終的な決着は `2026-09-22-b3-conclusion.md`（同ディレクトリ）を読むこと。
> 本ファイルで参照している診断用の計装（`TRELLIS_DBG_COND_NPY` / `TRELLIS_DBG_COND_OVERRIDE(_512)` /
> `TRELLIS_DBG_TEX_MODE` / `TRELLIS_DBG_MV_STEP` / `TRELLIS_DUMP_TEX_INPUTS` / `TRELLIS_DUMP_TEXSLAT`、
> `trellis-test-trellis2-mv-{tex,shape}-realweight-parity`、`trellis-tool-dump-threshold-matte`、
> `trellis-tool-png-mean`）は **`feat/trellis2-mv-hardening` には含めず、ブランチ `diag/trellis2-mv-b3`
> （tip 2649e61、0d4cfc5 + wip 10 コミット）にのみ存在する**。hardening 側で
> `grep -rn 'TRELLIS_DBG_COND_NPY\|TRELLIS_DBG_COND_OVERRIDE\|TRELLIS_DBG_TEX_MODE\|TRELLIS_DUMP_TEX' src/ CMakeLists.txt`
> が 0 件なのは意図どおり（`TRELLIS_DBG_COND` 単体は `src/pixal3d_cond_gpu.cpp:270` に 0d4cfc5 以前から在る別物で、
> B3 の計装とは無関係）。


**仮説a/b/c/d/e/fは全て静的・実測で棄却/精緻化し、真因を「tex flow (tex_flow_1024.gguf) の
forward計算自体がPyTorch参照とstep0から既に約1.3%ずれており、V=4 stochasticの12ステップ
逐次積分で5.1%まで拡大し、decoderを飽和させる」まで特定した。** コード上のロジックバグ
（cond取り違え・counter持ち越し・decoder入力破損）ではなく、**tex_flow_1024の数値精度
（f16 GGUF疑い）がV=4 stochasticという使われ方の下でのみ致命的に拡大する**という性質の
バグ/現象。まだ「どのレイヤ/演算が精度落ちの発生源か」までは特定できていない
（次の一手参照）。単純な1行修正では直らない可能性が高い。

## 二分の全記録（時系列）

### 1. 仮説列挙（`docs/design/...md` 本文）

a. stochasticカウンタがshape LR→HRからtexへ持ち越される → **棄却**（`docs/results/2026-09-21-trellis2-mv-reference.md`
   の凍結仕様「SS and texture each get a fresh counter」と、コード実測（`sample_bank`呼び出しで
   texにcounterを渡していない=nullptr）で確認）。
b. tex段がcond1024でなくcond512を参照 → **棄却**（実測 `mixed=0, tbank.size()=4, tlc=4101`）。
c. decoder/bake側の入力破損 → **保留→ほぼ棄却**（texlatは有限・統計正常範囲、mode分離実験で
   texlat自体はstoch/multidiffでほぼ同水準なのにdecode出力だけ激変=decoder"だけ"の問題ではなく
   texlatの構造的な違いにdecoderが超敏感、という形に精緻化）。
d. 背景除去失敗 → **棄却**（同じcond1024_bankをshape HRも使い、shape HRは正常）。
e. NaNの無音握り潰し（safety net） → **棄却**（全ステージ・全ステップでnonfinite=0、
   `TRELLIS_DBG_NAN`のdit-nanログも発火無し）。
f. 特定viewがtexモデルに対してだけ破綻、stochasticは希釈されず直撃 → **部分的に棄却**
   （per-step `[flow-mv-step]`ログでview0-3のmax|finite|が6.1-6.9と揃っており特定view突出なし）。

### 2. mode分離実験（advisor提案、diag pod v4で実施、pod名`trellis2mv-b3-diag`、削除済み）

`TRELLIS_DBG_TEX_MODE`環境変数を新設（`trellis_cli.cpp`、tex段のsample_bank呼び出しだけ
mode差し替え、shape/SS段は元のmodeのまま）。2実験:

| 実験 | shape/SS mode | tex mode | colors mean/std | 結果 |
|---|---|---|---|---|
| DIAG_shapeStoch_texMultidiff | stochastic V=4(B3幾何) | multidiffusion | 0.0254/0.0657 | **正常** |
| DIAG_shapeMultidiff_texStoch | multidiffusion V=4(B4幾何) | stochastic | 0.0065/0.0317 | **黒に近い** |

**結論: どちらの幾何(shape mode)を使っても、texステージ自身がstochasticなら壊れ、
multidiffusionなら正常。原因はtex段自身のstochastic実行に100%局在する**（shape/cond/counterは
無関係と確定）。

### 3. per-channel分析（同じpodログ、`slat_stats_per_channel`計装）

`DIAG_shapeMultidiff_texStoch`(黒)のtex_decode生出力(pbr, [6,Mv])のch別統計:

```
ch00(R) mean=-0.9868 std=0.0655   ch03(metallic)  mean=0.9973 std=0.0512
ch01(G) mean=-0.9869 std=0.0623   ch04(roughness) mean=0.9987 std=0.0302
ch02(B) mean=-0.9872 std=0.0624   ch05(alpha)     mean=0.9997 std=0.0006
```

RGBが-1(黒)に、MRAが+1に、**全1.36M voxelでほぼ一様に飽和**（stdが極小=分散が潰れている）。
一方その直前のtexlat(tex_decodeの入力, 32ch)は同じ幾何でtex=multidiffusion(正常)の場合と
ほぼ同水準（例: ch01 mean 6.21(正常) vs 6.32(黒)、ch14 mean -6.38(正常) vs -6.56(黒)、
全34chとも同程度の差）→ **texlatの周辺統計(ch別mean/std)だけでは黒化を説明できない**。
decoderが小さな入力差に対して超敏感、または合算統計に現れない構造(voxel間相関等)が原因。
生ログ: `logs/DIAG_shapeStoch_texMultidiff.log`, `logs/DIAG_shapeMultidiff_texStoch.log`。

### 4. 実重み・逐次12ステップ parity probe（★最新・最重要の発見）

新規テスト `src/test_trellis2_mv_tex_realweight_parity.cpp`
（`trellis-test-trellis2-mv-tex-realweight-parity`、CMakeLists.txtに登録済み・ローカルbuild確認済み）。
既存の`test_trellis2_mv_tex.cpp`はReplayModel（記録済みテンソルを返すだけ、cond中身もxも無視）で
1ステップ単独(`sp.steps=1`)のCFG算術しか検証しない、という設計doc既知の限界（fixtureテストの
「64ch concatは検証不能」という自己申告どおり）を突いて、**実際のDitRunner + 実GGUF重みで
fixtureのnoise_tex_0.npy/cond_1024.npy/shape_slat_feats.npy(要SHAPE_MEAN/STDで再正規化)を使い、
本物の12ステップ逐次sample_flow_multi(Stochastic)を1回走らせ**、trace出力から
`pred=(sample_before-sample_after)/dt`で各ステップの生速度を復元し、fixtureの
`tex_step{0,8,11}_cfg.npy`と比較する（cfgはtex production paramsのgs=1.0固定によりpos0と代数的に
同一）。

**結果（pod `trellis2mv-b3-diag` v5、削除済み。`logs/v5-parity-tail.txt`に全文）**:

```
N=4183 V=4 L=4101 Dc=1024
step 0: rel=1.2864e-02  (max|d|=7.81e-02, gmax=6.07)
step 8: rel=4.4532e-02  (max|d|=2.76e-01, gmax=6.19)
step11: rel=5.1215e-02  (max|d|=3.12e-01, gmax=6.08)
```

**解釈（重要）**: step0はまだ native 自身の累積が入っていない一発目のforward
（入力=fixtureのnoise_tex_0.npy、cond=view0そのもの）にも関わらず**既に1.29%の相対誤差**が
出ている。既存fixtureテスト（ReplayModelで記録済みpos0.npyをそのまま返す）がstep0/8/11で
rel~1e-8（実質ビット一致）を返すのとは根本的に違う対象を見ている:
- 既存テスト = 「記録済み正解テンソルを与えたときCFG混合の算術が合っているか」（合っている、ALL PASS）
- 今回のprobe = 「native自身が**その入力から実際にforward()して**同じテンソルを再現できるか」
  （**再現できていない**。step0から1.3%、step11で5.1%まで単調増加）

→ **tex_flow_1024.gguf の実 forward 計算そのものが PyTorch 参照と一致していない**
（cond/counter/decoder/bakeの配線ではなく、DiT forward の数値精度）。V=4 stochasticの
12ステップ逐次積分でこの誤差が拡大し（1.3%→4.5%→5.1%）、最終texlatがdecoderの飽和領域に
入るところまでが確認できた「黒くなる経路」。**V=2(B1)やmultidiffusion(B4)でこの誤差が
無害な範囲に収まる理由（ステップごとの希釈・平均化 or 単に閾値に届かない）はまだ未検証**。

`PARTIAL: B3 stage=` tex_flow_1024のDiT forward計算がPyTorch参照とstep0から1.3%(→11で5.1%)乖離、
cond/counter/decoder配線は正常と確定。次はshape_flow_1024で同じprobeを回し同水準の誤差率が
出るか比較しV=4 stochastic固有か全体精度の問題か切り分け。

## 【2026-09-21 23:4x 更新】判定基準(a)は反証で確定 — 「累積発散」説を撤回

統括指示で以下の判定基準を凍結してから実行:
(a) B1(V=2)でも同程度のrelが出れば「累積発散が原因」説は反証される。
(b) 5%程度の連続的乖離だけでは全面黒を説明するには弱いので、離散的原因（特定step・特定view・
counter異常）の有無をB1/B3/B4で並べて必ず確認する。

pod `trellis2mv-b3-diag`（v6、削除済み）で以下を実行・確認（NFS
`/nfs/trellis2-mv/b3-diag/realweight_parity_{tex_v2img,shapehr_v4img,shapehr_v2img}/`、
ローカル `logs/v6-pod-full.log`）:

| stage / config | step0 rel | step8 rel | step11 rel |
|---|---|---|---|
| tex_flow_1024, V=4 stochastic（B3、黒）| 1.29% | 4.45% | 5.12% |
| tex_flow_1024, V=2 stochastic（B1、正常）| 1.22% | 2.33% | **7.05%** |
| shape_flow_1024 HR, V=4 stochastic（B3幾何）| 4.11% | 57.7% | 83.2% |
| shape_flow_1024 HR, V=2 stochastic（B1幾何）| 3.97% | 81.0% | 63.4% |

**B1(正常)のstep11 rel(7.05%)がB3(黒)の5.12%を上回った → 判定基準(a)成立、「累積発散が
黒化の原因」説は反証・撤回する。** shape_flow_1024はV=4/V=2どちらも57〜83%という遥かに
大きい発散を示すのに幾何は正常（既知事実）ということも、発散量そのものが黒化の予測因子には
ならないことを補強する。

**現在地: 発散の「量」ではなく「構造」を疑うフェーズに切り替え。** 判定基準(b)の
decoder出力比較は新規pod無しで確定済み: `tex_decode base color post-clamp` の std が
B1=0.0428, B4=0.0337に対し**B3=0.0038（9〜11倍小さい）**、meanもB3だけ一桁小さい
（B1=0.0454, B4=0.0238, B3=0.0021）。nan/infは全run 0。texlatの集約(32ch)統計は
B3/B4でさほど違わないのにdecode出力だけ激変する（追補1既出）＝集約統計に現れない構造的な
偏りがdecoderを直撃している可能性が高い。詳細は
`docs/design/2026-09-21-trellis2-mv-b3-tex-black.md` の「追補2」参照。

## 【2026-09-21 23:5x 更新】advisor()指摘で判明した probe の盲点 — 次の一手を差し替え

advisor()に相談した結果、**v6のrealweight parity probeはPyTorch参照のcond_1024.npy（=正解の
DINOv3特徴）をそのまま使っている**ため、「native自身のtex forward計算が正しいか」しか検証して
おらず、「native自身が実際にB3のE2Eで作っているcond1024_bank自体が正しいか」は一度も検証して
いないことが判明した。これはこれまでの計装（`slat_stats("cond_1024 view0 ...")`)が**view0のみ**
（かつ標準化後のmean=0/std=1という、正規化されていれば当然そうなる自明な統計）しか見ておらず、
view1-3の統計を一度も取っていなかったことに起因する構造的な見落とし。

**新しい最有力仮説**: 全runが `birefnet.gguf not found -- falling back to threshold matte`
（閾値マット代替）で前処理しており、これが特定view（特にFrame2/Frame3）で劣化したDINOv3特徴を
生んでいる可能性。B1(正常)は Frame1+Frame4 の2枚しか使わない。B3(黒)はこれに加えて
Frame2・Frame3を使い、Stochastic modeではこの2枚が希釈されず単独でstep 1,5,9(Frame2)・
2,6,10(Frame3)を担当する。B4(正常)はmultidiffusionで4枚を平均するため単独view由来の異常が
薄まる可能性がある。この症状分布（「Frame2/3を単独使用するrunでだけ壊れる」）は観測事実
（B1正常・B3黒・B4正常）と整合する。ただし shape HR も同じ cond1024_bank(view1-3含む)を
参照して gs=7.5(cond差が7.5倍に増幅)で正常に見える、という反証材料もあるため、**決めつけず
測って決める**。

**tex forward計算の数値精度は原因候補から除外する**（v6で判定基準(a)が反証済み。前commitの
「累積発散」筋書きは撤回済み、次セッションはこれを再検証しない）。

## 【2026-09-21 23:5x 更新】matte目視確認 — 統括指示の安価な確認を実施、matte仮説は却下

統括指示で、advisor()のcond dump/差し替えpodへ進む前に**pod不要の最安確認**を実施した:
Mac上でCPUのみ（重み不要）で native の `threshold_cutout()`（実際にB1/B3/B4で使われた
関数）をFrame1〜4に単体適用し、参照側`preprocessed_{0..3}.png`と目視+alpha被覆率で比較
（新規ツール `trellis-tool-dump-threshold-matte`、CMakeLists.txt登録、画像は
`docs/results/2026-09-21-trellis2-mv-b3/matte-check/`）。

- alpha被覆率だけ見るとFrame2/Frame4が「ほぼ全て半透明」でFrame1/Frame3の「ほぼ全て不透明」と
  質的に違った → が、**目視するとFrame4はクリーン**（被覆率統計だけでは正常なソフトエッジと
  本物の欠陥を区別できないと判明。alpha%単独で判定しない教訓）。
- **Frame2だけ**顔・首に明瞭なゴースト状アーティファクト（灰色の影）が視覚的に確認できた。
- **決定的確認**: PyTorch参照の`preprocessed_1.png`（=Frame2の参照側前処理結果）にも
  **全く同じ位置・形状のゴーストが存在**した。よってこの欠陥はnativeのthreshold-matte
  fallback固有ではなく元のFrame2.png自体（または両実装共通の特性）に由来し、参照側も
  同じ欠陥入りの入力で正常なテクスチャを出している。

**結論: matte品質（birefnet.gguf欠如）仮説は却下。** 修正の形は「E2Eセットアップ問題」には
ならない。advisor()提案のcond dump/差し替えpodへ進む（native自身のcond1024_bankが
DINOv3実行後に参照のcond_1024.npyと数値一致するかは今回のCPU限定確認では検証できておらず、
別の切り口として引き続き必要）。

## 【2026-09-22 00:xx 更新】cond差し替え実験（pod v7）— cond原因説は部分的に支持、単独では説明不足

統括指示の3腕差し替え実験を実施（`trellis_cli.cpp`に`TRELLIS_DBG_COND_NPY`/
`TRELLIS_DBG_COND_OVERRIDE`（cond1024_bank用）/`TRELLIS_DBG_COND_OVERRIDE_512`（cond_bank用、
後日追加）を実装、pod `trellis2mv-b3-diag` v7、削除済み、
`docs/results/2026-09-21-trellis2-mv-b3/logs/v7-pod-cond-swap.log`）:

| arm | 内容 | out_base.png mean |
|---|---|---|
| (i) | native cond のまま（ベースライン） | **0.0730** |
| (ii) | view1,2（Frame2,Frame3）のcond1024だけ参照condに差し替え | **0.6045** |
| (iii) | 全viewのcond1024を参照condに差し替え | **7.1921** |

参考: 「正常」とされるB1/B2/B4のout_base.png meanは10〜65。

**native cond1024_bankの参照とのper-view rel（4view全て）**: view0=19.5%, view1=19.3%,
view2=14.0%, view3=20.0%。**Frame2/Frame3(view1/view2)だけが突出して悪いわけではない**
（view2はむしろ最も誤差が小さい）。全viewが同程度（14〜20%）ネイティブのDINOv3 cond生成が
参照からずれている。

**統括の判定基準に照らすと**: (ii)は「正常化」(10〜65到達)しなかった（0.60のまま）。
(iii)も「黒のまま」ではなく大幅改善(0.073→7.19、約100倍)したが、**依然として正常範囲
(10〜65)には届いていない**。つまり二択どちらにもきれいに当てはまらない中間的な結果。

**解釈**: condの誤差は明確に黒化の主要因の一つ（全view補正で100倍改善）だが、**単独では
「正常」まで説明しきれない**。全viewのcond1024を完全に参照へ差し替えてもまだ7.19なのは、
①SS/shape-LR段が使うcond_bank(512)はこの実験ではnativeのまま（今回は1024しか差し替えて
いない）で幾何(coords)がnative由来のまま、②幾何が違えばUVアトラス上の被覆面積も変わり
mean自体が単純比較できない、③tex/V=4 stochastic固有の残存問題、のいずれか（複合の可能性も）。
per-view relがFrame2/3限定ではなく全viewに一様に出ていることから、「Frame2/3特有の劣化」
仮説（前回のmatte疑いの延長）は後退し、「native DINOv3実行全体が参照から系統的に15〜20%
ずれており、それがV=4 stochastic texの文脈でだけ致命的に増幅される」という筋書きへ寄る。

**コード変更**: `cond_override_stage()`ラムダへリファクタリングし、cond1024_bank用と
cond_bank(512)用の両方に同じ仕組みを適用可能にした（`TRELLIS_DBG_COND_OVERRIDE_512`env var
追加）。ビルド確認済み、pod v8で(iv) 512+1024両方全view差し替えを試す準備完了。

## 【2026-09-22 00:3x 訂正】上のarm(iv)/(v)「決定的」は誤り — noise行番号結合による交絡未分離

advisor()相談で発覚（`src/trellis_cli.cpp:344-345,782-783`で確認済み: `std::mt19937 rng(run_seed)`
**単一ストリーム**を`noise(n)`が呼ばれるたびに逐次消費する設計。SS→shape-LR→shape-HR→tex の
順に`noise()`を呼ぶため、**cond_512を差し替えてSS段のactive voxel数が変わると、shape-LR以降
全ステージのnoiseが丸ごと別サンプルになる**——同じ`--seed 42`でも実質的に別サンプルの
シード掃引を1回引いたのと同じ）。よって下の「pod v8決定的」の節は**因果として読めない**:
arm(iv)/(v)が正常化したのは「cond_512を直したから」ではなく「パイプライン全体を再抽選した
結果、たまたま黒くならないサンプルを引いた」可能性を排除できていない。

**現時点で因果として言えるのはarm (i)→(ii)→(iii)（cond_1024実験、pod v7）のみ**: これらは
cond_512がnativeのまま=SS/shape-LRの出力（coords, noise消費量）が完全に同一なので、
shape-HR以降のnoiseストリームも同一。cond1024の精度がmeanに単調に効く(0.073→0.60→7.19、
正常域には届かず)という結論は維持できる。cond_512側の実験(pod v8)は**ノイズ床を測らずに
条件比較した**ため（lessons.md「確率的生成モデルを条件間で比較するときは対照群でノイズ床を
測る」に抵触）、次段でシード掃引によりB3 native単体の失敗率を測ってから再解釈する。

## 【2026-09-22 00:2x、要再解釈】pod v8 — cond_512（SS/shape-LR側）だけの補正で正常化した（訂正: 決定的ではない、上の訂正参照）

pod v8（削除済み、`docs/results/2026-09-21-trellis2-mv-b3/logs/v8-pod-cond-swap-512.log`）で
2腕追加実行:

| arm | 内容 | out_base.png mean |
|---|---|---|
| (iv) | 全viewのcond_512とcond_1024を両方参照に差し替え | **39.95**（正常範囲） |
| (v) | 全viewのcond_512**だけ**参照に差し替え（cond_1024はnativeのまま） | **29.84**（正常範囲） |

**(v)が決定的**: tex段が見るcond_1024はnativeの不正確なまま（前回実験で単独では7.19までしか
戻らなかった）なのに、**SS/shape-LR段が使うcond_512だけを補正するとmeanが正常範囲まで戻った**。
つまり黒化に一番効いているのはtex段のcond精度ではなく、**SS/shape-LR段のcond_512精度が
決める幾何（アクティブボクセル選択・LR SLat）**だと分かった。

cond_512のnative-vs-参照per-view relも計測: view0=16.0%, view1=18.2%, view2=14.0%,
view3=18.8%（cond_1024の14〜20%とほぼ同水準、こちらもFrame2/3特有ではない）。

**残された謎**: B1(V=2)はnativeの不正確なcond_512・cond_1024をそのまま使っても正常。
B3(V=4 stochastic)だけがこの同水準の不正確さに敏感で、cond_512を補正するだけで直る。
これは「V=4 stochastic texが幾何のわずかな違い（アクティブボクセル選択の閾値付近での
揺れ等）に対して特異的に脆弱」という、mode分離実験の結論（バグはtex自身のstochastic実行に
局在）と両立する——cond_512補正は幾何を変えることで、tex側の脆弱性を"避けた"だけかもしれず、
根本原因（なぜV=4 stochastic texだけがこの幾何感度を持つか）はまだ未解明。

コードのcond_512精度自体を直す（DINOv3実行やthreshold処理のバグを探す）のは、原因を
「native全体のDINOv3実行が系統的に15〜20%不正確」という一般論に帰着させるもので、
V=4 stochastic固有の脆弱性の説明にはならない可能性が高い。次段は「tex自身のstochastic
実行がなぜ幾何のわずかな違いにここまで敏感か」を直接見る方向へ戻すべき。

## 【2026-09-22 00:5x 最重要】pod v9 シード掃引 — 「V=4 stochasticがV=2より脆い」自体が誤りだった

advisor()の判定基準どおり、B3 nativeを6シード（42-47）、B1 nativeを3シード（42-44、対照）で
実行（pod `trellis2mv-b3-diag` v9、再ビルド不要でv8のNFSバイナリを流用、削除済み、
`docs/results/2026-09-21-trellis2-mv-b3/logs/v9-pod-seed-sweep.log`）。

| run | seed42 | seed43 | seed44 | seed45 | seed46 | seed47 |
|---|---|---|---|---|---|---|
| B3 (V=4 stochastic) mean | **0.0687** | 47.25 | **0.5782** | 38.80 | 37.90 | 50.46 |
| B1 (V=2 stochastic) mean | 10.999 | 31.27 | **1.7203** | — | — | — |

**B3は6本中2本（seed42, 44）が黒（mean<1）、4本は正常（38〜50）——シード非依存の構造的破綻
ではなく確率的な失敗モードだった。** advisor()の判定基準「B3で1本でも正常なら失敗率問題」に
該当 → これまでの「V=4 stochastic texは構造的に壊れている」という前提そのものが誤りと確定。

**さらに重大**: 「正常」の対照として使っていたB1(V=2 stochastic)も、**seed44で mean=1.72
という黒に近い結果**を出した（3本中1本）。E2E報告の「B1は正常」はseed42（mean=11.0、
正常域10〜65のギリギリ下限）という**たった1回の観測**に基づくものだった。

**この1本のシード掃引だけで、これまでの二分の枠組み全体が組み替わる**:
- B3とB1の失敗率はどちらも観測範囲でおよそ1/3（B3: 2/6, B1: 1/3）——**V=4がV=2より脆い、
  という当初のE2E報告の前提自体が、n=1比較のサンプリング運だった可能性が高い**。
- mode分離実験（tex=stochasticなら黒、tex=multidiffusionなら正常）はseed42固定で行った
  ため結論としては正しい可能性が高い（stochastic modeがmultidiffusionより本質的に脆い、
  という筋は生きている）が、「V=4」という条件がどれだけ効いているかは未分離のまま。
- pod v7/v8のcond実験（cond1024/512の精度がmeanに影響する）も、単に「別のnoise draw」を
  引いていただけの可能性が高く、cond精度そのものの因果効果は疑わしい。

**新しい統一的な説明（最有力仮説）**: stochastic modeは各stepで**1viewの生予測のみ**を使い
希釈しない（multidiffusionは毎step全view平均で常に希釈・平滑化）ため、**特定のnoise seedの
下で数値的に不安定な軌道に入り、decoderの飽和領域まで暴走することがある**——これはV=2でも
V=4でも起こりうる、stochastic modeというアルゴリズム自体が内在的に持つ頑健性の低さであり、
「B3特有のバグ」ではなく「stochastic modeの既知のはずの弱点」の可能性が高い。

## 【2026-09-22 01:1x 最重要・新発見】TASK-B3-REF向けdump完了 + fixtureディレクトリの取り違えを発見

統括指示（原因確定は保留、(a)アルゴリズムの確率的失敗 vs (b) native固有の数値問題を切り分ける
ため、seed42のB3とseed44のB1のtex段入力一式をnpyでdumpし、参照側はTASK-B3-REF.md担当が
PyTorchへ注入）を実施。pod `trellis2mv-b3-diag` v10（削除済み）+ 確認用の短命pod
`trellis2mv-b3-inspect`（削除済み）。**修正には着手していない。**

### 実装（テスト専用、env gate）

`src/trellis_cli.cpp`に`TRELLIS_DUMP_TEX_INPUTS=<dir>`を追加（tex段のsample_bank呼び出し
直前直後、`sample_bank`ラムダにオプショナルな`trace`引数を追加して`sample_flow_multi`の
`trace`出力を取得できるようにした）。dumpされるファイル（全てrow-major、PyTorch参照fixtureと
同じ規約）:

- `tex_hr_coords.npy` — int32 [N,4]（batch=0,x,y,z）
- `tex_noise0.npy` — float32 [N,32]（tex段sampler初期noise、channel-major→row-major変換済み）
- `tex_trace_step{0..11}.npy` — float32 [N,32]×12（各stepの x_t、sample_flow_multiのtrace出力）
- `tex_slat_final_denorm.npy` — float32 [N,32]（最終tex SLat、tex_decode入力と同じ状態）
- `tex_cond_view{v}.npy` — float32 [tlc,1024]×V（native自身のcond1024_bank、per view）
- `tex_sampler_params.txt` — steps/gs/gr/gi0/gi1/rescale_t/mode/V/tN/tlc

### 出力先・sha256一覧

NFS `/nfs/trellis2-mv/b3-dump/{B3_seed42,B1_seed44}/`。全ファイルのsha256は
`docs/results/2026-09-21-trellis2-mv-b3/logs/v10-pod-tex-dump.log`に記録済み。

| dump | N(tN) | V | out_base.png mean（このdump run自体の再確認） |
|---|---|---|---|
| B3_seed42 | 4123 | 4 | 0.0712（元のシード掃引0.0687と近似、黒を再現） |
| B1_seed44 | 4391 | 2 | 1.7438（元のシード掃引1.7203と近似、黒に近い値を再現） |

SamplerParams（両方共通）: steps=12, guidance_strength=1.0, guidance_rescale=0.0,
gi0=0.6, gi1=0.9, rescale_t=3.0, mode=stochastic。

### ★新発見（dump作業中に偶然発覚、コードのバグではなくfixtureデータの取り違え）

cond dumpのsha256を比較していて、**B1_seed44のview1（`Frame4.png`のはずのcond）が、
B3_seed42のview2（`Frame3.png`のcond）とビット完全一致**していることに気づいた
（両方とも`2916f49e...`で始まる同一sha256）。ソース画像を直接比較したところ:

```
yoimiya_4view/Frame3.png sha256=da945908...
yoimiya_2view/Frame4.png sha256=da945908...  <- Frame3と同一！
yoimiya_4view/Frame4.png sha256=d81de09e...  <- 別物（本来の"Frame4"）
```

**`~/nfs/pixal3d_trellis2mv_ref_v2/inputs/yoimiya_2view/Frame4.png`は、実際には
`yoimiya_4view/Frame3.png`のコピー（誤命名）であり、本来の"Frame4"（yoimiya_4view内の
別画像、側面〜背面のポーズ違い）ではない。** これはコード側のバグではなく**テスト入力
fixtureディレクトリのデータ取り違え**（誰かが2-view fixtureを作る際にFrame3をFrame4として
コピーした可能性が高い）。

**影響**: これまでの全てのB1(2-view)実行——E2E担当の元の報告、本セッションのシード掃引・
mode分離実験・cond差し替え実験も含め——は「Frame1+Frame4」ではなく**実際には
「Frame1+Frame3」**という2枚を使っていたことになる。コードのview0/view1ラベル自体は
正しく動作している（意図した2枚を正しく読み込み・処理している）ので、この取り違えが
黒化バグの直接原因ではないが、「B1とB3で異なる2視点セットを比較していたつもりが、
実際にはB1がB3のFrame1+Frame3のサブセットを使っていた」という点は、これまでの分析の
文脈情報として重要（例えばB1とB3で同じFrame3のcondを共有していたことが分かった今、
mode分離実験・シード掃引の解釈を見直す材料になりうる）。**修正はしていない**
（統括判断待ち。fixtureはTASK-B3-REF.md担当や他ワーカーも参照している可能性があるため
勝手に書き換えない）。

## 次の一手（v9のシード掃引所見を最優先で反映、上のdump完了・新発見を追加）

0. **【最優先】この所見を統括へ報告し、タスクの前提（「B3だけが黒くなる」）自体が
   崩れたことを確認する。** TASK-B3.mdの前提（B1/B2/B4正常・B3のみ黒）はE2E担当の
   単一シード観測に基づくもので、n=1の落とし穴（lessons.md該当）に当てはまる。
   「原因確定→最小修正」という枠組みが成立するかどうか自体、統括判断が必要
   （固定コードバグではなく確率的頑健性の問題なら、修正の形が変わる——fail-closed検出、
   stochastic modeのデフォルト非推奨化、リトライ、等）。
1. 参考として、B2/B4（multidiffusion）でも複数シードを回し、multidiffusionが本当に
   0%失敗率なのか（希釈により頑健なのか、単にまだ運良く当たっていないだけか）を
   確認する価値がある。
2. stochastic modeの失敗メカニズム自体（どのstep・どのviewの組み合わせで軌道が破綻するか）
   を、黒くなったseed（42, 44）と正常なseed（43, 45, 46, 47）を比較することで直接調べられる
   （`TRELLIS_DBG_MV_STEP`計装は既にある）。

## 次の一手（旧版、advisor()指摘のnoise交絡を受けて全面差し替え — 上のv9所見でさらに更新済み）

**発動条件（lessons.md）に該当**: 確率的生成モデルを条件間で比較する前に対照群でノイズ床を
測っていなかった。次の1本はまずこれを埋める。

1. **【最優先・ビルド不要】シード掃引でB3 nativeの失敗率を測る**: pod v8の`build/trellis-cli`
   がNFSに残っている（ソース変更なしなら再ビルド不要、`libgomp1`等のapt済み環境だけ要る）。
   env無し（native cond）のまま `--seed 42,43,44,45,46,47` でB3(4view stochastic)を6本、
   対照として `--seed 42,43,44` でB1(2view stochastic)を3本、`out_base.png mean`と
   `active voxels @res32` / quantized tokens数を1行ずつ表にする。
   - **判定基準（着手前に凍結）**: B3が6/6黒(mean<1)なら「V=4 stochastic texはnative上流なら
     シード非依存で壊れる」が確定し構造的原因を追う（次段: coords/tslat/condをnative B3固定で
     tex段のnoiseだけ別RNGから振る）。B3で1本でも正常なら「失敗率問題」（stochastic V=4の
     頑健性/fail-closed検出という別の修正の形になる）— この場合は結果が出た時点で統括へ
     即報告してから次を決める。
2. シード掃引の結果次第で、pod v7のcond1024実験（arm i→ii→iii、noise同一で因果として有効）
   の「単調改善するが正常域に届かない」という所見をどう位置付けるかを再検討する。
3. pod v8のarm(iv)/(v)は「noise交絡あり」を明記した上で参考情報として残す（再実験するなら
   coords/noiseを固定した上でcond_512だけを差し替える配線に直す必要がある）。

## 環境・再現手順メモ（次セッション用）

- worktree: `~/Downloads/pixal3d-mv-hardening`（`feat/trellis2-mv-hardening`）。
  計装は WIP commit 済み（`git log`で`wip(trellis2-mv): B3 diagnostic instrumentation`を確認）。
  **新規ファイル`src/test_trellis2_mv_tex_realweight_parity.cpp`とCMakeLists.txt変更は
  このHANDOVER保存と同時にコミットする**（下記参照）。
- NFS: `/nfs/trellis2-mv/src/0d4cfc5/`にこのツリーのソースをコピー済み（計装版で上書き済み、
  pristine 0d4cfc5ではない点に注意。他ワーカーが触ると混乱する）。ビルド済みバイナリ:
  `build/trellis-cli`, `build/trellis-test-trellis2-mv-tex-realweight-parity`。
- pod起動レシピ（apt依存衝突を解消済み、確認済みの手順）: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`
  イメージで、①`apt-get install cmake ninja-build git python3 libgomp1 build-essential g++`
  ②`apt-get install cuda-cudart-dev-12-8=12.8.57-1` ③`apt-get install libcublas-dev-12-8=12.8.3.14-1`
  ④`apt-get install cuda-nvcc-12-8`（この順でないと"held broken packages"で失敗する。詳細は
  削除済みpod YAMLのコメント参照、次にpodを立てるときは`cmake -B build`から再configureが要る
  ＝ヘッダmtime更新で実質フルリビルドになる、-j16で約3分）。
- 診断pod（`trellis2mv-b3-diag`、v1〜v5、すべて削除済み、GPU課金停止確認済み）で使った
  env var: `TRELLIS_DBG_NAN=1`(既存, DitRunner::forwardのNaN検出) `TRELLIS_DBG_MV_STEP=1`
  (新設, sample_flow_multiの毎ステップview/finite数ログ + texlatのper-channel統計)
  `TRELLIS_DBG_TEX_MODE=stochastic|multidiffusion`(新設, tex段だけmode上書き)
  `TRELLIS_DUMP_TEXSLAT=path`(新設, texlat denormをhr_slat.bin形式でdump)。
- 7a/7b（TASK-B3.md訂正版手順7、完了済み）: `logs/7a_{f04b33c,0d4cfc5}.log`
  （rel=4.3312e-01/2.3349e-01、READMEの数値と一致再現確認）、`logs/7b_fail_closed.log`
  （5パターン全てrc=1、メッセージ一致確認）。
- 全生ログは`docs/results/2026-09-21-trellis2-mv-b3/logs/`に保存済み。

## 禁止事項の遵守状況

- fixtureテストの期待値は一切変更していない（`test_trellis2_mv_tex.cpp`は未変更、
  再ビルド後もALL PASS維持確認済み）。
- `feat/trellis2-mv-e2e` worktreeは触っていない。push していない。
- 「直った」とはまだ言っていない（原因はtex_flow_1024のforward精度まで特定したが、
  修正はまだ実施していない）。
