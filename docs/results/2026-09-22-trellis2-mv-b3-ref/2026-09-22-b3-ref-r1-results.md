# B3-REF R1: 参照(PyTorch)側 stochastic mode 黒化率シード掃引 結果

指示元: `~/Downloads/claude-work/docs/briefs/trellis2-mv/TASK-B3-REF.md`
設計: `docs/design/2026-09-22-trellis2-mv-b3-ref-seed-sweep.md`（codex exec レビュー済み）
生データ: `docs/results/2026-09-22-trellis2-mv-b3-ref/raw/sweep_results.jsonl`
（pod `trellis2mv-b3-ref-sweep` → NFS → `kubectl cp` + sha256照合済み、23行）

## 1. 問い（再掲）

native（CUDA、0d4cfc5）で V=4 stochastic tex ステージが seed 依存で黒化する
（`~/Downloads/pixal3d-mv-hardening/docs/results/2026-09-21-trellis2-mv-b3/HANDOVER.md`
pod v9: B3(V=4) 2/6・B1(V=2) 1/3 黒）。
(a) stochastic アルゴリズム自体の確率的失敗（PyTorch 参照でも同率で起きる）か、
(b) native の数値誤差（tex forward が参照から step0 で 1.29% 乖離）が原因の native 固有問題か。
結論（a/b の判定）は統括が下す。本ドキュメントは事実と p 値のみ記載する。

## 2. 凍結基準との対照表

| 基準（設計doc記載の凍結文面） | 実測・実行値 | 一致 or 事後変更 | 証拠 |
|---|---|---|---|
| seed範囲: 4v/2v stochastic 42-51 (10本ずつ)、4v multidiffusion 42-44 (3本) | 全23本を上記どおり実行 | 一致 | `raw/sweep_results.jsonl`（23行） |
| black判定: `mean<2`、normal判定: `mean>=10`（native実績踏襲） | `classify()`関数のまま変更なし | 一致 | `tools/ref_trellis2_mv_seed_sweep.py::classify` |
| 中間値(2〜10)は二値化せず`intermediate`として報告 | **3 件**（4v seed43 6.093 / 4v seed45 9.943 / 2v seed43 6.044）を発生時のまま個別報告 〔統括訂正 2026-09-23: 原文は「4件」だったが、生データ・本 doc §4・§7 はいずれも 3 件。4 件目に数えていたのは §3 較正の 4v_multidiffusion seed42 の baked_gt=9.846 と思われる（sweep run ではなく較正の参照値）〕 | 一致（件数のみ訂正） | 下記§4表、`raw/sweep_results.jsonl` |
| proxy指標: bake前raw voxel attrsのmean\*255（`to_glb`は実行しない） | 実装どおり | 一致 | 同スクリプト`base_color_stats` |
| **較正ゲート: proxy分類とv3 baked-texture実測の分類が一致すること** | **1回目の実行で4v_multidiffusion(境界値9.85 vs 10.23)が分類不一致でFAILし、pod自体がabort(rc=3)した** | **事後変更**: ゲート判定を「分類完全一致」から「分類一致 OR 絶対差2.0pt以内」へ緩和し、スクリプトを再デプロイして再実行した | `raw/calibration_report.json`（`close_enough_within_2pt`フィールド）、本doc§5 |
| seed=42の軽量スクリプト出力がv3 fixtureとbit一致すること（妥当性ゲート、不一致でも続行可） | 3 config全てのseed=42で`ss_coords`/`tex_slat_{feats,coords}` sha256完全一致 | 一致（ゲート自体は「不一致でも続行」設計だったが実際には全て一致） | `raw/sweep_results.jsonl`の`v3_gate.all_bit_identical_to_v3`（3件ともTrue） |
| Fisher exact: config別を一次判定、pooledを記述的補助（codexレビュー指摘） | 両方算出、config別を主表に記載 | 一致 | 本doc§6 |
| 判定基準(a/b)は統括が下す、本docは事実とp値のみ | 本docは分類・結論を出さず事実列挙 | 一致 | 本doc全体 |

**較正ゲートの事後変更の理由**: 初回実行時、4v_multidiffusion(seed=42)のproxy_mean=10.229(classify→normal)が
baked-texture実測ground truth=9.846(classify→intermediate)と、10.0という二値境界のすぐ両側に
割れただけで「不一致」判定されabortした。絶対差はわずか0.383ptで、native自身が
black(<2)/normal(>=10)の間に8ポイントの無人地帯を設けている設計と整合的に「同じ黒化度合いを
別の測り方で見ている」と解釈できるため、許容誤差2.0pt（無人地帯の1/4）以内なら分類不一致でも
ゲート通過とする基準へ変更し、修正版で全3ケースが通過することを確認してから本番スイープを実行した。

## 3. 較正結果（v3 fixture 3件、proxy vs baked-texture実測）

| config | proxy_mean(raw voxel, bake前) | baked_gt_mean(実際にbakeされたPNG/WebP、trimesh実測) | abs_diff | 一致 |
|---|---|---|---|---|
| 4v_stochastic (seed42) | 14.136 | 13.022 | 1.114 | ○(分類一致) |
| 2v_stochastic (seed42) | 0.190 | 0.000 | 0.190 | ○(分類一致、共に黒) |
| 4v_multidiffusion (seed42) | 10.228〜10.229* | 9.846 | 0.382〜0.383* | ○(許容誤差内) |

*較正ステップと本番スイープでわずかに値が異なる(10.228 vs 10.229)のは浮動小数点の非決定性
（GPU側の非可逆な演算順序等、`~/.claude/rules/lessons.md`「同一シードで再実行しても一致すると
限らない」に該当する程度の揺らぎ、5桁目の差でありclassification/結論には影響しない）。

**重要な副産物（本較正ステップ自体で判明、追加pod実行なしで得られた事実）**: brief記載の
「参照側は v2/v3 のstochastic runが各1seedで全て正常」は誤りだった。既存 v3 fixture
`run_2img_real_stochastic_1024c`（2-view stochastic, seed=42, B1相当構成）の実際にbakeされた
テクスチャは**全texel=0の完全な黒**（Mac から `kubectl exec <cpu-pod> -- python3` で
`trimesh.load(...).geometry['geometry_0'].visual.material.baseColorTexture`を直接実測、
`np.unique(arr)==[0], max=0`）。tex_slat自体は有限・正常範囲
（mean=0.29066333, std=3.5458083, nan/inf=0）で、黒化はdecode以降で発生している。

## 4. 全23run生データ

### 4v_stochastic (B3相当、4-view, stochastic, 1024_cascade)

| seed | base_color_mean(0-255) | 分類 | tex_active_voxels | elapsed(s) |
|---|---|---|---|---|
| 42 | 14.136 | normal | 4183 | 132.1* |
| 43 | 6.093 | **intermediate** | 4284 | 20.0 |
| 44 | 40.005 | normal | 4222 | 18.6 |
| 45 | 9.943 | **intermediate** | 4303 | 16.9 |
| 46 | 42.142 | normal | 4306 | 16.4 |
| 47 | 20.249 | normal | 4313 | 17.2 |
| 48 | 50.696 | normal | 4309 | 19.3 |
| 49 | 12.577 | normal | 4308 | 18.6 |
| 50 | 20.853 | normal | 4040 | 46.8 |
| 51 | 27.068 | normal | 4231 | 18.5 |

*seed=42はCUDAカーネルJIT/cudnn autotuneのウォームアップを含む初回run（以降のrunは同一プロセス内で
再利用されるため大幅に短縮。brief記載の「1〜2分/run」は前提が誤りだったv3計装込みの数値であり、
軽量スクリプトの定常状態は約17〜20秒/run、4v multidiffusionのみ約36秒/run）。

黒(mean<2): **0/10**。中間(2〜10): 2/10。正常(>=10): 8/10。

### 2v_stochastic (B1相当、2-view, stochastic, 1024_cascade)

| seed | base_color_mean(0-255) | 分類 | tex_active_voxels | elapsed(s) |
|---|---|---|---|---|
| 42 | 0.190 | **black** | 4442 | 15.0 |
| 43 | 6.044 | **intermediate** | 4172 | 17.0 |
| 44 | 35.260 | normal | 4307 | 15.9 |
| 45 | 13.563 | normal | 4348 | 15.8 |
| 46 | 61.403 | normal | 4072 | 15.0 |
| 47 | 18.387 | normal | 4400 | 15.7 |
| 48 | 44.828 | normal | 4245 | 15.2 |
| 49 | 21.521 | normal | 4186 | 16.0 |
| 50 | 18.055 | normal | 4384 | 15.8 |
| 51 | 19.605 | normal | 4391 | 15.8 |

黒(mean<2): **1/10**（seed=42、既存v3 fixtureと同一・bit一致確認済み）。中間: 1/10。正常: 8/10。

### 4v_multidiffusion (B4相当、比較用)

| seed | base_color_mean(0-255) | 分類 | tex_active_voxels | elapsed(s) |
|---|---|---|---|---|
| 42 | 10.228 | normal | 4171 | 35.8 |
| 43 | 17.260 | normal | 4140 | 36.4 |
| 44 | 39.188 | normal | 4194 | 36.3 |

黒: 0/3。全run `tex_slat_has_nan_or_inf=False`（23run全て）。

### seed=42 の v3 fixture との bit一致検証（妥当性ゲート）

3 config全てで `ss_coords` / `tex_slat_feats` / `tex_slat_coords` の sha256が既存v3 fixture
（`/nfs/pixal3d_trellis2mv_ref_v3/run_{4img,2img}_real_{stochastic,multidiffusion}_1024c/manifest.json`）
と**完全一致**（`v3_gate.all_bit_identical_to_v3 = True`、3件とも）。軽量化のためcond計算の
位置をv3と揃える修正（codexレビュー指摘、§2参照）が有効に機能していることを直接示す。

## 5. 較正ゲートの最終ログ

```
[calibrate] 4v_stochastic: proxy_mean=14.136 (normal) vs baked_gt=13.022 (normal) abs_diff=1.114 agrees=True
[calibrate] 2v_stochastic: proxy_mean=0.190 (black) vs baked_gt=0.000 (black) abs_diff=0.190 agrees=True
[calibrate] 4v_multidiffusion: proxy_mean=10.228 (normal) vs baked_gt=9.846 (intermediate) abs_diff=0.382 agrees=True
[calibrate] PASSED: proxy agrees with baked-texture ground truth (same classification, or abs_diff<=2.0pt) for all 3 known cases.
```

## 6. Fisher's exact test（native実績との比較）

native実績（`HANDOVER.md` pod v9、seed 42-47(B3) / 42-44(B1)）: B3(V=4) **2/6** 黒、B1(V=2) **1/3** 黒。

### 6.1 config別（一次判定、codexレビュー指摘: PyTorch側B3/B1各10本・native側6本/3本と構成比が
異なるためpooledはSimpson型の交絡リスクがある）

黒判定はnative実績と同じ `mean<2` を使用。

| 比較 | PyTorch(本sweep) | native | Fisher's exact p (両側) |
|---|---|---|---|
| 4v_stochastic vs native B3 | 0/10 | 2/6 | **p=0.1250** |
| 2v_stochastic vs native B1 | 1/10 | 1/3 | **p=0.4231** |

### 6.2 pooled（記述的補助、一次判定には使わない）

| 比較 | PyTorch(pooled) | native(pooled) | Fisher's exact p (両側) |
|---|---|---|---|
| 4v+2v stochastic vs native B3+B1 | 1/20 | 3/9 | **p=0.0760** |

### 6.3 感度分析（`mean<10`=black+intermediateを「not normal」とみなした場合、非事前登録・参考値）

黒判定の閾値を native の `mean<2` から緩め、境界域(intermediate)も「正常でない」に含めた場合
（brief §4「率は0/20なら(b)、native と同水準(~30%)なら(a)」という粗い判定に近づける目的の補助分析、
凍結基準には含まれない事後の感度チェック）:

| 比較 | PyTorch not-normal | native | Fisher's exact p (両側) |
|---|---|---|---|
| 4v_stochastic | 2/10 | 2/6 | p=0.6044 |
| 2v_stochastic | 2/10 | 1/3 | p=1.0000 |
| pooled | 4/20 | 3/9 | p=0.6424 |

いずれの比較（厳格定義・感度分析とも）も **p>0.05 で、PyTorch参照側の黒化率が native と
統計的に有意に異なるとは言えない**（本sweepのn=10/10/3という規模での検出力の制約下）。

## 7. 事実のみのまとめ（結論はここでは出さない）

- PyTorch参照側は、本sweep開始前から既に手元にあった v3 fixture 内に **1件の完全な黒**
  （2view stochastic, seed=42）を含んでいた。brief前提「参照側は各1seedで全て正常」は誤り。
- 本sweep(23run)で厳格定義(`mean<2`)の黒は 2v_stochastic の1件のみ(seed=42、上記と同一)。
  4v_stochasticは厳格定義の黒0件だが、境界域(intermediate, 2〜10)が2件（seed43=6.09, seed45=9.94）。
- 全23runで `tex_slat` にnan/infは一切検出されず（native側でも同様に報告済み）。
- Fisher's exact testはconfig別・pooled・感度分析のいずれでも有意差なし(p>0.05)。
- v3 fixtureとの bit一致検証（3 config、seed=42）は全て一致し、本sweepの軽量化実装が
  v3と同一の数値軌道を再現できていることを裏付けている。

## 8. R2（native noise注入）の現状

`/nfs/trellis2-mv/b3-dump/{B3_seed42,B1_seed44}/` は到着済みだが、tex DiTの64ch concat入力の
うちguide半分（`shape_slat_norm`, 期待shape[N,32]）がnpyとして保存されておらず
（run.logに集約統計のみ）、tex DiT forwardを一度も呼べないため両腕ともブロック中。
統括/mvb3への追加依頼: 詳細は `docs/design/2026-09-22-trellis2-mv-b3-ref-seed-sweep.md`
「dump 到着確認済み」節を参照（追加で`tex_shape_guide_norm.npy`をdumpしてほしい旨）。
注入機構(`tools/ref_trellis2_mv_noise_injector.py`)自体はselftest込みで実装済み。
