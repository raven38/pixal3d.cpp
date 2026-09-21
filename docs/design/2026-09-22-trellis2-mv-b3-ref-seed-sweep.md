# B3-REF: 参照(PyTorch)側 stochastic mode 黒化率シード掃引 + native noise 注入

指示元: `~/Downloads/claude-work/docs/briefs/trellis2-mv/TASK-B3-REF.md`（COMMON.md 共通規約に従う）。
背景: `~/Downloads/pixal3d-mv-hardening/docs/results/2026-09-21-trellis2-mv-b3/HANDOVER.md`。

## 問い

native（CUDA、0d4cfc5）で V=4 stochastic tex ステージが seed 依存で黒化する（6 seed 中 2 本）。
native側の最新知見（pod v9, seed 42-47）: B3(V=4) 2/6・B1(V=2) 1/3 が黒。「V=4 stochastic の
方が V=2 より脆い」という前提は n=1 比較のサンプリング運だった可能性が高いところまで判明済み。
残る問い: (a) stochastic アルゴリズム自体の確率的失敗（PyTorch 参照でも同率で起きる）か、
(b) native の数値誤差（tex forward が参照から step0 で 1.29% 乖離）が原因の native 固有問題か。

## R1: 参照側 seed 掃引の設計

### 既存資産の再利用

`tools/ref_trellis2_mv_pod_run_v3.py` は使わない。理由: 実測で 1 run ≈ 464〜486 秒
（v3 fixture の `run_4img_real_stochastic_1024c` / `run_2img_real_stochastic_1024c` の
`manifest.json` `elapsed_sec`。brief 記載の「1〜2分」は誤り、全12step x_t capture +
自己検証(cfg_reconstruction_check等)込みのフル計装コストが支配的とみられる）。
23 run（4view stochastic ×10 + 2view stochastic ×10 + 4view multidiffusion ×3）を
このコストで回すと約3時間かかり、pod 予算・COMMON.md 予算に照らして避けたい。

### 新規スクリプト `tools/ref_trellis2_mv_seed_sweep.py`

v3 から計装を全て外した軽量版。目的は黒化率の分類のみで、per-step テンソル比較は不要
（R2 の noise 注入実験の方でstep単位比較を行う）。

- pipeline のロードと `get_cond`（512・1024 両方）は **画像セットごとに1回だけ**行い、
  seed ループの外に出す（cond は seed に依存しない値のため；v3 の重複計算を避ける）。
- seed ループ内: `torch.manual_seed(seed)` → `inject_sampler_multi_image` context →
  `sample_sparse_structure` → `sample_shape_slat_cascade`（4view/2viewとも 1024_cascade）→
  `sample_tex_slat` → `pipeline.decode_latent(shape_slat, tex_slat, res)`。
  StageRecorder 等の計装・x_t保存・mesh.glb export は行わない（decode_latent の戻り値
  `meshes[0].attrs` のみ使う）。
- **黒判定の定義とその較正**（advisor指摘: native の「mean<2 黒/10〜65正常」は GLB にbakeされた
  UV テクスチャ(`out_base.png`)の0-255 meanであり、voxel属性の mean をそのまま「踏襲」と
  書くのは実例確認なしの飛躍。着手前に Mac から `kubectl exec <cpu-pod>` で実測較正した:
  - `/nfs/repos/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:450`
    `decode_tex_slat`: `ret = self.models['tex_slat_decoder'](...) * 0.5 + 0.5` を実ソースで確認
    （`mesh.attrs` は既に`[0,1]`）。`docs/spec/12-ovoxel_mesh.md:62`と整合。
  - 既存 v3 fixture（seed=42、3 config とも既に生成済み）の `mesh.glb` を `trimesh.load` で開き
    `material.baseColorTexture`（bakeされたUVテクスチャそのもの、native の`out_base.png`と
    同じ値域）の実測 mean: **4view stochastic=13.02**(正常域)、**4view multidiffusion=9.85**
    (正常域下限付近)、**2view stochastic=0.0**（全texel unique=[0], max=0 — **完全に黒**、
    tex_slat自体は有限・正常範囲 mean=0.29/std=3.55 nan=0 で、黒化はdecode以降で発生）。
  - **重要な予備発見**: brief 記載の「参照側は v2/v3 のstochastic runが各1seedで全て正常」は
    誤りだった。既存 v3 の `run_2img_real_stochastic_1024c`（seed=42、B1相当構成）が
    **既に完全な黒**（bake済みテクスチャ全texel=0）。これは追加pod実行なしに得られた事実であり、
    「PyTorch参照でもstochastic modeがseed依存で黒化する」の直接証拠（仮説(a)を支持する方向）。
    R1完了報告で必ず言及する。
  - 軽量スイープでは `to_glb`（remesh+decimate 200k+1024 bake、v3 run 1本あたり ~470s の
    支配要因）を省き、`pipeline.decode_latent()` が返す生 `mesh.attrs`（decoder forward出力
    そのもの、bake前）を直接使う。`base_color_mean = (attrs[:, :3].clamp(0,1) * 255).mean()`。
    黒 run（2view seed42）で bake後実測(0.0)と近い値になることを pod 内で最初に確認してから
    残りの seed へ進む（proxy の妥当性ゲート、advisor指摘4番と合わせる）。
    black 判定閾値は native 実績どおり `mean<2` を暫定採用するが、**中間値（3〜8程度）が出たら
    二値化せず連続値のまま報告する**（advisor指摘、9.85のような境界値の実例あり）。
  - 併せて 6ch 全部の mean/std・saturation_rate（3ch RGBとも `<1/255` or `>254/255` の
    voxel比率）を `[0,1]` 空間で記録する（COMMON.md「否定的指標も後出ししない」対応）。
- 出力は 1 run = JSON 1行（seed, mode, num_images, base_color_mean/std, saturation_rate,
  active_voxels@各段, elapsed_sec, is_black）。mesh.glb や x_t は保存しない
  （ディスク・時間削減、v2/v3 fixture は一切変更しないので既存資産と衝突しない）。
- **禁止事項の遵守**: v2/v3 スクリプト自体は変更しない（新規ファイルとして追加）。
  v2/v3 の出力ディレクトリ（`/nfs/pixal3d_trellis2mv_ref_{v2,v3}/`）には書き込まない。
  新規出力は `/nfs/trellis2-mv/b3-ref-sweep/`（NFS、この worker 専用）に置く。

### 軽量スクリプトの妥当性ゲート（advisor指摘2）

計装を外し `get_cond` を seed ループ外に出す変更が v3 と同じ軌道を再現するか、各 config の
先頭 run（seed=42）で検証する。`docs/spec/33-trellis2-mv-reference.md`
「決定性・1-image collapseのv2での再検証」節で `run_1img_baseline_512` vs `_rerun`
（同一seed=42、同一環境）が `cond_512`/`ss_coords`/`shape_slat_{feats,coords}`/
`tex_slat_{feats,coords}`/`noise_*_0` 全て **sha256完全一致**した実績があるため、
このpodも同一GPU種別・同一環境（v2/v3と同じ pod レシピを再利用する前提）なら
**sha256完全一致をゲートに採用する**: 軽量スクリプトの seed=42 出力の
`ss_coords`/`tex_slat_{feats,coords}`のsha256を、既存v3 fixture
（`run_{4img,2img}_real_stochastic_1024c`, `run_4img_real_multidiffusion_1024c`の
`manifest.json`記載値）と直接比較する。**不一致でもスイープは止めない**
（黒化率の統計自体は意味を持つ。「v3と bit 一致」という主張だけ落として報告する）。

### R2 dump 仕様の欠落（advisor指摘3、統括へ要報告）

TASK-B3-REF.md が列挙した mvb3 dump 項目（HR coords / tex noise[N×C] / cond1024_bank per view /
各step x_t / 最終tex SLat）に **native の最終 shape SLat feats（64ch concat 前の正規化前後
いずれか）が含まれていない**。tex DiT は shape SLat を条件として使うため、これが無いと
「native coords + native noise」腕を正しく組めない。R1 pod 投入後、統括へ追加依頼を出す
（本ファイルの結論節 / 報告に明記）。

### pod の組み方（advisor指摘4）

- 1 pod・1 プロセスで pipeline を1回だけロードし、23 config×seed を順に処理する
  （HANDOVER のnative CUDAビルド用podではなく、v2/v3 を回した参照生成pod
  ＝`/nfs/envs/miniforge3/envs/trellis2`、`HF_HUB_OFFLINE=1`、イメージ・レシピは
  `/nfs/pixal3d_trellis2mv_ref_v2/scripts/` を再利用）。
- run 間で `torch.cuda.empty_cache()`。1 run 完了ごとに結果1行をJSONLへ即flush
  （Kueue preemption / deadline切れで部分結果が残るように）。
- 所要時間は事前見積りしない。seed=42（各configの先頭）の実測時間をログに出し、以降の見積りに使う。
- activeDeadlineSeconds は 4h 程度に設定。
- 投入直前に `kubectl get pods` で Running な GPU pod が無いことを再確認する
  （mvb3 が R2 dump 用に pod を立てる可能性があるため）。
- 完了待ちは `run_in_background` + 通知 or job-watch スキルを使い、ポーリングしない。

### 実行対象（brief 通り）

| config | images | mode | resolution | seed |
|---|---|---|---|---|
| B3相当 | yoimiya_4view (Frame1-4) | stochastic | 1024_cascade | 42-51 (10本) |
| B1相当 | yoimiya_2view (Frame1,4) | stochastic | 1024_cascade | 42-51 (10本) |
| B4相当(比較用) | yoimiya_4view (Frame1-4) | multidiffusion | 1024_cascade | 42-44 (3本) |

native 側は 42-47(B3) / 42-44(B1) だったが、brief が「native と揃えられるなら42..51」と
指示しているため 42-51 の10本に揃える（native 側 42-47 の6本はこの10本に包含される形になる
訳ではない=範囲が違う点に注意。native と同一seedでの直接比較は42-47の重なる範囲のみ有効、
42-51全体はPyTorch側の独立したnサイズ拡大として扱う）。

### 判定材料（brief 通り）

- 黒化率 0/20（4view+2view合算）なら (b) が濃厚、native と同水準(~30%)なら (a) が濃厚。
- Fisher's exact test で PyTorch 黒化率 vs native の差を検定する（scipy非依存の自前実装、
  既知値=tea-tasting問題[[3,1],[1,3]] p=0.485714…で自己テスト済み）。
  **codexレビュー指摘**: PyTorch側はB3/B1各10本・native側は6本/3本と構成比が異なり、
  config固有の効果があれば pooled(20 vs 3/9) の検定は Simpson型の交絡を起こしうる。
  さらに同一seedをB3/B1で使い回す設計のため「20 runを独立ベルヌーイ試行」という前提も弱い。
  よって **config別の層別検定（4view vs 4view=2/6、2view vs 2view=1/3）を一次判定材料とし、
  pooled(20 vs 9)は記述的な補助値に格下げする**。
- 結論（a/b）は統括が下す。ここでは表と p 値のみ書く。

## R2: native noise 注入

TASK-B3-REF.md 記載の通り、`mvb3` からの dump 到着待ち（2026-09-22 時点で
`/nfs/trellis2-mv/b3-dump/` 未着、`kubectl exec <cpu-pod> -- ls` で確認済み）。
到着後の実装方針:

- v2 の `NoiseCapture`（`tools/ref_trellis2_mv_pod_run_v2.py`）を継承・拡張した
  `NoiseInjector` を新設し、`torch.randn` を native 提供の npy 配列読み出しに置き換える
  （形状・呼び出し順が一致することを assert し、**注入した各テンソルの sha256 をログに残す**
  — brief 指示・v2 で patch 未発火だった実績があるため）。
- 腕1: native coords + native noise + **参照cond**（既存 cond_1024 をそのまま使用）。
- 腕2: native coords + native noise + **native cond**（mvb3 dump の cond1024_bank per view を
  そのまま注入、cond override は HANDOVER 記載の `TRELLIS_DBG_COND_OVERRIDE` 系と同じ発想を
  PyTorch 側にも実装）。
- 各 step の x_t を native の step ごと x_t と rel 比較（v3 の `save_tensor` 相当を流用）。
- 黒くなるかどうかと、native と最初に分岐する step を報告。

### dump 到着確認済み（2026-09-22 00:4x）、ただし欠落あり（codex指摘が的中）

`/nfs/trellis2-mv/b3-dump/{B3_seed42,B1_seed44}/` に到着済み。中身を実測確認:

```
tex_hr_coords.npy          (N,4) int32       HR座標
tex_noise0.npy             (N,32) float32    tex段noise
tex_trace_step{0..11}.npy  (N,32) float32    各stepのx_t
tex_slat_final_denorm.npy  (N,32) float32    最終tex SLat(denorm)
tex_cond_view{0..3}.npy    (Lc,1024) float32 native cond1024_bank per view
tex_sampler_params.txt     steps/guidance/mode等
run.log                    実行ログ(stats行含む)
```

**advisor指摘3が的中**: tex DiT の入力は64ch concat（32ch=denoise対象のtex x_t、
32ch=`shape_slat_norm`のguide、run.log実測 `tex slat guide (tslat, fixed shape_slat_norm half
of concat) n=131936 mean=-0.0024 std=0.9683`）だが、**このguide半分(32ch, 形状は
[N,32]のはず)がnpyとして保存されていない**（run.logにmean/std等の集約統計が印字されて
いるのみ）。これが無いとtex DiTのforwardを一度も呼べない（reference condを使う腕・native
condを使う腕のどちらでも共通して必要な入力のため、両腕とも現状ブロック）。

**統括/mvb3への追加依頼（このファイルに記録、報告にも明記）**: `TRELLIS_DUMP_TEX_INPUTS`の
dump処理に、tex DiT forward直前の`shape_slat_norm`（64ch concatのguide半分、期待shape
[N,32]、native実装上はSHAPE_MEAN/STDで正規化済みの値のはず）を追加で
`tex_shape_guide_norm.npy`として保存する1行を足して再dumpしてほしい。既存の計装
（tex_noise0.npy等）と同じ関数内に既にある値の可能性が高く、追加コストは小さいと見込む。

到着済みでも上記欠落のため、このセッションではR2の実装スケルトン（NoiseInjector クラス・
selftest）のみ先に作り、実際のstep比較実行は guide dump 到着後に行う。

## codex exec レビュー（実施済み、2026-09-22）

背景実行、`codex exec --sandbox read-only - < prompt.txt` で stdin を閉じて実施。指摘と反映:

1. **cond計算の位置（重要・修正済み）**: v3は`torch.manual_seed(seed)`の**後**に`get_cond`を
   呼ぶが、当初案は cond を seed ループの外でキャッシュしていた。get_cond が万一RNGを消費する
   実装なら、seed=42のv3 bit一致ゲートが偽陰性で壊れる恐れがある。**修正**:
   前処理（`preprocess_image`、v3でもmanual_seedより前に実行される、RNG非依存）だけキャッシュし、
   `get_cond`自体はv3と同じ位置（manual_seed直後）で毎seed呼び直す構成に変更した
   （`tools/ref_trellis2_mv_seed_sweep.py`のget_preprocessed_images/run_sweep）。
2. **proxy較正が1例(黒)だけでは不足（修正済み）**: raw voxel attrs meanとbake後texel meanは
   重み付けが本質的に異なるため、黒1例の一致だけでは判定器として採用する根拠が弱い。
   **修正**: 既存v3 fixture 3件（4v stochastic=正常、2v stochastic=黒、4v multidiffusion=境界値）
   全てについて、保存済みshape_slat/tex_slatを`pipeline.decode_latent()`のみ再実行し
   （サンプリングなし、数秒〜十数秒）、proxy分類とbake後実測分類が一致するかを
   `calibrate_against_v3_fixtures()`で検証し、**フルスイープ着手前のゲート**とした
   （1件でも不一致ならsweepをabortする。`--skip-calibration-gate`でデバッグ時のみ無効化可能）。
3. **統計設計（Fisher pooled vs 層別）**: 上記「判定材料」節に反映済み（config別を一次、
   pooledを記述的補助に格下げ）。

全指摘を設計・実装に反映済み。
