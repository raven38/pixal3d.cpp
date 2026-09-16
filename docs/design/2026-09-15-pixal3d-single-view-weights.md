# Pixal3D 単視点重みを CLI から選べるようにする

- 状態: **実装済み（2026-09-16）**。改訂 1 で codex レビューを反映
- 実測: CUDA/L4 で SV・MV とも V=1 生成が完走（190.3 s / 193.1 s）。同一入力・同一 seed・
  共有モデル共通で、SV は V=3,816,218 F=7,648,734、MV は V=3,979,224 F=7,977,482 と明確に
  異なる形状になった（5 視点レンダで目視確認）。入力が同一なので、この差は transforms.json
  では説明できない。ただし cyclops の正解形状を持たないため「どちらが正しいか」は未判定。
  既定（省略）と明示 mv は Metal で PLY バイト一致・GLB も生成時刻の 3 バイトを除き一致
- 対象 issue/マイルストーン: ROADMAP M2「productized single-view Pixal3D path」
- 変更ファイル: `include/trellis_args.h` / `src/trellis_args.cpp` / `src/trellis_cli.cpp` /
  `tools/convert.py`（SV エントリ、適用済み） / `README.md` / SV manifest 2 種

## 背景

公式 Pixal3D は 4 段の flow それぞれに単視点版（`_mv` なし）と多視点版（`_mv`）を配布している。
このリポジトリは多視点版しか扱えない。`src/trellis_cli.cpp` がファイル名を 4 箇所で
直接埋め込んでいるためで、単視点用の分岐は存在しない。

```
trellis_cli.cpp:264  M + "/pixal3d_ss_flow_mv.gguf"
trellis_cli.cpp:296  M + "/pixal3d_shape_flow_512_mv.gguf"
trellis_cli.cpp:352  M + "/pixal3d_shape_flow_1024_mv.gguf"
trellis_cli.cpp:401  M + "/pixal3d_tex_flow_1024_mv.gguf"
```

## 何を作るか（と、作らないか）

**作る**: 重みの系列（単視点 / 多視点）を CLI で選べるようにする。上記 4 つのファイル名だけが
変わる。**ただし入力は従来どおり `--views DIR`（`transforms.json` の frames が 1 件）に限る。**

**作らない**: 公式 SV の入力規約そのもの（後述）、新しい推論経路。

### 根拠（条件付けと DiT グラフは共通）

1. 公式チェックポイントの単視点版と多視点版は **config json が完全一致**し、safetensors の
   テンソル名・shape・dtype のシグネチャも一致する（ss_flow 701 / tex_flow 700 エントリ、
   HF のヘッダを Range 取得して照合、2026-09-15）。変換後の GGUF でも
   `general.architecture` / `trellis.config_json` / 700 テンソルの構造が多視点版と一致し、
   `dit_detect_proj_attn()` が要求する `blocks.0.cross_attn.proj_linear.weight` を持ち
   `d_proj` も同値（ss 1024 / slat 2048）。**同じローダ・同じグラフで読める。**
2. V=1 で条件付けが壊れないことをコードで確認した（レビュー指摘 1 の追試）:
   `pixal3d_cond.cpp:47` / `:95` は V=0 を先に弾いてから平均係数を作るので V=1 で係数 1。
   `pixal3d_cond_gpu.cpp:438` も同様。`proj_grid` と NAF に視点数依存の寸法や V>1 前提は無い。
   `trellis_args.cpp:126` が `--num-views 0` を、`transforms_json.cpp:212` が空 frames を、
   `pixal3d_input.cpp:13` が `n<=0` を弾くので、条件付け関数に V=0 は届かない。

### 根拠が及ばない範囲（レビューで判明・重要）

**公式 SV は入力規約が違う。** `inference.py:209` / `:248` は
`camera_params = {'camera_angle_x', 'distance', 'mesh_scale'}` を渡し、**`transform_matrix` を
使わない**。単一画像からカメラを推定する経路（`get_camera_params_wild_moge()`、MoGe 依存、
`inference.py:137`）まで持つ。一方 MV は `transform_matrix` を必須にする。

したがって本変更で得られるのは「**SV 重みを、MV と同じ `transforms.json` 入力規約で動かす**」
ことであり、「公式 SV パイプラインの再現」ではない。公式 SV との出力一致は主張しない。

後処理は差が無いことを確認済み: `inference.py:263` と `inference_mv.py:330` の `to_glb()` 引数は
**完全一致**（`decimation_target=1000000` / `remesh_band=1` / `texture_size` 同値）、最終の
`rot` 行列も同一。`inference.py:125` の回転は `distance_from_fov()` というカメラ距離推定
ヘルパの内部で、出力座標系とは無関係。よって `trellis_cli.cpp:444` / `:469` / `:504` / `:516`
の MV 既定値は SV でもそのまま使ってよい。

## インターフェース

```
--pixal3d-weights sv|mv     Pixal3D flow weights variant (default: mv)
```

`--weights` ではなく `--pixal3d-weights` にする（レビュー指摘 4）。`parse_args()` は CLI と
server が共有し（`trellis_args.cpp:99`）、Pixal3D 経路の選択は `views` の有無だけ
（`trellis_cli.cpp:527`）なので、`trellis-cli image.png out.glb --weights sv` が受理されて
TRELLIS.2 重みで走り、利用者が SV を実行したと誤認しうる。名前で対象を明示し、**`--views`
無しで指定されたらエラーにする**。

`sv` を選ぶと 4 つのファイル名の接尾辞が `_mv` から `_sv` へ変わる。共有の 5 モデル
（`dinov3` / `pixal3d_naf` / `ss_dec` / `shape_dec` / `tex_dec`）は系列に依存しないので変えない。

## 事前検査（レビュー指摘 3）

現行は 4 本を段階的にロードし、texture flow は `trellis_cli.cpp:401` まで到達しないと存在確認
されない。`Model::load()` は例外を投げるが（`trellis_model.cpp:124`）CLI main は捕捉しない
（`trellis_cli_main.cpp:17`）。`_sv` の texture flow だけ欠けていると、SS・Shape512・Shape1024 を
走らせた後に未捕捉例外で落ち、診断も出ない。

**推論開始前に、選択した系列で実際に必要なファイルを一括検査する。** `--no-texture` のときは
texture flow を必須にしない。欠けていれば系列名と探したパスを出して終了する（既定の `_mv` へ
黙って落ちない）。

## 誤用への手当て（レビュー指摘 1 を反映）

多視点重みに 1 視点を与えると、素直な劣化ではなく破綻する。実測（社内の比較実験、
2026-09-04 / 2026-09-11）では、カメラ・入力を共通にして視点数だけ 4→1 にすると
多視点重みが **85勝0敗**（d30 85体、p=5.170e-26）、**20勝0敗**（game 設定画 20体、
IoU 中央値 +0.32）で負け、目視でも頭部から棘状の破綻が出る。1 視点同士では多視点重みは
単視点重みに 3勝17敗で負ける。

当初案は V=2-3 の sv と V=5+ の mv を無警告にしていたが、どちらも根拠が無い。
`pixal3d_cond_ss/slat` は全視点を単純平均するので、SV 重みに V=2-3 を与えれば別分布の条件に
なる。MV も 4 視点でしか検証していない。**検証済みの組み合わせだけを無警告にする**:

| weights | 無警告 | それ以外 |
|---|---|---|
| `sv` | V=1 | 警告（単視点重みに複数視点を平均投入している） |
| `mv` | V=4 | 警告（多視点重みは 4 視点で検証済み） |

実行は止めない（判断はユーザーに残す）。

## 入力検証（レビュー指摘 5）

現行は `camera_angle_x` の存在しか見ず（`trellis_cli.cpp:137`）、行列も 4x4 の数値であることしか
検査しない（`transforms_json.cpp:194`）。さらに逆行列計算の失敗が無視される
（`proj_grid.cpp:127` / `:206`）。V=1 では他視点による平均緩和が無いので、壊れた 1 枚が
そのまま条件になる。

入力段で次を検証する: `camera_angle_x` が有限かつ正（0 だと `tan(fov/2)`→0 で focal が無限大）、
`transform_matrix` が有限、可逆（特異行列を拒否）、カメラ距離が正。

## 検証（レビュー指摘 6 を反映）

**既定の挙動が変わっていないこと**:
- `--pixal3d-weights` 省略と `--pixal3d-weights mv` を同一入力・同一 seed で流し、**GLB の
  sha256 が一致**すること（ファイルサイズと IoU だけでは、一部の段だけ SV に切り替わる回帰を
  検出できない）
- 公式 cyclops 4 視点で既存の結果と一致（GLB 31,020,608 B / silhouette IoU 0.9795）
- geometry-only（`--no-texture`）と texture あり の両方

**SV が動くこと**:
- `--pixal3d-weights sv` で 1 視点入力を通し GLB が出る
- 同じ 1 視点入力を `mv` でも通して比較する。多視点重みだけが破綻するなら、上記の実測が
  transforms.json の誤りではないことの直接の裏付けになる（現状この追試は未実施）
- 出力は**単一視点のレンダだけで判断しない**。front/back/left/right/top の複数視点で見る

**解決経路のテスト**:
- 4 本すべての解決パスを検査する単体テスト（混在の検出）
- 無効値、`--views` 無しでの指定、ファイル欠落、警告表の全セル

## 代替案と却下理由

- **ファイル名を全部 CLI 引数にする**: 柔軟だが 9 個の引数が増え、既存の `--models DIR` 規約を
  壊す。系列は 2 つしかないので接尾辞の選択で足りる。
- **models ディレクトリの中身から自動判定**: `_sv` しか無ければ sv、という推測は、両方を
  同じディレクトリに置いたときに黙って片方を選ぶ。どの重みで出た結果かを追えなくなる。
- **視点数から自動選択**: V=1 なら sv、という自動化は、単視点重みを持っていない利用者に
  「ファイルが無い」エラーを出す。明示選択にして、不利な組み合わせは警告で伝える。
- **系列ごとに別ディレクトリを `--models` で渡す**（レビュー提案）: 新しい引数が要らない点は
  優れるが、共有 5 モデル（6.0 GiB）を二重に置くか symlink 運用が要る。今回は採らないが、
  `--pixal3d-weights` と排他ではないので将来併用できる。
- **manifest の role 解決を native にも入れる**（レビュー提案）: ブラウザは既に role から
  ファイル名を解決する（`web/real_e2e/worker.js:27`）。系列・SHA256・サイズを一体で固定でき、
  4 本の混在を防げる点で接尾辞連結より堅い。**変更規模が大きいので本変更には含めないが、
  M2 の本実装で再検討する**。

## ブラウザ（WASM）

既存 MV への影響は無い。WASM の C ABI はモデルパスを個別引数で受けるので CLI の接尾辞切替に
依存しない（`pixal3d_real_geometry_wasm.cpp:171`）。ただし **SV をブラウザへ出す契約は未設計**
で、`worker.js:16` の legacy fallback と現行 manifest は MV 名。SV 用の manifest を別 model set
として用意済み（`pixal3d-sv-f16` / `pixal3d-sv-q8_0`、2026-09-15 生成）だが、UI からの選択は
本変更の範囲外とする。
