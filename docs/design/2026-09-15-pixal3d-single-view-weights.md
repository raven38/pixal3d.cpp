# Pixal3D 単視点重みを CLI から選べるようにする

- 状態: **実装済み（2026-09-16）**。改訂 2 で配布境界と FOV 契約を明文化
- 実測: CUDA/L4 で SV・MV とも V=1 生成が完走（190.3 s / 193.1 s）。同一入力・同一 seed・
  共有モデル共通で、SV は V=3,816,218 F=7,648,734、MV は V=3,979,224 F=7,977,482 と明確に
  異なる形状になった（5 視点レンダで目視確認）。入力が同一なので、この差は transforms.json
  では説明できない。ただし cyclops の正解形状を持たないため「どちらが正しいか」は未判定。
  既定（省略）と明示 mv は Metal で PLY バイト一致・GLB も生成時刻の 3 バイトを除き一致
- 対象 issue/マイルストーン: ROADMAP M2「productized single-view Pixal3D path」
- 変更ファイル: `include/trellis_args.h` / `src/trellis_args.cpp` / `src/trellis_cli.cpp` /
  `src/transforms_json.cpp` / `tools/convert.py`（SV エントリ） / `README.md` /
  `install/test_pixal3d_weight_selection.sh` / Linux CI
- 配布境界: **この PR は runtime + conversion support まで**。公開 installer / model manifest は
  引き続き MV-only で、SV の size/SHA 固定済み model-set 配布は follow-up とする。

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

**作らない**: 公式 SV の入力規約そのもの（後述）、新しい推論経路、SV model-set の公開配布。

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
   `trellis_args.cpp:126` が `--num-views 0` を、`transforms_json.cpp` が空 frames を、
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
ヘルパの内部で、出力座標系とは無関係。よって MV 経路の postprocess 既定値は SV でも
そのまま使ってよい。

## インターフェース

```
--pixal3d-weights sv|mv     Pixal3D flow weights variant (default: mv)
```

`--weights` ではなく `--pixal3d-weights` にする。`parse_args()` は CLI と server が共有し、
Pixal3D 経路の選択は `views` の有無なので、Pixal3D 専用名にして **`--views` 無しで指定されたら
エラー**にする。

`sv` を選ぶと 4 つのファイル名の接尾辞が `_mv` から `_sv` へ変わる。共有の 5 モデル
（`dinov3` / `pixal3d_naf` / `ss_dec` / `shape_dec` / `tex_dec`）は系列に依存しないので変えない。

### SV 重みの取得（この PR の配布境界）

通常の installer と公開 manifest はまだ MV-only である。したがって `--pixal3d-weights sv` を
試す source checkout では、TencentARC/Pixal3D の plain（非 `_mv`）safetensors/config を
`TRELLIS_MODELS` 配下の `ckpts/` に置き、次を実行する:

```bash
export TRELLIS_MODELS=/path/to/TencentARC/Pixal3D
export TRELLIS_GGUF_OUT=/path/to/pixal3d_models
python3 tools/convert.py \
  pixal3d_ss_flow_sv \
  pixal3d_shape_flow_512_sv \
  pixal3d_shape_flow_1024_sv \
  pixal3d_tex_flow_1024_sv
```

公開 installer に SV を載せる際は、4 flow のサイズ/SHA256を固定した別 model-set manifest を
作る。runtime selector はその follow-up と独立している。

## 事前検査（レビュー指摘 3）

現行は 4 本を段階的にロードし、texture flow は最後まで到達しないと存在確認されない。
`Model::load()` は例外を投げるため、`_sv` の texture flow だけ欠けていると、3 段走った後に
落ちうる。

**推論開始前に、選択した系列で実際に必要なファイルを一括検査する。** `--no-texture` のときは
texture flow を必須にしない。欠けていれば系列名と探したパスを出して終了する（既定の `_mv` へ
黙って落ちない）。

## 誤用への手当て（レビュー指摘 1 を反映）

多視点重みに 1 視点を与えると、素直な劣化ではなく破綻する。実測では、カメラ・入力を共通に
して視点数だけ 4→1 にすると多視点重みが **85勝0敗**（d30 85体、p=5.170e-26）、
**20勝0敗**（game 設定画 20体、IoU 中央値 +0.32）で負け、目視でも頭部から棘状の破綻が出る。
1 視点同士では多視点重みは単視点重みに 3勝17敗で負ける。

`pixal3d_cond_ss/slat` は全視点を単純平均するので、SV 重みに V=2-3 を与えれば別分布の条件に
なる。MV も 4 視点でしか検証していない。**検証済みの組み合わせだけを無警告にする**:

| weights | 無警告 | それ以外 |
|---|---|---|
| `sv` | V=1 | 警告（単視点重みに複数視点を平均投入している） |
| `mv` | V=4 | 警告（多視点重みは 4 視点で検証済み） |

実行は止めない（判断はユーザーに残す）。

## 入力検証（レビュー指摘 5 + 改訂 2）

壊れた 1 枚がそのまま条件になるので、入力段で次を検証する:

- `camera_angle_x` が有限かつ **`0 < fov < pi`**。0 は focal が無限大、`>= pi` は透視投影として
  不正なので parser 段階で拒否する。top-level/per-frame の両方に同じ契約を適用する。
- `transform_matrix` が有限、可逆（特異行列を拒否）
- カメラ距離が正

`install/test_pixal3d_weight_selection.sh` は実モデルなしで、FOV>=pi が model preflight より前に
落ちることも固定する。

## 検証（レビュー指摘 6 を反映）

**既定の挙動が変わっていないこと**:
- `--pixal3d-weights` 省略と `--pixal3d-weights mv` を同一入力・同一 seed で流し、PLY/GLB の
  実質内容が一致すること
- 公式 cyclops 4 視点で既存結果と一致
- geometry-only と texture あり の両方

**SV が動くこと**:
- `--pixal3d-weights sv` で 1 視点入力を通し GLB が出る
- 同じ 1 視点入力を `mv` でも通して差を確認する
- 出力は front/back/left/right/top の複数視点で見る

**weightless contract**:
- flag 省略時は MV ファイルだけを解決
- 明示 `sv` は SV ファイルだけを解決し、MV に黙ってフォールバックしない
- `camera_angle_x >= pi` はモデルロード前に拒否
- canonical-rig / natural-order の既存テストも維持

## 代替案と却下理由

- **ファイル名を全部 CLI 引数にする**: 柔軟だが 9 個の引数が増え、既存の `--models DIR` 規約を
  壊す。系列は 2 つしかないので接尾辞の選択で足りる。
- **models ディレクトリの中身から自動判定**: `_sv` しか無ければ sv、という推測は、両方を
  同じディレクトリに置いたときに黙って片方を選ぶ。どの重みで出た結果かを追えなくなる。
- **視点数から自動選択**: V=1 なら sv、という自動化は、単視点重みを持っていない利用者に
  「ファイルが無い」エラーを出す。明示選択にして、不利な組み合わせは警告で伝える。
- **系列ごとに別ディレクトリを `--models` で渡す**: 新しい引数が要らない点は優れるが、共有 5
  モデルを二重に置くか symlink 運用が要る。今回は採らない。
- **manifest の role 解決を native にも入れる**: 系列・SHA256・サイズを一体で固定できる点で
  接尾辞連結より堅い。変更規模が大きいので、本 PR では runtime selector と converter だけにし、
  公開 SV model-set の follow-up で再検討する。

## ブラウザ（WASM）

既存 MV への影響は無い。WASM の C ABI はモデルパスを個別引数で受けるので CLI の接尾辞切替に
依存しない。ただし **SV をブラウザへ出す契約は未設計**で、現行 Web manifest は MV 名のまま。
SV の公開 manifest / UI 選択は本変更の範囲外とする。
