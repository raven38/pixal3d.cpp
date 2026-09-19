# Flow DiT の op/block 別プロファイラ（`--profile`）と postprocess の段別計時

日付: 2026-09-20 / 対象ブランチ: `feat/desktop-0.10.0-sv-parity` / ステータス: 実装済み（codex レビュー `docs/reviews/2026-09-20_metal-flow-profiler_review.md` = 差し戻し → 下記「レビュー対応」で改訂）

## 概要

macOS (M4 Max / Metal) の MV 1024 生成は約 24〜28 分で、段別計測（`[flow] N steps, M forwards, X s`）
までは docs にあるが、その下（transformer block 単位・op 単位）の内訳は無い。
`trellis-cli --profile` を追加し、各 flow の 1 forward を node 単位に分割実行して
役割（self-attn / cross-attn+proj / MLP）別・op 種別・block 別に集計して印字する。
同時に `[6/6]` postprocess（weld / remesh / decimate / uv_bake / GLB 書き出し）の段別時間を
印字し、「4 段の flow 合計と `done in` の差 ≒ 5〜6 分」の内訳を機械的に出せるようにする。

## 背景（実測、`docs/results/canonical-rig/json_run.log`、M4 Max / Metal / q8_0 MV v1）

| 区間 | 実測 | うち `[flow]` |
|---|---:|---:|
| [2/6] SS cond + flow + decode | 95.8 s | 89.7 s |
| [3/6] LR shape SLAT | 109.4 s | 94.7 s |
| [3/6] HR shape SLAT（N = 17 612） | 677.3 s | 627.5 s |
| [4/6] shape decode | 23.2 s | — |
| [5/6] texture cond(NAF@1024) + flow + PBR decode | 573.0 s | 478.0 s |
| [1/6] load + [6/6] postprocess（差分） | 191.1 s | — |
| 合計 `done in` | 1669.8 s | 1289.9 s |

- Shape-1024 は 627.5 s / 20 forwards = 31.4 s/forward。同じ q8_0 セットの L4 CUDA は 113.1 s / 20 = 5.7 s。
- Mac 上の唯一の A/B（`docs/PIXAL3D_E2E_STATUS.md:489`）で SS flow が FA 144.4 s vs NOFA 89.3 s、
  Shape-512 が 102.2 s vs 63.5 s と **FlashAttention の方が遅い**。Shape-1024 / Texture は既定 FA。
- ggml-metal は op 単位の GPU タイムスタンプを公開していない（`GGML_METAL_CAPTURE_COMPUTE` の
  Xcode capture のみ）。`ggml_backend_graph_compute` は同期なので、node 単位に分割して壁時計で
  測れば op 単位の近似値が取れる。

## 提案

### 1. `--profile` フラグ（`trellis-cli` / `trellis-server` 共通の引数パーサ）

- `include/trellis_args.h`: `bool profile = false;` と `extern bool g_profile;`（`g_no_fa` と同じ置き方）。
- `src/trellis_args.cpp`: `--profile` の parse と help 1 行。
- `src/trellis_cli.cpp`: `trellis::g_profile = cfg.profile;` を `g_no_fa` を設定している 2 箇所
  （`trellis_run_mv` / `trellis_run_sv`）に並べる。
- `src/dit.cpp`: `bool g_profile = false;` を定義。
- `README.md` のフラグ一覧に 1 行追加（CLAUDE.md: 挙動を変えるものは env でなくフラグ）。

### 2. 役割マーカー（`src/dit.cpp::block()`）

`ggml_set_name` は名前を付けるだけでグラフの形は変えない。block ごとに残差加算後の `h` に
`blk{i}.res_msa` / `blk{i}.res_cross` を無条件で付ける。MLP 末尾は `build_dit_dense` の
`keep("after_block{i}")` が既に無条件で付けている。
`ggml_build_forward_expand` の DFS 後順では、あるマーカーの直前の未訪問ノード群がその区間の
計算になるので、「node j の役割 = index ≥ j の最初のマーカー」で帰属できる。
既知のずれ: t_embedder / adaLN の数ノードと block 0 の最初の layernorm は block 0 の self-attn 区間に
混ざる（コストは無視できる。設計上の既知事項として結果に注記する）。

### 3. プロファイラ本体（`src/flow_runner.cpp::DitRunner::forward`）

`g_profile` のとき:

- forward ごとに `ggml_backend_graph_compute` 全体の壁時計を印字（`[prof] fwd #k whole: X s`）。
- **forward #1**（#0 は Metal のパイプライン遅延コンパイルで汚れる。`json_run.log` に
  `ggml_metal_library_compile_pipeline` が最初の forward で出ている）を
  `ggml_graph_view(g_, j, j+1)` + `ggml_backend_graph_compute` で node 単位に実行し、各 node の時間を取る。
- 集計と印字:
  - `[prof] N=… Lc=… weights … MB | activations … MB | cond … MB`
  - 役割別（self_attn / cross_attn+proj / mlp / input / t_emb / final）: 秒と割合
  - op 別（`ggml_op_name`。`MUL_MAT` は `src0->type` で `MUL_MAT(q8_0)` / `MUL_MAT(f16)` /
    `MUL_MAT(f32)` に分ける — 重み GEMM と attention の活性化同士の GEMM を区別するため）
  - block 別 30 行（msa / cross / mlp の秒）
  - 上位 15 node（op・名前・shape・秒）
  - **node 合計 / whole の比**。Metal backend は `use fusion / use concurrency / use graph optimize = true`
    で動いており（同ログ）、1 node グラフでは三つとも効かないので、node 合計は whole より大きく出る。
    この比が歪みの上限で、比を付けずに op 割合を公表しない。
- 同じ forward の出力（速度）は分割実行でも whole でも同じテンソルに書かれるので、サンプラの
  結果には影響しない（プロファイル forward の出力をそのまま使う）。

### 4. postprocess の段別計時（`src/trellis_cli.cpp` `[6/6]`）

weld / fill / bvh / remesh / decimate / uv_bake / GLB+PNG 書き出しの各所に `(%.1fs)` を付ける。
`trellis_run_mv` の `t0` は views の読み込み前なので、`[1/6]` の時間も `(%.1fs)` で出す。
`--profile` に関係なく常時印字する（`[flow]` の行と同じ扱い。1 行ずつで冗長にならない）。

## 影響範囲

| ファイル | 変更 |
|---|---|
| `include/trellis_args.h` | `profile` フィールド、`g_profile` extern |
| `src/trellis_args.cpp` | `--profile` parse / help |
| `src/trellis_cli.cpp` | `g_profile` 設定 2 箇所、`[6/6]` の計時印字 |
| `src/dit.cpp` | `g_profile` 定義、block マーカー名 |
| `src/flow_runner.cpp` | forward カウンタ、node 分割計測と集計 |
| `include/flow_runner.h` | `fwd_count_` メンバ |
| `README.md` | フラグ 1 行 |

破壊的変更なし。`--profile` 無しでは `ggml_set_name` の追加以外に挙動差が無い
（名前は `trellis_graph_dump` の出力に載るだけ）。WASM ビルドは `g_profile` を触らない（既定 false）。

## テスト計画

1. ビルド（`build-metal`、Release）。
2. `--profile` 付きで cyclops 4 view（`docker/linux-webgpu-gate/e2e/views` + `transforms.json`、
   seed 1、`--res 1024`、q8_0 MV v1）を 1 回。各 flow で `[prof]` ブロックが出ること、
   node 合計 / whole 比が印字されること、`[flow]` の合計が `--profile` 無しの既知値
   （SS 89.7 / LR 94.7 / HR 627.5 / tex 478.0 s）から forward 1 本分程度しか増えないこと。
3. 同じ入力で `--profile --no-fa` を 1 回（FA/NOFA の A/B。N が SS decode の離散判定で僅かに
   変わるので、forward 時間は N で正規化して比べる）。
4. 出力 GLB の V/F が既知値の範囲（V≈640k, F≈950k）にあること（プロファイルがサンプラ結果を
   壊していない確認）。
5. 結果は `docs/results/2026-09-20-metal-flow-profile.md` に、backend / hardware / fixture / 重みセットを
   明記して残す（機械固有パスは `<home>` / `<scratchpad>` に置換）。

## 代替案

| 案 | 却下理由 |
|---|---|
| Xcode Instruments / `xctrace` Metal System Trace | コード変更なしで GPU カーネル単位が見えるが、trace が巨大で node → 役割の対応付けを手で行う必要がある。まず本案で粗い内訳を取り、必要なら特定 op に絞って使う |
| `GGML_METAL_CAPTURE_COMPUTE` | 1 graph の GPU capture。Xcode でしか読めず集計できない |
| `ggml_backend_sched` の eval callback | 本コードは sched を使わず 1 graph 直叩き。導入すると割り当てが変わり本番経路と別物になる |
| env `TRELLIS_PROFILE` | CLAUDE.md の「挙動を変えるものはフラグ」に反する |

## レビュー対応（codex 差し戻し → 改訂版、実装はこの改訂版）

| 指摘 | 対応 |
|---|---|
| `ggml_graph_view` は非公開 API（`ggml-impl.h`）で Windows DLL では export されない | 公開 API の `ggml_new_graph_custom` + `ggml_graph_clear` + `ggml_graph_add_node` でスライスを組む（DFS を伴わないので祖先を引き込まない） |
| node 単位の壁時計は submit/wait の固定費を含み、軽い op を過大評価する。sum/whole 比は上限にならない | **階層型**にした。役割・block 表は role×block の**連続スライス**（~93 本、スライス内の fusion / concurrency は保たれる）から取り、node 単位は op 種別の参考値として出力に `ISOLATED` と明記する。whole / segment / node の 3 つの合計を並べて印字する |
| forward #0 の whole は node 分割経路の warm-up にならない | node パスは 2 回回し、1 回目（非 fusion パイプラインのコンパイルを含む）は時間だけ印字、2 回目を採用 |
| 出力の契約が未定義 | whole を先に実行し `gout_` をホストへ読み出してから副次パスを回す。サンプラは常に whole の結果を使う |
| FA/NOFA は同一 N で比較すべき | N を印字する。SS decode の離散判定で N が数 % ずれる可能性はあるが、探している効果（SS で 1.6×）に対して無視できる。実測では両方とも N=17612 で一致した |
| マーカーの健全性検証 | 各 block に res_msa / res_cross / after_block の 3 つが揃っているかを実行時に検査し、欠けていれば警告 |
| 「Metal の graph optimize が失われる」は不正確 | 直接 `ggml_backend_graph_compute` を呼ぶ経路では graph optimize は元々適用されない。失われるのはスライス境界をまたぐ fusion と concurrency のみ、と訂正 |
| 影響範囲: 分類ロジックの単体テスト、他 backend でのコンパイル確認 | 分類は `[prof] warning` の実行時検査で代替。他 backend は公開 API のみ使用しているのでコンパイル上の差異は無い（CI の Linux / Windows ビルドで確認） |
| postprocess の accounted / residual 表示 | `[post]` の各段と `postprocess total`、`[1/6]` の `load` を印字する。段の合計と `done in` の差は log から機械的に出せる |

実装の所在: `src/flow_runner.cpp::DitRunner::profile_forward`（階層型計測）、`src/dit.cpp::block`（マーカー名）、
`src/trellis_cli.cpp`（`[post]` 計時）、`src/trellis_args.cpp` / `include/trellis_args.h`（`--profile`）。
