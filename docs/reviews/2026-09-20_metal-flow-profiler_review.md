### レビュー結果サマリー

- 判定: **差し戻し**
- 重大な問題: 5件
- 軽微な指摘: 5件
- 承認条件:
  1. `ggml_graph_view` の非公開 API 依存を解消する。
  2. node 分割時間を「本番 graph 内での寄与率」と扱わない。
  3. プロファイル実行結果をサンプラへ渡さず、通常 whole graph の出力を維持する。
  4. 分割実行専用の warm-up と複数回測定を追加する。
  5. FA/NOFA を同一 `N`・同一テンソル入力で比較する。
  6. forward 出力の数値同値性と複数視点レンダを受け入れ基準に加える。

設計目的と postprocess 計時は妥当ですが、中核となる node 単位プロファイルの数値をそのまま性能寄与として解釈できません。現状の出力仕様では誤った最適化判断につながる可能性があります。

### 詳細フィードバック

#### 1. 設計の完全性

**重大: whole 実行と node 分割実行の順序・返却値が未定義です。**

設計には次の3つが併記されています。

- forward ごとに whole graph を計時する
- forward #1 を node 単位で実行する
- 分割実行の出力をサンプラでそのまま使う

しかし、whole→分割、分割→whole、forward #1 は分割のみ、のどれかが明記されていません。現在の通常経路は compute 後すぐ `gout_` を読み出します（[flow_runner.cpp](src/flow_runner.cpp:140)）。

推奨する契約は次です。

1. 常に whole graph を実行する。
2. `gout_` をホストへコピーし、その値をサンプラへ返す。
3. 対象 forward だけ、その後に profiling side-pass を実行する。
4. side-pass の出力は破棄する。

これなら `--profile` が推論結果へ影響しないことを設計上保証できます。「分割結果も同じはず」という仮定に依存しません。

**役割分類の仕様も、実装可能な不変条件として不足しています。**

DFS 後順と「次のマーカー」で区間を割る考え方は、現在の直接実行経路では概ね成立します。[dit.cpp](src/dit.cpp:351) の block 順も設計の前提どおりです。

ただし、最低限以下を実行時に検証すべきです。

- 各 block に `res_msa`、`res_cross`、`after_block` が各1個存在する。
- マーカーの順序が30 blockすべてで正しい。
- 未分類 node が0件、または明示した `other` に収まる。
- blockごとの分類 node 数を出力し、0件区間をエラーにする。

将来 scheduler や明示的な graph optimize を導入すると node 順は変わり得るため、長期的には `ggml_tensor* → {block, role}` の明示的メタデータを構築時に記録する方が安全です。

#### 2. 技術的実現可能性

**重大: `ggml_graph_view` は公開 API ではありません。**

このリビジョンでは宣言が `thirdparty/ggml/src/ggml-impl.h` にあり、公開 `include/` にはありません（[ggml-impl.h](thirdparty/ggml/src/ggml-impl.h:349)）。実体も単に元 graph の node 配列を指す shallow view です（[ggml.c](thirdparty/ggml/src/ggml.c:7185)）。

したがって、現提案の影響範囲だけでは通常はコンパイルできません。次のいずれかが必要です。

- vendored ggml に公開 wrapper API を追加する。
- trellis_core から内部ヘッダへ依存することを明示する。
- profiler を ggml 側に実装する。
- scheduler callback など公開 API を使う構成へ変える。

推奨は小さな公開 wrapper の追加です。その場合、`thirdparty/ggml` の変更と API 互換テストを影響範囲に加える必要があります。

**node view の実行そのものは、順番を守れば成立します。**

`ggml_graph_view(g, j, j+1)` は依存 node を自動実行しません。元 graph のトポロジカル順ですべての node を前から実行して初めて正しい依存値が得られます。途中の nodeだけの任意実行や、該当 node の繰り返し測定は、それだけでは正しくありません。

また、`ggml_backend_graph_compute` が async compute の後に synchronize するため、壁時計がGPU完了を含む点は正しいです（[ggml-backend.cpp](thirdparty/ggml/src/ggml-backend.cpp:444)）。

**重大: 「node 合計 / whole が歪みの上限」という解釈は成立しません。**

node 単位では毎回、概ね次の固定費が入ります。

- command buffer生成・encode・commit
- CPU/GPU同期
- backend dispatch
- wall-clock計測自体
- autoreleaseやスケジューリングの揺れ

固定費は軽い `VIEW`、`RESHAPE`、`ADD` 等を相対的に大きく見せ、重い GEMM を相対的に小さく見せます。したがって、全体比がたとえば2倍でも、個別カテゴリの誤差が2倍以内とは限りません。カテゴリ順位すら変わり得ます。

出力名は以下のようにすべきです。

- `isolated_node_wall_time`
- `isolated_share`
- `production contributionではない`
- `whole graphとの比較からカテゴリ別誤差は推定できない`

op別割合を最適化判断に使うなら、別の検証手段が必要です。

**Metal graph optimize に関する説明は修正が必要です。**

Metal backend が `use graph optimize = true` と表示しても、現在の `DitRunner` は backend を直接呼んでいます。graph optimize は scheduler の split 構築経路などから明示的に呼ばれる構造で、直接の `ggml_backend_graph_compute` 自体は optimize を呼びません。

したがって「1 node化により fusion / concurrency / graph optimize の三つが失われる」は正確ではありません。

- fusion: node間 fusion は失われる
- concurrency: node間 concurrency は失われる
- graph optimize: 現在の直接実行経路では、そもそも適用されていない可能性が高い

#### 3. リスク・懸念事項

**重大: forward #0 の whole graph warm-up は、node分割経路の warm-upになりません。**

whole graphでは fused kernel が選択されますが、single-node graphでは別の非fusion pipelineが選択される可能性があります。最初の node 分割測定には、その経路固有のpipeline生成が混入します。

最低でも以下が必要です。

- unmeasured node-wise warm-upを1回
- measured node-wise passを3回以上
- nodeごとは中央値を採用
- CPU submit時間が支配的な node を明示

ただし、全1024 flowでこれを行うと非常に高コストです。そのため、後述する階層型測定が現実的です。

**プロファイル時間が「forward 1本分程度」という受け入れ基準と矛盾しています。**

設計自身が node 合計は whole より大きいとしています。数千nodeすべてで同期するなら、追加時間は whole 1本分を大幅に超える可能性があります。

安全策として以下を追加してください。

- node数と予想測定回数を開始前に表示
- `--profile-flow ss|lr|hr|texture`
- `--profile-level role|block|node`
- node測定の時間上限または明示確認
- serverでは既定で node level を拒否、または対象リクエストを1回に限定

#### 4. 影響範囲の網羅性

次が不足しています。

- `thirdparty/ggml`: 公開 graph-view wrapperを追加する場合
- CMake: profiler helperまたは内部ヘッダを扱う場合
- parserのテスト
- profiler分類ロジックの単体テスト
- CPU/CUDA/WebGPU/WASMでのコンパイル確認、またはMetal限定の明示的拒否
- `trellis-server` の長時間リクエスト、ログ量、複数リクエスト時の挙動
- ドキュメント上の出力形式・単位・測定限界

postprocessについても、現在のMV末尾には提案された段以外に以下があります（[trellis_cli.cpp](src/trellis_cli.cpp:556)）。

- `clean_mesh`
- `drop_small_components`
- decimate後の再weld/fill/drop
- reference frame変換
- UV失敗時のchart fallback
- vertex-color fallback
- PNG書き出し
- PLY書き出し

「done inとの差を機械的に説明する」が目的なら、全区間の合計と residual を最後に表示すべきです。

```text
[time] load=...
[time] ss_cond_flow_decode=...
[time] lr_shape=...
[time] hr_shape=...
[time] shape_decode=...
[time] texture=...
[time] postprocess.weld_fill=...
...
[time] accounted=...
[time] residual=...
[time] total=...
```

なお、通常TRELLIS経路には `[7/7]` の別postprocess呼び出しがあります。MVだけを対象とするなら明示的にスコープ外としてください。

#### 5. テスト計画の妥当性

**重大: FA/NOFAを `N` で正規化する比較は不適切です。**

attentionは単純な線形計算量ではありません。

- self-attention: 概ね `N²`
- MLP: 概ね `N`
- cross-attention: `N × Lc`
- FlashAttention: tile、padding、kernel選択で段差がある

したがって、異なる `N` の時間を `N` で割って比較しても公平になりません。同じ保存済み `xt / cond / proj / RoPE / N` を与える固定fixtureでFA/NOFAを比較してください。

**V/F範囲だけでは推論非破壊性を確認できません。**

`V≈640k, F≈950k` は大きな破損しか検出できず、latentや形状の局所差を見逃します。必要なゲートは以下です。

- 同一入力に対する whole-only と `--profile` の forward出力比較
  - finite率
  - max absolute error
  - max relative error
  - RMSE
- サンプラ全体の最終latent比較
- GLBの構造検査
  - V/F
  - finite座標
  - component数
  - boundary/non-manifold
  - PBR/texture存在
- front/back/left/right/top等の複数視点レンダ比較

また、タイミングは1回ではなく、最低3回の中央値とばらつきを記録してください。長いE2Eを反復しにくい場合、固定forward fixtureで統計を取り、フル生成はFA/NOFA各1回の統合確認に分けるのが現実的です。

#### 6. 代替案

最も現実的なのは、**階層型プロファイル**です。

1. block/role単位の連続 view を順に実行する。
2. 1区間内ではfusion/concurrencyを残す。
3. block/role測定からボトルネックを特定する。
4. 遅い区間だけnode分割する。
5. 最終確認にMetal captureを使う。

これにより同期回数は「全node数」から概ね `30 blocks × 3 roles` に減り、測定対象の実運用との乖離も小さくなります。

代替案表には、さらに以下を加えるべきです。

| 案 | 評価 |
|---|---|
| role/block単位のgraph view | 第一候補。区間内fusion/concurrencyを保ち、同期固定費を抑えられる |
| vendored ggmlに計測hook追加 | 保守コストはあるが、正確なnode対応と将来の再利用性が高い |
| Metal command-buffer GPU timestamps | CPU submit時間を除ける。node対応にはbackend側instrumentationが必要 |
| whole graph A/B ablation | 特定fusionやFA経路の実効果確認に向くが、網羅的op内訳には向かない |
| node単位wall-clock | 診断用の孤立latencyとしてのみ有効。本番寄与率には使わない |

結論として、postprocess計時は小修正で進められますが、DiT profilerは「role/blockの階層計測を主、node計測を限定診断」とする再設計が必要です。特に非公開API依存、歪みの解釈、出力非破壊性、同一 `N` 比較の4点が解消されるまでは承認できません。


