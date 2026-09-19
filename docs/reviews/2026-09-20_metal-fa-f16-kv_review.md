# codex 設計レビュー（1 周目）: docs/design/2026-09-20-metal-fa-f16-kv.md — 2026-09-20

（`codex exec -s read-only`、標準出力を保存。判定: 差し戻し → 設計を改訂して再確認へ）

### レビュー結果サマリー

- 判定: **差し戻し**
- ソース確認上、Metal の BF16 経路が `q=bfloat / o_t=half`、F16 経路が `q=half / o_t=float` であるという中心的な観察は正しいです。
- ただし、head dimension、固定出力メモリ、F32 案の安全性、他 backend への波及、統計的判定方法に実装前修正が必要です。

重大な問題（A）:

- **A1: head_dim は 64 ではなく 128**。RMSNorm 上界、gamma 閾値、Metal pipeline 名、メモリ見積もりが連鎖的に誤っています。
- **A2: 範囲計測の固定出力は約 5 GB ではない**。self-attention だけで約 9.1 GiB、cross-attention の Q まで含む下限で約 12.1 GiBです。
- **A3: F32 K/V でも Q は Metal カーネル内で F16 化される**ため、「範囲リスクゼロ」「最も安全」は成立しません。
- **A4: 全型への `set_prec(F32)` 適用は Metal 以外の実行意味を変える**。CUDA/Vulkan と既存 `TRELLIS_FA_FAST=1` の互換性を壊します。
- **A5: `--no-fa` は有力な F32 reference だが、厳密な oracle と断定する根拠が不足**しています。
- **A6: n=1 の品質判定、速度 ABAB 2 対では既定変更の根拠になりません**。独立標本、再実行ノイズ床、統計的判定規則が必要です。
- **A7: HR で coords が不一致になった場合の GLB ゲートが未定義**です。「近いか同等」「V/F・bbox が同等」は機械判定できません。
- **A8: `g_fa_kv` の backend 解決場所が不完全**です。CLI の2箒所だけでは多数の `Model::load`、テスト、WASM、server 経路を網羅できません。

軽微な指摘（B）:

- **B1:** `~17.6k` 項の累積は主に HR self-attention の説明であり、短い cond を読む cross-attentionには当てはまりません。
- **B2:** `|x| < 6.1e-5` は「F16 正規化数未満」であり、直ちにゼロ化を意味しません。padding のゼロも除外すべきです。
- **B3:** 「K を縮小すれば softmax は厳密に不変」は実数演算上の説明です。cast 前後の量子化誤差は変わります。
- **B4:** dump の形式に shape、stage、coords対応、commit、重みハッシュ、backend、型、失敗時の扱いがありません。
- **B5:** 文書の対象ブランチは `perf/metal-rope-layout` ですが、確認時の checkout は `perf/cond-profile`、SHA は同じ `d4ffdf4` でした。再現記録は SHA を正本にしてください。

承認条件:

1. head_dim=128 で上界、pipeline名、メモリ見積もりを全面修正する。
2. F32 arm の Q-F16 リスクを明記し、「range-risk-free」の記述を撤回する。
3. Metal の dtype 比較と、CUDA/Vulkan の accumulation policy 変更を分離する。
4. 範囲診断を約12 GiBの全テンソル固定方式から、縮約値または block/stage 単位の取得へ変更する。
5. `--no-fa` を「F32 reference」と定義し、小規模な倍精度CPU参照で妥当性を確認する。
6. 速度・品質の独立反復数、主要評価量、CI/検定、同等性マージンを凍結する。
7. coords 不一致時の GLB比較式と閾値を数値で定義する。
8. backend解決を `DitRunner` 構築時など、全runner共通の一箇所へ置く。

### 詳細フィードバック

#### 1. 設計の完全性

- **A: backend 解決の責務が分散しています。**  
  設計書 [docs/design/2026-09-20-metal-fa-f16-kv.md:59](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/docs/design/2026-09-20-metal-fa-f16-kv.md:59) は `trellis_cli.cpp` の2箇所での解決を提案していますが、runner はCLI以外からも多数作られます。実際、[src/flow_runner.cpp:372](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/flow_runner.cpp:372) が共通factoryで、WASM・各テスト・serverから呼ばれています。  
  `requested_fa_kv` と `effective_fa_kv` を分離し、`DitRunner` 構築時に `m.backend` から解決して、解決結果をgraphに固定してください。グローバルな `AUTO` を一度F16へ上書きする方式は、同一プロセスで別backendを扱えません。

- **A: dump仕様が再現記録として不足しています。**  
  [設計書:80-85](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/docs/design/2026-09-20-metal-fa-f16-kv.md:80) のraw配列だけでは、shapeやstageの取り違え、別runによる上書きを検出できません。各armごとのディレクトリとJSON manifestを用意し、少なくとも次を保存すべきです。

  - commit SHA、ggml SHA
  - backend名、hardware、OS
  - effective K/V型とaccumulation設定
  - モデルファイルのSHA-256
  - 入力ハッシュ、seed、stage、shape、dtype
  - coords件数・順序・ハッシュ
  - 完了マーカーとatomic rename

- **B: CLI/envの優先順位が未定義です。**  
  `--fa-kv`、`TRELLIS_FA_KV`、`TRELLIS_FA_FAST` が同時指定された場合の規則を明記してください。推奨は `CLI > TRELLIS_FA_KV > TRELLIS_FA_FAST > auto`、不正値はsilent fallbackではなく起動失敗です。

- **B: `--no-fa` と `--fa-kv` の併用時の表示が必要です。**  
  K/V型が無視されることをログに出さないと、誤ったarmを計測したように見えます。

#### 2. 技術的実現可能性

- **A: head_dim の前提が誤っています。**  
  実装は [include/dit.h:18](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/include/dit.h:18) で `head_dim=128`、`n_heads=12` です。設計書の `sqrt(64)=8` は `head_dim/2=64` との混同です。

  正しい上界は次です。

  ```text
  |Q_i|, |K_i| <= sqrt(128) * max|gamma|
                 ≈ 11.314 * max|gamma|
  ```

  したがって、

  ```text
  overflow境界: max|gamma| ≈ 65504 / sqrt(128) ≈ 5790
  8倍headroom境界: max|gamma| <= 65504 / (8*sqrt(128)) ≈ 724
  ```

  「gammaが8000を超えた場合だけ」という記述は修正が必要です。なお、既存の [tools/fa_gamma_bound.py:32](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/tools/fa_gamma_bound.py:32) は128を使っており、こちらが実装と一致しています。

- **A: メモリ見積もりが過小です。**  
  N=17612、head_dim=128、12 heads、F32では、

  ```text
  self Q: 128 * 17612 * 12 * 4 = 約103.2 MiB
  self K/V各: 128 * 17664 * 12 * 4 = 約103.5 MiB
  self 30 blocks合計 ≈ 9.1 GiB
  cross Q 30 blocksだけでも ≈ 3.0 GiB
  ```

  したがって、cross K/Vを無視しても約12.1 GiBです。通常のgallocr活性化、重み、ホスト読出しbufferも別に必要です。[src/flow_runner.cpp:125-133](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/flow_runner.cpp:125) のようにoutput指定後にgraph全体を割り当てるため、単なるログ追加ではありません。

  全Q/K/Vを固定せず、次のいずれかにしてください。

  - graph内でabs/sum等を用いて縮約し、小さな統計テンソルだけoutput化する
  - block単位の診断runnerを作る
  - flow/step/blockを指定して限定取得する
  - まず短いpilotで実測allocationを印字し、device budget内であることをゲートにする

- **A: F32 K/VでもQはhalfです。**  
  `FA_TYPES_F32` は [ggml-metal.metal:6574](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/thirdparty/ggml/src/ggml-metal/ggml-metal.metal:6574) で `q_t=half`、K/VとOだけfloatです。Qは [ggml-metal.metal:5983-5988](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/thirdparty/ggml/src/ggml-metal/ggml-metal.metal:5983) でF32から`q4_t`へcastされます。  
  よってF32 armはK/V範囲リスクを除きますが、Q overflowは残ります。またpipelineは `dk64` ではなく、このモデルでは `dk128_dv128` です。

- **A: accumulation設定の変更がA/Bを汚染します。**  
  現在は [src/dit.cpp:254](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/dit.cpp:254) でF16 fast pathに`set_prec(F32)`を付けません。これを全型へ付けると、Vulkanでは [ggml-vulkan.cpp:9940](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/thirdparty/ggml/src/ggml-vulkan/ggml-vulkan.cpp:9940) の`f32acc`選択が変わり、CUDAもprecision指定を参照します。  
  Metal A/Bだけが目的なら、まずK/V型だけを変え、他backendのprec条件は現状維持すべきです。変更するなら `--fa-accum` を別issue・別ゲートに分離してください。

- **確認済みの肯定点:**  
  Metal はQにF32を要求しK/V同型を要求しています（[ggml-metal-ops.cpp:2661-2663](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp:2661)）。F16/BF16/F32 K/Vはいずれも対応し、BF16はdevice capability依存です（[ggml-metal-device.m:1225-1240](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/thirdparty/ggml/src/ggml-metal/ggml-metal-device.m:1225)）。型切替自体は実現可能です。

#### 3. リスク・懸念事項

- **A: Vにも新しいF16範囲リスクがあります。**  
  self/crossともVはRMSNormされていません。[src/dit.cpp:324-329](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/dit.cpp:324) と [src/dit.cpp:338-347](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/dit.cpp:338) ではQ/Kだけが`rms_gamma`を通ります。`V_SCALE=1/256`は閾値を約 `65504*256 ≈ 1.68e7` に広げるだけで、保証ではありません。  
  「新たなリスクはQ/Kだけ」は撤回し、Vは動的ゲートのみで保証されると明記してください。

- **A: paddingがsubnormal比率を汚染します。**  
  permute+pad後のK/Vで `|x|<6.1e-5` を数えると、padding zeroが全て含まれます。実tokenだけを分母にし、さらに次を区別してください。

  - exact zero
  - `0 < |x| < 2^-24`（F16最小subnormal未満）
  - `2^-24 <= |x| < 2^-14`（subnormal域）
  - cast後にzero/nonfiniteになった数

- **A: thermal controlが不十分です。**  
  `ps`はGPU利用、温度、周波数、Spotlightや映像処理負荷を検出できません。既存結果でも [profile結果:81-92](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/docs/results/2026-09-20-metal-flow-profile.md:81) に大きな単調ドリフトがあります。  
  90秒冷却に加えて、arm順をblock内でランダム化またはLatin-square化し、AC接続、電源モード、開始時刻、前run時間を保存してください。pipeline lazy compileは測定前に各型1回ずつ完了させる必要があります。

- **B: RMSNorm上界には有限精度の但し書きを付けるべきです。**  
  数学的論法は正しいですが、gamma shapeの検証、全flowでのhead_dim確認、RoPE cos/sin誤差を含む実測headroomとの併用が必要です。静的上界だけでVや非有限値は保証できません。

#### 4. 影響範囲

- **A: 「Metal autoだけが変わる」は現設計では不正確です。**  
  `set_prec(F32)`の全型適用、明示的`--fa-kv f16/f32`、legacy aliasの意味変更はCUDA/Vulkan/WebGPU/CPUへ波及します。特にWebGPUはBF16 FA非対応で、F16/F32のみ対応しています（[ggml-webgpu.cpp:4268-4277](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/thirdparty/ggml/src/ggml-webgpu/ggml-webgpu.cpp:4268)）。

- **A: backend matrixが不足しています。**  
  少なくとも次を設計上固定してください。

  | Backend | auto | 明示bf16 | 明示f16 | 明示f32 |
  |---|---|---|---|---|
  | Metal BF16対応 | ゲート前BF16 / 後F16 | 実行 | 実行 | 実行 |
  | Metal BF16非対応 | F16または明示エラー | capability error | 実行 | 実行 |
  | CUDA | BF16現状維持 | smoke | smoke、prec現状維持 | smoke |
  | Vulkan | BF16現状維持 | smoke | 既知RADV制約を明示 | smoke/unsupported判定 |
  | WebGPU | no-FA既定 | 明示エラー | 対応時のみ | 対応時のみ |
  | CPU | BF16現状維持 | smoke | smoke | smoke |

  unsupportedの場合はgraph compute時の失敗ではなく、構築時に明示エラーにしてください。

- **A: server/Studioの運用可視性が不足しています。**  
  parser共有だけでは、実際に選ばれたbackend・K/V型・accumulationが分かりません。server起動ログと`/capabilities`にeffective policyを出し、Studio同梱serverでも確認できるようにしてください。

- **B: 既存文書の修正範囲を追加してください。**  
  [README.md:345](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/README.md:345) のM1向け`TRELLIS_FA_FAST=1`、[docs/spec/29-perf-profile.md:38](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/docs/spec/29-perf-profile.md:38) の「F16 accumulate」「HRはF16範囲超過」、WebGPU gap文書の型説明も同時更新対象です。

#### 5. テスト計画

- **A: 品質n=1では「F16がBF16よりoracleに近い」と一般化できません。**  
  配列要素数をnとして扱うこともできません。要素は強く相関しており、実験単位は入力×seedです。

  推奨する最低設計は次です。

  1. 同一入力・同一seed・同一armを2回以上実行し、非決定性のノイズ床を測る。
  2. 性質の異なる2入力クラス以上を用意する。
  3. pilot後、主要評価量を `log(RMS_error_f16 / RMS_error_bf16)` のようなpaired scalarに固定する。
  4. 最低でも **2クラス×5 seed = n=10 paired cases** を使い、片側95% CIの上限が0未満ならsuperiorityとする。
  5. nを削る場合は結論を「このfixture・seedでの1例観測」に限定し、既定変更には使わない。

  max errorは外れ値依存が強いため、主要評価量をRMS、maxを安全ゲートに分ける方が適切です。

- **A: 速度ABAB 2対は小さすぎます。**  
  F32を含む3腕なのに、記述されたABABにはF32がありません。[設計書:94](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/docs/design/2026-09-20-metal-fa-f16-kv.md:94) を修正し、各flowで少なくとも5～8 counterbalanced blocksを取り、paired speed ratioの95% CI全体が1を超えることをゲートにしてください。  
  「3 fwdの#1–#2平均」は、#0がcompile warm-upであることを明記してください。

- **A: coords不一致時の判定が機械化されていません。**  
  coordsの順序だけが違う場合は座標でjoinしてlatentを比較できます。set自体が違う場合は、まず以下を出すべきです。

  ```text
  voxel Jaccard(f16, oracle)
  voxel Jaccard(bf16, oracle)
  symmetric-difference count
  occupancy IoU
  ```

  GLBについても、V/Fは品質距離ではありません。bboxは中心差・各軸extent相対差、表面は双方向Chamfer/Hausdorff、レンダは6視点の対応画像を保存する、と定義してください。閾値は事前に既知good/badで妥当性確認が必要です。

- **A: `IoU差がseed間差±0.01以内`は現計画では評価不能です。**  
  seed間差を先に複数seedで実測し、その分布からマージンを決める必要があります。固定seed 1本からseed間差は得られません。

- **A: holdoutがありません。**  
  R/Qで使ったcyclops seed 1を、auto切替後の最終E2Eにも再利用しています。既定変更の最終確認は別入力・別seedをholdoutにしてください。

- **B: 既存テストの記述が曖昧です。**  
  「参照テンソルがあれば」ではなく、各コマンド、期待exit code、期待する数値上限、対象backendを列挙してください。CLI precedence、invalid value、legacy alias、backend auto resolution、unsupported dtypeのnegative testも必要です。

#### 6. 代替案

- **A: F32 K/Vをrange-risk-freeな代替案として扱えません。**  
  Qはhalfのままなので、正しくは「K/V castの範囲リスクを除く比較arm」です。

- **B: BF16 K/V + float Oを第4 armとして早期評価する価値があります。**  
  現象の因果を分離できます。

  ```text
  BF16現状: q=bfloat, K/V=bfloat, O=half
  BF16+float-O: q=bfloat, K/V=bfloat, O=float
  F16現状: q=half, K/V=half, O=float
  F32 K/V: q=half, K/V=float, O=float
  ```

  BF16+float-Oを加えると、改善がQ/Kの仮数差によるものか、O累積精度によるものか判別できます。forkパッチ増加は欠点ですが、原因同定用の実験armとしては有用です。

- **B: Kだけのpower-of-two scalingは片側対策です。**  
  Q overflow時には効きません。必要ならQまたはKを縮小してscore scaleを逆補正する一般形として記述してください。また、実数上は同値でもF16 cast誤差は変わるため「厳密に不変」は「cast前の実数演算では同値」に言い換えるべきです。

中心仮説には十分な技術的根拠がありますが、現状のまま実装・計測へ進むと、誤ったhead_dimと約2倍以上過小なメモリ見積もり、backend間で異なるaccumulation条件、小標本による既定変更が結果を支配します。上記A項目を設計書へ反映した後であれば、再レビュー可能です。


---

# 再確認（2 周目）: 改訂版に対する A1〜A8 / 承認条件 1〜8 の解消確認

改訂版と現 checkout の実装を照合しました。

### A1〜A8

| 項目 | 結果 | 理由 |
|---|---|---|
| A1 | 解消 | head_dim=128で上界・gamma閾値・`dk128_dv128`・メモリが再計算されています。実装も [dit.h](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/include/dit.h:18) で128です。 |
| A2 | 解消 | 固定出力を約12.1 GiBと正しく認識し、追加活性化を含む実測allocのpilot gateも追加されています。 |
| A3 | 解消 | F32 K/VでもQはhalfになることを明記し、「K/V castの範囲リスクだけを除く比較腕」と限定しています。 |
| A4 | 解消 | `set_prec(F32)`はbf16/f32だけ、f16には付けない従来条件へ戻っています。実装も [dit.cpp](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/dit.cpp:323) と一致します。 |
| A5 | 未解消 | `--no-fa`を依然として「exact SDPA、oracle」と呼び、承認条件で要求した小規模な倍精度CPU参照を明示的に実施しない設計です。既存golden一致は有力な根拠ですが、条件5そのものは満たしません。 |
| A6 | 未解消 | 品質は1 fixture×5 seedに留まり、前回要求した2クラス×5 seedを満たしません。速度もHRは各腕4標本で、前回提示した5〜8 block未満です。 |
| A7 | 未解消 | coords不一致時の相対Jaccard判定は追加されましたが、GLBのbbox/V/Fは記録だけで、数値閾値がありません。両候補が共に悪い場合を排除する絶対Jaccard下限もありません。 |
| A8 | 解消 | backend解決は [build_dit_dense](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/dit.cpp:497) 入口で `m.backend` から行われ、graphごとに固定されています。 |

### 承認条件1〜8

| 条件 | 結果 | 理由 |
|---|---|---|
| 1 | 解消 | head_dim=128に基づく各計算へ修正済みです。 |
| 2 | 解消 | F32 armに残るQ-F16リスクを明記済みです。 |
| 3 | 解消 | Metalのdtype比較と他backendのaccumulation変更を分離しています。 |
| 4 | 未解消 | 約12.1 GiBの全テンソル固定方式をそのまま採用しています。pilotによる予算確認は安全策ですが、「縮約値またはblock/stage単位へ変更」という条件は満たしません。 |
| 5 | 未解消 | F32 referenceへの言い換えはありますが、倍精度CPU参照による妥当性確認がありません。 |
| 6 | 未解消 | 評価量・CI・5%同等性マージンは凍結されましたが、品質の2クラス×5 seedと速度の最低反復数が不足しています。 |
| 7 | 未解消 | GLB比較の数値閾値が定義されていません。 |
| 8 | 解消 | 全runner共通の `build_dit_dense` 入口へ移動済みです。 |

B（1行）: [dit.h](/Users/s25705/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/include/dit.h:69) のコメントは「Metal AUTO = F16」と読めますが、設計と実装はゲート前AUTO = BF16で不一致です。

**判定: 差し戻し**

未解消条件は **4・5・6・7** です。特に既定変更の根拠に直結する条件5〜7が残っているため、現時点では条件付き承認にはできません。
