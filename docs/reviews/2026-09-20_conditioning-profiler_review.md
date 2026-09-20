# codex 設計レビュー: conditioning プロファイラ（issue #33）— 1 周目

日付: 2026-09-20 / 対象: `docs/design/2026-09-20-conditioning-profiler.md`（初版）/ 判定: 差し戻し → 改訂版で全項目対応（設計書末尾「レビュー対応」）

### レビュー結果サマリー

- 判定: **差し戻し**
- 必須セクション自体は揃っています。
- ただし、現案のまま実装すると、プロファイラが conditioning の入力・後続計算・熱状態を変える経路があり、得られた時間を本番経路のボトルネックとして解釈できません。

#### A: 実装を止める項目

1. **同じ `g_profile` が flow の大量 side-pass と conditioning 計測を同時に有効化する**

   現行 `--profile` は各 `DitRunner` の第2 forward後に segment pass と node pass 2回を追加します（[flow_runner.cpp:161](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/flow_runner.cpp:161)、[flow_runner.cpp:264](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/flow_runner.cpp:264)）。  
   この状態で設計書 §4 の E2E を走らせると、conditioning は通常の flow 後ではなく、追加 GPU 負荷で熱・メモリ状態を変えた後に測られます。既存実測でも profile pass が HR 段に約100秒加わっています（[metal-flow-profile.md:13](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/docs/results/2026-09-20-metal-flow-profile.md:13)）。

   `profile_flow_ops` と `profile_conditioning_stages` を別フラグまたは別レベルに分離してください。

2. **`cond_slat_gpu_chunked` の `naf_enc` 再実行は後続計算を変え得る**

   `naf_enc` の出力は永続バッファ `P_pooled` に書かれ、その後の `naf_qk` が消費します（[pixal3d_cond_gpu.cpp:310](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/pixal3d_cond_gpu.cpp:310)）。`run_graph` 内で node side-pass を実行すると、まだ本番経路で使用予定の `P_pooled` を再書き込みします。

   gallocr による scratch/input 再利用後の単一 node 再実行は、元入力が保存されている保証もありません。したがって「同じ値を再計算するだけ」は成立していません。以下のいずれかが必要です。

   - production graphとは別の診断 invocation でのみ op profileする
   - profiling後に入力を再アップロードして whole graphを再実行し、その結果だけを後続へ渡す
   - 永続出力を保存・復元する

3. **提案された共通関数の契約では、安全な再実行も既存出力形式の維持もできない**

   `trellis_profile_graph_ops(backend, graph, label)` には、次が渡されません。

   - gallocr再利用後に必要な入力を復元する callback
   - side-pass前後で保護すべき出力
   - flow profilerの role/block区間
   - whole graph時間
   - op時間を既存の `op_role_s` へ戻す手段

   現在の flow profilerは node時間を role/block情報と結合しています（[flow_runner.cpp:279](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/flow_runner.cpp:279)）。提示された3引数APIへ単純に切り出して「出力形式不変」は実現できません。

   共通関数は印字関数ではなく、少なくとも `std::vector<NodeTiming>` を返す測定 primitiveとし、入力復元 callback・warm-up回数・side-effect契約を明示すべきです。

4. **wall-clock計測とop side-passを同一実行で有効化すると、外側の時間が汚染される**

   `dinov3_encode` や `naf_upsample_ggml` の内部でop再実行すると、外側の `dino` / `naf` / `pixal3d_cond_slat` wall timeはproduction処理とside-passの合計になります。これは設計書 §3 の「§2の壁時計が先」という前提に反します。

   次の二相を別プロセスで実施してください。

   - Phase 1: exclusive wall-clockのみ
   - Phase 2:選択した単一graphのop profileのみ。Phase 1の時間表には混ぜない

5. **時間の照合式が成立しない**

   テスト計画では `[cond]` 合計と `HR shape SLAT` / `texture SLAT + decode` の一致を求めています（[設計書:160](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/docs/design/2026-09-20-conditioning-profiler.md:160)）。しかし既存の段合計は flow computeも含み、新しい `[cond]` 一覧にはflow computeがありません。

   正しい不変条件は概ね次です。

   ```text
   existing_stage_total
     ≈ exclusive_non_flow_substages
     + flow_whole_forwards
     + explicitly_reported_residual
   ```

   さらに現在の案では以下が未計上です。

   - `pixal3d_cond_slat` の巨大な `out.proj.assign`、正規化、camera setup
   - `decode_unet` の `from_latent` と最終 `output_layer`
   - `sparse_c2s` 内部の2個目の neighbor table、mask/coords構築
   - graph cleanup / gallocr free
   - debug readback
   - `[prof]` の入れ子による二重計上

   inclusive/exclusiveを出力名で区別し、各段に `accounted / residual / total` を出してください。

6. **A/B計測手順に順序効果と反復不足がある**

   `trellis-test-pixal3d-cond-tex --host` はGPU版を先に走らせ、その直後にhost版を走らせます（[test_pixal3d_cond_tex.cpp:111](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/test_pixal3d_cond_tex.cpp:111)、[test_pixal3d_cond_tex.cpp:126](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/test_pixal3d_cond_tex.cpp:126)）。host版もDINO/NAFのdevice computeを使うため、後順の値は前順の熱・メモリ圧を受けます。

   各方式を別プロセスにし、最低でも ABBA、各3回、median・min/maxまたはMADを記録してください。`ps` の開始前スナップショットだけでは、実行中のGPU共有やWindowServer負荷を除外できません。

7. **非破壊性ゲートが弱い**

   V/Fとbboxが一致しても、局所形状・latent・PBRの変化は検出できません。プロファイル有無で次を比較すべきです。

   - conditioningの `global` / `proj`: finite、max abs、RMSE、L2rel
   - 各flowの返却latent
   - GLB: V/F、成分数、finite座標、boundary/non-manifold、PBR有無
   - front/back/left/right/topの複数視点レンダ

   特にchunked経路については、op side-pass直後の永続バッファが後続結果を変えないことを直接検証する必要があります。

8. **「段階2を入れる条件」が未決定のまま acceptance と衝突する**

   issue acceptance は「stage/op内訳」ですが、§3は「device compute支配と分かった経路だけ」と条件付きです。どの割合なら実施するか、実施しない場合でもacceptanceを満たすかが未決定です。

   例として「exclusive computeが段時間の30%以上ならop profileを行う」のように事前固定してください。

#### B: 後から直せるが、本PR内で解消したい項目

1. **WASMの説明が不正確**

   「WASMも同じ `trellis_core`」ではありません。トップレベルCMakeはWASM時にreturnし、`web/ss/CMakeLists.txt` が専用ソース集合を直接コンパイルします（[CMakeLists.txt:35](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/CMakeLists.txt:35)、[web/ss/CMakeLists.txt:16](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/web/ss/CMakeLists.txt:16)）。

   現在は `dit.cpp` も集合に含まれるため `g_profile` のリンクは成立しそうですが、WASM build/linkテストが必要です。

2. **Metal unified memoryを「host往復」と表現するのは不正確**

   PCIe転送ではなくても、4 GiBのhost materialization、コピー、cache/page pressure、un-permuteのメモリ帯域負荷は残ります。「device↔host転送」ではなく「backend readback APIによるhost materializationとメモリ圧」と記録してください。

3. **stdoutについて実装条件が不足**

   CLI/serverは既にstdoutをunbufferedにしています（[trellis_cli.cpp:301](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/trellis_cli.cpp:301)、[trellis-server.cpp:255](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/trellis-server.cpp:255)）。一方、テストバイナリには同じ保証がありません。

   各計測時刻をprintf前に確定し、必要なら1行ごとに `fflush(stdout)` してください。また `TRELLIS_DBG_NAF` と `g_profile` が同時に有効でも重複印字しない条件が必要です。

4. **全backendへのコンパイル影響が未検証**

   変更対象はMetal専用ではなく、CPU/CUDA/Vulkan/WebGPU/WASM共通ソースです。最低限以下をテスト計画へ追加してください。

   - native CPU build
   - Metal Release build
   - `trellis-server` buildとcontract test
   - WASM full/partial targetのbuild/link
   - CUDA/VulkanはCI対象、または未検証と明記

5. **`--profile` 無しログの単純diffはテストにならない**

   時間値や実行環境由来の行は毎回変わります。ログを正規化し、追加可能なprefix集合と既存行の順序を機械検査する方が妥当です。

6. **top 1〜3の選定規則がない**

   exclusive median wall timeで順位付けするのか、削除可能時間の上限で順位付けするのかを固定してください。候補(b)のような共有案では、単純なlap合計が実現可能speedupとは限りません。

### 承認条件

以下を設計へ反映すれば、再レビュー可能です。

1. flow side-passとconditioning wall-clock/op profileを別モードに分離する。
2. wall-clock計測とop計測を別実行にする。
3. chunked `naf_enc` の永続出力をside-passから保護する。
4. 共通op profiler APIに入力復元・出力保護・測定結果返却の契約を定義する。
5. exclusive/inclusive、accounted/residual、flowとの照合式を定義する。
6. ABBA・3反復以上・別プロセス・実行順と共有負荷の記録を採用する。
7. profile有無のconditioning/latent/GLB/複数視点レンダ比較を追加する。
8. stage 2の発火基準とissue acceptanceとの関係を確定する。
9. server/WASM/native CPUを含むbuild matrixを追加する。

### 詳細フィードバック

#### 1. 設計の完全性

概要・背景・提案・影響範囲・テスト計画・代替案は存在します。ただし、以下が未決定です。

- 段階2を実施する定量条件
- op profile時の入力復元方法
- chunked経路の永続出力保護
- exclusive/inclusive計時の定義
- top 1〜3の順位基準
- 失敗時・時間上限超過時の扱い
- freshを何秒・何条件で定義するか

したがって、形式上は完全でも、実装者が意味を変えずに実装できる程度には凍結されていません。

#### 2. 技術的実現可能性

`ggml_backend_graph_compute` は内部でasync compute後にsynchronizeするため、壁時計で同期computeを測る前提自体は成立します（[ggml-backend.cpp:444](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/thirdparty/ggml/src/ggml-backend.cpp:444)）。

同じgallocr上のnode再実行も、バッファ解放前なら機械的には可能です。しかし正しさには次が必要です。

- input tensorがgallocr再利用で破壊されていない
- nodeを依存順に実行する
- pass開始前に全inputを再投入する
- profiling出力がproduction用永続バッファを上書きしない
- whole出力をside-pass前にhostへ退避する

現設計は最後の項目しか満たしていません。

#### 3. リスク・懸念

最大リスクは、計測そのものが次に測る区間を変えることです。

```text
flow whole
  → flow segment/node side-pass
  → 熱・メモリ圧上昇
  → conditioning wall time
  → conditioning op side-pass
  → 次viewまたは次stage
```

この順序では、stage時間もA/B差もproduction経路の値ではありません。外部GPU共有だけでなく、自分自身のプロファイラ負荷が主要な交絡です。

また、入れ子の `[prof]` と `[cond]` を同じ集計表で足すと二重計上になります。結果文書には必ず「階層」と「足してよい列」を定義してください。

#### 4. 影響範囲

列挙されたファイルは主要箇所を押さえていますが、次が不足しています。

- `web/ss/CMakeLists.txt` とWASM build scripts
- serverのログ・長時間リクエスト契約
- profiler API/出力schemaの単体テスト
- parser/helpのテスト
- CPU/CUDA/Vulkan/WebGPUのbuild影響
- 結果を再現するコマンドとraw log保存先
- `decode_unet` 内の `sparse_c2s` host処理

`trellis-server` はmutexで生成を直列化しているため、同一server内の同時生成競合は抑えられています（[trellis-server.cpp:393](/Users/<redacted-user>/Downloads/pixal3d.cpp-feat-browser-partial-e2e-integrate/src/trellis-server.cpp:393)）。ただし、profile requestがサーバを長時間占有する運用影響は別途明記すべきです。

#### 5. テスト計画

現計画の良い点は、host/GPU数値比較、既存fixture、E2E統合確認を含むことです。一方、以下が不足しています。

- side-pass前後の内部出力比較
- profiler単体の入力復元テスト
- profiler対象graphの一部nodeだけを再実行した場合の破壊検出
- 時間計測の反復と順序入替え
- residualの自動上限
- 複数視点の成果物比較
- server/WASM build
- `--profile` と `--profile` 無しの同一commit比較

E2E 1本は統合確認には使えますが、性能差の定量には使えません。

#### 6. 代替案

Instrumentsを完全に却下する必要はありません。壁時計ラップと競合せず、次の補助証拠として有用です。

- `unpermute` / bilinearのCPUサンプル比率
- page fault・memory pressure
- printfやallocation/freeの影響
- host側が支配的という壁時計結果の反証確認

最も安全な代替案は、production E2Eには常時の粗いexclusive lapだけを入れ、op再実行は専用テストバイナリの単一graph診断に限定する構成です。これならproduction経路の数値・熱状態・後続永続バッファを守りつつ、issue #33のボトルネック特定を満たせます。

今回は設計と現ソースの静的照合のみで、ビルド・実測は実施していません。


---

# 再確認（2 周目）

- 判定: **差し戻し**
- A1: **解消** — flow side-pass と conditioning 計測が別フラグ・別実行になっています。
- A2: **解消** — 本番 chunked 経路での op 再実行を廃止し、診断専用グラフに限定しています。
- A3: **解消** — 再アップロード callback、周回数、副作用契約、測定結果返却、既存集計の維持が定義されています。
- A4: **解消** — L0/L1 と L2 を別プロセスで実施します。
- A5: **解消** — flow を含む隙間なしの排他的ラップと `accounted` 照合が定義されています。
- A6: **部分的** — ABBA・別プロセス・反復は追加されましたが、監視対象が `trellis` プロセスだけで、WindowServerや他のGPU利用プロセスによる共有負荷を検出・記録できません。
- A7: **部分的** — conditioning の bit 一致とflow再実行結果は追加されましたが、GLBの成分数・finite・boundary/non-manifold・PBRおよび複数視点レンダ比較がありません。
- A8: **解消** — 30%という発火基準と、該当なしの場合のacceptance上の扱いが固定されています。
- B1: **解消** — WASMの構成説明が訂正され、build/link確認が追加されています。
- B2: **解消** — host materialization、コピー、ページ圧という表現に修正されています。
- B3: **解消** — 行単位のflushと重複印字防止が明記されています。
- B4: **部分的** — Metal・CPU・WASM・CUDA/Vulkanは扱われていますが、WebGPUビルドとWASM full/partial両targetの確認が明示されていません。
- B5: **解消** — 時間値の単純diffではなく、接頭辞集合の機械検査になっています。
- B6: **解消** — 排他的ラップのfresh中央値による上限と、実測可能な実現見込みを分離して順位付けしています。
- 承認条件: GPU共有監視を`trellis`以外にも拡張し、元のA7で要求したGLB構造検査と5視点レンダ比較、WebGPUおよびWASM full/partialのbuild/link確認をテスト計画へ追加すること。
