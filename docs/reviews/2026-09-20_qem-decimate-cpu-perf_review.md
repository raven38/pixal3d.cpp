### レビュー結果サマリー

- 判定: **差し戻し**
- 重大な問題（A）: 7件
- 軽微な指摘（B）: 6件

設計の中心である「頂点・面を共有しない勝者 collapse」の論理は、通常の三角形入力について成立します。一方、エッジ抽出の同値条件、WASM、GPU/Vulkan fallback、テストの機械判定、メモリ、性能下限が設計に落ちていません。このまま実装すると「実メッシュでは速く見えるが、異常系・別ビルドで壊れる」可能性があります。

重大な問題は次です。

- A1: 頂点局所エッジ抽出は、縮退面の自己ループ辺を再現せず、現行実装と同じ集合・面数になりません。
- A2: 挿入ソートと「スタック上」の作業領域に上限がなく、高次数頂点で時間・スタックが破綻します。
- A3: `std::atomic<uint64_t>` の実装契約、lock-free 性、WASM 非 pthread ビルドでの成立性が未確定です。
- A4: CUDA/HIP/Vulkan は「影響なし」ではありません。失敗時に変更後の CPU 経路へ fallback します。
- A5: テスト計画が非多様体・縮退面・孤立頂点・空入力・スレッド数差をカバーせず、既存比較ツールも不一致時に exit 0 です。
- A6: 目標下限 15秒は、提示された実測値から導かれる理想下限と矛盾します。
- A7: WASM を含むピークメモリ予算と失敗時挙動がありません。

承認条件:

1. 入力契約を「全ての面は3つの相異なる有効頂点」と明示して入口で検査するか、縮退面を含め現行3半エッジ抽出と完全同値にする。
2. 局所ソートの高次数フォールバックとメモリ上限を決める。
3. atomic CAS の擬似コード、memory order、lock-free 検査、WASM の非 atomic 直列分岐を設計へ追加する。
4. CPU/Metal、CUDA、HIP、Vulkan、WASM のビルド・fallback テストを追加する。
5. 異常系、反復決定論、複数スレッド数、TSANを含むテストを機械的な失敗判定にする。
6. 25秒を hard gate、15秒を stretch goal とするなど、性能目標を実測値と整合させる。
7. ピークRSSを計測し、特に4 GiB上限のWASMについて許容入力サイズまたは graceful failure を定める。

### 詳細フィードバック

#### 1. 設計の完全性

**[A] エッジ抽出の入力前提が未定義です。**

現行実装は各面から常に `(a,b)`, `(b,c)`, `(c,a)` の3本を数えます。[src/decimate_qem.cpp:93](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:93)

例えば面 `(0,0,1)` は次になります。

| 実装 | 辺と面数 |
|---|---|
| 現行 | `(0,0):1`, `(0,1):2` |
| 提案の `other > a` | `(0,1):2`、`(0,0)` は消失 |

つまり、通常三角形では同値でも縮退面では同値ではありません。`(0,0,0)` なら現行は自己ループ辺1本、新方式は `E=0` です。boundary 判定、stall、最終出力まで変わり得ます。

対処はどちらかです。

- `decimate_qem()` の契約として有効index・相異なる3頂点を検査し、不正面を事前除去する。
- 各面の3半エッジを直接列挙する方式にして自己ループを含め完全同値にする。

**[A] 「スタック上で挿入ソート」が実装可能な仕様になっていません。**

頂点次数を `d` とすると候補数は最大 `2d`、挿入ソートは最悪 `O(d²)` です。collapse 後には局所次数が増える可能性があります。動的長の標準C++スタック配列もありません。

少なくとも以下を決める必要があります。

- 例: 64件までは固定配列、超過時はスレッドローカル `std::vector` と `std::sort`。
- 最大観測次数、p50/p95/p99/max と、高次数メッシュでの時間。
- 2パスで同じソートを2回行うコスト。
- 作業領域確保失敗時の扱い。

**[B] `parallel_for` の共通化方針が中途半端です。**

設計は共通ヘッダを新設する一方、[src/remesh_dc.cpp:45](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/remesh_dc.cpp:45) の既存実装は置換しないとしています。同名で挙動の異なる実装が残るため、将来ドリフトします。今回置換しないなら、名称を `decimate_parallel_for` に限定する方が正確です。

#### 2. 技術的実現可能性

**collapse の排他性: 通常入力では成立します。**

現行の所有条件は、エッジ `t` の両端に接続する全ての面について `prop[f] == pack(cost[t], t)` を要求します。[src/decimate_qem.cpp:186](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:186)

2本の異なる勝者エッジ `x`,`y` が頂点 `v` を共有すると仮定すると、`v` に接続する全ての面で同時に

```text
prop[f] = pack(cost[x], x)
prop[f] = pack(cost[y], y)
```

が必要です。しかし下位32 bitのIDが異なるので packed 値は一致せず、矛盾します。面を共有する場合も同じです。

したがって、勝者間で以下は互いに素です。

- `verts[e0]`
- `vdead[e1]`
- 端点に接続する `faces`
- `fdead`

非多様体辺でも、全隣接面が `v2f` に入る限りこの証明は崩れません。

**[B] 設計書には証明の前提を明記してください。**

必要な前提は、有効なCSR、エッジIDの一意性、collapse 開始前に全prop計算が完了していること、collapse中に隣接情報を更新しないことです。

**legacy(sorted) との bit 一致: 条件付きで妥当です。**

QEMは頂点ごと、skinnyはエッジごとのローカル累積です。[src/decimate_qem.cpp:111](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:111) [src/decimate_qem.cpp:129](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:129)

以下を維持すれば、並列化自体はfloat加算順を変えません。

- `off/v2f` の内容と順序が同一。
- 各頂点内のQEM加算順が同一。
- 各エッジで `process(e0)` → `process(e1)` の順序が同一。
- 同じコンパイラ、最適化、FP contraction/FENV。
- エッジ集合、面数、昇順IDが完全一致。

問題は最初のA1により、縮退面ではエッジ集合が一致しないことです。

**[A] bit一致テストの前に、各ラウンドで次を直接比較してください。**

- `edges`
- 辺ごとの出現数
- `boundary`
- `off/v2f`
- `cost` のビット列
- `prop`
- 勝者ID集合
- compact前後のV/F

最終出力の `memcmp` だけでは、途中の相違が偶然消えた場合を検出できません。

**[A] atomic CAS min の正しさは実装仕様が必要です。**

正しい基本形は次です。

```cpp
uint64_t old = slot.load(std::memory_order_relaxed);
while (candidate < old &&
       !slot.compare_exchange_weak(
           old, candidate,
           std::memory_order_relaxed,
           std::memory_order_relaxed)) {
}
```

全scatter完了後にworkerを `join()` してからcollapseへ進めるなら、値のmin-reduceには relaxed で十分です。

ただし以下を固定してください。

- `prop` は本物の `std::atomic<uint64_t>` 配列にする。通常の`uint64_t`をキャストしてatomicアクセスしない。
- `sizeof(std::atomic<uint64_t>)` と `is_always_lock_free` を確認する。
- 非lock-free環境では直列propへfallbackする。
- WASM直列時はatomic型を使わず、現行の通常minをコンパイルする。
- CAS試行回数・成功回数を段別計測する。

CUDAも同じpacked keyを `atomicMin` しています。[src/decimate_qem.cu:190](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cu:190) ただしCPU上で同程度の高速化が得られる根拠にはなりません。高次数頂点周辺では同じcache lineへの競合が集中します。

#### 3. リスク・懸念

**[A] 15秒という目標下限は実測値と両立しません。**

提示値では段合計117.5秒、関数全体128.7秒なので、段外の残差が11.2秒あります。並列対象はおよそ

```text
qem 2.5 + cost 61.7 + prop 18.3 + collapse 4.2 = 86.7秒
```

理想的な16倍でも5.4秒です。さらに直列adj/compactが1.6秒あります。

```text
11.2 + 1.6 + 86.7 / 16 = 18.2秒
```

これは新エッジ抽出を0秒、atomic競合・帯域制約・Eコア差を0とした非現実的な下限です。したがって15秒は計算上ほぼ不可能です。`≤25秒`を受け入れ基準、15秒をstretch goalに分離すべきです。

**[A] メモリ評価がピークRSSを表していません。**

`eoff + eopen` は約

```text
(V+1)*4 + E*1
= 9.18M*4 + 27.6M
≈ 64.3 MB
```

で、`prop` のatomic化は `sizeof(atomic<uint64_t>) == 8` なら既存147 MBの置換です。「+130 MB未満」の内訳が不明です。

一方、同一ラウンドで `v2f` 約220 MB、edges約221 MB、QEM約367 MB、cost約110 MB、vnew約331 MB、prop約147 MBなどが共存し、入力・compact用バッファを含めると概算2 GB級です。unordered_map削除により減る可能性は高いものの、推測ではなくピークRSSを測る必要があります。

WASMは最大4 GiB設定で、decimateを実際に含みます。[web/ss/CMakeLists.txt:21](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/web/ss/CMakeLists.txt:21) [web/ss/CMakeLists.txt:43](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/web/ss/CMakeLists.txt:43)

**[B] packed float順序の前提を明記してください。**

`pack_cost` はfloatのビット列を符号なし整数として比較しています。[src/decimate_qem.cpp:65](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:65) 非負有限値では数値順と一致しますが、負値やNaNでは一般のfloat順ではありません。現行互換として維持するのは妥当ですが、「costのmin」ではなく「packed unsigned keyのmin」と記載し、負値・NaNの計数をデバッグ検査してください。

#### 4. 影響範囲

**[A] CUDA/HIP/Vulkanは影響なし、という記載は誤りです。**

GPU経路が失敗するとCPUへfallbackします。[src/decimate_qem.cpp:237](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:237) [src/decimate_qem.cpp:247](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:247)

したがって変更後CPUコードはCUDA/HIP/Vulkanビルドでも、

- 必ずコンパイル・リンクされる。
- GPU不在、OOM、カーネル不一致、64-bit atomic非対応時に実行される。
- fallback結果と性能を変える。

各バックエンドのビルドと、CPU fallbackを強制するテストが必要です。

**[A] 呼び出し側が1箇所漏れています。**

設計は`trellis_cli.cpp` 2箇所と`post_replay.cpp`だけを列挙していますが、[src/pixal3d_postprocess.cpp:67](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/pixal3d_postprocess.cpp:67) にも呼び出しがあります。

**[A] WASMについて「直列ならlock有無は無関係」は不十分です。**

pthread無しでもatomic型のコンパイル・リンクが成立するとは限りません。64-bit atomic helperやtarget featureを要求する可能性があります。`#ifdef __EMSCRIPTEN__` ではpropそのものを非atomic実装へ切り替え、実際のWASM対象をビルドしてください。

**[B] スレッドリンク条件も明示してください。**

`trellis_core`自体には`Threads::Threads`がリンクされておらず、現在は一部実行ファイルだけです。[CMakeLists.txt:443](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/CMakeLists.txt:443) macOSで通ることだけでなくLinux CPU/CUDA/HIP構成も確認対象です。

#### 5. テスト計画

**[A] 異常系テストを追加してください。**

最低限、次の固定fixtureが必要です。

- 1三角形の開いたメッシュ。
- 1辺を3面で共有する非多様体メッシュ。
- 同じ面を2回持つメッシュ。
- `(0,0,1)` と `(0,0,0)` の縮退面。
- 孤立頂点を含むメッシュ。
- `V>0, F=0`。
- `target == F`、`target > F`、`target == 0`。
- 不正indexはrejectするのかpreconditionとするのか。
- 高次数fan/starメッシュ。
- 全コストがタイになる対称メッシュ。

非多様体辺は、現行と同じく面数3ならboundaryではなくnonmanifoldとして扱うことを直接assertしてください。

**[A] スレッド数を制御して反復決定論を確認してください。**

`1, 2, 3, 8, 16` threadsで各3回以上実行し、全出力がbit一致することを確認します。`hardware_concurrency()`固定ではこの試験ができないため、テスト専用envまたは引数が必要です。加えて小規模fixtureをTSANで実行してください。

**[A] 現在のcompareは不一致でも成功終了します。**

[src/test_decimate_bench.cpp:134](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/test_decimate_bench.cpp:134) は結果を表示しますが、[src/test_decimate_bench.cpp:138](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/test_decimate_bench.cpp:138) で常に`return 0`です。CIの機械判定には使えません。

また、現状のCMakeはベンチしか登録しておらず、CPU回帰テストやCTest登録がありません。[CMakeLists.txt:434](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/CMakeLists.txt:434)

**[A] T3のbbox/V/Fだけでは局所破綻を検出できません。**

bboxが同一でも局所スパイク、穴、自己交差、薄部欠損は起こり得ます。少なくとも以下が必要です。

- 対応不要の双方向surface distanceまたはsampled Hausdorff。
- boundary/nonmanifold/成分数。
- 面積・体積・法線反転率。
- 固定カメラ・固定rendererによるfront/back/left/right/top画像。
- 画像差の機械閾値。目視判定は補助扱い。

**[B] T5のコマンドは再現可能な形になっていません。**

`post-replay`にはdumpと出力GLBの位置引数が必要です。[src/post_replay.cpp:39](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/post_replay.cpp:39) 完全なコマンド、ビルドSHA、fixture SHA-256、出力パス、renderer設定を記録してください。

#### 6. 代替案

**[A] 次の2案を比較対象へ追加してください。**

1. **3F半エッジのflat配列 + global sort/RLE**

   メモリは増えますが、現行の辺集合・自己ループ・多重度を最も直接的に再現できます。局所抽出のoracleとしても有効です。並列radix sortを使える場合は性能候補にもなります。

2. **face-parallel gatherによるprop**

   各面が自分の`prop[f]`を1回だけ書く方式です。vertex-to-edge CSRが追加で必要ですが、CAS競合とlock-free依存を排除できます。atomic版との実測比較が必要です。

**[B] 段階導入も検討すべきです。**

最初にqem/cost/collapseだけを並列化し、edge抽出とpropを現行のままにしてbit一致を確認した後、edge抽出、atomic propを個別に導入すると、相違の原因を段ごとに限定できます。

**[B] Metal案の却下理由は限定的すぎます。**

「ホスト側29秒が残る」のは現行Vulkan型の設計をそのまま移植した場合です。CUDA同様にedge sort/RLEまでGPU化するMetal案なら残りません。今回採用する必要はありませんが、代替案としては「ホスト抽出を残す限定案」と「全GPU化」を分けて評価すべきです。
