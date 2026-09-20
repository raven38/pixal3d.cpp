### 判定: 差し戻し

並列 collapse の排他性と native CPU 上の atomic 可視性は妥当です。ただし、Emscripten 分岐に実際の競合経路があり、入力検査の公開契約も満たしていません。

### 重大（A）

1. Emscripten でも環境変数により並列実行でき、非 atomic `prop` が競合する  
   [src/decimate_qem.cpp:78](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:78)、[src/decimate_qem.cpp:260](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:260)、[parallel_for.h:25](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/include/parallel_for.h:25)

   `parallel_threads()` は Emscripten で 1 を返しますが、`decim_threads()` は `TRELLIS_DBG_DECIM_THREADS=2` などを優先します。`parallel_for()` も明示された `nt` を強制的に 1 へ戻しません。

   再現シナリオ:

   - Emscripten build
   - `TRELLIS_DBG_DECIM_THREADS=2`
   - `E >= 4096`
   - pthread 無しなら `std::thread` 生成で例外・停止
   - pthread 有りなら `prop_min()` が通常の read-modify-writeなので、同じ面へ伝播する複数 edge 間でデータ競合し、lost update・非決定結果になる

   修正は、少なくとも `decim_threads()` または `parallel_for()` で `__EMSCRIPTEN__` 時の明示 `nt` も必ず 1 に固定することです。

2. 「入口で不正面を除去する」という公開契約を pass-through が迂回する  
   [uv_bake.h:55](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/include/uv_bake.h:55)、[src/decimate_qem.cpp:329](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:329)、[src/decimate_qem.cpp:352](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:352)

   具体例:

   ```cpp
   V0 = 3;
   F0 = 1;
   target_faces = 1;
   faces = {0, 0, 99};
   ```

   `F0 <= target_faces` により検査前に返り、縮退面と範囲外 index がそのまま出力されます。後段が有効メッシュを前提にすると範囲外アクセスにつながります。

   `decimate_qem()` 側と `decimate_qem_cpu()` 側の双方に早期 return があるため、検査を共通化してから pass-through 判定する必要があります。現在の異常系テストは必ず簡略化を走らせる設定なので、この経路を検出しません。

### 軽微（B）

- [src/decimate_qem.cpp:254](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:254) の「cost は非負有限」は浮動小数点上では保証されません。QEM 展開式は大座標で cancellation により負値になり得ます。負の IEEE-754 bit pattern を unsigned 比較すると数値順になりません。legacy と同じ挙動ではありますが、コメントの前提を弱めるか、cost の有限性・非負性を明示的に正規化すべきです。

- [src/test_decimate_cpu.cpp:111](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/test_decimate_cpu.cpp:111) の  
  `F <= max(input F, target)` は、通常 `target < input F` なので実質「面数が増えていない」しか検査しません。目標面数到達を要求するケースでは `F <= target` が必要です。

- `local_edges` の集合・多重度は最終メッシュの bit 一致を介してしか検査されていません。ランダムな有効三角形列、重複面、3面以上共有辺、境界辺について、`unordered_map` の `(edge,count)` と直接一致させる単体テストが望まれます。

- 非多様体 fixture は `E < 4096` なので直列カットオフに入り、非多様体入力での並列 collapse を検証していません。非多様体 fan を複製して `E >= 4096` にするケースが必要です。

- スレッド決定論は各スレッド数につき1回だけです。断続的競合を拾うには複数回反復し、失敗した子プロセスの出力は比較対象から除外して、比較チェック自体も失敗させる方が明瞭です。

- 「45ラウンドを通して bit 一致」は最終出力の `memcmp` だけでは証明できません。一度発散して再収束する可能性を排除するなら、各ラウンドの `V/F/verts/faces` hash を比較する必要があります。

- `off`、`eoff`、`E` は `int` です。[src/decimate_qem.cpp:148](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:148) 以降で `3F` や edge 数が `INT_MAX` を超える入力は符号付き overflow になります。対応しないなら最大入力サイズを契約として検査してください。

- [src/decimate_qem.cpp:265](/Users/<redacted-user>/Downloads/pixal3d-qem-decimate/src/decimate_qem.cpp:265) の `is_always_lock_free` は正しさには不要で、mutex 実装の atomic が有効なプラットフォームまでコンパイル不能にします。性能上の必須条件なら、対象プラットフォーム限定であることを明示すべきです。

- テスト executable は作られていますが、CMake の `add_test()` 登録が見当たりません。標準の CTest/CI が自動で実行する保証がありません。

### 確認できた正しさ（何をどう確かめたか）

- collapse の native CPU 並列書き込みは互いに素です。2本の勝者が頂点または面を共有すると、その共有面の `prop` が異なる2つの packed key と同時に一致する必要があります。edge ID が異なるため不可能です。非多様体辺でも全 incident face が `own` 判定対象なので証明は維持されます。

- `local_edges` は、有効で3頂点が相異なる各面について、最小頂点から2辺、中間頂点から1辺、最大頂点から0辺を列挙します。したがって各無向辺は小さい端点で面ごとにちょうど1回現れ、重複面・境界・非多様体でも `unordered_map` 版と集合・多重度が一致します。

- native CPU の relaxed CAS は min-reduce として正しいです。CAS失敗時には `cur` が観測値へ更新され、spurious failure もループで再試行されます。propagate の全 worker を `join()` 後、collapse の worker を生成するため、スレッド完了・join・次の thread start を通じた happens-before があり、collapse の relaxed load から確定値を読めます。

- float の累積順は、CSR を面順に直列構築しているため、各頂点QEMと各edgeの skinny 累積について legacy と同じです。edge ID も `(a,b)` 順なので、有効入力では参照実装との bit 一致という論理は成立します。

- `n < 4096` の直列カットオフは、各要素内の演算順を変えず、propagate だけは同じ整数 min-reductionなので、native CPU の結果を変えません。`remesh_dc` の該当ループも出力要素ごとの独立書き込みです。

- `F=0,target=0` は早期 return で安全です。仮に `simplify_round` が `F=E=0` で呼ばれても、atomic 配列を最低1要素確保し、アクセスループは空です。

- sandbox 外で `./build-metal/trellis-test-decimate-cpu 120` を実行し、closed/open torus、非多様体、重複面、高次数、異常面、スレッド数 1/2/3/8/16 の全既存チェックが `PASS` することを確認しました。TSAN と Emscripten build は今回再実行していません。
