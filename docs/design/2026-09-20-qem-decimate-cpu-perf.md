# CPU QEM decimation の高速化（issue #29）— エッジ抽出のソート化と段の並列化

日付: 2026-09-20 / ブランチ: `perf/qem-decimate`（origin/main d54de75 起点、#28 の RoPE ブランチとは独立）/
ステータス: 実装済み（codex レビュー `docs/reviews/2026-09-20_qem-decimate-cpu-perf_review.md` = 差し戻し →
末尾「レビュー対応」で改訂。計測結果は `docs/results/2026-09-20-qem-decimate-cpu-perf.md`）

## 概要

macOS / Metal の MV 1024 生成で postprocess 177 s のうち `decimate_qem`（18.4M 面 → 1M 面）が 118 s を占める
（issue #28 の E2E プロファイル）。Mac には CUDA / Vulkan の decimate カーネルが無いので、常に
`src/decimate_qem.cpp` の CPU 経路が走る。この経路は**完全にシングルスレッド**で、毎ラウンド
`std::unordered_map` に 3F 本の半エッジを挿入してエッジを数え直している。

本設計は**アルゴリズム（quadric・コスト式・独立集合の選び方・閾値ラダー・λ）を一切変えず**、
(1) エッジ抽出を CSR ベース（頂点ごとの局所ソート）に置き換え、(2) 頂点独立・エッジ独立な段を
`std::thread` で並列化する。出力は現行実装と同一アルゴリズムの結果であり、差はエッジ ID の付番順
（現行は `unordered_map` の走査順 = もともと任意）によるタイ処理のみ。

目標: cyclops MV 1024 / seed 1 の同一 fixture で `decimate_qem` **≤ 25 s を受け入れ基準**（現行 128.7 s、
issue の目標は −60〜90 s）、15 s を stretch goal とする（レビュー A6 を受けて分離。実測は 7.2 s、§レビュー対応）。

## 背景（実測、2026-09-20、M4 Max 16 コア、アイドル状態、origin/main ビルド）

段別タイマー（`TRELLIS_DBG_DECIM=1`、本ブランチで追加済み。挙動は変えない）で現行実装を測った:

**実メッシュ**（`trellis-cli --views docker/linux-webgpu-gate/e2e/views -m gguf-q8_0 --res 1024 --seed 1 --dump-post`
の fixture、`~/nfs/pixal3d/fixtures/cyclops_mv1024_seed1_post.bin`、V 9,183,186 / F 18,372,632）→ 1M 面:

| 段 | 合計秒 | 割合 | 内容 |
|---|---:|---:|---|
| cost | 61.7 | 52% | エッジごとの QEM 評価 + 反転判定 + skinny 累積（`process`） |
| edges | 29.1 | 25% | `unordered_map` による一意エッジ抽出 + 境界頂点判定 |
| prop | 18.3 | 16% | (cost,id) を両端点の隣接面へ min-reduce する scatter |
| collapse | 4.2 | 4% | 勝者エッジの collapse（頂点移動・面の付け替え・dead マーク） |
| qem | 2.5 | 2% | 頂点 quadric の累積 |
| adj / compact | 0.8 / 0.8 | 1% | CSR 隣接 / 頂点・面の圧縮 |
| ラウンド計 | 117.5 | | 45 ラウンド（thresh 1e-8 → 1e-7） |
| **`decimate_qem` 全体** | **128.7 s** | | 入力コピー + 最終圧縮を含む。出力 V 478,776 / F 961,394（baseline log と一致） |

同規模の手続き torus（V 9.0M / F 18.0M、53 ラウンド）でも比率は同じ（cost 46% / edges 29% / prop 15%）。

## 提案

### 1. `simplify_round` の書き換え（`src/decimate_qem.cpp`）

段ごとに次のように置き換える。データの意味・順序依存の計算順は現行と同一に保つ。

| 段 | 現行 | 新 | 決定論 |
|---|---|---|---|
| adj（v→f CSR） | 直列 counting sort | **そのまま**（直列。面順で決まる v2f の順序は qem / skinny の float 累積順に効くので変えない） | 同一 |
| edges | `unordered_map` に 3F 挿入 → 走査 | **頂点ごとの局所抽出**（`local_edges`）: 頂点 a の隣接面の他 2 頂点のうち `> a` のものをスレッドローカルの `std::vector` に集め（重複あり、多重度 = そのエッジを含む面数）、n ≤ 32 は挿入ソート・それ以上は `std::sort` → RLE で一意エッジと面数を得る。2 パス（頂点ごとの一意数を数える → prefix sum → 書き込み）で `edges[]` を **(a, b) 昇順の CSR**（`eoff[a]..eoff[a+1]`）に並べる。面数 == 1 のエッジを `eopen[t]` に記録し、後段の直列走査で両端点を boundary にする。面数 ≥ 3（非多様体辺）は現行どおり boundary にしない。**前提: 各面は相異なる有効な 3 頂点**（入口で保証、§レビュー対応 A1） | エッジ ID が **(a,b) 昇順**に固定される（CUDA 版 `cub_sort_keys` + RLE と同じ順序） |
| qem | 直列 | 頂点で並列（各頂点は自分の v2f を CSR 順に累積 → 現行と同じ float 順） | 同一 |
| cost | 直列 | エッジで並列（`cost[t]`, `vnew[t]` を各自書く。読むのは verts / faces / qem / boundary のみ） | 同一 |
| prop | 直列 scatter | **`std::atomic<uint64_t>` の CAS min** による並列 scatter（CUDA `k_propagate` の `atomicMin` と同じ）。比較は `pack_cost` の符号なし 64 bit キー（cost の float ビット列 << 32 ｜ id。cost は非負有限なので数値順と一致し、同コストは id の小さい方が勝つ。INF/NaN は collapse の `cost <= thresh` で落ちる）。`load(relaxed)` → `p < cur` の間 `compare_exchange_weak(cur, p, relaxed, relaxed)`。値の min-reduce だけで、読み手は `join()` 後なので relaxed で足りる。`static_assert(is_always_lock_free)`。Emscripten（直列）では atomic 型を使わず素の `uint64_t` 配列に min を書く | 同一 |
| collapse | 直列 | エッジで並列。勝者エッジ t は e0 / e1 の全隣接面を「所有」し、2 つの勝者が面または頂点を共有することは無い（共有頂点は全隣接面の共有を意味し、両方が同じ面の min にはなれない）ので、`verts[e0]`・`vdead[e1]`・`fdead`・`faces` への書き込みは互いに素 | 同一（collapse の適用順に依存しない） |
| compact | 直列 | **そのまま**（直列 0.8 s。後で必要なら 2 パス化） | 同一 |

エッジ ID の付番が変わるので、**同じ float コストを持つ 2 本のエッジが同じ面を取り合う**場合だけ勝者が
現行と変わり得る（`pack_cost` は下位 32 bit に id を入れて min を取るため）。実メッシュ（DC 出力）では
コストが bit 一致することは実質無く、A/B で確認する（§テスト計画）。手続き的な torus では規則構造の
ためタイが起き得るので、そこでは V/F・bbox・開放辺・成分数の一致で見る。

### 2. 並列化の道具

`src/remesh_dc.cpp` にあった `parallel_for`（`std::thread` で均等分割、Emscripten は直列）を
`include/parallel_for.h` に移し、`remesh_dc.cpp` と `decimate_qem.cpp` の両方がそれを使う（同名別実装の
ドリフトを避けるため置き換えた。remesh 側の差分は関数本体の削除と `remesh_threads()` の委譲だけ）。
`parallel_for(n, fn, nt)` はスレッド数を引数に取り、decimate は `TRELLIS_DBG_DECIM_THREADS=N`
（テスト用の固定。結果はスレッド数に依らない）が無ければ `hardware_concurrency()`（M4 Max で 16）。
毎段スレッドを起こす固定費は 45 ラウンド × 6 段 × 16 本 ≈ 4,300 回 ≈ 0.2 s で無視できる。
trellis_core は既に `remesh_dc.cpp` で `std::thread` を使っているので、リンク条件（Linux の pthread）は
新たに増えない。

### 3. 変えないもの

- 閾値ラダー（1e-8 開始、除去率 < 1% で ×10、stall 2 回で ×10、1e12 で打ち切り）、`lam_len = 1e-2`、
  `lam_skinny = 1e-3`、boundary 重み、反転判定、skinny の式。
- `decimate_qem()` の入出力・シグネチャ、GPU / Vulkan への dispatch（`TRELLIS_HAVE_GPU_DECIMATE` /
  `TRELLIS_HAVE_VK_DECIMATE` はそのまま先に試す）。
- 後段の `weld_vertices` → `fill_small_holes` → `drop_small_components(0.03)` と呼び出し側
  （`trellis_cli.cpp` 2 箇所、`post_replay.cpp`）。

## 実装計画

1. `include/parallel_for.h`（`trellis::parallel_for(int64_t n, fn(b,e), nt)` と `trellis::parallel_threads()`）。
   `decimate_qem()` を「GPU/Vulkan dispatch → `decimate_qem_cpu()`」に分け、`decimate_qem_cpu` を
   `uv_bake.h` で公開する（テストが GPU ビルドでも CPU 経路 = fallback を直接検証できるように）。
2. `src/test_decimate_bench.cpp`（本ブランチで追加済み）に **legacy 参照実装**を同梱する: 変更前の
   `simplify_round` をそのまま写し、エッジ抽出のあとに `std::sort(edges)` を 1 行足したもの
   （`--legacy` で選択）。エッジ順が同じなら新実装とは全段で float 計算順が一致するので、
   **出力は bit 一致**するはず。これを新実装の正しさの機械判定に使う。
3. `simplify_round` を上表どおりに書き換える（段別タイマーは維持）。
4. `trellis-test-decimate`（CMake 登録、CUDA 不要）: torus R=300（180k 面）と、境界付きメッシュ
   （torus の 1 列を抜いて開いたもの）で legacy(sorted) vs 新が bit 一致、かつ有限・開放辺数・成分数が
   入力と整合することを assert。数十秒で終わる。
5. 実 fixture で A/B（§テスト計画）→ `docs/results/2026-09-20-qem-decimate-cpu-perf.md` に記録 → PR。

## 影響範囲

| ファイル | 変更 |
|---|---|
| `src/decimate_qem.cpp` | `simplify_round` の書き換え、`decimate_qem_cpu` の分離と入口検査（ラダー・λ・dispatch 順は不変） |
| `include/uv_bake.h` | `decimate_qem_cpu` の宣言 |
| `include/parallel_for.h` | 新規（`remesh_dc.cpp` から移設） |
| `src/remesh_dc.cpp` | ローカルの `parallel_for` を削除しヘッダに委譲 |
| `src/test_decimate_cpu.cpp`, `src/test_decimate_qem_legacy.inc` | 新規: 回帰テストと変更前の参照実装（テスト専用） |
| `src/test_decimate_bench.cpp` | 新規: ベンチ / A-B（`--legacy`、`compare` は不一致で exit 3） |
| `src/post_replay.cpp` | `--decimated MESH.bin`（間引き済みメッシュから bake 以降だけ再生。GLB A/B 用） |
| `src/test_decimate.cpp`（既存、CUDA 手動ビルド） | 触らない |
| `CMakeLists.txt` | `trellis-test-decimate-cpu` / `trellis-test-decimate-bench` の登録 |
| `docs/design/`, `docs/reviews/`, `docs/results/` | 本設計・レビュー・計測結果 |

- 呼び出し側（不変）: `src/trellis_cli.cpp`（MV と cascade の 2 箇所）、`src/post_replay.cpp`、
  `src/pixal3d_postprocess.cpp`（WASM/browser の postprocess）。
- CUDA / HIP / Vulkan ビルド: GPU 経路が先に試され、**失敗時（デバイス無し・alloc/kernel エラー・
  64-bit atomics 非対応）は本 CPU 経路へ fallback する**ので、これらのビルドでも新コードはコンパイル・
  リンクされ実行され得る。`trellis-test-decimate-cpu` は `decimate_qem_cpu` を直接呼ぶので、どのビルド
  でも fallback 経路を検証できる。`decimate_qem_vk.cpp` はホスト側で同じ `unordered_map` エッジ抽出を
  持つ（29 s 相当）。本 issue では触らず、Vulkan-only ビルド向けの追い作業として issue に残す。
- WASM（Emscripten）: `decimate_qem.cpp` は `web/ss` の postprocess にも入る。`#ifdef __EMSCRIPTEN__` で
  スレッド数 1（直列）かつ prop を非 atomic にする。`em++ -c`（pthread 無し）でのコンパイルを確認済み。
- メモリ: 実測ピーク RSS（§レビュー対応 A7）。unordered_map（27.6M ノード）が消える分だけ減る。
- 決定論: 新実装は入力に対して決定論的（並列の interleaving は min-reduce と互いに素な書き込みにしか
  現れない）。現行実装との差はタイ処理のみ。

## テスト計画

| # | テスト | 判定 |
|---|---|---|
| T1 | `trellis-test-decimate-cpu`: torus R=300（閉）、1 列抜いた torus（開）、目標以下、極小目標、目標 0、三角形 1 枚、F=0、非多様体辺（3 面共有）、高次数 fan（200 面、`std::sort` フォールバック）、重複面 + 孤立頂点、縮退面 `(0,0,1)`/`(0,0,0)` + 範囲外 index、スレッド数 1/2/3/8/16 | legacy(sorted) と **bit 一致**（`memcmp`、縮退入力は「事前に除去した入力」との一致）。有限・有効 index、閉メッシュで open_edges == 0、成分数保存、スレッド数間で bit 一致。加えて同テストを `-fsanitize=thread` で実行 |
| T2 | 実 fixture（cyclops、18.37M 面 → 1M）: legacy(sorted) vs 新 | bit 一致（一致しなければ差分の最初のラウンドを特定して原因を潰す） |
| T3 | 実 fixture: 現行（hash 順）vs 新 | V/F・bbox・開放辺・非多様体辺・成分数を並置。タイ処理の差なので V/F は数個以内、bbox は 1e-4 以内を期待。乖離が大きければ T2 の前提（タイの希少性）を疑う |
| T4 | 壁時計（アイドル、他セッションの `build-metal/trellis` が走っていないことを `ps` で確認してから） | 新実装 `decimate_qem` を 3 回計測して中央値。受け入れ ≤ 25 s（現行 128.7 s）、stretch 15 s。`/usr/bin/time -l` でピーク RSS も並置 |
| T5 | 同じ dump から現行 / 新の間引きメッシュ（bench の out.bin）を作り、`post-replay <dump> <out.glb> --no-weld --no-fill --no-remesh --faces 1000000 --atlas 4096 --decimated <mesh.bin>` で同じ bake を通して GLB にする | `tools/glb_metrics.py` の幾何・UV・材質指標、両方向のサンプル表面距離（Hausdorff 相当）、固定カメラの複数視点レンダ並置と画像差分。目視は補助 |
| T6 | E2E（`trellis-cli` 同入力・同 seed、アイドル）| `done in` と decimate 区間の差分。出力 GLB の V/F を baseline log（Vo=639713 Fo=950170）と並置 |

計測はすべて「別セッションの Metal 負荷が無いこと」を確認して行う（同じ GPU/CPU を共有しているため、
重なったら汚染と明記して再計測）。

## 代替案

| 案 | 内容 | 却下理由 |
|---|---|---|
| B. vertex clustering で 4〜5M 面へ前減量 → QEM | 既存 `decimate_cluster` を前段に置く | 格子スナップで形状・トポロジが変わり（issue の "visible topology regressions" に直撃）、QEM 部分だけでも ~30 s 残る。本案（A）が目標未達の場合の追加策として温存 |
| C1. Metal compute への限定移植（`decimate_qem_vk.cpp` 相当、ホストでエッジ抽出） | 4 カーネルを Metal で | ホスト側のエッジ抽出（29 s + map 破棄 11 s）が残る |
| C2. Metal 全 GPU 化（CUDA 版同様に sort/RLE まで GPU） | 全段を Metal で | 新規コード量が最大（Metal API のホストコード + 8 カーネル）。CPU 案で 7 s まで下がったので投資対効果が無い |
| D. ラウンド数の削減（独立集合を緩める / 閾値ラダーの初期値変更） | 収束を速める | 出力が変わる（参照 CuMesh からの逸脱）。issue の「品質劣化なし」に反する |
| E. `unordered_map` の差し替えだけ（robin-hood 等） | 最小差分 | edges 29 s しか対象にならず、cost 62 s / prop 18 s が残る |

## レビュー対応（codex `docs/reviews/2026-09-20_qem-decimate-cpu-perf_review.md`、判定: 差し戻し）

| # | 指摘 | 対応 |
|---|---|---|
| A1 | 頂点局所抽出は縮退面の自己ループ辺 `(a,a)` を再現しない | 入力契約を「各面は相異なる有効な 3 頂点」とし、`decimate_qem_cpu` の入口で範囲外 index / 縮退面を落とす（各ラウンド末の圧縮と同じ規則）。変更前は `(a,a)` が勝つと a とその全隣接面を消していた（穴が開く）ので再現しない。テスト `degenerate-input` で「事前に除去した入力」と bit 一致を確認。**契約の非対称**: 除去は間引きを実行するときだけで、`F <= target` の pass-through は入力をそのまま返す（変更前と同じ）。除去したら `decimate_qem: dropped N invalid/degenerate input faces` を stdout に出す（既存の `decimate_qem(...)` 行と同じ流儀） |
| A2 | 挿入ソートと作業領域に上限が無い | スレッドローカル `std::vector`（再利用）、n ≤ 32 は挿入ソート・超過は `std::sort`。実メッシュの次数は p50 6 / p99 10 / max 14（候補 ≤ 28）。高次数 fan（200 面）のテストでフォールバックを通す |
| A3 | atomic の契約・lock-free・WASM | `std::atomic<uint64_t>` 配列を明示的に確保、`static_assert(is_always_lock_free)`、CAS ループは relaxed（値の min-reduce のみ、読み手は join 後）。Emscripten は `#ifdef` で素の配列 + 通常 min。`em++ -c`（pthread 無し）でコンパイル確認 |
| A4 | CUDA/HIP/Vulkan は fallback で CPU 経路を使う | 影響範囲に明記。`decimate_qem_cpu` を公開し、テストがどのビルドでも CPU 経路を直接検証できるようにした。`pixal3d_postprocess.cpp` の呼び出しも列挙 |
| A5 | 異常系・スレッド数・機械判定 | T1 に 12 ケース + スレッド数 1/2/3/8/16（`TRELLIS_DBG_DECIM_THREADS`、子プロセス）+ TSAN を追加。`compare` は不一致で exit 3 |
| A6 | 15 s は実測から導ける下限と矛盾 | 受け入れ ≤ 25 s、stretch 15 s に分離。実測 7.22 s（レビューの下限見積り 18.2 s は「ラウンド外 11.2 s」を固定費と置いたが、その正体は `unordered_map` の破棄で新実装では消える） |
| A7 | ピーク RSS | `/usr/bin/time -l`: 変更前（参照）3.90 GB → 新 3.26 GB（bench プロセス全体、fixture 読み込み込み） |
| B | `parallel_for` の重複 | `remesh_dc.cpp` のローカル実装を削除しヘッダに統一 |
| B | packed key の順序前提 | 表に明記（非負有限 cost、INF/NaN は閾値判定で除外） |
| B | T5 のコマンド | 結果ドキュメントに完全なコマンドと fixture の出自を記録 |
| B | 3F 半エッジ flat sort / face-parallel gather の代替 | 局所抽出の oracle は変更前の `unordered_map` 実装そのもの（bit 一致で検証済み）。prop は atomic 版で 18.3 → 1.1 s なので gather 版は不要と判断 |
| B | Metal 全 GPU 化 | 代替案 C を「ホスト抽出を残す限定案」と「全 GPU 化」に分けて再記述。今回は CPU 案で目標を大きく超えたので見送り |

## 実装レビュー対応（codex `docs/reviews/2026-09-20_qem-decimate-cpu-perf_impl_review.md`、判定: 差し戻し → 修正済み）

| # | 指摘 | 対応 |
|---|---|---|
| A1 | Emscripten でも `TRELLIS_DBG_DECIM_THREADS` で並列化でき、非 atomic の prop が競合する | `decim_threads()` は `__EMSCRIPTEN__` で env を無視して 1 を返し、`parallel_for` も `__EMSCRIPTEN__` では引数の nt を 1 に固定 |
| A2 | `F <= target` の pass-through が入口検査を迂回し、不正面をそのまま返す | 検査を `sanitize_faces()` に共通化し、`decimate_qem` / `decimate_qem_cpu` の両方で **pass-through 判定より前**に実行（pass-through は「位置そのまま + 検査済みの面」）。GPU / Vulkan 経路にも検査済みの面を渡す。テスト `below-target-invalid` で両入口を確認 |
| B | 「cost は非負有限」は丸めで破れる | コメントを「負になった cost は符号ビットでキーが大きくなり負けるだけ（変更前と同じ）」に修正 |
| B | `F <= max(F, target)` は緩い | `F > target` なら `F <= target`、そうでなければ入力と同一、に戻した |
| B | 非多様体入力の並列 collapse（E ≥ 4096）が未検証 | `nonmanifold-parallel`（torus R=60 の 1/8 を複製、非多様体辺 ≥ 1000、E ≈ 12k）を追加 |
| B | スレッド決定論が各 1 回 | 各スレッド数 3 回（計 15 子プロセス）、失敗した子の出力は比較を失敗させる |
| B | `int` 添字の上限 | `sanitize_faces` で `3F > INT32_MAX` と面バッファ不足を弾き、メッシュを無変更で返す |
| B | `is_always_lock_free` の static_assert は不要 | 削除（mutex 実装の atomic でも正しい。主要ターゲットでは lock-free） |
| B | `local_edges` の直接検証 / ラウンドごとの hash 比較 / CTest 登録 | 見送り: 集合・多重度はレビュー自身が導出で確認しており、最終出力の bit 一致（478k 頂点の float 位置 = collapse 履歴そのもの）が実質的な機械判定。CTest はこの repo の慣習（`trellis-test-*` は CTest 未登録）に合わせる |
