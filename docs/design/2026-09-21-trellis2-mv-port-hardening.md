# trellis2-mv: PR #71 への検証済み修正・テストハーネスの移植（TASK-PORT）

日付: 2026-09-21 / 対象ブランチ: `feat/trellis2-mv-hardening`（`origin/feat/trellis2-mv` = PR #71 の
f04b33c から分岐）/ ステータス: 設計中（実装前レビュー待ち）/
親: `docs/briefs/trellis2-mv/{COMMON,TASK-PORT}.md`

## 背景

trellis2-mv エピックは2トラックが並走し、ユーザー決定で **PR #71 のブランチ（40コミット、#59〜#67 + CI）
を土台に採用**した。もう一方の旧トラック（複数 worktree に分散、`~/Downloads/pixal3d-mv-{59,60,61,62}`）が
持つ「反証検証を通した修正・fixture・テストハーネス」を PR #71 のコードへ移植するのが本タスク。
**旧トラックの実装コードそのもの（#60 cond bank、#61 サンプラ、#62 SS ステージの本体実装）は捨てる**。
残すのは修正・fixture・テストである。旧トラックの worktree は読むだけで変更・push しない。

## P0（着手前に統括から追加指示、完了・別コミット済み）

PR #71 の CI `linux-cpu` job FAILURE の原因: `src/trellis_cli_main.cpp` の入力必須チェックが
`p.trellis2_mv` を見ておらず、`--trellis2-mv DIR` 単独指定でも「no input image」で exit 1 していた。
`&& p.trellis2_mv.empty()` を追加し、`tests/trellis2_mv_cli_contract.sh build/trellis-cli` で
`TRELLIS2_MV_CLI_CONTRACT_OK`（7/7 ok）を確認済み。commit `0b341fb`。

## 現状の事実確認（このリポジトリの HEAD、f04b33c + P0 で直接確認したもの）

| 項目 | 統括のブリーフ記載 | 実測（このセッションでコード直読） |
|---|---|---|
| float32 t-schedule 境界バグ | 残存2箇所（`sample_flow` :398、MVサンプラ :501） | **一致**。両方とも `float t = 1.0f - (float)i/steps` のfloat32演算（`flow_runner.cpp:391-393`単一版、`:498-502` MV版）。ヘルパ関数への切り出し無し |
| OOD クランプ `no_fix` ガード | `:456` 既存はガードあり、`:535` MV経路はガード無しでハードコード | **食い違い**。現HEADでは `:533-536` の MV 側クランプも `if (!no_fix) { ... }` で既にガード済み（`git log -p -- src/flow_runner.cpp` で確認: `7aa4c31 feat(trellis2-mv): implement stochastic and multidiffusion fusion` の時点から一貫してガード付き）。ブリーフの表はクランプ行だけを見て囲みの `if` を見落としたと判断する。**P2はコード変更不要**、PARTIAL報告のみで訂正する |
| counter 境界（PR#104継承） | 実装済み | 確認省略（このタスクの対象外、変更しない） |

## P1: 共有ヘルパの抽出とdouble化

### 目的
旧トラック c4775c8（`fix(flow): compute the t-schedule and guidance-interval decision in double`）を
PR #71 のコードへ意味移植する。単純 cherry-pick は #71 のMVサンプラが別実装のためコンフリクトする前提。

### 変更内容（`include/flow_runner.h` / `src/flow_runner.cpp`）
1. `SamplerParams::gi0/gi1/rescale_t` を `float` → `double` に変更。`sigma_min`/`guidance_strength`/
   `guidance_rescale` は `float` のまま（旧トラックと同じ、変更範囲を必要最小限にする）。
2. 新規公開関数（ヘッダに追加、`namespace trellis` スコープ）:
   - `std::vector<double> flow_t_schedule(int steps, double rescale_t)` — numpy
     `linspace(1,0,steps+1)` + rescale を double で逐語再現。最終要素は計算せず `0.0` を直接代入
     （linspace の endpoint forcing と一致させる）。
   - `inline bool flow_in_guidance_interval(double t, double gi0, double gi1)`
3. `apply_guidance_rescale(sample, pos, pred, t, sigma_min, guidance_rescale, no_fix)` を
   `src/flow_runner.cpp` 内の static ヘルパとして抽出し、**既存 `sample_flow`（FlowFwdProjオーバーロード）
   と #71 の `sample_flow_multi` の両方**がこれを呼ぶようにする。今 `sample_flow_multi` 内にラムダで
   重複定義されている `apply_rescale` を削除し、このヘルパに置き換える。
   - 唯一の意図的な挙動差: 現在の MV版ラムダには `Nst < 2` の早期returnがあるが、単一版には無い
     （単一版は実運用でNst>=2以外あり得ないため実害なし）。共有ヘルパにこのガードを残す。
     単一画像パスへの**唯一の意図的変更点**としてここに明記する（ゼロ除算防止のno-op、実latentでは発火しない）。
4. `sample_flow`（FlowFwdProj版）と `sample_flow_multi` のt-schedule計算・guidance区間判定を
   `flow_t_schedule()`/`flow_in_guidance_interval()` 呼び出しに置き換える。t/tprevのfloat化・
   `1000*t`のfloat化は判定の**後**（narrowing point は参照と同じ位置）。
5. `sample_flow`（FlowFwd版、`FlowFwdProj`ラッパー）は変更不要（内部で上のFlowFwdProj版を呼ぶだけ）。
6. `CMakeLists.txt`: `set_source_files_properties(src/flow_runner.cpp PROPERTIES COMPILE_OPTIONS
   "-ffp-contract=off")` を `if(NOT MSVC)` でガードして追加（`trellis_core` ターゲット定義の直後）。

### f サフィックス一掃（gi0/gi1/rescale_t 代入箇所）
grep で確認済みの全ファイル・全箇所を対象に `0.6f`→`0.6` 等へ変更（`sp.gi0 = 0.6;` のように
`f` サフィックスを外す。doubleへ暗黙変換可能なfloatリテラルではなく、double リテラルを直接書く）:
- `src/flow_runner.cpp`（sample_flow_multiのts配列、guided判定）は上記3-4で対応済み
- `src/pixal3d_full_e2e_wasm.cpp`（2箇所）
- `src/pixal3d_real_geometry_wasm.cpp`（2箇所）
- `src/pixal3d_texture_wasm.cpp`（2箇所 + `tex_sampler_params.npy` オーバーライドをdrift警告のみに変更）
- `src/pixal3d_wasm.cpp`（2箇所）
- `src/test_pixal3d_slat_sample.cpp`（2箇所 + `sampler_params.npy` オーバーライドをdrift警告のみに変更
  + `--probe-steps` を `flow_t_schedule()`/`flow_in_guidance_interval()` 呼び出しに置き換え）
- `src/test_pixal3d_ss_sample.cpp`（1箇所）
- `src/test_slat_shape.cpp`（1箇所）
- `src/test_ss_full.cpp`（1箇所）
- `src/test_ss_sample.cpp`（1箇所）
- `src/trellis_cli.cpp`（このリポジトリでの実測: **6箇所**、`grep -n "gi0\|gi1\|rescale_t"` で列挙済み。
  旧トラックの c4775c8 は同ファイル4箇所のみ変更していたが、#71 はMVサンプラ用に2箇所追加している
  ため6箇所すべてを対象にする）
- `src/test_trellis2_mv_sampler.cpp`（#71独自、3箇所: `rescale_t = 3.0f; gi0 = 0.35f; gi1 = 0.9f;`）

`test_trellis2_mv_sampler.cpp` は `gi0=0.35, gi1=0.9` という #71 独自パラメータを使っており、
float→double化でどのstepの `inside` 判定が変わるか自明ではない。**P1適用前後でこのバイナリの
出力を並べてPARTIALに貼る**（期待値がPython float64オラクル由来か、実装の自己整合か不明なため、
値が変わったら「バグを直した」ではなく「境界の丸めが変わった」事実として報告する）。

### 検証（P1単体）
- 既存 `trellis-test-trellis2-mv-sampler` のbefore/afterを比較（-j4ビルド、CPU、スレッド既定）。
- `-ffp-contract=off` 込みで `trellis-cli`/主要testターゲットがビルド通ることを確認。
- 真の検出器（`shape_slat_sampler_production` の `inside==9` assert）はP3で移植されるため、
  P1のPARTIALでは「flow_t_schedule(12, 3.0)を印字してts[8]==0.6ちょうど（inside, 9/3）になること」を
  10行程度の確認コードで示す（P3を待たずに一次確認）。

## P2: MV経路クランプのno_fixガード

現状確認の通り**既に満たされている**（`flow_runner.cpp:533-536`）。P1でヘルパを共有化すれば
「MV経路だけガードが外れる」という状態は構造的に発生しなくなる。**コード変更なし、コミットなし**。
`PARTIAL: PORT P2` で「brief記載と現状の食い違い」「P1のヘルパ共有化で構造的に保証される」ことのみ報告する。

## P3: `trellis-test-flow-multi` の移植

### 対象
旧トラック `~/Downloads/pixal3d-mv-61/src/test_flow_multi.cpp`（699行、13関数、74チェック）を
`src/test_flow_multi.cpp` として#71ツリーへ移植。fixture 2点をコピー:
- `tests/fixtures/trellis2_mv_sampler/sampler_oracle.txt`（4373 bytes）
- `tools/gen_trellis2_mv_sampler_fixture.py`（11923 bytes、生成スクリプト。実行はしない、参照用）

### API差分と移植方針（実際に読んで確認した差分のみ）
この test は `flow_runner.h` の公開API（`sample_flow`/`sample_flow_multi`/`flow_t_schedule`/
`flow_in_guidance_interval`）だけに依存し、`Trellis2MvCondBank` 等の#60系型には依存しない
（`make_cond_buf`/生のfloatバッファで手組みしている）。したがって**P1の完了だけで大半はそのまま
コンパイルが通る**はずで、修正が要るのは1点のみ:

- 旧: `sample_flow_multi(fwd, noise, conds, neg, sp, mode, StochasticCounter* stochastic_counter = nullptr, trace = nullptr)`
  — 7番目が `StochasticCounter*`、8番目が `trace`。
- 新（#71）: `sample_flow_multi(fwd, sample, conds, neg_cond, sp, mode, std::vector<std::vector<float>>* trace = nullptr, int* stochastic_counter = nullptr)`
  — 7番目が `trace`、8番目が `int*`。

`StochasticCounter` 型（`{ std::size_t next_view = 0; }`）は#71に存在せず、#71は生の `int*` を
使う。**旧トラックの型を持ち込まず、#71のint*に合わせて書き換える**（実装コードは捨てる方針に従う）。
具体的な機械的変更:
- `using trellis::StochasticCounter;` の using宣言を削除。
- `StochasticCounter ctr;` → `int ctr = 0;` に変更する箇所: `test_stochastic_g0_counter_advances`、
  `test_cascade_counter_continuation`、`run_fixture_case`のcascade分岐（計3箇所）。
- `&ctr` を渡す呼び出しを7番目位置→8番目位置に移動し、7番目に `nullptr`（trace不要）を明示:
  `sample_flow_multi(fwd, noise, conds, neg.data(), sp, mode, &ctr)` →
  `sample_flow_multi(fwd, noise, conds, neg.data(), sp, mode, nullptr, &ctr)`。
- それ以外の全呼び出し（`&ctr`を渡さないもの）はデフォルト引数のみなので**無変更でコンパイル可能**。

### 検証観点（旧テストの構成をそのまま維持）
- 構造的自己完結チェック（呼び出し回数・順序・NaN安全網・入力検証・スレッド決定性）
- `STAGES[]` の `expect_inside/expect_outside`（SS 10/2、shape 9/3、tex 6/6）は **P1のdouble化が
  無ければ shape=8/4・tex=5/7 になる既知の境界バグ**の回帰ガード。P1が正しく効いていることの
  直接証拠になる。
- 独立fixture（`sampler_oracle.txt`）との突き合わせ（5 CASE + 3 TSEQ、tol=1e-4 と bit-exact 併用）
- `test_tseq_bitexact`: `flow_t_schedule()` の出力を独立生成のdouble列とbit-exact比較 + 
  `shape_slat_sampler_production` の `inside==9` 明示assert（P1単体PARTIALで先行確認した内容の
  正式な回帰テスト化）

### CMakeLists.txt
```
add_executable(trellis-test-flow-multi src/test_flow_multi.cpp)
target_link_libraries(trellis-test-flow-multi PRIVATE trellis_core)
set_target_properties(trellis-test-flow-multi PROPERTIES BUILD_RPATH "${GGML_RPATH}")
```
（`trellis-test-trellis2-mv-sampler` の直後に追加。既存ターゲットは削除しない、併存。）

**重み・GPU不要、CPU実行、数秒**（旧トラック実績通り）。COMMON.mdの「1分超テストは10分間隔」制約の
対象外と判断する（fixture置換テストと合わせても軽量）。

## P4: `trellis-test-trellis2-mv-ss` の移植（実fixture SS parity）

### スコープ判定（実装コードを読んで確定）
旧トラック `include/trellis2_mv_ss.h`（125行）は6個の関数/型を宣言しているが、
`src/test_trellis2_mv_ss.cpp`（572行、`main()`まで含めて全読了）が**実際に呼ぶのは4個だけ**:
`trellis2_mv_ss_transpose_for_decode` / `trellis2_mv_sample_ss_flow` / `trellis2_mv_decode_ss` /
`trellis2_mv_sample_ss_core`。5個目の `trellis2_mv_sample_ss(models_dir, ..., Trellis2MvCondBank, ...)`
（models_dir を取る本番エントリポイント、`Trellis2MvCondBank` 型に依存）はこのテストファイルの
どこからも呼ばれていない（ヘッダのコメントで「build verification only」と書かれているだけで、
実際のビルド検証コードも main() には無い）。

**したがって `Trellis2MvCondBank` 型への依存を一切持ち込まずに済む。** #60系の型・cond bank実装は
本当に捨てられる。移植するのは次の4関数+型+定数のみ、新規ファイル:

`include/trellis2_mv_ss.h`（新規、#71ツリー）:
```cpp
#pragma once
#include <array>
#include <vector>
#include "flow_runner.h"

namespace trellis {
struct Model;
constexpr int kSsInCh = 8, kSsRes = 16, kSsL = kSsRes*kSsRes*kSsRes, kSsLatentSize = kSsInCh*kSsL;

struct Trellis2MvSsResult {
    std::vector<float> latent;
    std::vector<std::array<int,3>> coords;
};

std::vector<float> trellis2_mv_ss_transpose_for_decode(const std::vector<float>& latent);

std::vector<float> trellis2_mv_sample_ss_flow(const FlowFwd& fwd,
                                              const std::vector<const float*>& conds,
                                              const float* neg_cond, MultiCondMode mode,
                                              const SamplerParams& sp, std::vector<float> noise,
                                              std::vector<std::vector<float>>* trace = nullptr);

Trellis2MvSsResult trellis2_mv_decode_ss(const Model& ss_dec_model, const std::vector<float>& zdec);

Trellis2MvSsResult trellis2_mv_sample_ss_core(const FlowFwd& fwd, const Model& ss_dec_model,
                                              const std::vector<const float*>& conds,
                                              const float* neg_cond, MultiCondMode mode,
                                              const SamplerParams& sp, std::vector<float> noise,
                                              std::vector<std::vector<float>>* trace = nullptr);
} // namespace trellis
```
`src/trellis2_mv_ss.cpp`: 上記4関数の実装。`trellis2_mv_sample_ss_flow` は内部で `int counter = 0;`
を用意し `sample_flow_multi(fwd, noise, conds, neg_cond, sp, mode, trace, &counter)` を呼ぶ
（SSは他ステージとcounterを共有しない、旧ヘッダのコメント通り。counterは`trellis2_mv_sample_ss_flow`
のローカル変数に閉じる=毎回view0から開始）。`trellis2_mv_ss_transpose_for_decode`/`trellis2_mv_decode_ss`/
`trellis2_mv_sample_ss_core` は旧実装をそのまま(呼び出しAPIの型だけ#71に合わせて)移植。
**`trellis_cli.cpp` は一切変更しない**（新規ファイル追加のみなので、production の振る舞いに触れない。
「#71の実装の振る舞いを変える修正はP1/P2以外行わない」を厳守する形）。

### npy.h への追加
`ss_coords.npy`（torch SparseTensor coords、int32 `[N,4]`）を読むため、`npy::ArrayI32` 構造体と
`npy::load_i32()` を `include/npy.h` に追加する（#71には無い。`save_i32` は既にあるので対称な
読み込み関数を足すだけの**純粋な追加**、既存コードへの影響なし）。旧トラックの実装をそのまま移植。

### `src/test_trellis2_mv_ss.cpp` の移植
旧テスト（572行）をほぼそのまま移植。P1完了後は `flow_t_schedule`/`flow_in_guidance_interval` が
公開済みなのでそのまま使える。`SamplerParams::gi0/gi1` はP1でdouble化済みなので
`ss_production_params()` の `sp.gi0 = 0.6; sp.gi1 = 1.0;` はそのまま通る。

二段構えのゲートをそのまま維持:
1. **parity gate**（`--internal-parity-subprocess` で自己再exec、`TRELLIS_NOFIX=1`を強制）:
   v2 fixtureの6 run全てで rel ~1e-6・coords完全一致を主張。
2. **production regression**（クランプON、既定実行）: 既知の乖離（min_ratio<0.2、max_rel帯、
   sym_diff帯）を「情報表示」でなく`check()`でassertする（旧テストの2026-09-21マージレビュー修正を
   引き継ぐ）。

### 実行時の留意（COMMON.md制約）
- v2 fixture `~/nfs/pixal3d_trellis2mv_ref_v2/` は実在確認済み（6 run ディレクトリ全て存在）。
  **リポジトリへコピーしない**（770MB、ディスク11GiB制約）。テストのデフォルト引数として
  `fixture_root` を絶対パスで参照する既存の旧テストの流儀をそのまま使う。
- `ss_dec.gguf` を要する（重み使用）。1分超の可能性があるため実行は1回、10分間隔ルールに従う。
- 本番実装が旧トラックと異なる挙動をしてもこのタスクでは**直さない**。乖離が出たら
  `FINDING: <run/step/乖離量/file:line>` で報告するに留める（禁止事項に明記済み）。

## P5: list_images の2..8下限

### 結論（fork調査+このセッションでのコード確認、両方で一致）
**#71にこの特定バグは存在しない。** #71は旧トラックの専用関数 `trellis2_mv_list_images()` 自体を
持たず、汎用 `trellis::list_view_images(dir, &err)`（下限チェック無し、列挙のみ）+ CLI側
(`trellis_cli.cpp:768-780`付近)の `if (names.size() < 2 || names.size() > 8)` という構成。
1枚ディレクトリは正しく `--trellis2-mv requires 2..8 images; found 1` でreject される。

### 作業内容（テスト新規作成のみ、プロダクションコード変更なし）
旧トラック `~/Downloads/pixal3d-mv-60` の `git show e92311a` が追加した `src/test_trellis2_mv_cond.cpp`
相当の回帰テスト観点（natural sort・ゼロ埋めタイ・非画像/隠しファイル除外・2..8境界両側・決定性）を、
**`trellis::list_view_images()` + CLI相当の `< 2 || > 8` チェック**という#71の実際の構成に対して
新規に書く。加えて `tests/trellis2_mv_cli_contract.sh`（P0で確認済み、既にnatural order・1枚拒否・
2..8境界を一部カバー）で重複していない観点（ゼロ埋めタイ、隠しファイル除外、決定性の複数回実行）だけを
補うテストにする。

## 全体の依存順序とコミット計画

```
P0 (完了, commit 0b341fb)
  -> P1 (flow_runner.cpp/.h 共有ヘルパ + double化 + f-suffix sweep)      [1 commit]
  -> P2 (コード変更なし、PARTIAL報告のみ)
  -> P3 (test_flow_multi.cpp + fixture 2点、CMakeLists 追加)             [1 commit]
  -> P4 (trellis2_mv_ss.h/.cpp 新規 + npy.h ArrayI32/load_i32 + test)    [1 commit]
  -> P5 (list_images 回帰テスト新規)                                     [1 commit]
```
P3・P4はP1完了に依存（float64 t-schedule公開関数を使うため）。P4はP3と独立(fixtureも別)だが、
同じ`flow_runner.h`変更に依存するため順序はP1が先であれば良い。ブリーフ指定の順序(P1→P2→P3→P4→P5)
をそのまま守る。

## codex exec設計レビュー反映（2026-09-21、advisor判断で確定）

`codex exec`（read-onlyサンドボックス、実コード読了ベース）でレビューを受けた。判定は「LGTMではない」。
指摘の大半（P1のnarrowing順序明示、fサフィックスgrepの精度、P3の入力検証差分、P5のテスト配置）は
そのまま採用する。P4のみ、codexの二択（テスト専用化 or trellis_cli.cpp抽出リファクタ）のどちらでもなく
第三案（advisor判断）を採る。2周目のcodex execは回さない — 変更は「codexの指摘を採る」か
「理由付きで採らない」のいずれかで、新規の設計判断は次の1点（P4のstatic化）のみのため。

### P1: 受け入れ基準をコード形で凍結
narrowing順序を曖昧な散文でなくコードで固定する（両サンプラで同一）:
```cpp
const double t_d = ts[i], tprev_d = ts[i + 1];
const bool inside = flow_in_guidance_interval(t_d, sp.gi0, sp.gi1);
const float t = (float)t_d;
const float dt = (float)(t_d - tprev_d);        // 1回のdouble減算をnarrowする（float引き算後にnarrowしない）
const float tscaled = (float)(1000.0 * t_d);    // 1000倍はdoubleで行ってからnarrowする
```
`float tprev`という中間変数は作らない（作ると`t - tprev`と書きたくなり、narrowing順序が壊れる）。
`flow_t_schedule(int steps, double rescale_t)` の先頭に `if (steps <= 0) throw std::invalid_argument(...)`
を追加する（新規公開関数の契約として。既存呼び出し箇所は全てsteps=12等の正の定数のため実害なし）。

fサフィックス箇所は `rg -n '\b(gi0|gi1|rescale_t)\s*=\s*[^;]*[0-9](\.[0-9]*)?[fF]\b' --glob '*.{h,hpp,cc,cpp,cxx}'`
で適用前に全件列挙し、EVIDENCEとして貼る（適用後に同コマンドが実質0件になることを確認して貼るのがP1完了の証跡）。
適用前の実測件数（このセッションで実行済み）: `pixal3d_wasm.cpp`×2、`test_pixal3d_ss_sample.cpp`×1、
`pixal3d_real_geometry_wasm.cpp`×3、`test_slat_shape.cpp`×1、`test_trellis2_mv_sampler.cpp`×3、
`pixal3d_full_e2e_wasm.cpp`×3、`pixal3d_texture_wasm.cpp`×3(default分、npy override分は別扱い)、
`test_ss_full.cpp`×1、`trellis_cli.cpp`×6、`test_ss_sample.cpp`×1、`test_pixal3d_slat_sample.cpp`×2
(npy override分は別扱い)、`include/flow_runner.h`×3(構造体定義そのもの)。

NPY override（`tex_sampler_params.npy`/`sampler_params.npy`からのgi0/gi1/rescale_t読み込み）は
**旧トラックc4775c8と同じくwarn-onlyを維持する**（fail-closedへ強化するというcodexの提案は採らない）。
理由: (1) この挙動は旧トラック自身のレビュー済み判断であり、今回移植する対象そのもの。
(2) P3/P4のparity/production gateはこのNPY override経路を一切通らない（fixture読み込みは別の
`npy::load`経路）ため、fail-closed化してもP4の検証強度には寄与しない。

### P2: 変更なし（現状確認のみ、codexもLGTM）

### P3: 74/74ではなく73実行(うち72 PASS・1 FAIL)+1 SKIPが期待値
codexが実コードで確認した通り、#71の`sample_flow_multi`は`std::invalid_argument`を投げる
（`std::runtime_error`ではない）ため、旧テストの`catch (const std::runtime_error&)`は捕捉できず
プロセスがterminateする。移植時は`catch (const std::exception&)`に変更する（期待値=「例外が飛ぶこと」
は変えていないので「勝手に期待値を変える」禁止事項には抵触しない、#71の例外型に合わせるAPI適合）。

2件は安全に実行できない/実行しても意味が異なる:
- **「null cond element throws」**: #71はconds要素のnullを検証しない。nullを渡すとFakeModelの
  `cond[0]`参照でSIGSEGVし、プロセスが落ちて残り全チェックが失われる。`check()`を呼ばず
  `printf("SKIP ...")`にして1件減らす（74→73実行）。FINDING: `flow_runner.cpp:492-494`は
  `conds`の各要素nullを検証しない（`std::invalid_argument`は空リストのみ対象）。
- **「forward return-size mismatch throws」**（stochasticモード）: #71のstochastic分岐
  （`:550-567`）はmultidiffusion分岐（`:574`）と違いforward戻り値サイズを検証しない。
  ただしこのテストのfake modelはNst+1（大きい方）を返すためOOB読みは起きず、安全に実行できる。
  実行すると例外が飛ばず`check()`が自然にFAILする。これは直さずFAILのまま残し、FINDINGで報告する:
  `flow_runner.cpp:550-567`（stochastic分岐）は`:574`と同じサイズ検証を持たない
  （fwdが**小さい**ベクタを返した場合はOOB読みのリスクがある、というのが実害）。

期待される最終結果: **73 checks実行、72 PASS、1 FAIL（既知・FINDING報告）、1 SKIP（安全のため未実行・FINDING報告）**。

`StochasticCounter`(旧)→`int*`(新)の意味的差分（旧は`[0,V)`に正規化、新は無制限に増加）は
コード上は等価に動作する（`(*counter)++ % conds.size()`で剰余を取るのは呼び出し側）ため、
移植した74チェックの範囲では検出できない設計上の契約差。FINDING 1行のみ報告し、テスト対象外とする。

### P4: 新規ヘッダを作らず、テストファイル内staticとして完結させる（設計変更）
codexの指摘（4関数が#71の本番SS実装(`trellis_cli.cpp:906-920`)と並行する未使用の複製になり、
テストしても本番経路を検証しない）は妥当。ただしcodexが提示した二択のどちらも取らない:
- 「テスト専用ヘルパに限定」: 本番コードとの対応関係が薄れる
- 「trellis_cli.cpppを抽出リファクタして共有」: 本番ファイルへの侵襲が大きく、`sample_bank`ラムダの
  `t2mv`分岐（単一画像[3/6]ブロックにも影響する）を巻き込むため、「#71の実装の振る舞いを変える
  修正はP1/P2以外行わない」の精神に反するリスクが高い

**第三案**: `include/trellis2_mv_ss.h`/`src/trellis2_mv_ss.cpp`という新規production APIは作らず、
4関数(`trellis2_mv_ss_transpose_for_decode`/`trellis2_mv_sample_ss_flow`/`trellis2_mv_decode_ss`/
`trellis2_mv_sample_ss_core`)を**`src/test_trellis2_mv_ss.cpp`内のstatic関数**として降格して移植する。
これにより:
- 本番APIとして誤認される未使用コードを一切追加しない（`trellis_cli.cpp`は無変更のまま）
- `sample_flow_multi`/`ss_decode`/`ss_coords`という#71の共有関数は直接呼ぶため、本番SS経路の
  「サンプラ本体+デコーダ」部分は実際にテストが踏む。テストされないのは
  「transpose 3行」（`trellis_cli.cpp:914`と同じ添字式`zdec[c*4096+sp]=z[c+8*sp]`をコメントで
  行番号ごと参照して書く、`test_transpose_pure`のindex-patternテストが式の同一性を守る）と
  「CLI配線・実DitRunner・実ss_flow.gguf」（このマシンには実際`ss_flow.gguf`が存在しないため
  いずれにせよ実行不可 — `~/nfs/weights/pixal3d/gguf-q8_0/`確認済み、`pixal3d_ss_flow_mv.gguf`
  はあるが`ss_flow.gguf`という名前のファイルは無い）
- この限界を**FINDINGとして明記**: 「本テストはreplay FlowFwd（記録済み予測値の再生）でCFG/rescale/
  Euler積分とデコードだけを検証しており、実DiT forward・実cond bank構築・CLI引数配線は検証していない。
  実ss_flow.gguf不在のためこのマシンでは検証不可能（旧トラックの`test_trellis2_mv_ss.cpp`冒頭コメントと
  同じ制約）」

codex追加指摘のうち採用するもの:
- 6 runいずれかのディレクトリ・必須ファイルがmissingならFAILにする（旧テストはSKIP+g_fail不変のため
  「6 run全部無くてもALL PASS」になり得た。これは移植時に修正する — 旧の**バグ**ではなく
  旧テストが想定していなかった「fixtureが全く無い環境」での誤ったPASSを防ぐテスト自体の頑健化であり、
  #71の実装コードには触れないので禁止事項の対象外）
- 実行済みrun数が厳密に6であることをassert
- 各runで12 CFGファイルすべて(`cfg_files_checked==steps`)を確認したことをassert（既に旧テストの
  `strict`分岐にある`cfg_files_checked == steps`条件を、production regression側にも同様に追加）
- `result.coords.size() == coords_mine.size()`（set化で重複座標が隠れるのを防ぐ）

codex追加指摘のうち採用しないもの（理由付き）:
- **manifest.jsonのSHA/model revision照合**: v2 fixtureのmanifest形式を未確認。今回のスコープでは
  「テスト移植」であり新規の照合ロジック設計は含まない。FINDINGに「manifest照合は未実施」と明記する。
- **5面投影並置での目視確認**: SS coordsは32³格子の活性ボクセル集合であり、レンダリング可能な
  メッシュではない（decimate/UV展開前の中間表現）。exact-set比較が本タスクの検証水準として妥当
  （旧トラックの設計判断を踏襲）。FINDINGに「視覚的並置は未実施、exact-set比較のみ」と明記する。
- **閾値への失敗注入**: `min_ratio`/`max_rel`/`sym_diff`の帯は「既知の乖離のpin」（検出器ではない）
  と旧テスト自身がコメントで明記している。pinの妥当性を注入テストで確認するのは本タスク（移植）の
  範囲を超える。FINDINGに「これらの帯はpinであり失敗モード検出器としての検証はしていない」と明記する。

### P5: codexの分割案を採用
`< 2 || > 8`の下限上限チェックをテスト内に複製しない（本番CLIの回帰を検出できないため）。
- 2枚/8枚/9枚の境界ケースは`tests/trellis2_mv_cli_contract.sh`に3ケース追加し、実際に
  `build/trellis-cli --trellis2-mv`を叩いて確認する（P0で確認した既存7チェックに追加、10/10がEVIDENCE）。
- natural sort・ゼロ埋めタイ・隠しファイル/非画像除外・決定性は`trellis::list_view_images()`の
  単体テストとして`src/test_list_view_images.cpp`（新規、既存test命名規約に合わせる）を作成する。

## 実装順序（codexレビュー反映後）
```
P1 (flow_runner.cpp/.h 共有ヘルパ + double化 + f-suffix sweep)      [1 commit]
  -> P3 (test_flow_multi.cpp、73実行/72PASS/1FAIL/1SKIP)            [1 commit]
  -> P5 (list_images単体テスト + CLIコントラクト3ケース追加、独立)   [1 commit]
  -> P4 (test_trellis2_mv_ss.cpp、4関数はstatic降格、重み使用)       [1 commit]
```
P2はコード変更なし（PARTIALのみ）。P4は重み実行を伴うため最後に配置し、10分間隔ルールの対象は
このセッションで1回のみ（parity subprocessの自己re-execを含めても1回のテストバイナリ実行として扱う）。

## リスクと軽減

- **P1がSamplerParams::gi0/gi1/rescale_tの型を変える**: 全呼び出し箇所を上記の一覧で洗い出し済み
  （grepで機械的に確認、漏れがあればビルドエラーで検出できる — float→doubleの暗黙変換は許容される
  方向なので、`f`サフィックス除去を忘れてもコンパイルは通ってしまう点に注意。**ビルド後に
  再度grep -n "gi0 = .*f;\|gi1 = .*f;\|rescale_t = .*f;"で0件になることを確認**してから完了とする）。
- **P4で新規ファイルを追加するがtrellis_cli.cppは触らない**ため、本番挙動への影響はゼロ。
- **ディスク11GiB制約**: v2 fixtureはコピーしない。ビルドは対象ターゲットのみ（`-j4`）。
- **Metal GPU共有**: P4のss_dec.gguf実行はCPU(`gpu=-1`)既定で行う（旧テストのデフォルト引数通り）。
