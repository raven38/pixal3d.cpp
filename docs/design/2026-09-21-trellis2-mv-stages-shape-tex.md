# trellis2-mv: shape SLat（LR/HR cascade）・texture SLat の実重み段階比較（TASK-STAGES）

日付: 2026-09-21 / 対象ブランチ: `feat/trellis2-mv-stages`（`feat/trellis2-mv-hardening` の
3f0b405 から分岐 = PR #71 + 移植5コミット） / ステータス: 設計中（実装前レビュー待ち） /
親: `/Users/<redacted-user>/Downloads/claude-work/docs/briefs/trellis2-mv/{COMMON,TASK-STAGES}.md`
（リポジトリ外の絶対パス。このリポジトリ内には `docs/briefs/` は存在しない）

## 背景・スコープ

`test_trellis2_mv_ss.cpp`（P4、SS ステージ）が確立した方式 —— #71 の本番ビルディングブロック
（`sample_flow_multi` / 各デコーダ関数）だけを呼ぶ file-local static ヘルパでラップし、
`~/nfs/pixal3d_trellis2mv_ref_v2` の実重み fixture を `ReplayModel` で再生してサンプラ数学だけを
native で検証する —— を **shape SLat（512 単独解像度 / 1024 cascade の LR→HR）と texture SLat**
に広げる。#63/#64 自体（実装）は PR #71 で完了済み（weightless テストのみ）。本タスクは実装を
変更せず、実重みでの段階比較テストを追加する。

## 既存コードの実測（このツリーの HEAD で直読）

- `src/trellis_cli.cpp:894-902` の `sample_bank` ラムダが SS/shape/tex 全ステージ共通で
  `sample_flow_multi`（単画像なら `sample_flow`）を呼ぶ唯一の入口。file-local ヘルパもこれだけを
  呼ぶ（自己比較禁止の原則、P4 と同じ）。
- shape: `shape_flow` ラムダ（:932-945）。`in_ch=out_ch=32`、`SamplerParams{steps=12,
  gs=cfg.gsh(既定7.5f), gr=0.5f, gi=[0.6,1.0], rescale_t=3.0}`。cascade 時のみ
  `shape_stochastic_counter`（:930）を **LR→HR で共有**（Stochastic モードの時だけ渡す、:940-942）。
- cascade 分岐（:954-1000）: (1) LR (`shape_flow_512.gguf`, `coords`=SS coords, `cond_bank`) →
  denorm（`SHAPE_MEAN/STD`、:970-972） → (2) `trellis::shape_upsample`（共有関数、:977） →
  (3) 量子化 backoff ループ（:982-997、**関数化されていない純算術**: grid=hr_res/16 で
  `std::set` 量子化、`hr_res<=1024 || 件数<max_tokens` で確定、それ以外は `-128` ずつ backoff） →
  (4) HR (`shape_flow_1024.gguf`, `shc`, `cond1024_bank`)。非 cascade は `shc=coords` のまま
  `shape_flow_512.gguf` を1回。
- texture: :1077-1095。`in_ch=64`（`[noise(32);shape_slat_norm(32)]` concat、:1082-1090）、
  `out_ch=32`、`SamplerParams{steps=12, gs=1.0f, gr=0.0f, gi=[0.6,0.9], rescale_t=3.0}`。
  `sample_bank` 呼び出し（:1092）に `stochastic_counter` 引数を渡していない → デフォルト
  `nullptr` → `sample_flow_multi`（`flow_runner.cpp:545`）で `seq = i`（ステップ番号そのもの、
  実質「呼び出しごとに新規カウンタ」）。issue #64 の「テクスチャ開始時にカウンタをリセット」を
  そのまま満たす。**concat 前段の 64ch 入力は `ReplayModel` の `x` を無視する契約
  （P4 と同じ）では観測不能・検証対象外** —— `ReplayModel` は `(x,t,cond)` を読まず記録済み
  テンソルを返すだけなので、64ch concat の正しさそのものは本テストの範囲外（それは
  `DitRunner`/実 `tex_flow_*.gguf` を使う E2E テスト（#65/#68）の担当）。
- `include/flow_runner.h:87-106` の `sample_flow_multi` は P4 で既に一次検証済み
  （SS ステージの fixture replay）。本タスクは同じ関数を疎（sparse, N 可変）入力・
  cascade 2段呼びで exercise するだけで、サンプラ数学自体を再実装しない。
- **（レビュー後の修正）`trellis::shape_upsample` は本タスクでは exercise しない。**
  fixture に参照の LR 最終 SLat が無いため（下の「レビュー反映」2番）、HR ステージの
  座標は参照の `shape_slat_coords.npy` をそのまま使う。`shape_upsample` 自体の検証は
  別タスクのギャップとして `NEXT:` に残す。

## fixture のレイアウト確認（実測、`python3` でヘッダ読み）

- 512 解像度 run: `shape_slat_coords.npy` == `ss_coords.npy`（同一 `[N,4]`、実測 963 行の run
  で確認）。shape/tex ステージの疎 coords は SS の出力そのもの。cascade run は最終
  `shape_slat_coords.npy` が HR コード（`shc` 相当）で、LR 単独の coords は
  `noise_shape_cascade_0.npy` の行数（=`ss_coords.npy` と同じ）から復元する。
- 予測テンソル（`shape_step*_*.npy` / `shape_lr_step*_*.npy` / `shape_hr_step*_*.npy` /
  `tex_step*_*.npy`）は numpy shape `(N, 32)`、C order → メモリ上 `index = n*32 + c`
  （c が内側/連続）。native 側の疎入力レイアウトは `DitRunner::forward` のコメント
  （`include/flow_runner.h:48`）どおり `[in_ch*N]` **channel-major**（同じ
  `index = c + in_ch*n = c + 32*n`）。**両者は代数的に同一の式**（`n*32+c == c+32*n`）—— つまり
  SS の dense ケース（torch `[8,16,16,16]` は channel-outer、native は channel-inner で
  transpose が必要だった）と異なり、**疎ケースは transpose 不要、`.npy` を読んだバイト列をそのまま
  `std::vector<float>` として使える**。P4 の `to_native_ss`/`trellis2_mv_ss_transpose_for_decode`
  に相当する変換関数は shape/tex には作らない（作ると「存在しない変換を『適用した』と称する」
  向きの誤りになる — 作らないこと自体を明記する）。
- cascade run の HR 量子化 backoff: 実測 6 run のうち直接確認した run
  （`run_4img_real_multidiffusion_1024c`）の最終 voxel 数は 4171（`shape_slat_coords.npy`
  の shape を `python3` で直接確認）。旧トラック側の記録（このリポジトリ外、
  `~/Downloads/pixal3d-mv-59/docs/spec/33-trellis2-mv-reference.md` §2.2、参考情報として
  引用のみ）では6run中の最大が16516（1img T.png HRケース）とあり、いずれも
  `cfg.max_tokens` 既定49152（`include/trellis_args.h:71`）を大きく下回る。かつ `hr_res<=1024` が常に真
  （既定 `cfg.hr_res=1024`、`include/trellis_args.h:70`、これらの run は `--hr-res` 指定なし）
  なので、backoff ループの break 条件 `(int)q.size() < max_tok || hr_res <= 1024`
  （`trellis_cli.cpp:988`）は**常に1回目（grid=64）で成立する**。それでもループ自体は
  本番と同じ形（1回で終わる特殊ケースとして）を file-local ヘルパにそのまま複製し、
  「1回で抜ける」を実測 assert する（ループが実際に周回する経路は本 fixture では検証できない
  ことを結果ドキュメントに明記する — 事後の言い訳ではなく既知の fixture 限界として先に書く）。

## レビュー反映（codex exec、2026-09-21、着手前レビュー）

`codex exec --sandbox read-only` によるレビューで「要修正」判定。指摘は全て採用し、
下記のとおりスコープを大幅に絞った（実装はまだ着手していない状態でのレビューだったため、
コード変更は発生していない）。

1. **P0（最重要・確定した fixture 上の制約）**: shape/texture の予測テンソルは
   `{step0, step8, step11}` の3点のみ（`ls ~/nfs/pixal3d_trellis2mv_ref_v2/run_*/shape_lr_step*.npy`
   等で直接確認、`shape_step1..7,9,10` は存在しない）。SS は12点全部あるため12ステップ
   Euler積分をそのまま replay できたが、shape/tex は**9ステップ分の速度場が欠落**しており、
   「最終 SLat 一致」「LR→HR 通し replay」を fixture から再構成することは**不可能**
   （逆算も不可能 — 中間 `sample` 状態のダンプも無い）。→ **「最終 shape_slat/tex_slat が一致」
   という検証項目は削除する**（TASK-STAGES.md の記載自体が本 fixture の実際の内容と
   食い違っている。事前決めの受け入れ基準を実装都合で緩めたのではなく、着手前レビューの
   時点で「そもそも実行不可能」と判明したもの。統括への `FINDING:` として報告する）。
2. **さらに自分で追跡した波及（P0 の直接の帰結、レビューでは明示されていないが同じ根）**:
   cascade の LR→HR 座標受け渡し（`shape_upsample`→quantize backoff→`shc`）を検証するには
   **参照の LR 最終 denormalized SLat** が必要だが、fixture には存在しない
   （`shape_lr_step{0,8,11}` は予測テンソルのみ、LR 最終結果のダンプなし）。**したがって
   「HR コード集合が参照の `shape_slat_coords.npy` と一致する」という自己導出の検証も不可能**。
   一方、**HR ステージの初期ノイズ（`noise_shape_cascade_1.npy`）は LR の出力に依存しない
   独立ノイズ**であることを実測確認した（`noise_shape_cascade_1.npy` の行数 4171 が
   `shape_slat_coords.npy` の行数 4171 と一致、かつ LR ノイズ `noise_shape_cascade_0.npy`
   行数948とは無関係）——つまり **HR の座標は参照の `shape_slat_coords.npy` をそのまま使い
   （自己導出しない）、そのコード集合上で HR sampler の step0 CFG 検証は独立して成立する**。
   本タスクは HR コード集合の自己導出検証を落とし、参照コードをそのまま使う方式に変更する。
3. **P0 の帰結としてのスコープ確定**: 検証できるのは
   **各ステージ・各stepの CFG 後出力が、その time step の直前の `sample` 状態を仮定した
   ときに算術的に正しいか**、かつ**その直前 `sample` 状態が独立に既知の場合のみ**。
   - shape の `step0`: `sample`=`noise_shape_{0,cascade_0,cascade_1}.npy`（既知）→
     **検証可能**（LR/HR 両方、production パラメータ gs=7.5/gr=0.5 込み）。
   - shape の `step8`/`step11`: 直前 `sample` は steps 1-7 (または 9-10) の実行結果に
     依存し、それらの速度場が無いので**直前状態を再構成できない → 検証不可能、削除**。
   - texture の `step0/8/11` **全て検証可能**: production パラメータが `gs=1.0, gr=0.0`
     のため、`sample_flow_multi` の CFG 式は `pred = 1.0*pos + 0.0*neg = pos`
     （`gr=0.0` なので `apply_guidance_rescale` は呼ばれない — `flow_runner.cpp:560/584`
     の `if (sp.guidance_rescale > 0.0f)` ガード）——**`sample` 状態に一切依存しない**
     純代数式になる。よって texture は3点全部、`sample` 未知でも `cfg.npy == pos0.npy`
     （stochastic）/ `avg.npy`（multidiffusion, interval内）を直接比較できる。これは
     TASK-STAGES.md 項目4「gs=1.0/gr=0.0でクランプがno-op」の検証そのもの。
4. **P1（テクスチャ 64ch concat 未検証）**: 採用。probe `FlowFwd` を追加し、
   `sample_bank`/`sample_flow_multi` に渡す前に concat ヘルパ（`trellis_cli.cpp:1083-1090` の
   複製、pure data movement）が毎ステップ `x64[0:32]==現在state, x64[32:64]==固定shape_norm`
   を満たすことを構造的に assert する。
5. **P1（coords/feats の行対応）**: 採用。coords は `std::set` でなく `std::vector` の
   完全一致（順序込み）で比較し、別途「重複なし」を `set` サイズとの比較で確認する。
   自己導出した coords は使わない（上記2番）ため、実質「参照の coords 行数と対応する
   noise/prediction/最終feats の行数が一致する」という N の整合性チェックになる。
6. **P2（"共有関数だけ"の主張の限定）**: 採用。ただし下の「追記」節のとおり CFG 本体は
   複製せず本番 `sample_flow_multi` を `sp.steps=1` で直接呼ぶ方式に変更したため、
   denorm・CFG混合・rescale・clamp は複製しない（denorm はそもそも実装しない——本タスクは
   最終値を検証しないため不要）。**quantize backoff ループと texture concat の2つだけが
   実際の複製コード**であり、これらは `trellis_cli.cpp` 自身の配線の検証ではないことを
   明記する（quantize backoff は実 fixture で exercise せず synthetic structural test のみ、
   texture concat は probe `FlowFwd` で構造検証）。`min_ratio`（OOD clamp 診断、shape の
   step0 のみ算出可能）は診断値としてのみ出力し、production gate の pass/fail 条件には
   使わない。
7. **P3（文書参照の誤り）**: 修正済み。`docs/spec/33-trellis2-mv-reference.md` は
   このツリーには存在しない（旧トラック worktree `~/Downloads/pixal3d-mv-59` にのみ存在し、
   このツリーの `docs/results/2026-09-21-trellis2-mv-reference.md` は**別内容**——diff で確認、
   算法プロトコルの記述で fixture の実際の捕捉ステップ集合は書かれていない）。
   以降の記述は `~/nfs/pixal3d_trellis2mv_ref_v2` 配下の実ファイルを直接
   `ls`/`python3` ヘッダ読みで確認した事実のみを根拠とする（旅路ドキュメントへの参照は外す）。
   デコーダ検証（`shape_decode`/`tex_decode`）はスコープ外であることも明記する
   （`shape_upsample` も上記2番の理由で本タスクでは exercise しない — 別タスクの
   ギャップとして `NEXT:` に残す）。
8. **P1（backoff は1回で抜けるだけ）**: 採用。加えて量子化 backoff ループ自体は
   参照 LR feats が無く実行しないため、**ループのロジック単体は synthetic 座標での
   pure structural test（モデル・fixture 不要）としてのみ検証**し、「実 fixture で
   backoff が exercise された」とは主張しない。

## 追記: step0/8/11 CFG 検証の実装方式を「複製算術」から「本番 `sample_flow_multi` 呼び出し」へ変更

レビュー後にさらに検討した結果、上の「レビュー反映」1・3番で「`apply_guidance_rescale` 相当を
ファイルローカルに複製」と書いた方式は、**複製せずに本番 `sample_flow_multi` を直接呼んで
実現できる**ことが分かった（自己比較禁止の原則をより強く満たせるため、この方式に変更する）。

- `flow_t_schedule(steps, rescale_t)` の第0ノット（`i=0`）は **`steps` の値に関わらず常に `t=1.0`**
  （`linspace(1,0,steps+1)` の最初の点は常に1、rescale後も `rescale_t*1/(1+(rescale_t-1)*1)=1`）。
  かつ `apply_guidance_rescale`/CFG混合式は `t` と `sample`（=直前状態）だけに依存し `dt` には
  依存しない。→ **`sp.steps=1` で本番 `sample_flow_multi` を呼び、`sample`=既知の初期ノイズ
  （`noise_shape_0/cascade_1.npy` 等）を渡せば、step0 の `guided` 判定・CFG混合・rescale・
  clamp まで全て本番コードパスで再現できる**。返り値（1ステップ積分後の `sample`）から
  `pred = (noise - result) / dt1`（`dt1` も `flow_t_schedule(1, rescale_t)` から実測）で
  逆算し、`step0_cfg.npy` と比較する。denorm は不要（sampler の中間テンソルは全て正規化空間、
  denorm は12ステップ完了後の最終後処理のみ — 本タスクは最終値を検証しないため denorm 自体を
  実装しない。上の「レビュー反映」6番の「denorm はテスト側の複製」という記述は誤りなので撤回する）。
- **texture の step8/11 も同じ `sp.steps=1` トリックで検証できる**（shape の step8/11 は
  不可能、上記1番参照）。理由: texture の production パラメータは `gs=1.0` 固定
  （`t`/`guided` に関わらず変化しない）かつ `gr=0.0`（rescale自体を呼ばない）なので、
  `pred` は **`t`/`sample` に一切依存しない**（`stochastic` は `gs==1` 分岐で単一 positive
  呼び出しのみ、`multidiffusion` は `guided` の真偽に関わらず `pred=pos(avg)` に収束する—
  `guided=true` 分岐でも `1.0*pos+(1-1.0)*neg=pos` で neg 項が代数的に消える）。
  `sp.steps=1` は `t=1.0` を強制するため実際の step8（`t=0.6`, guided=true）とは
  **異なるコード分岐**（`!guided` 分岐、neg呼び出し無し）を通るが、texture の場合その分岐差は
  出力値に影響しない（上記の代数的等価性）。**これは意図的な簡略化であり、
  「guided分岐でneg項が正しくゼロ化される」こと自体の分岐別カバレッジは持たない**——
  正直に明記する（texture の `!guided` 分岐と `guided`(gs=1) 分岐、どちらも本番
  `sample_flow_multi` 内の実在コードだが、本テストが実際に exercise するのは前者のみ）。
- shape の step0（LR・HR 双方）は実際の step0 が `guided=true`（`gi1=1.0` 境界含む）で
  `sp.steps=1` の `t=1.0` も同じ境界のため、**実際に real step0 と同一のコード分岐**
  （`pos`/`neg` 両方呼び出し、CFG混合、rescale、clamp）をそのまま通る。分岐差の懸念は無い。
- texture の step0/11 も同様に「実際に `!guided`」なので `sp.steps=1` は分岐を含めて完全一致
  （`tex_step0`/`tex_step11` の fixture に `neg.npy` が無いことで確認済み）。step8 だけが
  上記の「分岐は変わるが値は変わらない」ケース。

## テスト方式（P4 と同じ4本柱、スコープはレビュー後に縮小）

1. **file-local static ヘルパ**: `trellis_cli.cpp` の `shape_flow` ラムダ・cascade 分岐・
   texture ブロックが持つ「純粋な配線」（denorm 算術、quantize backoff ループ）だけを
   `static` 関数として複製し、モデル数学は必ず共有関数（`sample_flow_multi`,
   `shape_upsample`）を呼ぶ。`trellis_cli.cpp` は変更しない。
2. **fixture replay**: `ReplayModel`（P4 から意味的に流用、疎ゆえ transpose 無し版）で
   `shape_lr_step*/shape_hr_step*/shape_step*/tex_step*` の pos/avg/neg を記録順に再生し、
   `sample_flow_multi` の CFG/rescale/Euler 積分を実データで駆動する。
3. **parity gate（`TRELLIS_NOFIX=1` 自己再実行）+ production gate（クランプ ON）** の二段構え
   （P4 と同じ subprocess 方式、`flow_runner.cpp` の `no_fix` が関数内 `static const bool` で
   プロセス起動後不変なため）。
4. **禁止事項の遵守**: 不一致は `FINDING:`、期待値を書き換えない、`trellis_cli.cpp` 変更なし、
   push なし。

## ファイル構成

- `src/test_trellis2_mv_shape.cpp`（新規）: shape-LR（512、6 run）+ shape-HR（1024 cascade、
  6 run。LR→HR 配線含む）。CMake ターゲット `trellis-test-trellis2-mv-shape`。
- `src/test_trellis2_mv_tex.cpp`（新規）: texture SLat（512 6run + 1024cascade 6run の
  計12run、shc/tcoords は対応する shape run の最終 coords をそのまま使う）。CMake ターゲット
  `trellis-test-trellis2-mv-tex`。
- 2ファイルに分けるのは #63/#64 の issue 分割、および CMake ターゲット単位でのビルド予算
  （このマシンのディスク/スレッド制約、COMMON.md §6）に合わせるため。SS/sampler が既に
  `trellis-test-trellis2-mv-ss` / `trellis-test-trellis2-mv-sampler` の2ターゲットに分かれている
  前例に揃える。

## 検証項目 → 実装対応（TASK-STAGES.md の番号に対応、レビュー後の縮小スコープ）

TASK-STAGES.md の項目1・2 が要求する「最終 shape_slat/tex_slat 一致」「cascade 境界の
受け渡しを含めた通し検証」は、上の「レビュー反映」1-2番の理由で**この fixture からは
実行不可能**と判明した。この2項目は削除ではなく**実行不可能という結論そのものを
`FINDING:` として報告する**（統括が fixture 再生成（全12ステップ捕捉）を指示するかどうかの
判断材料）。以下は fixture の実際の内容で検証可能な範囲。

1. **shape step0 CFG（LR/HR 両方、production パラメータ gs=7.5/gr=0.5）**: `sample`=
   `noise_shape_0.npy`（512単独・cascade LR）/ `noise_shape_cascade_1.npy`（cascade HR、
   LR出力に依存しない独立ノイズ）という既知値から、`apply_guidance_rescale` 相当の算術
   （ファイルローカルに複製、上記2番参照）で `step0_cfg.npy` を再現し rel 誤差を測る。
   6run(512) + 6run(1024cascade、LR/HR両方=12箇所) の計18箇所。HR は参照の
   `shape_slat_coords.npy` を座標としてそのまま使う（自己導出しない、上記2番）。
   `TRELLIS_NOFIX=1` の parity gate + production gate（clamp ON、`min_ratio` は診断のみ）
   の二段構えは P4 と同じ。
2. **cascade 境界（縮小版）**: LR→HR の共有カウンタ配線（Stochastic モード時に LR の
   最終カウンタ値を HR へ渡す）を trellis_cli.cpp の配線どおりに複製し、**API として
   例外なく動作する**ことのみを real fixture 上で確認する（座標・最終値の参照一致は
   検証しない、上記の通り不可能）。continued/reset の判別自体は
   `src/test_trellis2_mv_sampler.cpp:75-97` と `src/test_flow_multi.cpp:369-398`
   （`test_cascade_counter_continuation`）が synthetic fixture で既に担保済みなので
   再検証しない（重複実装を避ける、これは codex レビューでも妥当と判定済み）。
3. **shape の i=8 境界**: `flow_in_guidance_interval(ts[8], 0.6, 1.0)` を shape の
   `SamplerParams`（`rescale_t=3.0`）で直接呼び `true`（inside）を assert。
   `shape_step8_neg.npy`/`shape_lr_step8_neg.npy`/`shape_hr_step8_neg.npy` の実在
   （inside の観測的裏付け）も確認する。
4. **texture SLat step{0,8,11} 全点**（`gs=1.0/gr=0.0`、512+1024cascade 計12run）:
   `gr=0.0` で `apply_guidance_rescale` が呼ばれない（`flow_runner.cpp:560/584` の
   ガード）ため CFG 式は `sample` に依存しない純代数（`pred=pos` または `pred=avg`）。
   3点全てで `cfg.npy` を直接再現し rel 誤差 ~1e-6 を assert（`TRELLIS_NOFIX` の有無で
   結果が変わらないことも確認 — このステージはそもそも clamp 分岐に入らない）。
   texture の 64ch concat 配線（`x64[0:32]=state, x64[32:64]=固定shape_norm`）は
   probe `FlowFwd` で構造的に assert（上記4番）。
5. **OOD クランプ（issue #77）診断出力**: shape の `min_ratio` を step0（検証可能な唯一の
   step）で計算し出力する。0.2 未満なら `FINDING:`。**production gate の pass/fail には
   使わない**（上記6番、診断値に格下げ）。texture はそもそも `gr=0.0` で clamp 分岐に
   入らないため対象外。
6. **1-image 回帰（`trellis-test-slat-shape`）**: 対象外・変更なし。「ビルド確認もしない」
   と明記する（既存テストで別スコープ）。
7. **決定性**: 「同じ replay 入力（recorded tensors + noise）を2回 native に通して
   bit-identical」を assert する（native 側のみの決定性確認。fixture 自体の決定性は
   `~/nfs/pixal3d_trellis2mv_ref_v2` 側の既存検証に依存し、本タスクでは再確認しない）。
8. **coords/feats の行整合（P1、新規追加）**: 各 run で `noise_shape_*.npy` /
   `shape_*_step{0,8,11}_*.npy` / `shape_slat_feats.npy` / `tex_slat_feats.npy` の
   行数が、対応する coords ファイル（`ss_coords.npy` または `shape_slat_coords.npy`）の
   行数と一致すること、かつ該当 coords に重複行が無いこと（`std::vector` サイズ vs
   `std::set` サイズ）を assert する。texture の coords は対応する shape run の
   最終 coords と行数一致することも確認する。
9. **quantize backoff ループの structural test（P1/P8、新規追加）**: 実 fixture を使わず
   synthetic 座標で `trellis_cli.cpp:982-997` の複製ループ単体をテストする
   （token 数がわざと `max_tokens` を超える synthetic ケースを与えて実際に `-128` backoff が
   起こることを確認 — 実 fixture では起こらない分岐を、複製コードの structural test として
   カバーする）。「実 fixture で backoff が exercise された」とは主張しない。

## 実装結果（2026-09-21、実行済み）

`src/test_trellis2_mv_shape.cpp` / `src/test_trellis2_mv_tex.cpp` を実装し、
`cmake --build build --target trellis-test-trellis2-mv-{shape,tex} -j4` でビルド、
`TRELLIS_THREADS=4 ./build/trellis-test-trellis2-mv-{shape,tex}`（gpu=-1相当、CPU backend、
このMac、重み無し・fixture-replayのみ）で実行し、**両方とも `ALL PASS`（exit 0、FAIL行0件）**。

- shape: pure test 2件（quantize backoff structural、i=8境界）+ 全12run(512 x6 +
  1024cascade LR/HR x6) の step0 CFG が本番 `sample_flow_multi`（`sp.steps=1`直接呼び）経由で
  rel誤差 3.9e-8〜2.8e-7（production gate, clamp ON）/ 同オーダー（parity gate, `TRELLIS_NOFIX=1`
  subprocess）で参照と一致。coords行数・重複なしも全run確認。
- shape の OOD clamp診断（issue #77, step0のみ）: 全18箇所（6run×LR/HR、512は6run分)の
  `min_ratio` は **0.2462〜0.5918、いずれも0.2フロアを超過**——step0では clamp は発火していない
  （step8/11は本設計の制約により未測定、既知の限界として別途 FINDING で報告）。
  `FINDING:` は「clampが発火した」ではなく「step8/11は原理的に測定不能」という fixture 側の
  制約について出す。
- tex: pure test 2件（concat structural、guidance-interval境界）+ 全12run×3step(0,8,11)=36箇所の
  CFG が rel誤差 2.1e-8〜8.5e-8で参照と一致（`gr=0.0`のためclamp分岐自体に入らず、production/parity
  gateの区別は不要——両者は同じコードパスを通る）。
- 実装中に1件バグを発見・修正: texture の multidiffusion step8 チェックで、実fixtureのneg.npy
  （guided分岐の「gs=1でも常にneg評価」quirkによる記録）を`sp.steps=1`replay（not-guided強制、
  neg呼び出し無し）に誤って含めていたため「recorded callを全て消費」チェックが4/4 runで失敗した
  （4 FAILURE(S)）。`build_replay_one_step`に`include_neg`引数を追加し、
  `real_branch_guided==true`のケースではneg.npyを読み込まないよう修正、再ビルド・再実行で
  ALL PASS を確認した。

## 禁止事項の再確認

- `trellis_cli.cpp` / `flow_runner.cpp` / `shape_decoder.cpp` は変更しない。
- 不一致は `FINDING:` で報告し、実装は直さない。テストの期待値（fixture の値）を書き換えない。
- push しない。旧トラック worktree（`~/Downloads/pixal3d-mv-59` 等）は読むだけ。

## ビルド・実行の制約遵守

- `cmake --build build --target trellis-test-trellis2-mv-shape -j4` /
  `... --target trellis-test-trellis2-mv-tex -j4`（個別ターゲットのみ、`-j4`）。
- 実行は `TRELLIS_THREADS=4 ./build/trellis-test-trellis2-mv-shape ...`（`src/trellis_model.cpp:47`
  の既存 env var、コード変更不要）。
- 重いテスト（1分超）の起動間隔10分以上。本タスクは2つの新規バイナリをそれぞれ1回
  （+ 内部 parity subprocess 再実行1回）ずつ動かす想定 = 実行イベント2回。
- E2E は回さない（担当外）。

## 未決事項（実装中に判明したら FINDING: で報告、ここでは決めない）

- LR/HR で `cond_dec_bank`/`Lc_dec` の解決（512→1024 の Lc 遷移）が fixture の
  `cond_1024.npy` の列数と一致するかは実装時に読んで確認する。
