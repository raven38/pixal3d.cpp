# B3 決着: stochastic mode のテクスチャ黒化は seed 依存の確率的失敗として文書化する（native 固有バグの証拠なし）

- 対象: E2E `B3`（`--trellis2-mv <4view> --trellis2-mv-mode stochastic --res 1024`、CUDA A100、0d4cfc5）で
  base color テクスチャが全面黒（`out_base.png` mean 0.07）になった件。issue #58（epic）の派生。
- 指示元: `~/Downloads/claude-work/docs/briefs/trellis2-mv/TASK-B3.md`（native 側二分）、同 `TASK-B3-REF.md`
  （参照側掃引 R1 / noise 注入 R2）。統括 doc `ORCHESTRATION.md` §14。
- 入力 doc: `HANDOVER.md`（同ディレクトリ、native 側の全記録・pod v4〜v10）、
  `../2026-09-22-trellis2-mv-b3-ref/2026-09-22-b3-ref-r1-results.md`（参照側 23 run 掃引、以下「R1 doc」）。
- 決定者: 統括（2026-09-22）。本 doc の数値は全て R1 doc の生データ `raw/sweep_results.jsonl` と
  `logs/v9-pod-seed-sweep.log` から本セッションで再計算して照合済み（§3）。

## 0. 結論（先に、限定付き）

**修正は行わない。stochastic mode は seed 依存で tex ステージが黒化しうる既知の制限として記録する。**

根拠は次の 2 点に尽きる:

1. **参照実装（PyTorch、pinned `75fbf018`）でも黒化が起きる**。v3 fixture の `run_2img_real_stochastic_1024c`
   （2-view stochastic、seed 42）の bake 済みテクスチャは全 texel = 0 で、R1 掃引の seed 42 と bit 一致
   （`ss_coords` / `tex_slat_{feats,coords}` の sha256）。参照側の黒 run（2v seed 42）は `saturation_rate` 0.9999911、
   native 側 HANDOVER が観測した「decoder 飽和」と同じ機構。
   ただし飽和率は黒 run（n=1、0.9999911）以外では分離が弱い — 境界域 3 run（4v seed43 0.3975 / 4v seed45 0.4898 /
   2v seed43 0.5007）に対し、正常 19 run の範囲は 0.0〜0.3675 で、両者の飽和率はほぼ地続き。飽和率で黒を
   判定しているのではなく、完全な黒 1 件だけが飽和率でも突出している、という事実に留める。
2. **native と参照の黒化率に統計的な差は検出されなかった**（Fisher exact 両側、config 別 p = 0.125 / 0.423、
   pooled p = 0.076、感度分析 p = 0.60〜1.00。§3）。

**限定**: n = 10（参照）/ 6（native）の規模では差を検出できなかっただけで、**率が同じと証明されたわけではない**。
4-view では参照 0/10 vs native 2/6 と、点推定は native が高い（p = 0.125）。native 固有の寄与（tex DiT 実重み forward の
step 0 で 1.29% の乖離、§6.2）が黒化率を上げている可能性は**排除されていない**。決定打となる R2（native の noise を
参照に注入して同一軌道で黒くなるかを見る実験）は未実施（§5）。したがって本 doc の主張は
「**参照側でも起き、n = 10/6 の規模では差を検出できなかった。native 固有バグの証拠は無い**」であり、
「再現しない」「問題なし」ではない。

## 1. 凍結基準との対照表

| 凍結文面（出典） | 実測・実行値 | 一致 or 事後変更 # | 証拠ファイル |
|---|---|---|---|
| TASK-B3 手順 2「tex SLat の統計を B1/B2/B3/B4 で並べ、flow/cond 側か decoder/bake 側かを二分」 | 実施。tex SLat は全 run で finite、B3 の最終 SLat が decoder 飽和域（HANDOVER §1〜3） | 一致 | `HANDOVER.md` §「mode分離実験」「per-channel分析」、`logs/DIAG_*.log` |
| TASK-B3 手順 4「原因が確定したら最小修正 + weightless 回帰テスト」 | **修正せず**。原因は「固定コードバグ」ではなく「stochastic mode の確率的失敗」と判定 | **事後変更 #1**: 手順 4〜5 は「原因確定」を前提にしており、前提（B3 のみ黒・決定論的）が v9 掃引で崩れたため統括判断で修正を追わない | `logs/v9-pod-seed-sweep.log`、本 doc §0 |
| TASK-B3 手順 5「1 run だけ pod で B3 を再実行して mean が正常範囲になることを確認」 | 実施せず（修正が無いので確認対象が無い） | 事後変更 #1 に従属 | — |
| TASK-B3 禁止「原因未確定のまま『直った』と言わない（黒くなる最初の段を示す）」 | 「直った」とは書かない。黒くなる段 = tex ステージの sampler 軌道が decoder 飽和域へ入る（SLat 自体は finite） | 一致 | `HANDOVER.md` §「per-channel分析」、R1 doc §3（参照側も `tex_slat` finite で decode 以降で黒） |
| TASK-B3 手順 7a「Part A の `trellis-test-slat-shape` を f04b33c / 0d4cfc5 の両バイナリで再実行し永続化」 | 実施。0d4cfc5: 21 forwards / rel 2.3349e-01、f04b33c: 20 forwards / rel 4.3312e-01（fix(flow) で inside 9 vs 8 の直接証拠） | 一致 | `logs/7a_0d4cfc5.log`、`logs/7a_f04b33c.log` |
| TASK-B3 手順 7b「fail-closed 5 パターンを実重み経路で叩き exit code を永続化」 | 実施。5 パターン全て rc=1 | 一致 | `logs/7b_fail_closed.log` |
| TASK-B3-REF R1「4-view / 2-view × stochastic × res 1024 で seed 10 個ずつ、multidiffusion 4-view × 3」 | 4v 10 + 2v 10 + 4v-md 3 = 23 run、seed 42〜51 / 42〜44 | 一致 | `../2026-09-22-trellis2-mv-b3-ref/raw/sweep_results.jsonl`（23 行） |
| TASK-B3-REF R1「黒判定 mean < 2」 | `classify()` は mean < 2 → black、2〜10 → intermediate、≥ 10 → normal | 一致 | `tools/ref_trellis2_mv_seed_sweep.py::classify` |
| TASK-B3-REF R1 の凍結分岐「**黒化率 0/20 なら (b) が濃厚、native と同水準（~30%）なら (a)**」 | 実測は strict 黒 **1/20（5%）**、intermediate 込み **5/20（25%）**。native は strict 3/9（33%） | **事後変更 #2**: どちらの分岐にも落ちない（0/20 でも ~30% でもない）ため**凍結分岐では判定不能**。(a) の採用は Fisher 有意差なし + 参照側の完全黒 1 件を根拠とする統括判断 | R1 doc §4・§6、本 doc §3 |
| TASK-B3-REF R1「Fisher exact で native（3/9）との差を検定し p を添える」 | config 別 p = 0.1250 / 0.4231、pooled p = 0.0760。本セッションで生データから再計算し一致 | 一致 | 本 doc §3（再計算スクリプトは §3 末尾） |
| TASK-B3-REF R1 較正ゲート「proxy 分類と baked 実測の分類が一致」 | 4v multidiffusion が境界値（9.85 vs 10.23）で不一致 → 「分類一致 or 絶対差 2.0pt 以内」へ緩和して再実行 | **事後変更 #3**（R1 担当が R1 doc §2 に申告済み。本 doc はそれを引き継ぐ） | R1 doc §2・§5、`raw/calibration_report.json` |
| TASK-B3-REF R2「native の noise + 参照 cond を PyTorch tex に注入、PyTorch も黒くなるなら (a)」 | **未実施**。dump に `tex_shape_guide_norm.npy`（64ch concat の guide 半分）が無く tex DiT forward を呼べない | **事後変更 #4**: R2 なしで決着。R2 は本 doc の反証経路として §5 に残す | R1 doc §8、`HANDOVER.md` §「TASK-B3-REF向けdump完了」 |
| 統括 doc §14 の advisor 判定基準「B3 で 1 本でも正常なら失敗率問題（構造バグでない）」 | native B3 6 seed 中 4 本正常（mean 37.9〜50.5） | 一致 | `logs/v9-pod-seed-sweep.log` L16〜21 |
| COMMON.md「pod は自然終了型、完了後削除、kubectl cp は sha256 照合」 | R1: 自然終了型 driver、sha256 照合済み（R1 doc §冒頭）。native 側は同一 pod 名 `trellis2mv-b3-diag` を v1〜v10 で使い回しており（HANDOVER の各節が同名 pod を参照）、自然終了型では成立しないので**常駐型だったと推定される**。全 pod 削除済み（本セッション 2026-09-22 の `kubectl get pods` で trellis2/b3 系 0 件を再確認） | R1 一致 / native 側は**事後変更 #5**。**ただし HANDOVER §「禁止事項の遵守状況」は pod 形態に触れておらず、担当の自己申告は無い**（本 doc で統括が後から推定して記録した） | R1 doc、`HANDOVER.md` §「禁止事項の遵守状況」（L433〜、pod 形態の記載無しを確認） |
| TASK-B3 禁止「fixture テストの期待値を触らない」 | `src/test_trellis2_mv_tex.cpp` 未変更（HANDOVER §「禁止事項の遵守状況」の自己申告、`git diff 0d4cfc5 diag/trellis2-mv-b3 -- src/test_trellis2_mv_tex.cpp` が空） | 一致 | `HANDOVER.md` L433〜 |
| TASK-B3 禁止「`feat/trellis2-mv-e2e` worktree を変更しない」 | 触れていない（同自己申告。統括も本セッションで読み取りのみ） | 一致 | 同上 |
| TASK-B3 / TASK-B3-REF 禁止「push しない」 | 未 push。hardening / diag / mv-59 のいずれも remote へ送っていない（push は go-queue #66 で GO 待ち） | 一致 | `git rev-list --count origin/feat/trellis2-mv..HEAD` |
| TASK-B3-REF 禁止「v2/v3 fixture を変更しない」 | 変更していない。§6.1 の取り違えも**発見して記録しただけで rename していない**（GO 待ち） | 一致 | 本 doc §6.1、go-queue #66f |
| TASK-B3-REF 禁止「pod を 2 本同時に走らせない」 | R1 の pod `trellis2mv-b3-ref-sweep` は 1 本。**native 側 pod との重なりは確認できない** — sweep の開始時刻がどこにも記録されておらず（`sweep.log` に wall-clock 行が 0、`sweep_summary.json` は `finished_at` のみ）、native v9 は 15:15:06〜15:24:54 UTC、v10 は 15:31:54〜15:35:56 UTC、sweep の finish は 16:07:43 UTC。sweep の run 時間合計は 590.2 s だが pipeline load・較正 3 件・pod bootstrap が加わるため開始時刻を逆算できず、v10 と重なっていた可能性を否定できない | **未確認**（「一致」とは書けない） | `raw/sweep.log`、`raw/sweep_summary.json`、`logs/v9-pod-seed-sweep.log`、`logs/v10-pod-tex-dump.log` の各タイムスタンプ |
| TASK-B3-REF R3「黒 run と正常 run の tex SLat 統計（mean / std / **max abs**、飽和割合）を並べる」 | 参照側の生データで **tex SLat を見ているキーは `tex_slat_mean` / `tex_slat_std` / `tex_slat_has_nan_or_inf` の 3 つだけ**（`tools/ref_trellis2_mv_seed_sweep.py:370-378`、`tex_feats_np` から算出）。`saturation_rate` と `per_channel_mean_0_1` / `per_channel_std_0_1` は **tex SLat ではなく decode 後の PBR 属性**の統計（同 `base_color_stats()`:95-109、`mesh.attrs [Mv,6]` から算出）で、R3 が求めた tex SLat 統計ではない。飽和率は黒 1 件 0.9999911、境界域 3 件 0.3975〜0.5007、正常 19 件 0.0〜0.3675。native 側は per-channel 統計が HANDOVER §「per-channel分析」にある | **事後変更 #6（未達を含む）**: ①並置表そのものが R1 doc に作られなかった（表は `base_color_mean` と `tex_active_voxels` のみ）②**`max abs` は生データのキーにも R1 doc にも存在せず、測定されていない**（`sweep_results.jsonl` の全キーを列挙して確認）。tex SLat の mean/std は生データから引けるが、**`max abs` は保存されておらず mean/std からは再構成できない**（測るには pod 再実行が要る） | `raw/sweep_results.jsonl`（キー一覧）、`HANDOVER.md` §「per-channel分析」 |
| TASK-B3 手順 5「pod は同時に 1 本のみ」 | native 側の v4〜v10 は同一 pod 名の使い回しで 1 本ずつ。**参照側 R1 と native v10 の重なりは確認できない**（上の「pod を 2 本同時に走らせない」行と同じ理由 — sweep の開始時刻が記録されていない） | **未確認** | 同上 |
| TASK-B3 手順 1「仮説を先に列挙して doc に書く」 | 仮説 a〜f を列挙し、codex exec と advisor のレビューを反映した設計 doc が着手前に作られている | 一致 | `docs/design/2026-09-21-trellis2-mv-b3-tex-black.md`（冒頭に superseded 注記あり） |
| TASK-B3 手順 3「flow/cond 側なら cond / counter / SamplerParams を dump して比較」 | 発動した（手順 2 の二分で cond 側の疑いが残ったため）。**3 つの対象で経路が違う**: **cond** は pod v7（cond_1024 差し替え）と v8（cond_512 差し替え）で差し替え実験、v10 で `tex_cond_view{v}.npy` として dump。**SamplerParams** は v10 で `tex_sampler_params.txt` として dump。**counter は dump していない** — コード実測（`sample_bank` 呼び出しで tex に counter を渡していない = nullptr）と凍結仕様「SS and texture each get a fresh counter」で棄却済み。手順 3 が求めた「各 step で選ばれた view index」は `[flow-mv-step] i= 0 seq=0 view=0/4 …` として DIAG ログに出ている | 一致（実施済み）。**ただし cond 差し替えの結果は結論に使っていない**。v8 は cond_512 を差し替えたことで SS の active voxel 数が変わり、shape-LR 以降の noise が丸ごと別サンプルになるため因果解釈不能（HANDOVER 00:3x 訂正）。v7 は cond_512 が native のままで noise ストリームが同一なので**この機構には当たらず**、00:3x は「因果として言えるのは v7 のみ」と明記している。その v7 も 00:5x の v9 シード掃引の後に「単に別の noise draw を引いただけの可能性が高い」へ格下げされており、**HANDOVER 内で v7 の扱いが 00:3x と 00:5x で食い違っている**（本 doc は v7/v8 いずれの結果も結論の根拠にしていない） | `HANDOVER.md` L241-244（00:3x、v7 は noise 同一）・L303-304（00:5x、v7/v8 とも別 draw 説）・L320-332（v10 の dump 一覧、counter 無し）・L36-37（counter nullptr）、`logs/v7-pod-cond-swap.log`、`logs/v8-pod-cond-swap-512.log`、`logs/v10-pod-tex-dump.log`、`logs/DIAG_B3_4view_stochastic.log` L22-25 |

**行にしなかった凍結文面**（Q1 結論依存・Q2 無言逸脱可能性がともに No、または既存行と証拠が重複するもの）。
行の採否は「黙って省略・変更したときに本 doc の結論・限定文・数値表が変わりうるか」で判定した:
- TASK-B3 手順 6「view 順ログは追加不要・何も足さない」— 証拠が「hardening の src/ に計装が無いこと」で、
  上の「push しない」「fixture テストの期待値を触らない」行および §7 の「診断用計装は `diag/trellis2-mv-b3` にのみ存在」と重複。
- **運用指示の類**（結論に関わらないもの）: 両 brief の報告フォーマット（`PARTIAL:` 各 2 種・`DONE:` 3 行・日本語で応答）、
  冒頭の「まず COMMON.md を読んでから着手すること」、worktree / cwd の指定、TASK-B3-REF R1「率は『N 本中 k 本』と書き」、
  同 R2「注入した noise が使われたことを sha256 でログに残す」（R2 自体が未実施なので従属）、同 R3「結論（a/b）は統括が下すので
  事実と p 値だけ書く」（R1 doc がそのとおりに書かれていることは §0 の引用で担保済み）。


事後変更 #1〜#6 のうち、結論に影響しうるのは #2（凍結分岐で判定不能）と #4（決定打未実施）。どちらも §0 の限定文に反映した。
#6（tex SLat 統計の並置表が未作成）は結論を変えないが、飽和の機構を主張する材料が生データ止まりであることを意味する。

## 2. 事実（native 側と参照側を並べる）

| 系 | 構成 | seed | n | 黒（mean < 2） | 境界域（2〜10） | 正常（≥ 10） | 出典 |
|---|---|---|---|---|---|---|---|
| native CUDA 0d4cfc5 | B3: 4-view stochastic 1024c | 42〜47 | 6 | **2**（42: 0.069、44: 0.578） | 0 | 4（37.9 / 38.8 / 47.3 / 50.5） | `logs/v9-pod-seed-sweep.log` |
| native CUDA 0d4cfc5 | B1: 2-view stochastic 1024c | 42〜44 | 3 | **1**（44: 1.720） | 0 | 2（11.0 / 31.3） | 同上 |
| PyTorch 参照 | 4v stochastic 1024c | 42〜51 | 10 | **0** | 2（43: 6.09、45: 9.94） | 8（12.6〜50.7） | R1 doc §4 |
| PyTorch 参照 | 2v stochastic 1024c | 42〜51 | 10 | **1**（42: 0.190、v3 fixture と bit 一致） | 1（43: 6.04） | 8（13.6〜61.4） | 同上 |
| PyTorch 参照 | 4v multidiffusion 1024c | 42〜44 | 3 | 0 | 0 | 3（10.2 / 17.3 / 39.2） | 同上 |
| native CUDA 0d4cfc5 | B2 / B4（multidiffusion） | 42 | 各 1 | 0 | 0 | 1 | E2E 結果 doc（`feat/trellis2-mv-e2e`） |

補足事実:

- 全 23 参照 run と native 全 run で `tex_slat` に NaN/Inf は無い。黒化は SLat が finite のまま decoder の飽和域に入ることで起きる
  （native: HANDOVER §「per-channel分析」、参照: R1 `saturation_rate`）。
- 同 seed で multidiffusion に切り替えると native も正常（HANDOVER §「mode分離実験」、seed 42）。参照側 multidiffusion は 3/3 正常。
- native の seed と参照の seed は **RNG が別**（native は `std::mt19937` + `std::normal_distribution`（`trellis_cli.cpp:323/761`）、参照は `torch.manual_seed`）で、同じ seed 番号でも
  noise は対応しない。2 系は独立標本であり、run 単位の対応比較はできない。

## 3. 検定（Fisher exact、両側）

| 比較 | 参照 | native | p | 判定 |
|---|---|---|---|---|
| 4v stochastic（B3） | 0/10 | 2/6 | **0.1250** | 有意差なし |
| 2v stochastic（B1） | 1/10 | 1/3 | **0.4231** | 有意差なし |
| pooled（記述的補助） | 1/20 | 3/9 | **0.0760** | 有意差なし |
| 感度: mean < 10 を「not normal」とした 4v | 2/10 | 2/6 | 0.6044 | 有意差なし |
| 感度: 同 2v | 2/10 | 1/3 | 1.0000 | 有意差なし |
| 感度: 同 pooled | 4/20 | 3/9 | 0.6424 | 有意差なし |

再計算（本セッション、超幾何分布の直接和、`math.comb`）: `raw/sweep_results.jsonl` から黒 0/10・1/10・0/3 と境界域 2/10・1/10・0/3、
`logs/v9-pod-seed-sweep.log` から native 2/6・1/3 を読み、上表 6 個の p 全てが R1 doc §6 の値と 4 桁一致した。

## 4. 隠さない観察差（解釈はしない）

n 不足のためどれも「差がある」とは言えないが、記録として残す:

- **分布の形**: native の 9 run は双峰（0.07 / 0.58 / 1.72 と 11.0〜50.5、2〜10 に 0 件）。参照 4v は 6〜51 に連続的に散り、
  境界域 2 件。2v は 0.19 と 6.0〜61 に散る。
- **測り方**: native は bake 後 `out_base.png` の平均、参照は bake 前 raw voxel 属性の平均 ×255（proxy）。較正 3 件で
  絶対差 0.19〜1.11pt（R1 doc §3）。
- **フレーム構成**: B1 の 2-view は実際には Frame1 + **Frame3**（§6.1）。参照 2v も同じ 2 枚なので比較は対称。
- **native の B3 vs B1**: 2/6 vs 1/3 で「V=4 が V=2 より脆い」根拠は無い（当初 E2E 報告の前提は n = 1 のサンプリング運、HANDOVER §「v9」）。

## 5. 反証経路（この結論が覆る条件）

- **R2（native noise 注入）**: native seed 42 B3 の `tex_noise0.npy` + HR coords + 参照 cond を PyTorch tex ステージに注入して
  12 step 回し、参照が**正常**なら「同じ noise で native だけ黒 → native 固有」で (b) が復活する。逆に参照も黒なら (a) が確定する。
  必要な追加 dump は `tex_shape_guide_norm.npy`（64ch concat の guide 半分、[N,32]）。dump 実装は `diag/trellis2-mv-b3`
  （`TRELLIS_DUMP_TEX_INPUTS`）、注入機構は `tools/ref_trellis2_mv_noise_injector.py`（selftest 済み）。
- **n を増やす**: 「参照 0% vs native 33%」という点推定が真だとしても、Fisher 両側で p < 0.05 に届くのは各側 15 run 以上
  （0/15 vs 5/15 で p = 0.042、0/20 vs 7/20 で p = 0.008。本セッションで計算）。A100 で参照 ~20 s/run、native は 1 run 数分。
- **B2/B4 の多 seed**: multidiffusion が本当に 0% か（希釈で頑健か、まだ当たっていないだけか）は native 側 n = 1、参照側 n = 3。

## 6. 付随して確定した事実（別トラックへ）

### 6.1 fixture 取り違え: `yoimiya_2view/Frame4.png` は `yoimiya_4view/Frame3.png` のコピー

本セッションで再検証（`shasum -a 256`、`~/nfs/pixal3d_trellis2mv_ref_v2/inputs/`）:

```
49a99ae240ecdf7d  yoimiya_4view/Frame1.png
bffaeef7b1247d50  yoimiya_4view/Frame2.png
da945908a20a1afb  yoimiya_4view/Frame3.png
d81de09e0df3b87f  yoimiya_4view/Frame4.png
49a99ae240ecdf7d  yoimiya_2view/Frame1.png
da945908a20a1afb  yoimiya_2view/Frame4.png   <- 4view/Frame3.png と同一
```

これまでの全 B1（2-view）実行と参照 2v run は Frame1 + Frame3 を使っていた。native と参照が同じバイトを読んでいるので
本 doc の比較は無効化されないが、E2E 結果 doc・HANDOVER・R1 doc・旧トラックのカタログ（`feat/trellis2-mv-59` の
`docs/spec/33-trellis2-mv-reference.md` L390「`Frame{1,4}`」）の表記は全て誤り。**fixture は共有物（ローカル + NFS
`/nfs/pixal3d_trellis2mv_ref_v{2,3}`、他 worktree が参照）なので rename は GO 待ち**（go-queue #66f）。

### 6.2 tex DiT の実重み forward が参照と step 0 で 1.29% 乖離（未解決、別 issue 候補）

`diag/trellis2-mv-b3` の `trellis-test-trellis2-mv-tex-realweight-parity`（pod v5、`logs/v5-parity-tail.txt`）: fixture の
`noise_tex_0.npy` + `cond_1024.npy` を実 GGUF（`/nfs/trellis2-mv/gguf-f16/tex_flow_1024.gguf`、**f16**）で forward した
生速度が fixture の `tex_step0_cfg.npy` と rel 1.29e-2、step 8 で 4.45e-2、step 11 で 5.12e-2。既存の
`trellis-test-trellis2-mv-tex` は ReplayModel（記録済みテンソルを返す）なので CFG 算術しか見ておらず、この乖離は見えない。

- 本件の黒化との因果は**未確立**（参照側でも黒化が起き、n=10/6 では率の差を検出できなかったため、1.29% が native の黒化率を押し上げているかどうかは R2 待ち。§0 のとおり「率が同じ」とは言えない）。
- f16 重み量子化 / FlashAttention（bf16、`--no-fa` 未試行。FA 設定はログ未記録）/ CUDA 演算順のどれが主因かは未切り分け。
- 継承 trellis.cpp には tex flow の実重み parity テストが無い（`trellis-test-shape-flow` はある）ので、
  **single-image 経路にも共通する可能性**がある。#65 の範囲外 → 別 issue として起票する候補（go-queue #66g）。

## 7. 成果物の所在

| 物 | 場所 |
|---|---|
| native 側の全記録（pod v4〜v10、二分の時系列、7a/7b） | `docs/results/2026-09-21-trellis2-mv-b3/{HANDOVER.md,logs/,matte-check/}`（本ブランチ） |
| 仮説列挙・codex/advisor レビュー | `docs/design/2026-09-21-trellis2-mv-b3-tex-black.md`（本ブランチ） |
| 参照側 R1 掃引（doc・生データ・tool） | `docs/results/2026-09-22-trellis2-mv-b3-ref/`、`tools/ref_trellis2_mv_seed_sweep.py`、`tools/run_b3_ref_sweep_driver.sh`（c9044b6 を本ブランチへ cherry-pick） |
| R2 用 noise 注入機構 | `tools/ref_trellis2_mv_noise_injector.py`（同上） |
| 診断用計装・probe テスト・ad-hoc tool（**PR に含めない**） | ブランチ `diag/trellis2-mv-b3`（tip 2649e61 = 0d4cfc5 + wip 10 コミット） |
| native の tex 入力 dump（seed 42 B3 / seed 44 B1） | NFS `/nfs/trellis2-mv/b3-dump/{B3_seed42,B1_seed44}/`（sha256 は `logs/v10-pod-tex-dump.log`） |
| 既知の制限としての登録 | `docs/spec/28-divergence-matrix.md` Addendum 4 |

## 8. 後続（GO 待ち、go-queue #66 に登録）

- 66e: issue #58 に本 doc の事実と p 値をコメント（文案は go-queue に置く）。
- 66f: fixture `yoimiya_2view/Frame4.png` → `Frame3.png` へ rename（ローカル v2 + NFS v2/v3 + カタログ + E2E/HANDOVER/R1 doc の表記訂正）。
- 66g: tex DiT 実重み forward 1.29% 乖離の issue 起票（single-image を含む tex flow の実重み parity テスト追加を提案）。
