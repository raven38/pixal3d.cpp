# Pixal3D conditioning（NAF@1024 / HR cond / upsample / PBR decode）の段別プロファイル（issue #33）

日付: 2026-09-20 / 対象ブランチ: `perf/cond-profile`（`perf/metal-rope-layout` d4ffdf4 起点、`--profile` f2d4bfe を含む）/
ステータス: 実装・計測完了（2026-09-20、結果は `docs/results/2026-09-20-conditioning-profile.md`。設計の codex レビュー 1 周目 = 差し戻し、再確認で A6/A7/B4 の部分指摘をテスト計画へ追加 → 実装はこの改訂版。結果の codex 反証レビュー `docs/reviews/2026-09-20_conditioning-profile_results_review.md` = 差し戻し → 汚染の機械分類・実行ごとゲートでの再計測・GroupNorm 単独の対照・出力比較・`nth` 反証テストを追加して改訂）/
親: #28、本 issue: #33

## 概要

MV 1024 の E2E（M4 Max / Metal / q8_0 MV v1 / cyclops 4 view）で、4 段の flow DiT の外側に残る時間のうち
**texture 段の「NAF@1024 conditioning + PBR decode」≈ 95〜109 s** と **HR shape 段の「cond / upsample」≈ 50 s** は
RoPE 修正（#28）では縮まらず、内訳も未計測。issue #33 の acceptance は
(1) 同じ cyclops MV 1024 fixture で M4 Max / Metal の stage/op 内訳を出す、
(2) 実測見込み付きで top 1〜3 のボトルネックを特定する、(3) 最適化は follow-up PR で行う（本 PR には入れない）。

本設計は (1)(2) のための**計測**を対象にする。計測は次の 3 層に分け、**互いに別の実行で取る**（同じ実行に混ぜると
外側の時間が内側の計測コストを含んでしまう）:

| 層 | 何を測るか | どこで | 有効化 |
|---|---|---|---|
| L0 常時ラップ `[cond]` | 段ごとの flow 以外の排他的な区間（load / cond / gather / upsample / decode …） | `trellis_run_mv`（本番経路） | 常時（`[post]` と同じ） |
| L1 sub-stage ラップ `[cond-v]` | view ごと / decoder stage ごとの排他的な区間 | `pixal3d_cond_slat` / `cond_slat_gpu_chunked` / `naf_upsample_ggml` / `decode_unet` / `dinov3_encode` | 新フラグ `--profile-cond`（printf のみ、計算を足さない） |
| L2 op 別 | 1 グラフを node 単位に再実行して op 種別で集計 | **診断バイナリのみ**（`trellis-test-pixal3d-cond-tex --naf-ops`）。本番経路では決して回さない | 診断バイナリの引数 |

## 背景（実測と、コードを読んで確定した事実）

### E2E の段別（`[flow]` 以外の時間）

| 区間 | `docs/results/2026-09-20-metal-flow-profile.md`（legacy RoPE、`--profile`） | 新 RoPE の E2E（別セッション `rope_e2e.log`、2026-09-20、他プロセスと GPU/CPU を共有） |
|---|---:|---:|
| [2/6] SS: cond + SS decode | 111.5 − 104.8 = 6.7 s | 43.6 − 37.7 = 5.9 s |
| [3/6] LR: cond（S=512, R=32, NAF T=512） | 165.7 − 150.4 = 15.3 s | 56.1 − 41.3 = 14.8 s |
| [3/6] HR: upsample + quantize + cond（S=1024, R=64, NAF T=512） | 825.0 − 772.3 = **52.7 s** | 678.3 − 630.8 = **47.5 s** |
| [4/6] shape decode | 22.2 s | 25.2 s |
| [5/6] texture: cond（S=1024, R=64, NAF T=1024） + PBR decode | 581.1 − 484.3 = **96.8 s** | 482.1 − 373.3 = **108.8 s** |

flow が RoPE 修正で縮むほど conditioning（合計 ≈ 175 s）の比率は上がる（新 RoPE の E2E では全体 ≈ 1470 s の 12%）。

### コードを読んで分かったこと（`src/trellis_cli.cpp::trellis_run_mv`）

1. **CLI の 3 つの SLAT conditioning（L381 / L437 / L486）は host 版 `pixal3d_cond_slat` を呼んでいる。**
   WASM / WebGPU 側だけが device 常駐・sparse 版 `pixal3d_cond_slat_gpu(..., &coords)` を使う。
   host 版は view ごとに
   - `dinov3_encode`（device）→ `[Ntok, 1024]` を host へ読み出し（`tensor_to_f32`）→ `patch_tokens_to_chw`（host 転置）
   - `proj_grid_sample`（host、double 精度 bilinear）で **dense R³ = 262 144 token × 1024 ch** の lr
   - `naf_upsample`（Metal では `naf_upsample_ggml`: 1 グラフ。T=1024 で gallocr 11.3 GB、出力 `[1024, 1024²]` = 4 GiB）
     → 4 GiB を host へ materialize（unified memory なので PCIe 転送ではないが、backend readback API による
     コピーとページ圧はそのまま残る）→ **stride T² の un-permute**（`out[ch*TT + ras]`、1 画素につき 1024 本の
     4 MiB 離れた書き込み）
   - `proj_grid_sample` で NAF map（4 GiB）から dense R³ × 1024 ch の hr
   - dense `[R³, 2048]`（2 GiB）へ累積
   を行い、最後に `pixal3d_gather_proj` で N ≈ 17.6k token（R³ の **6.7%**）だけを拾う。
   `docs/PIXAL3D_WEBGPU_MEMORY.md` §11 には同じ Metal で GPU 版（分割グラフ・sparse）が 4 view 41.8 s（N=16 384）
   という実測が既にある。
2. **DINOv3@1024 は HR shape 段と texture 段で同じ `views1024` に対して 2 回走る**（違いは NAF の T だけ）。
   dinov3 / pixal3d_naf の GGUF は 4 回 load される（SS / LR / HR / tex）。
3. `shape_upsample` は `decode_unet(coords_only=true)` で、shape decoder の 4 stage（ConvNeXt + C2S）を
   **全部**回して coords だけ取る（参照実装 `decoder.upsample` と同じ構成。res32 → res512 なので 4 段が必要）。
4. HR の quantize は `std::set<std::array<int,3>>` へ 1.2M 要素の挿入。
5. `[5/6]` の時間は cond と `tex_decode`（M = 4.82M voxel の U-Net）が未分離。
6. `naf_upsample_ggml` には `TRELLIS_DBG_NAF` の lap（graph alloc / inputs / graph compute / readback+free）が既にある。
7. **gallocr は入力バッファもグラフ内で再利用する**（`src/flow_runner.cpp:80` のコメント「re-upload each forward
   (gallocr reuses input buffers across runs)」）。したがって whole 実行後に node を再実行するには**入力を再アップロード
   してから node 0 から順に**回す必要があり、永続バッファへ cpy で書くグラフ（分割グラフ版の `_dino` / `_naf_attn` の
   accumulator 加算、`_naf_enc*` の `P_pooled`、`_naf_qk` の `Q_bm` / `K_rows`）は、再実行が壊れた入力を読むと後続の
   本番計算を汚す。**本番経路の中で op 別再実行をしてはいけない。**

### 先行計測（新規コードなし、`trellis-test-pixal3d-cond-tex --host`）

`build-metal/trellis-test-pixal3d-cond-tex <dinov3> <naf> docker/linux-webgpu-gate/e2e/views 0 --S 1024 --R 64
--naf-t {1024,512} --coords hr_coords_f32.npy --host`（N=17 489、`synth_tex_fixture/hr_coords.npy` を f32 化。
E2E の cyclops は N=17 612〜17 614 で 1% 差）。

1 回目（02:18〜02:24、**他セッションの E2E texture flow（GPU）・decimate ベンチ（CPU）と同時実行 → 汚染あり**。
同一プロセス内で GPU 版 → host 版の順に走る現行のテスト構成なので順序効果も含む。数値の一致判定だけは汚染に無関係）:

| 条件 | GPU 版（sparse・分割グラフ） | host 版 | host 版の NAF 1 view（`TRELLIS_DBG_NAF`） | 数値差 GPU vs host |
|---|---:|---:|---|---|
| S=1024 R=64 T=1024（tex） | 139.2 s（最遅 view 78 s = GPU 競合） | 90.5 s | alloc+inputs ≈ 1 s / compute 7.6 s / readback+unpermute **9.3 s**、gallocr 11.3 GB | global bit 一致、proj L2rel 4.9e-8 PASS |
| S=1024 R=64 T=512（HR shape） | 32.7 s（最遅 view 8.3 s） | 46.3 s | compute 6.4 s / readback+unpermute 1.5 s、gallocr 5.6 GB | 同上 |

読み取り（1 例観測、§4 の反復で確定させる）: (i) NAF の device compute は T=512 でも 6.4 s/view で、T=1024 との差は
1.2 s しかない → **encoder（S=1024 の 3×3 conv ×4 + 1×1 conv ×5、im2col 経路）が 1 view 6 s** を占め、host 版・GPU 版の
両方に共通する。FLOP は 1.3 TFLOP/view 程度なので Metal で 6 s は 20 倍以上遅く、kernel / レイアウト側の問題
（L2 を NAF に入れる根拠）。(ii) T=1024 の host 版は 4 GiB の materialize + stride-T² un-permute で 9.3 s/view。
(iii) GPU 版の T=1024 は汚染で判定不能（§11 の既存値 41.8 s と 3 倍違う）。

## 提案

### 1. フラグ（`include/trellis_args.h` / `src/trellis_args.cpp` / `src/trellis_cli.cpp`）

| フラグ | 意味 | グローバル |
|---|---|---|
| `--profile`（既存） | flow DiT の side-pass（segment / node pass）。**flow の後に来る conditioning の熱状態を変える**ので、
  conditioning の時間を取るときは使わない | `g_profile`（既存） |
| `--profile-cond`（新規） | L1 の sub-stage ラップを印字する。計算は一切足さない（printf と `steady_clock` だけ） | `g_profile_cond`（新規、`dit.cpp` で定義、`trellis_args.h` で extern） |

`--profile` は `--profile-cond` を含意**しない**（同時指定は可能だが、その E2E の cond 値は「flow side-pass 後」と記録する）。
`trellis_run_mv` / `trellis_run_sv` で `g_no_fa` / `g_profile` を設定している箇所に `g_profile_cond = cfg.profile_cond;` を並べる。
`README.md` のフラグ一覧に 1 行。

### 2. L0: 常時ラップ `[cond]`（`src/trellis_cli.cpp::trellis_run_mv`）

`[post]` と同じ lambda（`      [cond] <what> (%.1fs)`、呼ぶたびに `t_lap = now()` へリセット）で、**段の中を隙間なく**
区切る。flow も 1 つのラップ（`flow (load + sampler)`）として取るので、**ラップの和 = 段合計**が構成上成り立ち、
残差は出ない（段末の既存 `HR shape SLAT (%.1fs)` 等は残し、直前に `[cond] <stage> accounted = Σlaps` を印字して
±0.1 s で一致することを機械確認できるようにする）。

| 段 | ラップ（この順で隙間なし） |
|---|---|
| [2/6] | `load dinov3` / `cond_ss (V views)` / `flow (load + sampler)` / `ss_decode (load + decode + coords)` |
| [3/6] LR | `load dinov3+naf` / `cond_slat S=512 T=512 (V views)` / `gather_proj` / `flow (load + sampler)` / `denorm + stats` |
| [3/6] HR | `load shape_dec` / `shape_upsample` / `quantize (set)` / `load dinov3+naf` / `cond_slat S=1024 T=512 (V views)` / `gather_proj` / `flow (load + sampler)` / `denorm + stats` |
| [4/6] | `load shape_dec` / `shape_decode` / `dual_grid_to_mesh` / `fill_holes` |
| [5/6] | `load dinov3+naf` / `cond_slat S=1024 T=1024 (V views)` / `gather_proj` / `flow (load + sampler)` / `load tex_dec` / `tex_decode` / `pbr unpack` |

`[flow] 12 steps, …` の既存行はサンプラが出す inclusive な値で、`flow (load + sampler)` ラップの内側。結果表では
「足してよいのは同じ接頭辞の行だけ」と明記する。

### 3. L1: `--profile-cond` の sub-stage ラップ（printf のみ、計算を足さない）

出力接頭辞は `[cond-v]`（view / stage 単位）。L0 の `[cond]` の**内側**にあり、`[cond]` と足してはいけない
（結果表の階層で区別する）。各ラップは排他的（直前のラップ終了時刻から）で、区間の先頭から末尾まで隙間なく区切る。
テストバイナリは stdout がブロックバッファなので、L1 の印字は行ごとに `fflush(stdout)` する。

| 場所 | 1 行の内容（排他的ラップ、秒） |
|---|---|
| `pixal3d_cond_slat`（host 版） | 先頭で `alloc out.proj (…MB) + calc_mats`、view ごとに `normalize / dino / chw / lr_proj / naf / hr_proj / accum` |
| `pixal3d_cond_ss`（host 版） | `alloc + calc_mats`、view ごとに `normalize / dino / chw / proj / accum` |
| `naf_upsample_ggml` | 既存 `TRELLIS_DBG_NAF` の lap を `g_profile_cond` でも出す（両方有効でも 1 回だけ印字）。`readback+free` を
  `readback`（`tensor_to_f32`）/ `unpermute`（stride-T² 転置）/ `free` に分ける。`naf` ラップの内側 |
| `cond_slat_gpu_chunked` | `run_graph` の中で `build+alloc / upload / compute / free` を計時し、view ごとにグラフ種別
  `dino / naf_enc / naf_qk (stripes) / naf_attn (chunks run / skipped)` の合計と、host 側 `taps (bilinear_taps ×2 + sparse 選択)`、
  `chunk_prep (ci/cw 生成)` を 1 行で出す。末尾に `readback + interleave` |
| `decode_unet`（`shape_upsample` / `shape_decode` / `tex_decode`） | `from_latent`、stage ごとに `neighbor_table / convnext / c2s` と N、末尾 `output_layer`（融合時は 0）。
  `sparse_c2s` の内側（2 つ目の neighbor table、mask / coords 構築）は c2s ラップに含まれる inclusive 値として扱う |
| `dinov3_encode` | `build+alloc / upload / compute / readback / free` |

### 4. L2: op 別プロファイル（診断バイナリのみ）

#### 共通 primitive（`src/graph_dump.cpp` / `include/graph_dump.h`）

```cpp
// 割り当て済み・実行済みのグラフ g を node 単位に再実行して各 node の秒を返す。
//   passes: 何周回すか（1 周目は非 fusion パイプラインのコンパイルを含むので捨て、最後の周を返す）
//   reupload: 各周の先頭で呼ぶ。gallocr は入力バッファもグラフ内で再利用するので、呼び出し側は
//             ggml_set_input したテンソル全部をここで再アップロードしなければならない
// 契約: (1) g の出力は呼ぶ前に host へ読み出しておく（再実行後の中身は本番の値ではない）。
//       (2) g が gallocr 外の永続テンソルへ書く（ggml_cpy 先が永続バッファ）なら呼んではいけない。
//           呼び出し側が保証する（関数側では判定できない）。
std::vector<double> trellis_graph_node_times(ggml_backend_t backend, ggml_cgraph* g, int passes,
                                             const std::function<void()>& reupload);
// op 種別の集計表（MUL_MAT は src0 型で分割、FLASH_ATTN_EXT は K/V 型で分割）と上位 node を印字。
// whole_s は同じグラフの whole 実行の秒（node 合計との比を出す。isolated である旨を明記）。
void trellis_print_op_table(ggml_cgraph* g, const std::vector<double>& node_s, double whole_s,
                            const char* label, int top_nodes = 15);
```

`DitRunner::profile_forward` の node pass（`for j: dt[j] = run_slice(j, j)` を 2 周）は
`trellis_graph_node_times(m_.backend, g_, 2, [&]{ set x/t/cond/rope/proj })` に置き換える。role × block の
segment pass、`op_role_s` の集計、印字は `profile_forward` に残す（出力形式は変わらない。入力再アップロードが
加わるぶん node pass の時間は微増する）。

#### NAF encoder の診断（`trellis-test-pixal3d-cond-tex --naf-ops [--naf-direct-conv]`）

`NafGgmlOpts` に `bool profile_ops = false;` を足す。`naf_upsample_ggml` は `profile_ops` のとき、出力を host へ読み出した
後・`ggml_gallocr_free` の前に、入力（img / rope_cos / rope_sin / win_idx / blk_idx / v_rows）を再アップロードして
`trellis_graph_node_times(…, 2, …)` を回し `trellis_print_op_table` を印字する。本番経路（`naf_upsample` からの呼び出し、
`naf_ggml_opts_for`）は `profile_ops` を立てないので影響しない。
`--naf-ops` は view 0 について `dinov3_encode` → `naf_upsample_ggml(opts.profile_ops = true)` を 1 回だけ回す
（wall-clock 計測とは別プロセス）。`--naf-direct-conv` は `opts.direct_conv = true` を強制し、encoder の im2col vs
direct conv を同じ診断で A/B する（WebGPU が既に使っている経路。数値差は既存の `trellis-test-naf --generic` の範囲）。

#### L2 を回す基準（事前固定）

L1 の結果で、**1 つの ggml グラフの compute が、その段の flow 以外の時間の 30% 以上**を占めたら、そのグラフを L2 の
対象にする。先行計測では NAF（encoder 6 s/view × 4 = 24 s、HR 段の flow 以外 ≈ 47〜53 s の 45%以上、tex 段でも
30%前後）が該当するので **NAF は L2 を実施する**。`dinov3_encode`（1〜2 s/view の見込み）は基準を満たさない見込みで、
満たさなければ L2 は行わず L1 の compute 値を内訳として報告する（issue の "stage/op breakdown" は「段 = L0/L1、
op = 基準を満たしたグラフ」と読む。基準を満たすグラフが無ければ op 別は「該当なし」と書く）。

### 5. 計測手順

共通: 開始前・実行中（30 s おき）・終了後に `ps -Ao pid,pcpu,etime,args -r | head -8`（CPU 上位 8 プロセス、trellis 以外の
Chrome / WindowServer / codex 等も含む）と `grep build-metal/trellis` の両方を取り、他の trellis プロセスが 1 つでも居た計測、
または trellis 以外に CPU 50% 超のプロセスが継続して居た計測は「汚染」と記録して採用しない（他セッションが同じ GPU を使う:
`docs/results/…-metal-flow-profile.md` §4 の熱ドリフトに加え、今回 GPU 版が 3 倍ずれた。macOS はプロセス別 GPU 使用率を
非 root で出せないので CPU 上位で代替し、その限界を結果に書く）。`fresh` = 直前 90 s 以上、GPU を使う trellis プロセスが無い状態。

1. **A/B（L0/L1 相当、標準ハーネス、fresh）**: `trellis-test-pixal3d-cond-tex` に `--mode gpu|host|both`（既定 both、
   従来と同じ）と `--profile-cond` を足す。GPU 版と host 版を**別プロセス**で、**ABBA ×… 最低各 3 回**、条件間 60 s 冷却。
   条件: S=1024/R=64 × T=1024（tex）/ T=512（HR shape）、S=512/R=32/T=512（LR）。中央値と min / max を記録。
2. **L2（診断、fresh）**: `--naf-ops` を T=1024 / T=512 で各 1 回（+ `--naf-direct-conv` で各 1 回）。
3. **E2E（throttled、統合確認）**: `trellis-cli --views docker/linux-webgpu-gate/e2e/views -m <q8_0 MV v1> --res 1024
   --seed 1 --profile-cond` を 1 本（`--profile` は付けない）。E2E は 1 本なので**性能差の定量には使わず**、
   (a) L0 の `accounted` が段合計と ±0.1 s で一致、(b) 各段の L0 値が A/B の fresh 値と比べてどれだけ throttled か、
   (c) GLB の V/F・bbox が既知値（V≈640k / F≈950k）と一致（`--profile-cond` は printf のみなので計算は変わらない。
   新 RoPE の E2E 検証そのものは #28 側の作業で、本 PR では V/F・bbox の一致以上を主張しない）、に使う。
4. 結果は `docs/results/2026-09-20-conditioning-profile.md` に backend / hardware / 重みセット / fixture / N / 実行順 /
   他プロセス有無 / raw log の置き場 / 再現コマンドを明記して保存。

### 6. top 1〜3 の選定規則

順位は「**その候補で消せる時間の上限** = 候補が置き換える排他的ラップの fresh 中央値の合計」で付け、
「**実現見込み**」は A/B で実測できるものだけ数値で書く（測れないものは上限のみ）。
候補 (a) は host 版 − GPU 版の差がそのまま実現見込み。候補 (b)（DINO 共有）は `dino` ラップ × 4 view + `load` ラップの
合計が上限で、実現見込みは同値（計算をそのまま省くだけ）。候補 (c)（NAF encoder のカーネル）は L2 の op 表の
該当 op 合計が上限、`--naf-direct-conv` の差が実現見込み。

| 候補 | 根拠 | 見込みの出し方 |
|---|---|---|
| (a) CLI の 3 段を `pixal3d_cond_slat_gpu(..., &coords)` に切り替える（dense R³ と 4 GiB の host materialize をなくす） | §背景 1 | `--mode host` − `--mode gpu` の fresh 中央値差。数値差は既存テストの tol（L2rel 5e-3）で確認済み（4.9e-8） |
| (b) DINOv3@1024 の patch map を HR shape 段と texture 段で共有（`[1024, 4096]` × 4 view = 64 MB）、dinov3/naf の再 load を減らす | §背景 2 | `dino` ラップ × 4 view + `load dinov3+naf` ラップ |
| (c) NAF encoder のカーネル / レイアウト（im2col → direct conv、f16 im2col の帯域、GroupNorm） | 先行計測 (i)、L2 | L2 op 表 + `--naf-direct-conv` A/B |
| (d) `naf_upsample_ggml` の un-permute と host bilinear | 先行計測 (ii) | `unpermute` / `hr_proj` ラップ。(a) を採ればまとめて消える |
| (e) `tex_decode` / `shape_upsample` の stage 別 | §背景 3・5 | `decode_unet` ラップ。参照実装と同構成なので削れるのはカーネル側だけ |
| (f) quantize の `std::set` | §背景 4 | ラップ値が 1 s 未満なら候補から外す |

## 影響範囲

| ファイル | 変更 |
|---|---|
| `include/trellis_args.h` / `src/trellis_args.cpp` | `profile_cond` フィールド、`--profile-cond` の parse / help、`g_profile_cond` extern |
| `src/dit.cpp` | `g_profile_cond` の定義 |
| `src/trellis_cli.cpp` | `g_profile_cond` 設定 2 箇所、`trellis_run_mv` に L0 `[cond]` ラップ + `accounted` 行 |
| `src/pixal3d_cond.cpp` | L1 ラップ（`#include "trellis_args.h"`） |
| `src/pixal3d_cond_gpu.cpp` | `run_graph` の計時と view ごとの L1 行 |
| `src/naf_gpu.cpp` / `include/naf.h` | lap を `g_profile_cond` でも出す（重複印字なし）、readback / unpermute / free の分離、`NafGgmlOpts::profile_ops` と L2 呼び出し。結果レビュー後に追加: `NafGgmlOpts::generic_groupnorm`（GroupNorm だけ `ggml_norm` 再表現、診断用。`naf_ggml_opts_for` は立てない） |
| `src/shape_decoder.cpp` | `decode_unet` の L1 ラップ |
| `src/dinov3.cpp` | `dinov3_encode` の L1 ラップ |
| `src/graph_dump.cpp` / `include/graph_dump.h` | `trellis_graph_node_times` / `trellis_print_op_table` |
| `src/flow_runner.cpp` | node pass を primitive に置き換え（出力形式不変、入力再アップロードを追加） |
| `src/test_pixal3d_cond_tex.cpp` | `--mode gpu|host|both`、`--profile-cond`、`--naf-ops`、`--naf-direct-conv`。結果レビュー後に追加: `--naf-generic` / `--naf-generic-gn`（lowering の切り替え）、`--naf-compare`（同一入力で native をもう 1 回回して出力差 max\|d\| / L2rel を印字） |
| `README.md` | `--profile-cond` の 1 行 |
| `docs/results/2026-09-20-conditioning-profile.md` | 結果 |

破壊的変更なし。フラグ無しの差分は L0 の `[cond]` printf 行だけ（`[post]` と同じ扱い）。`trellis-server` は同じ
`trellis_run_mv` を呼ぶので同じ行が stdout に出る（既存 `[post]` と同じ）。server の `--profile-cond` は
`trellis-cli` と同じ引数パーサで受けるが、生成を mutex で直列化しているので同時実行の競合は無く、
プロファイル付きリクエストがサーバを占有する時間は printf 分（無視できる）。
WASM: トップレベル CMake は WASM で return し `web/ss/CMakeLists.txt` が専用ソース集合を持つ。`dit.cpp` はその集合に
含まれるので `g_profile_cond` のリンクは成立する見込みだが、build/link で確認する（テスト計画 1）。
CPU / CUDA / Vulkan / WebGPU: 変更は printf と `steady_clock` と公開 API（`ggml_graph_add_node` 等、既存プロファイラで使用中）
のみ。native CPU ビルドで確認し、CUDA / Vulkan は CI（Linux / Windows）に任せる旨を明記する。

## テスト計画

1. **ビルド行列**: `build-metal`（Release、`trellis-cli` / `trellis-server` / `trellis-test-pixal3d-cond-tex` /
   `trellis-test-pixal3d-cond-slat` / `trellis-test-dit-bench`）、native CPU ビルド（`-DGGML_METAL=OFF`、`trellis-cli` と
   `trellis-server`）、native WebGPU（既存 `build-webgpu`、`trellis-cli`）、WASM は full / partial の両 target
   （`build-wasm` / `build-wasm-rg` の既存構成で link まで）。CUDA / Vulkan は CI。
2. **引数**: `--profile-cond` / `--mode` / `--naf-ops` の parse と `--help` の表示。未知の値は既存と同じエラー経路。
3. **非破壊**（計算を足さないことの確認）: `trellis-test-pixal3d-cond-tex --mode both --profile-cond` の `COND_TEX: PASS`
   と、`--profile-cond` 無しの同条件で `--save-prefix` に保存した `global` / `proj` の npy が **bit 一致**すること
   （printf しか足していないので一致する。しなければラップの入れ方が計算を動かしている）。
   `trellis-test-pixal3d-cond-slat`（S=512 host 版 vs PyTorch fixture）も回す。
4. **L2 の非破壊**: `--naf-ops` は診断専用で本番経路に入らない。primitive の契約テストとして `trellis-test-dit-bench --profile`
   の出力形式が変わらないこと、node pass 後の次 forward の結果が profile 無しの forward と bit 一致すること
   （既存の「whole を先に読む」契約 + 入力再アップロードの確認）。
5. **L0 の照合**: E2E 1 本で、各段の `[cond] … accounted` が段合計（`HR shape SLAT (…s)` 等）と ±0.1 s で一致すること。
   `--profile-cond` 無しの E2E ログとの比較は「行の接頭辞集合の差が `[cond]` だけ」を機械検査する（時間値は毎回変わるので
   単純 diff はしない）。GLB は `tools/glb_metrics.py` で V/F・bbox・成分数・finite 座標・PBR テクスチャの有無を出し、
   同一 commit の新 RoPE E2E（`--profile-cond` 無し、同入力・同 seed）と比較する。加えて front/back/left/right/top の
   5 視点レンダ（`tools/render_glb*.py`）を並置して差が無いことを目視する（printf しか足していないので差は出ない想定。
   出たらラップの入れ方が計算を動かしている）。
6. **反復と順序**: A/B は §5 の ABBA ×3・別プロセス・`ps` スナップショット付きで行い、結果ドキュメントに実行順を残す。

## 代替案

| 案 | 却下理由 |
|---|---|
| いきなり CLI を `pixal3d_cond_slat_gpu` に切り替えて E2E で before/after を取る | issue の acceptance は「最適化前にボトルネックを特定」。切り替えは follow-up PR の候補 (a) として、A/B の数値で見込みを出す |
| 本番経路（`naf_upsample_ggml` / `cond_slat_gpu_chunked`）の中で op 再実行 | §背景 7: gallocr の入力再利用と永続バッファへの cpy のため、後続の本番計算を汚し得る。診断バイナリに限定する |
| `--profile` 1 本で flow と conditioning を同時に取る | flow の side-pass（HR で +100 s の GPU 負荷）が後続 conditioning の熱状態を変える。別フラグ・別実行にする |
| Xcode Instruments / `xctrace` | host 側（un-permute・bilinear）の CPU サンプル比率・page fault の補助証拠としては有用。L1 の壁時計で host 側が支配的と出た区間の反証確認に限って使う（主計測にはしない） |
| env `TRELLIS_PROFILE_COND` | CLAUDE.md の「挙動を変えるものはフラグ」 |

## レビュー対応（codex 1 周目 = 差し戻し → 本改訂版。`docs/reviews/2026-09-20_conditioning-profiler_review.md`）

| # | 指摘（A = 実装を止める） | 対応 |
|---|---|---|
| A1 | 同じ `g_profile` が flow side-pass と cond 計測を同時に有効化し、cond は熱状態が変わった後に測られる | `--profile-cond` / `g_profile_cond` を分離（§1）。E2E の cond 計測は `--profile` を付けない（§5-3） |
| A2 | chunked の `naf_enc` 再実行は `P_pooled` を再書き込みし、gallocr の入力再利用で元入力の保証も無い | 本番経路での op 再実行を全廃（§4、§背景 7）。L2 は診断バイナリで `naf_upsample_ggml`（永続バッファ無し）のみ、入力を再アップロードしてから回す |
| A3 | 3 引数の共通関数では入力復元・出力保護・既存出力形式の維持ができない | 測定 primitive（`trellis_graph_node_times`: passes + reupload callback、契約をコメントに明記）と印字（`trellis_print_op_table`）に分離。flow profiler は role/block 集計を自前に残す（§4） |
| A4 | 壁時計と op side-pass を同一実行で回すと外側の時間が汚れる | L0/L1（壁時計）と L2（op）を別プロセスに分離（概要の 3 層、§5） |
| A5 | `[cond]` 合計と段合計の照合式が成立しない（flow が入っていない、未計上区間が多い） | L0 を隙間なしの排他的ラップにして flow も 1 ラップにする → Σlaps = 段合計（`accounted` 行で機械確認、§2）。未計上だった `out.proj.assign` / `from_latent` / `output_layer` / free を L1 に追加、`sparse_c2s` 内部は c2s の inclusive 値と明記（§3） |
| A6 | A/B の順序効果と反復不足 | `--mode gpu|host` で別プロセス、ABBA ×3、中央値と min/max、`ps` を前・中・後で取る（§5-1） |
| A7 | 非破壊ゲートが V/F・bbox だけで弱い | `--profile-cond` は printf のみなので `global` / `proj` npy の bit 一致で確認（テスト 3）。L2 は本番経路外。flow の latent / 複数視点レンダの比較は「計算を変えるプロファイル」が本 PR に無いので対象外と明記 |
| A8 | L2 の発火条件が未決定で acceptance と衝突 | 「1 グラフの compute が段の flow 以外の 30% 以上」で固定、NAF は該当（§4）。該当なしの場合の書き方も定義 |
| B1 | WASM は `trellis_core` ではなく `web/ss/CMakeLists.txt` の専用集合 | 影響範囲を訂正、build/link をテスト 1 に追加 |
| B2 | unified memory を「host 往復」と書くのは不正確 | 「backend readback API による host materialize とページ圧」に修正（§背景 1） |
| B3 | テストバイナリの stdout はブロックバッファ、`TRELLIS_DBG_NAF` との重複印字 | L1 は行ごとに `fflush`、重複印字は 1 回に（§3） |
| B4 | 全 backend へのコンパイル影響 | ビルド行列をテスト 1 に追加（Metal / native CPU / WASM、CUDA / Vulkan は CI） |
| B5 | `--profile` 無しログの単純 diff はテストにならない | 接頭辞集合の機械検査に変更（テスト 5） |
| B6 | top 1〜3 の選定規則が無い | 「消せる時間の上限」と「実現見込み」を分けて定義（§6） |

再確認（2 周目、同レビューファイル末尾）: A1〜A5・A8・B1〜B3・B5・B6 は解消、A6 / A7 / B4 が「部分的」→
A6: 監視を trellis 以外の CPU 上位プロセスにも広げ、macOS で GPU 別使用率が取れない限界を明記（§5）。
A7: GLB の構造検査（`tools/glb_metrics.py`）と 5 視点レンダ比較をテスト 5 に追加。B4: WebGPU ビルドと WASM full / partial の
両 target をテスト 1 に追加。周回上限（1 周 + 再確認 1 回）に達したので、これらを承認条件として取り込んだうえで実装に入る。
