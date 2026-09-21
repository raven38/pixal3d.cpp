# Pixal3D conditioning（NAF@1024 / HR cond / upsample / PBR decode）の段別プロファイル — 2026-09-20（issue #33）

環境: macOS 26.5 / MacBook Pro M4 Max（Mac16,6、64 GB）/ Metal（`MTL0`）。重み `pixal3d-q8_0 v1`（MV、
`~/nfs/weights/pixal3d/gguf-q8_0/`、dinov3 も q8_0）。入力は cyclops 4 view（`docker/linux-webgpu-gate/e2e/views` +
`transforms.json`、`mesh_scale` 1.0）。コード: `perf/cond-profile`（`perf/metal-rope-layout` 4853a55 = 新 RoPE + `--profile`
の上に 151cd21 / 16c29cd / db0623b。§1 の 1 回目・§5 の E2E は db0623b のバイナリ、§1 の再計測（`clean2/` / `clean3/`）と §4 の
`--naf-ops` は診断フラグを足した c2bff7f のバイナリ。差分は printf と診断専用の分岐だけで、計測対象の経路は同じ）。設計と計測手順: `docs/design/2026-09-20-conditioning-profiler.md`（codex レビュー
`docs/reviews/2026-09-20_conditioning-profiler_review.md`）。

raw ログ・ps スナップショット・レンダ・計測スクリプトは `docs/results/2026-09-20-cond-profile/`（`clean/` = 1 回目の A/B と
`--naf-ops`、`clean2/` = 実行ごとゲートの再計測（A/B + naf-ops）、`clean3/` = 前窓検査後の置き換え、`nth/` = ggml-metal `nth` の
反証テスト、`firstlook/` = 数値一致の出典、`e2e/` = E2E と参照ログ、`scripts/` = `ab_clean2.sh`（gate / snapshot / run のテンプレ）/
`ab_clean3.sh` / `ab_clean.sh` / `ab_rerun_lr.sh` / `e2e_profile_cond.sh` / `nth_experiment.sh` + `nth_patch.py` / `classify_runs.py` /
`summarize_ab.py` + `accepted_ab.txt` / `verify_e2e.py` / `render5.py`）。再現は `scripts/ab_clean2.sh`（実行ごとに 90 s ゲート、
A/B + naf-ops、静かなら約 1 時間）→ `scripts/e2e_profile_cond.sh`（E2E、約 30 分）→ `python3 scripts/classify_runs.py <dir>/ps_snapshots.log`
で採用判定 → `python3 scripts/summarize_ab.py <dirs> --accept scripts/accepted_ab.txt` / `python3 scripts/verify_e2e.py e2e/e2e_cond.log e2e/ref_rope_e2e.log`。

計測の 3 層（設計の L0 / L1 / L2）は**別の実行**で取った。同じ表の中でだけ足してよい（`[cond-v]` は `[cond]` の内側、
`[prof]` の node pass は isolated）。

### 汚染の判定（事前基準 + 事後の例外 1 つで機械分類。1 回目の版は「全実行 clean」と書いていたが誤りだったので差し替え）

設計 §5 の事前基準は「(a) 他の trellis プロセスが 1 つでも居た計測、(b) trellis 以外に CPU 50% 超のプロセスが継続して居た計測は
採用しない」。各実行の前・中（30 s おき）・後の `ps` スナップショット（`clean/ps_snapshots.log`、`clean2/ps_snapshots.log`、
`e2e/ps_snapshots.log`）を `scripts/classify_runs.py` で機械的に分類した（(b) の「継続」= その実行のスナップショットのうち
2 つ以上で 50% 超）。codex の反証レビュー（`docs/reviews/2026-09-20_conditioning-profile_results_review.md`）が、1 回目の
版で「clean」とした実行に他セッションの `trellis-test-decimate-bench` / `post-replay` / `build-metal-fa/trellis-test-*` が
重なっていることを指摘したのを受けて、**採用集合を全部作り直した**（§1 の表は再分類後 + 再計測分）。

- 1 回目の A/B（`clean/`、03:01〜03:34、`ab_clean.sh`）: 静穏ゲートをスクリプト開始時の 1 回しか掛けておらず、途中から
  別セッションの CPU ベンチ（`trellis-test-decimate-bench`、#29）・`post-replay`・FA テスト（#30）が断続的に重なった。
  26 実行中 **clean 14 / 汚染 12**（内訳は下表）。`--naf-ops` の 4 本は全部汚染（post-replay 196〜246% CPU、decimate-bench 100%）。
- 再計測（`clean2/`、05:06〜06:10、`ab_clean2.sh`）: **実行ごとに** 90 s の静穏ゲートを掛け、事後に同じ分類器で判定（18 本全部 clean）。
- 前窓の検査（設計の fresh = 直前 90 s 以上 GPU を使う trellis 無し）: 分類器に「before の 90 s 前までのスナップショットに
  他の trellis バイナリが居たか」を足して全実行を再判定したところ、1 回目の採用分のうち 4 本（T1024 host 03:10、LR gpu 03:22、
  LR host 03:28、再実行の T512 gpu 04:41）が前窓に他セッションの trellis（decimate-bench / FA テスト）を持っていたので不採用にし、
  置き換え計測（`clean3/`、06:32〜09:17、`ab_clean3.sh`、実行ごとにゲート）で 6 本回した → 4 本採用、2 本（T512 gpu 07:01 = 30.1 s、
  T1024 host 08:50 = 124.7 s）は実行中に共有チェックアウトの trellis テストが入り除外。**採用した全実行は、実行中の
  スナップショットで分類器を通り、1 回目の分は前窓（前の実行の after / periodic のスナップショット）も通る**。`clean2/` /
  `clean3/` / `nth/` の前窓は `gate()`（10 s おきの `ps`、90 s 連続で他 trellis 無しになるまで待つ）が保証するもので、ゲート中の
  `ps` は保存していないので分類器では事後検証できない（次回はゲートのサンプルもログに残す）。
- **事後の例外（事前基準からの逸脱）**: macOS の `suggestd`（CoreSuggestions、pid 953、elapsed 2 日以上）が 1 コア 92〜98% で
  上位に居る時間帯が長い。分類器の出力で数えると、採用した `clean2/` 18 本・`nth/` 4 本は全スナップショットで 50% 超、`clean3/` は
  6 本中 5 本、1 回目の `clean/` は 23 本中 10 本が全スナップショット・5 本が一部・**8 本（03:03〜03:16）は上位 8 に出ない**。
  つまり「全実行で一様」ではなく、1 回目の host 6 本のうち 5 本は suggestd 無しで測っている。基準 (b) を文字どおり適用すると
  再計測分が全部不採用になるので、**suggestd だけは「1 コアの CPU 負荷で GPU を使わない常駐プロセス」として基準から外した**
  （device 版は GPU バウンドで影響は小さく、host 版の CPU 部分には効きうる。§1 の host 版の幅の一因になりうる）。
  それ以外の 50% 超プロセス（git、他セッションの codex 等）は基準どおり扱う。
- 限界: macOS はプロセス別 GPU 使用率を非 root で出せないので、GPU 競合の検出は「trellis バイナリの有無」と CPU 上位の代替監視。
  ユーザーの Chrome（GPU プロセス 35%）等は E2E（§5）の条件に記す。

| 1 回目（`clean/`） | clean | 汚染（理由） |
|---|---|---|
| T1024 gpu ×3 | 03:01（31.3） | 03:07（31.4、他セッションの git > 50% ×2）、03:09（33.2、decimate-bench） |
| T1024 host ×3 | 03:03（74.1）、03:05（73.3）、03:10（71.9） | — |
| T512 gpu ×3 | 03:12（33.3） | 03:17（28.1、decimate-bench）、03:19（27.9、decimate-bench --legacy） |
| T512 host ×3 | 03:14（42.4）、03:16（45.4） | 03:20（41.9、共有チェックアウトの trellis テスト） |
| LR gpu ×3 | 03:22（7.0） | 03:26 / 03:27（segfault、decimate-bench とも重なる） |
| LR host ×3 | 03:23（13.6）、03:25（13.8）、03:28（13.7） | — |
| naf-ops ×4 | — | 全部（post-replay / decimate-bench / 共有チェックアウトの trellis） |
| 再実行 04:38〜（db0623b） | LR gpu 04:38（7.0） | LR gpu 04:40（22.9、`build-metal-fa/trellis-test-*`）、T512 gpu 04:41（27.7、前窓に同上） |
| 前窓で不採用にした 1 回目の分 | — | T1024 host 03:10（71.9）、LR gpu 03:22（7.0）、LR host 03:28（13.7） |
| 再計測 `clean2/`（05:06〜06:10） | 18 本全部 | — |
| 置き換え `clean3/`（06:32〜09:17） | T1024 host 06:32（66.8）、LR host 06:57（13.5）、LR gpu 06:59（7.0）、LR host 09:16（13.5） | T512 gpu 07:01（30.1）、T1024 host 08:50（124.7）: 実行中に共有チェックアウトの trellis テスト |

## 0. 結論（先に）

1. **CLI の 3 つの SLAT conditioning は host 経路（dense R³ + 4 GiB の NAF map を host で転置・bilinear）を使っており、
   既存の device 常駐 sparse 経路 `pixal3d_cond_slat_gpu(..., &coords)` に切り替えると、合成 fixture の fresh A/B で
   128.9 s → 66.0 s、**−63 s（中央値、tex −42.1 / HR −14.2 / LR −6.6、n=3〜4）。host 版の tex は実行間変動が大きく
   （66.8〜74.1、1 view の GPU compute が +3.7 s になるスパイク）、最速の host run を使う保守ケースでは −56 s**。数値差は global bit 一致・proj L2rel 4.9e-8（既存テストの tol 5e-3）。
   **PR #34（`feat/desktop-vram-8gb`、2026-09-20、CUDA 8 GB 向け）が HR / tex の 2 段を同じ経路へ切り替え済み**
   （4090 で tex 段 116 → 55 s）。本計測はその Metal 側の裏付け（HR + tex で −56 s、保守ケース −50 s）で、#34 の対象外の LR（S=512、−6.6 s、実 N では未測）が残る。
2. **残る device 側の支配項は NAF encoder の `GROUP_NORM`**（S=1024 の 1 view で 3.62 s / 6.27 s = 58%、n=2 で一致）。
   ggml-metal の `ggml_metal_op_group_norm` は group ごとに 1 threadgroup × **32 スレッド固定**（`nth = 32`、スケールする
   ループはコメントアウト）で、512 MB のテンソルを 8 × 32 = 256 スレッドで 3 パス舐めるので 1 op 453 ms。
   **GroupNorm だけ**を既存の `ggml_norm` 再表現（`[W*H*C/8, 8]` reshape、WebGPU 用の generic lowering の一部）に変えた対照で
   NAF グラフが **6.27 → 2.71 s / view（T=1024）、6.00 → 2.45 s（T=512）**、full generic は 2.60 / 2.68 s。利得の 97% が GroupNorm
   単独。出力差は native 比 L2rel 1.2e-4（max|d| 1.2e-2、既存 NAF テストの tol 3e-3 の内側、bit 一致ではない）。
   E2E 換算の実測分は tex 4 × 3.67 + HR 4 × 3.55 = **−29 s**（LR の S=512 は未測、上限のみ）。(1) と独立に効き、host 経路にも同じだけ効く。
   `nth` だけを（group_norm の dispatcher に限定して）1024 にした反証テスト（§4.3）で GROUP_NORM 453 → 16 ms/op・グラフ
   6.27 → 2.78 s となり、原因は確定。
3. その次は `IM2COL`（1.72 s / view、27%。K に依らず ≈210 ms/op の dispatch 律速で、1×1 conv にも im2col を使っている）と
   `CONT`（0.63 s、レイアウトの往復）。上限 ≈ 20 s、実現見込みは未測。`ggml_conv_2d_direct` は Metal では逆に +3.3 s / view
   （CONV_2D 527 ms/op、clean 実測）で採らない。
4. DINOv3@1024 は 0.8 s / view で、HR と tex で二重に走っている分を共有しても −3 s 程度。優先度は低い。
5. E2E での段別内訳と tex_decode / shape_upsample の取り分は §5。

## 1. fresh A/B: host 版（CLI の現行経路）vs device 版（sparse）

`trellis-test-pixal3d-cond-tex <dinov3> <naf> <views> 0 --mode {gpu|host} --profile-cond` を**別プロセス**で回し、
条件間 60 s 冷却、N=17 489（`synth_tex_fixture/hr_coords.npy` を f32 化。E2E の cyclops は 17 612〜17 614 で 1% 差）。
LR は `--stride 3`（合成 3 872 token、実 E2E は 4 438）。1 回目（`clean/`、ABBAAB、ゲートは開始時 1 回）と再計測
（`clean2/`、実行ごとに 90 s ゲート）を合わせ、冒頭の事前基準で clean と判定した実行だけを採用（採用リスト
`scripts/accepted_ab.txt`、集計 `python3 scripts/summarize_ab.py clean clean2 clean3 --accept scripts/accepted_ab.txt`（出力 `summary_accepted.md`））。

| 条件（4 view） | device 版 中央値（min / max、n） | host 版 中央値（min / max、n） | 差 | 比 |
|---|---:|---:|---:|---:|
| tex: S=1024 R=64 T=1024 | **31.2**（31.1 / 31.3、n=4） | **73.3**（66.8 / 74.1、n=3） | −42.1 s（min 差 −35.6） | 2.35× |
| HR shape: S=1024 R=64 T=512 | **27.8**（27.7 / 33.3、n=3） | **42.0**（41.6 / 45.4、n=4） | −14.2 s | 1.51× |
| LR shape: S=512 R=32 T=512 | **7.0**（7.0 / 7.0、n=3） | **13.6**（13.5 / 13.8、n=4） | −6.6 s | 1.94× |
| 合計（中央値） | 66.0 | 128.9 | **−62.9 s** | |

個別値（実行順）: tex device 31.3 / 31.1 / 31.2 / 31.2、host 74.1 / 73.3 / 66.8。HR device 33.3 / 27.8 / 27.7、
host 42.4 / 45.4 / 41.6 / 41.7。LR device 7.0 / 7.0 / 7.0、host 13.6 / 13.8 / 13.5 / 13.5。

**host 版 tex の幅（66.8〜74.1）の正体**: `[cond-v]` の view ごとのラップを開くと、74.1 s と 73.3 s の実行はそれぞれ 1 view だけ
NAF の `graph compute` が **10.0 s**（他の view は 6.29 s、+3.7 s の同じ大きさのスパイク）で、un-permute も 5.5〜6.5 s と
ばらつく。66.8 s の実行は 4 view とも compute 6.27〜6.28 s（`--naf-ops` の whole graph と同じ）、un-permute 5.3〜5.7 s。
スパイクは device 版 4 本（31.1〜31.3）には出ず、HR device の 33.3（他の 2 本 +5.6 s）も同種と思われる。原因は特定できない
（分類器に掛からない未観測の GPU 外乱、熱、OS 状態を含む実行間変動。macOS ではプロセス別 GPU 使用率が取れない）。
**中央値 73.3 はこの変動込み、最も静かな 1 本が 66.8**。感度: tex host に中央値を置くと 3 段合計の差は −62.9 s、最速の 1 本
（66.8）を置く保守ケースは −56.4 s、最遅（74.1）を置くと −63.7 s（統計的な区間ではなく、tex host の 1 本を差し替えた感度）。
汚染と判定して除外した実行の値（参考、採用しない）: tex device 31.4 / 33.2、host 71.9 / 124.7、HR device 28.1 / 27.9 / 27.7 / 30.1、
HR host 41.9、LR device 7.0 / 22.9、LR host 13.7。

**再現条件の注記**: 1 回目（`clean/`）は db0623b 以前のバイナリ（LR device は 16c29cd、他は 151cd21 相当。差分は printf のみ）、
`clean2/` / `clean3/` は c2bff7f（診断フラグ追加、`--mode` 経路は不変）。fixture は合成 N（tex / HR は
E2E の N と 1% 差、LR は 14.6% 少ない）。device 版で N 非依存なのは DINO / NAF encoder の compute（view 時間の 98%）で、
projection の tap 作成・upload・`get_rows` gather・accumulator（`pixal3d_cond_gpu.cpp` L497 / L529 / L580 の N3 サイズの
テンソル）は N 依存。LR の N=3 872 → 4 438 での影響量は**未測**（host 版も dense R³ なので N 依存部分は小さいが、差 −6.6 s を
実 N でそのまま主張はしない）。
**数値の一致**（結論 1 の根拠）は同一プロセスの gpu → host 比較（`firstlook/ab_T1024.log` / `ab_T512.log`、時間は汚染されるが
数値比較は汚染と無関係）: `global max|d|=0 L2rel=0`、`proj[lr||hr] max|d|=5.7e-6 mean|d|=9.9e-9 L2rel=4.9e-8 cos=1.0`（T=1024）、
T=512 も同じ（`L2rel=4.85e-8`）。これは**合成 fixture 上の host 版 vs device 版の一致**であり、実 E2E 座標で CLI を device
経路に切り替えた統合一致は PR #34 側（CUDA、proj L2rel 7.3e-8）の確認と、Metal では切り替え PR で行う。

汚染ありの最初の 1 回（02:18〜02:24、他セッションの E2E texture flow と同時、同一プロセス内で gpu → host）は tex で
device 139.2 s（最遅 view 78 s）/ host 90.5 s、HR で 32.7 / 46.3 s だった。**GPU を共有した瞬間の値は 3〜4 倍ずれる**。

### 1a. db0623b での再実行（`ab_rerun_lr.sh`、04:38:55 ゲート通過 → 04:41:54）と再計測（`ab_clean2.sh`、05:06〜05:34）

| 実行 | 結果 | 判定 |
|---|---:|---|
| LR gpu 04:38:55 | 7.0 s（view 1.71 / 1.71 / 1.72 / 1.72） | 採用 |
| LR gpu 04:40:03 | 22.9 s（最遅 view 9.7） | **除外**: 開始時の ps に別セッションの `build-metal-fa/trellis-test-*`（#30 の FA テスト、25 s 経過） |
| T=512 gpu 04:41:26 | 27.7 s（view 6.87 / 6.88 / 6.88 / 6.91） | 採用 |
| 再計測 8 本（05:06〜05:34、実行ごとにゲート） | tex device 31.1 / 31.2 / 31.2、HR device 27.8 / 27.7、LR device 7.0、HR host 41.6 / 41.7 | 全部採用（分類器で clean、前窓も ok） |
| 置き換え 6 本（06:32〜09:17、実行ごとにゲート） | tex host 66.8、LR host 13.5 / 13.5、LR device 7.0 | 採用。T512 device 07:01（30.1）と tex host 08:50（124.7）は実行中に他 trellis が入り除外 |

## 2. host 版の view 内訳（`[cond-v]`、L1、全実行・全 view の中央値）

| 条件 | dino | lr_proj（host bilinear、dense R³） | naf（合計） | hr_proj（host bilinear、dense R³） | accum | view 合計 |
|---|---:|---:|---:|---:|---:|---:|
| tex T=1024 | 0.81 | 0.63 | **13.39** | 2.12 | 0.07 | 17.22 |
| HR T=512 | 0.80 | 0.64 | **7.60** | 1.36 | 0.05 | 10.47 |
| LR S=512 | 0.12 | 0.05 | **2.95** | 0.23 | 0.01 | 3.38 |

`naf` の内訳（`naf_upsample_ggml` の lap、ms、中央値）:

| 条件 | graph alloc | inputs | **graph compute（device）** | readback（`tensor_to_f32`） | **unpermute（host、stride-T² 転置）** | free |
|---|---:|---:|---:|---:|---:|---:|
| tex T=1024（map 4 GiB） | 208 | 622 | **6 293** | 481 | **5 707** | 12 |
| HR T=512（map 1 GiB） | 108 | 165 | **6 010** | 73 | 1 224 | 3 |
| LR S=512 T=512（map 1 GiB） | 51 | 143 | 1 543 | 68 | 1 148 | 3 |

読み取り（tex、73 s の host 版、中央値ベース）: device の NAF compute 25 s（34%）、**host の un-permute 23 s（31%）**、host bilinear
（lr + hr、262 144 token × 1024 ch × 4 tap、double）11 s（15%）、NAF の alloc + inputs + readback 5.4 s（7%）、DINO 3.2 s
（4%）。device 版はこのうち un-permute / bilinear / readback をそのまま消す（sparse な tap を `get_rows` で device 上で集める）。
NAF の compute は T=1024 と T=512 でほぼ同じ（6.3 vs 6.0 s）= **encoder が支配的で attention は 0.3 s 程度**（§4）。

## 3. device 版（sparse）の view 内訳（`[cond-v]`、L1）

tex（T=1024、分割グラフ版 `cond_slat_gpu_chunked`、採用 4 実行 × 4 view の中央値、秒）:

| host taps | chunk_prep | dino compute | **naf_enc compute** | naf_qk compute（16 stripe） | naf_attn compute（16 chunk、skip 0） | build + upload + free 合計 | view 合計 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.00 | 0.75 | **5.87（77%）** | 0.20 | 0.37 | 0.40 | 7.62 |

HR（T=512）と LR は単一グラフ経路（`pixal3d_cond_slat_gpu` の T ≤ 512 分岐: DINO + NAF + sparse tap 収集を 1 グラフ）。
§1a の再実行（db0623b）での view ごとの内訳（秒）:

| 条件 | host taps | build+alloc | upload | **compute** | free | view 合計 | graph（gallocr） | nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HR S=1024 R=64 T=512 | 0.00 | 0.10〜0.14 | 0.01 | **6.76** | 0.00 | 6.87〜6.91 | 5583 MB | 1652 |
| LR S=512 R=32 T=512 | 0.00 | 0.05〜0.07 | 0.01 | **1.65** | 0.00 | 1.71〜1.72 | 2836 MB | 1651 |

どちらも 98% が compute で、view 間のばらつきは 0.01 s。HR の 6.76 s は「DINO@1024 0.8 + NAF S=1024 T=512 ≈ 6.0（§2 の host 版
naf compute と同じ）」に一致するので、T=512 でも device 側の支配項は NAF encoder（§4）。LR（S=512）は NAF が 1.5 s 程度で、
host 版（3.40 s/view）との差 1.7 s は un-permute 1.16 + bilinear 0.28 + readback 0.07 + alloc/inputs 0.2。

## 4. L2: NAF グラフの op 別（`--naf-ops`、診断バイナリ、view 0、node 再実行 = ISOLATED、fresh）

L2 の発火基準（設計 §4「1 グラフの compute が段の flow 以外の時間の 30% 以上」）: HR は NAF compute 6.0 s × 4 view = 24 s /
flow 以外 48.9 s = **49%**、tex は 25.2 s / 93.0 s = 27%（同じグラフなので HR の発火で兼ねる）、LR は 6.2 / 14.4 = 43%。
→ NAF encoder グラフだけが対象。DINO（0.8 × 4 = 3.2 s、7%）と decoder（別グラフ、issue 対象外）は発火しない。

`trellis-test-pixal3d-cond-tex ... --naf-ops --views 1 [--naf-generic | --naf-generic-gn | --naf-direct-conv] [--naf-compare]`。
1 回目（`clean/nafops_*`、03:29〜03:34）は 4 本とも他セッションの `post-replay` / `decimate-bench` と重なって**汚染**
（値は下と一致していたが採用しない）。以下は再計測（`clean2/nafops *`、05:36〜06:1x、実行ごとにゲート、全部 clean）。
whole = 本番と同じ 1 グラフの compute、node pass = 1 node ずつ再実行（入力を再アップロードしてから）。両者の比は ×1.00〜1.01。
**op 別の秒数と % は node pass（ISOLATED）の合計を分母にした値**で、whole graph とは 0.02〜0.04 s ずれる。

### 4.1 S=1024, T=1024（whole graph、秒。n=2 は 2 本の値、差 ≤ 0.01 s）

| 構成 | whole graph | GROUP_NORM / NORM | IM2COL | CONT | CONV_2D | MUL_MAT(f16) | 出力差 vs native（4 GiB 全要素） |
|---|---:|---:|---:|---:|---:|---:|---|
| **native**（im2col + `ggml_group_norm` / `pad_reflect_1d` / `pool_2d`、本番の Metal 設定） | **6.27 / 6.27** | **3.62（57.5%）**, 8 op × 453 ms | 1.72（27.2%）, 10 op × 172 ms | 0.63, 28 op | — | 0.11 | — |
| **GroupNorm だけ generic**（`--naf-generic-gn`: `ggml_norm` over `[W*H*C/8, 8]`、pad / pool は native） | **2.71 / 2.72** | 0.06, 8 op × 8 ms | 1.71 | 0.63 | — | 0.11 | max\|d\| 1.18e-2（値 −1.85、refmax 24.5）, mean\|d\| 7.7e-5, **L2rel 1.2e-4** |
| **full generic**（`--naf-generic`: 上 + pad を concat、pool を sum_rows。WebGPU の設定） | **2.60 / 2.60** | 0.06 | 1.72 | 0.49, 38 op | — | 0.11 | 同じ max\|d\| 1.18e-2（同じ要素）, L2rel 1.2e-4 |
| direct conv（`--naf-direct-conv`: `ggml_conv_2d_direct`、GroupNorm は native） | **9.56** | 3.62 | — | 0.47 | **5.27**, 10 op × 527 ms | — | max\|d\| 1.08e-2, L2rel 9.6e-5 |

- **GROUP_NORM 単独で 3.56 s / view**（6.27 → 2.71）。full generic との差 0.11 s が pad / pool の再表現分（CONT が 0.63 → 0.49）。
  出力差の統計は GroupNorm だけ変えた構成と full generic で max|d| の値も位置も一致する（同じ要素 327 678 159 で 1.1848e-2）ので、
  数値差は GroupNorm の再表現（reduction 順の違いが 5 層 + attention を通って増幅）に由来し、pad / pool の再表現は出力を変えて
  いないと**強く示唆される**（2 つの出力の直接比較 max|d|=0 は未実施）。L2rel 1.2e-4 は既存 `trellis-test-naf` の出力 tol
  3e-3 の内側だが、**bit 一致ではない**（1 回目の版の「厳密に同値」は数式の話で、浮動小数点ではこの値。`naf.h` / `naf_gpu.cpp` の
  コメントも「数学的には同じ、縮約順は異なる」に直した）。
- direct conv は Metal では im2col + GEMM（1.72 + 0.11）の 2.9 倍（5.27 s）で、グラフ全体 +3.3 s。**採らない**（PR #34 は
  VRAM 削減のために CUDA だけで有効化しており、Metal では im2col のままが正しい）。
- gallocr 11 329 MB（im2col / generic とも）、direct conv は活性が小さい（PR #34 の動機）。

### 4.2 S=1024, T=512

| 構成 | whole graph | GROUP_NORM / NORM | IM2COL | CONT | SUM_ROWS | 出力差 vs native |
|---|---:|---:|---:|---:|---:|---|
| native | **6.00** | 3.62（60.1%）, 8 op × 453 ms | 1.72 | 0.41 | — | — |
| **GroupNorm だけ generic** | **2.45** | 0.06 | 1.72 | 0.41 | — | max\|d\| 1.04e-2, mean\|d\| 7.5e-5, L2rel 1.2e-4 |
| full generic | 2.68 | 0.06 | 1.72 | 0.31 | 0.33（3 op、stride-2 pool の再表現。T=1024 では pool が無い） | — |

T=512 では pool の `sum_rows` 再表現（0.33 s）が native の `pool_2d` より高いので、**GroupNorm だけ lowering する構成が最速**
（2.45 s、−3.55 s/view）。T=1024 では pool が無く、concat の pad が `pad_reflect_1d` + CONT より 0.1 s 安いので full generic が
僅かに速い（2.60 vs 2.71）。どちらも GroupNorm の置き換えが利得のほぼ全部。

### 4.3 GROUP_NORM の正体と、`nth` を変えた反証テスト

- `thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp::ggml_metal_op_group_norm` は `dispatch_threadgroups(ngrp, 1, 1, nth, 1, 1)`
  で **`nth = 32` 固定**（`ne00` に応じて増やすループはコメントアウト）。カーネル `kernel_group_norm_f32` は group
  （`gs = ne00*ne01*ceil(ne02/ngrp)` = 1024×1024×16 = 16.7M 要素、64 MB）を 1 threadgroup が 3 パス（sum / 分散 / 正規化）で
  舐める。8 group × 32 thread = 256 スレッドで 512 MB → 450 ms。カーネル自体は `ntg > 32` のとき SIMD group 間を 32 float の
  threadgroup メモリで縮約する経路を持っている（smem = 32 floats、最大 1024 thread）。
- CUDA（`ggml-cuda/norm.cu::group_norm_f32_cuda`）も group ごとに 1 block だが 1024 thread（`group_norm_f32<1024><<<num_groups, 1024>>>`）
  で Metal の 32 倍、Vulkan も 1 workgroup / group の同型。**他 backend は未計測**。
- **反証テスト（結果: 仮説どおり）**: `nth` だけを 32 → 1024（pipeline の上限まで倍々、他は変えない）にした診断パッチで
  同じ `--naf-ops` を回した。**1 回目の試み（`nth/`、06:13〜06:25）は `int nth = 32; // SIMD width` の全置換で、同じ行を持つ
  9 つの dispatcher（sum / sum_rows / set_rows / soft_max / l2_norm / group_norm / norm / argmax / tri）が一緒に変わっていた**
  （codex の再確認で発覚。NAF グラフでは SOFT_MAX が 128 ms に悪化する副作用が混ざり、whole 2.90 s）。
  **2 回目（`nth2/`、09:28〜09:32）は `ggml_metal_op_group_norm` の文脈で一意に置換し、適用後の差分が 1 ハンクだけであることを
  検査してから計測**（`nth2/patch_diff.txt`、`scripts/nth_patch.py check`）。submodule には残さず実験後に復元・再ビルド済み。

  | 構成 | whole graph | GROUP_NORM | 備考 |
  |---|---:|---:|---|
  | native、`nth = 32`（現状） | 6.27 / 6.27 s（T=1024）、6.00 s（T=512） | 3.62 s, 8 op × 453 ms | — |
  | native、`nth = 1024`、**group_norm だけ**（`nth2/`） | **2.78 / 2.78 s（T=1024）、2.51 s（T=512）** | **0.13 s, 8 op × 16 ms（28 分の 1）** | 他の op は native と同じ（IM2COL 1.71、CONT 0.63、MUL_MAT 0.11） |
  | （参考）9 op 一括の 1 回目（`nth/`） | 2.90 / 2.90 s、2.54 s | 0.13 s, 16 ms/op | SOFT_MAX 128 ms の副作用込み。GroupNorm だけ generic との出力差 L2rel 7.4e-5 |

  group_norm の dispatch 1 行だけで GROUP_NORM が 453 → 16 ms/op、グラフ全体が 6.27 → 2.78 s（GroupNorm だけ generic の 2.71 s と
  0.07 s 差）。**32 スレッド固定が原因**で確定。残る 16 ms/op は `ggml_norm` 再表現の 8 ms/op の 2 倍（8 threadgroup しか出ないので
  GPU コアの一部しか使えない。3 パス）。
- 対処は 2 通り: (i) 既存の `NafGgmlOpts::generic_lowering`（または今回足した GroupNorm だけの `generic_groupnorm`）を Metal でも
  有効にする（host 経路 `naf_upsample_ggml` も device 経路 `pixal3d_cond_gpu.cpp` の分割グラフ L197 / 単一グラフ L525 も
  `naf_ggml_opts_for(naf)` を通すので、判定を変えれば両経路に効く。`TRELLIS_DBG_NAF_GENERIC=1` が今の入口）、
  (ii) ggml-metal 側の `nth` を直す（`patches/` の流儀で、pad / pool は native のまま、数値は native と同じ reduction 幅に近い）。
  速いのは (i)（T=1024: 2.60〜2.71 s vs 2.78 s、T=512: 2.45 vs 2.51 s）、native op のまま・pad / pool を触らないのは (ii)。
  (ii) は `thirdparty/ggml` の 1 行変更で、`docs/PIXAL3D_UPSTREAM_POLICY.md` の流儀（`patches/` + upstream 提案）が要る。
  follow-up PR では (i) を既定にし、(ii) は upstream ggml へ提案する（コメントアウトされたスケール処理を戻すだけ）。
  → **issue #55 / `docs/results/2026-09-20-metal-groupnorm.md`**（2026-09-20）: (i) を Metal の既定にした。op 単体の float64 参照で
  native `ggml_group_norm` は max|d| 2.2e-3（1/std を過大評価）、再表現は 9.5e-7 なので、上の「native 比 L2rel 1.2e-4」は
  ほぼ native カーネル側の誤差。
- `IM2COL`: node 別では 1×1 conv（`encoder` 枝、C_in=128 の 4 層）が **215 ms/op** で 256 MB（`[128, 1M]` f16）を書き、
  3×3（`sem_encoder` 枝、C_in=128 の 4 層）が 211 ms/op で 2.4 GB（`[1152, 1M]` f16）を書く（残り 2 op は C_in=3 の初段で小さい）。
  **K に依らず ≈210 ms** なのは、Metal の `kernel_im2col` が threadgroup grid = (IC, OH, OW) = 128 × 1024 × 1024 = 1.34 億
  threadgroup を、各 (N, KH, KW) = 1×1 スレッド（3×3 なら 9 スレッド）で dispatch する形状のため（帯域ではなく dispatch で律速。
  1×1 は 1.2 GB/s、3×3 でも 11 GB/s）。1×1 conv の im2col は `[W*H, C]` への permute + f16 cast と同じ物なので、
  `ggml_mul_mat` を直接当てれば消せる（4 op × 215 ms ≈ 0.86 s/view が上限。permute + f16 cast の CONT/CPY が 1 op 残るので
  実現は ≈ 200 ms/op、0.8 s/view。**未実測**）。3×3 側はカーネルの dispatch 形状（1 threadgroup で複数ピクセル）を変える必要がある。
- `MUL_MAT(f16)` 0.11 s: 畳み込みの GEMM 自体は 1.3 TFLOP を 0.11 s（12 TFLOPS）で回っており問題ない。

## 5. E2E 1 本（`--profile-cond`、`--profile` 無し、throttled、統合確認）

`trellis-cli --views docker/linux-webgpu-gate/e2e/views -m ~/nfs/weights/pixal3d/gguf-q8_0 --res 1024 --seed 1 --profile-cond`
（`e2e_profile_cond.sh`、04:07:34 ゲート通過 → 04:37:21 終了、`done in 1786.6s`。raw: `docs/results/2026-09-20-cond-profile/e2e/e2e_cond.log`、
`cond.glb`、`ps_snapshots.log`）。比較対象は同じ commit（4853a55 の内容）で `--profile-cond` 無しの E2E
（別セッション 2026-09-20 03:37 終了、`ref_rope_e2e.log` / `ref_rope_4853a55.glb`）。

**条件の注記（E2E は 1 本で、性能の定量根拠にはしない。以下は観測値の記録）**: 他の trellis プロセスは全期間 0 件だが、
(a) 直前まで別セッションの E2E（03:35〜04:07）が同じ GPU を回していて GPU が温まった状態で開始、(b) Chrome の renderer
（pid 88460）≈100% CPU と Chrome GPU プロセス（pid 1061）≈35% が全期間居た（ユーザーの対話中のブラウザ。止められない）、
(c) suggestd（冒頭の常駐例外）。参照 E2E（別セッション）との flow の差（HR 881.6 vs 630.8 s、tex 459.9 vs 373.3 s、step ごとの
遅れは一様で突発ではない）は環境差として記録するだけで、どちらの値も速度の主張には使わない。この E2E で使うのは設計どおり
(1) `accounted` の照合、(2) conditioning ラップが fresh A/B の host 値とどれだけ離れているか（tex 72.0 vs 73.3 = −2%、HR 44.5 vs 42.0 = +6%、
LR 14.4 vs 13.6 = +6% → conditioning 側の結論は E2E の熱状態に依存しない）、(3) 出力の非破壊確認、の 3 点。

### 5.1 段別（`[cond]`、秒）と `accounted` の照合

| 段 | 段合計 | `accounted` | 差 | 参照 E2E（`--profile-cond` 無し） | `[cond]` 内訳（排他的、実行順） |
|---|---:|---:|---:|---:|---|
| SS | 44.3 | 44.3 | 0.0 | 43.6 | load dinov3 0.1 / cond_ss S=512 R=16 (host) 0.5 / flow 38.7 / ss_decode 5.0 |
| LR shape SLAT | 57.5 | 57.5 | 0.0 | 56.1 | load 0.0 / **cond_slat S=512 T=512 (host) 14.4** / gather+noise 0.0 / flow 43.1 / denorm 0.0 |
| HR shape SLAT | 930.9 | 930.9 | 0.0 | 678.3 | load shape_dec 0.2 / **shape_upsample 4.2** / quantize 0.0 / load 0.0 / **cond_slat S=1024 T=512 (host) 44.5** / gather+noise 0.0 / flow 881.9 / denorm 0.0 |
| shape decode | 27.4 | 27.4 | 0.0 | 25.2 | load 0.1 / **shape_decode 25.4** / dual_grid_to_mesh 0.6 / fill_holes 1.3 |
| texture SLAT + decode | 553.3 | 553.3 | 0.0 | 482.1 | load 0.1 / **cond_slat S=1024 T=1024 (host) 72.0** / gather 0.0 / flow 460.3 / load tex_dec 0.3 / **tex_decode 20.6** / pbr unpack 0.0 |
| postprocess | 173.1 | — | — | 230.6 | weld 4.9 / bvh 1.9 / remesh_dc 12.5 / decimate 115.5 / uv+bake 33.1 / GLB 3.5 / PLY 1.8 |

5 段すべて `accounted` = 段合計（差 0.0 s、`verify_e2e.py`）。flow 以外の合計は 189.4 s（段合計 − flow ラップ）で、内訳は
**cond_slat host 3 段 = 130.9 s**（fresh A/B の host 合計 128.9 s と 1.6% 差）、decoder 3 つ（upsample 4.2 + shape 25.4 +
tex 20.6）= 50.2 s、ss_decode 5.0、その他（load / mesh / fill / cond_ss）3.3 s。issue の「≈95 s（tex cond + NAF@1024 + PBR decode）」は
72.0 + 20.6 + 0.4（load）= 93.0 s、「≈50 s（HR cond + upsample）」は 44.5 + 4.2 + 0.2 = 48.9 s に分解できた。

### 5.2 E2E 内の `[cond-v]`（host 経路、view ごと、秒）

| 段 | dino | lr_proj | naf | hr_proj | accum | view 合計 | 4 view 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LR S=512 T=512 | 0.12〜0.15 | 0.05〜0.06 | 3.13〜3.27 | 0.22〜0.24 | 0.01 | 3.55〜3.69 | 14.4 |
| HR S=1024 T=512 | 0.84〜1.05 | 0.64〜0.91 | 7.73〜8.28 | 1.25〜1.51 | 0.05〜0.13 | 10.82〜11.39 | 44.4 |
| tex S=1024 T=1024 | 0.83〜0.91 | 0.65〜0.72 | 13.45〜15.72 | 2.01〜2.24 | 0.05〜0.29 | 17.07〜19.76 | 71.9 |

`dinov3 S=1024 ntok=4101`: compute 0.81〜1.02 s / view、activations 1121 MB（HR と tex で同じ入力に対して 2 回 = 8 回走る）。

decoder（`[cond-v]`、stage ごと、秒。N は入力 token 数、`-> N` は出力）:

| decoder | stage0 C=1024 | stage1 C=512 | stage2 C=256 | stage3 C=128 | 合計 |
|---|---|---|---|---|---:|
| sparse_upsample（LR→HR coords、N=4438→1.21M） | convnext×4 0.15 / c2s 0.17 | convnext×16 0.83 / c2s 0.29 | convnext×8 0.71 / c2s 0.44 | convnext×4 0.57 / c2s 0.91 | 4.2 |
| shape_dec（N=17614→4.82M） | convnext×4 1.60 / c2s 1.15 | convnext×16 5.39 / c2s 1.68 | convnext×8 4.03 / c2s 2.24 | convnext×4 3.92 / c2s+head 5.05 | 25.4 |
| tex_dec（N=17614→4.82M） | convnext×4 0.79 / c2s 0.90 | convnext×16 4.72 / c2s 1.22 | convnext×8 3.47 / c2s 2.12 | convnext×4 3.04 / c2s+head 4.01 | 20.6 |

decoder はどちらも convnext 合計（shape 14.9 / tex 12.0 s）と c2s 合計（shape 10.1 / tex 8.3 s）に二分され、stage1（C=512、16 block）
が最大。neighbor_table は 0.3 s 以下。**1 本の E2E の観測値**で、順位付け・定量の根拠にはしない（decoder は参照実装と同じ構成で
issue の対象外。#28 の別 issue で fresh に測り直す際の当たりとしてだけ残す）。

### 5.3 出力の一致（`--profile-cond` は printf のみ）

- `tools/glb_metrics.py`: V 660 440 / F 962 544 / components 22 144 / boundary edges 333 680 / non-manifold 0 / winding 100% /
  bbox [-0.3782, -0.4228, -0.4918]〜[0.3783, 0.4225, 0.4926] / charts 22 099 / atlas 4096² WebP × 2 / metallicRoughness 統計、
  すべて参照と同一。`tools/compare_glb_pair.py`: nn 距離 mean / p95 / max = 0.000000（両方向）、nonfinite 0。
- **GLB の BIN チャンク（33 813 372 bytes、頂点・UV・index・WebP 2 枚）は sha256 が一致**（`fc017b19a54a36f8…`）。差は JSON の
  `asset.extras`（generator 文字列・commit・build 時刻）の 8 bytes だけ。注意: `cond.glb` の `asset.extras.commit` は 4853a55 と
  出るが、これは cmake configure 時のスタンプ（`build-metal` を 4853a55 で configure し、以後は再 configure せずにビルド）で、
  実際のバイナリは db0623b。provenance にこの欄を使わない。
- 5 視点レンダ（y 回転 0/90/180/270° + 上面、`render5.py`、上段 参照・下段 今回。この GLB では 'back' 列が顔）:
  `docs/results/2026-09-20-cond-profile/e2e/render5_ref_vs_cond.png`（BIN が bit 一致なので同一。記録のために出した）。
- ログの行種: 数値・パス・空白幅を正規化した行接頭辞集合の差は `[cond]` / `[cond-v]` の 46 種だけで、参照にしか無い行は 0。

## 6. 候補の順位（消せる時間の上限 / 実現見込み、fresh、E2E 換算 = 4 view × 該当段）

設計 §6 の規則: 上限 = 候補が置き換える排他的ラップの fresh 中央値の合計、実現見込み = **A/B で実測できたものだけ**数値で書く。

| # | 候補 | 上限（置き換える排他的ラップの合計） | 実現見込み（実測） | 根拠 |
|---|---|---:|---:|---|
| 1 | **CLI の 3 段を `pixal3d_cond_slat_gpu(..., &coords)` に切り替える**（dense R³・4 GiB の host materialize・stride-T² 転置・host bilinear をなくす） | host 版合計 128.9 s | **−63 s（中央値）、保守ケース −56 s**（合成 fixture の host − device、n=3〜4、tex host の実行間変動の感度。うち HR + tex の −56.3 s は PR #34 が CUDA 向けに実装済み、LR の −6.7 s が残り） | §1。数値差は合成 fixture で global bit 一致 / proj L2rel 4.9e-8（実 E2E 座標での統合一致は切り替え PR で確認）。WASM 側は既にこの経路。#34 は Metal 未実行なので、#34 を Metal で回して §1 の device 版の値（tex 31.2 / HR 27.8）が出ることを確認するのが最短 |
| 2 | **NAF encoder の GroupNorm を Metal でも `ggml_norm` 再表現にする**（`generic_groupnorm`、または T=1024 では full `generic_lowering`） | GROUP_NORM 3.62 s × 4 view × 2 段（S=1024）= 29.0 s + LR（S=512、未測） | **−29 s**（whole graph の実測差 × 4 view: tex 4 × 3.67（full generic）+ HR 4 × 3.55（GroupNorm だけ）。LR は未測なので含めない） | §4。#1 の後も device 版の 77% が naf_enc なので独立に効く。出力差 L2rel 1.2e-4（既存 tol 3e-3 内）。代替は ggml-metal の `nth` 修正（§4.3） |
| 3 | NAF encoder の IM2COL / CONT（1×1 conv の im2col 撤去 → `mul_mat` 直接、3×3 im2col の dispatch 形状、レイアウト往復） | (1.72 + 0.63) s × 4 × 2 段 = 18.8 s + LR | **未測**（1×1 の 4 op × 215 ms = 0.86 s/view が ISOLATED 時間からの上限。`mul_mat` 直接化後の whole graph A/B は未実施） | §4。#2 の後は残りの 90% |
| 4 | DINOv3@1024 の patch map を HR と tex で共有 | dino 0.8 s × 4 ≈ 3.2 s | 同値（計算をそのまま省く） | §2、§5。dinov3 / naf の再 load は E2E で合計 0.8 s（mmap）なので対象外 |
| 5 | tex_decode / shape_upsample の stage 別 | E2E 観測値のみ（§5、定量根拠ではない） | — | 参照実装と同構成。issue 対象外 |

#1 と #2 を両方入れると conditioning 3 段の fresh 合計は 128.9 → 66.0 → **≈ 37 s**（66.0 − 29）の実測ベース見込み
（LR の GroupNorm 分と #3 は上限のみで、この数字に入れていない）。E2E の flow 以外 189 s のうち conditioning 131 s がこの対象で、
残り（decoder 50 s、ss_decode 5 s）は別 issue。

## 7. まだやっていないこと・注意

- `generic_groupnorm` / `generic_lowering` を Metal 本番で有効にしたときの PyTorch fixture との一致確認。fixture（`tools/ref_pixal3d_naf.py`
  の出力）は pod 側（`/mnt/hdd1/pixal3d/ref/pixal3d/naf`）にあり、この Mac には無い。本ドキュメントで確認したのは
  **native（Metal）との差 L2rel 1.2e-4**（§4.1）まで。CPU backend では `trellis-test-naf --generic` で fixture 一致済み。
- LR（S=512）の GroupNorm lowering の実測（候補 #2 の LR 分は上限のみ）。設計 §5-2 が予定していた `--naf-direct-conv` の T=512 も
  未実施（T=1024 で +3.3 s と結論が出たので省いた）。
- IM2COL / CONT の削減案の実測（候補 #3、follow-up PR）。
- 候補 #1 の実 E2E 座標での統合一致（Metal で PR #34 を回す）。
- WebGPU（native Dawn）ビルドは、この Mac に prebuilt Dawn が無くなっていたので未確認（設計のテスト計画 B4 の一部）。
  CPU（`-DGGML_METAL=OFF`）と WASM（`build-wasm-ss`: partial / full E2E の両 target）は link まで確認済み。CUDA / Vulkan は
  ビルド環境が無く未確認（変更は backend 非依存の printf と `ggml_backend_*` 呼び出しのみ）。
- 計測手順で踏んだこと（次回の投入前ゲート）:
  1. 計装した経路（T ≤ 512 の単一グラフ経路）を投入前に 1 回も実行せずに A/B を回し、`ggml_free` 後の `ggml_graph_n_nodes(g)` で
     segfault した（16c29cd → db0623b）。→ 計装した全経路を 1 view で smoke してから投入する。
  2. A/B 実行中に同じ build ディレクトリのバイナリを再ビルドした（LR device の 1 本目だけ 16c29cd。差分は printf のみ）。→ 計測中は
     ビルドしない（再計測・nth 実験では守った）。
  3. 1 回目の A/B は静穏ゲートを開始時にしか掛けておらず、他セッションの CPU / GPU ベンチが 26 本中 12 本に重なった。しかも 1 回目の
     版のドキュメントでそれを「全実行 clean」と書いた（codex の反証レビューで発覚）。→ ゲートは実行ごとに掛け、汚染判定は
     手で読まず `classify_runs.py` で機械的に出す。
  4. E2E は直前まで別セッションの E2E が GPU を使っていた直後に始まり、flow が参照より遅かった（§5、観測値のみ）。
     GPU の冷却待ち（数分）もゲートに含める。
