# trellis2-mv B3（4-view + stochastic, res1024）テクスチャ全面黒バグ — 仮説と診断計画

日付: 2026-09-21 / 対象ブランチ: `feat/trellis2-mv-hardening`（HEAD `0d4cfc5`）/
ステータス: 設計中（実装前レビュー待ち） / 親: `/Users/<redacted-user>/Downloads/claude-work/docs/briefs/trellis2-mv/{COMMON,TASK-B3}.md`

> **【2026-09-22 統括追記・superseded】** 本 doc は調査着手時の仮説・診断計画で、下記「事実」節の前提
> （B3 のみ黒・B1/B2/B4 正常・参照は正常 = trellis.cpp 固有のバグ）は**その後の実測で全て崩れている**:
> native B3 は 6 seed 中 4 本が正常、B1 も 3 本中 1 本が黒、参照側も 2-view seed 42 が全 texel 0 で黒化する。
> 最終的な決着は `../results/2026-09-21-trellis2-mv-b3/2026-09-22-b3-conclusion.md` を読むこと。
> ここに書かれた診断計装は `diag/trellis2-mv-b3`（tip 2649e61）にのみ存在する。

## 事実（担当 `mve2e` の報告、読むだけ。`~/Downloads/pixal3d-mv-e2e/docs/results/2026-09-21-trellis2-mv-e2e/README.md`）

- CUDA pod（A100）、`0d4cfc5`、`--seed 42 --res 1024`、v2 fixture 実写4-view。
- B3（4view, stochastic, res1024）**のみ** `out_base.png` が全面黒（mean=0.07, max=159。他 run は mean 10〜65）。
- B1（2view stochastic）・B2（2view multidiffusion）・B4（4view multidiffusion）は正常。
- 同一条件の PyTorch 参照 `run_4img_real_stochastic_1024c` は正常（trellis.cpp 固有のバグ）。
- `trellis-test-trellis2-mv-tex`（12 fixture run、`run_4img_real_stochastic_1024c` 含む）は ALL PASS。
- run.log に NaN/Inf/error/warning は一切なし（静かな失敗）。

## このツリーの HEAD での静的解析（結論）

`src/trellis_cli.cpp` のテクスチャ段（1043-1121行）と `src/flow_runner.cpp` の
`sample_flow_multi`（519-616行）、および凍結済み参照仕様
（`docs/results/2026-09-21-trellis2-mv-reference.md`）を突き合わせて確認した。

1. **stochastic カウンタの持ち越し（当初の主仮説 a）は棄却**。
   - 参照仕様が明記: 「SS and texture each get a fresh counter; cascade shape LR + HR share
     one injection context」。C++ 側もこの通り実装されている
     （`trellis_cli.cpp:894-902` の `sample_bank` はテクスチャ呼び出し時に
     `stochastic_counter` を渡していない → `flow_runner.cpp:558` の `seq = i`）。
   - shape の LR→HR 共有カウンタも「V が12を割り切らない場合のみ HR 開始 view が変わる」と
     明記されており、V=4/V=2 はどちらも 12%V==0 で無関係。
   - さらに SS・shape（cascade HR 込み）は同じ `sample_flow_multi` の Stochastic 分岐を
     **V=4 で正常に通過している**（B3 のジオメトリは見た目上正常）。関数自体に V=4 固有の
     バグがあるなら SS/shape も壊れるはずで、そうなっていない。
2. **cond1024 vs cond512 の取り違え（仮説 b）は可能性低いが未確認**。
   `mixed` 分岐は `so.coords.size() > 9,000,000` のみで決まり V/mode に依存しない。
   B3 の SS active voxels（936）は B1（958）と近く、decoded voxels も同程度と推測されるため
   閾値を超える可能性は低いが、実測未確認（診断計画で確認する）。
3. **decoder/bake 側の入力破損（仮説 c）は可能性低い**。
   `trellis::tex_decode` は V/mode を一切知らない共通関数で、B1/B2/B4 と全く同じコードパスを
   通る。texlat（decode 直前の tex SLat）自体が壊れていれば decoder は無実というのが自然な読み。
4. **背景除去(matte)の失敗（仮説 d）は棄却**。
   テクスチャ段が使う cond1024_bank は shape HR 段と全く同じバンクであり、shape HR は
   正常に見えるジオメトリを出している。同じ入力画像・同じ V=4 stochastic で shape は
   壊れずテクスチャだけ壊れるので、cond bank 自体（≒背景除去の結果）に問題があるとは考えにくい。
5. **新仮説 e（本命）**: テクスチャ段の 64ch concat（`[noise(32); shape_slat_norm(32)]`、
   `trellis_cli.cpp:1082-1090`）は、設計ドキュメント
   （`docs/design/2026-09-21-trellis2-mv-stages-shape-tex.md`）が明記する通り
   **`ReplayModel` ベースの fixture テストでは検証不能な唯一の箇所**（`ReplayModel` は
   `x`/`cond` の中身を無視して記録済みテンソルを返すだけ）。つまり
   「テクスチャ flow の数値計算（CFG 混合）」と「テクスチャ flow が実際に受け取る入力
   （64ch concat・実 `cond`）」は別物で、後者は E2E（本タスク）でしか検証できない。
   V=4 stochastic 固有で、DiT 内部のどこかで非有限値が発生し、
   `flow_runner.cpp:605` の safety net（`if (!std::isfinite(v)) v = 0.0f;`）で
   **ログ無しに黙って握り潰されている**可能性が高い（症状の「NaN/Inf が一切ログに出ない静かな失敗」
   と整合する）。この safety net はまさにそういうケースを想定して書かれたものだが、
   握り潰した事実そのものをログしないため、通常運転時は見分けがつかない。

## codex exec レビュー反映（2026-09-21）

`要修正` の指摘を受けて以下を訂正する。

- **仮説a/c/dの「棄却」は言い過ぎだった**。shape が正常でも texture は別重み
  （`tex_flow_1024.gguf`）・別 in_ch（64）・別トークン数系列であり、flow/DiT 実装全体を
  免責する根拠にはならない。cond が有限でも decoder/bake 側でモデル感度差により黒化し得る
  （仮説c）。以降は「疑い低い」程度の扱いに弱め、実測で決着させる。
- **新仮説f（codex 提案、本命候補に追加）**: 4view の中の**特定1枚**が `tex_flow_1024` モデル
  （shape とは別重み）に対してだけ不安定/NaN を誘発する。stochastic は12ステップ中その view を
  選ぶ約3ステップで**希釈されず単独採用**するため致命的になり得る一方、multidiffusion は
  毎ステップ4view全部を平均するため、同じ NaN が発生しても他3viewとの平均で薄まり
  see safety net後は局所的な speckle 程度に留まる、という説明が
  「V=4×stochasticの組み合わせでのみ発生・V=4×multidiffusion(B4)は正常」に最も整合する。
  ただし B4 が本当に無傷か（微小な異常が「目視破綻なし」で見逃されていないか）は未確認。
- **二分計装を decode/bake 出力側まで拡張する**（入力側 texlat だけでは、texlat が全て有限
  なのに black という結果が出たとき原因を確定できないという指摘の通り）: `tex_decode` の
  生出力 (`pbr`, clamp前) と clamp後 `pbr6`/`colors` にも `slat_stats` 相当の統計を追加する。
- **`[dit-nan]` にステージ名を付ける**（DitRunner クラス自体は変更せず、`make_sparse_runner`
  呼び出し直前に `fprintf(stderr, "[stage] entering <name> DitRunner ...")` を打つだけの
  軽量な方法で代替 — 複数 DitRunner が同一ログに出力されるパイプライン全体で、どの
  `[dit-nan]`/`[flow-mv-step]` 行がどのステージのものか区別できるようにする）。
- **view 分離テストを同一 pod セッション内に追加**: B3 の4枚の入力画像それぞれを単独
  （V=1、`--trellis2-mv-mode stochastic` でも実質単一condになる）で res1024 通し実行し、
  texlat/decode 統計を比較する。B1+B3 の計装だけで全て有限だった場合の決着手段として、
  同一 pod 内（複数回の `trellis-cli` 呼び出し、pod 自体は1つ）で実施する。

## 診断計画（二分探索、pod 1回で完結させる）

現行コードには **tex SLat（texlat, decode 直前）の統計ログが存在しない**
（shape SLat には `slat_stats("HR slat", slat_dn)` があるが tex には無い）。
brief が前提とする「既存 dump」は実際には存在しないため、最小の計装を追加してから
1 回の診断 pod 実行でエビデンスを取る。

### 追加する計装（診断専用、フラグ無しで常時出力 — 数行の printf のみ）

1. `trellis_cli.cpp` のテクスチャ段: `sample_bank` 呼び出し直後・denorm 前後で
   `slat_stats("tex slat (norm)", texlat)` を追加。既存の `slat_stats` ヘルパー
   （40行目、mean/std/min/max/finite率を出す）をそのまま使う。
2. `flow_runner.cpp` の `sample_flow_multi` Stochastic 分岐: 各ステップで選ばれた
   view index（`seq % conds.size()`）と、その `pred` の non-finite 数を
   `TRELLIS_DBG_MV_STEP` 環境変数でゲートしたログとして出す（既存の `TRELLIS_DBG_NAN`/
   `TRELLIS_DBG_STEP` と同じ命名規約）。

### pod での実行（1回、自然終了型、完了後削除）

- 既存ビルド済みバイナリ（NFS `/nfs/trellis2-mv/src/0d4cfc5/build/`）は計装前のものなので
  再ビルドが必要。ソース差分を tar で送り、NFS 上の既存ビルドディレクトリを再利用して
  incremental rebuild（フルビルドしない）。
- 実行: B3（4view stochastic res1024）を `TRELLIS_DBG_NAN=1 TRELLIS_DBG_MV_STEP=1` で1回。
  比較用に B1（2view stochastic res1024、正常系）も同条件で1回（同一 pod 内で直列、
  「同時に1本」原則は維持）。
- 確認する二分点:
  - `[dit-nan] *** first NaN forward` が **tex_flow_1024 の DitRunner** で出るか（出れば
    「flow 側」に確定、次はどの層・どの step で NaN が発生するかを内訳ログで見る）。
  - `tex slat (norm)` の finite 率・mean/std が B1/B3 で明確に異なるか（texlat 自体が
    壊れているなら decoder より手前に確定）。
  - `mixed` 分岐の実際の値（so.coords.size() と 9,000,000 の関係）をログに残し、
    仮説 b を実測で確定的に棄却/採用する。

### 原因確定後

- 最小修正（当該バグに限定）+ weightless 回帰テスト（V=4 stochastic で tex ステージが
  受け取る cond/counter/finite性を検査する構造テスト、または既存テストへの assert 追加）。
- 修正後、**追加でもう1回だけ** pod で B3 を再実行し `out_base.png` mean が正常範囲になることを
  確認する（brief 手順5、本診断runとは別カウント）。
- view順ログ（#65 acceptance gap）も同じビルドで stage log に1行追加する（別コミット）。

## 禁止事項の遵守

- fixture テストの期待値は変更しない。`feat/trellis2-mv-e2e` worktree は触らない。push しない。
- 原因未確定のまま「直った」と報告しない。

## 追補（2026-09-21 22:5x、統括指示で次段の判定基準を先に凍結）

実重み・逐次12ステップ parity probe（`trellis-test-trellis2-mv-tex-realweight-parity`）で
tex_flow_1024 の forward が PyTorch 参照から step0=1.29%→step11=5.12% 単調発散することを確認した
（`docs/results/2026-09-21-trellis2-mv-b3/HANDOVER.md` 参照）。この「累積発散」説を検証・補強する
次段に着手する前に、判定基準を凍結する（着手後に動かさない。動かした場合はその旨を明記する）。

### 凍結する判定基準

- **(a) 累積発散説の反証条件**: B1（V=2 stochastic、テクスチャは正常）で同じ probe を回し、
  tex step0/8/11 の rel が B3（V=4）と**同程度（目安: 概ね同じ桁・傾向、5%前後）**なら、
  「V=4 stochasticでの累積発散が黒化の原因」という説は**反証される**（V=2でも同程度発散するのに
  V=2は黒くならないため）。逆に B1 の rel が B3 より明確に小さい（例: 1%未満で頭打ち）なら、
  V=4/V=2 間の発散量の違いが黒化と相関する、という部分的な裏付けになる（それでも「相関」であり
  「機構の証明」ではない点は明記する）。
- **(b) 発散量だけでは「全面黒」を説明するには弱いという前提**: 5%程度の連続的な値のずれだけで
  decoder が全チャンネル・全 voxel でほぼ一様に飽和する（既知: R/G/B≈-0.987、M/R/A≈+0.999、
  std が健全run の1/10以下）のは非直感的。そのため発散量の比較と**並行して**、
  離散的原因（特定 step で値が跳ぶ、特定 view の cond だけ発散、counter が想定外の値を取る）が
  無いか B1/B3/B4 の tex SLat・decoder 入力の max abs・NaN/Inf 有無・飽和割合を並べて確認する。
  ここで「離散的原因あり」の証拠が出たら、発散量ではなくその離散的原因を主因として扱う
  （発散量の緩やかな増加は「随伴して観測される現象」に格下げする）。
- **12点 step rel 推移の制約（判明済み、着手前に明記）**: pinned v2 fixture が PyTorch 参照の
  CFG 後テンソルを保存しているのは **tex/shape_hr/shape_lr とも各 stage 0/8/11 の3点のみ**
  （SS stage のみ全12点）。よって「B1/B3/B4 で12点の rel 推移を表にする」は PyTorch 参照との
  比較としては**3点までしか作れない**（新規に PyTorch 参照を12点分生成するのは本タスクの
  スコープ外・別途相談）。凍結する代替: ①PyTorch 参照との rel は0/8/11の3点表、②native 自身の
  12ステップ全部の trace（forward予測の max\|finite\|・選択view・non-finite数）は既存の
  `TRELLIS_DBG_MV_STEP` ログとして B1/B3/B4 の実 E2E 診断ログに既にあり（新規 pod 不要）、
  これを12点表として使う（PyTorch 参照とは比較しない、native 内の離散異常検出専用）。
- **shape_flow_1024 との比較**: tex 固有かどうかを切り分けるため、同じ parity probe 手法を
  shape_flow_1024（HR、V=4、B3 と同じ幾何・cond）にも適用し、0/8/11 の rel を tex と並べる。
  shape の baseline（`trellis-test-slat-shape`、非MV・単一cond、GS=7.5/GR=0.5）が既に
  23.3%/43.3% という**tex より遥かに大きい**発散を報告している点（7a で再現確認済み）は、
  「発散量が大きいほど黒くなる」という単純な閾値仮説に対する事前の反証材料として扱う
  （shape は大きく発散しても出力メッシュは正常に見えるため）。ただしこれは異なる測定条件
  （非MV・GS=7.5 vs MV・GS=1.0、モデルも別）なので参考情報に留め、MV・V=4のshape_hr自身の
  rel を新たに測ってから結論する。

### 追補2（2026-09-21 23:4x、pod v6 実行結果 — 判定基準(a)は反証で確定）

`trellis-test-trellis2-mv-tex-realweight-parity` を B1(V=2) にも回し、新規
`trellis-test-trellis2-mv-shape-realweight-parity`（shape_flow_1024 HR、上記追補1のとおり
gs=7.5/gr=0.5/gi=[0.6,1.0]で実装）をB3(V=4)・B1(V=2)双方の幾何で回した（pod
`trellis2mv-b3-diag` v6、NFS `/nfs/trellis2-mv/b3-diag/realweight_parity_{tex_v2img,
shapehr_v4img,shapehr_v2img}/parity.log`、ローカル `logs/v6-pod-full.log`）。

| stage / config | step0 rel | step8 rel | step11 rel |
|---|---|---|---|
| tex_flow_1024, V=4 stochastic（B3、黒）| 1.29% | 4.45% | 5.12% |
| tex_flow_1024, V=2 stochastic（B1、正常）| 1.22% | 2.33% | **7.05%** |
| shape_flow_1024 HR, V=4 stochastic（B3幾何）| 4.11% | 57.7% | 83.2% |
| shape_flow_1024 HR, V=2 stochastic（B1幾何）| 3.97% | 81.0% | 63.4% |

**判定基準(a)（反証条件）が成立 → 「累積発散が黒化の原因」説は反証された。**
B1（正常）の step11 rel(7.05%)はB3（黒、5.12%）と同程度どころか**上回っている**。
「V=4 stochasticでtexの前方計算誤差が閾値を超えて黒くなる」という筋書きは、V=2でも
同等以上に誤差が出るのに黒くならない事実と矛盾するため維持できない。

さらに shape_flow_1024 は V=4/V=2 どちらでも 8〜11 ステップ目で 57〜83% という
**texより遥かに大きい**発散を示すが、shapeのジオメトリはB1/B3双方とも正常に見える
（既知事実、task brief記載）。これは「PyTorch参照からのforward発散量」という指標**単体**が
黒化/破綻の予測因子として機能しないことを補強する（shapeは巨大な発散を許容して正常動作する
系であり、texのV=4での発散(5.12%)はshape基準からすれば非常に小さい）。

**結論: 発散の"量"ではなく"構造"を疑うフェーズへ移行する。** 判定基準(b)で予告した離散的原因の
確認は、既存の diagnostic ログ（新規pod不要、本セッションで既に収集済み）で部分的に果たせる:

- decoder出力（`tex_decode base color, post-clamp (colors, [3,Mv])`）の std がB3だけ突出して
  小さい: B1 std=0.0428, B4 std=0.0337, **B3 std=0.0038**（約9〜11倍小さい＝ほぼ一様＝視覚的に
  「のっぺり」で報告された黒に整合）。mean も B1=0.0454, B4=0.0238, B3=0.0021 と一貫して
  B3だけ一桁小さい。NaN/Infはどのrunも0（`nan/inf=0`）。
- B3とmodeだけ違うB4のtexlat集約統計（32ch mean/std）は大きく違わないのに decode 出力だけ
  激変する（追補1既出）＝ **texlatの集約統計に現れない構造的な違い**（特定チャンネル・特定
  voxel群への偏り等）がdecoderを直撃している可能性が高い。
- 前方計算誤差の**大きさ**はV=4/V=2で同水準・同方向に増加する「ノイズ的」ふるまいに見えるが、
  誤差の**符号・チャンネル分布**（系統的バイアスか、平均ゼロのノイズか）は今回のprobeでは
  未計測（aggregate rel/mean|d|/max|d|のみで符号情報を捨てている）。

### 追補3（2026-09-21 23:5x、pod不要のmatte目視確認 — 統括指示、advisor()指摘のcond dump/差し替えpodの前段）

advisor()の新仮説（native自身のcond1024_bank、特にFrame2/Frame3のDINOv3特徴がbirefnet.gguf
未変換のthreshold matte fallbackで劣化している可能性）を検証する前に、**pod不要で最も安い確認**
として、Mac上でCPUのみ（重み不要）でnativeのthreshold matteをFrame1〜4に単体適用し、
参照側の`preprocessed_{0..3}.png`と目視+alpha被覆率で比較した。

新規ツール `trellis-tool-dump-threshold-matte`（CMakeLists.txtに追加、`src/`配下、
テスト対象ではなくad-hocツールと明記）で `trellis::threshold_cutout()`（実際にB1/B3/B4
E2Eで使われた関数そのもの）を各Frameに適用し alpha被覆率を測定:

| frame | native opaque% | native transparent% | native partial% |
|---|---|---|---|
| Frame1 | 13.35% | 86.65% | 0.00% |
| Frame2 | 0.52% | 86.92% | **12.56%** |
| Frame3 | 13.62% | 86.38% | 0.00% |
| Frame4 | 0.61% | 86.79% | **12.60%** |

Frame2とFrame4は「ほぼ全て半透明（partial）」というFrame1/Frame3と質的に異なる統計を示した
（premultiply前のalphaマスクの粗さの指標）。これだけ見るとFrame2固有の劣化に見えるが、
**Frame4も同水準の「partial優勢」統計を示しており、Frame4はB1(正常)で使われている**ため、
この統計だけでは「Frame2/3だけ劣化」という仮説を支持しない。

**目視確認（本文の鉄則: 可視化なしに結論しない）**: 4枚を黒背景へpremultiplyして並べたところ
（`docs/results/2026-09-21-trellis2-mv-b3/matte-check/native_premul_Frame{1,2,3,4}.png`）、
**Frame2だけ**顔・首まわりに明瞭な半透明の「ゴースト状」アーティファクト（灰色っぽい影）が
乗っていた。Frame1/3/4は全てクリーンな切り抜きだった（Frame4もクリーン——上のalpha統計の
「partial優勢」はFrame4では単なる輪郭のアンチエイリアシングで、視覚的な欠陥ではなかった。
alpha被覆率という指標だけでは「本物の劣化」と「正常なソフトエッジ」を区別できないことが判明
——今後この種の判定に生の被覆率%を単独で使わない）。

**決定的な追加確認**: 参照側の`preprocessed_1.png`（PyTorch参照のview1=Frame2の前処理結果、
`docs/results/2026-09-21-trellis2-mv-b3/logs/`には無いが `~/nfs/pixal3d_trellis2mv_ref_v2/
run_4img_real_stochastic_1024c/preprocessed_1.png`）を目視したところ、**native premul Frame2と
全く同じ位置・形状のゴーストアーティファクトが存在した**。つまりこの欠陥はnativeの
threshold-matte fallback固有ではなく、**元のFrame2.png自体（もしくは両実装が独立に再現する
何らかの共通の前処理特性）に由来**しており、PyTorch参照側にも同一の形で存在する。

**結論: matte品質（birefnet.gguf欠如によるthreshold matte fallback）はB3黒化の原因ではない
と判断する。** 参照側は同じ欠陥入りのFrame2前処理結果を使ってなお正常なテクスチャを出しており
（brief記載の既知事実）、native側だけがこの欠陥のせいで壊れているとは言えない。この仮説は
**却下**し、advisor()提案のcond dump/差し替えpodへ進む（native自身のcond1024_bankがPyTorch
参照のcond_1024.npyと数値的に一致するかどうかは、matte欠陥の有無とは別の切り口としてまだ
未検証——「同じ見た目の入力画像」から「同じDINOv3特徴」が出るかはGPU上のDINOv3実行の話で、
今回のCPU限定の目視確認では検証できていない点に注意）。

### 次の一手（優先順、判定基準(a)反証を受けて更新）

1. **「発散が系統的バイアスかランダム誤差か」を確認する**: 今回のparity probeを拡張し、
   `pred_native - pred_ref` の**符号付き平均**（チャンネル別、全体別）をstep0/8/11で出す。
   V=4だけ特定チャンネルに一貫した符号のバイアスが乗っていて、V=2はゼロ近傍でばらついている
   なら「バイアス蓄積」が黒化の直接原因になりうる（decoderのGroupNorm/tanh的な飽和は
   ランダム誤差より系統的バイアスに弱いことが多い — 類似事例: [[metal-groupnorm-lowering-findings]]）。
2. 1で有意なバイアスが見つからない場合: texlatの**per-channel**統計をB1(V=2,正常)でも取得し
   （現状B1のper-channel breakdownは未取得、B3相当のmodeMultidiff_texStoch分のみ取得済み）、
   どのチャンネルがB3で異常かをB1と比較する。
3. それでも手がかりがなければ、V=4固有の何か（例: view cycling `0,1,2,3,0,1,2,3,...`という
   周期がSLatの空間パターンと干渉する、V=4のcond bank自体に特定view由来の異常がある等）を
   個別viewでの単体stochastic(V=1相当、counterを1viewに固定)実行と比較して切り分ける。

### 追補4（2026-09-22、pod v7/v8 cond差し替え実験の実施とnoise交絡の発覚）

統括指示でcond1024_bank/cond_bank(512)の差し替え実験（pod v7, v8）を実施した。数値・詳細は
`docs/results/2026-09-21-trellis2-mv-b3/HANDOVER.md`参照。要点:

- pod v7: cond1024_bankのみ差し替え。native(mean=0.073)→Frame2/3のみ差し替え(0.60)→
  全view差し替え(7.19、正常域10〜65には未達)。per-view native-vs-参照relは全4view
  14〜20%でFrame2/3特有ではない。
- pod v8: cond_bank(512)も差し替え。全view512+1024差し替え(39.95、正常)、**512だけ**
  差し替え・1024はnativeのまま(29.84、正常)。

**重要な訂正（advisor()指摘、`src/trellis_cli.cpp:344-345,782-783`で確認済み）**:
`std::mt19937 rng(run_seed)`は**パイプライン全体で単一ストリーム**であり、`noise(n)`が
呼ばれるたびに逐次消費される（SS→shape-LR→shape-HR→texの順）。cond_512を差し替えると
SS段のactive voxel数が変わり、shape-LR以降全ステージのnoise消費量・内容が丸ごと変わる。
**よってpod v8のarm(iv)/(v)「正常化」は、cond_512を直した効果か、単にパイプライン全体を
再抽選した効果か、分離できていない**。pod v7のarm(i)→(ii)→(iii)はcond_512がnativeのまま
（coords/noise消費量が同一）なので、この交絡を受けない——cond1024精度がmeanに単調に効く
（正常域には届かない）という結論だけは維持できる。

**教訓（lessons.md該当・次回優先）**: 確率的生成モデルを条件間で比較する前に、
同一入力・シードだけ変更した対照群でノイズ床を2回以上測るべきだった。次段は
まずビルド不要（NFSに残る v8 バイナリを流用）のシード掃引（B3 native ×6seed、
B1 native ×3seed対照）でB3単体の失敗率を測ってから、cond実験の解釈をやり直す。

### 追補5（2026-09-22、pod v9シード掃引 — タスクの前提自体が崩れた）

pod v9（NFSの既存ビルド流用、再ビルド不要）でB3 native×6seed・B1 native×3seedを実行。
結果は`docs/results/2026-09-21-trellis2-mv-b3/HANDOVER.md`「最重要」節に詳細記載。要点:

- **B3（V=4 stochastic）は6本中2本（seed42,44）が黒（mean<1）、4本は正常（38〜50）**。
  シード非依存の構造的破綻ではなく確率的な失敗モード。
- **対照のB1（V=2 stochastic）も3本中1本（seed44）が黒に近い（mean=1.72）**。
  E2E報告の「B1は常に正常」はseed42（mean=11.0、正常域の下限ギリギリ）という
  たった1回の観測に基づいていた。
- 観測された失敗率はB3(2/6)・B1(1/3)でほぼ同水準——**「V=4がV=2より脆い」という
  TASK-B3.mdの前提自体が、n=1比較のサンプリング運だった可能性が高い**。

**最有力の統一的説明**: stochastic modeは各stepで1viewの生予測のみを使い希釈しない
（multidiffusionは毎step全view平均で常に平滑化）ため、特定のnoise seed下で数値的に
不安定な軌道に入り、decoderの飽和領域まで暴走することがある——V=2でもV=4でも起こりうる
アルゴリズム自体の頑健性の問題であり、「B3固有のバグ」ではない可能性が高い。

**これまでの実験の再解釈**:
- mode分離実験（tex=stochasticなら黒、multidiffusionなら正常、seed42固定）は
  「stochastic modeがmultidiffusionより脆い」という筋としては生き残るが、「V=4」という
  条件の寄与は未分離。
- pod v7/v8のcond精度実験は、cond差し替えでSS voxel数が変わりnoiseストリームが
  丸ごと再抽選された（`trellis_cli.cpp`の単一RNGストリーム設計、追補4で確認済み）ため、
  「cond精度を直したから正常化した」のではなく「別のシードを引いただけ」の可能性が高い。
  cond精度自体の因果効果は本追補時点では確認できていない。

**次段は統括判断を仰ぐ**: 「原因確定→最小修正」というTASK-B3.mdの枠組みは、確率的頑健性の
問題には単純に当てはまらない（コードの1バグではなくアルゴリズム特性の可能性）。修正の形は
fail-closed検出・リトライ・stochastic mode非推奨化等に変わりうる。
