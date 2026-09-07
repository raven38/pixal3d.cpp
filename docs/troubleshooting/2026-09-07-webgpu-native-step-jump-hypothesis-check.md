# ネイティブ Dawn/WebGPU の間欠的 step-jump 破損: 仮説の検証記録（2026-09-07）

対象仮説: 「Shape-1024 / Texture Flow のネイティブ Dawn 実行で、同一入力なのに 12 ステップ中 1 forward が
突然ずれて以後単調に誤差が増える。Chrome/WASM は安定。原因は共有バックエンド/ランタイム側にあり、
(1) subgroup-matrix 経路 → (2) in-flight バッチ無制限 → (3) 1 GiB attention chunk の順で A/B すべき」

検証環境: Apple M4 Max / 64 GB / macOS 26.5、Dawn prebuilt `18eb229`（spec 31 §7 と同一アーカイブ）、
vendored ggml `737e88f2` + `patches/ggml-webgpu/0001-0005`。この checkout は Downloads の zip 展開で
git 管理外。`hr_sample` fixture（PyTorch 参照）と M1 Pro 実機はこのマシンに無いため、**元の障害そのものは
再現できない**。以下は「コード事実の照合」「モデル不要の合成ストレス」「HF から取得した実モデル GGUF +
合成 fixture での決定性テスト（§3.1）」「同じ入力での Chrome/WASM 完走と native との突き合わせ（§3.2）」で
言えることの範囲。**M4 Max では native・browser とも clean で、両者は M1 Pro の clean run と同じ水準で一致した**
（§3.2）。

## 1. 結論

| 項目 | 判定 | 根拠 |
|---|---|---|
| (1) subgroup-matrix 経路が原因 | **反証**（提案の A/B は既に実施済みの構成） | root `CMakeLists.txt:116-118` が `PIXAL3D_WEBGPU_SUBGROUP_MATRIX` 既定 OFF で `GGML_WEBGPU_SUBGROUP_MATRIX=0` を ggml-webgpu に付与。本ビルドの `compile_commands.json` でも確認。spec 31 §9（L855「root CMake sets 0」、L1045「remain off」）は障害を観測したビルド群で同じ。`ggml-webgpu.cpp:3891` で `supports_subgroup_matrix = valid_subgroup_matrix_config`（=false）となり、mul_mat 高速経路の key（shader-lib `:2028`）と FA 経路選択（`:2726`/`:4263`）は両方この値を見る。さらに障害 run は `TRELLIS_NOFA=1` の exact-SDPA なので FA 自体を通らない |
| (2) in-flight バッチ無制限が原因 | **コード事実は正しいが、native/Chrome 差を単独では説明しない** | `ggml-webgpu.cpp:462-464` は `UINT32_MAX`、`graph_compute` は待たずに返る（`:3230-3321`）。ただし param arena への書き込みは `queue.WriteBuffer`（`:562`）でキュー順序保証があり CPU→GPU の競合は無い。in-flight ロジックに `__EMSCRIPTEN__` 分岐は無く、Chrome/WASM も同じ投入パターンで動く。残るのは「Metal 側の資源圧迫を増幅する」役割のみ |
| (3) attention chunk サイズが原因 | **機構なし** | `src/dit.cpp:160-186` のチャンク分割はグラフ構築時の query 分割で、query 間に reduction が無くビット等価。sweep する根拠が無い |
| 「共有バックエンド/ランタイムのバグ」という総論 | **支持。ただし ggml-webgpu 層だけでなく Dawn/Metal 層に無言失敗の経路がある** | §2 |

## 2. 新知見: Dawn の Metal バックエンドは Metal コマンドバッファの失敗を一切見ない

`google/dawn@18eb229` の `src/dawn/native/metal/` 全 44 ファイル（.mm/.h）を取得して grep した結果、
`MTLCommandBufferStatus` / `[cmdBuf error]` / `status` を参照する箇所は皆無。`QueueMTL.mm:236` の
`addCompletedHandler` は完了 serial を進めるだけで、エラーの有無を確認しない。

```objc
[*pendingCommands addCompletedHandler:^(id<MTLCommandBuffer>) {
    this->UpdateCompletedSerialTo(QueuePriority::Lowest, pendingSerial);
}];
```

したがって Metal 層でコマンドバッファが失敗した場合（macOS 26.5 SDK `MTLCommandBuffer.h` の
`MTLCommandBufferError`: `Timeout`=2「実行が長すぎて中断・打ち切り」、`PageFault`=3、
`AccessRevoked`=4「timeout/hang を起こしすぎたクライアントのデバイスアクセス剥奪」、`OutOfMemory`=8
「実行に必要なメモリが不足」。unified log では `kIOGPUCommandBufferCallbackErrorTimeout` 等として出る。ggml-metal では "command buffer N failed with status 5"
として検出されるクラス。同じ vendored ggml の `ggml-metal-context.m:258-276` は `MTLCommandBufferStatusError` を
明示的に検査しており、ggml-metal と Dawn-Metal でこの点が非対称）、WebGPU 側は `OnSubmittedWorkDone` = Success を受け取り、
`ggml_backend_webgpu_check_wait_status`（`:443-459`、timeout/Error/callback 失敗は全て abort）も
`SetUncapturedErrorCallback`（`:3893`、abort）も発火しない。gallocr はバッファを forward 間で再利用する
（`src/flow_runner.cpp:65-68`、同一 forward 内でもスロットを再利用）ので、未実行カーネルの出力バッファには
**そのスロットに直前に書かれた値**（同一 forward の先行 op か前 forward の値）が残る。
これは観測された特徴と整合する:

- NaN ではなく「もっともらしい大きさ」の誤差（rel 3e-3〜1e-2）が 1 forward で発生
- 以後 Euler 更新で単調に伝播
- 破損 forward が遅い（213-240 s vs 193-200 s）: GPU リセット/再スケジュールの時間。ただし §3.2 で
  「遅いが clean」の反例（60-239 s のばらつきで全ステップ一致）を観測したので、遅さは必要条件でも十分条件でもない
- 共有 GPU で悪化するが必須ではない（カーネル時間が伸びるほど `Timeout` / `OutOfMemory` に近づく）
- in-flight 無制限（仮説 (2)）の実際の役割はここにある: GPU エラー発生時点で commit 済み・未実行の
  コマンドバッファが多いほど巻き添えの範囲が広がる。差別化要因ではなく増幅要因

Chrome が安定な理由はこの機構では説明し切れていない（Chrome の forward は 245 s と遅く、時間ベースの
watchdog だけなら露出はむしろ増える）。生き残る差分は: Chrome 同梱 Dawn の版が `18eb229` と異なる、
GPU プロセス分離（別プロセスの working set）、Chrome 側の GPU プロセス watchdog が失敗を context loss
として顕在化させる、の 3 点。いずれも未測定。

## 3. 合成ストレス（M4 Max、モデル不要）

`scratchpad/stress/webgpu_stress.cpp`: Texture Flow と同形状（N=17,489、d_model 1536、12 heads、hd 128、
exact-SDPA の 1 GiB chunk = 14 chunk/attention、MLP 8192）の DiT 風ブロック 4 層、f16 重み。同一入力で
反復実行し出力 107 MB の FNV-1a ハッシュを照合。グラフ 600 ノード（≈10 コマンドバッファ/forward）、
gallocr 1,741 MB + 重み 264 MB。

| 条件 | 反復 | ビット一致 | forward 時間 | unified log の GPU/Metal エラー |
|---|---|---|---|---|
| 単独（16:02:21-16:04:05） | 20 | 20/20 | 4.2 → 5.8 s（単調増） | 0 件 |
| 3 プロセス同時（16:04:18-16:13:32） | 20 × 3 | 60/60 | 25.6 / 33.4 / 34.8 s（min/med/max） | 0 件 |

log の述語: `IOGPU` / `GPU Timeout` / `command buffer` / `kIOGPU` / `com.apple.Metal` の error 以上。

### 3.1 実モデル（Texture Flow 1024 MV、native WebGPU、M4 Max）— 追記 17:50-18:20

HF `TencentARC/Pixal3D` の `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv.safetensors`（5.55 GB）を
`tools/convert.py pixal3d_tex_flow_1024_mv` で f16 GGUF 化（2.78 GB、700 tensors、spec 32 と同サイズ）。
PyTorch 参照 latent は無いので、合成 fixture（`stress/make_synth_fixture.py`: N=17,489 の重複なし
ボクセル座標、N(0,1) の noise / cond / concat_cond、norm mean 0 / std 1、サンプラは tex 既定
gs=1.0）で `trellis-test-pixal3d-slat-sample --stage tex --backend webgpu --dump` を
`TRELLIS_NOFA=1` で 2 回連続実行し、per-step latent を比較した。

| run | 12 forward の合計 | forward あたり | NaN | unified log の GPU/Metal エラー |
|---|---|---|---|---|
| A（17:50:11-18:04:44） | 872.9 s | 59〜82 s | 0 | 0 件 |
| B（18:04:44-18:19:51） | 906.0 s | 66〜88 s | 0 | 0 件 |

`cpp_tex_x_step1..12.npy` と `cpp_tex_x_final.npy` の 13 テンソル全てが **ビット一致**（max|d| = 0）。
latent std の推移は 0.970 → 0.685（step 10）→ 0.865（final）で有限。メモリは spec 32 と同じ
（重み 2,647 MB、活性化 1,863 MB、デバイス常駐 4.79 GB）。step 3 の jump は出ていない。
verdict 行は「f32 参照なし」で FAIL 表示になるが、これは fixture 由来で計算の失敗ではない。

読み方: **M4 Max では実モデルでも再現せず**（2 run、24 forward）。M1 Pro（32 GB、GPU 4 GiB バッファ上限は同じ）より速く、メモリも 2 倍の
環境なので「バックエンドは健全」の証拠にはならない。「破損が起きるなら Metal エラーが log に出るか」の
判別も、破損が出なかったので未決。

### 3.2 ブラウザ（Chrome/WASM）— 完走・native と一致（19:29-20:49、途中 launchd 障害で中断）

**結論から: ブラウザ run は `RESULT: OK` で完走し、native run A と全 12 ステップで一致した。**
M4 Max では native も browser も clean で、step jump はどちらにも出ない。

| step | browser vs native run A（L2 相対） | cos | 参考: spec 32 の M1 Pro（browser vs native run 3） |
|---|---|---|---|
| 1 | 1.134e-08 | 1.0000000 | 1.1e-8 |
| 2 | 1.592e-05 | 1.0000000 | 5.1e-6 |
| 6 | 5.907e-05 | 1.0000000 | 2.6e-5 |
| 12 = final | **2.292e-04** | 1.0000000 | **2.0e-4** |

非有限値は全ステップで 0、`max|d|` は step 1 の 2.4e-7 から final の 3.4e-3 まで単調。
M1 Pro の clean run と同じ水準で、両プラットフォームの残差は f16 重み演算によるもの（spec 32 §11 の
「2 つの Dawn ビルドは互いに f32 より 3 倍近い」という観察と整合）。メモリはページ側も native と同一
（重み 2,647 MB、活性化 1,863 MB、デバイス常駐 4,789.9 MB）。

**forward 時間のばらつきは破損の指標にならない（反証データ）。** ブラウザの forward は
60.7 / 174.2 / 121.7 / 238.7 s … と大きくばらつき（平均 140.4 s、モジュール内合計 1,687.4 s、
実時間 80 分）、native の 59-82 s に対し最大 4 倍だったが、**結果は完全に clean だった**。
spec 32 §11 は「破損 forward は clean forward より遅い（213-240 s vs 193-200 s）」を
fault のシグネチャ候補として挙げているが、今回「遅いが clean」の実例が得られたので、
**時間だけで破損を検出することはできない**。ここは §5 の検出戦略にも効く（時間ベースの
カナリアは偽陽性を出す）。

以下は完走に至るまでの経緯（同じ症状に当たったときのため）。

WASM モジュールは emcc 6.0.9 でビルド成功（`web/texture/pixal3d_texture.{js,wasm}`、`web/ss/pixal3d_ss.*`、
`web/smoke/pixal3d_smoke.wasm`）。`web/` を 8199 で配信し、`web/texture/run_playwright.js` を
native と同じ GGUF・合成 fixture で起動した。

| 試行 | 結果 |
|---|---|
| 18:30:46 起動（既存ドライバ） | Chrome は起動しページも WASM も読み込んだが、14 分 20 秒後にページが閉じて中断。ドライバの catch 節は `page.$eval` を先に呼ぶため console ログの書き出し（`console.txt`）に到達せず、証拠が残らなかった |
| 18:47 以降 4 回（計装ドライバ / CDP ポート / Chrome for Testing 151 / サンドボックス無効） | **Chrome が起動直後に SIGABRT**。`bootstrap_check_in org.chromium.crashpad.child_port_handshake…: unknown error code (141)` → `ReadExactly: expected 4, observed 0` → `Received signal 6` |

**根本原因: launchd のユーザードメインが "Reentrancy avoided" で詰まっている。**
`launchctl limit` が Chrome と同一のエラーを返す:

```
$ launchctl limit
Could not print resource limits: 141: Reentrancy avoided
$ launchctl list          # 出力なし（同様に失敗）
```

エラー **141 = Reentrancy avoided** が Chrome の `bootstrap_check_in` が受け取る 141 と同じもの。
launchd がサービスのチェックインを拒否するため、mach サービスを登録するプロセスは軒並み失敗する:

| 症状 | 由来 |
|---|---|
| Chrome が起動直後に SIGABRT | crashpad の `child_port_handshake` が check-in できない |
| `pgrep` が `sysmond service not found` | sysmond の mach サービスを引けない |
| `nohup` が `can't detach from console` | 制御端末の切り離しに失敗 |
| Playwright の `kill EPERM` | 自分の子プロセスすら kill できない |

**ユーザー自身のターミナルでも同一の SIGABRT が再現**したので、シェル固有ではなくマシン全体。
リポジトリ・WASM ビルド・WebGPU とは無関係。ディスク 40 GB 空き、スワップ 0、総 RSS 52.9/64 GB で
メモリ枯渇でもない。稼働 8 時間、load average 10.87（5 分）/ 16.58（15 分）、プロセスは
node 48 / Chrome Helper 43 / zsh 40 / mcp@latest 32 / claude 32 / uv 27 と蓄積しており、
serena/mcp 系の約 30 プロセスが 7 時間 10 分前から常駐していた。

launchd は再起動できないので、**復旧には OS の再起動が要る**。18:45:23 にユーザーの Chrome が
再起動し（app 内に framework が 151.0.7922.174 / 152.0.7977.75 / 152.0.7977.76 の 3 版同居）、
最初の run が閉じた 18:45:06 と近接するのは、この時点で launchd が詰まって実行中の Chrome ごと
巻き込まれたと見るのが整合的。

**復旧（19:22 頃）**: OS 再起動なしで launchd が回復した（ユーザーが別手段で解消。`launchctl limit` が
正常値を返し `pgrep` も通るようになった）。ただし**障害中から生きていたシェルは古い bootstrap ポートを
握ったままで、そこから起動する Chrome は失敗し続けた**（mach bootstrap は親から継承されるため）。
新しいシェルから起動したら通った。同じ症状に当たったら、サービス側の復旧を確認したあと
**シェルを開き直す**こと。

**実行手順**:

```sh
cd <repo>/web && python3 -m http.server 8199 &     # 既に起動中なら不要
cd <repo>/web
node texture/run_playwright.js ~/nfs/weights/pixal3d/gguf/pixal3d_tex_flow_1024_mv.gguf \
     <scratchpad>/synth_tex_fixture <出力先> - 0
```

ブラウザ側 latent は `browser_tex_x_step<k>.npy` として出るので、native の
`<scratchpad>/tex_run_A/cpp_tex_x_step<k>.npy` と
`stress/compare_dumps.py <A> <browser> cpp_tex_ browser_tex_` で突き合わせる。spec 32 の M1 Pro では
browser vs native が step 1 で 1.1e-8、step 12 で 2.0e-4 だった。

### 3.3 PyTorch 参照との parity 判定 — native / browser とも PASS（21:00-21:30、A100 pod）

§3.1/§3.2 は「決定性」と「両バックエンドの一致」までで、**数値の正しさは未判定**だった。
PyTorch 参照を作って埋めた。

**参照の作り方**（`scratchpad/stress/ref_tex_from_synth.py`）。`tools/ref_pixal3d_hr_sample.py` の
`run_tex_stage()` をそのまま抜き出し、concat_cond だけ「前段 shape SLAT から計算」ではなく
**合成 fixture の `f32_tex_concat_cond.npy` をそのまま使う**形にした（合成 fixture に前段が無いため）。
サンプラは `TEX_SAMP`（steps=12, gs=1.0, gr=0.0, gi=(0.6,0.9), rescale_t=3.0）で
`pipeline_mv.json` 由来の公式値、pixal3d.cpp のテスト既定と同一。conditioning（DINOv3/NAF）と
decoder を通らないので natten / spconv を必要としない。共有 GPU クラスタの A100 80GB pod で実行。

**upstream への改変を 1 箇所入れた（要記録）**: FlashAttention は fp16/bf16 しか受け付けず、
参照の f32 run（`model.convert_to(torch.float32)`）が `RuntimeError: FlashAttention only support
fp16 and bf16 data type` で落ちる。upstream の `pixal3d/modules/sparse/attention/full_attn.py` は
xformers / flash_attn 系しかバックエンドを持たない（非 sparse 側の `ATTN_BACKEND=sdpa` とは別物）。
そこで varlen をシーケンス境界で切って `F.scaled_dot_product_attention` に流す `sdpa` 分岐を足した
（`scratchpad/stress/add_sdpa_backend.py`）。シーケンス間は元々 attend しないので
FlashAttention の varlen と数学的に同値。

**その改変の妥当性検証**: 同じ bf16 run を flash_attn と sdpa の両方で回して比較した。
step 1 で L2 相対 3.79e-4（cos 0.9999999）、12 ステップの Euler で累積して final 5.19e-3
（cos 0.9999866）。bf16 の丸め差として説明できる大きさで、**後述の判定量（2.3e-3）より
一桁大きいわけではない**点は注意（bf16 参照はしきい値の較正にしか使っていない）。

**判定**（テスト定義の rel = `max|d| / max|f32 参照|`、`test_pixal3d_slat_sample.cpp:165-170`）。
PASS 基準は `rel(mine, f32) <= max(2·rel(f32, bf16), 5e-2)`。

| 量 | 値 |
|---|---|
| f32 参照の max\|x\| | 4.8219 |
| rel(f32, bf16)（較正元） | 1.2841e-02 |
| **しきい値** = max(2×上, 5e-2) | **5.0000e-02** |
| **rel(native WebGPU, f32)** | **2.3309e-03 → PASS**（しきい値の 4.66%） |
| **rel(browser WASM, f32)** | **2.1191e-03 → PASS**（しきい値の 4.24%） |
| 参考: native と browser の差 | 6.9996e-04 |

L2 相対でも native 4.724e-04 / browser 4.739e-04（final、cos 0.9999999）で、
spec 32 §11 の native run 3（テスト定義 1.4616e-3）と同水準。per-step std は PyTorch 側
（0.9703 0.9397 0.9073 0.8724 0.8352 0.7960 0.7559 0.7180 0.6894 0.6849 0.7318 0.8650）と
native WebGPU が 4 桁目まで一致する。

**この節が言えること**: 合成 fixture という人工入力に対してだが、**native WebGPU と Chrome/WASM は
どちらも PyTorch f32 に対して正しい**（しきい値の 5% 未満）。§3.1/§3.2 の「決定的」「両者一致」と
合わせて、M4 Max 上のこの経路には step-jump 破損も数値の誤りも観測されない。
実写由来の `hr_sample` fixture での判定は依然として未実施（§6）。

## 4. spec 32 §11 の時刻から読める相関（未検証・観察のみ）

| run | lock 保持 | 結果 |
|---|---|---|
| Chrome/WASM | 05:07:58-05:57:28 | clean |
| native run 1 | **05:57:36**-06:39:52 | step 3 破損（Chrome run 終了の 8 秒後に開始） |
| native run 2 | 06:54:08-07:34:33 | step 3-4 破損 |
| native run 3 | 08:08:27-08:47:15 | clean |

Chrome の GPU プロセスは spec 31 §12.8 で 10-15 GB を保持していた。run 1 は Chrome run 直後で、
Playwright/Chrome の解放が完了していない可能性がある。また「sibling session の次ジョブが lock 待ち」は
GPU compute の co-tenant ではないが、既に起動していれば**メモリの co-tenant** ではある。
「no other GPU tenant」= 「メモリ圧迫なし」ではない点を、M1 Pro 側で確認する価値がある。

## 5. 推奨するデバッグ順（提案の順序を並べ替え）

1. **M1 Pro の unified log を破損 run の時刻窓で確認する**（最も安く、最も判別力が高い）:

   ```sh
   /usr/bin/log show --start "2026-09-07 05:50:00" --end "2026-09-07 08:50:00" \
     --predicate 'eventMessage CONTAINS "IOGPU" OR eventMessage CONTAINS[c] "GPU Timeout" OR eventMessage CONTAINS[c] "command buffer" OR eventMessage CONTAINS "kIOGPU" OR (subsystem == "com.apple.Metal" AND (messageType == error OR messageType == fault))' \
     --style compact
   ```

   zsh では `log` が組み込みに取られるので `/usr/bin/log` を使う。述語の動作は M4 Max で確認済み
   （`messageType == error` 単独で直近 1 日 819,301 件がヒット。空振りする述語ではない）。run 1/2 の step 3 付近（開始 +7〜12 分）に
   Metal/IOGPU エラーがあり run 3 に無ければ §2 の機構で確定。何も無ければ §2 は落ち、ggml-webgpu/Dawn
   内部の競合に絞る。
2. 破損 run の時刻に Chrome / Playwright / sibling セッションのプロセスとメモリ（`vm_stat`、
   `memory_pressure`）がどうだったかを突き合わせる（§4）。
3. **1 forward 内の first-divergence capture**（提案の手順 6）を前倒しする。既存の `TRELLIS_DBG_*` /
   `inter_` ダンプ（`src/flow_runner.cpp:60-61`）でブロック単位の出力をハッシュし、破損 forward で
   「どのバッチ以降が古い値のままか」を見る。§2 が真なら合否は次で決まる: 発散は **64 カーネル単位の
   コマンドバッファ境界ちょうどで始まり**、影響を受けた各テンソルは**そのバッファ位置に直前に書かれていた
   内容とバイト一致**する（gallocr はスロットを同一 forward 内でも再利用するので、直前の書き手は同一
   forward 内の先行 op か前 forward）。丸め程度の差やランダムなビット化けなら §2 の機構は否定される。
4. Dawn 側に検出を入れる（fork または local patch）: `QueueMTL.mm` の completed handler で
   `[cmdBuf status] == MTLCommandBufferStatusError` なら `[cmdBuf error]` をログし device loss へ倒す。
   これで無言失敗が「検出される失敗」になる。ggml-webgpu 側は device lost callback が
   ログのみ（`:3883-3892`）なので、abort に変えるか次の map で失敗させる。
5. `max_inflight_batches` の A/B（1 / 4 / 8 / 無制限）は、1〜3 で「Metal 層の資源圧迫」が示唆された
   場合にのみ意味を持つ。subgroup-matrix の A/B は不要（既に OFF）。attention chunk の sweep は
   機構が無いので後回し。
6. 破損検出の自動化: **forward 時間をカナリアに使わない**（§3.2 で「遅いが clean」の反例を観測済み）。
   各 forward の出力ハッシュを前 forward と比較するだけでも検出できない（値が変わる
   のは正常）。代わりに同一 forward を 2 回実行して一致を要求する「二重実行ゲート」を、疑わしい run
   だけに使う（コスト 2 倍）。

## 6. 実施していないこと

- M1 Pro 実機での再現・log 確認（このマシンに無い）
- **実写由来の** `hr_sample` fixture での parity 判定。§3.3 で合成 fixture に対する PyTorch f32 参照を
  作って native/browser とも PASS を確認したが、これは人工入力（乱数の cond/noise）に対する判定。
  実画像から DINOv3/NAF を通した条件での判定には `tools/ref_pixal3d_cond_slat.py` →
  `ref_pixal3d_slat_sample.py` → `ref_pixal3d_hr_sample.py` のチェーン全体が要る（natten/spconv が
  必要なので GPU pod で回す。ckpt は shape_hr 5.55 GB + 両デコーダ 1.9 GB の追加取得）
- GPU watchdog を意図的に起こす無限ループシェーダでの「Dawn が失敗を飲み込む」実証
  （ユーザー端末の表示が止まる／GPU 再起動を誘発しうるので GO 待ち）
- `max_inflight_batches=1` の A/B（破損が再現しないため判別不能）
- Chrome 同梱 Dawn のコマンドバッファ処理の確認

## 7. 生成物の所在

- 合成ストレス: `<scratchpad>/stress/webgpu_stress.cpp`、ログ `solo.log` / `cot_{A,B,C}.log`
- 実モデル: GGUF `~/nfs/weights/pixal3d/gguf/pixal3d_tex_flow_1024_mv.gguf`（元 ckpt は同ディレクトリ
  `ckpts/`）、合成 fixture `<scratchpad>/synth_tex_fixture/`、dump `<scratchpad>/tex_run_{A,B}/`、
  比較 `stress/compare_dumps.py`。fixture 付きの parity テストには依然 `hr_sample/`（PyTorch 参照）が要る
- emscripten 6.0.9 を `brew install emscripten` で導入済み（`/opt/homebrew/bin/emcc`、docs 指定版と一致）
- 本 checkout の `build-webgpu/`（gitignore 対象）: Dawn prebuilt は scratchpad `dawn/extracted/`
- Dawn ソース（metal/ 44 ファイル）: scratchpad `dawn-src/`
