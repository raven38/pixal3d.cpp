# Pixal3D browser E2E status

`feat/browser-partial-e2e` の現状。数値は測ったものだけを書き、未実施は未実施と書く。

## 0. 何が blocker だったか（解決済み）

`pixal3d_cond_slat_gpu(... {S=1024, R=64, naf_T=1024})`（texture 段）は、単一グラフのままだと
ggml gallocr が **1 本 11 344 MB** のバッファを要求する。主犯は NAF 出力 `[1024, 1024²]` f32
= 4 294 967 296 B で、実測済みの WebGPU `maxBufferSize` 4 294 967 292 B を **4 バイト**超える。

解決:

1. **sparse coords**: `pixal3d_cond_slat_gpu` に active voxel の coords を渡せるようにし、
   dense `R³` accumulator（2 GiB）を作らない。dense + `pixal3d_gather_proj` と全成分ビット一致。
2. **分割グラフ**: 1 view を 4 種のグラフに割る（DINOv3 / NAF encoder 2 枝 / block 行 stripe の
   RoPE / block chunk の neighborhood attention）。NAF 出力は一度も materialize せず、
   chunk ごとに projection tap だけ回収して accumulator へ足す。

単一バッファ最大 **11 344 MB → 3 332 MB**。数値差は hr 側だけで L2 rel 8.5e-09 / cos 1.0000000000
（chunk をまたぐ tap 加算の順序差）、global と lr はビット一致。詳細と内訳表は
`docs/PIXAL3D_WEBGPU_MEMORY.md` §11。

## 1. Real-input full E2E（本命）

Browser target: `pixal3d-webgpu-real-geometry-wasm`
Harness: `web/real_e2e/`（`run_playwright.js` あり）
C ABI: `pixal3d_real_full_run()`（geometry のみの `pixal3d_real_geometry_run()` も残置）

```text
RGBA images + transforms.json
 -> live SS conditioning / SS Flow / SS decode
 -> live Shape-512 conditioning / Flow
 -> shape decoder upsample + Pixal3D grid64 quantization
 -> live Shape-1024 conditioning (NAF T=512) / Flow
 -> live Texture-1024 conditioning (NAF T=1024, 分割グラフ)
 -> Texture Flow
 -> Shape Decode / Texture Decode
 -> production postprocess (weld / hole fill / narrow-band remesh / cleanup / QEM / UV / PBR bake)
 -> textured GLB
```

fixture 由来の neural 出力の注入は無い。report は
`REAL_INPUT_LIVE_TEXTURE_COND=1` / `REAL_INPUT_TEXTURE_COND_BLOCKED=0` を出す。

## 2. Fixture-input full model E2E

Browser target: `pixal3d-webgpu-full-e2e-wasm`
Harness: `web/full_e2e/`（`run_playwright.js` あり）
C ABI: `pixal3d_full_fixture_run()`

Texture conditioning は live に切り替え済み（`FULL_E2E_LIVE_TEXTURE_COND=1`）。
旧 fixture 経路は debug チェックボックス（C ABI の `tex_cond_fixture=1`）のときだけ通り、
そのとき report は `FULL_E2E_LIVE_TEXTURE_COND=0` を出す。

このターゲットは `s1024_images.npy` / `images_512.npy` / `camera_angle_x.npy` /
`transform_matrix.npy` / `mesh_scale.npy` と各段の noise を持つ fixture bundle を要求する。

## 3. Partial E2E（最小の integration gate）

Browser target: `pixal3d-webgpu-partial-e2e-wasm`、harness `web/partial_e2e/`

```text
Shape-1024 fixture -> Texture Flow -> Shape Decode -> Texture Decode -> production postprocess -> textured GLB
```

fixture は `PIXAL3D_DUMP_FIXTURE=<dir>` を付けた native の
`trellis-test-pixal3d-real-e2e` が実データから書き出せる
（`hr_coords` / `f32_shape_slat` / `f32_tex_concat_cond` / `tex_cond_global` /
`tex_cond_proj` / `tex_noise` / `tex_norm_{mean,std}`）。

## 4. Production postprocess

共通実装 `include/pixal3d_postprocess.h` / `src/pixal3d_postprocess.cpp`。
partial / fixture full / real full の 3 経路が同じ tail を使う。

## 5. 検証の位置づけ（重要）

- 分割グラフ版の conditioning は、**同一 backend の単一グラフ版**（PyTorch 参照で検証済みの経路）
  との同値性で確認している。
- **PyTorch 参照（`tex_cond_global.npy` / `tex_cond_proj.npy`）との突き合わせは未実施**。
  `tools/ref_pixal3d_cond_slat.py` は DINOv3 + NAF(natten) + spconv が要り CUDA 必須で、
  検証に使った Mac では生成できない。GPU pod で回すまで「PyTorch 基準の rel / cosine」は出せない。

## 7. 実測（2026-09-08, M4 Max / Chrome WebGPU vs Metal）

入力: `transforms.json` + pre-matted RGBA 4 view（1024²）。`mesh_scale` はどの手元データにも
無いので既定 1.0。view 0 が canonical front view である保証も無い —— **以下は統合とメモリの
gate であって、形状品質の評価ではない**。

以下は §7b の SS decoder 修正**後**の true full E2E（browser: `web/real_e2e/`、seed 1、
native: `trellis-test-pixal3d-real-e2e`、同じ 4 view）。

| stage | browser (Chrome/WebGPU) | native (Metal) |
|---|---|---|
| SS Flow 12 step | 115.3 s (22 forwards) | 144.4 s (FA) / 89.3 s (NOFA) |
| Shape-512 Flow | 64.2 s | 102.2 s / 63.5 s |
| Shape-1024 tokens | 11 907 | 11 964 / 12 083 |
| Shape-1024 Flow | 755.7 s | 600.9 s |
| **live Texture-1024 cond** | **41.4 s, graph peak 1 538 MB, resident 2 741 MB** | 37.9 s, 3 332 MB, 2 741 MB |
| Texture Flow 12 step | 489.3 s (12 forwards) | 331.8 s |
| production postprocess | remesh_res=512 → remesh 3 319 396 面 → QEM 490 292 面, atlas 4096 | remesh_res=1024 → remesh 15 422 816 面 → QEM 955 372 面 |
| textured GLB | V=399 782 F=490 292 29 035 652 B | V=831 007 F=955 372 47 992 476 B |
| **textured GLB bbox** | **(0.9560, 0.9781, 0.5833)** | (0.9571, 0.9815, 0.5965) |

postprocess の予算がブラウザだけ違う（remesh_res 512 / target_faces 500 000 対 1024 / 1 000 000）
のは wasm32 の 4 GiB ヒープ制約によるもので、face 数とメッシュの粗さの差はここから来ている。
神経回路の段も**完全に同条件ではない**: browser は `g_no_fa`（plain softmax）で回り SS decoder
だけ CPU backend、native は FlashAttention + 全段 Metal。したがって上の表は backend 統合と
メモリの gate であって、backend 間の数値 parity の測定ではない（parity は §7a の同一 fixture
比較で測っている）。

token 数がばらつくのは、SS decode の閾値が離散判定で、backend 間（および FA / SDPA 間）の
わずかな数値差がそのまま active voxel 数を変えるため。

ブラウザで踏んだ 3 つの落とし穴（いずれも「これらのターゲットが一度もブラウザで実行されて
いなかった」ことの現れ）:

1. `dit_N4096_dcond5_proj1: 184 unsupported node(s)` —— WebGPU backend に BF16 K/V の
   FLASH_ATTN_EXT が無い。full / real の E2E が `g_no_fa` を立てていなかった。
2. `std::bad_alloc` —— res=1024 の narrow-band remesh が 7.8M 頂点 / 15.6M 面を作り、
   wasm32 の 4 GiB ヒープに収まらない。`remesh_res` を足してブラウザは 512 で回す。
3. `thread constructor failed: Not supported` —— wasm ターゲットは pthread 無しでリンク
   しているので `std::thread` が生成できない。remesh の並列化を直列に落とした
   （候補ビットセットがスレッドごとに res³/8 バイト要るので、メモリ的にも効く）。

## 7a. Texture Flow の native vs browser per-step parity（同一 fixture）

fixture は native の real-input run が吐いた Shape-1024 fixture（N=12 083）。native は
`pixal3d-texture-run --texture ... <dump_prefix>`（Metal, `g_no_fa` は texture 経路が常に立てる）、
browser は `web/texture/run_playwright.js`（Chrome/WebGPU）。同じ条件・同じ noise。

| step | max abs | mean abs | L2 rel | cosine | bit identical |
|---|---|---|---|---|---|
| 1 | 3.184e-04 | 4.963e-06 | 6.708e-06 | 1.000000000 | no |
| 4 | 4.334e-04 | 1.433e-05 | 2.134e-05 | 1.000000000 | no |
| 8 | 1.572e-03 | 3.910e-05 | 7.608e-05 | 0.999999997 | no |
| 12 / x_final | 3.258e-03 | 1.394e-04 | 2.546e-04 | 0.999999968 | no |

非有限値は両側とも 0。誤差は step とともに単調に積み上がるだけで、跳ぶ step は無い
（spec 32 の step-jump 破損の兆候は出ていない）。実時間は browser 496.0 s / native 12 forwards。

## 7b. 解決済み: ブラウザの live geometry が X 軸に潰れていた（2026-09-08）

**症状**: browser の real-input E2E は完走して GLB を書くのに、形状が X 軸に潰れていた
（最終 GLB の bbox [0.101, 0.996, 0.980]。native は [0.905, 0.971, 0.578]）。

**切り分け**: 実入力パイプラインの各段に計測（座標の bbox / 重心、テンソルの
mean/std/min/max/非有限数）を入れて browser と native を並べた。

| stage | native (Metal) | browser 修正前 | browser 修正後 |
|---|---|---|---|
| `ss_cond_proj` | std 0.966560 | 0.966559 | 0.966559 |
| `ss_latent` | mean +0.007359 std 0.532235 | +0.007341 / 0.532413 | 同左 |
| **`ss_logits`** | mean −146.74 std 56.01 **nonfinite 0** | mean −27.60 std **12518** **nonfinite 21143** | mean −146.78 std 56.07 **nonfinite 0** |
| `ss_coords@64` | x[0..31] y[0..31] z[5..24] centroid(15.2,18.4,15.3) | **x[2..4]** y[0..31] z[0..31] | x[0..31] y[0..31] z[5..24] centroid(15.2,18.4,15.3) |
| `hr_coords@64` | N=12 083 x[1..63] z[11..48] | N=10 901 **x[4..10]** | N=11 907 x[1..62] z[11..48] |
| raw mesh bbox | (0.9652, 0.9692, 0.5763) | — | **(0.9537, 0.9750, 0.5799)** |
| 最終 textured GLB bbox | (0.9571, 0.9815, 0.5965) | **(0.1006, 0.9961, 0.9801)** | **(0.9560, 0.9781, 0.5833)** |

**根本原因**: ggml WebGPU backend は `GGML_OP_IM2COL_3D` も `GGML_OP_CONV_3D` も実装して
いない（`docs/PIXAL3D_WEBGPU_OP_GAP.md`: "every Conv3d in this file is unsupported
regardless of which branch compiles"）。SS decoder (`src/ss_decoder.cpp`) は全段が Conv3d で、
かつ `ss_decode` は `trellis_graph_dump` は呼ぶが **`check_graph_supported` を呼んでいなかった**
ため、FlashAttention のときのように実行前に弾かれず、黙って NaN 混じりの occupancy logits を
返していた。その logits から取った active voxel が薄いスラブになり、以降の Shape-512 /
Shape-1024 / decode が全部その上に積み上がっていた。

**修正**:

1. wasm ビルドでは **SS decoder だけ CPU backend に載せる**（`Model::load(path, -1)`。
   `trellis-test-pixal3d-ss-sample` の `dec_gpu=-1` と同じ扱い）。他の段は WebGPU のまま。
2. `ss_decode` に `check_graph_supported` を追加し、同種の「黙って壊れる」を実行前のエラーにする。

**native への回帰が無いことの確認**: guard を足した `src/ss_decoder.cpp` でリビルドした
`build-metal/trellis-test-pixal3d-real-e2e` は Metal で `REAL_GEOMETRY_RESULT: OK`
（V=4 098 362 F=8 244 512、`ss_logits nonfinite=0`）。Metal backend は Conv3D を持つので
guard は素通りする。CUDA は手元に無いので未確認（`ggml-cuda` は IM2COL_3D を実装しているので
通るはずだが実測していない）。`ss_decode` を呼ぶ翻訳単位は
`src/pixal3d_real_geometry_wasm.cpp` と `src/pixal3d_full_e2e_wasm.cpp` の 2 つだけで、
どちらも `SS_DEC_BACKEND` 対応済み（`grep -ln ss_decode src/*.cpp` で確認）。

**修正後の true full E2E**: browser が textured GLB を bbox (0.9560, 0.9781, 0.5833) で出力
（native (0.9571, 0.9815, 0.5965)）。4 視点レンダも native と同じシルエット・同じ色・
同じ破綻の仕方（頭部が欠ける = 入力ビュー由来で backend 由来ではない）になった。

**教訓**: `check_graph_supported` を通っているグラフ（DiT / conditioning / NAF）は未対応 op を
実行前に弾くが、通っていないグラフは黙って壊れる。新しいグラフを足したら必ず通すこと。

## 6. 走らせ方（このリポジトリでの実行手順）

WASM ビルド:

```sh
emcmake cmake -S . -B build-wasm-ss -DGGML_WEBGPU=ON -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DTRELLIS_WASM_SS=ON
cmake --build build-wasm-ss --target pixal3d-webgpu-ss-wasm pixal3d-webgpu-texture-wasm \
  pixal3d-webgpu-partial-e2e-wasm pixal3d-webgpu-full-e2e-wasm pixal3d-webgpu-real-geometry-wasm -j
```

必要な GGUF は 9 本: `dinov3` / `pixal3d_naf` / `pixal3d_ss_flow_mv` / `ss_dec` /
`pixal3d_shape_flow_512_mv` / `shape_dec` / `pixal3d_shape_flow_1024_mv` /
`pixal3d_tex_flow_1024_mv` / `tex_dec`。NAF だけ HF に無く、`valeoai/NAF` の release
`naf_release.pth` を safetensors 化してから `tools/convert.py pixal3d_naf`。

ブラウザ（リポジトリルートを配信してから）:

```sh
python3 -m http.server 8200 --directory .   # 別シェル
cd web
PIXAL3D_BASE_URL=http://127.0.0.1:8200/web/real_e2e/ \
  node real_e2e/run_playwright.js <gguf_dir> <views_dir> 1        # real-input full E2E
PIXAL3D_BASE_URL=http://127.0.0.1:8200/web/partial_e2e/ \
  node partial_e2e/run_playwright.js <tex_flow.gguf> <shape_dec.gguf> <tex_dec.gguf> <fixture_dir>
```

native の同一 C++ 経路（ブラウザへ投げる前の dry-run。browser と同じ SDPA 経路を踏むなら
`TRELLIS_NOFA=1`）:

```sh
cmake --build build-metal --target trellis-test-pixal3d-real-e2e
PIXAL3D_DUMP_FIXTURE=<dir> ./build-metal/trellis-test-pixal3d-real-e2e \
  <dinov3> <naf> <ss_flow> <ss_dec> <shape512> <shape_dec> <shape1024> \
  <views_dir> <out.glb> 1 <tex_flow> <tex_dec>
```

conditioning のメモリと数値だけを見るとき:

```sh
cmake --build build-metal --target trellis-test-pixal3d-cond-tex
TRELLIS_DBG_COND=1 ./build-metal/trellis-test-pixal3d-cond-tex <dinov3> <naf> <views_dir> 0 \
  --S 1024 --R 64 --naf-t 1024 --views 4 --stride 4 --save-prefix /tmp/tex1024
```
