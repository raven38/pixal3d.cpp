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

| stage | browser (Chrome/WebGPU) | native (Metal) |
|---|---|---|
| SS Flow 12 step | 190.6 s | 144.4 s (FA) / 89.3 s (NOFA) |
| Shape-512 Flow | 161.7 s | 102.2 s / 63.5 s |
| Shape-1024 tokens | 10 901 | 11 964 / 12 083 |
| Shape-1024 Flow | 704.0 s | 600.9 s |
| **live Texture-1024 cond** | **47.3 s, graph peak 1 538 MB** | 37.9 s, 3 332 MB |
| Texture Flow 12 step | 487.3 s (39.8 s/forward) | 331.8 s |
| production postprocess | remesh_res=512 → QEM 484 398 面, atlas 4096 | remesh_res=1024 → 955 372 面 |

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

## 7b. 未解決: ブラウザの live geometry が X 軸につぶれる（2026-09-08 発見）

browser の real-input full E2E は**完走して textured GLB を書く**が、出てくる形状は板状につぶれている。

| GLB | bbox サイズ (x,y,z) | 出所 |
|---|---|---|
| browser full E2E | **[0.101, 0.996, 0.980]** | ブラウザの live shape latent |
| browser partial E2E | [0.968, 0.972, 0.579] | native の shape latent を fixture で注入 |
| native full E2E (NOFA) | [0.905, 0.971, 0.578] | native の live shape latent |

X 方向だけ約 1/9 に潰れている。**レンダの見え方ではなく実際の頂点座標**（4 視点レンダも板を映す）。

切り分けられている範囲:

- **shape decode / dual grid / texture flow / texture decode / production postprocess は
  ブラウザで正しい** —— partial E2E が native 由来の latent から native と同じ形を出している。
- **live Texture-1024 conditioning も正しい**（本文書 §0 の同値性、および partial との整合）。
- 疑わしいのは **ブラウザ側の SS → Shape-512 → upsample → Shape-1024 のどこか**。
  Shape-1024 の token 数も browser 10 901 / native 12 083 と 1 割少なく、潰れた形状と整合する。
- ブラウザ 2 run で token 数は 10 901 で一致するので、**再現性はある**（乱数の揺らぎではない）。

これは今回のタスクの blocker（Texture-1024 conditioning）とは別の、**新たに見つかった browser 固有の
不具合**。v0.9 を tag する前に潰す必要がある。次の一手は、ステージごとの latent を browser と
native で dump して最初に食い違う段を特定すること（`web/ss` の harness が SS / Shape-512 の
latent を吐けるので、そこから）。

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
