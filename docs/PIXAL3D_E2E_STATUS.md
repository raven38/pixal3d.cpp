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
