# Pixal3D browser E2E status

Implemented on `feat/browser-partial-e2e` in the requested order.

## 1. Fixture-input full model E2E

Browser target: `pixal3d-webgpu-full-e2e-wasm`
Harness: `web/full_e2e/`
C ABI: `pixal3d_full_fixture_run()`

Live stages:

```text
fixture views
 -> SS conditioning + SS Flow + SS decode
 -> Shape-512 conditioning + Flow
 -> shape decoder upsample + Pixal3D grid64 quantization
 -> Shape-1024 conditioning (NAF T=512) + Flow
```

Then:

```text
Texture conditioning: fixture fallback
 -> Texture Flow
 -> Shape Decode
 -> Texture Decode
 -> GLB
```

The report explicitly emits `FULL_E2E_LIVE_TEXTURE_COND=0`. This is **not** the v0.9 gate yet.

For deterministic fixture bundles, provide the stage noise files matching the reference run. The HR reference generator saves `shape_noise.npy` and `tex_noise.npy`; if a dedicated LR reference noise is used for the combined full-E2E bundle, store it as `lr_shape_noise.npy` to avoid ambiguity.

## 2. Real-input E2E

Shared C++ loader:

- `include/pixal3d_input.h`
- `src/pixal3d_input.cpp`

It loads `transforms.json` plus pre-matted RGBA frame files from a native directory or browser WORKERFS and produces the 512/1024 `Pixal3dView` arrays.

Browser target: `pixal3d-webgpu-real-geometry-wasm`
Harness: `web/real_e2e/`
C ABI: `pixal3d_real_geometry_run()`

Currently live:

```text
RGBA images + transforms.json
 -> SS
 -> Shape-512
 -> Shape-1024
 -> Shape Decode
 -> geometry GLB
```

The report emits `REAL_INPUT_TEXTURE_COND_BLOCKED=1`. A textured real-input full E2E cannot be truthfully declared until Texture-1024 conditioning is live on browser WebGPU.

## 3. Production postprocess

Shared implementation:

- `include/pixal3d_postprocess.h`
- `src/pixal3d_postprocess.cpp`

Pipeline:

```text
raw dual-grid mesh + decoded voxel PBR
 -> weld vertices
 -> fill small holes
 -> narrow-band dual-contour remesh
 -> component cleanup
 -> QEM decimation
 -> UV unwrap / box projection
 -> voxel PBR bake
 -> textured GLB
```

The fixture-injected partial E2E now uses this production-quality postprocess path instead of the earlier vertex-color-only GLB.

## Remaining blocker for true full browser E2E

`pixal3d_cond_slat_gpu(... {S=1024, R=64, naf_T=1024})` currently builds a NAF map whose source tensor exceeds the adapter/browser ~4 GiB `maxBufferSize` limit. The existing WebGPU path therefore cannot compute Texture-1024 conditioning from views.

Required next implementation:

- on-demand or block-chunked NAF@1024, and/or
- sparse-coordinate-gathered accumulation that never materializes the full `[1024, 1024^2]` map.

Once this lands, replace the fixture `tex_cond_global/tex_cond_proj` boundary with the live producer. Only then should `v0.9.0-browser-e2e` be tagged.
