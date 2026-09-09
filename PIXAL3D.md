# pixal3d.cpp

Portable C++ runtime for Pixal3D, targeting native GPU execution, browser WebGPU/WASM, CLI, and desktop/web applications.

> Development strategy: start from `pwilkin/trellis.cpp` because Pixal3D's current implementation is TRELLIS.2-backbone based, then add Pixal3D-specific conditioning and WebGPU support while preserving a single common C++ pipeline.

## Current status (2026-09-09)

The native Pixal3D multiview cascade and browser WebGPU/WASM full pipeline are implemented end-to-end. `main` now includes the initial release infrastructure from PR #24/#25/#26/#27/#28/#29/#31/#33:

- native MV CLI and `/generate-mv`
- fail-closed explicit positive `mesh_scale`
- Desktop MV preflight for metadata, matching, matrices/FOV and alpha mattes
- Studio-managed model cache with external-file-safe deletion
- real Tauri/WebKitGTK Xvfb lifecycle CI
- browser SS → Shape-512 → Shape-1024 → Texture-1024 → decode → production postprocess → textured GLB
- production Web app with verified manifest/size/SHA256 OPFS install and GLB viewer/download
- fast Playwright production UI/OPFS gate
- Linux/GCC C++ build gate

Initial release gates are tracked in [`docs/PIXAL3D_RELEASE_CHECKLIST.md`](docs/PIXAL3D_RELEASE_CHECKLIST.md).

### Initial Desktop alpha scope

- Windows x64 and Linux x86-64
- native resident `trellis-server`
- 1024 / 1536 MV generation
- pre-matted RGBA + `transforms.json`
- explicit `mesh_scale` (automatic estimation is not a supported release path)
- CUDA/Vulkan only where the release candidate is actually validated
- native WebGPU is not advertised while issue #2 remains open

### Initial Web alpha scope

- Chrome / Chromium only
- WebGPU required
- 1024 only
- pre-matted RGBA + `transforms.json`
- browser-safe wasm32 postprocess/memory budgets
- verified local OPFS model-set installation is implemented; automatic persistent model delivery/version lifecycle remains tracked by #9

Planned prerelease tags: `v0.9.0-desktop-alpha` and `v0.9.0-web-alpha`.

## Repository strategy

This repository preserves the full `trellis.cpp` Git history and continues development as an independent private repository.

Recommended remotes:

```text
origin    git@github.com:raven38/pixal3d.cpp.git
upstream  https://github.com/pwilkin/trellis.cpp.git
```

## Browser architecture

The browser implementation is **WASM-first**. It is not a second inference pipeline written in TypeScript.

```text
TypeScript UI
  -> pixal3d.wasm
  -> common C++ pipeline
  -> ggml WebGPU backend
  -> WebGPU
```

JavaScript/TypeScript remains a thin application/binding/file layer.

## High-level architecture

```text
                    PyTorch Pixal3D
                    golden reference
                          |
                          v
                     pixal3d.cpp
                common C++ pipeline
                          |
          +---------------+---------------+
          |               |               |
        CUDA            Vulkan          WebGPU
          |               |               |
         CLI           Desktop       WASM / Browser
```

## Pixal3D-specific work on top of trellis.cpp

Primary additions:

- Pixal3D model/config/weight loading
- camera-aware pixel-aligned 3D projection
- `ProjGrid` / multiview projection
- projected conditioning / `ProjectAttention`
- multiview fusion by averaging projected 3D conditions
- NAF high-resolution image features
- native and browser end-to-end Pixal3D paths
- browser memory chunking/budget gates
- CLI/server/desktop/web integration

Reusable trellis.cpp areas include DINOv3, flow transformers, Euler/CFG sampling, dense SS decoder, sparse tensor infrastructure, sparse 3D convolution, shape/texture decoders, dual-grid/mesh extraction, remeshing/decimation/UV/GLB, CLI/server/desktop patterns, and GGML backend infrastructure.

## Target cascade

```text
SS
 -> Shape 512
 -> Shape 1024
 -> Texture 1024
 -> mesh / GLB
```

## Validation principle

Never debug only the final mesh. Compare tensors stage-by-stage:

```text
image
 -> DINO
 -> projection
 -> MV fusion
 -> ProjectAttention
 -> every flow block
 -> SS decoder
 -> sparse flow
 -> sparse decoder
 -> mesh
```

Use identical serialized initial noise for PyTorch/native/WebGPU comparisons. Release support claims additionally require the acceptance matrix in `docs/PIXAL3D_RELEASE_CHECKLIST.md`.

See `docs/PIXAL3D_*.md` for detailed architecture, memory measurements, E2E status, and porting notes.
