# pixal3d.cpp

Portable C++ runtime for Pixal3D, targeting native GPU execution, browser WebGPU/WASM, CLI, and desktop/web applications.

> Development strategy: start from `pwilkin/trellis.cpp` because Pixal3D's current implementation is TRELLIS.2-backbone based, then add Pixal3D-specific conditioning and WebGPU support while preserving a single common C++ pipeline.

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

JavaScript/TypeScript should remain a thin application/binding layer.

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

ORT WebGPU remains an optional component if it materially simplifies an isolated submodel such as DINOv3; it is not the architectural center.

## Pixal3D-specific work on top of trellis.cpp

Primary additions:

- Pixal3D model/config/weight loading
- camera-aware pixel-aligned 3D projection
- `ProjGrid` / multiview projection
- projected conditioning
- `ProjectAttention`
- multiview fusion by averaging projected 3D conditions
- NAF high-resolution image features
- Pixal3D single-view and multiview pipeline parity
- WebGPU support for missing operations

Expected reusable areas from trellis.cpp include DINOv3, flow transformers, Euler/CFG sampling, dense SS decoder, sparse tensor infrastructure, sparse 3D convolution, shape/texture decoders, dual-grid/mesh extraction, remeshing/decimation/UV/GLB, CLI/server/desktop patterns, and GGML backend infrastructure.

## Target cascade

```text
SS
 -> Shape 512
 -> Shape 1024
 -> Texture 1024
 -> mesh / GLB
```

## Development order

1. Establish PyTorch golden tensor dumps.
2. Validate the inherited trellis.cpp baseline.
3. Add Pixal3D model metadata and weight conversion/loading.
4. Add `ProjectAttention`.
5. Add single-view camera projection and reach PyTorch parity.
6. Add NAF conditioning.
7. Add multiview projection and average fusion.
8. Reach native CUDA end-to-end Pixal3D parity.
9. Integrate/enable GGML WebGPU.
10. Port only unsupported Pixal3D operations to WGSL.
11. Build the WASM/browser target.
12. Build thin Web UI and native desktop UI around the same runtime.

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

Use identical serialized initial noise for PyTorch/native/WebGPU comparisons.

See `docs/PIXAL3D_*.md` for the detailed architecture and porting plan.
