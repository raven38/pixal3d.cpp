# Pixal3D Porting Plan

## Phase 0 — Reference infrastructure

Create PyTorch scripts that dump:

- preprocessed image tensors
- DINO tokens/features
- projected feature grids
- fused MV conditioning
- ProjectAttention outputs
- per-block flow outputs
- latent tensors after every Euler step
- SS occupancy
- sparse coordinates/features
- decoder outputs
- mesh vertices/faces/attributes

Use deterministic fixtures. Generate one initial latent/noise tensor and feed the exact same tensor to all implementations.

## Phase 1 — Establish trellis.cpp baseline

Before modifying architecture:

- build trellis.cpp
- validate available CUDA path
- run component tests
- record baseline commit
- record `thirdparty/ggml` fork/branch/commit
- identify local patches made by trellis.cpp to ggml

## Phase 2 — Pixal3D weight/config support

Add Pixal3D model identifiers, stage configs, tensor-name mapping, converter changes, projected conditioning dimensions, and Pixal3D-specific attention metadata.

Do not implement MV yet.

## Phase 3 — ProjectAttention

Implement and validate independently against PyTorch:

```text
global_out = cross_attention(x, global_context)
proj_out   = linear(projected_context)
output     = global_out + proj_out
```

## Phase 4 — Single-view pixel-aligned projection

Implement camera-aware projection from the shared 3D grid into image feature maps, preserving camera convention, FOV projection, bilinear sampling, border behavior, coordinate convention, and dtype behavior.

Target: single-view conditioning parity.

## Phase 5 — NAF

Bring up in stages:

1. temporary DINO-feature duplication fallback to preserve expected channel count
2. exact NAF model and neighborhood attention
3. quality and numerical comparison

The fallback is only a debug path.

## Phase 6 — Multiview

Implement multiview projection and average fusion. Process views sequentially where possible to reduce peak memory.

Validate duplicate cameras, orthogonal synthetic views, canonical front-view convention, and variable view counts.

## Phase 7 — Native end-to-end parity

Run full:

```text
SS -> Shape 512 -> Shape 1024 -> Texture 1024 -> mesh/GLB
```

on CUDA before WebGPU optimization.

## Phase 8 — WebGPU backend integration

Inventory the exact ggml version/fork used by trellis.cpp.

Preferred path: bring upstream WebGPU backend changes into the trellis ggml fork while preserving trellis-specific patches.

Alternative: port trellis patches onto a newer upstream ggml with WebGPU.

Keep this integration separate from Pixal3D model logic.

## Phase 9 — Fill missing WebGPU ops

Likely high-risk operations:

- 3D coordinate RoPE semantics
- pixel-aligned projection
- dense Conv3D
- sparse Conv3D
- NAF neighborhood attention
- sparse gather/scatter helpers

Implement them in the backend rather than as a separate TypeScript compute pipeline.

## Phase 10 — WASM/browser target

Build the C++ runtime via Emscripten and expose a small stable API.

Conceptual API:

```cpp
pixal3d_context * pixal3d_create(const pixal3d_create_params *);
pixal3d_result    pixal3d_generate(pixal3d_context *, const pixal3d_input *);
void              pixal3d_destroy(pixal3d_context *);
```

JavaScript bindings should remain thin.

## Phase 11 — Applications

Web: upload views, camera input/transforms.json, progress, stage memory/status, 3D preview, GLB export.

Desktop: same UX concepts, native runtime, CUDA/Vulkan device selection.
