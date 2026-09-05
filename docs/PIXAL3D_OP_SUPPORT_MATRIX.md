# Pixal3D Operation / Backend Support Matrix

This matrix describes the GPU operations required by the **browser WASM build** of `pixal3d.cpp`.

The browser architecture is fixed as:

```text
TypeScript UI
  -> pixal3d.wasm
  -> common C++ pipeline
  -> ggml WebGPU backend
  -> WebGPU
```

Revalidate only the **WebGPU operator coverage** against the exact ggml commit used by `pixal3d.cpp`, because backend support changes over time. This does **not** mean reconsidering whether the browser build uses WASM.

| Component | Preferred native | Preferred browser | Notes |
|---|---|---|---|
| DINOv3 | existing trellis/ggml path | ggml WebGPU; ORT optional | benchmark before committing |
| timestep embedding | C++/ggml | C++ host/WASM or backend | tiny tensor |
| Flow Linear/MLP | ggml | ggml WebGPU | expected straightforward |
| self-attention | ggml | ggml WebGPU | validate head_dim=128 |
| cross-attention | ggml | ggml WebGPU | used by ProjectAttention |
| ProjectAttention proj/add | ggml | ggml WebGPU | Pixal3D graph wiring |
| 3D RoPE | ggml/custom | custom backend if semantics differ | coordinate-aware |
| camera matrices | CPU | WASM | small |
| 3D -> image projection | CUDA/custom | WGSL backend op | fuse projection + sampling where useful |
| bilinear feature sampling | CUDA/custom | WGSL/backend | exact coordinate conventions matter |
| MV average fusion | GPU | WGSL/backend | ideally fused with projection accumulation |
| NAF | CUDA/custom | WGSL/backend | neighborhood attention is high-risk |
| SS dense Conv3D | trellis existing | WGSL/backend extension as needed | verify exact ggml coverage |
| pixel shuffle 3D | existing/custom | WGSL/backend | possible fusion candidate |
| occupancy -> coords | CPU/GPU | WASM first | optimize compaction later |
| sparse coordinate hash/dedup | CPU | WASM | cache maps |
| sparse neighbor map | CPU | WASM first | upload integer map to GPU |
| SparseConv3D feature math | trellis existing | WGSL/backend | major custom kernel if unsupported |
| C2S/S2C topology | CPU + GPU features | WASM topology + GPU features | avoid large feature readback |
| shape decoder activations | GPU | WebGPU | sigmoid/softplus/comparisons |
| flexible dual grid | CPU | WASM | geometry/topology |
| remesh/QEM/UV/GLB | CPU | WASM | reuse portable C++ |
| Euler / CFG orchestration | C++ | WASM C++ | common pipeline logic |
