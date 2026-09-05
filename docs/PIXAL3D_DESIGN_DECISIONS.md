# Pixal3D Design Decisions

## Browser runtime: WASM-first

Status: **Accepted**

The browser implementation will not reimplement the Pixal3D pipeline in TypeScript.

```text
Browser UI (TypeScript)
        |
        v
   pixal3d.wasm
        |
        v
common C++ pipeline
        |
        v
ggml WebGPU backend
        |
        v
      WebGPU
```

Consequences:

- pipeline sequencing, Euler/CFG sampling, model state, sparse topology logic, geometry, and GLB handling stay in shared C++ where practical;
- heavy tensor operations remain GPU-resident and are dispatched through WebGPU;
- Pixal3D-specific missing GPU operations are added to the WebGPU backend as WGSL kernels;
- TypeScript owns UI/application concerns, not inference logic;
- ORT WebGPU is optional for isolated submodels only when it has a clear benefit.

## Repository strategy

Status: **Accepted**

The repository preserves the full `pwilkin/trellis.cpp` Git history and develops independently as `raven38/pixal3d.cpp`.

Recommended remotes:

```text
origin    raven38/pixal3d.cpp
upstream  pwilkin/trellis.cpp
```

## WebGPU support matrix

Status: **Clarified**

The WebGPU operation matrix tracks what the WASM build can offload to the GPU. Changes in ggml operator coverage may change the number of custom WGSL kernels required, but they do not change the WASM-first architecture.
