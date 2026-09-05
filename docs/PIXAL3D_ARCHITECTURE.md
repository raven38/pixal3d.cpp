# Pixal3D Architecture

## Core design decision

`pixal3d.cpp` is one runtime with multiple execution backends.

Do not design this as separate native C++ and JavaScript inference implementations.

```text
                         pixal3d.cpp
                    common model/pipeline
                           |
             +-------------+-------------+
             |             |             |
           CUDA          Vulkan        WebGPU
             |             |             |
            CLI         Desktop      WASM/browser
```

TypeScript owns application UX, file selection, camera UI, progress reporting, viewer integration, and invoking the WASM API. Inference logic should stay in shared C++ where practical.

## Browser target

The browser target is **WASM-first**. WebGPU is the GPU execution backend used from that WASM runtime; it is not a replacement for WASM.

```text
Web application (TypeScript)
        |
        v
pixal3d WASM bindings
        |
        v
common C++ pipeline
        |
        +--> GGML WebGPU backend
        |
        +--> Pixal3D-specific WGSL kernels
        |
        +--> CPU/WASM topology + geometry
```

ORT WebGPU is optional only if it clearly simplifies an isolated submodel such as DINOv3.

## Pixal3D MV conditioning

Each input view is processed independently, projected into a common 3D grid using camera parameters, then fused by averaging.

```text
view_0 -> image features -> project to 3D --+
view_1 -> image features -> project to 3D --+--> average --> projected condition
...                                        |
view_V -> image features -> project to 3D --+
```

The fused condition size is independent of the number of views. For browser memory use, process views sequentially where possible.

## Pipeline

```text
SS
 -> Shape 512
 -> Shape 1024
 -> Texture 1024
```

Stage weights should be unloadable between stages, especially in the browser.

## CPU/WASM vs GPU split

Good CPU/WASM candidates:

- camera matrix preparation
- sparse coordinate hashing
- unique/dedup/index-map generation
- subdivision topology
- dual-grid topology
- remeshing / decimation orchestration
- GLB serialization

GPU candidates:

- dense transformer math
- projected conditioning
- DINO features
- dense Conv3D
- sparse Conv3D feature calculation
- sparse gather/scatter involving large feature tensors
- neighborhood attention

Rule: coordinates/topology are small and may move through WASM; large feature tensors should remain GPU-resident.
