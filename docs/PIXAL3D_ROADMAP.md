# Pixal3D Roadmap

## M0 — Repository/bootstrap

- [x] Create private `raven38/pixal3d.cpp`
- [x] Preserve full `pwilkin/trellis.cpp` Git history on `main`
- [ ] Configure local remotes as `origin=raven38/pixal3d.cpp`, `upstream=pwilkin/trellis.cpp`
- [x] Add Pixal3D planning documents
- [ ] Record exact trellis.cpp and ggml baseline SHAs
- [ ] Keep upstream MIT notices

## M1 — Native baseline

- [ ] Build inherited trellis.cpp unchanged
- [ ] Validate CUDA on RTX 4090
- [ ] Run available component tests
- [ ] Generate known-good TRELLIS.2 output
- [ ] Document ggml fork/patch delta

## M2 — Pixal3D single-view

- [ ] model/config loading
- [ ] projected conditioning tensor layout
- [ ] ProjectAttention
- [ ] camera-aware projection
- [ ] single-view golden tensor parity
- [ ] end-to-end shape parity

## M3 — NAF

- [ ] DINO-only fallback for bring-up
- [ ] exact NAF feature extraction
- [ ] neighborhood attention implementation
- [ ] high-resolution conditioning parity

## M4 — Pixal3D MV

- [ ] transforms.json parser
- [ ] camera convention tests
- [ ] per-view projection
- [ ] average fusion
- [ ] sequential-view memory path
- [ ] MV end-to-end parity

## M5 — Native release candidate

- [ ] SS
- [ ] Shape 512
- [ ] Shape 1024
- [ ] Texture 1024
- [ ] mesh/GLB
- [ ] `pixal3d-cli generate`
- [ ] benchmark native CUDA

## M6 — ggml WebGPU

- [ ] choose ggml integration strategy
- [ ] WASM/Emscripten build
- [ ] backend op inventory against Pixal3D graph
- [ ] automated WebGPU backend tests

## M7 — Missing WebGPU kernels

- [ ] 3D RoPE
- [ ] projection
- [ ] MV accumulation
- [ ] dense Conv3D
- [ ] sparse gather/scatter
- [ ] SparseConv3D
- [ ] NAF neighborhood attention

## M8 — Browser end-to-end

- [ ] image loading
- [ ] camera loading
- [ ] stage weight load/unload
- [ ] full generation
- [ ] GLB returned to JS
- [ ] memory profiling
- [ ] numerical comparison against CUDA

## M9 — Apps

### Web
- [ ] input UI
- [ ] MV UI
- [ ] progress/stage UI
- [ ] Three.js viewer
- [ ] GLB export

### Desktop
- [ ] native backend binding
- [ ] CUDA/Vulkan selector
- [ ] same core UX
