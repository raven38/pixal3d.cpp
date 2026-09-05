# Pixal3D Roadmap

## M0 — Repository/bootstrap

- [x] Create private `raven38/pixal3d.cpp`
- [x] Preserve full `pwilkin/trellis.cpp` Git history on `main`
- [x] Configure local remotes as `origin=raven38/pixal3d.cpp`, `upstream=pwilkin/trellis.cpp`
- [x] Add Pixal3D planning documents
- [x] Record exact trellis.cpp and ggml baseline SHAs (see `PIXAL3D_UPSTREAM_POLICY.md`)
- [x] Keep upstream MIT notices (`LICENSE` retained)

## M1 — Native baseline

- [x] Build inherited trellis.cpp unchanged (Metal, macOS, 2026-09-06; CUDA on RTX 4090 pending)
- [ ] Validate CUDA on RTX 4090
- [x] Run available component tests (CUDA: c2s/sparse-conv/dinov3/ss-dec OK; shape-dec FAIL on a random-latent stand-in fixture — open)
- [ ] Generate known-good TRELLIS.2 output
- [ ] Document ggml fork/patch delta

## M2 — Pixal3D single-view

- [x] model/config loading (`pixal3d_*` converter components, `dit_detect_proj_attn`)
- [x] projected conditioning tensor layout (spec 30 §1)
- [x] ProjectAttention (`trellis-test-pixal3d-{ss,slat}-flow`; see spec 30 §5 for precision limits)
- [x] camera-aware projection (`proj_grid`, `trellis-test-proj-grid`)
- [x] single-view golden tensor parity (`trellis-test-pixal3d-cond-ss` V=1)
- [ ] end-to-end shape parity

## M3 — NAF

- [ ] DINO-only fallback for bring-up
- [x] exact NAF feature extraction (`naf.cpp`, T=128/512 parity)
- [x] neighborhood attention implementation (CPU, NATTEN clamped-window semantics)
- [ ] high-resolution conditioning parity

## M4 — Pixal3D MV

- [ ] transforms.json parser
- [x] camera convention tests (calc_mat exact vs reference)
- [x] per-view projection
- [x] average fusion (`pixal3d_cond_ss`, 4-view parity rel 2.5e-4)
- [x] sequential-view memory path (views accumulated one at a time)
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
