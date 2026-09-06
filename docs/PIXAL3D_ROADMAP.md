# Pixal3D Roadmap

## M0 — Repository/bootstrap

- [x] Create private `raven38/pixal3d.cpp`
- [x] Preserve full `pwilkin/trellis.cpp` Git history on `main`
- [x] Configure local remotes as `origin=raven38/pixal3d.cpp`, `upstream=pwilkin/trellis.cpp`
- [x] Add Pixal3D planning documents
- [x] Record exact trellis.cpp and ggml baseline SHAs (see `PIXAL3D_UPSTREAM_POLICY.md`)
- [x] Keep upstream MIT notices (`LICENSE` retained)

## M1 — Native baseline

- [x] Build inherited trellis.cpp unchanged (Metal on macOS; clean `2516c48` CUDA build on RTX 4090 at `/mnt/hdd1/pixal3d/pixal3d.cpp-baseline`)
- [x] Validate CUDA on RTX 4090 (trellis-cli, fa-mask-overflow / fa-bf16-range PASS)
- [x] Run available component tests (CUDA: c2s/sparse-conv/dinov3/ss-dec OK; the shape-dec FAIL on a random-latent stand-in fixture was an artifact — `trellis-test-pixal3d-shape-decode` matches the real-SLAT reference mesh to 0.001 voxel)
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
- [x] high-resolution conditioning parity (`pixal3d_cond_slat`, shape_512 fixture rel <=3e-4)

## M4 — Pixal3D MV

- [x] transforms.json parser (`include/transforms_json.h`/`src/transforms_json.cpp`, wired into `trellis-cli --views`)
- [x] camera convention tests (calc_mat exact vs reference)
- [x] per-view projection
- [x] average fusion (`pixal3d_cond_ss`, 4-view parity rel 2.5e-4)
- [x] sequential-view memory path (views accumulated one at a time)
- [x] MV end-to-end parity (per-stage: SS voxel IoU 0.998, shape-512/HR and texture latents, shape/tex decoders; `trellis-cli --views` GLB matches the reference GLB's bbox axis-for-axis, 946k vs 976k faces; the flat metallic channel matches 2 of 3 reference seeds — the seed-42 reference's small metallic region is a sampling outlier)

## M5 — Native release candidate

- [x] SS (`trellis-test-pixal3d-ss-sample`)
- [x] Shape 512 (`trellis-test-pixal3d-slat-sample`)
- [x] Shape 1024 (`trellis-test-pixal3d-slat-sample --stage shape_hr`)
- [x] Texture 1024 (`trellis-test-pixal3d-slat-sample --stage tex`)
- [x] mesh/GLB (`trellis-cli --views`, reference postprocess defaults: 1M faces, 4096 atlas, band 1)
- [x] `pixal3d-cli generate` → implemented as `trellis-cli --views DIR out.glb` (single shared pipeline binary)
- [x] benchmark native CUDA: RTX 4090, 4 views, 1024 cascade = 11.5 min wall / 9.9 GB host RSS with the GPU NAF (reference PyTorch low_vram: 5.2 min); HR shape 236 s and texture 171 s dominate

## M6 — ggml WebGPU

- [x] choose ggml integration strategy (vendored fork's own `ggml-webgpu`, unchanged pin + `patches/ggml-webgpu/`; `docs/spec/31-webgpu-bringup.md` §3/§7)
- [x] WASM/Emscripten build (`scripts/build_wasm_smoke.sh`, `scripts/build_wasm_ss.sh`)
- [x] backend op inventory against Pixal3D graph (`docs/PIXAL3D_WEBGPU_OP_GAP.md`; §8 = validated set)
- [x] automated WebGPU backend tests (`trellis-webgpu-smoke`; the `trellis-test-pixal3d-{cond-ss,ss-flow,ss-sample}` / `trellis-test-proj-grid` binaries run unchanged on the WebGPU device, spec 31 §9)

## M7 — Missing WebGPU kernels

- [x] 3D RoPE (no kernel needed: host-built even/odd index input replaces `ggml_arange`, `dit_rope_index`)
- [x] projection (no kernel needed: `get_rows` x4 + weighted sum on the device, `pixal3d_cond_ss_gpu`; bit-level parity with the host path)
- [x] MV accumulation (device-resident running average, `pixal3d_cond_ss_gpu`; peak memory flat in V)
- [ ] dense Conv3D (SS decoder still runs on the CPU backend after a WebGPU flow)
- [x] large-dispatch ops (`patches/ggml-webgpu/0003`: 2D dispatch for soft_max/sum_rows/norm/get_rows/concat/pad/repeat, cpy `gid.y` fix; `trellis-webgpu-ops`)
- [ ] sparse gather/scatter
- [ ] SparseConv3D
- [x] NAF neighborhood attention (no kernel needed: block-window formulation as `get_rows` + batched `mul_mat` + `soft_max`, `naf_build`; GroupNorm/reflect-pad/avg-pool lowered to `norm`/`concat`/`sum_rows`; spec 31 §10.3)

## M8 — Browser end-to-end

- [x] SS stage in the browser: DINOv3 -> projection -> MV fusion -> ProjectAttention SS flow -> 12-step sampler, one WASM module on ggml WebGPU (`web/ss/`, spec 31 §9); JS only mounts files and prints
- [x] Shape-512 stage in the browser: DINOv3 -> NAF -> lr/hr projections -> MV fusion -> sparse ProjectAttention shape flow -> 12-step sampler, same module (`pixal3d_shape512_run`, `web/shape512/`, spec 31 §10.7)
- [ ] image loading (fixture `.npy` views today; PNG + BiRefNet/RMBG cutout not ported)
- [ ] camera loading (fixture `transform_matrix` today; `transforms.json` parsing not wired)
- [x] stage weight load/unload (SS stage: DINOv3 freed before the flow weights load, WORKERFS-backed GGUF streaming)
- [ ] full generation
- [ ] GLB returned to JS
- [x] memory profiling (SS stage, buffer-allocation accounting: `docs/PIXAL3D_WEBGPU_MEMORY.md` §6; Shape-512 stage §7)
- [x] numerical comparison against CUDA (SS stage: spec 31 §9 -- voxel IoU 1.000 vs the CUDA production run; Shape-512 stage: spec 31 §10.6/10.7)

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
