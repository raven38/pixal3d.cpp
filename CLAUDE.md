# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`pixal3d.cpp` is a private fork of [`pwilkin/trellis.cpp`](https://github.com/pwilkin/trellis.cpp)
(a GGML-based C++ port of Microsoft TRELLIS.2-4B image→3D). The fork's purpose is to build
**Pixal3D** — a TRELLIS.2-backbone model with multiview, camera-aware conditioning — on top of the
inherited trellis.cpp pipeline, while keeping one common C++ pipeline across CUDA, Vulkan, and
future WebGPU/WASM targets. Read `PIXAL3D.md` first; it is the top-level map to the
Pixal3D-specific docs (`docs/PIXAL3D_ARCHITECTURE.md`, `PIXAL3D_PORTING_PLAN.md`,
`PIXAL3D_OP_SUPPORT_MATRIX.md`, `PIXAL3D_VALIDATION.md`, `PIXAL3D_DESIGN_DECISIONS.md`,
`PIXAL3D_UPSTREAM_POLICY.md`, `PIXAL3D_ROADMAP.md`).

Remotes: `origin` = private dev repo (`raven38/pixal3d.cpp`), `upstream` = read-only
(`pwilkin/trellis.cpp`), used only to inspect/selectively cherry-pick. See
`docs/PIXAL3D_UPSTREAM_POLICY.md` before merging or diverging from upstream — keep merges
infrequent and pin exact upstream/ggml SHAs while Pixal3D parity work is in progress; don't update
ggml and Pixal3D graph logic in the same commit.

Until native Pixal3D parity is reached, most work is porting/extending the inherited
trellis.cpp TRELLIS.2 pipeline (see `docs/PIXAL3D_PORTING_PLAN.md` phases) rather than modifying
Pixal3D-specific code that doesn't exist yet — check which phase is current before assuming
Pixal3D conditioning code (ProjectAttention, projection, NAF, multiview fusion) is present.

## Build

```sh
git submodule update --init --recursive
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON   # or -DGGML_CUDA=ON / -DGGML_HIP=ON
cmake --build build -j
```

macOS/Apple Silicon needs no backend flag — Metal is enabled automatically. WebGPU (native Dawn):
`-DGGML_WEBGPU=ON -DGGML_METAL=OFF -DCMAKE_PREFIX_PATH=<prebuilt Dawn>`; the configure step then
applies `patches/ggml-webgpu/*.patch` to the submodule working tree (`docs/GGML_FORK_DIFF.md`
"Local patches" — never commit the resulting submodule diff). Browser SS path: `scripts/build_wasm_ss.sh`
→ `web/ss/` (Worker + WORKERFS + JSPI; `web/ss/run_playwright.js` drives Chrome). A local, gitignored
`build-baseline-metal/` may exist from the Phase 1 baseline build; don't assume it's current.
The vendored `thirdparty/ggml` (branch `trellis-patches` of `pwilkin/ggml`, v0.15.1+10) already
ships the upstream `ggml-webgpu` backend (`-DGGML_WEBGPU=ON`, needs Dawn natively or emdawnwebgpu under
Emscripten); see `docs/spec/31-webgpu-bringup.md` for its state, gaps and build paths.

Studio (desktop/web frontend, `app/`):

```sh
cd app && npm install
npm run dev          # browser dev server
npm run tauri dev    # desktop shell
npm run build         # tsc --noEmit + production bundle
```

## Tests

Tests are standalone `trellis-test-*` executables (`src/test_*.cpp`), not registered in CTest.
Build and run only the target relevant to your change:

```sh
cmake --build build --target trellis-test-<name> && ./build/trellis-test-<name>
```

Available: `trellis-test-ss-flow`, `trellis-test-shape-flow`, `trellis-test-ss-sample`,
`trellis-test-birefnet`, `trellis-test-preprocess`, `trellis-test-deform`, `trellis-test-ss-dec`,
`trellis-test-slat-shape`, `trellis-test-dinov3`, `trellis-test-sparse-conv`, `trellis-test-bigop`,
`trellis-test-c2s`, `trellis-test-sparse-degenerate`, `trellis-test-shape-dec`, `trellis-test-ss-full`,
plus CUDA-only `trellis-test-fa-mask-overflow` / `trellis-test-fa-bf16-range`. Pixal3D stages:
`trellis-test-proj-grid`, `trellis-test-pixal3d-{cond-ss,cond-slat,cond-tex,ss-flow,ss-sample,slat-flow,slat-sample,shape-decode,tex-decode}`,
plus `trellis-test-pixal3d-real-e2e` (the browser real-input E2E's shared C++ driven natively —
`PIXAL3D_DUMP_FIXTURE=<dir>` writes the Shape-1024 fixture the browser partial E2E consumes)
(`[gpu]` arg: -1 = CPU backend, 0 = the build's GPU device; on WebGPU pass `TRELLIS_NOFA=1` to the
flow tests — the backend has no BF16 FlashAttention — and `dec_gpu=-1` to `ss-sample`, whose SS
decoder needs Conv3D). `trellis-webgpu-smoke` / `web/smoke` are the backend smoke tests.

`trellis-smoke` exercises the broader pipeline. Most neural parity tests need reference tensors
dumped by the matching `tools/ref_*.py` script (run via the `uv` venv — numpy, safetensors, gguf,
pillow, torch, timm) before the C++ test can compare against them. There is no coverage threshold;
when reporting test results, state the backend, hardware, and fixtures used — high-memory CUDA
regressions are explicit manual checks, not automated.

**Never validate only the final mesh/GLB.** When adding or debugging a pipeline stage, compare
intermediate tensors stage-by-stage against the PyTorch reference (image → DINO → projection → MV
fusion → ProjectAttention → every flow block → SS decoder → sparse flow → sparse decoder → mesh),
using identical serialized initial noise across PyTorch/native/WebGPU. This is the project's
validation principle (`PIXAL3D.md`), not optional diligence.

## High-level architecture

```
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

One runtime, multiple backends — never a separate C++ and TypeScript inference implementation.
The browser target will be WASM-first: TypeScript UI → `pixal3d.wasm` → common C++ pipeline →
ggml WebGPU backend → WebGPU. JS/TS stays a thin binding/application layer; inference logic stays
in shared C++.

### Inherited trellis.cpp pipeline (`src/`)

```
RGB image
  │  BiRefNet / RMBG  (background removal)        → RGBA cutout       [birefnet.cpp, deform_conv*]
  ▼
DINOv3 ViT-L/16 feature extractor                  → patch tokens     [dinov3.cpp]
  ▼
① Sparse-Structure flow DiT (dense 16³)            → active voxels    [dit.cpp, ss_decoder.cpp]
  ▼
② Shape-SLAT flow DiT (sparse) → FlexiDualGrid dec  → dual grid/mesh   [shape_decoder.cpp, dual_grid.cpp]
  ▼
③ Texture-SLAT flow DiT (sparse) → sparse U-Net dec → 6-ch PBR/voxel  [dit.cpp, sparse.cpp]
  ▼
textured mesh
  │ weld → narrow-band UDF dual-contour remesh → QEM decimate (CPU/CUDA/HIP/Vulkan) →
  │ cluster + xatlas unwrap → trilinear PBR bake (BVH snap) → Telea inpaint  [remesh_dc.cpp,
  │                                                            decimate_qem*, uv_bake.cpp, tri_bvh.cpp]
  ▼
UV-textured GLB (WebP PBR textures)                                        [mesh_glb.cpp]
```

All three flow stages share one Euler/CFG sampler (`FlowEulerGuidanceIntervalSampler`, 12 steps).
SDPA is FlashAttention with padded K/V (`src/dit.cpp::sdpa`) — required to fit the 1024 cascade's
~53k HR tokens in VRAM without NaNs (ggml's tiled FA NaNs on the unpadded last key-tile at that
token count); `--no-fa` restores the plain-softmax path for A/B tests. Every neural component here has a
matching PyTorch reference test — check `docs/spec/` (numbered `NN-<component>.md`, reverse-engineered
per-component specs, plus `IMPL_NOTES_*.md`) before touching `dit.cpp`, `dinov3.cpp`, `sparse.cpp`,
or the decoders; the divergence from the reference postprocess is tracked explicitly in
`docs/spec/27-reference-postprocess.md` / `28-divergence-matrix.md`.

Entry points: `trellis_cli_main.cpp`/`trellis_cli.cpp`/`trellis_args.cpp` build `trellis-cli`;
`trellis-server.cpp` (same CLI/args files) builds the resident HTTP server
(`GET /health`, `POST /generate`). `post-replay` / `post_replay.cpp` replays the postprocess alone
from a `TRELLIS_DUMP_POST` dump — use it to iterate on remesh/decimate/UV/bake without re-running
the flow stages.

### Pixal3D additions on top of this baseline (in progress — see porting plan phase status)

Per `PIXAL3D.md` / `docs/PIXAL3D_ARCHITECTURE.md`: Pixal3D model/config/weight loading,
camera-aware pixel-aligned 3D projection, `ProjGrid`/multiview projection, `ProjectAttention`
(`global_out = cross_attention(x, global_context); proj_out = linear(projected_context); output =
global_out + proj_out`), multiview fusion by averaging projected 3D conditions, NAF high-res image
features. Each input view is processed independently → projected into a shared 3D grid via camera
params → averaged; the fused condition size is independent of view count. Process views
sequentially where possible to bound memory, especially for the eventual browser target. Rule of
thumb for the CPU/WASM-vs-GPU split (`docs/PIXAL3D_ARCHITECTURE.md`): coordinates/topology (camera
matrices, sparse coord hashing, dedup/index maps, subdivision/dual-grid topology, GLB
serialization) are CPU/WASM candidates; large feature tensors (dense transformer math, projected
conditioning, DINO features, dense/sparse Conv3D, neighborhood attention) must stay GPU-resident.

### Studio app (`app/`)

Vite/TypeScript frontend (`app/src/`: `api.ts`, `viewer.ts`, `store.ts`, `settings.ts`, `tauri.ts`)
plus a Tauri Rust shell (`app/src-tauri/`) for the desktop build. Talks to `trellis-server` over
HTTP; gallery/settings persist via IndexedDB/local config, not through the C++ core.

### Tools (`tools/`, Python via `uv` venv)

`convert.py` (safetensors → GGUF, per-model key remap + 5D-conv reshapes), `quantize_gguf.py`,
`ref_*.py` (per-component PyTorch reference dumps consumed by the matching `trellis-test-*`
binary), `glb_metrics.py` (geometry/UV/material metrics for ours-vs-reference GLB comparison),
`render_glb*.py`/`render_mesh.py`/`render_points.py`/`render_color.py` (quick renders),
`mv_preview/` (Playwright-driven `<model-viewer>` multi-view capture, see its own README).

## Coding conventions

Four-space indent in C++/Rust, two spaces in TypeScript, braces on the same line. `snake_case` for
C++ files/functions, `PascalCase` for types/classes, `camelCase` in TypeScript. Test sources are
`src/test_<component>.cpp`. No repo-wide formatter/linter — match adjacent style and keep diffs
focused. Commit subjects are concise/imperative with scoped prefixes (`fix:`, `feat(app):`,
`cuda:`, `ci:`, `release:`); one logical change per commit. `.gitignore` excludes generated builds,
model weights, and reference dumps — keep those and machine-specific config out of commits/PRs.

`TRELLIS_DBG_*` env vars are debug-logging toggles only; there are otherwise no
behavior-driving env vars beyond the flags documented in `README.md` (`--res`, `--bg-removal`,
`--no-texture`, `--decim`, `--atlas`, `--box-uv`, `--seed`, `--require-gpu`, `--f32`, `--no-fa`) —
prefer flags over new env vars for anything that changes pipeline behavior.
