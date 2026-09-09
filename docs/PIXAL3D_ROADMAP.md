# Pixal3D Roadmap

> Current release acceptance criteria live in [`PIXAL3D_RELEASE_CHECKLIST.md`](PIXAL3D_RELEASE_CHECKLIST.md).
> Refreshed after PR #24/#25/#26/#27/#28/#29/#31/#33.

## M0 — Repository/bootstrap

- [x] Private `raven38/pixal3d.cpp` with preserved trellis.cpp history
- [x] Upstream policy / baseline SHAs / MIT notices
- [x] ggml WebGPU patch strategy documented

## M1 — Native baseline

- [x] Native Metal/CUDA builds and component tests
- [x] CUDA RTX 4090 validation
- [x] Linux/GCC build gate on `main` (#33)
- [ ] Known-good generic TRELLIS.2 product output (non-Pixal3D follow-up)

## M2 — Pixal3D single-view

- [x] model/config loading
- [x] projected conditioning + ProjectAttention
- [x] camera-aware projection
- [x] single-view golden tensor parity
- [ ] productized single-view Pixal3D path (MV is release focus)

## M3 — NAF

- [x] exact NAF feature extraction
- [x] neighborhood attention
- [x] high-resolution conditioning parity
- [ ] optional DINO-only debug fallback

## M4 — Pixal3D multiview

- [x] `transforms.json` parser / camera convention tests
- [x] per-view projection / average fusion / sequential-view memory path
- [x] MV end-to-end parity
- [x] `/generate-mv`
- [x] fail-closed explicit positive `mesh_scale`
- [x] Desktop client refuses missing `mesh_scale` before request (#19 / PR #27)
- [ ] automatic `mesh_scale` estimator: **not in release path**; PR #5 failed real calibration and is closed

## M5 — Native cascade / release candidate

- [x] SS / Shape-512 / Shape-1024 / Texture-1024
- [x] shape + texture decode
- [x] textured GLB export
- [x] native CUDA benchmark
- [x] CUDA `ss_decode` graph-support path exercised on NVIDIA L4 (#11)
- [ ] clean-install Windows release-candidate E2E (#18)
- [ ] clean-install Linux release-candidate E2E (#18)

## M6 — ggml WebGPU

- [x] WASM/Emscripten build
- [x] WebGPU op inventory and smoke tests
- [x] projection/MV accumulation
- [x] sparse gather/scatter / SparseConv3D
- [x] sparse texture/PBR decoder
- [x] NAF neighborhood attention
- [x] SS decoder routed to CPU in wasm because WebGPU Conv3D path was unsafe

## M7 — Browser memory/correctness hardening

- [x] sparse/chunked Texture-1024 conditioning
- [x] SS decoder silent-corruption workaround / support guard
- [x] output-head fusion / attention chunking / device-budget gate
- [x] DINO precision and Q8_0 fixes
- [x] MLP chunking implemented as **opt-in** (#12); measured WebGPU/NOFA peak did not improve, so it is not a release blocker
- [ ] native Dawn/WebGPU issue #2 remains under investigation; Chrome/browser path is separate

## M8 — Browser end-to-end

- [x] real RGBA image loading
- [x] `transforms.json` loading
- [x] SS → Shape-512 → Shape-1024 → Texture-1024
- [x] shape + texture decode
- [x] production postprocess
- [x] textured GLB returned to JS
- [x] official cyclops-class browser full E2E
- [x] browser-safe `remesh_res=512`
- [ ] browser 1536: explicitly outside initial Web alpha scope

## M9 — Apps

### Desktop / Trellis Studio

- [x] resident native `trellis-server` integration
- [x] single-image UI / viewer / gallery / GLB export
- [x] Pixal3D MV input / matching / reorder
- [x] explicit `mesh_scale` UI
- [x] 1024 / 1536 control
- [x] progress / cancel
- [x] managed model cache + safe delete (#16 / PR #25)
- [x] real Tauri/WebKitGTK Xvfb lifecycle smoke (#17 / PR #26)
- [x] MV preflight gate (#19 / PR #27)
- [ ] clean-install Windows/Linux E2E (#18)

### Web

- [x] production app shell + viewer/download (#20 / PR #28)
- [x] verified manifest/size/SHA256 OPFS installation (#15 / PR #28)
- [x] fast production browser UI/OPFS gate (#22 / PR #29)
- [ ] automatic persistent model delivery/version invalidation/safe delete (#9)
- [ ] storage quota + WebGPU/device-budget preflight UI (#21)
- [ ] real Chrome second-launch/no-retransfer/cache-delete release smoke

## M10 — Initial releases

### Desktop alpha

- [x] shared model manifest contract (#15 / PR #24)
- [x] safe managed model cache (#16 / PR #25)
- [x] Tauri Xvfb/WebKitGTK smoke (#17 / PR #26)
- [x] MV preflight (#19 / PR #27)
- [x] CUDA graph-support hardware validation (#11)
- [ ] generate final manifest from exact release model directory
- [ ] clean-install Windows/Linux E2E (#18)
- [ ] synchronize package/Tauri version with `v0.9.0-desktop-alpha`
- [ ] verify installer asset lookup for prerelease/tag-specific release
- [ ] tag `v0.9.0-desktop-alpha`

### Web alpha

- [x] production UI (#20 / PR #28)
- [x] verified local OPFS model install (#15 / PR #28)
- [x] production UI CI (#22 / PR #29)
- [ ] automatic persistence/version lifecycle (#9)
- [ ] storage/WebGPU preflight (#21)
- [ ] real Chrome/WebGPU full generation + cache reuse/delete release gate
- [ ] tag `v0.9.0-web-alpha`
