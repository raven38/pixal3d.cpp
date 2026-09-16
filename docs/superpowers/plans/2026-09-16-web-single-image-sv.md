# Web Single-Image Pixal3D SV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a browser-only one-image Pixal3D SV path that creates camera metadata automatically and feeds the existing WebGPU/WASM Pixal3D runtime without ever running MV weights at V=1.

**Architecture:** Split input/camera logic into pure modules that can be tested in Node, keep the existing WASM worker role-based model loading, and gate generation on manifest-derived `sv`/`mv` compatibility. Automatic browser FOV estimation uses the official MoGe-2 ViT-S ONNX export through pinned ONNX Runtime WebGPU; the public MV release remains unchanged until verified SV GGUF manifests are published.

**Tech Stack:** ES modules, browser Canvas/File APIs, Web Workers, WebGPU, ONNX Runtime Web 1.29.0, OPFS, existing Emscripten Pixal3D worker, Node contract tests, Playwright smoke tests.

**Spec:** `docs/superpowers/specs/2026-09-16-web-single-image-sv-design.md`

## Global Constraints

- Existing transforms.json multiview and four-view canonical-rig behavior must remain unchanged.
- One image must use the dedicated SV flow family; MV@V=1 is never a fallback.
- Single-view `mesh_scale` defaults to exactly 1.0 and is a projection gauge, not physical size.
- Invalid FOV, alpha matte, manifest family, or camera-estimator output fails closed before generation.
- Public MV model download remains the default until a verified public SV model manifest exists.
- Do not invent SV GGUF sizes or hashes.
- MoGe ONNX is lazy-loaded only when automatic one-image FOV estimation is needed.

---

### Task 1: Lock model-family and single-view camera contracts

**Files:**
- Create: `web/app/single_view.js`
- Create: `web/app/test_single_view.mjs`
- Modify: `web/package.json`

**Interfaces:**
- Produces `modelVariant(manifest): 'sv'|'mv'|'invalid'`
- Produces `validateFov(fov): boolean`
- Produces `singleViewDistance(fov, meshScale=1): number`
- Produces `singleViewTransforms(imageName, fov, meshScale=1): object`
- Produces `classifyInput(files): { mode, images, jsonFile, error }`

- [ ] Write Node tests for SV/MV/mixed manifests, 1-image/4-image/ambiguous routing, valid/invalid FOV, distance equation, and canonical front transform.
- [ ] Run `node web/app/test_single_view.mjs`; verify RED because the module does not exist.
- [ ] Implement the minimal pure functions with no DOM dependencies.
- [ ] Re-run the test and verify PASS.
- [ ] Add `test:single-view` to `web/package.json`.

### Task 2: Port official MoGe focal recovery as a pure tested module

**Files:**
- Create: `web/app/moge_focal.js`
- Create: `web/app/test_moge_focal.mjs`

**Interfaces:**
- `recoverFocalShift(points, mask, width, height, options?): { focal, shift }`
- `focalToNormalizedFx(focal, width, height): number`
- `focalToHorizontalFov(focal, width, height): number`

- [ ] Generate a synthetic point map using a known focal and shift under the official normalized-view-plane convention.
- [ ] Test recovery within tight numerical tolerance and rejection of degenerate support.
- [ ] Run the test and verify RED.
- [ ] Implement deterministic <=64x64 sampling, closed-form focal for a candidate shift, and bounded 1D golden-section search for shift.
- [ ] Re-run and verify PASS.

### Task 3: Add reference-compatible browser RGBA preprocessing

**Files:**
- Modify: `web/app/single_view.js`
- Modify: `web/app/test_single_view.mjs`

**Interfaces:**
- `preprocessSingleViewFile(file): Promise<{ file, blackImageData, width, height }>`

- [ ] Add browser-oriented contract tests for alpha bbox math through exported pure crop-bound helpers.
- [ ] Implement max-side 1024 resize, `alpha > 0.8` bbox, centered square 1.1-margin crop, RGBA PNG staging, and black compositing.
- [ ] Reject opaque/no-real-alpha, fully transparent, and no-pixel-above-threshold inputs.
- [ ] Verify pure tests; browser smoke will exercise File/Canvas behavior.

### Task 4: Add a lazy verified MoGe-2 ONNX browser estimator

**Files:**
- Create: `web/app/moge_estimator.js`
- Create: `web/app/moge_worker.js`
- Modify: `web/package.json`
- Modify: `scripts/build_web_dist.sh`

**Interfaces:**
- `estimateSingleViewFov({ blackImageData, onProgress }): Promise<{ fov, source, diagnostics }>`

- [ ] Pin `onnxruntime-web` to `1.29.0`.
- [ ] Add constants for official `Ruicheng/moge-2-vits-normal-onnx/model.onnx`, exact size 140852051, SHA256 `24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0`.
- [ ] Implement separate OPFS cache `pixal3d-camera-models-v1`, stream verification, and fail closed on mismatch.
- [ ] Worker loads vendored ORT Web assets, prefers `webgpu`, runs NCHW float32 image + `num_tokens`, returns raw points/mask.
- [ ] Main estimator calls `recoverFocalShift`, derives normalized fx and horizontal FOV, validates `0<fov<pi`, and reports diagnostics.
- [ ] Build script copies required ORT JS/WASM assets and the new modules into `web/dist`.
- [ ] Run `npm --prefix web install`, `npm --prefix web run build`, and verify no remote JS runtime dependency is introduced.

### Task 5: Extend the calibration/input controller with explicit SV mode

**Files:**
- Modify: `web/real_e2e/calibration.js`
- Modify: `web/app/main.js`
- Modify: `web/app/index.html`
- Modify: `web/app/test_canonical_rig.mjs`
- Create: `web/app/test_single_view_ui.mjs`

**Interfaces:**
- Calibration/controller exposes `getMode(): 'sv'|'mv'|null`
- `prepareSingleView({ estimateFov, manualFov }): Promise<File[]>`
- existing `getPatchedFiles()` remains MV-compatible

- [ ] Add tests proving one image enters SV mode while existing 4-image canonical behavior remains unchanged.
- [ ] Update UI copy to advertise “1 image (SV) or 4-view turntable (MV)”.
- [ ] Single image displays canonical scale 1.0 with no scale confirmation UI.
- [ ] Add advanced manual FOV input; empty value means automatic MoGe.
- [ ] Before run, preprocess/estimate camera and construct `[transforms.json, input.png]` in memory.
- [ ] Surface camera source/FOV diagnostics and estimator errors.

### Task 6: Gate worker execution on checkpoint family

**Files:**
- Modify: `web/app/main.js`
- Modify: `web/real_e2e/worker.js`
- Modify: `web/app/test_single_view.mjs`

**Interfaces:**
- Manifest family derived before enabling Generate.
- Worker message includes `mode: 'sv'|'mv'`.

- [ ] Generate remains disabled for SV input + MV manifest and MV input + SV manifest.
- [ ] Mixed/unknown flow families are rejected as invalid model sets.
- [ ] Worker independently rechecks family/mode before invoking WASM, so UI bypass cannot cause MV@V=1.
- [ ] Existing role resolution is retained; no duplicate SV worker or C++ graph.

### Task 7: Production smoke, docs, and PR

**Files:**
- Modify: `web/package.json`
- Modify: `web/app/run_release_gate.mjs` and/or existing production smoke harness as needed
- Modify: `README.md`
- Modify: `docs/PIXAL3D_RELEASE_CHECKLIST.md`

- [ ] Add test scripts so CI/local verification runs `test:rig`, `test:single-view`, `test:moge-focal`, and the new UI contract.
- [ ] Run `npm --prefix web run build` and all weightless Node tests.
- [ ] Run the existing production UI smoke to verify four-view MV regression behavior.
- [ ] Document that automatic camera estimation downloads ~141 MB MoGe ONNX lazily and that the default public GGUF release is still MV-only.
- [ ] Document exact local SV installation requirement: verified manifest + four `_sv.gguf` flows + shared models.
- [ ] Open PR against `main`, link Issue #4 and PR #5, and explicitly state the public SV-model publication blocker.
- [ ] Do not merge until CI is green and a real browser run with an actual SV model set has been recorded.
