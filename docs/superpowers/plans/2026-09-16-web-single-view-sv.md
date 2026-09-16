# Web Single-View Pixal3D SV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deployable browser single-image Pixal3D path using dedicated SV flow weights, canonical `mesh_scale=1.0`, and default/manual FOV without requiring `transforms.json`.

**Architecture:** Keep the existing WASM worker ABI and role-based model resolution. Add a Web input-mode adapter that synthesizes the one-frame camera package for SV, make model storage/source selection mode-aware, and preserve the current MV calibration path untouched.

**Tech Stack:** JavaScript ES modules, OPFS, Web Workers, Emscripten/WASM, WebGPU, Playwright/Node browser contract tests, Cloudflare Workers static assets.

**Spec:** `docs/superpowers/specs/2026-09-16-web-single-view-sv-design.md`

## Global Constraints

- Single-image mode must never silently run the MV flow family.
- Single-image default `mesh_scale` is exactly `1.0`.
- Single-image default FOV is exactly `0.3490658503988659` radians (20 degrees).
- Manual FOV must satisfy `0 < fov < pi`.
- Existing 4-view canonical-rig and explicit-transforms MV behavior must remain unchanged.
- No browser MoGe dependency in this PR.
- Public SV download must not be claimed unless a verified SV manifest with exact size/SHA256 exists.

---

### Task 1: Add single-view camera/input contract

**Files:**
- Create: `web/app/single_view.js`
- Create: `web/app/test_single_view.mjs`
- Modify: `web/package.json`

**Interfaces:**
- Produces `DEFAULT_SV_FOV`, `makeSingleViewTransforms(imageName, fovRad=DEFAULT_SV_FOV)`, `prepareSingleViewFile(file, fovRad)`.
- `makeSingleViewTransforms` returns `{camera_angle_x, mesh_scale:1.0, frames:[...]}` using canonical front c2w and the PR #5 distance equation.

- [ ] Write tests for exact 20-degree default, `mesh_scale=1.0`, expected distance, one frame, and rejection of `fov<=0`/`fov>=pi`.
- [ ] Run `node app/test_single_view.mjs` and verify RED.
- [ ] Implement the module with browser-safe APIs and deterministic metadata.
- [ ] Run the test and verify PASS.
- [ ] Add `test:sv` to `web/package.json`.

### Task 2: Make input UI mode-aware

**Files:**
- Modify: `web/app/index.html`
- Modify: `web/app/main.js`
- Modify: `web/real_e2e/calibration.js`
- Modify: `web/app/test_canonical_rig.mjs`

**Interfaces:**
- Input mode enum is `sv|mv`.
- SV mode supplies one image plus synthesized `transforms.json`; MV delegates to the existing calibrator.
- Main code can query `{mode, isReady(), getPatchedFiles(), getViewCount()}`.

- [ ] Extend browser contract tests so one image is accepted only in SV mode and existing four-view behavior remains bit-for-bit unchanged.
- [ ] Verify tests fail before implementation.
- [ ] Add mode selector and advanced FOV input; hide MV scale controls in SV mode.
- [ ] Wire SV file staging through `single_view.js`; keep MV calibrator logic intact.
- [ ] Run `npm run test:rig` and `npm run test:sv`.

### Task 3: Make model cache/source mode-aware

**Files:**
- Modify: `web/app/model_store.js`
- Modify: `web/app/release_store.js`
- Modify: `web/app/main.js`
- Create: `web/app/test_model_mode.mjs`
- Modify: `web/package.json`

**Interfaces:**
- Manifest compatibility helper maps model set to family `sv|mv` using explicit manifest metadata when present, falling back to `_sv.gguf`/`_mv.gguf` flow filenames.
- `resolveModelSource(mode)` returns family-specific manifest/base URLs.
- Active mode Generate requires cached manifest family to match input mode.

- [ ] Write tests proving SV rejects MV manifest and MV rejects SV manifest; source query parameters are independent.
- [ ] Verify RED.
- [ ] Implement family detection and separate SV source keys (`sv_manifest_url`, `sv_model_base_url`, `PIXAL3D_SV_MODEL_*`).
- [ ] Keep current MV defaults unchanged.
- [ ] In SV mode with no configured public manifest, disable release download with an actionable message while retaining local verified install.
- [ ] Run all Node contract tests.

### Task 4: Production build and browser smoke

**Files:**
- Modify: `scripts/build_web_dist.sh`
- Modify: `web/app/README.md`
- Modify: `.github/workflows/web-production-smoke.yml` if needed by the existing smoke harness.

**Interfaces:**
- Build continues bundling the existing MV manifest.
- SV public manifest is optional; no fake hash/size data is generated.

- [ ] Add production-smoke assertions for SV mode UI, one-image readiness, and fail-closed public SV download when no verified SV manifest exists.
- [ ] Run the smoke test and verify RED.
- [ ] Update build/README without weakening existing MV release checks.
- [ ] Run `npm run build` and existing production browser smoke.

### Task 5: PR review and deploy

**Files:** no new implementation files unless review finds a bounded bug.

- [ ] Open PR against `main` documenting that public one-click SV model download remains gated on a verified SV manifest, while local/custom verified SV sets work now.
- [ ] Review the complete diff for accidental MV regressions and silent MV fallback.
- [ ] Verify all GitHub Actions checks on the exact head SHA.
- [ ] Squash merge with `expected_head_sha` protection.
- [ ] Run the repository's production deployment command (`npm run deploy` from `web/`) using the existing Cloudflare configuration/authorized account.
- [ ] Open the deployed site and verify: MV page loads, SV mode is visible, one image synthesizes the 20-degree camera, and missing public SV manifest produces the intended actionable model message rather than running MV weights.
