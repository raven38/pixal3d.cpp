# Pixal3D initial release checklist

Status baseline: `main` after PR #24/#25/#26/#27/#28/#29/#31/#33. This file is the release acceptance source of truth; the roadmap is descriptive.

## Planned prerelease tags

- Desktop: `v0.9.0-desktop-alpha`
- Web: `v0.9.0-web-alpha`
- Model set: independent manifest version, e.g. `pixal3d-q8_0-v1`

Before publishing a `v0.9.0-*` Desktop artifact, synchronize the Tauri/npm package version (currently 0.6.0) with the release version. Installer paths that rely on `releases/latest` must also be verified for prerelease/tag-specific assets instead of assuming the latest stable release.

Every release note must record runtime commit, model-set/version, manifest SHA256, validated hardware/browser surfaces, and known issues.

## Common gates

| Requirement | Status | Evidence / owner |
|---|:---:|---|
| Shared model manifest contract: exact filenames/roles/bytes/SHA256 | ✅ | #15 / PR #24 |
| Manifest rejects incomplete/corrupt/mixed production set | ✅ | model-manifest CI |
| Linux/GCC C++ build gate on `main` | ✅ | PR #33 |
| Explicit `mesh_scale`; no silent production fallback | ✅ | #19 / PR #27 |
| Release model directory gets a real generated manifest | ⬜ | release packaging |
| Official cyclops + views4 result recorded for exact runtime/model set | ⬜ | release candidate run |

## Desktop alpha

Supported target: Trellis Studio, Windows x64 + Linux x86-64, resident native `trellis-server`, Pixal3D MV 1024/1536. CUDA/Vulkan may be advertised only on surfaces actually exercised by the release candidate. Native Dawn/WebGPU is outside the Desktop alpha support matrix while #2 remains open.

| Requirement | Status | Test class |
|---|:---:|---|
| Managed cache cannot delete external user files | ✅ #16 / PR #25 | CI: temp dir / traversal / symlink |
| Real Tauri/WebKitGTK window + Studio-spawned server lifecycle | ✅ #17 / PR #26 | Xvfb CI |
| MV metadata/image/alpha/mesh_scale preflight | ✅ #19 / PR #27 | headless UI CI |
| CUDA `ss_decode` graph-support path exercised on NVIDIA hardware | ✅ #11 | NVIDIA L4 validation |
| Clean-install Windows: installer → models → MV → textured GLB → viewer/save | ⬜ #18 | **real machine** |
| Clean-install Linux: installer → models → MV → textured GLB → viewer/save | ⬜ #18 | **real machine** |
| Installer downloads/verifies the exact release manifest/model set | ⬜ | clean-install release candidate |
| Package/Tauri version matches `v0.9.0-desktop-alpha` | ⬜ | packaging |

Desktop alpha deliberately requires an explicit positive `mesh_scale`. Automatic estimation from PR #5 is not part of the release path because the real calibration datasets produced large errors (cyclops expected ~1.0 → 1.261568; views4 expected ~0.206 → 0.381288).

## Web alpha

Supported target: Chrome/Chromium + WebGPU, resolution 1024 only. Inference stays in the common C++/WASM pipeline; JS is UI/file orchestration.

| Requirement | Status | Test class |
|---|:---:|---|
| Production app shell + GLB viewer/download | ✅ #20 / PR #28 | Chromium |
| Verified OPFS model-set install consumes manifest + size + SHA256 | ✅ #15 / PR #28 | Playwright fixture |
| Fast production UI + OPFS + manifest integrity gate | ✅ #22 / PR #29 | Playwright CI |
| Persistent automatic model delivery/version invalidation/safe cache delete | ⬜ #9 | CI + real Chrome storage |
| Storage quota + WebGPU/device-budget preflight UI | ⬜ #21 | headless + real browser |
| Full Chrome/WebGPU 1024 known-input generation → textured GLB | ⬜ | real Chrome release gate |
| Second launch reuses cached ~7.8 GB model set without retransferring it | ⬜ #9 | real Chrome |
| Cache deletion frees only origin-owned Pixal3D storage | ⬜ #9 | real Chrome |

Web alpha known limits: Chrome/Chromium only, WebGPU required, 1024 only, wasm32 4 GiB host address-space constraints, browser-safe 512 postprocess, and pre-matted RGBA + `transforms.json` input.

## Non-blockers / known issues

- #2 native Dawn/WebGPU intermittent corruption: not a Web/Chrome blocker and not a Desktop blocker unless native WebGPU is advertised.
- #12 MLP chunking exists as opt-in and reduces FA-path memory, but measured WebGPU/NOFA peak did not improve; it is not an initial release blocker.
- PR #5 auto `mesh_scale`: not release-ready; explicit/manual scale remains the supported contract.
- Browser 1536: unsupported for initial Web alpha.

## Release note template

```text
Runtime tag: <git tag>
Commit: <sha>
Model set: <manifest.model_set> <manifest.version>
Model manifest SHA256: <sha256 of manifest file>
Validated surfaces:
  Desktop Windows: <GPU/backend/result or N/A>
  Desktop Linux:   <GPU/backend/result or N/A>
  Web Chrome:      <browser/GPU/result or N/A>
Known issues: <issue numbers>
```

A blocker may be waived only by narrowing the advertised support matrix; do not silently waive a failed gate.
