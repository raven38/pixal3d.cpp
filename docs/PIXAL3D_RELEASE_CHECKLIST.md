# Pixal3D initial release checklist

Status baseline: `main` after PR #24/#25/#26/#27/#28/#29/#31/#33. This file is the release acceptance source of truth; the roadmap is descriptive.

## Planned prerelease tags

- Desktop: `v0.9.0-desktop-alpha`
- Web: `v0.9.0-web-alpha`
- Model set: Desktop `pixal3d-f16` / `v1`, Web `pixal3d-q8_0` / `v1` (see "Fixed initial release model sets" below)

Desktop artifact metadata carries the **numeric** version `0.9.0` (`app/package.json`,
`app/package-lock.json`, `app/src-tauri/tauri.conf.json`, `app/src-tauri/Cargo.toml`,
`app/src-tauri/Cargo.lock` — all four in agreement, was 0.6.0). The `-desktop-alpha` prerelease
marker lives only in the git tag, because Tauri's Windows bundler rejects a version containing a
prerelease identifier (`app version cannot have build metadata or pre-release identifier`,
tauri-apps/tauri#5286, closed as not planned). So the mapping is: tag `v0.9.0-desktop-alpha` →
artifact metadata `0.9.0`, and a release note must state both. Rust is not installed on the
reference machine, so `cargo`/`tauri build` was not run locally for this change; `npm run build`
(tsc --noEmit + vite) passes, and the Linux `tauri build --no-bundle` gate in
`.github/workflows/studio-tauri-xvfb.yml` covers the Rust side. The installers resolve a concrete release before downloading anything (#36). `--tag`/`-Tag` takes
an explicit tag, `latest` the newest stable, `latest-prerelease` the newest release *including*
prereleases; `/releases/latest` (which skips prereleases) is reached only by the explicit `latest`
channel, so an explicitly named tag or `latest-prerelease` can never fall back to stable assets.
An explicit tag is rejected unless the API returns that exact tag, and tags are restricted to
`[A-Za-z0-9._+-]` so a tag cannot smuggle a URL fragment or path segment. Every expected asset must
be present in the resolved release — the check is an exact literal match — otherwise the install
stops with the resolved tag and the actual asset list. A receipt is written to
`<config dir>/release.json`: repo, tag, release id, the tag's commit SHA (resolved via
`/commits/<tag>`, because `target_commitish` may be a branch name), requested tag, runtime bundle,
backend and model set.

Behaviour change: the desktop app asset is now required. Previously a missing AppImage / setup.exe
only warned and left a runtime-only install; now the install stops before downloading anything.
Pass `--skip-app` / `-SkipApp` for a runtime-only install. A prerelease Desktop alpha is therefore installed as:

```sh
./install/install.sh --repo raven38/pixal3d.cpp --tag v0.9.0-desktop-alpha    # Linux
./install/install.ps1 -Repo raven38/pixal3d.cpp -Tag v0.9.0-desktop-alpha     # Windows
```

While this repository is private, both installers need `GITHUB_TOKEN`/`GH_TOKEN` with `repo`
access — the public asset URL 404s, so with a token they download through the API asset endpoint.
Making the alpha release public is the alternative; that is a distribution decision, not a script
change.

Every release note must record runtime commit, model-set/version, manifest SHA256, validated hardware/browser surfaces, and known issues.

## Fixed initial release model sets

Two manifests are fixed, both generated from real files with `tools/model_manifest.py` and both
using the consumer-mandated filename `pixal3d-models.json` (hardcoded in
`app/src-tauri/src/model_cache.rs` and `web/app/model_store.js`):

| | Desktop release identity | Web release identity |
|---|---|---|
| `model_set` / `version` | `pixal3d-f16` / `v1` | `pixal3d-q8_0` / `v1` |
| Manifest | `models/pixal3d-f16-v1/pixal3d-models.json` | `models/pixal3d-q8_0-v1/pixal3d-models.json` |
| Manifest SHA256 | `722f94c360da5e0dad2651a070489cd8c3de9e00c616b5f88607db41bb0588cd` | `5648ee78259808ff2b664574a666cd880c2c2b584ab8cd225bf80480e17d669f` |
| Total model bytes | 13,660,816,128 (12.72 GiB) | 8,091,975,360 (7.54 GiB) |

**This deviates from #34's original acceptance condition** ("Desktop/Web consume the same
manifest"). Both frontends consume the same *contract* — one schema, one filename, one generator,
the same nine roles — but not the same file, because a single set cannot satisfy both requirements:
the browser cannot afford F16 weights and native must not silently ship quantized weights while
parity work is open. Narrowing to one shared set means either shipping Q8_0 on Desktop (losing
reference parity) or F16 on Web (3.4 % budget headroom). Recorded here rather than waived silently.

The split follows `docs/PIXAL3D_WEBGPU_MEMORY.md` §12: native stays F16 for reference parity, while
the browser needs Q8_0 because the F16 texture flow leaves only 3.4 % of the 4095 MB WebGPU budget
at N=17690 tokens (Q8_0: 34 %). Both manifests describe the same nine required roles.

### Per-file bytes

| role | file | f16 bytes | Q8_0 bytes | Q8_0 GiB |
|---|---|---:|---:|---:|
| image_encoder | `dinov3.gguf` | 606,773,440 | 323,657,920 | 0.30 |
| naf | `pixal3d_naf.gguf` | 1,334,656 | 1,334,656 | 0.00 |
| ss_flow | `pixal3d_ss_flow_mv.gguf` | 2,681,053,184 | 1,426,559,744 | 1.33 |
| ss_decoder | `ss_dec.gguf` | 147,379,392 | 147,379,392 | 0.14 |
| shape_flow_512 | `pixal3d_shape_flow_512_mv.gguf` | 2,775,572,640 | 1,476,761,760 | 1.38 |
| shape_decoder | `shape_dec.gguf` | 948,745,344 | 881,361,568 | 0.82 |
| shape_flow_1024 | `pixal3d_shape_flow_1024_mv.gguf` | 2,775,572,640 | 1,476,761,760 | 1.38 |
| texture_flow_1024 | `pixal3d_tex_flow_1024_mv.gguf` | 2,775,670,944 | 1,476,813,984 | 1.38 |
| texture_decoder | `tex_dec.gguf` | 948,713,888 | 881,344,576 | 0.82 |

The F16 set is F16 weights with F32 norms/biases throughout (`gguf.GGUFReader`, 2026-09-09):
`dinov3` F32:219/F16:99, `pixal3d_naf` F32:27/F16:10, each flow DiT F16:365/F32:335,
`ss_dec` F32:54/F16:20, `shape_dec` F32:182/F16:110, `tex_dec` F32:178/F16:106.

**The Q8_0 set was regenerated on 2026-09-09 and is not byte-identical to the one measured in
§13.** `tools/quantize_gguf.py` quantized 96 `dinov3` tensors here (606,773,440 → 323,657,920 B),
whereas §13 lists `dinov3` as untouched at 579 MB; the flow DiTs and decoders match §13's counts
(244–245 / 66–70 tensors). §13's native E2E evidence (mean silhouette IoU 0.9105 vs. F16 0.9089,
gate PASS) therefore does **not** transfer byte-for-byte to `pixal3d-q8_0 v1`; re-running the
official E2E against this exact manifest is the open release-candidate gate below.

### E2E quality gate for the fixed sets (2026-09-10, macOS/M4 Max, Metal)

`trellis-cli --views <views4_fixed> --models <set> --seed <n> --res 1024`, judged with
`tools/silhouette_iou.py --res=256 --min-iou=0.85 --max-scale-error=0.05`. Input: four views
(front/back x left/right) rebuilt from the 252-frame `yoimiya_blender_1024` capture with
`mesh_scale: 0.206` — **not** byte-identical to the `views4_fixed` directory §13 used, which no
longer exists, so these numbers are not comparable to §13's absolute values. The paired
Q8_0-vs-F16 comparison below is internally consistent because both sets ran on the same input and
the same seeds.

| seed | `pixal3d-q8_0 v1` mean IoU | `pixal3d-f16 v1` mean IoU | Q8_0 − F16 |
|---|---:|---:|---:|
| 1 | 0.8967 (scale_err 0.0020, V=699,807 F=940,746) | 0.8948 (0.0040, V=703,702 F=960,356) | +0.0019 |
| 2 | 0.8976 (0.0060, V=706,794 F=947,454) | 0.9040 (0.0040, V=730,694 F=966,044) | −0.0064 |
| seed spread within the set | 0.0009 | 0.0092 | |

All four runs report `SILHOUETTE_GATE: PASS`. **No quality difference between the two sets is
detectable**: the sign of the difference flips between seeds and its magnitude is smaller than the
F16 set's own seed-to-seed spread (0.0092). This supports shipping Q8_0 to the browser without a
shape-quality regression; it does not show Q8_0 is better. Runtime was 744–778 s per run
(SS ~126 s, shape ~150 s, shape decode ~10 s at ~1.73 M voxels, texture 92–236 s, postprocess the
remainder).

Still open: the official `cyclops` sample has not been run against these manifests, and no Chrome
/WebGPU run has been done at all — the rows below stay ⬜.

### Regenerate / verify

Never edit a manifest by hand.

```sh
python tools/quantize_gguf.py <f16>/<name>.gguf <q8>/<name>.gguf Q8_0   # for each of the nine
python tools/model_manifest.py generate <models_dir> --model-set <set> --version v1 \
  --source "<provenance>" --output models/<set>-v1/pixal3d-models.json
python tools/model_manifest.py verify <models_dir> models/<set>-v1/pixal3d-models.json
```

### Fail-closed re-check against the real files (2026-09-09, macOS/M4 Max)

| case | `pixal3d-f16 v1` | `pixal3d-q8_0 v1` |
|---|---|---|
| intact set | `MODEL_MANIFEST_OK pixal3d-f16 v1`, rc 0 | `MODEL_MANIFEST_OK pixal3d-q8_0 v1`, rc 0 |
| corrupt (file truncated) | `size mismatch pixal3d_naf.gguf: 1000 != 1334656`, rc 2 | `size mismatch ss_dec.gguf: 4096 != 147379392`, rc 2 |
| missing role | `missing: tex_dec.gguf`, rc 2 | `missing: dinov3.gguf`, rc 2 |
| mixed set | entry SHA256 from another set → `sha256 mismatch pixal3d_ss_flow_mv.gguf`, rc 2 | F16 `pixal3d_tex_flow_1024_mv.gguf` in a Q8_0 dir → `size mismatch … 2775670944 != 1476813984`, rc 2 |
| JSON Schema | `REAL_MANIFEST_SCHEMA_OK` | `REAL_MANIFEST_SCHEMA_OK` |
| Web consumer | `validateManifest()` accepts (9 files) | `validateManifest()` accepts (9 files) |

`tools/model_manifest.py self-test` / `schema-test` also pass (`MODEL_MANIFEST_SELF_TEST_OK`,
`MODEL_MANIFEST_SCHEMA_TEST_OK`), and `tools/model_manifest.py check-committed` gates the committed
manifests themselves in CI: shape + required roles, directory name matching `<model_set>-<version>`,
and the SHA256 above being present in this file. **Wiring it into
`.github/workflows/model-manifest.yml` is still pending** — the PR author's token lacks GitHub's
`workflow` scope, so the workflow file could not be pushed. The follow-up commit adds
`models/*/pixal3d-models.json` and `docs/PIXAL3D_RELEASE_CHECKLIST.md` to the workflow's `paths`,
runs `python tools/model_manifest.py check-committed`, and validates every
`models/*/pixal3d-models.json` against the schema. Note what CI cannot do: it has no access to the
real weights, so byte/SHA256 agreement with the actual GGUFs is only the manual re-check recorded
above. Desktop consumes the same file: `model_cache.rs` reads `version`
and `files[].name` from `pixal3d-models.json` in the managed model directory, so a release manifest
is copied next to the GGUFs unchanged. Installer download/verify (#18) and OPFS lifecycle (#9)
consume these manifests but are tracked in their own rows.

## Common gates

| Requirement | Status | Evidence / owner |
|---|:---:|---|
| Shared model manifest contract: exact filenames/roles/bytes/SHA256 | ✅ | #15 / PR #24 |
| Manifest rejects incomplete/corrupt/mixed production set | ✅ | model-manifest CI |
| Linux/GCC C++ build gate on `main` | ✅ | PR #33 |
| Explicit `mesh_scale`; no silent production fallback | ✅ | #19 / PR #27 |
| Release model directory gets a real generated manifest | ✅ | #34: `models/pixal3d-f16-v1/` + `models/pixal3d-q8_0-v1/` |
| Official cyclops + views4 result recorded for exact runtime/model set | 🔶 | views4-style input re-run against both manifests (E2E table above, 4 runs, all PASS); official `cyclops` sample still not run |

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
| Installer resolves the exact prerelease tag and fails closed on missing assets | ✅ | #36: tag resolution + asset preflight + receipt. Verified against the live GitHub API for tag/`latest`/`latest-prerelease` resolution, tag-injection rejection, missing-asset and unknown-tag exit 1, and compact-JSON parsing. `install.ps1` is unverified beyond static checks — no PowerShell on the reference machine |
| Installer downloads/verifies the exact model set against its manifest | ⬜ | still open: the installers fetch the inherited TRELLIS.2 weights from HF and only record `model_set`/`version` from an existing `pixal3d-models.json`; they do not download or SHA256-verify the Pixal3D set |
| Package/Tauri version matches `v0.9.0-desktop-alpha` | ✅ | #35: metadata `0.9.0` in all four files; tag carries `-desktop-alpha` (see above) |

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
| Second launch reuses the cached 7.54 GiB `pixal3d-q8_0 v1` set without retransferring it | ⬜ #9 | real Chrome |
| Cache deletion frees only origin-owned Pixal3D storage | ⬜ #9 | real Chrome |

The Web release model set is `pixal3d-q8_0 v1` (7.54 GiB), per §12's GPU-budget headroom finding. Its native E2E gate now passes against this exact manifest (see the E2E table above); no Chrome/WebGPU run has been done against it.

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
