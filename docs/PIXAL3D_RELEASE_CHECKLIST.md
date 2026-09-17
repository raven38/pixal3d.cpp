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

### Manifest-verified model set install

`--model-manifest PATH|URL|release` (`-ModelManifest`) installs the model set a manifest describes
and verifies **every file's exact size and SHA256** before the install is allowed to succeed;
`--model-base-url` (`-ModelBaseUrl`) says where the files live. `--verify-models`
(`-VerifyModels`) verifies an existing models directory against its manifest and exits without
downloading anything or writing any config — that is the gate a clean-install E2E (#18) can run
before and after installing.

Fail-closed properties, each verified (2026-09-10, macOS):

| case | result |
|---|---|
| real `pixal3d-q8_0 v1`, 9 files / 7.54 GiB, all present | `model set OK — pixal3d-q8_0 v1`, rc 0, 4.7 s |
| one file corrupted | `does not verify: tex_dec.gguf` + `(8 of 9 files match the manifest)`, rc 1 |
| one file missing | `does not verify: ss_dec.gguf`, rc 1 |
| manifest names `../escape.gguf` | `unsafe file name in model manifest`, rc 1 |
| `schema_version: 2` | `unsupported model manifest schema_version: 2`, rc 1 |
| served file tampered mid-download | prints expected vs. got size/SHA256, **deletes the bad file**, rc 1 |
| download into an empty dir | 9 files fetched, verified, manifest copied in last, rc 0 |

The manifest is copied into the models directory only after every file verifies, so its presence
means "complete, verified model set" — which is exactly how `app/src-tauri/src/model_cache.rs` and
`web/app/model_store.js` read it. The receipt records `manifest_sha256` and a `verified` flag that
is true only when this run actually hashed the files.

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

The F16 files themselves no longer live on the reference machine (they were moved to shared storage
after verifying all nine SHA256 digests); the manifest is what makes them recoverable and
identifiable. The Q8_0 set stays local because it is the Web release identity.

**This deviates from #34's original acceptance condition** ("Desktop/Web consume the same
manifest"). Both frontends consume the same *contract* — one schema, one filename, one generator,
the same nine roles — but not the same file, because a single set cannot satisfy both requirements:
the browser cannot afford F16 weights and native must not silently ship quantized weights while
parity work is open. Narrowing to one shared set means either shipping Q8_0 on Desktop (losing
reference parity) or F16 on Web (3.4 % budget headroom). Recorded here rather than waived silently.

The split follows `docs/PIXAL3D_WEBGPU_MEMORY.md` §12: native stays F16 for reference parity, while
the browser needs Q8_0 because the F16 texture flow leaves only 3.4 % of the 4095 MB WebGPU budget
at N=17690 tokens (Q8_0: 34 %). Both manifests describe the same nine required roles.

### Single-view Web model set `pixal3d-sv-q8_0 v1` (2026-09-17, #9)

The browser `Single image (SV)` mode (PR #8) refuses to run multiview flow weights at V=1, so it
needs its own verified set. `pixal3d-sv-q8_0 v1` is the single-view counterpart of the Web
release identity: the four flow DiTs are the official Pixal3D single-view checkpoints (the
`ckpts/*_bf16.safetensors` **without** the `_mv` suffix) converted with `tools/convert.py` and
quantized with `tools/quantize_gguf.py … Q8_0`; the five shared models (`dinov3.gguf`,
`pixal3d_naf.gguf`, `ss_dec.gguf`, `shape_dec.gguf`, `tex_dec.gguf`) are byte-identical to
`pixal3d-q8_0 v1` (same SHA256 as the table below).

| | Web single-view identity |
|---|---|
| `model_set` / `version` / `model_family` | `pixal3d-sv-q8_0` / `v1` / `sv` |
| Manifest | `models/pixal3d-sv-q8_0-v1/pixal3d-models.json` |
| Manifest SHA256 | `e127c0f70e6dc43c07b4c1afc896a7e14353e5c2e7aa6ce0fd82406a6a94c61e` |
| Total model bytes | 8,091,975,360 (7.54 GiB) — the SV Q8_0 flows have exactly the MV Q8_0 sizes |
| Hosted at | `https://huggingface.co/raven38/pixal3d-sv-q8_0-v1/resolve/main` (`sv_manifest_url` = `…/pixal3d-models.json`) |

`model_family` is a new optional manifest key (schema + `tools/model_manifest.py`): absent means
`mv`, so the two existing manifests are unchanged; `sv` switches the required flow file names to
`*_sv.gguf`, and a manifest whose family and file names disagree is rejected by `validate_manifest_shape`
(self-test) and by `web/app/model_family.js` `modelFamilyForManifest` (re-exported from `single_view.js`; the browser enforces the same name↔role contract, see `web/app/README.md`).

Verification chain for the published bytes: SHA256 of the four `*_sv.gguf` recomputed from the
files on shared storage before upload (4/4 match), Hugging Face LFS object ids after upload equal
to those digests (9/9, the five shared files were copied server-side from `pixal3d-q8_0-v1`), and
the manifest generated by `tools/model_manifest.py generate --model-family sv` from a local set
whose four SV files were downloaded back from the published repository (`verify` rc 0).
This set has **not** yet been through the real Chrome/WebGPU SV release gate — that is #11.

**Native V=1 acceptance run (#9 validation steps 2–4, 2026-09-17).** The exact published set was
run natively on Apple M4 Max (Metal; `trellis-cli` built from this branch at `90f8c95`, whose last `src/` change is `7fe6e10`, ggml submodule `737e88f2`):

| step | result |
|---|---|
| Served manifest fetched from Hugging Face vs. committed file | byte-identical, SHA256 `e127c0f7…94c61e` |
| Local 9-file set (`~/data/weights/pixal3d/gguf-sv-q8_0/`, the five shared files symlinked to `gguf-q8_0/`) verified against the *served* manifest | `MODEL_MANIFEST_OK pixal3d-sv-q8_0 v1`; the directory contains no `*_mv.gguf` |
| Input | official `assets/mv_images/Belle_front.png` (2146², real alpha), one image, PR #5 crop (`preprocess_prematted_rgba` → 1124² RGBA) + `estimate_transforms_moge.py --manual-fov 0.3490658503988659` (20°, `mesh_scale=1.0`, distance 2.8356 — the same one-frame camera `web/app/single_view.js` synthesizes) |
| `trellis-cli --views … --models <sv set> --pixal3d-weights sv --seed 42 --res 1024` | **exit 0**, 1,525 s wall; log: `V=1 views, mesh_scale=1.0000`, `flow weights: sv`, no view-count warning; decode V=1,375,499 F=2,941,810 → GLB V=729,031 F=956,332, 37.0 MB |
| Flow files actually opened (`lsof -p` sampled every 1 s while running — a lower bound, small files can load between samples) | `pixal3d_ss_flow_sv.gguf`, `pixal3d_shape_flow_512_sv.gguf`, `pixal3d_shape_flow_1024_sv.gguf`, `pixal3d_tex_flow_1024_sv.gguf`, plus shared `ss_dec.gguf` / `shape_dec.gguf` resolved through the symlinks; **no `_mv.gguf`** (structurally impossible too: `mv_weight_paths` builds `<models>/pixal3d_*_sv.gguf` and no `_mv` file exists under `--models`) |
| Visual check (six orthographic views front/back/left/right/top/bottom, normal-shaded) | consistent A-pose figure matching the input in every view, bbox centred at the origin, extents 0.671 × 0.894 × 0.272, `uv_bake: 1 components`; **no spike-like protrusions or detached geometry** |
| Control: same input, same seed, `--pixal3d-weights mv` (the known MV@V=1 failure mode; the CLI prints its `85-0` warning) | exit 0, 1,417 s; a **detached debris blob in front of the body** visible in the left/right/top/bottom views, `uv_bake: 2 components`, extents 0.626 × 0.842 × **0.818** (depth 3× the SV result). The same renderer therefore does show the failure mode when it is present |

Artifacts (both GLBs — SV `d7c9963a…2ba90`, MV control `5bfa675d…` — PLYs, six-view renders and
the SV-vs-MV side-by-side, staged `input.png`/`transforms.json`, CLI logs, `lsof` lists) are kept
outside the repo at `~/data/pixal3d-sv-acceptance/2026-09-17-pr13/`. This run shows completion,
weight selection and the absence of the MV@V=1 failure mode on one subject; it is not a parity
claim against the PyTorch reference, and one subject is not a quality statistic.

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

### Official `cyclops` sample (2026-09-10, same machine)

`assets/mv_images/example` from `TencentARC/Pixal3D` (4 RGBA views at 1024², `mesh_scale: 1.0`
declared), seed 1, res 1024:

| | `pixal3d-q8_0 v1` | `pixal3d-f16 v1` |
|---|---:|---:|
| mean silhouette IoU | 0.9801 | 0.9805 |
| scale_error | 0.0031 | 0.0032 |
| final GLB | V=639,713 F=950,170 | V=629,701 F=955,022 |
| runtime | 1443.7 s | 1559.6 s |
| gate | PASS | PASS |

Both match the 0.9785 recorded for this sample in `docs/PIXAL3D_E2E_STATUS.md`, and the two model
sets differ by 0.0004 — again no detectable quality difference, on a second subject class (a large,
bulky subject vs. the thin-limbed figure of the views4-style input above), which is what makes the
Q8_0-for-the-browser decision safe to state.

The same exact `pixal3d-q8_0 v1` manifest was then exercised in real Chrome/WebGPU by the release
gate below: full 1024 generation passed at mean IoU 0.9795, and a browser restart reused the OPFS
cache with zero GGUF retransfers.

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
and the SHA256 above being present in this file. `.github/workflows/model-manifest.yml` is wired on
`main` (PR #43) to run `check-committed` and validate committed `models/*/pixal3d-models.json`
against the schema. CI does not have the real weights, so byte/SHA256 agreement with the actual
GGUFs remains the manual re-check recorded above. Desktop consumes the same file: `model_cache.rs`
reads `version` and `files[].name` from `pixal3d-models.json` in the managed model directory, so a
release manifest is copied next to the GGUFs unchanged. Installer download/verify (#18) and OPFS
lifecycle (#9) consume these manifests but are tracked in their own rows.

## Common gates

| Requirement | Status | Evidence / owner |
|---|:---:|---|
| Shared model manifest contract: exact filenames/roles/bytes/SHA256 | ✅ | #15 / PR #24 |
| Manifest rejects incomplete/corrupt/mixed production set | ✅ | model-manifest CI |
| Linux/GCC C++ build gate on `main` | ✅ | PR #33 |
| Explicit `mesh_scale`; no silent production fallback | ✅ | #19 / PR #27 |
| Release model directory gets a real generated manifest | ✅ | #34: `models/pixal3d-f16-v1/` + `models/pixal3d-q8_0-v1/` |
| Official cyclops + views4 result recorded for exact runtime/model set | ✅ | both manifests, both inputs: views4-style at seeds 1/2 and the official cyclops sample — 6 runs, all PASS (tables above) |

## Desktop alpha

Supported target: Trellis Studio, Windows x64 + Linux x86-64, resident native `trellis-server`, Pixal3D MV 1024/1536. CUDA/Vulkan may be advertised only on surfaces actually exercised by the release candidate. Native Dawn/WebGPU is outside the Desktop alpha support matrix while #2 remains open.

| Requirement | Status | Test class |
|---|:---:|---|
| Managed cache cannot delete external user files | ✅ #16 / PR #25 | CI: temp dir / traversal / symlink |
| Real Tauri/WebKitGTK window + Studio-spawned server lifecycle | ✅ #17 / PR #26 | Xvfb CI |
| MV metadata/image/alpha/mesh_scale preflight | ✅ #19 / PR #27 | headless UI CI |
| CUDA `ss_decode` graph-support path exercised on NVIDIA hardware | ✅ #11 | NVIDIA L4 validation |
| Clean-install Windows: installer → models → MV → textured GLB → viewer/save | ⬜ #18 | **real machine** |
| Clean-install Linux: installer → models → MV → textured GLB → viewer/save | ✅ headless (see below) / ⬜ viewer | **real machine** |
| Installer resolves the exact prerelease tag and fails closed on missing assets | ✅ | #36: tag resolution + asset preflight + receipt. Verified against the live GitHub API for tag/`latest`/`latest-prerelease` resolution, tag-injection rejection, missing-asset and unknown-tag exit 1, and compact-JSON parsing. `install.ps1` verify-only path is now executed in Windows CI by PR #44; full clean-install remains #18 |
| Installer downloads/verifies the exact model set against its manifest | ✅ | `--model-manifest` / `--verify-models` (see below); verified against the real `pixal3d-q8_0 v1` set and fail-closed cases, plus Linux/Windows tiny-set CI in PR #44 |
| Package/Tauri version matches `v0.9.0-desktop-alpha` | ✅ | #35: metadata `0.9.0` in all four files; tag carries `-desktop-alpha` (see above) |

Desktop alpha deliberately requires an explicit positive `mesh_scale`. Automatic estimation from PR #5 is not part of the release path because the real calibration datasets produced large errors (cyclops expected ~1.0 → 1.261568; views4 expected ~0.206 → 0.381288).

### Clean-install Linux, headless — Vulkan and CUDA (2026-09-12, NVIDIA L4 / Ubuntu 24.04 container)

`install.sh --skip-app --backend vulkan --tag v0.9.0-desktop-alpha` + `trellis-cli --views` in a bare
`ubuntu:24.04`-based image (no apt at run time), models from Hugging Face:

| step | result |
|---|:---:|
| installer | rc 0; Vulkan runtime tarball from the release, 9 GGUF from HF, `pixal3d-q8_0 v1` verified |
| runtime starts on a minimal image | **failed on the first attempt** — `libgomp.so.1: cannot open shared object file` (ggml-cpu links OpenMP, `libgomp1` absent). Fixed by PR #50 (bundle it + fail packaging when `ldd` reports anything unresolved); the rebuilt archive starts |
| real GPU | `ggml_vulkan: 0 = NVIDIA L4 (NVIDIA) \| fp16: 1 \| matrix cores: NV_coopmat2`, `using Vulkan0 (23034 MB)` |
| MV 1024 generation | rc 0, 1188.8 s total, 34,254,872-byte GLB (V=670,138 F=991,056, atlas 4096), sha256 verified after transfer |
| stage times | SS 31.2 s / Shape-512 25.9 s / Shape-1024 251.9 s / Texture 150.7 s — **the fastest of the three paths measured** (macOS Metal native 89.1/93.3/581.5/345.2; L4 browser WebGPU 169.6/179.1/1279.0/768.2) |
| known gap | the NAF@1024 encoder falls back to CPU on Vulkan (4-D `MUL_MAT`/`PERMUTE`/`CONT`/`RESHAPE` unsupported, once per view) — correct output, but this stage is not GPU-resident on Vulkan |

Re-run on the portable-CPU archives (PR #54, `GGML_NATIVE=OFF`): same times within 0.15 %
(31.1 / 25.8 / 252.1 / 151.3 s, total 1187.0 s) and the same GLB size, so the portability fix costs
no measurable speed. The GLB bytes are not identical between the two builds (sha256
`49c71902…` vs `2281f30c…`) — CPU kernel selection changes the float rounding; bit-identical output
across builds is not claimed anywhere in this project.

**CUDA 13.1 leg, same pod recipe** (`--backend cuda`, `trellis-cuda-linux-x64.tar.gz`, portable build):
install rc 0, `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 22563 MiB)`, generation rc 0 in
**626.7 s**, GLB 35,617,428 B (V=677,419 F=996,282, atlas 4096), sha256 verified after transfer.
Stage times **14.4 / 15.5 / 113.1 / 67.4 s** — 2.2x the Vulkan leg on the same GPU, and the NAF CPU
fallback of the Vulkan path does not occur.

| path (same input, seed 1) | SS | Shape-512 | Shape-1024 | Texture | total |
|---|---:|---:|---:|---:|---:|
| L4 CUDA 13.1 native | **14.4 s** | **15.5 s** | **113.1 s** | **67.4 s** | **626.7 s** |
| L4 Vulkan native | 31.1 s | 25.8 s | 252.1 s | 151.3 s | 1187.0 s |
| macOS M4 Max Metal native | 89.1 s | 93.3 s | 581.5 s | 345.2 s | ~1440 s |
| L4 browser WebGPU | 169.6 s | 179.1 s | 1279.0 s | 768.2 s | 3209 s |

Not covered: the Studio GUI on Linux (WebKitGTK window, gallery/save) — only the runtime and CLI.
Logs/renders: `docs/results/linux-webgpu-gate/l4-clean-vulkan-20260912/` and
`.../l4-clean-cuda-20260912/`.

## macOS Desktop alpha (2026-09-10)

Supported target: Trellis Studio on **macOS 26.5 / Apple Silicon**, Metal runtime, resident native
`trellis-server`, Pixal3D MV 1024. This is the surface that is actually verified end to end on real
hardware; Windows and Linux clean-install (#18) remain open, so they must not be advertised until a
real machine runs them.

| requirement | status | evidence |
|---|:---:|---|
| Studio builds on macOS | ✅ | `npx tauri build --bundles app` → `Trellis Studio.app` 3.7 MB, `CFBundleShortVersionString` 0.9.0 (matches #35), id `cpp.trellis.studio`, ad-hoc signature |
| Real window on macOS | ✅ | one window titled "Trellis Studio", 1200×820, process alive |
| Studio spawns + supervises the server | ✅ | reads `~/Library/Application Support/trellis-studio/config.json`, launches `trellis-server` as its child (parent pid confirmed), `GET /health` → `ok` |
| **MV 1024 generation through the desktop server** | ✅ | `POST /generate-mv` with the official cyclops sample, seed 1: HTTP 200, 32,899,856-byte GLB in 1987 s; **mean IoU 0.9801, scale_error 0.0031 → PASS**, V=639,713 F=950,170 — identical to the native CLI run |
| Installer path | ✅ | `install/install.sh` on Darwin: Metal-only backend, config to `~/Library/Application Support`, dmg mounted and copied to `/Applications`, quarantine attribute cleared; ran to completion (rc 0) against real assets |
| Release artifacts | ✅ | `.github/workflows/release.yml` includes `macos-14` in both runtime and Studio matrices, producing `trellis-metal-macos-arm64.tar.gz` and `trellis-studio-macos-arm64.dmg`; packaging rejects a missing `@loader_path` or a residual build-machine rpath |
| Runtime tarball is self-contained | ✅ | the binaries resolve `@rpath/libggml*.dylib` through `@loader_path`; verified by deleting the build-directory rpath and still reaching `/health` → `ok`. Packaging strips that absolute build path and fails if `@loader_path` is missing |
| Code signing / notarization | ⬜ | ad-hoc signature only, `TeamIdentifier=not set`. The installer clears `com.apple.quarantine`, so a scripted install works; a user double-clicking a downloaded dmg still meets Gatekeeper. Advertise as unsigned or notarize before a public release |
| Intel macs | ⬜ | rejected explicitly by the installer (`uname -m != arm64`); only Apple Silicon is built and tested |

### Clean install from the published CI artifacts (2026-09-12, macOS/M4 Max)

The prerelease `v0.9.0-desktop-alpha` was created on `main` `c7848d1` and `release.yml` ran on the
`release: published` event (run 34617440907). With `~/Library/Application Support/trellis-studio`
absent and no `Trellis Studio.app` installed:

| step | result |
|---|:---:|
| `install.sh --repo … --tag v0.9.0-desktop-alpha -y --model-manifest <HF manifest> --model-base-url <HF>` (the installer as of the tag) | rc 0, 8 min; runtime tarball + dmg from the release API, 9 GGUF from Hugging Face, manifest `5648ee78…` verified, app copied to `/Applications` |
| runtime self-contained | `LC_RPATH` is `$ORIGIN` / `@loader_path` only |
| signature | `adhoc`, `TeamIdentifier=not set` (as the release note says) |
| Studio launch | `Trellis Studio.app` spawns `runtime/trellis-server` (child pid confirmed), `GET /health` → `ok` on 127.0.0.1:8080 |
| MV 1024 generation through that server | `POST /generate-mv`, views4, seed 1: HTTP 200, 32,958,796-byte GLB in 28 min; GLB `generator` = `trellis.cpp v0.9.0-desktop-alpha` |

Same head/nose ring/teeth/ears as the native CLI and browser runs on the same input.

The tag stays `v0.9.0-desktop-alpha` with artifact metadata `0.9.0` (see the versioning note above);
the release note must say macOS/Apple Silicon only.

## Web alpha

Supported target: Chrome/Chromium + WebGPU, resolution 1024 only. Inference stays in the common C++/WASM pipeline; JS is UI/file orchestration.

| Requirement | Status | Test class |
|---|:---:|---|
| Production app shell + GLB viewer/download | ✅ #20 / PR #28 | Chromium |
| Verified OPFS model-set install consumes manifest + size + SHA256 | ✅ #15 / PR #28 | Playwright fixture |
| Fast production UI + OPFS + manifest integrity gate | ✅ #22 / PR #29 | Playwright CI |
| Persistent automatic model delivery/version invalidation/safe cache delete | ✅ #9 | PR #42 implementation + `test_cache_failure_modes.mjs` + the real-Chrome gate |
| Storage quota + WebGPU/device-budget preflight UI | ✅ #21 | PR #42 `preflight.js` + device probe (2026-09-11): the preflight now requests a throwaway device with the runtime's own feature set (`shader-f16`, plus `subgroups` when advertised) and the adapter's limits; `requestDevice()` failure and missing `shader-f16` are rejected before the model install (`test_cache_failure_modes.mjs` cases 7–8). Real M4 Max: `device ok (shader-f16, subgroups)` |
| Full Chrome/WebGPU 1024 known-input generation → textured GLB | ✅ | official cyclops, Chrome 153, IoU 0.9795 PASS |
| Second launch reuses the cached 7.54 GiB `pixal3d-q8_0 v1` set without retransferring it | ✅ | browser restart on the same profile: 0 GGUF re-transfers |
| Cache deletion frees only origin-owned Pixal3D storage | ✅ | real Chrome: usage → 0; headless: an unrelated OPFS directory survives |
| Public single-view (SV) Q8_0 model set + manifest published | ✅ #9 | `pixal3d-sv-q8_0 v1` on Hugging Face (see "Single-view Web model set" above); real-Chrome SV gate is #11, production deploy is #10 |
| Production deployment reproducible from a clean checkout + merged SV UI deployed | ✅ #10 | `web-wasm-runtime` CI artifact (emsdk 6.0.9), `docs/runbooks/web-deploy.md`, version `3933e60f…` (2026-09-17), `smoke_production.mjs` 13/13 |
| Real Chrome/WebGPU single-image SV release gate | ✅ #11 | `run_release_gate.mjs --mode sv` on the deployed dist: IoU 0.9754, 1 component, no spike failure, 0 re-transfers (table below) |
| MV release gate re-run on the rebuilt runtime | ✅ | production URL, IoU 0.9795 / GLB 31,020,608 B — identical to 2026-09-11 (table below) |

The Web release model set is `pixal3d-q8_0 v1` (7.54 GiB), per §12's GPU-budget headroom finding. Its native E2E gate and the real Chrome/WebGPU release gate both pass against this exact manifest.

Web alpha known limits: Chrome/Chromium only, WebGPU required, 1024 only, wasm32 4 GiB host
address-space constraints, browser-safe 512 postprocess, and pre-matted RGBA input.
`transforms.json` is optional for exactly four turntable views (front/right/back/left,
elevation 0, FOV 20°); any other camera setup or view count still requires it, and the
synthesized case requires the user to confirm `mesh_scale` explicitly — no default is assumed.

**Linux + NVIDIA needs a Chrome flag.** Dawn does not expose `shader-f16` on NVIDIA/Vulkan
without `--enable-dawn-features=vulkan_enable_f16_on_nvidia` (crbug.com/42251215, Chromium 153 /
Dawn main 2026-09). With the flag the full gate passes on an L4 (see M12); without it the #21
device preflight stops before the 7.54 GiB install instead of failing at generation, which is
the intended behaviour but means stock Chrome on Linux/NVIDIA cannot run the Web alpha.

### Web alpha release gate, end to end (2026-09-10)

`web/app/run_release_gate.mjs` drives real Chrome with real WebGPU and the real `pixal3d-q8_0 v1`
set: fetch 7.54 GiB over HTTP into OPFS with streaming SHA-256, generate the official cyclops sample
at 1024, download the GLB, **restart the browser on the same profile**, and delete the cache — then
write one JSON record (browser, adapter, commit, model_set, manifest SHA256).

| step | measured |
|---|---|
| browser | Chrome for Testing 153.0.8010.12, headed, `--enable-unsafe-webgpu` |
| adapter | `maxBufferSize` 4,294,967,292 · `maxStorageBufferBindingSize` 4,294,967,292 |
| fetch + verify + store 7.54 GiB | 154–229 s across runs, **9 GGUF requests (one per file)** |
| generation | `complete` in **3688 s (61 min)**, seed 1, res 1024 |
| GLB | 31,020,608 bytes via the browser download |
| acceptance | `tools/silhouette_iou.py`: **mean IoU 0.9795, scale_error 0.0066 → PASS** (V=306,258 F=494,436 — identical to the file-input run) |
| **restart, same profile** | `Ready` immediately, **0 GGUF re-transfers** (only manifest reads) |
| generation after restart | started from the cached set with 0 transfers, then cancelled (a second 61-minute run adds nothing) |
| cache delete | usage 3,797,013,283 → **0**, state back to "Not ready" |

Texture Flow peaked at `resident 2517 MB` against the 4095 MB budget (38 % headroom) at
`tokens=17795` — the token scale at which §12 of `docs/PIXAL3D_WEBGPU_MEMORY.md` measured F16 at
3.4 % headroom. This run is therefore direct evidence for shipping Q8_0 to the browser, not a
restatement of the earlier estimate.

**A measured caveat about `navigator.storage.estimate()`.** It reported `usage` 3,797,013,283 for a
set whose files total 8,091,975,360 bytes while the app was reading that set successfully, and
`quota` tracked `usage + 10 GiB` in both runs (14.53 GB here, 18.83 GB when usage read
8,091,980,579). Chromium's OPFS accounting under-reported by more than 2× on one run, and delete then
freed exactly the number it had reported. So the estimate is self-consistent but is not a measure of
what is stored: treat `storagePreflight` as advisory and rely on `QuotaExceededError` at write time.

**`WebGPU (4095 MB)` is not a memory budget.** The runtime line comes from
`thirdparty/ggml/src/ggml-webgpu/ggml-webgpu.cpp:3772`, which returns `maxBufferSize` as both free
and total memory with an explicit `TODO` (gpuweb/gpuweb#5505). `web/app/preflight.js` is right not to
invent an aggregate-VRAM threshold; the per-buffer and per-binding limits are the checkable part.

### Web alpha release gate against the public deployment (2026-09-11)

Same gate (`web/app/run_release_gate.mjs --models-url …`), but the app is served from the public
Cloudflare Pages deployment and the model set from the public Hugging Face repository — no local
HTTP server anywhere in the path.

| | value |
|---|---|
| app | `https://pixal3d-web.raven38.workers.dev/` (Cloudflare Pages, `wrangler.jsonc`, 12 static files / 4.6 MB from `scripts/build_web_dist.sh`) |
| models | `https://huggingface.co/raven38/pixal3d-q8_0-v1/resolve/main` (public; 9/9 files match the committed manifest by size + SHA256) |
| browser | Chrome for Testing 153.0.8010.12, headed, `--enable-unsafe-webgpu`, M4 Max / Metal |
| adapter | `maxBufferSize` 4,294,967,292 · `maxStorageBufferBindingSize` 4,294,967,292 |
| fetch throughput | **6.9 MiB/s** measured mid-download (HF → this machine), i.e. ~18–20 min for a cold 7.54 GiB install |
| resume after a hard kill | the first run was killed at 7.5 GiB; the rerun with the same profile reached `Ready` in 1210 s with **`gguf_requests: 1`** — eight of nine files were re-hashed from OPFS and reused |
| generation | `complete` in **4902 s**, seed 1, res 1024 (3688 s on the local-HTTP run; same machine, other load present) |
| Texture Flow | `tokens=17795`, `resident 2517.0 MB`, graph peak 1538.0 MB — identical to the local-HTTP run |
| shape decode | 4,798,704 voxels → raw mesh V=4,798,704 F=9,607,898; the browser tail completed |
| GLB | **31,020,608 bytes** — byte-for-byte the same size as the local-HTTP run |
| acceptance | `tools/silhouette_iou.py`: **mean IoU 0.9795, scale_error 0.0066 → PASS** — identical to the local-HTTP run |
| restart, same profile | `Ready` immediately, **0 GGUF re-transfers**, generation started from cache |
| cache delete | usage 3,797,012,676 → **0** |

One real bug was found and fixed on the way. Hugging Face answers a request that carries a
`Referer` header with **404 and no CORS headers** (hotlink protection), so the browser reported
`Failed to fetch` while every `curl` probe succeeded. `web/app/release_store.js` now fetches with
`referrerPolicy: 'no-referrer'`. This is invisible to any test that does not run a real browser
against a real cross-origin host.

### Production deployment with the SV UI and a reproducible runtime (2026-09-17, #10)

Deployed the merged single-image SV mode (PR #8) to the existing Worker, with the WebGPU/WASM
runtime rebuilt from a clean checkout for the first time (design:
`docs/design/2026-09-17-web-deploy-reproducible.md`, runbook: `docs/runbooks/web-deploy.md`).

| | value |
|---|---|
| Worker / URL | `pixal3d-web` · `https://pixal3d-web.raven38.workers.dev/` (unchanged) |
| Cloudflare version | `3933e60f-7bd4-420b-8cdf-cf06f426a941` (100 %, 2026-09-17T09:47:48Z); previous active version `65d1d151-889d-4e6c-aaf1-e520ec0fa9bc` (rollback target) |
| App files | identical to `origin/main` `8592a56` (PR #8); deploy tooling from PR #14 |
| Runtime | CI artifact of `web-wasm-runtime` run 35189943680 (emsdk **6.0.9**, merge ref `baa6796`): `pixal3d_real_geometry.js` `e5c462c61a3fd27f9e895a6137b0d97884e223665309dd546f9d44f13a868394`, `.wasm` `e43a4aa5238cdb8eb7be0f499b90f5c40f9e0bd270b232f02cf4c1697ab3a208` (3,650,886 B). Two independent CI runs produced identical digests; the macOS Homebrew emcc 6.0.9 build differs in the wasm, so production uses the CI artifact only |
| Previous runtime | `46b2016392aa7b34e49c5d9ea8ae7a6bbc0a4ff80a8fb381223bb6cdefd1b073` (3,616,383 B, built 2026-09-09; the runtime behind every gate row above) |
| Model sources (pinned) | MV `https://huggingface.co/raven38/pixal3d-q8_0-v1/resolve/1f82a6b7e0b64de4fe7c96c66db7977fc61a1203` · SV `https://huggingface.co/raven38/pixal3d-sv-q8_0-v1/resolve/c5dfd4c2352e39a392c5ff2f90bdcbb96d3c7403` (+ `/pixal3d-models.json`, SHA256 `e127c0f7…c61e`) |
| Served bytes | 14/14 files equal the pre-deploy receipt (`/index.html` is a 307 to `/`; compare via `/`) |

Deploy receipt (`web/dist`, the bytes the SV gate ran against and that were uploaded):

| file | SHA256 |
|---|---|
| `index.html` | `d4c16d61e11145983bd14bb10efa6ea8e01167ecfa72613a80cde5d8f6498091` |
| `main.js` | `4802239b871bb632a7ab850dd08f5c20958d9be769bbade25514bdac8b53cc2c` |
| `model_store.js` | `391ec063c0b5c673a3de09f3700a99b84b670cb83bc76f4d9f3796c480c5ecf6` |
| `models/pixal3d-q8_0-v1/pixal3d-models.json` | `5648ee78259808ff2b664574a666cd880c2c2b584ab8cd225bf80480e17d669f` |
| `preflight.js` | `a41f96c1b86f08509f8140c67d58b16c97e947910a55c3a30ba497f262b1be45` |
| `real_e2e/build-info.json` | `cffef2929aae6973b98871b951d08ff884da8fcf471cf8b25a9ab9459be8d5f1` |
| `real_e2e/calibration.js` | `e48752ea6959d7b28d2db980b329ab4fb2ba408983060dfe8a22d5d86a5d18d6` |
| `real_e2e/pixal3d_real_geometry.js` | `e5c462c61a3fd27f9e895a6137b0d97884e223665309dd546f9d44f13a868394` |
| `real_e2e/pixal3d_real_geometry.wasm` | `e43a4aa5238cdb8eb7be0f499b90f5c40f9e0bd270b232f02cf4c1697ab3a208` |
| `real_e2e/worker.js` | `d042ef6b862909c7f20bd4a57a06dde3e020ef42a065f36585a9258b2e669c8f` |
| `release_store.js` | `b36fc1817b3f08eae1cbfc83d2d01ddd324c57af913cb0b4fc5bfff6de5e37cf` |
| `sha256.js` | `805e98ffe8e0f0974192fbda2c6282dc659d3cc4f9ffd890cf37d9d832fcab37` |
| `single_view.js` | `e22fdcb5f53067ad7121270edeff163fb4c15fcd072138b82b9f83df06a11b73` |
| `vendor/model-viewer.min.js` | `283b0672384614b4847636c306fc93fe4b1fcadc76d668b4e47f0ca76bcf033b` |
| Smoke | `web/app/smoke_production.mjs`: `WEB_PRODUCTION_SMOKE_OK 13 checks` (both URLs injected, `build-info.json` served and equal to the wasm, input-mode visible, default `sv` because SV is configured, SV/MV manifests resolve to their family, Download enabled in both) |

#### Single-image SV release gate on the deploy candidate (2026-09-17, #11, M4 Max / Metal)

`web/app/run_release_gate.mjs --mode sv` against the exact `web/dist` that was deployed (served
locally, byte-identical receipt), models from the pinned SV repository, input = the official cyclops
front view (`view00_azim000.png`, 1024², RGBA), Chrome for Testing 153.0.8010.12 headed,
`--enable-unsafe-webgpu`. Report `ok=true`.

| | value |
|---|---|
| adapter | `maxBufferSize` 4,294,967,292 · `maxStorageBufferBindingSize` 4,294,967,292 · `device ok (shader-f16, subgroups)` |
| install | 826 s, 9 GGUF requests, `Ready · pixal3d-sv-q8_0 v1 · SV · 7.54 GB · 9 verified model(s)` |
| browser preprocessing | 1024² → object-centric crop 926² (1.1 margin), 874,820 B; `Canonical front camera · FOV 20.0° · mesh_scale 1.0 · distance 2.8356`, `V=1` |
| SS | active voxels **3,786** (64³), `nonfinite=0` |
| Shape-512 → 1024 | upsample 1,010,474 coords → **14,871** tokens (MV cyclops: 17,795) |
| raw mesh | V 3,953,680 / F 7,937,104, bbox (0.7327, 0.8160, 0.9171) |
| Texture Flow | tokens 14,871, graph peak 1538.0 MB, resident 2494.1 MB |
| final GLB | **29,333,892 B**, V 235,404 / F 471,892, one geometric component after position weld, bbox extents (0.7362, 0.8195, 0.9207) |
| generation | `complete` in 6119 s (SS 252 s · Shape-512 328 s · Shape-1024 2396 s · Texture 1617 s; the machine had other load) |
| front silhouette IoU | **0.9754** (`tools/silhouette_iou.py` with the replicated crop + canonical camera), projected-size ratio 0.974 |
| renders | front/back/left/right/top: symmetric ears, single eye, nose ring, closed back — **no spike-like MV@V=1 failure** |
| restart, same profile | `Ready`, **0 GGUF re-transfers** (manifest only), generation started from cache |
| cache delete | usage 3,797,012,618 → **0** |

The SV gate above ran against the deploy candidate served locally (byte-identical to production per the
receipt). The SV model path through the **production origin** was then exercised without generation:
`web/app/smoke_sv_install.mjs https://pixal3d-web.raven38.workers.dev/` → install 260 s, 9 GGUF requests
(HF 302 → CDN), `Ready · pixal3d-sv-q8_0 v1 · SV · 7.54 GB · 9 verified model(s)`; restart 0 re-transfers;
delete 3,797,012,618 → 0 (`ok=true`).

The first attempt of this gate ended with `Target page, context or browser has been closed` during
Shape-1024 (the headed Chrome window was closed externally); the rerun on the same profile skipped
the install and reproduced SS 3,786 / 14,871 tokens exactly.

#### MV release gate on the deployed runtime (2026-09-17)

Same gate as 2026-09-11 (`run_release_gate.mjs`, now opened with an explicit `?mode=mv` because a
configured SV source makes SV the default mode), against the **production URL** after the deploy,
models from the pinned MV repository, official cyclops 4 views, seed 1, same machine/browser.
Report `ok=true`.

| | 2026-09-11 (runtime `46b20163…`) | 2026-09-17 (runtime `e43a4aa5…`) |
|---|---|---|
| install | 1210 s (resume) | 582 s, 9 GGUF requests, `Ready · pixal3d-q8_0 v1 · MV · 7.54 GB` |
| SS active voxels | — | 4,438 |
| Shape-1024 tokens / raw mesh | 17,795 / V 4,798,704 | **17,795 / V 4,798,704** (identical) |
| Texture Flow resident | 2517.0 MB | **2517.0 MB** |
| generation | 4902 s | 7228 s (SS 264 · Shape-512 362 · Shape-1024 1814 · Texture 1136; other load on the machine) |
| GLB | 31,020,608 B | **31,020,608 B** (identical size) |
| mean silhouette IoU / scale ratio | 0.9795 / 0.993 | **0.9795 / 0.993** (per view 0.9791 · 0.9779 · 0.9820 · 0.9789) |
| restart / delete | 0 re-transfers / usage → 0 | 0 re-transfers / 3,797,012,676 → **0** |

No regression of the MV path on the rebuilt runtime; both families now have a real-Chrome PASS on
the same wasm SHA256 (`e43a4aa5238cdb8eb7be0f499b90f5c40f9e0bd270b232f02cf4c1697ab3a208`), which
is the acceptance condition of `docs/design/2026-09-17-web-deploy-reproducible.md`.

### Failure modes of the browser cache (`web/app/test_cache_failure_modes.mjs`)

`test_headless.mjs` covers the happy path (remote lifecycle → OPFS → SHA → commit marker, delete,
local fallback). This gate covers the other side, and the current implementation passes all of it:

| case | result |
|---|---|
| same-size, different content served | `Downloaded model failed verification: … (size 27/27, sha256 …)`, cache stays not ready |
| refresh interrupted (HTTP 500 mid-set) | fails loudly; **no mixed set reads as ready** |
| role reassignment with identical hashes | not treated as the same set — all 9 files re-fetched |
| a required file removed from OPFS | not ready |
| delete | removes only the Pixal3D namespace; an unrelated OPFS directory survives |
| **reload with a warm cache** | **0 GGUF re-transfers** |

One real gap the gate found, now closed (#21, 2026-09-11): a `requestDevice()` failure used to pass
the preflight, which only probed the adapter and its limits. `webgpuPreflight()` now requests a
throwaway device the way `ggml_webgpu_init` does as far as the pipeline depends on it: the same
adapter selection (default `requestAdapter()`, no `powerPreference`), `shader-f16` required,
`subgroups` when the adapter advertises it, and `maxBufferSize` / `maxStorageBufferBindingSize` as
`requiredLimits`; the device is destroyed immediately. Two known differences, both deliberate:
the runtime passes the adapter's *full* limits object, and a `GGML_WEBGPU_GPU_PROFILE` build also
requires `timestamp-query` (not a release build). A GPU that would fail at generation time now fails
at preflight, before the 7.54 GiB install. `test_cache_failure_modes.mjs` cases 7–9 pin the contract:
device refusal and missing `shader-f16` must be rejected, and a healthy adapter must pass — the mock
throws if the preflight stops requesting `shader-f16` or the two limits.

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
