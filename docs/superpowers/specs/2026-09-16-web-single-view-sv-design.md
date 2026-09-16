# Web Single-View Pixal3D SV Design

## Goal

Make the public Pixal3D Web application accept one pre-matted RGBA image and run the dedicated Pixal3D single-view flow checkpoint family entirely in the browser, without a user-authored `transforms.json`.

The first browser release intentionally does **not** port MoGe-2. It uses a canonical single-view camera gauge with `mesh_scale=1.0`, horizontal FOV 20 degrees by default, and an advanced manual FOV override. MoGe/WebGPU camera estimation is a follow-up.

## Existing contracts

- The browser worker already resolves model files by semantic roles (`ss_flow`, `shape_flow_512`, `shape_flow_1024`, `texture_flow_1024`, etc.), so SV and MV can share the same worker ABI as long as the selected manifest maps those roles to the right filenames.
- The browser runtime consumes a view directory containing image file(s) plus `transforms.json`.
- The current public release manifest is MV-only and contains exactly nine required models.
- The current four-view canonical rig remains supported and unchanged.
- PR #5 provides the canonical single-view camera convention and fixes `mesh_scale=1.0` as the default single-view projection gauge.

## Architecture

### Input mode adapter

Add a mode-aware input layer that supports:

1. **Single image / SV**
   - exactly one pre-matted RGBA image;
   - object-centric alpha crop matching PR #5 in browser JS;
   - synthesize a private one-frame `transforms.json` with canonical front orientation;
   - `mesh_scale=1.0`;
   - default horizontal FOV = 20 degrees (`0.3490658503988659` rad);
   - advanced FOV input must satisfy `0 < fov < pi`;
   - no user scale control in the normal single-view UX.

2. **Four-view / MV**
   - preserve the existing front/right/back/left canonical rig path;
   - preserve explicit mesh-scale confirmation.

3. **Explicit transforms / MV**
   - preserve current behavior.

### Model-set selection

Keep the existing role-based worker unchanged. The app chooses a model family by input mode:

- MV -> existing `pixal3d-q8_0-v1` manifest.
- SV -> a separate manifest URL/source or a locally installed verified SV manifest.

The existing OPFS namespace continues to hold one active verified manifest at a time. On every mode switch and before inference, the app derives the cached manifest family and requires it to match the selected input mode. A cache containing MV weights is therefore not usable for SV, and vice versa.

Until a verified public SV Q8_0 manifest is published, the deployed UI supports local verified installation of an SV manifest + GGUF set and clearly reports that one-click public SV download is not configured. It must never silently run MV weights for a single image.

### Worker

No new WASM inference ABI is required. `worker.js` continues resolving all files by role from the supplied manifest, mounts the generated one-image view package, and calls the existing Pixal3D real geometry/full entrypoint. Legacy hard-coded filenames remain MV-only developer fallback; production SV always requires a manifest.

## Model source configuration

Expose independent source configuration for the two model families:

- MV default manifest: existing `/models/pixal3d-q8_0-v1/pixal3d-models.json`.
- SV manifest: deployment-configurable via `PIXAL3D_SV_MODEL_MANIFEST_URL` / query parameter `sv_manifest_url`.
- SV base URL: `PIXAL3D_SV_MODEL_BASE_URL` / query parameter `sv_model_base_url` / dedicated persisted setting.

If no public SV manifest is configured, the SV release-download button is disabled with actionable text and the local verified install path remains usable.

## UX

The input card has a mode selector:

- `Single image (SV)`.
- `4-view / transforms (MV)`.

To preserve the currently usable public deployment, **MV remains the default when no verified public SV manifest is configured**. When a deployment supplies an SV manifest, the app may default to SV. `?mode=sv` or `?mode=mv` explicitly selects a mode.

SV mode shows one drop target, a compact advanced FOV field, and camera summary (`front`, `20.0 deg`, `mesh_scale 1.0`). MV mode renders the existing calibration UI.

Model storage status always names the installed model set/family. If the cached set does not match the active input mode, Generate stays disabled and the UI tells the user which family is required.

## Validation

Weightless browser tests cover:

- one RGBA image -> browser crop + one-frame transforms metadata;
- default FOV 20 degrees, mesh_scale 1.0 and expected camera distance;
- invalid manual FOV fails closed;
- single-image mode rejects an installed MV manifest;
- MV canonical-rig / explicit-transforms behavior remains unchanged;
- model source resolution distinguishes MV and SV;
- without a configured SV manifest, the public SV download action is disabled and local verified installation remains available;
- production build includes `single_view.js` and injects optional SV source globals only when supplied, without fabricating a manifest.

Existing browser production smoke and canonical-rig tests must remain green.

## Deployment gate

A deploy is allowed when:

- browser unit/smoke tests are green;
- the built site loads and MV remains usable;
- SV mode cannot silently use MV weights;
- if no verified public SV model manifest is available, the deployed site clearly exposes SV as local/custom-model capable rather than claiming one-click public SV download.

A later release can publish the verified SV Q8_0 manifest/weights and enable one-click SV downloads without changing this inference architecture.
