# Pixal3D Web production app

The production browser app uses the shared C++/WASM runtime. JavaScript owns only input, model delivery/storage, preflight, progress, and result presentation.

## Browser inference modes

### Single image (SV)

- one pre-matted RGBA image
- no user-authored `transforms.json`
- browser object-centric alpha crop
- canonical front camera
- `mesh_scale = 1.0`
- horizontal FOV 20° by default, with an advanced manual override
- dedicated Pixal3D **single-view** flow weights only

The browser synthesizes the same one-frame camera metadata consumed by the existing Pixal3D WASM path. The first Web release does **not** port MoGe-2; browser-native camera/FOV estimation is a follow-up.

Single-image mode never falls back to the multiview checkpoints. The cached model manifest must identify an SV family (`model_family: "sv"` or `_sv.gguf` flow filenames) before Generate can be enabled.

### Multiview (MV)

The existing modes remain unchanged:

- pre-matted RGBA multiview input + `transforms.json`, or
- exactly four canonical turntable views (front/right/back/left) without JSON, with explicit positive `mesh_scale` confirmation.

Resolution is 1024 in the browser for both families.

## Model delivery

### MV public release

The committed MV release manifest remains:

```text
/models/pixal3d-q8_0-v1/pixal3d-models.json
```

The default MV GGUF base URL can be configured with:

```js
window.PIXAL3D_MODEL_BASE_URL = 'https://models.example.com/pixal3d-q8_0-v1';
```

or:

```text
?model_base_url=https://models.example.com/pixal3d-q8_0-v1
```

or the Model download base URL field, which persists to origin-local `localStorage`.

A custom MV manifest can still be selected with `window.PIXAL3D_MODEL_MANIFEST_URL` or `?manifest_url=...`.

### SV model source

The verified public SV Q8_0 set is `pixal3d-sv-q8_0 v1` on Hugging Face
([`raven38/pixal3d-sv-q8_0-v1`](https://huggingface.co/raven38/pixal3d-sv-q8_0-v1); manifest
`models/pixal3d-sv-q8_0-v1/pixal3d-models.json`, `model_family: sv`). The production build does **not**
bake it in: the SV source must be configured explicitly at deploy time (`scripts/build_web_dist.sh`
arguments 2/3 or `PIXAL3D_SV_MODEL_MANIFEST_URL` / `PIXAL3D_SV_MODEL_BASE_URL`), and the app never
silently reuses the MV set for a single image:

```text
sv_manifest_url   = https://huggingface.co/raven38/pixal3d-sv-q8_0-v1/resolve/main/pixal3d-models.json
sv_model_base_url = https://huggingface.co/raven38/pixal3d-sv-q8_0-v1/resolve/main
```

An SV source can be configured with:

```js
window.PIXAL3D_SV_MODEL_MANIFEST_URL = 'https://models.example.com/pixal3d-sv-q8_0-v1/pixal3d-models.json';
window.PIXAL3D_SV_MODEL_BASE_URL = 'https://models.example.com/pixal3d-sv-q8_0-v1';
```

or query parameters:

```text
?mode=sv&sv_manifest_url=https://models.example.com/pixal3d-sv-q8_0-v1/pixal3d-models.json&sv_model_base_url=https://models.example.com/pixal3d-sv-q8_0-v1
```

When no public SV manifest is configured, select **Single image (SV)** and use **Install local verified set…** with `pixal3d-models.json` plus all GGUFs. The manifest must describe the SV family.

### Model-host requirements

The model host must:

- serve the manifest-listed GGUF files by exact filename
- allow CORS from the Pixal3D Web origin
- preserve exact bytes; the browser verifies exact `size_bytes` and SHA-256
- support normal streaming responses for multi-GB objects
- use HTTPS for production deployments

A failed or interrupted file is never marked complete. The browser stores a pending manifest in the Pixal3D OPFS namespace and, on retry, re-hashes already-complete files and reuses them instead of transferring them again. Partial/corrupt files are removed and fetched again. The production `pixal3d-models.json` commit marker is written only after every required file verifies.

## OPFS safety

Only the origin-owned `pixal3d-models-v1` OPFS namespace is managed or recursively deleted. User-selected filesystem files/directories are never deletion targets. Cache deletion is disabled while model installation or generation is active.

The cache currently holds one active model-set manifest at a time. Switching SV/MV modes checks manifest family compatibility and keeps Generate disabled when the cached set belongs to the other family.

## Preflight

Before a model download, the UI checks `navigator.storage.estimate()` against the manifest's required bytes plus a 512 MiB safety margin. Already-occupied bytes from a resumable target set are credited.

Before generation, the UI requires a WebGPU adapter and records its exposed buffer limits. WebGPU does not expose aggregate VRAM, so the C++ runtime's existing device/token-budget checks remain the authoritative long-run memory gate.

## Release verification

The Playwright gate covers both families without downloading real models:

```text
MV remote manifest -> OPFS streaming write -> incremental SHA-256 -> Ready
                   -> safe OPFS delete
                   -> local verified MV -> mock inference -> GLB download
                   -> switch to SV (MV cache rejected)
                   -> local verified SV -> one RGBA input
                   -> browser crop + synthesized camera JSON
                   -> mock SV inference
```

A real-device release gate with the full public SV checkpoint set remains separate from this weightless browser contract.
