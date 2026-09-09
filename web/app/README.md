# Pixal3D Web production app

The production browser app uses the shared C++/WASM runtime. JavaScript owns only input, model delivery/storage, preflight, progress, and result presentation.

## Initial release surface

- Chrome / Chromium
- WebGPU required
- resolution 1024 only
- pre-matted RGBA multiview input + `transforms.json`
- explicit positive `mesh_scale`
- release model set: `models/pixal3d-q8_0-v1/pixal3d-models.json`

## Model delivery

The committed manifest is the release identity. The default manifest URL is:

```text
/models/pixal3d-q8_0-v1/pixal3d-models.json
```

The GGUF bytes themselves are intentionally not committed to this repository. A deployment supplies their base URL by one of these mechanisms, in priority order:

```js
window.PIXAL3D_MODEL_BASE_URL = 'https://models.example.com/pixal3d-q8_0-v1';
```

or:

```text
?model_base_url=https://models.example.com/pixal3d-q8_0-v1
```

or through the Model download base URL field, which persists the value in origin-local `localStorage`.

A custom manifest can be selected with `window.PIXAL3D_MODEL_MANIFEST_URL` or `?manifest_url=...`, but release deployments should use the committed Q8_0 v1 manifest unless intentionally testing another signed/verified set.

### Model-host requirements

The model host must:

- serve the nine manifest-listed GGUF files by exact filename
- allow CORS from the Pixal3D Web origin
- preserve exact bytes; the browser verifies exact `size_bytes` and SHA-256
- support normal streaming responses for multi-GB objects
- use HTTPS for production deployments

A failed or interrupted file is never marked complete. The browser stores a pending manifest in the Pixal3D OPFS namespace and, on retry, re-hashes already-complete files and reuses them instead of transferring them again. Partial/corrupt files are removed and fetched again. The production `pixal3d-models.json` commit marker is written only after all nine files verify.

## OPFS safety

Only the origin-owned `pixal3d-models-v1` OPFS namespace is managed or recursively deleted. User-selected filesystem files/directories are never deletion targets. Cache deletion is disabled while model installation or generation is active.

## Preflight

Before a model download, the UI checks `navigator.storage.estimate()` against the manifest's required bytes plus a 512 MiB safety margin. Already-occupied bytes from a resumable target set are credited so an interrupted 7.5 GiB install can continue without requiring another full 7.5 GiB of free quota.

Before generation, the UI requires a WebGPU adapter and records its exposed buffer limits. WebGPU does not expose aggregate VRAM, so the C++ runtime's existing `TRELLIS_DEVICE_BUDGET_MB` / token-budget checks remain the authoritative long-run memory gate.

## Release verification

The fast Playwright gate uses a tiny nine-file HTTP model set and covers:

```text
HTTP fetch -> OPFS streaming write -> incremental SHA-256 -> Ready
           -> safe OPFS delete
           -> local verified fallback -> mock inference -> GLB download
```

The final real-device gate remains issue #37: real Chrome/WebGPU with the full 7.54 GiB Q8_0 set, full 1024 inference, browser restart/cache reuse, and cache deletion.
