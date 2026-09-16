# Web Single-Image Pixal3D SV Design

## Goal

Let the existing Pixal3D Web application accept one pre-matted RGBA image and run the dedicated Pixal3D single-view flow family entirely in the browser, without a user-authored `transforms.json`.

The existing multiview paths remain unchanged:

- `transforms.json` + posed RGBA views -> current MV path
- exactly four RGBA images without JSON -> current canonical front/right/back/left rig

A one-image input is a distinct SV path, never `MV@V=1`.

## Input routing

The browser input resolver has three explicit modes:

1. **Single image, no JSON** -> single-view mode.
2. **Exactly four images, no JSON** -> existing canonical multiview rig.
3. **`transforms.json` present** -> existing posed multiview mode.

Two, three, or five-plus images without JSON remain errors. A malformed JSON never falls back to either synthetic mode.

The one-image path initially requires a real alpha matte, matching PR #5. It performs the same object-centric preprocessing contract:

- downscale so the longest side is at most 1024 px,
- foreground is `alpha > 0.8`,
- centered square crop around the foreground bbox,
- 1.1 margin,
- preserve RGBA for Pixal3D conditioning,
- black-composite the same crop for camera estimation.

## Single-view camera contract

For one image, the browser fixes `mesh_scale = 1.0` as the canonical single-view projection gauge. It does not claim a real-world object size.

Given horizontal FOV `fov_x`, camera distance is

```text
distance = 1 / (2 * mesh_scale * tan(fov_x / 2))
```

and the canonical front camera-to-world matrix is

```text
[1, 0,  0, 0]
[0, 0, -1, -distance]
[0, 1,  0, 0]
[0, 0,  0, 1]
```

The browser constructs an in-memory `transforms.json` plus the processed `input.png` and feeds those files to the existing WASM/WebGPU worker. No second projection implementation is introduced.

## Browser FOV estimation

Use the official MoGe-2 ONNX export rather than adding a server dependency.

Initial browser model:

- `Ruicheng/moge-2-vits-normal-onnx`
- `model.onnx`
- expected size: 140,852,051 bytes
- expected SHA-256: `24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0`

The model runs through pinned `onnxruntime-web` with WebGPU preferred. The model is fetched lazily only for the single-image automatic-camera path, verified by exact size and SHA-256, and cached in OPFS separately from the Pixal3D GGUF set.

The ONNX export returns raw point map / mask rather than the full Python `.infer()` postprocess. Port the official MoGe focal-recovery math to JavaScript:

- use the official normalized view-plane UV convention,
- keep mask values above 0.5,
- sample/downsample to at most 64x64 support points,
- solve the 1D camera shift while computing the optimal focal analytically for each candidate shift,
- convert recovered focal to normalized `fx`, then `fov_x = 2 * atan(0.5 / fx)`.

The browser exposes the resulting FOV and estimator source in diagnostics. Invalid/non-finite FOV or degenerate point maps fail closed.

A manual FOV advanced field is retained as a deterministic fallback and for parity/debugging; when supplied it bypasses MoGe.

## Model-set compatibility

The existing Web worker already resolves model files by semantic roles, so the WASM graph does not need a second SV worker. Model variant is derived from the four flow-role filenames:

- all four end in `_sv.gguf` -> `sv`
- all four end in `_mv.gguf` -> `mv`
- mixed/unknown -> invalid manifest

Rules:

- single-image mode requires `sv` and refuses `mv` or mixed sets,
- canonical/posed multiview mode requires `mv` and refuses `sv`,
- no silent checkpoint-family substitution.

The currently published Web release model set is MV-only. This PR therefore supports an installed local verified SV manifest/set, but does not invent size/SHA values for a public SV release. Publishing a public SV model set is a separate release task.

## UI

Reuse the existing input card rather than adding a second uploader.

For a single image the card displays:

- `Single image · Pixal3D SV`
- processed preview,
- camera source (`MoGe-2 WebGPU` or `manual FOV`),
- estimated FOV when available,
- `mesh_scale 1.0 (canonical gauge)`.

Generation stays disabled until:

- WebGPU preflight passes,
- a complete model set is cached,
- the model set is `sv`,
- the single image has a valid alpha matte and preprocessing succeeded,
- a valid FOV is available or can be estimated.

With the default public MV set installed, selecting one image shows an actionable message to install a verified SV model set instead of running MV weights.

## Storage and deployment

Keep the existing Pixal3D OPFS model cache contract. The MoGe ONNX cache is separate so deleting/replacing the Pixal3D model set does not force a re-download of the camera model.

`build_web_dist.sh` vendors the pinned ONNX Runtime Web browser assets from `node_modules` and copies the new camera modules. No runtime JavaScript CDN is required.

The public deployment remains safe before an SV GGUF release: four-view MV keeps working; one-image mode is present but gated on installing an SV model set.

## Testing

### Pure Node tests

Cover:

- model-set variant detection (`sv`, `mv`, mixed rejection),
- one-image routing vs four-image canonical routing,
- canonical camera JSON and `mesh_scale=1.0`,
- FOV validation,
- focal recovery on a synthetic projected point map with known focal and shift,
- preprocessing geometry helpers where DOM-independent.

### Browser smoke

Extend production smoke with mocked camera estimation/model metadata to verify:

- one RGBA image selects SV mode,
- one image + MV manifest never enables Generate,
- one image + SV manifest produces one processed image + one generated transforms JSON for the worker,
- four-view canonical MV path remains unchanged,
- malformed/ambiguous input fails closed.

### Real browser acceptance

Before enabling a public SV default model set:

- run the browser MoGe FOV estimator against known-camera examples and record FOV error,
- run WebGPU SV generation with real SV GGUFs,
- compare against PR #5 / official Python single-view runs,
- confirm no `MV@V=1` checkpoint path is reachable.

## Non-goals

- public hosting of the SV GGUF set without verified size/SHA metadata,
- RGB background removal in this PR,
- arbitrary two/three-view pose estimation,
- real-world scale estimation,
- replacing the existing four-view canonical rig.
