# Image Camera Transform Estimation Design

## Goal

Allow the Pixal3D camera-aware path to run without a user-provided `transforms.json` by estimating camera metadata from the supplied image(s), while keeping `mesh_scale` explicit and preserving PR #3's fixed 4-view canonical rig as a separate path.

The immediate product target is the original request: **one pre-matted image, no `transforms.json`, Pixal3D inference**. Arbitrary multi-view pose estimation is supported by the estimator protocol but is not guessed by the bundled backend until it is validated.

## Existing contracts

- `TransformsFile` is the single downstream camera contract: top-level/per-frame `camera_angle_x`, `mesh_scale`, and Blender/NeRF-style camera-to-world matrices.
- PR #3 resolves missing metadata for exactly four known turntable views by assigning the fixed front/right/back/left rig.
- `mesh_scale` remains explicit. Camera estimation must not infer or overwrite absolute object scale.
- A provided `transforms.json` always wins. Malformed/unreadable JSON fails closed and never falls back to estimation.

## Architecture

### C++ resolver

Add a small camera-estimation module that can produce `TransformsFile` in two ways:

1. **Manual FOV**: for exactly one image, `camera_angle_x` is supplied explicitly. The runtime computes camera distance using the same equation as official Pixal3D single-image inference with `extend_pixel=0`:

   `distance = 1 / (2 * mesh_scale * tan(camera_angle_x / 2))`

   The remaining gauge freedom is fixed to the canonical front orientation used by the existing Pixal3D rig.

2. **External estimator**: an administrator/user-controlled estimator executable receives `--views-dir`, `--mesh-scale`, and `--output`. It writes a normal Pixal3D `transforms.json` to the private output path. C++ parses that file through the existing parser, verifies that it kept the caller's explicit `mesh_scale`, verifies the returned frame set matches the image files in the directory, and then deletes the temporary file.

The estimator executable is invoked with argv (`fork`/`execvp` on POSIX, `_spawnvp` on Windows), not through a shell, so uploaded paths are never shell-interpreted.

### Bundled MoGe-2 estimator

`tools/estimate_transforms_moge.py` implements the first backend. It follows official Pixal3D `inference.py` for the one-image case:

- load the RGB image (transparent pixels composited onto white),
- run `Ruicheng/moge-2-vitl`,
- derive horizontal FOV from normalized `intrinsics[0,0]`,
- compute distance from FOV and the explicit `mesh_scale`,
- emit one front-view `TransformsFile`.

The helper deliberately rejects two/three/arbitrary multi-view inputs for now instead of inventing relative poses. A different executable implementing the same output contract can handle them later without changing Pixal3D runtime code.

### Resolution order when `transforms.json` is absent

1. `--camera-fov F` (exactly one image) → synthesize estimated single-view metadata without Python.
2. `--estimate-camera --camera-estimator PATH` → run the external estimator.
3. Otherwise, existing PR #3 behavior remains: exactly four views + explicit `mesh_scale` → fixed canonical rig.
4. Everything else fails with an actionable error.

Estimation is opt-in for the first release. This prevents a machine without MoGe dependencies from unexpectedly spawning Python and keeps the known-camera 4-view path deterministic.

## CLI / server integration

### CLI

New flags:

- `--estimate-camera`: enable the external estimator when no JSON is present.
- `--camera-estimator PATH`: estimator executable/script. `PIXAL3D_CAMERA_ESTIMATOR` is the environment fallback.
- `--camera-fov RAD`: manual single-image FOV, matching official Pixal3D's manual-FOV fallback.

`--mesh-scale` remains required whenever camera metadata is synthesized/estimated.

### Server

`POST /generate-mv` may receive `estimate_camera=1` or `camera_fov=<rad>`. The estimator executable is a **server launch setting**, never a request-provided command. The server estimates/validates metadata before model execution, writes the resulting metadata into its private staging directory, then the existing model path consumes it once; this avoids running a heavy estimator twice.

## Error handling

Fail closed on:

- missing/non-positive/non-finite `mesh_scale`,
- invalid FOV (`<=0`, `>=pi`, NaN/inf),
- estimator executable missing or non-zero exit,
- missing/malformed estimator output,
- output `mesh_scale` differing from the explicit input,
- output frame list not matching the actual input image files,
- arbitrary multi-view input when the bundled MoGe backend is used.

Do not fall back from a failed estimator to the fixed 4-view rig: the user explicitly chose estimated-camera semantics.

## Validation

TDD covers the C++ contract without downloading model weights:

- official distance equation at a known FOV/scale,
- exact front c2w orientation/translation for manual FOV,
- invalid FOV/scale rejection,
- external-estimator output validation and frame-set validation using a tiny fake estimator,
- resolver precedence: JSON > manual FOV > estimator > canonical rig.

The bundled Python helper has a manual-FOV mode used in CI so its JSON shape/math can be tested without torch/MoGe. A real MoGe parity run is documented separately because downloading a ViT-L checkpoint is not suitable for routine CI.

## Non-goals of this PR

- estimating absolute `mesh_scale`,
- silently forcing one image into the fixed four-view rig,
- adding separate Pixal3D single-view flow checkpoints,
- claiming arbitrary multi-view pose estimation from the bundled MoGe backend,
- WebGPU-porting MoGe-2.

The external estimator boundary is intentionally the seam for a later multi-view SfM/VGGT/MoGe alignment backend and for a future browser-native estimator.