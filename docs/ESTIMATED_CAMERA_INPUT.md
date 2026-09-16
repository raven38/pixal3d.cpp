# One-image camera estimation (no transforms.json)

Issue #4 tracks removing the requirement for user-authored camera metadata. This branch adds the first milestone: a **single pre-matted image** can be cropped/normalized like official Pixal3D, converted into camera metadata, and run with the dedicated Pixal3D single-view flow weights.

This branch depends on the runtime support from PR #6 (`--pixal3d-weights sv`). It deliberately refuses the older `V=1` + multiview-weight path because that combination was measured to break down badly.

## What is estimated

The bundled helper follows TencentARC/Pixal3D `inference.py`:

1. The wrapper mirrors `pipeline.preprocess_image()` for pre-matted RGBA input: downscale the longest side to at most 1024, find the `alpha > 0.8` object bbox, and take the centered square crop with the reference 1.1 margin.
2. The crop stays RGBA for pixal3d.cpp. Its conditioner premultiplies RGB by alpha, so transparent pixels become black.
3. MoGe-2 (`Ruicheng/moge-2-vitl`) sees the same crop explicitly alpha-composited onto black. Hidden RGB values under transparent pixels never leak into the camera estimator.
4. MoGe predicts normalized camera intrinsics and horizontal FOV is derived from normalized `fx`:

   `fov_x = 2 * atan(1 / (2 * fx_normalized))`

5. The one-image projection gauge uses **`mesh_scale = 1.0` by default**, matching official Pixal3D. This is not an estimate of physical object size. `--mesh-scale` remains available as an advanced override.
6. Camera distance is derived from FOV and scale using the official single-image equation (with `extend_pixel=0`):

   `distance = 1 / (2 * mesh_scale * tan(fov_x / 2))`

7. The remaining one-view gauge freedom is fixed to the canonical front-view camera orientation already used by Pixal3D.

Because the grid normalization and camera distance both scale with `1 / mesh_scale`, changing `mesh_scale` while recomputing the corresponding distance is a single-view projection-gauge change, not a claim about real-world meters or centimeters. The result is written as a private ordinary `transforms.json`, so no downstream projection math is duplicated.

## Generate a GLB from one image

The input must already be a pre-matted RGBA image with a real alpha channel. This milestone does not run background removal in Python.

```bash
python3 tools/run_pixal3d_estimated.py \
  --image character.png \
  --models pixal3d_models \
  --trellis-cli ./build/trellis-cli \
  --output character.glb
```

No scale argument is required for the normal one-image path; the wrapper uses the canonical `mesh_scale=1.0` gauge. To override it for experiments:

```bash
python3 tools/run_pixal3d_estimated.py \
  --image character.png \
  --models pixal3d_models \
  --mesh-scale 0.5 \
  --output character.glb
```

The override is still validated as finite and positive, and the camera distance is recomputed from the same official equation.

The wrapper creates a private temporary views directory, writes the official-style object crop as `input.png`, estimates camera metadata, then invokes:

```text
trellis-cli --views <private-dir> --pixal3d-weights sv ...
```

The temporary `transforms.json` is not a user-facing input. The selected model directory must contain the four SV flow files introduced by PR #6 (`pixal3d_*_sv.gguf`) plus the shared Pixal3D models. Until an SV release manifest is published, prepare that model directory using PR #6's conversion/download instructions rather than the current MV-only installer set.

### Manual FOV fallback

To bypass MoGe and supply horizontal FOV directly (radians):

```bash
python3 tools/run_pixal3d_estimated.py \
  --image character.png \
  --models pixal3d_models \
  --fov 0.3490658503988659 \
  --output character.glb
```

Manual-FOV mode skips torch/MoGe, but the wrapper still uses Pillow for the reference-compatible RGBA crop.

## Dependencies

The wrapper always needs:

- Pillow (object crop / RGBA staging)

Automatic FOV estimation additionally imports:

- PyTorch
- NumPy
- MoGe-2 (`from moge.model.v2 import MoGeModel`)

The default model is `Ruicheng/moge-2-vitl`. The helper chooses CUDA when available and otherwise falls back to CPU; `--device` can override this.

These Python dependencies and the MoGe checkpoint are **not** bundled into the standalone C++ runtime in this milestone.

## Important scope limits

- The wrapper fixes the unobservable **single-view projection gauge** to `mesh_scale=1.0` by default. It does **not** estimate real-world object size.
- `--mesh-scale` is an advanced positive finite override only; it is not needed for the normal one-image UX.
- This milestone estimates the one-image camera/FOV case only. The bundled helper rejects multiple images instead of inventing relative poses.
- PR #3's exact four-view front/right/back/left canonical rig remains a separate known-camera shortcut; its scale handling is unchanged.
- Generation uses the dedicated Pixal3D **SV flow checkpoint family** through PR #6, not the multiview weights with `V=1`.
- Arbitrary multi-image relative-pose estimation and browser-native camera estimation remain follow-ups under Issue #4.

## Validation

`tools/test_estimate_transforms_moge.py` verifies, without downloading MoGe weights, that:

- the official FOV-to-distance equation is reproduced,
- the generated JSON has the expected one-frame camera convention,
- invalid `mesh_scale` fails closed,
- multiple images are rejected by the bundled one-image backend,
- transparent hidden RGB is black-composited before MoGe.

`tools/test_run_pixal3d_estimated.py` uses a real RGBA fixture and a recording fake CLI to verify that:

- the object-centric alpha crop is applied,
- the staged crop preserves alpha while transparent pixels resolve to black,
- the private image is normalized to `input.png`,
- the CLI is invoked with `--pixal3d-weights sv`,
- omitting `--mesh-scale` produces canonical `mesh_scale=1.0`,
- an explicit positive override changes both `mesh_scale` and the derived camera distance consistently,
- invalid explicit scale still fails closed,
- the wrapper refuses a CLI build that predates PR #6.

A full quality acceptance run should compare official Pixal3D single-view inference against pixal3d.cpp using the same image, camera metadata, canonical scale gauge, seed, and SV checkpoint family. That run is separate from routine CI because it requires the full model set/GPU.
