# One-image camera estimation (no transforms.json)

Issue #4 tracks removing the requirement for user-authored camera metadata. This branch adds the first milestone: a **single pre-matted image** can be converted into the camera metadata consumed by the existing Pixal3D camera-aware pipeline.

## What is estimated

The bundled helper follows TencentARC/Pixal3D `inference.py`:

1. MoGe-2 (`Ruicheng/moge-2-vitl`) predicts normalized camera intrinsics.
2. Horizontal FOV is derived from normalized `fx`:

   `fov_x = 2 * atan(1 / (2 * fx_normalized))`

3. `mesh_scale` remains an explicit user input.
4. Camera distance is derived from FOV and scale using the official single-image equation (with `extend_pixel=0`):

   `distance = 1 / (2 * mesh_scale * tan(fov_x / 2))`

5. The remaining one-view gauge freedom is fixed to the canonical front-view camera orientation already used by Pixal3D.

The result is written as an ordinary `transforms.json`, so no downstream projection math is duplicated.

## Generate a GLB from one image

The current Pixal3D `--views` loader requires a real alpha channel, so the image should already be a pre-matted RGBA PNG.

```bash
python3 tools/run_pixal3d_estimated.py \
  --image character.png \
  --models pixal3d_models \
  --mesh-scale 1.0 \
  --trellis-cli ./build/trellis-cli \
  --output character.glb
```

The wrapper creates a private temporary views directory, estimates camera metadata, then runs the existing `trellis-cli --views ...` pipeline. The temporary `transforms.json` is not a user-facing input.

### Manual FOV fallback

To bypass MoGe and supply horizontal FOV directly (radians):

```bash
python3 tools/run_pixal3d_estimated.py \
  --image character.png \
  --models pixal3d_models \
  --mesh-scale 1.0 \
  --fov 0.3490658503988659 \
  --output character.glb
```

Manual-FOV mode uses only the Python standard library in the estimator helper and is the mode exercised by routine CI.

## Automatic estimator dependencies

Automatic FOV estimation imports the same components used by the official Pixal3D path:

- PyTorch
- NumPy
- Pillow
- MoGe-2 (`from moge.model.v2 import MoGeModel`)

The default model is `Ruicheng/moge-2-vitl`. The helper chooses CUDA when available and otherwise falls back to CPU; `--device` can override this.

These Python dependencies and the MoGe checkpoint are **not** bundled into the standalone C++ runtime in this milestone.

## Important scope limits

- `mesh_scale` is never guessed. Single-image absolute object scale is ambiguous, so a finite positive value remains required.
- This milestone estimates the one-image camera/FOV case only. The bundled helper rejects multiple images instead of inventing relative poses.
- PR #3's exact four-view front/right/back/left canonical rig remains a separate known-camera shortcut.
- The downstream generation path is the repository's existing Pixal3D camera-aware multiview cascade with `V=1`; this change solves the **camera-metadata input problem**, not the separate model-quality question of dedicated single-view checkpoints.
- Arbitrary multi-image relative-pose estimation and browser-native camera estimation remain follow-ups under Issue #4.

## Validation

`tools/test_estimate_transforms_moge.py` verifies, without downloading model weights, that:

- the official FOV-to-distance equation is reproduced,
- the generated JSON has the expected one-frame camera convention,
- invalid `mesh_scale` fails closed,
- multiple images are rejected by the bundled one-image backend.

`tools/test_run_pixal3d_estimated.py` verifies the wrapper orchestration using a no-op CLI executable, so CI checks staging and estimator invocation without loading GGUF weights.