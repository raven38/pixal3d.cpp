# Camera/FOV estimation without user-authored transforms.json

Issue #4 tracks removing the requirement for user-authored camera metadata. Two bounded helpers are available without changing the downstream Pixal3D camera contract:

- one pre-matted image -> MoGe-2 FOV -> canonical front camera -> dedicated SV flow weights;
- explicit front/right/back/left images -> MoGe-2 per-view focal estimates -> one shared FOV -> fixed canonical 4-view extrinsics -> MV flow weights.

Both paths write a private ordinary `transforms.json` and feed the existing runtime rather than duplicating projection math.

## Shared preprocessing contract

The wrappers mirror the Pixal3D object-centric preprocessing used for camera estimation:

1. downscale the longest side to at most 1024;
2. find the `alpha > 0.8` foreground bbox;
3. take a centered square crop with the existing 1.1 margin;
4. preserve RGBA for pixal3d.cpp;
5. explicitly alpha-composite the same staged crop onto black before MoGe.

This keeps hidden RGB under transparent pixels out of the estimator and makes the generator and camera estimator observe the same crop.

MoGe-2 (`Ruicheng/moge-2-vitl`) predicts normalized camera intrinsics. Horizontal FOV is derived from normalized `fx` as:

```text
fov_x = 2 * atan(1 / (2 * fx_normalized))
```

## One-image path

The single-image path fixes the unobservable projection gauge to `mesh_scale = 1.0` by default, matching official Pixal3D. This is not an estimate of physical object size. `--mesh-scale` remains an advanced override.

Camera distance uses the official single-image equation (`extend_pixel=0`):

```text
distance = 1 / (2 * mesh_scale * tan(fov_x / 2))
```

Generate with:

```bash
python3 tools/run_pixal3d_estimated.py \
  --image character.png \
  --models pixal3d_models \
  --trellis-cli ./build/trellis-cli \
  --output character.glb
```

The private runtime call is equivalent to:

```text
trellis-cli --views <private-dir> --pixal3d-weights sv ...
```

To bypass MoGe and provide horizontal FOV directly (radians):

```bash
python3 tools/run_pixal3d_estimated.py \
  --image character.png \
  --models pixal3d_models \
  --fov 0.3490658503988659 \
  --output character.glb
```

Manual-FOV mode skips torch/MoGe, but still uses Pillow for the matched RGBA crop.

## Canonical 4-view shared-FOV path

For known turntable views, camera orientation is already observed by construction. Do not inject free pose-estimation noise into front/right/back/left. Instead, pass the four identities explicitly:

```bash
python3 tools/run_pixal3d_canonical4_estimated.py \
  --front front.png \
  --right right.png \
  --back back.png \
  --left left.png \
  --models pixal3d_models \
  --mesh-scale 0.7 \
  --trellis-cli ./build/trellis-cli \
  --output character.glb
```

The wrapper stages the files as `01_front.png` through `04_left.png`, runs MoGe-2 once across those four staged crops, and records each normalized focal estimate. The downstream camera uses one shared focal length:

```text
fx_shared = median(fx_front, fx_right, fx_back, fx_left)
fov_x = 2 * atan(1 / (2 * fx_shared))
```

The four canonical rotations remain exactly the existing 0° / 90° / 180° / 270° rig. `mesh_scale` remains explicit and keeps its existing MV meaning; no physical object scale is inferred.

### Canonical distance when FOV changes

The existing no-JSON canonical rig is calibrated at:

```text
FOV_0      = 20°
distance_0 = 3.1192049980163574
```

When a different shared FOV is selected, the helper preserves that rig's image-plane projection gauge by scaling only the orbit distance:

```text
distance = distance_0 * tan(FOV_0 / 2) / tan(fov_x / 2)
```

This leaves the canonical rotations and MV `mesh_scale` contract unchanged.

### Inconsistent focal estimates

The automatic 4-view path computes the maximum relative deviation from the median focal estimate:

```text
spread = max_i |fx_i / fx_shared - 1|
```

The default fail-closed threshold is `0.25` (25%). If the four views disagree more strongly, the helper refuses to write camera metadata instead of silently trusting a questionable median. The threshold can be adjusted with `--max-focal-spread` for experiments.

For orthographic-looking character sheets or inputs where MoGe should not be trusted, keep the deterministic fixed/manual FOV path:

```bash
python3 tools/run_pixal3d_canonical4_estimated.py \
  --front front.png --right right.png --back back.png --left left.png \
  --models pixal3d_models \
  --mesh-scale 0.7 \
  --fov 0.3490658503988659 \
  --output character.glb
```

The private runtime call uses the normal MV family:

```text
trellis-cli --views <private-dir> --pixal3d-weights mv ...
```

## Dependencies

Both wrappers always need:

- Pillow (object crop / RGBA staging)

Automatic FOV estimation additionally imports:

- PyTorch
- NumPy
- MoGe-2 (`from moge.model.v2 import MoGeModel`)

The default model is `Ruicheng/moge-2-vitl`. The estimator chooses CUDA when available and otherwise falls back to CPU; `--device` can override this.

These Python dependencies and the MoGe checkpoint are not bundled into the standalone C++ runtime.

## Important scope limits

- Neither helper estimates physical object scale.
- The one-image path estimates FOV only and fixes the remaining pose gauge to the canonical front camera.
- The canonical 4-view path estimates one shared FOV only; front/right/back/left extrinsics remain fixed.
- Arbitrary multi-image relative-pose estimation remains a separate part of Issue #4 and should use a joint camera/geometry estimator rather than this canonical helper.
- Browser-native FOV estimation remains separate (#12).
- Near-orthographic character sheets may be better served by fixed/manual 20° than by a pinhole FOV estimate; do not treat MoGe as authoritative there without validation.

## Validation

`tools/test_estimate_transforms_moge.py` verifies without downloading MoGe weights that:

- the official one-image FOV-to-distance equation is reproduced;
- single-image JSON keeps the expected camera convention;
- hidden RGB is black-composited before MoGe;
- canonical 4-view focal estimates are aggregated in focal space;
- strongly inconsistent focal estimates fail closed;
- changing canonical FOV rescales distance while preserving all four canonical rotations;
- invalid `mesh_scale` fails closed.

`tools/test_run_pixal3d_estimated.py` verifies the single-image crop, SV weight selection, scale gauge, and failure handling with a recording fake CLI.

`tools/test_run_pixal3d_canonical4_estimated.py` verifies that:

- all four input images receive the object-centric RGBA crop;
- transparent pixels resolve to black for conditioning/estimation;
- explicit view identity maps to `01_front` / `02_right` / `03_back` / `04_left`;
- generated camera metadata uses the canonical rotations and shared FOV;
- `mesh_scale` stays explicit;
- the runtime is invoked with `--pixal3d-weights mv`;
- invalid scale and missing inputs fail closed.

The no-model CI validates contracts only. The remaining #15 quality gate is a GPU/model comparison of the existing fixed-20° canonical baseline against automatic shared-FOV MoGe on representative perspective and near-orthographic inputs, including downstream silhouette/render/mesh metrics.
