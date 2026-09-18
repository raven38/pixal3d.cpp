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

The wrapper stages the files as `01_front.png` through `04_left.png`, runs MoGe-2 once across those four staged crops, and records each normalized focal estimate. The downstream camera uses one shared focal length. The default aggregation is the **front/back mean**:

```text
fx_shared = (fx_front + fx_back) / 2          # --aggregation front_back (default)
fx_shared = median(fx_front, fx_right, fx_back, fx_left)   # --aggregation median4
fov_x     = 2 * atan(1 / (2 * fx_shared))
```

Why front/back only: on real perspective turntables (85 clothed full-body sets with known 40° FOV) MoGe-2's right/left estimates were biased by about +12° in FOV on 84/85 sets while front/back medians were within 0.3° of the truth. The four-view median therefore lands about +5° high and its spread gate rejected 65% of those inputs; the front/back mean brought the median absolute FOV error from 5.65° to 2.49° (90th percentile 10.2° → 6.3°). Right/left estimates are still computed and reported, but only as a diagnostic.

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

### Inconsistent focal estimates and the fallback policy

Consistency is judged only on the views the aggregate uses; disagreement of the side views is a warning, not a rejection:

```text
front_back:  spread_used = max(fx_front, fx_back) / min(fx_front, fx_back) - 1   rejected when > --max-front-back-diff (default 0.25)
median4:     spread_used = max_i |fx_i / fx_shared - 1|         rejected when > --max-focal-spread (default 0.25)
always:      spread_all  = max_i |fx_i / fx_shared - 1|         reported; right/left deviation > --max-focal-spread only warns
```

When the automatic estimate is rejected:

- `run_pixal3d_canonical4_estimated.py` falls back to the fixed 20° canonical rig (`--fallback-fov`, default `0.3490658503988659`) with a warning, so a batch does not stop on one bad set. Pass `--no-fallback` to fail closed instead.
- `estimate_transforms_moge.py` fails closed unless `--fallback-fov` is given explicitly.

### `camera_estimation.json`

Experiment metadata is not mixed into `transforms.json`. The helper writes a sibling `camera_estimation.json` (the wrapper places it next to the GLB as `<output>.camera_estimation.json`, or at `--camera-json`) with the fields Issue #15 asks to expose:

```json
{
  "source": "moge-2-front_back",
  "estimator": "MoGe-2", "model": "Ruicheng/moge-2-vitl", "package": "3.0.0",
  "package_commit": "<moge git sha>", "revision": "<HF snapshot revision actually loaded, or null>",
  "image_names": ["01_front.png", "02_right.png", "03_back.png", "04_left.png"],
  "fx": {"front": 1.42, "right": 0.91, "back": 1.39, "left": 0.88},
  "fov_deg": {"front": 38.8, "right": 57.7, "back": 39.6, "left": 59.3},
  "aggregation": "front_back",
  "estimated_shared_fx": 1.405, "estimated_fov_rad": 0.6838, "estimated_fov_deg": 39.2,
  "spread_all": 0.37, "spread_used": 0.022, "side_spread": 0.37, "accepted": true,
  "thresholds": {"max_focal_spread": 0.25, "max_front_back_diff": 0.25},
  "warnings": ["right/left focal estimates deviate ... side views are diagnostic only"],
  "selected_fov_rad": 0.6838, "selected_fov_deg": 39.2, "fallback_used": false, "fallback_fov": 0.3490658503988659,
  "mesh_scale": 0.7, "distance": 1.545, "transforms_file": "transforms.json"
}
```

`estimated_*` always hold the MoGe aggregate (also when it was rejected); `selected_*` is what `transforms.json` uses. When the automatic estimate is rejected the wrapper keeps the diagnostic as `<output>.camera_estimation.failed.json` (fail-closed) or writes it with `"source": "fallback"` (fallback used); after a `trellis-cli` failure the diagnostic also goes to the `.failed.json` name so a stale GLB never sits next to a fresh success sidecar.
`source` is `moge-2-front_back` / `moge-2-median4`, `fallback` (automatic estimate rejected, fallback FOV used), `manual` (`--fov`), or `rejected` (written before failing closed).

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
- canonical 4-view focal estimates are aggregated in focal space (front/back mean by default, median4 kept for comparison);
- front/back inconsistency is rejected, side-view disagreement only warns, and the rejected case either falls back to an explicit FOV or fails closed;
- `camera_estimation.json` is written in manual and automatic modes (the automatic path is exercised with `--focals-json`, which aggregates precomputed per-view fx without MoGe);
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
