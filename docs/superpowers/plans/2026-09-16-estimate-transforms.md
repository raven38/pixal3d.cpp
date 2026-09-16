# Camera Transform Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Pixal3D camera-aware path accept one image without `transforms.json` by estimating FOV with a swappable external estimator and converting the result into the existing `TransformsFile` contract.

**Architecture:** Keep all downstream Pixal3D projection code unchanged. Add a focused `camera_estimator` module for manual-FOV synthesis and external-estimator execution; add a MoGe-2 Python backend that emits normal transforms metadata. Missing JSON resolution stays deterministic: explicit manual FOV, explicit estimator, existing 4-view canonical rig, otherwise error.

**Tech Stack:** C++17, `std::filesystem`, POSIX `fork/execvp` / Windows `_spawnvp`, existing transforms parser, Python 3 helper using MoGe-2/torch/Pillow for real inference.

**Spec:** `docs/superpowers/specs/2026-09-16-estimate-transforms-design.md`

## Global Constraints

- A supplied `transforms.json` always wins; malformed/unreadable JSON never falls back.
- `mesh_scale` is explicit and finite/positive for every synthesized or estimated camera path.
- Camera estimation is opt-in in this PR.
- Estimator executables are configured by the process owner, never by an HTTP request.
- The bundled MoGe backend supports exactly one image; unsupported multi-view input fails closed.
- PR #3's fixed 4-view canonical rig remains unchanged unless estimation is explicitly requested.

---

### Task 1: Lock the camera math and resolver contract with failing tests

**Files:**
- Create: `src/test_camera_estimator.cpp`
- Modify: `CMakeLists.txt`
- Modify: `.github/workflows/linux-build.yml`

**Interfaces:**
- Consumes: existing `TransformsFile`, `list_view_images`, `synthesize_canonical_rig`.
- Produces expected API for Task 2:
  - `float single_view_distance_from_fov(float fov_x, float mesh_scale)`
  - `bool synthesize_single_view_camera(const std::string&, float, float, TransformsFile&, std::string&)`
  - `bool estimate_transforms_external(const std::string&, float, const std::string&, TransformsFile&, std::string&)`

- [ ] **Step 1: Write the failing C++ test**

Cover:

```cpp
check_close(single_view_distance_from_fov(0.3490658503988659f, 1.0f),
            2.8356409f, 1e-5f, "official distance equation");

TransformsFile one;
std::string err;
check(synthesize_single_view_camera("front.png", 0.3490658503988659f, 1.0f, one, err),
      "single-view camera synthesis");
check(one.frames.size() == 1, "one frame");
check_close(one.frames[0].transform_matrix[7], -2.8356409f, 1e-5f,
            "front camera distance");
check(!synthesize_single_view_camera("front.png", 0.0f, 1.0f, one, err),
      "zero FOV rejected");
check(!synthesize_single_view_camera("front.png", 0.3f, 0.0f, one, err),
      "zero scale rejected");
```

Also create a temporary fake estimator script that writes a valid one-frame `transforms.json`, call `estimate_transforms_external`, and verify the frame set and scale. A second fake output with the wrong `mesh_scale` must be rejected.

- [ ] **Step 2: Push the test-only commit and verify RED in Linux CI**

Expected: `Linux C++ build` fails because `camera_estimator.h` / functions do not exist yet.

- [ ] **Step 3: Keep the test target in routine CI**

After implementation, CI must execute:

```bash
./build/trellis-test-camera-estimator
```

Expected: PASS without model weights.

---

### Task 2: Implement the C++ camera-estimator module

**Files:**
- Create: `include/camera_estimator.h`
- Create: `src/camera_estimator.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
float single_view_distance_from_fov(float fov_x, float mesh_scale);

bool synthesize_single_view_camera(
    const std::string& file_path,
    float fov_x,
    float mesh_scale,
    TransformsFile& out,
    std::string& error);

bool estimate_transforms_external(
    const std::string& views_dir,
    float mesh_scale,
    const std::string& estimator_executable,
    TransformsFile& out,
    std::string& error);
```

- [ ] **Step 1: Implement official single-view camera math**

Use:

```cpp
return 1.0f / (2.0f * mesh_scale * std::tan(fov_x * 0.5f));
```

and the canonical front camera rotation:

```text
[1,0,0,0]
[0,0,-1,-distance]
[0,1,0,0]
[0,0,0,1]
```

Reject non-finite values, `mesh_scale <= 0`, `fov <= 0`, and `fov >= pi`.

- [ ] **Step 2: Implement argv-based subprocess execution**

For `.py` estimator paths invoke `python3` on POSIX and `python` on Windows; otherwise execute the path directly. Pass only argv values:

```text
--views-dir <dir> --mesh-scale <value> --output <private-temp-json>
```

Use `fork` + `execvp` + `waitpid` on POSIX and `_spawnvp(_P_WAIT, ...)` on Windows. Do not invoke a shell.

- [ ] **Step 3: Validate estimator output**

Parse the private JSON with `load_transforms_json`, then verify:

- output `mesh_scale` equals caller scale within `1e-5 * max(1, abs(scale))`,
- output frame count equals the discovered input image count,
- sorted output `file_path` basenames equal `list_view_images(views_dir)`,
- every FOV and all 16 transform elements are finite.

Always remove the private output file.

- [ ] **Step 4: Run `trellis-test-camera-estimator`**

Expected: PASS.

---

### Task 3: Integrate resolver precedence into CLI/runtime

**Files:**
- Modify: `include/trellis_args.h`
- Modify: `src/trellis_args.cpp`
- Modify: `include/transforms_json.h`
- Modify: `src/transforms_json.cpp`
- Modify: `include/pixal3d_input.h`
- Modify: `src/pixal3d_input.cpp`
- Modify: `src/trellis_cli.cpp`

**Interfaces:**
- Add params:

```cpp
bool estimate_camera = false;
std::string camera_estimator;
float camera_fov = 0.0f;
bool camera_fov_set = false;
```

- [ ] **Step 1: Add strict CLI parsing**

Flags:

```text
--estimate-camera
--camera-estimator PATH
--camera-fov RAD
```

`--camera-fov` must parse as finite `0 < F < pi`. `PIXAL3D_CAMERA_ESTIMATOR` is the environment fallback for the executable path.

- [ ] **Step 2: Extend metadata resolution only for missing JSON**

Add optional resolution inputs to `load_views_metadata` / `pixal3d_load_input_views`. Precedence:

```text
transforms.json exists -> parse existing file
else camera_fov_set    -> require exactly one image, synthesize_single_view_camera
else estimate_camera   -> estimate_transforms_external
else                    -> existing 4-view canonical-rig behavior
```

A failed explicit estimator must return its error and must not fall through to canonical-rig synthesis.

- [ ] **Step 3: Log the selected camera source**

CLI prints one of:

```text
camera source: transforms.json
camera source: manual FOV
camera source: estimated
camera source: canonical 4-view rig
```

- [ ] **Step 4: Run existing and new weightless tests**

Expected:

```bash
./build/trellis-test-transforms-synth docker/linux-webgpu-gate/e2e/views
./build/trellis-test-camera-estimator
```

both PASS.

---

### Task 4: Add the bundled MoGe-2 one-image backend

**Files:**
- Create: `tools/estimate_transforms_moge.py`
- Create: `tools/test_estimate_transforms_moge.py`

**Interfaces:**
- CLI:

```text
estimate_transforms_moge.py --views-dir DIR --mesh-scale F --output PATH [--manual-fov RAD]
```

- [ ] **Step 1: Write the Python math/JSON test first**

The test creates a temporary directory with one tiny PNG and calls the helper with `--manual-fov 0.3490658503988659`, which must not import torch/MoGe. Assert:

```python
assert data["mesh_scale"] == 1.0
assert len(data["frames"]) == 1
assert abs(data["camera_angle_x"] - 0.3490658503988659) < 1e-12
assert abs(data["frames"][0]["transform_matrix"][1][3] + 2.8356409) < 1e-5
```

Also assert two images are rejected by the bundled backend.

- [ ] **Step 2: Implement manual mode**

Use only Python stdlib + Pillow in manual mode. Composite transparent pixels onto white and emit the standard transforms JSON.

- [ ] **Step 3: Implement MoGe mode**

Lazy-import `torch`, `numpy`, `PIL`, and `moge.model.v2.MoGeModel`; load `Ruicheng/moge-2-vitl`; derive:

```python
fx_normalized = intrinsics[0, 0]
fov_x = 2 * math.atan(1.0 / (2.0 * fx_normalized))
distance = 1.0 / (2.0 * mesh_scale * math.tan(fov_x / 2.0))
```

Emit the same JSON as manual mode. Print FOV in radians/degrees and distance to stderr for diagnostics.

- [ ] **Step 4: Run the no-model Python test**

Expected: PASS without downloading MoGe weights.

---

### Task 5: Integrate server without double-running the estimator

**Files:**
- Modify: `src/trellis-server.cpp`
- Modify: `include/transforms_json.h`
- Modify: `src/transforms_json.cpp`

**Interfaces:**
- New writer:

```cpp
bool write_transforms_json(const std::string& path,
                           const TransformsFile& tf,
                           std::string& error);
```

- [ ] **Step 1: Add server request parsing**

Accept form fields:

```text
estimate_camera=1|true|on
camera_fov=<radians>
```

Never accept the estimator executable path from a request. It comes only from server launch params/environment.

- [ ] **Step 2: Estimate during staging/validation**

When no transforms part exists and estimation/manual FOV is requested, resolve the `TransformsFile` before loading model weights and write it to the private staging directory. Then clear the per-request estimation flag before `trellis_run` so the existing runtime only parses the staged JSON.

- [ ] **Step 3: Preserve HTTP error contract**

Estimator failures return HTTP 400 with the actionable reason. Internal process/staging failures return 500. Do not expose private staging paths.

- [ ] **Step 4: Build and run weightless tests**

Expected: all Linux CPU targets build; camera and canonical-rig tests pass.

---

### Task 6: Wire source-checkout generation helper and document the feature

**Files:**
- Modify: `install/generate.sh`
- Modify: `README.md`
- Modify: `docs/spec/30-pixal3d-cond.md`
- Modify: `.github/workflows/linux-build.yml`

**Interfaces:**
- `generate.sh --estimate-camera --mesh-scale F --views DIR`
- `generate.sh --camera-fov RAD --mesh-scale F --views DIR`

- [ ] **Step 1: Pass new flags in local mode**

When `--estimate-camera` is selected and no explicit estimator is given, source checkouts pass:

```text
--camera-estimator <repo>/tools/estimate_transforms_moge.py
```

if that file exists; otherwise fail with instructions to set `PIXAL3D_CAMERA_ESTIMATOR`.

- [ ] **Step 2: Document dependency boundary**

README must state that automatic estimation requires Python + MoGe-2 dependencies and is not part of the standalone runtime bundle yet. Manual `--camera-fov` has no such dependency.

- [ ] **Step 3: Add CI execution**

Linux CI runs:

```bash
./build/trellis-test-camera-estimator
python3 tools/test_estimate_transforms_moge.py
```

The Python test uses manual mode and must not download model weights.

- [ ] **Step 4: Final verification**

Required green checks:

```text
Linux C++ build
  - Build everything configured
  - Run the weightless canonical-rig test
  - Run the camera-estimator test
  - Run the MoGe helper no-model test
```

Then open/update the PR against `main` and link `#4`, explicitly noting that bundled arbitrary multi-view pose estimation remains a follow-up while the external estimator contract already supports a replacement backend.