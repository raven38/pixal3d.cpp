#!/usr/bin/env bash
# Weightless contract test for Pixal3D SV/MV selection and camera FOV validation.
# Usage: install/test_pixal3d_weight_selection.sh path/to/trellis-cli
set -euo pipefail

CLI="${1:?usage: $0 path/to/trellis-cli}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="$ROOT/docker/linux-webgpu-gate/e2e/views/view00_azim000.png"

if [ ! -x "$CLI" ]; then
  echo "FAIL: trellis-cli not executable: $CLI" >&2
  exit 1
fi
if [ ! -f "$FIXTURE" ]; then
  echo "FAIL: missing RGBA fixture: $FIXTURE" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
views="$tmp/views"
models="$tmp/models"
mkdir -p "$views" "$models"
cp "$FIXTURE" "$views/front.png"

write_tf() {
  local fov="$1"
  cat > "$views/transforms.json" <<EOF
{
  "camera_angle_x": $fov,
  "mesh_scale": 1.0,
  "frames": [
    {
      "file_path": "front.png",
      "transform_matrix": [
        [1, 0, 0, 0],
        [0, 0, -1, -2.8356409],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
      ]
    }
  ]
}
EOF
}

run_fail() {
  local log="$1"; shift
  set +e
  "$CLI" "$@" >"$log" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    echo "FAIL: command unexpectedly succeeded: $*" >&2
    cat "$log" >&2
    exit 1
  fi
}

# Default remains MV. With an empty model dir, the fail-fast diagnostic must name
# the MV flow family and must not silently look for the SV family.
write_tf 0.3490658503988659
run_fail "$tmp/default.log" --views "$views" --models "$models" --res 1024 "$tmp/default.glb"
grep -q 'pixal3d_ss_flow_mv\.gguf' "$tmp/default.log"
if grep -q 'pixal3d_ss_flow_sv\.gguf' "$tmp/default.log"; then
  echo "FAIL: default MV path mentioned SV weights" >&2
  cat "$tmp/default.log" >&2
  exit 1
fi

# Explicit SV selection must fail on the SV family only; never substitute MV.
run_fail "$tmp/sv.log" --views "$views" --models "$models" --pixal3d-weights sv --res 1024 "$tmp/sv.glb"
grep -q 'pixal3d_ss_flow_sv\.gguf' "$tmp/sv.log"
if grep -q 'pixal3d_ss_flow_mv\.gguf' "$tmp/sv.log"; then
  echo "FAIL: explicit SV path mentioned MV weights" >&2
  cat "$tmp/sv.log" >&2
  exit 1
fi

# Perspective FOV is physically valid only for 0 < fov < pi. This must fail at
# camera validation, before model preflight, even though the model directory is empty.
write_tf 3.141592653589793
run_fail "$tmp/fov.log" --views "$views" --models "$models" --pixal3d-weights sv --res 1024 "$tmp/fov.glb"
grep -q 'camera_angle_x' "$tmp/fov.log"
grep -qi 'pi' "$tmp/fov.log"
if grep -q 'missing .*flow model' "$tmp/fov.log"; then
  echo "FAIL: invalid FOV reached model preflight" >&2
  cat "$tmp/fov.log" >&2
  exit 1
fi

echo "ok   default selects MV only"
echo "ok   explicit SV selects SV only"
echo "ok   camera FOV >= pi fails before model loading"
echo "PASSED"
