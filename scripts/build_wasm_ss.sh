#!/usr/bin/env bash
# Configure + build the browser Pixal3D SS path (docs/spec/31-webgpu-bringup.md §9).
# Produces web/ss/pixal3d_ss.{js,wasm}. Same toolchain notes as scripts/build_wasm_smoke.sh
# (emcc 6.0.9, built-in emdawnwebgpu port).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build-wasm-ss"

cd "${REPO_ROOT}"

emcmake cmake -S . -B "${BUILD_DIR}" \
  -DGGML_WEBGPU=ON \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRELLIS_WASM_SS=ON

cmake --build "${BUILD_DIR}" --target pixal3d-webgpu-ss-wasm -j

echo
echo "Built:"
ls -la "${REPO_ROOT}/web/ss/pixal3d_ss.js" "${REPO_ROOT}/web/ss/pixal3d_ss.wasm"
