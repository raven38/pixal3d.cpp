#!/usr/bin/env bash
# Configure + build the browser WebGPU smoke test (docs/spec/31-webgpu-bringup.md §8).
# Produces web/smoke/pixal3d_smoke.{js,wasm}. Requires an Emscripten toolchain (emcmake/em++)
# on PATH -- tested with emcc 6.0.9 (Homebrew), which auto-downloads the built-in
# `emdawnwebgpu` Emscripten port (v20260423.175430 as of that emcc version) on first use.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build-wasm"

cd "${REPO_ROOT}"

emcmake cmake -S . -B "${BUILD_DIR}" \
  -DGGML_WEBGPU=ON \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRELLIS_WASM_SMOKE=ON

cmake --build "${BUILD_DIR}" --target pixal3d-webgpu-smoke-wasm -j

echo
echo "Built:"
ls -la "${REPO_ROOT}/web/smoke/pixal3d_smoke.js" "${REPO_ROOT}/web/smoke/pixal3d_smoke.wasm"
