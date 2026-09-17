#!/usr/bin/env bash
# 本番 Web 配信物が使う WebGPU/WASM ランタイム（web/real_e2e/pixal3d_real_geometry.{js,wasm}）を
# クリーンなチェックアウトから決定的に組み立てる。CI（.github/workflows/web-wasm-runtime.yml）と
# 手元で同じ手順を使い、出所（source commit / emcc 版 / 両ファイルの SHA256）を
# web/real_e2e/build-info.json に書き出す。build_web_dist.sh はこの JSON を dist に同梱する。
#
#   scripts/build_wasm_real_geometry.sh [--fresh] [--build-dir DIR]
#
# 前提: emsdk が有効化済み（emcmake/emcc が PATH にある）。emdawnwebgpu は emcc 4.0.10 以降の
# 組み込み port（--use-port=emdawnwebgpu）なので追加取得は不要。ピン留め版は
# .github/workflows/web-wasm-runtime.yml の EMSDK_VERSION が正。
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo}/build-wasm-ss"
fresh=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) fresh=1; shift ;;
    --build-dir) build_dir="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v emcmake >/dev/null || { echo "emcmake not found: activate emsdk first" >&2; exit 1; }
[[ -f "${repo}/thirdparty/ggml/CMakeLists.txt" ]] || { echo "thirdparty/ggml is empty: git submodule update --init --recursive" >&2; exit 1; }

if [[ "$fresh" == 1 ]]; then rm -rf "$build_dir"; fi
cd "$repo"
# docs/PIXAL3D_E2E_STATUS.md §6 と同じ構成。patches/ggml-webgpu/*.patch は configure 時に適用される。
emcmake cmake -S . -B "$build_dir" \
  -DGGML_WEBGPU=ON -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DTRELLIS_WASM_SS=ON
cmake --build "$build_dir" --target pixal3d-webgpu-real-geometry-wasm -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

out_js="${repo}/web/real_e2e/pixal3d_real_geometry.js"
out_wasm="${repo}/web/real_e2e/pixal3d_real_geometry.wasm"
[[ -f "$out_js" && -f "$out_wasm" ]] || { echo "expected outputs missing: $out_js / $out_wasm" >&2; exit 1; }

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
commit="$(git -C "$repo" rev-parse HEAD)"
# thirdparty/ggml は configure 時の patch 適用で常に dirty になるので別枠で記録する。
dirty="$(git -C "$repo" status --porcelain --untracked-files=no -- src include web/ss patches CMakeLists.txt | grep -c . || true)"
emcc_version="$(emcc --version | head -1)"
ggml_commit="$(git -C "$repo/thirdparty/ggml" rev-parse HEAD)"
ggml_patches="$(cd "$repo/patches/ggml-webgpu" && ls *.patch | tr '\n' ' ')"
python3 - "$out_js" "$out_wasm" "$commit" "$dirty" "$emcc_version" "$ggml_commit" "$ggml_patches" "$(sha "$out_js")" "$(sha "$out_wasm")" <<'PY'
import json, os, sys
js, wasm, commit, dirty, emcc, ggml, patches, sha_js, sha_wasm = sys.argv[1:10]
info = {
    "target": "pixal3d-webgpu-real-geometry-wasm",
    "source_commit": commit,
    "source_dirty_files": int(dirty),
    "ggml_commit": ggml,
    "ggml_patches": patches.split(),
    "emcc": emcc,
    "files": {
        os.path.basename(js): {"size_bytes": os.path.getsize(js), "sha256": sha_js},
        os.path.basename(wasm): {"size_bytes": os.path.getsize(wasm), "sha256": sha_wasm},
    },
}
out = os.path.join(os.path.dirname(js), "build-info.json")
with open(out, "w") as f:
    json.dump(info, f, indent=2); f.write("\n")
print(json.dumps(info, indent=2))
PY
