#!/usr/bin/env bash
# Pixal3D Web の静的配信物を web/dist/ へ組み立てる。
# アプリ本体だけを平坦に置き、モデル(数 GiB)は外部ホストから取得する。
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/web/dist"
# MV の既定モデル配信元。第1引数で上書きできる。
mv_base_url="${1:-https://huggingface.co/raven38/pixal3d-q8_0-v1/resolve/main}"
# SV は verified manifest が公開されるまで空でよい。環境変数または第2/3引数で注入する。
sv_manifest_url="${PIXAL3D_SV_MODEL_MANIFEST_URL:-${2:-}}"
sv_base_url="${PIXAL3D_SV_MODEL_BASE_URL:-${3:-}}"
allow_missing_wasm="${PIXAL3D_WEB_ALLOW_MISSING_WASM:-0}"

rm -rf "$out"
mkdir -p "$out/real_e2e" "$out/vendor"

cp "$root"/web/app/{index.html,main.js,model_store.js,release_store.js,preflight.js,sha256.js,single_view.js} "$out/"
cp "$root"/web/real_e2e/{calibration.js,worker.js} "$out/real_e2e/"
wasm_js="$root/web/real_e2e/pixal3d_real_geometry.js"
wasm_bin="$root/web/real_e2e/pixal3d_real_geometry.wasm"
if [[ -f "$wasm_js" && -f "$wasm_bin" ]]; then
  cp "$wasm_js" "$wasm_bin" "$out/real_e2e/"
elif [[ "$allow_missing_wasm" == "1" ]]; then
  echo "warning: generated WebGPU WASM artifacts absent; building UI-only smoke bundle" >&2
else
  echo "missing generated WebGPU runtime: $wasm_js / $wasm_bin" >&2
  echo "build pixal3d-webgpu-real-geometry-wasm before production deploy" >&2
  exit 1
fi
cp "$root/app/public/vendor/model-viewer.min.js" "$out/vendor/"

# MV manifest は既存 release identity として同梱する。SV manifest は size/SHA 検証済みの
# public set が別途公開されるまで捏造しない。
mkdir -p "$out/models/pixal3d-q8_0-v1"
cp "$root/models/pixal3d-q8_0-v1/pixal3d-models.json" "$out/models/pixal3d-q8_0-v1/"

# main.js の外部参照を、平坦化した配置に合わせて書き換える。
python3 - "$out/main.js" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8').read()
subs = [("'../real_e2e/calibration.js'", "'./real_e2e/calibration.js'"),
        ("'../../app/public/vendor/model-viewer.min.js'", "'./vendor/model-viewer.min.js'"),
        ("'../real_e2e/worker.js'", "'./real_e2e/worker.js'")]
for old, new in subs:
    if s.count(old) != 1:
        raise SystemExit(f"想定と異なる参照数: {old} -> {s.count(old)}")
    s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
PY

# deployment model sources を注入する。SV 値が空なら Web UI は local/custom verified
# model-set install のみを許可し、MV へフォールバックしない。
python3 - "$out/index.html" "$mv_base_url" "$sv_manifest_url" "$sv_base_url" <<'PY'
import io, json, sys
p, mv_base, sv_manifest, sv_base = sys.argv[1:5]
s = io.open(p, encoding='utf-8').read()
tag = '<script type="module" src="main.js"></script>'
if s.count(tag) != 1:
    raise SystemExit("main.js の script タグが見つからない")

def lit(v):
    return json.dumps(v).replace('<', '\\u003c')
lines = [f'window.PIXAL3D_MODEL_BASE_URL = {lit(mv_base)};']
if sv_manifest:
    lines.append(f'window.PIXAL3D_SV_MODEL_MANIFEST_URL = {lit(sv_manifest)};')
if sv_base:
    lines.append(f'window.PIXAL3D_SV_MODEL_BASE_URL = {lit(sv_base)};')
inject = '<script>' + ''.join(lines) + '</script>\n' + tag
io.open(p, 'w', encoding='utf-8').write(s.replace(tag, inject))
PY

printf '%s\n' "built: $out" "MV model base URL: $mv_base_url" "SV manifest URL: ${sv_manifest_url:-<not configured>}" "SV model base URL: ${sv_base_url:-<not configured>}"
du -sh "$out"; find "$out" -type f | sed "s|$out/||" | sort
