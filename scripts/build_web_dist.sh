#!/usr/bin/env bash
# Pixal3D Web の静的配信物を web/dist/ へ組み立てる。
# アプリ本体だけを平坦に置き、モデル(7.54 GiB)は外部ホストから取得する。
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/web/dist"
# 既定のモデル配信元。第1引数で上書きできる。
base_url="${1:-https://huggingface.co/raven38/pixal3d-q8_0-v1/resolve/main}"

rm -rf "$out"
mkdir -p "$out/real_e2e" "$out/vendor"

cp "$root"/web/app/{index.html,main.js,model_store.js,release_store.js,preflight.js,sha256.js} "$out/"
cp "$root"/web/real_e2e/{calibration.js,worker.js,pixal3d_real_geometry.js,pixal3d_real_geometry.wasm} "$out/real_e2e/"
cp "$root/app/public/vendor/model-viewer.min.js" "$out/vendor/"

# manifest はリリース識別子であり、サイト相対パスで参照されるので同梱する。
mkdir -p "$out/models/pixal3d-q8_0-v1"
cp "$root/models/pixal3d-q8_0-v1/pixal3d-models.json" "$out/models/pixal3d-q8_0-v1/"

# main.js の3つの外部参照を、平坦化した配置に合わせて書き換える。
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

# デプロイ既定のモデル配信元を注入する（README の解決優先順位に従う）。
python3 - "$out/index.html" "$base_url" <<'PY'
import io, sys
p, base = sys.argv[1], sys.argv[2]
s = io.open(p, encoding='utf-8').read()
tag = '<script type="module" src="main.js"></script>'
if s.count(tag) != 1:
    raise SystemExit("main.js の script タグが見つからない")
inject = f'<script>window.PIXAL3D_MODEL_BASE_URL = {base!r};</script>\n' + tag
io.open(p, 'w', encoding='utf-8').write(s.replace(tag, inject))
PY

printf '%s\n' "built: $out" "model base URL: $base_url"
du -sh "$out"; find "$out" -type f | sed "s|$out/||" | sort
