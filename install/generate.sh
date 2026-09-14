#!/usr/bin/env bash
# 1 コマンドで「取得 → 生成 → GLB」まで通す（uv の install.sh と同じ curl | sh 型）。
#
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/install/generate.sh \
#     | bash -s -- --views ./views -o out.glb
#
# 2 つのモード:
#   local  (既定) : install.sh で runtime + 重みを $DEST に用意（2 回目以降はキャッシュ）し、
#                   trellis-cli --views を実行する。GPU が要る。
#   remote         : --server URL を渡すと、常駐 trellis-server の POST /generate に
#                   multipart で投げて GLB を受け取る。ローカルに GPU も重みも要らない。
#
# 入力は Pixal3D の multiview ディレクトリ（transforms.json + RGBA view PNG）。
set -euo pipefail

REPO="${TRELLIS_REPO:-raven38/pixal3d.cpp}"
TAG="${TRELLIS_TAG:-latest-prerelease}"
MODEL_BASE_URL="${PIXAL3D_MODEL_BASE_URL:-https://huggingface.co/raven38/pixal3d-q8_0-v1/resolve/main}"
VIEWS=""; OUT="pixal3d.glb"; SEED=1; NUM_VIEWS=""; SERVER=""; RES=""; EXTRA=()
DEST="${TRELLIS_DEST:-${XDG_DATA_HOME:-$HOME/.local/share}/trellis-studio}"

usage() {
  cat <<USAGE
generate.sh — fetch runtime + weights once, then generate a GLB

  --views DIR        multiview input dir (transforms.json + view PNGs)   [required]
  -o, --output PATH  output .glb (default $OUT)
  --seed N           RNG seed (default $SEED)
  --num-views N      use only the first N frames
  --res 1024|1536    geometry resolution (local mode only)
  --server URL       remote mode: POST to a running trellis-server instead of running locally
  --repo OWNER/NAME  release repo for the runtime (default $REPO)
  --tag TAG          release tag (default $TAG)
  --model-base-url U where the manifest's GGUF files live (default: Hugging Face q8_0 v1)
  --dest DIR         local install dir (default $DEST)
  -- ARGS...         passed through to trellis-cli (local mode)
USAGE
}
while [ $# -gt 0 ]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2;;
    -o|--output) OUT="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --num-views) NUM_VIEWS="$2"; shift 2;;
    --res) RES="$2"; shift 2;;
    --server) SERVER="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --model-base-url) MODEL_BASE_URL="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage; exit 1;;
  esac
done
[ -n "$VIEWS" ] || { echo "--views DIR is required" >&2; usage; exit 1; }
[ -f "$VIEWS/transforms.json" ] || { echo "$VIEWS/transforms.json not found" >&2; exit 1; }
info() { echo "[generate] $*" >&2; }

# ---- remote: 常駐サーバへ投げるだけ -------------------------------------------
if [ -n "$SERVER" ]; then
  # transforms.json の frame file_path と同じ相対名で各 view を multipart に載せる
  # （trellis-server はファイル名を transforms.json 相対で staging する）。
  # POST /generate-mv: part 名は view0, view1, …、filename は transforms.json の file_path。
  args=(-F "transforms=@$VIEWS/transforms.json"); i=0
  while IFS= read -r f; do
    args+=(-F "view$i=@$VIEWS/$f;filename=$f"); i=$((i+1))
  done < <(grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' "$VIEWS/transforms.json" | sed 's/.*"\([^"]*\)"$/\1/')
  [ -n "$NUM_VIEWS" ] && args+=(-F "num_views=$NUM_VIEWS")
  [ -n "$RES" ] && args+=(-F "resolution=$RES")
  args+=(-F "seed=$SEED")
  info "POST $SERVER/generate-mv ($i views) -> $OUT"
  code=$(curl -sS -X POST "${SERVER%/}/generate-mv" "${args[@]}" -o "$OUT" -w '%{http_code}')
  [ "$code" = "200" ] || { echo "server returned HTTP $code: $(head -c 300 "$OUT")" >&2; exit 1; }
  info "wrote $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
  exit 0
fi

# ---- local: install.sh で runtime + 重みを用意（既にあれば manifest で全ファイル検証）→ trellis-cli ----
CLI="$DEST/runtime/trellis-cli"
MODELS="$DEST/models"
installer="$(dirname "${BASH_SOURCE[0]:-$0}")/install.sh"
if [ ! -f "$installer" ]; then
  installer="$(mktemp)"; curl -fsSL "https://raw.githubusercontent.com/$REPO/main/install/install.sh" -o "$installer"
fi
# キャッシュ有効の判定は「CLI が実行可能」かつ「models/ の全ファイルが manifest のサイズ・SHA256 と一致」。
# manifest だけ残った部分キャッシュや壊れた GGUF は、ここで弾いて installer に補完させる。
need_install=0
if [ ! -x "$CLI" ] || [ ! -f "$MODELS/pixal3d-models.json" ]; then
  need_install=1
elif ! bash "$installer" --dest "$DEST" --verify-models >/dev/null 2>&1; then
  info "existing weights under $MODELS do not match their manifest — re-running the installer"
  need_install=1
fi
if [ "$need_install" = 1 ]; then
  info "runtime or weights missing/invalid under $DEST — running the installer"
  bash "$installer" --repo "$REPO" --tag "$TAG" --dest "$DEST" --skip-app -y \
    --model-manifest release --model-base-url "$MODEL_BASE_URL"
fi
[ -x "$CLI" ] || { echo "trellis-cli not found at $CLI (the release archive must ship it)" >&2; exit 1; }

cmd=("$CLI" --views "$VIEWS" --models "$MODELS" --seed "$SEED" --require-gpu -o "$OUT")
[ -n "$NUM_VIEWS" ] && cmd+=(--num-views "$NUM_VIEWS")
[ -n "$RES" ] && cmd+=(--res "$RES")
info "${cmd[*]} ${EXTRA[*]:-}"
"${cmd[@]}" "${EXTRA[@]}"
info "wrote $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
