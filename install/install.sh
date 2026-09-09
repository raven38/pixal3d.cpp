#!/usr/bin/env bash
# Trellis Studio — one-command setup for Linux (x86-64).
#
#   curl -fsSL https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.sh | bash
#
# Detects the GPU runtime (CUDA / ROCm / Vulkan), downloads the matching
# trellis-server bundle + the TRELLIS.2 weights (~16.5 GB), installs the
# Trellis Studio desktop app, and writes the config the app reads on launch.
set -euo pipefail

# Release identity. `releases/latest` is never used: it silently skips
# prereleases, which is how a Desktop alpha tag would resolve to an older stable
# release's assets. The tag is always resolved to a concrete value first, then
# every expected asset is checked before anything is downloaded.
REPO="${TRELLIS_REPO:-pwilkin/trellis.cpp}"
TAG="latest"          # a tag, or "latest" (newest stable) / "latest-prerelease"
GH_API="https://api.github.com"
REL_BASE=""           # set by resolve_release()
HF_BASE="https://huggingface.co/ilintar/trellis2-gguf/resolve/main"
MODELS=(birefnet.gguf dinov3.gguf ss_flow.gguf ss_dec.gguf \
        shape_flow_512.gguf shape_flow_1024.gguf shape_dec.gguf \
        tex_flow_512.gguf tex_flow_1024.gguf tex_dec.gguf)
# ROCm gfx arches covered by the published rocm bundle (see .github/workflows/release.yml).
ROCM_GFX="gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1200 gfx1201"

# ---- defaults / args -------------------------------------------------------
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/trellis-studio"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/trellis-studio"
BACKEND=""; GPU=0; PORT=8080; MODELS_DIR=""; SKIP_MODELS=0; SKIP_APP=0; ASSUME_YES=0; QUANT=""

usage() {
  cat <<EOF
Trellis Studio installer (Linux)

  --repo OWNER/NAME            GitHub repo to install from (default $REPO)
  --tag TAG                    release tag to install; "latest" = newest stable,
                               "latest-prerelease" = newest release including
                               prereleases (default: latest). Prerelease tags
                               such as v0.9.0-desktop-alpha must be named
                               explicitly or via latest-prerelease.
  --backend cuda|cuda12|rocm|vulkan
                              force a runtime (default: auto-detect; cuda12 is
                              for NVIDIA Pascal/Volta GPUs such as the P100)
  --gpu N                      GPU index (default 0; <0 = CPU)
  --port P                     server port (default 8080)
  --dest DIR                   install location (default $DEST)
  --models-dir DIR             where to put weights (default <dest>/models)
  --quant q8|q4                download quantized weights instead of f16:
                                 q8 ~9.5 GB (near-lossless), q4 ~6 GB (smaller,
                                 slight quality loss). Default: f16 (~16.5 GB).
  --skip-models                don't download the weights
  --skip-app                   don't download the desktop app
  -y, --yes                    don't prompt for confirmation
  -h, --help                   this help
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --backend) BACKEND="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --models-dir) MODELS_DIR="$2"; shift 2;;
    --quant) QUANT="$2"; shift 2;;
    --skip-models) SKIP_MODELS=1; shift;;
    --skip-app) SKIP_APP=1; shift;;
    -y|--yes) ASSUME_YES=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage; exit 1;;
  esac
done
MODELS_DIR="${MODELS_DIR:-$DEST/models}"
RUNTIME_DIR="$DEST/runtime"
# Quantized weights live in q8/ and q4/ subpaths of the HF repo, same filenames.
case "$QUANT" in
  "")   WEIGHTS_LABEL="f16 (~16.5 GB)";;
  q8)   WEIGHTS_LABEL="Q8 (~9.5 GB, near-lossless)";;
  q4)   WEIGHTS_LABEL="Q4 (~6 GB, slight quality loss)";;
  *)    die "invalid --quant: $QUANT (use q8 or q4)";;
esac

# ---- logging ---------------------------------------------------------------
# Status messages go to stderr so command substitution (e.g. $(detect_backend))
# only ever captures the intended stdout value.
c_g=$'\e[32m'; c_y=$'\e[33m'; c_r=$'\e[31m'; c_b=$'\e[36m'; c_0=$'\e[0m'
log()  { echo "${c_g}==>${c_0} $*" >&2; }
info() { echo "${c_b} - ${c_0} $*" >&2; }
warn() { echo "${c_y}warn:${c_0} $*" >&2; }
die()  { echo "${c_r}error:${c_0} $*" >&2; exit 1; }

for t in curl tar; do command -v "$t" >/dev/null || die "'$t' is required"; done

# ---- backend detection -----------------------------------------------------
detect_backend() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -qi gpu; then
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
    if [ -n "$cc" ] && awk -v cc="$cc" 'BEGIN { exit !(cc >= 6.0 && cc < 7.5) }'; then
      info "detected NVIDIA compute capability $cc — selecting the CUDA 12 legacy runtime"
      echo cuda12; return
    fi
    echo cuda; return
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    local gfx; gfx="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1 || true)"
    if [ -n "$gfx" ] && echo "$ROCM_GFX" | grep -qw "$gfx"; then
      # ROCm is available, but the published rocm bundle needs a matching TheRock
      # runtime; Vulkan is self-contained and often faster on these iGPUs, so it's
      # the safe auto default. Pass --backend rocm to force ROCm.
      info "detected AMD $gfx — ROCm-capable (use --backend rocm to force it)"
    fi
  fi
  echo vulkan
}

if [ -z "$BACKEND" ]; then
  BACKEND="$(detect_backend)"
  log "auto-detected backend: ${c_b}${BACKEND}${c_0}"
else
  log "backend (forced): ${c_b}${BACKEND}${c_0}"
fi
case "$BACKEND" in cuda|cuda12|rocm|vulkan) ;; *) die "invalid backend: $BACKEND";; esac

echo
info "install dir : $DEST"
info "models dir  : $MODELS_DIR $([ "$SKIP_MODELS" = 1 ] && echo '(skipped)')"
info "weights     : $WEIGHTS_LABEL"
info "backend/gpu : $BACKEND / $GPU     port: $PORT"
echo
if [ "$ASSUME_YES" != 1 ] && [ -t 0 ]; then
  read -r -p "Proceed? [Y/n] " ans; case "${ans:-y}" in [nN]*) exit 0;; esac
fi

# ---- download helper (resumable) ------------------------------------------
download() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  info "↓ $(basename "$dest")"
  curl -fL --retry 3 --retry-delay 2 --progress-bar -C - -o "$dest" "$url" \
    || die "download failed: $url"
}


# ---- release resolution ----------------------------------------------------
# A token is optional for public repos and required for private ones.
GH_TOKEN_VALUE="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

gh_api() {
  local path="$1" auth=()
  [ -n "$GH_TOKEN_VALUE" ] && auth=(-H "Authorization: Bearer $GH_TOKEN_VALUE")
  curl -fsSL --retry 3 --retry-delay 2 "${auth[@]}" \
    -H "Accept: application/vnd.github+json" "${GH_API}${path}"
}

# Reads one field out of a JSON object.
#
# Two traps this avoids:
#   * `| head -1` would SIGPIPE the producing grep under `set -o pipefail` on a
#     large payload, aborting the script with rc 141 and no message;
#   * `grep -o -m1` stops after the first matching *line*, but still prints every
#     match on it — with compact (single-line) JSON that returns all ids at once.
# So: read every match, then take the first line inside the shell.
first_line() { local v="$1"; printf '%s' "${v%%$'\n'*}"; }
json_matches() { grep -o "$2" <<<"$1" || true; }
json_str() {
  local m; m="$(json_matches "$1" "\"$2\"[[:space:]]*:[[:space:]]*\"[^\"]*\"")"
  m="$(first_line "$m")"
  [ -n "$m" ] && sed 's/.*:[[:space:]]*"//; s/"$//' <<<"$m" || true
}
json_num() {
  local m; m="$(json_matches "$1" "\"$2\"[[:space:]]*:[[:space:]]*[0-9]\+")"
  m="$(first_line "$m")"
  [ -n "$m" ] && sed 's/.*:[[:space:]]*//' <<<"$m" || true
}

REQUESTED_TAG="$TAG"     # what the user asked for, kept for the receipt
RESOLVED_TAG=""; RELEASE_ID=""; RELEASE_BRANCH=""; RELEASE_COMMIT=""
ASSETS_JSON=""; ASSET_NAMES=""

# A tag goes into a URL path, so anything that could change which resource is
# addressed (a '#' fragment, a '?' query, '..', a slash) is rejected outright
# rather than encoded — release tags never need those characters.
check_tag_syntax() {
  case "$1" in
    ""|*[!A-Za-z0-9._+-]*)
      die "invalid release tag: '$1'
       tags may contain only letters, digits and . _ + - characters";;
    .|..) die "invalid release tag: '$1'";;
  esac
}

resolve_release() {
  local why release_json=""
  case "$TAG" in
    latest)
      why="newest stable release"
      # This endpoint intentionally excludes prereleases; it is only ever used
      # for the explicit "latest" channel.
      release_json="$(gh_api "/repos/${REPO}/releases/latest" || true)"
      ;;
    latest-prerelease)
      why="newest release including prereleases"
      # /releases is newest-first and, unlike /releases/latest, includes
      # prereleases. Take its first tag, then fetch that one release.
      local newest
      newest="$(json_str "$(gh_api "/repos/${REPO}/releases?per_page=1" || true)" tag_name)"
      [ -n "$newest" ] || die "no releases found for ${REPO} (or none visible to this token)"
      check_tag_syntax "$newest"
      release_json="$(gh_api "/repos/${REPO}/releases/tags/${newest}" || true)"
      ;;
    *)
      why="tag $TAG"
      check_tag_syntax "$TAG"
      release_json="$(gh_api "/repos/${REPO}/releases/tags/${TAG}" || true)"
      ;;
  esac

  if [ -z "$release_json" ]; then
    die "could not resolve a release for ${REPO} (${why}).
       If the repository is private, export GITHUB_TOKEN with 'repo' access.
       If the Desktop alpha is published as a prerelease, pass its tag:
         --repo OWNER/NAME --tag v0.9.0-desktop-alpha"
  fi

  RESOLVED_TAG="$(json_str "$release_json" tag_name)"
  RELEASE_ID="$(json_num "$release_json" id)"
  RELEASE_BRANCH="$(json_str "$release_json" target_commitish)"
  [ -n "$RESOLVED_TAG" ] || die "release payload for ${REPO} (${why}) has no tag_name"
  # The resolved tag also ends up in URLs, so it gets the same check even when
  # it came from the API rather than the command line.
  check_tag_syntax "$RESOLVED_TAG"
  [ -n "$RELEASE_ID" ] || die "release payload for ${REPO} (${why}) has no id"

  # An explicitly named tag must come back unchanged. Anything else means the
  # request addressed a different resource than intended.
  case "$TAG" in
    latest|latest-prerelease) ;;
    *) [ "$RESOLVED_TAG" = "$TAG" ] || die "requested tag '$TAG' but the API returned '$RESOLVED_TAG' — refusing to install";;
  esac

  # target_commitish may be a branch name, so it is not the tag's commit. Resolve
  # the tag to a real SHA for the receipt; a failure here is not fatal (the tag
  # and release id already identify the artifacts) but it is reported.
  local commit_json
  commit_json="$(gh_api "/repos/${REPO}/commits/${RESOLVED_TAG}" 2>/dev/null || true)"
  RELEASE_COMMIT="$(first_line "$(json_matches "$commit_json" '[0-9a-f]\{40\}')")"
  [ -n "$RELEASE_COMMIT" ] || warn "could not resolve ${RESOLVED_TAG} to a commit SHA"

  # Assets come from the dedicated endpoint, which returns only an array of
  # assets — the release object's own "name" (its title) can never leak in.
  ASSETS_JSON="$(gh_api "/repos/${REPO}/releases/${RELEASE_ID}/assets?per_page=100" || true)"
  ASSET_NAMES="$(grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' <<<"$ASSETS_JSON" \
    | sed 's/.*:[[:space:]]*"//; s/"$//' || true)"
  REL_BASE="https://github.com/${REPO}/releases/download/${RESOLVED_TAG}"
  log "release: ${c_b}${REPO}@${RESOLVED_TAG}${c_0} (${why}, id ${RELEASE_ID}, commit ${RELEASE_COMMIT:-unknown})"
}

# -F/-x: exact literal match. A plain `grep -x` would let the dots in an asset
# name match any character and report a missing asset as present.
have_asset() { printf '%s\n' "$ASSET_NAMES" | grep -Fxq -- "$1"; }

require_assets() {
  local missing=""
  for a in "$@"; do have_asset "$a" || missing="${missing} $a"; done
  if [ -n "$missing" ]; then
    die "release ${REPO}@${RESOLVED_TAG} is missing expected asset(s):${missing}
       assets present: $(printf '%s ' $ASSET_NAMES)
       Pick a different --tag, or check the release workflow's asset names."
  fi
}

# Downloads a release asset. For a private repo the public download URL 404s, so
# the API asset endpoint is used whenever a token is available. With a token the
# asset id must be found — falling back to a URL that cannot work would only
# turn a clear error into a confusing 404.
download_asset() {
  local name="$1" dest="$2" id
  if [ -n "$GH_TOKEN_VALUE" ]; then
    local obj
    # One asset object per line, then the object naming this asset. The payload
    # is pretty-printed, so newlines are removed *before* splitting on '{' —
    # splitting alone would leave "name" and "id" on separate lines and the id
    # would be lost.
    obj="$(first_line "$(tr -d '\n' <<<"$ASSETS_JSON" | tr '{' '\n' | grep -F -- "\"$name\"" || true)")"
    id="$(json_num "$obj" id)"
    [ -n "$id" ] || die "could not find the asset id for '$name' in ${REPO}@${RESOLVED_TAG}"
    mkdir -p "$(dirname "$dest")"
    info "↓ $name (api asset $id)"
    # A fresh temp file, never `-C -` onto $dest: resuming into a previous
    # version's file of a different length silently splices the two.
    rm -f "$dest.part"
    curl -fL --retry 3 --retry-delay 2 --progress-bar -o "$dest.part" \
      -H "Authorization: Bearer $GH_TOKEN_VALUE" \
      -H "Accept: application/octet-stream" \
      "${GH_API}/repos/${REPO}/releases/assets/${id}" \
      || { rm -f "$dest.part"; die "download failed: $name (api asset $id)"; }
    mv -f "$dest.part" "$dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  info "↓ $name"
  rm -f "$dest.part"
  curl -fL --retry 3 --retry-delay 2 --progress-bar -o "$dest.part" "${REL_BASE}/${name}" \
    || { rm -f "$dest.part"; die "download failed: ${REL_BASE}/${name}"; }
  mv -f "$dest.part" "$dest"
}

# ---- 1. server runtime bundle ---------------------------------------------
resolve_release
BUNDLE="trellis-${BACKEND}-linux-x64.tar.gz"
APP_ASSET="trellis-studio-linux-x86_64.AppImage"
EXPECTED=("$BUNDLE")
[ "$SKIP_APP" = 1 ] || EXPECTED+=("$APP_ASSET")
require_assets "${EXPECTED[@]}"

log "downloading trellis-server ($BACKEND) runtime"
mkdir -p "$RUNTIME_DIR"
TMP_BUNDLE="$(mktemp)"
download_asset "$BUNDLE" "$TMP_BUNDLE"
tar -C "$RUNTIME_DIR" -xzf "$TMP_BUNDLE"
rm -f "$TMP_BUNDLE"
SERVER_BIN="$RUNTIME_DIR/trellis-server"
[ -f "$SERVER_BIN" ] || die "trellis-server not found after extract"
chmod +x "$SERVER_BIN" || true

if [ "$BACKEND" = "rocm" ]; then
  warn "ROCm bundle needs a TheRock ROCm runtime on LD_LIBRARY_PATH."
  warn "See docs/getting-started.md; if the server fails to start, install ROCm 7.x"
  warn "(TheRock, gfx-matched) or re-run with --backend vulkan."
fi

# ---- 2. weights ------------------------------------------------------------
if [ "$SKIP_MODELS" = 1 ]; then
  warn "skipping model download (--skip-models); set them in the app's Settings."
else
  log "downloading TRELLIS.2 weights [$WEIGHTS_LABEL, resumable] -> $MODELS_DIR"
  mkdir -p "$MODELS_DIR"
  # ${QUANT:+$QUANT/} -> "q8/" or "q4/" when set, empty for f16.
  for m in "${MODELS[@]}"; do download "${HF_BASE}/${QUANT:+$QUANT/}${m}" "$MODELS_DIR/${m}"; done
fi

# The manifest (#34) is the model-set identity; record it in the receipt when the
# models directory already carries one.
MODEL_SET_JSON="null"
if [ -f "$MODELS_DIR/pixal3d-models.json" ]; then
  MODEL_SET_JSON="$(sed -n 's/.*"model_set"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$MODELS_DIR/pixal3d-models.json" | head -1)"
  MS_VER="$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$MODELS_DIR/pixal3d-models.json" | head -1)"
  MODEL_SET_JSON="{\"model_set\": \"$MODEL_SET_JSON\", \"version\": \"$MS_VER\"}"
fi

# ---- 3. desktop app --------------------------------------------------------
if [ "$SKIP_APP" = 1 ]; then
  warn "skipping desktop app download (--skip-app)."
else
  log "downloading Trellis Studio desktop app"
  APP="$DEST/Trellis Studio.AppImage"
  # require_assets already proved this asset exists in the resolved release, so a
  # failure here is a real error and must not degrade into a partial install.
  download_asset "$APP_ASSET" "$APP"
  chmod +x "$APP"
  info "app installed: $APP"
fi

# ---- 4. config -------------------------------------------------------------
log "writing config"
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.json" <<JSON
{
  "serverBin": "$SERVER_BIN",
  "modelsDir": "$MODELS_DIR",
  "backend": "$BACKEND",
  "gpu": $GPU,
  "host": "127.0.0.1",
  "port": $PORT,
  "outputDir": "$DEST/output"
}
JSON
info "config: $CONFIG_DIR/config.json"

# Release receipt: what was actually installed, so a bug report or a release note
# can name the exact runtime tag/commit alongside the model set.
cat > "$CONFIG_DIR/release.json" <<JSON
{
  "repo": "$REPO",
  "tag": "$RESOLVED_TAG",
  "release_id": ${RELEASE_ID:-null},
  "commit": "$RELEASE_COMMIT",
  "target_commitish": "$RELEASE_BRANCH",
  "requested_tag": "$REQUESTED_TAG",
  "runtime_bundle": "$BUNDLE",
  "backend": "$BACKEND",
  "models_dir": "$MODELS_DIR",
  "model_set": ${MODEL_SET_JSON:-null},
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
info "release receipt: $CONFIG_DIR/release.json"

echo
log "${c_g}done${c_0} — launch Trellis Studio${SKIP_MODELS:+ (add your models dir in Settings)}."
[ -f "$DEST/Trellis Studio.AppImage" ] && info "run: \"$DEST/Trellis Studio.AppImage\""
