#!/usr/bin/env bash
# Trellis Studio — one-command setup for Linux (x86-64) and macOS (Apple Silicon).
#
#   curl -fsSL https://raw.githubusercontent.com/raven38/pixal3d.cpp/main/install/install.sh | bash
#
# On macOS the runtime is Metal (no backend choice) and the app ships as a .dmg.
# Detects the GPU runtime (CUDA / ROCm / Vulkan / Metal), downloads the matching
# trellis-server bundle + the TRELLIS.2 weights (~16.5 GB), installs the
# Trellis Studio desktop app, and writes the config the app reads on launch.
set -euo pipefail

# Release identity. `releases/latest` is never used: it silently skips
# prereleases, which is how a Desktop alpha tag would resolve to an older stable
# release's assets. The tag is always resolved to a concrete value first, then
# every expected asset is checked before anything is downloaded.
REPO="${TRELLIS_REPO:-raven38/pixal3d.cpp}"
TAG="latest"          # a tag, or "latest" (newest stable) / "latest-prerelease"
# Verification escape hatch: install straight from a directory of release assets
# served over HTTP, skipping GitHub release resolution. This is how the macOS
# path is exercised before a release exists; it is not a distribution mechanism.
ASSET_BASE_URL=""
GH_API="https://api.github.com"
REL_BASE=""           # set by resolve_release()
HF_BASE="https://huggingface.co/ilintar/trellis2-gguf/resolve/main"
MODELS=(birefnet.gguf dinov3.gguf ss_flow.gguf ss_dec.gguf \
        shape_flow_512.gguf shape_flow_1024.gguf shape_dec.gguf \
        tex_flow_512.gguf tex_flow_1024.gguf tex_dec.gguf)
# ROCm gfx arches covered by the published rocm bundle (see .github/workflows/release.yml).
ROCM_GFX="gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1200 gfx1201"

# ---- platform --------------------------------------------------------------
# The app resolves its config with dirs::config_dir() (app/src-tauri/src/config.rs),
# which is ~/Library/Application Support on macOS and $XDG_CONFIG_HOME on Linux —
# so the installer must write config.json to the matching place per platform.
OS="$(uname -s)"
case "$OS" in
  Linux)  PLATFORM=linux;  ASSET_OS="linux-x64";       APP_ASSET="trellis-studio-linux-x86_64.AppImage";;
  Darwin) PLATFORM=macos;  ASSET_OS="macos-arm64";     APP_ASSET="trellis-studio-macos-arm64.dmg";;
  *) echo "unsupported platform: $OS (Linux and macOS only)" >&2; exit 1;;
esac
if [ "$PLATFORM" = macos ] && [ "$(uname -m)" != "arm64" ]; then
  echo "error: only Apple Silicon (arm64) is supported on macOS; this is $(uname -m)" >&2
  exit 1
fi

# ---- defaults / args -------------------------------------------------------
if [ "$PLATFORM" = macos ]; then
  DEST="$HOME/Library/Application Support/trellis-studio"
  CONFIG_DIR="$HOME/Library/Application Support/trellis-studio"
else
  DEST="${XDG_DATA_HOME:-$HOME/.local/share}/trellis-studio"
  CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/trellis-studio"
fi
BACKEND=""; GPU=0; PORT=8080; MODELS_DIR=""; SKIP_MODELS=0; SKIP_APP=0; ASSUME_YES=0; QUANT=""
# Pixal3D model set (#34): a manifest fixes the exact bytes of every model file.
MODEL_MANIFEST=""     # path, URL, or "release" (an asset of the resolved release)
MODEL_BASE_URL=""     # where the model files themselves live
VERIFY_ONLY=0         # verify an existing models dir and exit

usage() {
  cat <<EOF
Trellis Studio installer (Linux)

  --repo OWNER/NAME            GitHub repo to install from (default $REPO)
  --asset-base-url URL         install assets from this base URL instead of a
                               GitHub release (pre-release verification)
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
  --model-manifest PATH|URL|release
                               install the Pixal3D model set described by this
                               manifest (pixal3d-models.json) and verify every
                               file's exact size and SHA256. "release" takes the
                               manifest from the resolved GitHub release.
  --model-base-url URL         where the model files listed in the manifest live
                               (required with --model-manifest unless every file
                               is already present and only needs verifying)
  --verify-models              verify the existing models dir against its
                               manifest and exit; no downloads, no config writes
  --skip-models                don't download the weights
  --skip-app                   don't download the desktop app
  -y, --yes                    don't prompt for confirmation
  -h, --help                   this help
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    --asset-base-url) ASSET_BASE_URL="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --backend) BACKEND="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --models-dir) MODELS_DIR="$2"; shift 2;;
    --quant) QUANT="$2"; shift 2;;
    --model-manifest) MODEL_MANIFEST="$2"; shift 2;;
    --model-base-url) MODEL_BASE_URL="$2"; shift 2;;
    --verify-models) VERIFY_ONLY=1; shift;;
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

if [ "$PLATFORM" = macos ]; then
  # Metal is enabled unconditionally by CMake on Apple Silicon (CMakeLists.txt:277);
  # there is no other backend to pick, so an explicit --backend is rejected rather
  # than silently ignored.
  if [ -n "$BACKEND" ] && [ "$BACKEND" != metal ]; then
    die "--backend $BACKEND is not available on macOS (Metal only)"
  fi
  BACKEND=metal
  log "backend: ${c_b}metal${c_0} (macOS)"
elif [ -z "$BACKEND" ]; then
  BACKEND="$(detect_backend)"
  log "auto-detected backend: ${c_b}${BACKEND}${c_0}"
else
  log "backend (forced): ${c_b}${BACKEND}${c_0}"
fi
case "$BACKEND" in cuda|cuda12|rocm|vulkan|metal) ;; *) die "invalid backend: $BACKEND";; esac

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
  if [ -n "$ASSET_BASE_URL" ]; then
    REL_BASE="${ASSET_BASE_URL%/}"
    RESOLVED_TAG="asset-base-url"
    RELEASE_ID=0
    # 資産一覧は問い合わせ先が無いので空にする。存在確認は download 時の 404 で
    # 落ちる（require_assets を通ったふりはしない）。
    ASSET_NAMES=""
    log "assets: ${c_b}${REL_BASE}${c_0} (pre-release verification mode)"
    return
  fi
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
  if [ -n "$ASSET_BASE_URL" ]; then
    warn "asset presence is not pre-checked in --asset-base-url mode; a missing asset fails at download"
    return
  fi
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
  if [ -n "$GH_TOKEN_VALUE" ] && [ -z "$ASSET_BASE_URL" ]; then
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


# ---- Pixal3D model set (manifest-verified) ---------------------------------
# The manifest is the model-set identity (#34): exact byte size and SHA256 for
# each of the required roles. Nothing here trusts a filename or a length alone.

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else die "need sha256sum or shasum to verify the model set"; fi
}

# Manifest entries are read as one line per file so a malformed manifest cannot
# silently produce fewer files than it lists.
manifest_entries() {
  tr -d '\n' <"$1" | tr '{' '\n' | grep -F '"sha256"' | while IFS= read -r obj; do
    local name size digest
    name="$(json_str "$obj" name)"; size="$(json_num "$obj" size_bytes)"; digest="$(json_str "$obj" sha256)"
    [ -n "$name" ] && [ -n "$size" ] && [ -n "$digest" ] || continue
    printf '%s\t%s\t%s\n' "$name" "$size" "$digest"
  done
}

check_manifest() {
  local mf="$1" sv ms mv count
  sv="$(json_num "$(cat "$mf")" schema_version)"
  [ "$sv" = "1" ] || die "unsupported model manifest schema_version: ${sv:-missing}"
  ms="$(json_str "$(cat "$mf")" model_set)"; mv="$(json_str "$(cat "$mf")" version)"
  [ -n "$ms" ] && [ -n "$mv" ] || die "model manifest is missing model_set/version"
  count="$(manifest_entries "$mf" | grep -c . || true)"
  [ "${count:-0}" -gt 0 ] || die "model manifest lists no files"
  MODEL_SET_NAME="$ms"; MODEL_SET_VERSION="$mv"; MODEL_FILE_COUNT="$count"
  info "model set: ${ms} ${mv} (${count} files, manifest sha256 $(sha256_of "$mf"))"
}

# Verifies one file against the manifest. Returns 0 = matches, 1 = differs.
verify_one() {
  local path="$1" size="$2" digest="$3" actual_size
  [ -f "$path" ] || return 1
  actual_size="$(wc -c <"$path" | tr -d ' ')"
  [ "$actual_size" = "$size" ] || return 1
  [ "$(sha256_of "$path")" = "$digest" ] || return 1
}

install_model_set() {
  local mf="$1" ok=0 fetched=0 failed=""
  check_manifest "$mf"
  mkdir -p "$MODELS_DIR"
  while IFS="$(printf '\t')" read -r name size digest; do
    # A manifest comes from the network, so its names must never escape the
    # models directory.
    case "$name" in ""|.|..|*/*|*\\*) die "unsafe file name in model manifest: '$name'";; esac
    local dest="$MODELS_DIR/$name"
    if verify_one "$dest" "$size" "$digest"; then
      info "ok $name (already present, verified)"
      ok=$((ok+1)); continue
    fi
    if [ "$VERIFY_ONLY" = 1 ]; then failed="${failed} $name"; continue; fi
    [ -n "$MODEL_BASE_URL" ] || die "$name is missing or does not match the manifest, and --model-base-url was not given"
    rm -f "$dest.part"
    info "↓ $name ($size bytes)"
    curl -fL --retry 3 --retry-delay 2 --progress-bar -o "$dest.part" "${MODEL_BASE_URL%/}/$name" \
      || { rm -f "$dest.part"; die "download failed: ${MODEL_BASE_URL%/}/$name"; }
    mv -f "$dest.part" "$dest"
    if verify_one "$dest" "$size" "$digest"; then
      ok=$((ok+1)); fetched=$((fetched+1))
    else
      # Keep the bad file out of the models dir: a wrong-but-plausible model
      # file is worse than a missing one.
      local got_size got_sha
      got_size="$(wc -c <"$dest" | tr -d ' ')"; got_sha="$(sha256_of "$dest")"
      rm -f "$dest"
      die "model file does not match the manifest: $name
       expected size $size sha256 $digest
       got      size $got_size sha256 $got_sha
       the downloaded file was removed."
    fi
  done < <(manifest_entries "$mf")

  if [ -n "$failed" ]; then
    die "model set ${MODEL_SET_NAME} ${MODEL_SET_VERSION} does not verify:${failed}
       ($ok of $MODEL_FILE_COUNT files match the manifest)"
  fi
  [ "$ok" = "$MODEL_FILE_COUNT" ] || die "only $ok of $MODEL_FILE_COUNT model files verified"

  # The manifest lands in the models dir only after every file verified, so its
  # presence means "this directory is a complete, verified model set" — which is
  # what the Studio model cache and the Web store both read it as.
  if [ "$VERIFY_ONLY" != 1 ]; then
    cp "$mf" "$MODELS_DIR/pixal3d-models.json"
  fi
  MODEL_MANIFEST_SHA="$(sha256_of "$MODELS_DIR/pixal3d-models.json" 2>/dev/null || sha256_of "$mf")"
  log "model set verified: ${c_b}${MODEL_SET_NAME} ${MODEL_SET_VERSION}${c_0} ($ok files, $fetched newly downloaded)"
}

# Resolves --model-manifest to a local file.
fetch_manifest() {
  local src="$1" out="$2"
  case "$src" in
    release)
      resolve_release
      require_assets pixal3d-models.json
      download_asset pixal3d-models.json "$out"
      ;;
    http://*|https://*) download "$src" "$out";;
    *) [ -f "$src" ] || die "model manifest not found: $src"; cp "$src" "$out";;
  esac
}

# ---- 0. verify-only --------------------------------------------------------
# Verify an existing models dir and stop. No release resolution, no downloads,
# no config writes — this is the gate a clean-install E2E (#18) can run before
# and after an install to prove the model set on disk is the released one.
if [ "$VERIFY_ONLY" = 1 ]; then
  MF="${MODEL_MANIFEST:-$MODELS_DIR/pixal3d-models.json}"
  case "$MF" in
    release|http://*|https://*)
      MF_TMP="$(mktemp)"; fetch_manifest "$MF" "$MF_TMP"; MF="$MF_TMP";;
    *) [ -f "$MF" ] || die "no manifest to verify against: $MF";;
  esac
  install_model_set "$MF"
  log "${c_g}model set OK${c_0} — ${MODEL_SET_NAME} ${MODEL_SET_VERSION} in $MODELS_DIR"
  exit 0
fi

# ---- 1. server runtime bundle ---------------------------------------------
resolve_release
BUNDLE="trellis-${BACKEND}-${ASSET_OS}.tar.gz"
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

if [ "$BACKEND" = "rocm" ] && [ "$PLATFORM" = linux ]; then
  warn "ROCm bundle needs a TheRock ROCm runtime on LD_LIBRARY_PATH."
  warn "See docs/getting-started.md; if the server fails to start, install ROCm 7.x"
  warn "(TheRock, gfx-matched) or re-run with --backend vulkan."
fi

# ---- 2. weights ------------------------------------------------------------
if [ -n "$MODEL_MANIFEST" ]; then
  log "installing the Pixal3D model set from the manifest -> $MODELS_DIR"
  MF_TMP="$(mktemp)"
  fetch_manifest "$MODEL_MANIFEST" "$MF_TMP"
  install_model_set "$MF_TMP"
  rm -f "$MF_TMP"
elif [ "$SKIP_MODELS" = 1 ]; then
  warn "skipping model download (--skip-models); set them in the app's Settings."
else
  log "downloading TRELLIS.2 weights [$WEIGHTS_LABEL, resumable] -> $MODELS_DIR"
  mkdir -p "$MODELS_DIR"
  # ${QUANT:+$QUANT/} -> "q8/" or "q4/" when set, empty for f16.
  for m in "${MODELS[@]}"; do download "${HF_BASE}/${QUANT:+$QUANT/}${m}" "$MODELS_DIR/${m}"; done
fi

# The manifest (#34) is the model-set identity. "verified" is only true when this
# run actually checked every file's size and SHA256 against it.
MODEL_SET_JSON="null"
if [ -n "${MODEL_SET_NAME:-}" ]; then
  MODEL_SET_JSON="{\"model_set\": \"$MODEL_SET_NAME\", \"version\": \"$MODEL_SET_VERSION\", \"manifest_sha256\": \"${MODEL_MANIFEST_SHA:-}\", \"verified\": true}"
elif [ -f "$MODELS_DIR/pixal3d-models.json" ]; then
  MS_NAME="$(json_str "$(cat "$MODELS_DIR/pixal3d-models.json")" model_set)"
  MS_VER="$(json_str "$(cat "$MODELS_DIR/pixal3d-models.json")" version)"
  MODEL_SET_JSON="{\"model_set\": \"$MS_NAME\", \"version\": \"$MS_VER\", \"manifest_sha256\": \"$(sha256_of "$MODELS_DIR/pixal3d-models.json")\", \"verified\": false}"
fi

# ---- 3. desktop app --------------------------------------------------------
if [ "$SKIP_APP" = 1 ]; then
  warn "skipping desktop app download (--skip-app)."
else
  log "downloading Trellis Studio desktop app"
  # require_assets already proved this asset exists in the resolved release, so a
  # failure here is a real error and must not degrade into a partial install.
  if [ "$PLATFORM" = macos ]; then
    DMG="$(mktemp -d)/$APP_ASSET"
    download_asset "$APP_ASSET" "$DMG"
    MP="$(mktemp -d)"
    hdiutil attach -nobrowse -quiet -mountpoint "$MP" "$DMG" || die "could not mount $APP_ASSET"
    SRC_APP="$(find "$MP" -maxdepth 1 -name '*.app' -print -quit)"
    if [ -z "$SRC_APP" ]; then hdiutil detach -quiet "$MP" || true; die "no .app inside $APP_ASSET"; fi
    APP="/Applications/$(basename "$SRC_APP")"
    rm -rf "$APP"
    cp -R "$SRC_APP" /Applications/ || { hdiutil detach -quiet "$MP" || true; die "could not copy the app to /Applications"; }
    hdiutil detach -quiet "$MP" || true
    rm -f "$DMG"
    # The release is not notarized, so first launch is quarantined. Clearing the
    # attribute here is the difference between "it works" and a Gatekeeper dialog
    # that offers no way forward.
    xattr -dr com.apple.quarantine "$APP" 2>/dev/null || \
      warn "could not clear the quarantine attribute; open the app once via right-click > Open"
    info "app installed: $APP"
  else
    APP="$DEST/Trellis Studio.AppImage"
    download_asset "$APP_ASSET" "$APP"
    chmod +x "$APP"
    info "app installed: $APP"
  fi
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
# `cmd && info ...` as the last statement would make a successful install exit 1
# whenever the AppImage is absent (e.g. --skip-app), which now matters because
# callers gate on the exit code.
if [ "$PLATFORM" = macos ] && [ -d "/Applications/Trellis Studio.app" ]; then
  info "run: open \"/Applications/Trellis Studio.app\""
elif [ -f "$DEST/Trellis Studio.AppImage" ]; then
  info "run: \"$DEST/Trellis Studio.AppImage\""
fi
exit 0
