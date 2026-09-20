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
BACKEND=""; GPU=""; PORT=8080; MODELS_DIR=""; SKIP_MODELS=0; SKIP_APP=0; ASSUME_YES=0; QUANT=""
# Pixal3D model set (#34): a manifest fixes the exact bytes of every model file.
MODEL_MANIFEST=""     # path, URL, or "release" (an asset of the resolved release)
MODEL_BASE_URL=""     # where the model files themselves live
VERIFY_ONLY=0         # verify an existing models dir and exit
# 0.10.0: the single-view (SV) model set is a second, independent directory.
# Five of its nine file names are shared with the MV set but hold different
# bytes, so the two sets can never be merged into one directory.
MODELS_DIR_SV=""      # default <dest>/models-sv
MODEL_MANIFEST_SV=""  # path, URL, or "release" (asset pixal3d-models-sv.json)
MODEL_BASE_URL_SV=""

usage() {
  cat <<EOF
Trellis Studio installer (Linux)

  --repo OWNER/NAME            GitHub repo to install from (default $REPO)
  --asset-base-url URL         install assets from this base URL instead of a
                               GitHub release (pre-release verification)
  --tag TAG                    release tag to install; "latest" = newest stable,
                               "latest-prerelease" = newest release including
                               prereleases (default: latest). Prerelease tags
                               such as v0.10.0-desktop-alpha must be named
                               explicitly or via latest-prerelease.
  --backend cuda|cuda12|rocm|vulkan
                              force a runtime (default: auto-detect; cuda12 is
                              for NVIDIA Pascal/Volta GPUs such as the P100)
  --gpu N                      GPU index (default: auto — the server prefers a discrete GPU; <0 = CPU)
  --port P                     server port (default 8080)
  --dest DIR                   install location (default $DEST)
  --config-dir DIR             where config.json / release.json are written
                               (default $CONFIG_DIR; the app reads that one)
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
  --models-dir-sv DIR          where to put the single-view (SV) model set
                               (default <dest>/models-sv). Never the MV dir:
                               five file names collide with different bytes.
  --model-manifest-sv PATH|URL|release
                               also install the Pixal3D single-view model set
                               (family sv, e.g. pixal3d-sv-q8_0 v1) into
                               --models-dir-sv, verified the same way. "release"
                               takes pixal3d-models-sv.json from the release.
  --model-base-url-sv URL      where the SV model files live
  --verify-models              verify the existing models dir(s) against their
                               manifest(s) and exit; no downloads, no config
                               writes. Checks the MV dir and, when it holds a
                               manifest or --model-manifest-sv is given, the SV
                               dir too; both must pass.
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
    --config-dir) CONFIG_DIR="$2"; shift 2;;
    --models-dir) MODELS_DIR="$2"; shift 2;;
    --quant) QUANT="$2"; shift 2;;
    --model-manifest) MODEL_MANIFEST="$2"; shift 2;;
    --model-base-url) MODEL_BASE_URL="$2"; shift 2;;
    --models-dir-sv) MODELS_DIR_SV="$2"; shift 2;;
    --model-manifest-sv) MODEL_MANIFEST_SV="$2"; shift 2;;
    --model-base-url-sv) MODEL_BASE_URL_SV="$2"; shift 2;;
    --verify-models) VERIFY_ONLY=1; shift;;
    --skip-models) SKIP_MODELS=1; shift;;
    --skip-app) SKIP_APP=1; shift;;
    -y|--yes) ASSUME_YES=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage; exit 1;;
  esac
done
MODELS_DIR="${MODELS_DIR:-$DEST/models}"
MODELS_DIR_SV="${MODELS_DIR_SV:-$DEST/models-sv}"
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
# The `cuda` bundle is built with CUDA 13.1, whose corresponding driver branch is R590
# (NVIDIA CUDA Toolkit Release Notes, Table 2 "CUDA Toolkit and Corresponding Driver
# Branch": 13.0 -> R580, 13.1 -> R590). Minor-version compatibility lets 13.x binaries
# run on >= 580 drivers for SASS, but PTX JIT-compiled by an older driver is not
# covered. On WSL2 the user-mode CUDA libraries come from the Windows driver package
# (/usr/lib/wsl/lib) and can lag the Windows kernel-mode driver: with 580.178.04
# user-mode libraries trellis-server segfaults in libnvidia-ptxjitcompiler at the first
# kernel (#35). Below this major version the CUDA 12.9 `cuda12` bundle is used instead.
CUDA13_MIN_DRIVER_MAJOR=590

is_wsl() { grep -qi microsoft /proc/version 2>/dev/null || uname -r 2>/dev/null | grep -qi microsoft; }

# First line of `nvidia-smi`: "NVIDIA-SMI 580.178.04  Driver Version: 591.86  CUDA Version: 13.1".
# The NVIDIA-SMI field is the user-mode library version; on WSL2 it can differ from
# "Driver Version" (the Windows kernel-mode driver). Prints the major number or nothing.
usermode_driver_major() {
  nvidia-smi 2>/dev/null | sed -n 's/.*NVIDIA-SMI[[:space:]]*\([0-9][0-9]*\)\..*/\1/p' | head -1
}

# On WSL2 with a user-mode driver older than the CUDA 13.1 runtime needs, print the
# offending major version; otherwise print nothing (stdout is the value, like detect_backend).
wsl_cuda13_driver_too_old() {
  [ "$PLATFORM" = linux ] && is_wsl || return 0
  local drv; drv="$(usermode_driver_major)"
  if [ -n "$drv" ] && [ "$drv" -lt "$CUDA13_MIN_DRIVER_MAJOR" ]; then echo "$drv"; fi
}

detect_backend() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -qi gpu; then
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${GPU:-0}" 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
    if [ -n "$cc" ] && awk -v cc="$cc" 'BEGIN { exit !(cc >= 6.0 && cc < 7.5) }'; then
      info "detected NVIDIA compute capability $cc — selecting the CUDA 12 legacy runtime"
      echo cuda12; return
    fi
    local old; old="$(wsl_cuda13_driver_too_old)"
    if [ -n "$old" ]; then
      warn "WSL2: the user-mode CUDA driver is $old.x (nvidia-smi) but the CUDA 13.1 runtime corresponds to the R$CUDA13_MIN_DRIVER_MAJOR driver branch — selecting the CUDA 12 runtime (cuda12). Update the Windows NVIDIA driver to use the cuda runtime."
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
if [ "$BACKEND" = cuda ]; then
  OLD_DRV="$(wsl_cuda13_driver_too_old)"
  [ -z "$OLD_DRV" ] || die "WSL2: the user-mode CUDA driver is $OLD_DRV.x (nvidia-smi) but the CUDA 13.1 runtime (--backend cuda) corresponds to the R$CUDA13_MIN_DRIVER_MAJOR driver branch; trellis-server would segfault in libnvidia-ptxjitcompiler (#35). Use --backend cuda12, or update the Windows NVIDIA driver so /usr/lib/wsl/lib catches up."
fi

echo
info "install dir : $DEST"
info "models dir  : $MODELS_DIR $([ "$SKIP_MODELS" = 1 ] && echo '(skipped)')"
[ -z "$MODEL_MANIFEST_SV" ] || info "SV models   : $MODELS_DIR_SV"
info "weights     : $WEIGHTS_LABEL"
info "backend/gpu : $BACKEND / ${GPU:-auto}     port: $PORT"
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
         --repo OWNER/NAME --tag v0.10.0-desktop-alpha"
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
# silently produce fewer files than it lists. Columns: name, size, sha256, role,
# required ("true" unless the entry says otherwise).
manifest_entries() {
  tr -d '\n' <"$1" | tr '{' '\n' | grep -F '"sha256"' | while IFS= read -r obj; do
    local name size digest role req
    name="$(json_str "$obj" name)"; size="$(json_num "$obj" size_bytes)"; digest="$(json_str "$obj" sha256)"
    role="$(json_str "$obj" role)"
    req="true"; grep -q '"required"[[:space:]]*:[[:space:]]*false' <<<"$obj" && req="false"
    [ -n "$name" ] && [ -n "$size" ] && [ -n "$digest" ] || continue
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$size" "$digest" "$role" "$req"
  done
}

# name <-> role contract of a model family (mirror of web/app/model_family.js and
# src/model_manifest.cpp; the conformance vectors in web/app/manifest_conformance.json
# are run against this script by install/test_installer_manifest.sh).
MODEL_ROLES="image_encoder naf ss_flow ss_decoder shape_flow_512 shape_decoder shape_flow_1024 texture_flow_1024 texture_decoder"
FLOW_ROLES="ss_flow shape_flow_512 shape_flow_1024 texture_flow_1024"
expected_name_for_role() {  # role family
  case "$1" in
    image_encoder)     echo dinov3.gguf;;
    naf)               echo pixal3d_naf.gguf;;
    ss_flow)           echo "pixal3d_ss_flow_$2.gguf";;
    ss_decoder)        echo ss_dec.gguf;;
    shape_flow_512)    echo "pixal3d_shape_flow_512_$2.gguf";;
    shape_decoder)     echo shape_dec.gguf;;
    shape_flow_1024)   echo "pixal3d_shape_flow_1024_$2.gguf";;
    texture_flow_1024) echo "pixal3d_tex_flow_1024_$2.gguf";;
    texture_decoder)   echo tex_dec.gguf;;
    *) echo "";;
  esac
}
expected_role_for_name() {  # name family
  local r
  for r in $MODEL_ROLES; do
    [ "$(expected_name_for_role "$r" "$2")" = "$1" ] && { echo "$r"; return; }
  done
  echo ""
}
# Family inferred from the four flow-role file names: mv / sv / "" (mixed or none).
infer_family() {  # manifest
  local names="" n fam
  # (awk rather than a case statement: bash 3.2 cannot parse `case` inside `$( )`.)
  names="$(manifest_entries "$1" | awk -F'\t' -v fr=" $FLOW_ROLES " 'index(fr, " " $4 " ") { print $1 }')"
  [ -n "$names" ] || { echo ""; return; }
  for fam in mv sv; do
    local all=1
    while IFS= read -r n; do case "$n" in *"_${fam}.gguf") ;; *) all=0;; esac; done <<<"$names"
    [ "$all" = 1 ] && { echo "$fam"; return; }
  done
  echo ""
}

# Validates a manifest and pins its family. Sets MODEL_SET_NAME / MODEL_SET_VERSION /
# MODEL_FILE_COUNT / MODEL_FAMILY. Every rejection here happens before any file is
# touched or downloaded.
check_manifest() {
  local mf="$1" expected_family="$2" sv ms mv count explicit inferred family
  sv="$(json_num "$(cat "$mf")" schema_version)"
  [ "$sv" = "1" ] || die "unsupported model manifest schema_version: ${sv:-missing}"
  ms="$(json_str "$(cat "$mf")" model_set)"; mv="$(json_str "$(cat "$mf")" version)"
  [ -n "$ms" ] && [ -n "$mv" ] || die "model manifest is missing model_set/version"
  count="$(manifest_entries "$mf" | grep -c . || true)"
  [ "${count:-0}" -gt 0 ] || die "model manifest lists no files"

  explicit="$(json_str "$(cat "$mf")" model_family)"
  case "$explicit" in ""|mv|sv) ;; *) die "model_family must be one of: mv, sv (got '$explicit')";; esac
  inferred="$(infer_family "$mf")"
  if [ -n "$explicit" ] && [ -n "$inferred" ] && [ "$explicit" != "$inferred" ]; then
    die "model_family $explicit does not match the flow file names ($inferred)"
  fi
  family="${explicit:-${inferred:-mv}}"
  [ "$family" = "$expected_family" ] || die "manifest $ms $mv is a $family model set; this option expects the $expected_family set
       (use --model-manifest for the multi-view set and --model-manifest-sv for the single-view set)"

  # Known names carry a fixed role and are required; a known role must carry its
  # known name; no duplicate names or roles; the family's nine files must all be listed.
  local seen_names="" seen_roles="" present="" r
  while IFS="$(printf '\t')" read -r name size digest role req; do
    case " $seen_names " in *" $name "*) die "duplicate file name: $name";; esac
    seen_names="$seen_names $name"
    if [ -n "$role" ]; then
      case " $seen_roles " in *" $role "*) die "duplicate role: $role";; esac
      seen_roles="$seen_roles $role"
    fi
    local want_role want_name
    want_role="$(expected_role_for_name "$name" "$family")"
    if [ -n "$want_role" ]; then
      [ "$role" = "$want_role" ] || die "$name must have role $want_role, not '${role:-missing}'"
      [ "$req" = "true" ] || die "$name must be required"
      present="$present $name"
    else
      want_name="$(expected_name_for_role "$role" "$family")"
      [ -z "$want_name" ] || die "role $role must be $want_name, not $name"
    fi
  done < <(manifest_entries "$mf")
  local missing=""
  for r in $MODEL_ROLES; do
    local n; n="$(expected_name_for_role "$r" "$family")"
    case " $present " in *" $n "*) ;; *) missing="$missing $n";; esac
  done
  [ -z "$missing" ] || die "manifest $ms $mv is missing required $family files:$missing"

  MODEL_SET_NAME="$ms"; MODEL_SET_VERSION="$mv"; MODEL_FILE_COUNT="$count"; MODEL_FAMILY="$family"
  info "model set: ${ms} ${mv} (family ${family}, ${count} files, manifest sha256 $(sha256_of "$mf"))"
}

# Verifies one file against the manifest. Returns 0 = matches, 1 = differs.
verify_one() {
  local path="$1" size="$2" digest="$3" actual_size
  [ -f "$path" ] || return 1
  actual_size="$(wc -c <"$path" | tr -d ' ')"
  [ "$actual_size" = "$size" ] || return 1
  [ "$(sha256_of "$path")" = "$digest" ] || return 1
}

free_bytes_in() {  # dir (must exist)
  df -Pk "$1" | awk 'NR==2{print $4 * 1024}'
}

# Installs (or, with --verify-models, only verifies) one model set.
#   install_model_set MANIFEST DIR BASE_URL EXPECTED_FAMILY
# Results: MODEL_SET_NAME / MODEL_SET_VERSION / MODEL_FAMILY / MODEL_MANIFEST_SHA.
# Order: validate the manifest, verify what is on disk, check free space for what
# is missing, then download. Nothing is written before all of those pass, and a
# failing SV install therefore leaves an already verified MV directory untouched.
install_model_set() {
  local mf="$1" dir="$2" base_url="$3" expected_family="$4"
  local ok=0 fetched=0 failed="" need_bytes=0 missing_list=""
  check_manifest "$mf" "$expected_family"
  mkdir -p "$dir"
  while IFS="$(printf '\t')" read -r name size digest role req; do
    # A manifest comes from the network, so its names must never escape the
    # models directory.
    case "$name" in ""|.|..|*/*|*\\*) die "unsafe file name in model manifest: '$name'";; esac
    if verify_one "$dir/$name" "$size" "$digest"; then
      info "ok $name (already present, verified)"
      ok=$((ok+1)); continue
    fi
    failed="${failed} $name"
    missing_list="${missing_list}${name}"$'\t'"${size}"$'\t'"${digest}"$'\n'
    need_bytes=$((need_bytes + size))
  done < <(manifest_entries "$mf")

  if [ "$VERIFY_ONLY" = 1 ] && [ -n "$failed" ]; then
    die "model set ${MODEL_SET_NAME} ${MODEL_SET_VERSION} does not verify:${failed}
       ($ok of $MODEL_FILE_COUNT files match the manifest)"
  fi
  if [ -n "$failed" ]; then
    [ -n "$base_url" ] || die "these files are missing or do not match the manifest, and no base URL was given:${failed}"
    # Free-space precheck (design D7): bytes still to download + 10 %, checked
    # before the first byte is fetched.
    local free need_with_margin
    free="$(free_bytes_in "$dir")"
    need_with_margin=$((need_bytes + need_bytes / 10))
    if [ -n "$free" ] && [ "$free" -lt "$need_with_margin" ]; then
      die "not enough free space in $dir for ${MODEL_SET_NAME} ${MODEL_SET_VERSION}:
       need $((need_with_margin / 1048576)) MiB ($((need_bytes / 1048576)) MiB + 10 %), have $((free / 1048576)) MiB"
    fi
    while IFS="$(printf '\t')" read -r name size digest; do
      [ -n "$name" ] || continue
      local dest="$dir/$name"
      rm -f "$dest.part"
      info "↓ $name ($size bytes)"
      curl -fL --retry 3 --retry-delay 2 --progress-bar -o "$dest.part" "${base_url%/}/$name" \
        || { rm -f "$dest.part"; die "download failed: ${base_url%/}/$name"; }
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
    done <<<"$missing_list"
  fi
  [ "$ok" = "$MODEL_FILE_COUNT" ] || die "only $ok of $MODEL_FILE_COUNT model files verified"

  # The manifest lands in the models dir only after every file verified, so its
  # presence means "this directory is a complete, verified model set" — which is
  # what the Studio model cache, trellis-server /capabilities and the Web store
  # all read it as.
  if [ "$VERIFY_ONLY" != 1 ]; then
    cp "$mf" "$dir/pixal3d-models.json"
  fi
  MODEL_MANIFEST_SHA="$(sha256_of "$dir/pixal3d-models.json" 2>/dev/null || sha256_of "$mf")"
  log "model set verified: ${c_b}${MODEL_SET_NAME} ${MODEL_SET_VERSION}${c_0} (family ${MODEL_FAMILY}, $ok files, $fetched newly downloaded) in $dir"
}

# Resolves --model-manifest / --model-manifest-sv to a local file.
#   fetch_manifest SRC OUT [RELEASE_ASSET]
fetch_manifest() {
  local src="$1" out="$2" asset="${3:-pixal3d-models.json}"
  case "$src" in
    release)
      [ -n "$RESOLVED_TAG" ] || resolve_release
      require_assets "$asset"
      download_asset "$asset" "$out"
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
  # The MV dir is always checked. The SV dir is checked when it already holds a
  # manifest or one is named explicitly — and then both must pass.
  verify_dir() {  # manifest_arg dir family release_asset
    local mf="$1"
    case "$mf" in
      release|http://*|https://*)
        MF_TMP="$(mktemp)"; fetch_manifest "$mf" "$MF_TMP" "$4"; mf="$MF_TMP";;
      *) [ -f "$mf" ] || die "no manifest to verify against: $mf";;
    esac
    install_model_set "$mf" "$2" "" "$3"
    log "${c_g}model set OK${c_0} — ${MODEL_SET_NAME} ${MODEL_SET_VERSION} (${MODEL_FAMILY}) in $2"
  }
  verify_dir "${MODEL_MANIFEST:-$MODELS_DIR/pixal3d-models.json}" "$MODELS_DIR" mv pixal3d-models.json
  if [ -n "$MODEL_MANIFEST_SV" ] || [ -f "$MODELS_DIR_SV/pixal3d-models.json" ]; then
    verify_dir "${MODEL_MANIFEST_SV:-$MODELS_DIR_SV/pixal3d-models.json}" "$MODELS_DIR_SV" sv pixal3d-models-sv.json
  else
    info "no single-view model set at $MODELS_DIR_SV (nothing to verify)"
  fi
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
MV_SET_NAME=""; MV_SET_VERSION=""; MV_MANIFEST_SHA=""
if [ -n "$MODEL_MANIFEST" ]; then
  log "installing the Pixal3D multi-view model set from the manifest -> $MODELS_DIR"
  MF_TMP="$(mktemp)"
  fetch_manifest "$MODEL_MANIFEST" "$MF_TMP" pixal3d-models.json
  install_model_set "$MF_TMP" "$MODELS_DIR" "$MODEL_BASE_URL" mv
  rm -f "$MF_TMP"
  MV_SET_NAME="$MODEL_SET_NAME"; MV_SET_VERSION="$MODEL_SET_VERSION"; MV_MANIFEST_SHA="$MODEL_MANIFEST_SHA"
elif [ "$SKIP_MODELS" = 1 ]; then
  warn "skipping model download (--skip-models); set them in the app's Settings."
else
  log "downloading TRELLIS.2 weights [$WEIGHTS_LABEL, resumable] -> $MODELS_DIR"
  mkdir -p "$MODELS_DIR"
  # ${QUANT:+$QUANT/} -> "q8/" or "q4/" when set, empty for f16.
  for m in "${MODELS[@]}"; do download "${HF_BASE}/${QUANT:+$QUANT/}${m}" "$MODELS_DIR/${m}"; done
fi

# 0.10.0: optional single-view set, always in its own directory. It is installed
# after the MV set so a failure here cannot leave the MV directory half-written
# (the script dies before the config is written, and MV files are never touched).
SV_SET_NAME=""; SV_SET_VERSION=""; SV_MANIFEST_SHA=""
if [ -n "$MODEL_MANIFEST_SV" ]; then
  case "$MODELS_DIR_SV" in "$MODELS_DIR") die "--models-dir-sv must differ from the MV models dir ($MODELS_DIR): five file names collide with different contents";; esac
  log "installing the Pixal3D single-view model set from the manifest -> $MODELS_DIR_SV"
  MF_TMP="$(mktemp)"
  fetch_manifest "$MODEL_MANIFEST_SV" "$MF_TMP" pixal3d-models-sv.json
  install_model_set "$MF_TMP" "$MODELS_DIR_SV" "$MODEL_BASE_URL_SV" sv
  rm -f "$MF_TMP"
  SV_SET_NAME="$MODEL_SET_NAME"; SV_SET_VERSION="$MODEL_SET_VERSION"; SV_MANIFEST_SHA="$MODEL_MANIFEST_SHA"
fi
# The config only names an SV directory that holds a verified set (installed now
# or by an earlier run); otherwise it stays empty and trellis-server gets no
# --models-sv, which Studio reports as "single-view model set is not installed".
CONFIG_MODELS_DIR_SV=""
if [ -n "$SV_SET_NAME" ] || [ -f "$MODELS_DIR_SV/pixal3d-models.json" ]; then CONFIG_MODELS_DIR_SV="$MODELS_DIR_SV"; fi

# The manifest (#34) is the model-set identity. "verified" is only true when this
# run actually checked every file's size and SHA256 against it.
model_set_json() {  # name version sha dir
  if [ -n "$1" ]; then
    echo "{\"model_set\": \"$1\", \"version\": \"$2\", \"manifest_sha256\": \"$3\", \"verified\": true}"
  elif [ -f "$4/pixal3d-models.json" ]; then
    echo "{\"model_set\": \"$(json_str "$(cat "$4/pixal3d-models.json")" model_set)\", \"version\": \"$(json_str "$(cat "$4/pixal3d-models.json")" version)\", \"manifest_sha256\": \"$(sha256_of "$4/pixal3d-models.json")\", \"verified\": false}"
  else
    echo null
  fi
}
MODEL_SET_JSON="$(model_set_json "$MV_SET_NAME" "$MV_SET_VERSION" "$MV_MANIFEST_SHA" "$MODELS_DIR")"
MODEL_SET_SV_JSON="$(model_set_json "$SV_SET_NAME" "$SV_SET_VERSION" "$SV_MANIFEST_SHA" "$MODELS_DIR_SV")"

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
  "modelsDirSv": "$CONFIG_MODELS_DIR_SV",
  "backend": "$BACKEND",
  "gpu": ${GPU:-null},
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
  "models_dir_sv": "$CONFIG_MODELS_DIR_SV",
  "model_set_sv": ${MODEL_SET_SV_JSON:-null},
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
info "release receipt: $CONFIG_DIR/release.json"

echo
log "${c_g}done${c_0} — launch Trellis Studio$([ "$SKIP_MODELS" = 1 ] && echo ' (add your models dir in Settings)')."
# `cmd && info ...` as the last statement would make a successful install exit 1
# whenever the AppImage is absent (e.g. --skip-app), which now matters because
# callers gate on the exit code.
if [ "$PLATFORM" = macos ] && [ -d "/Applications/Trellis Studio.app" ]; then
  info "run: open \"/Applications/Trellis Studio.app\""
elif [ -f "$DEST/Trellis Studio.AppImage" ]; then
  info "run: \"$DEST/Trellis Studio.AppImage\""
fi
exit 0
