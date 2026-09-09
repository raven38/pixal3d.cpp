<#
.SYNOPSIS
  Trellis Studio — one-command setup for Windows (x64).

.DESCRIPTION
  Detects the GPU runtime (CUDA / ROCm / Vulkan), downloads the matching
  trellis-server bundle + the TRELLIS.2 weights (~16.5 GB), installs the
  Trellis Studio desktop app, and writes the config the app reads on launch.

.EXAMPLE
  irm https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.ps1 | iex
  # or, with options:
  ./install.ps1 -Backend vulkan -SkipModels
#>
[CmdletBinding()]
param(
  # Validated manually below: an empty default (auto-detect) is not a member of the
  # allowed set, and a [ValidateSet] default mismatch is a fatal bind error under
  # `irm ... | iex` (ValidateSetFailure).
  [string]$Backend = "",
  # Release identity. `releases/latest` is deliberately unused: it skips
  # prereleases, so a Desktop alpha tag would silently resolve to an older
  # stable release's assets.
  [string]$Repo = "",
  # A tag, or "latest" (newest stable) / "latest-prerelease" (newest release
  # including prereleases). Prerelease tags such as v0.9.0-desktop-alpha must be
  # named explicitly or reached via latest-prerelease.
  [string]$Tag = "latest",
  [int]$Gpu = 0,
  [int]$Port = 8080,
  [string]$Dest = "$env:LOCALAPPDATA\trellis-studio",
  [string]$ModelsDir = "",
  # Quantized weights: "q8" (~9.5 GB, near-lossless) or "q4" (~6 GB). Default f16.
  [string]$Quant = "",
  [switch]$SkipModels,
  [switch]$SkipApp,
  [switch]$Yes
)

$ErrorActionPreference = "Stop"
if (-not $Repo) { $Repo = if ($env:TRELLIS_REPO) { $env:TRELLIS_REPO } else { "pwilkin/trellis.cpp" } }
$GhApi = "https://api.github.com"
$RelBase = ""   # set by Resolve-Release
$HfBase = "https://huggingface.co/ilintar/trellis2-gguf/resolve/main"
$Models = @("birefnet.gguf", "dinov3.gguf", "ss_flow.gguf", "ss_dec.gguf",
  "shape_flow_512.gguf", "shape_flow_1024.gguf", "shape_dec.gguf",
  "tex_flow_512.gguf", "tex_flow_1024.gguf", "tex_dec.gguf")

if (-not $ModelsDir) { $ModelsDir = Join-Path $Dest "models" }
$RuntimeDir = Join-Path $Dest "runtime"

# Quantized weights live in q8/ and q4/ subpaths of the HF repo, same filenames.
switch ($Quant) {
  ""   { $WeightsLabel = "f16 (~16.5 GB)" }
  "q8" { $WeightsLabel = "Q8 (~9.5 GB, near-lossless)" }
  "q4" { $WeightsLabel = "Q4 (~6 GB, slight quality loss)" }
  default { Die "invalid -Quant: $Quant (use q8 or q4)" }
}
$QuantPath = if ($Quant) { "$Quant/" } else { "" }
$ConfigDir = Join-Path $env:APPDATA "trellis-studio"

function Log($m)  { Write-Host "==> $m" -ForegroundColor Green }
function Info($m) { Write-Host " -  $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "warn: $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "error: $m" -ForegroundColor Red; exit 1 }

# ---- backend detection -----------------------------------------------------
function Detect-Backend {
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
      if (nvidia-smi -L 2>$null) {
        $capText = nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i $Gpu 2>$null |
          Select-Object -First 1
        $cap = 0.0
        if ([double]::TryParse(
              "$capText".Trim(),
              [System.Globalization.NumberStyles]::Float,
              [System.Globalization.CultureInfo]::InvariantCulture,
              [ref]$cap) -and $cap -ge 6.0 -and $cap -lt 7.5) {
          Info "detected NVIDIA compute capability $capText — selecting the CUDA 12 legacy runtime"
          return "cuda12"
        }
        return "cuda"
      }
    } catch {}
  }
  try {
    $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    if ($gpus | Where-Object { $_.Name -match "AMD|Radeon" }) {
      # ROCm is possible on AMD, but the published bundle needs a matching TheRock
      # runtime; Vulkan is self-contained and robust, so it's the safe auto default.
      Info "detected an AMD GPU — ROCm-capable (use -Backend rocm to force it)"
    }
  } catch {}
  return "vulkan"
}

if (-not $Backend) {
  $Backend = Detect-Backend
  Log "auto-detected backend: $Backend"
} else {
  if ($Backend -notin @("cuda", "cuda12", "rocm", "vulkan")) { Die "invalid backend: $Backend (use cuda|cuda12|rocm|vulkan)" }
  Log "backend (forced): $Backend"
}

Write-Host ""
Info "install dir : $Dest"
Info ("models dir  : {0}{1}" -f $ModelsDir, $(if ($SkipModels) { " (skipped)" } else { "" }))
Info "weights     : $WeightsLabel"
Info "backend/gpu : $Backend / $Gpu     port: $Port"
Write-Host ""
if (-not $Yes) {
  $ans = Read-Host "Proceed? [Y/n]"
  if ($ans -match "^[nN]") { exit 0 }
}

# ---- download helper (resumable via BITS, IWR fallback) --------------------
function Download($url, $dest) {
  New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
  Info "down $(Split-Path $dest -Leaf)"
  try {
    Start-BitsTransfer -Source $url -Destination $dest -DisplayName (Split-Path $dest -Leaf)
  } catch {
    Warn "BITS failed, falling back to Invoke-WebRequest"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
  }
}


# ---- release resolution ----------------------------------------------------
# A token is optional for public repos and required for private ones.
$GhToken = if ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } elseif ($env:GH_TOKEN) { $env:GH_TOKEN } else { "" }

function Gh-Api($path) {
  $headers = @{ Accept = "application/vnd.github+json" }
  if ($GhToken) { $headers["Authorization"] = "Bearer $GhToken" }
  Invoke-RestMethod -Uri "$GhApi$path" -Headers $headers -UseBasicParsing
}

$script:Release = $null
$script:RequestedTag = $Tag
$script:ReleaseCommit = ""

# A tag goes into a URL path, so anything that could change which resource is
# addressed ('#', '?', '/', '..') is rejected rather than encoded — release tags
# never need those characters.
function Check-TagSyntax($t) {
  if (-not $t -or $t -notmatch '^[A-Za-z0-9._+-]+$' -or $t -eq '.' -or $t -eq '..') {
    Die "invalid release tag: '$t' (letters, digits and . _ + - only)"
  }
}

function Resolve-Release {
  $why = ""
  try {
    # -CaseSensitive: PowerShell's switch is case-insensitive by default, so
    # `-Tag LATEST` would take the stable channel and skip the exact-tag check
    # that the Linux script applies.
    switch -CaseSensitive ($Tag) {
      "latest" {
        $why = "newest stable release"
        $script:Release = Gh-Api "/repos/$Repo/releases/latest"
      }
      "latest-prerelease" {
        $why = "newest release including prereleases"
        # /releases is newest-first and includes prereleases, unlike /releases/latest.
        $newest = @(Gh-Api "/repos/$Repo/releases?per_page=1")[0]
        if (-not $newest) { Die "no releases found for $Repo (or none visible to this token)" }
        Check-TagSyntax $newest.tag_name
        $script:Release = Gh-Api "/repos/$Repo/releases/tags/$($newest.tag_name)"
      }
      default {
        $why = "tag $Tag"
        Check-TagSyntax $Tag
        $script:Release = Gh-Api "/repos/$Repo/releases/tags/$Tag"
      }
    }
  } catch {
    Die @"
could not resolve a release for $Repo ($why): $($_.Exception.Message)
       If the repository is private, set GITHUB_TOKEN to a token with 'repo' access.
       If the Desktop alpha is published as a prerelease, name its tag:
         -Repo OWNER/NAME -Tag v0.9.0-desktop-alpha
"@
  }
  if (-not $script:Release.tag_name) { Die "release payload for $Repo ($why) has no tag_name" }
  if (-not $script:Release.id) { Die "release payload for $Repo ($why) has no id" }
  # The resolved tag also ends up in URLs, so it gets the same check even when it
  # came from the API rather than the command line.
  Check-TagSyntax $script:Release.tag_name

  # An explicitly named tag must come back unchanged; anything else means the
  # request addressed a different resource than intended.
  if ($Tag -cne "latest" -and $Tag -cne "latest-prerelease" -and
      -not [string]::Equals($script:Release.tag_name, $Tag, [System.StringComparison]::Ordinal)) {
    Die "requested tag '$Tag' but the API returned '$($script:Release.tag_name)' — refusing to install"
  }

  # target_commitish may be a branch name, so resolve the tag to a real SHA for
  # the receipt. Not fatal if it fails: tag + release id already identify the
  # artifacts.
  try {
    $script:ReleaseCommit = (Gh-Api "/repos/$Repo/commits/$($script:Release.tag_name)").sha
  } catch { Warn "could not resolve $($script:Release.tag_name) to a commit SHA" }

  $script:RelBase = "https://github.com/$Repo/releases/download/$($script:Release.tag_name)"
  Log "release: $Repo@$($script:Release.tag_name) ($why, id $($script:Release.id), commit $(if ($script:ReleaseCommit) { $script:ReleaseCommit } else { 'unknown' }))"
}

function Require-Assets([string[]]$names) {
  # -contains / -eq are case-insensitive in PowerShell; asset names are not.
  $have = @($script:Release.assets | ForEach-Object { $_.name })
  $missing = @($names | Where-Object { $n = $_; -not ($have | Where-Object { [string]::Equals($_, $n, [System.StringComparison]::Ordinal) }) })
  if ($missing.Count -gt 0) {
    Die @"
release $Repo@$($script:Release.tag_name) is missing expected asset(s): $($missing -join ' ')
       assets present: $($have -join ' ')
       Pick a different -Tag, or check the release workflow's asset names.
"@
  }
}

# Downloads a release asset. Private-repo public URLs 404, so the API asset
# endpoint is used whenever a token is available.
function Download-Asset($name, $dest) {
  if ($GhToken) {
    $asset = $script:Release.assets |
      Where-Object { [string]::Equals($_.name, $name, [System.StringComparison]::Ordinal) } |
      Select-Object -First 1
    if (-not $asset) { Die "could not find the asset id for '$name' in $Repo@$($script:Release.tag_name)" }
    if ($asset) {
      New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
      Info "down $name (api asset $($asset.id))"
      # Download to a fresh temp file and move it into place: writing straight to
      # $dest would leave a half-updated file behind on failure.
      $part = "$dest.part"
      Remove-Item $part -Force -ErrorAction SilentlyContinue
      Invoke-WebRequest -Uri "$GhApi/repos/$Repo/releases/assets/$($asset.id)" `
        -Headers @{ Authorization = "Bearer $GhToken"; Accept = "application/octet-stream" } `
        -OutFile $part -UseBasicParsing
      Move-Item $part $dest -Force
      return
    }
  }
  $part = "$dest.part"
  Remove-Item $part -Force -ErrorAction SilentlyContinue
  Download "$RelBase/$name" $part
  Move-Item $part $dest -Force
}

# ---- 1. server runtime bundle ---------------------------------------------
Resolve-Release
$bundle = "trellis-$Backend-windows-x64.zip"
$appAsset = "trellis-studio-windows-x64-setup.exe"
$expected = @($bundle)
if (-not $SkipApp) { $expected += $appAsset }
Require-Assets $expected

Log "downloading trellis-server ($Backend) runtime"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
$tmp = Join-Path $env:TEMP $bundle
Download-Asset $bundle $tmp
Expand-Archive -Path $tmp -DestinationPath $RuntimeDir -Force
Remove-Item $tmp -Force
$ServerBin = Join-Path $RuntimeDir "trellis-server.exe"
if (-not (Test-Path $ServerBin)) { Die "trellis-server.exe not found after extract" }

if ($Backend -eq "rocm") {
  Warn "ROCm bundle needs a TheRock ROCm runtime on PATH; see docs/getting-started.md."
  Warn "If the server fails to start, re-run with -Backend vulkan."
}

# ---- 2. weights ------------------------------------------------------------
if ($SkipModels) {
  Warn "skipping model download (-SkipModels); set the models dir in the app's Settings."
} else {
  Log "downloading TRELLIS.2 weights [$WeightsLabel, resumable] -> $ModelsDir"
  New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
  foreach ($m in $Models) { Download "$HfBase/$QuantPath$m" (Join-Path $ModelsDir $m) }
}

# ---- 3. desktop app --------------------------------------------------------
if ($SkipApp) {
  Warn "skipping desktop app download (-SkipApp)."
} else {
  Log "downloading Trellis Studio desktop app"
  $setup = Join-Path $env:TEMP $appAsset
  # Require-Assets already proved this asset exists in the resolved release, so a
  # failure here is a real error and must not degrade into a partial install.
  Download-Asset $appAsset $setup
  Info "launching installer (silent, per-user)"
  # -PassThru so a non-zero installer exit is an error instead of a silent
  # partial install that still gets a success receipt.
  $proc = Start-Process $setup -ArgumentList "/S" -Wait -PassThru
  if ($proc.ExitCode -ne 0) { Die "Trellis Studio installer failed with exit code $($proc.ExitCode)" }
}

# The manifest (#34) is the model-set identity; record it in the receipt when the
# models directory already carries one.
$modelSet = $null
$manifestPath = Join-Path $ModelsDir "pixal3d-models.json"
if (Test-Path $manifestPath) {
  try {
    $m = Get-Content -Raw $manifestPath | ConvertFrom-Json
    $modelSet = [ordered]@{ model_set = $m.model_set; version = $m.version }
  } catch { Warn "could not read $manifestPath : $($_.Exception.Message)" }
}

# ---- 4. config -------------------------------------------------------------
Log "writing config"
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
$cfg = [ordered]@{
  serverBin = $ServerBin
  modelsDir = $ModelsDir
  backend   = $Backend
  gpu       = $Gpu
  host      = "127.0.0.1"
  port      = $Port
  outputDir = (Join-Path $Dest "output")
}
# Release receipt: what was actually installed, so a bug report or a release note
# can name the exact runtime tag/commit alongside the model set.
$receipt = [ordered]@{
  repo           = $Repo
  tag            = $script:Release.tag_name
  release_id     = $script:Release.id
  commit           = $script:ReleaseCommit
  target_commitish = $script:Release.target_commitish
  requested_tag  = $script:RequestedTag
  runtime_bundle = $bundle
  backend        = $Backend
  models_dir     = $ModelsDir
  model_set      = $modelSet
  installed_at   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}
[System.IO.File]::WriteAllText((Join-Path $ConfigDir "release.json"),
  (($receipt | ConvertTo-Json -Depth 4) + "`n"),
  (New-Object System.Text.UTF8Encoding($false)))
Info "release receipt: $(Join-Path $ConfigDir 'release.json')"

# Write UTF-8 *without* a BOM: Windows PowerShell 5.1's `Set-Content -Encoding UTF8`
# prepends a BOM, which serde_json (the app's config reader) refuses to parse, so the
# app would silently fall back to an empty/"unknown" config. WriteAllText with an
# explicit no-BOM encoding works on both Windows PowerShell 5.1 and PowerShell 7.
[System.IO.File]::WriteAllText(
  (Join-Path $ConfigDir "config.json"),
  ($cfg | ConvertTo-Json),
  (New-Object System.Text.UTF8Encoding($false)))
Info "config: $(Join-Path $ConfigDir 'config.json')"

Write-Host ""
Log "done — launch Trellis Studio from the Start menu."
