# install/install.ps1 の model-set 契約テスト（0.10.0: MV + SV の 2 ディレクトリ）。
# install/test_installer_manifest.sh の -VerifyModels 部分と conformance を PowerShell 側でも回す。
# （install.ps1 には --asset-base-url 相当が無いので、ダウンロード経路・config 書き出しは
# ここでは検証しない。free-space precheck は Install-ModelSet を dot-source して直接叩く。）
#
#   pwsh -NoProfile -File install/test_installer_manifest.ps1
#
# 最後に INSTALLER_MANIFEST_PS_OK を出す。
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent (Resolve-Path $MyInvocation.MyCommand.Path))
$Install = Join-Path $Root 'install/install.ps1'
$Conf = Join-Path $Root 'web/app/manifest_conformance.json'
$Work = Join-Path ([System.IO.Path]::GetTempPath()) ("pixal3d-installer-test-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$Cfg = Join-Path $Work 'cfg'

function Fail($m) { Write-Host "FAIL: $m" -ForegroundColor Red; exit 1 }
function Pass($m) { Write-Host "ok   $m" }

function Roles($fam) {
  @(@('dinov3.gguf','image_encoder'), @('pixal3d_naf.gguf','naf'),
    @("pixal3d_ss_flow_$fam.gguf",'ss_flow'), @('ss_dec.gguf','ss_decoder'),
    @("pixal3d_shape_flow_512_$fam.gguf",'shape_flow_512'), @('shape_dec.gguf','shape_decoder'),
    @("pixal3d_shape_flow_1024_$fam.gguf",'shape_flow_1024'),
    @("pixal3d_tex_flow_1024_$fam.gguf",'texture_flow_1024'), @('tex_dec.gguf','texture_decoder'))
}
function Make-Set($fam, $explicit) {
  $d = Join-Path $Work "src-$fam"; New-Item -ItemType Directory -Force -Path $d | Out-Null
  $files = @(); $i = 0
  foreach ($pair in (Roles $fam)) {
    $name = $pair[0]; $role = $pair[1]
    $data = [System.Text.Encoding]::ASCII.GetBytes("installer-test-$fam-$i-$role")   # 同名でも MV/SV で中身が違う
    [System.IO.File]::WriteAllBytes((Join-Path $d $name), $data)
    $sha = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($data)).Replace('-','').ToLower()
    $files += [ordered]@{ name = $name; role = $role; required = $true; size_bytes = $data.Length; sha256 = $sha }
    $i++
  }
  $m = [ordered]@{ schema_version = 1; model_set = "pixal3d-test-$fam"; version = 'v1'; files = $files }
  if ($explicit) { $m['model_family'] = $fam }
  [System.IO.File]::WriteAllText((Join-Path $d 'pixal3d-models.json'), ($m | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
  return $d
}
$MvSrc = Make-Set 'mv' $false
$SvSrc = Make-Set 'sv' $true
$Mv = Join-Path $Work 'mv'; $Sv = Join-Path $Work 'sv'; $Empty = Join-Path $Work 'empty'
Copy-Item -Recurse $MvSrc $Mv; Copy-Item -Recurse $SvSrc $Sv; New-Item -ItemType Directory -Force -Path $Empty | Out-Null

$script:LastOut = ''
function Run($wantRc, $pattern, [string[]]$argv) {
  $out = (& pwsh -NoProfile -File $Install @argv -ConfigDir $Cfg 2>&1 | Out-String)
  $rc = $LASTEXITCODE
  $script:LastOut = $out
  if ($rc -ne $wantRc) { Write-Host $out; Fail "rc=$rc want=${wantRc}: install.ps1 $($argv -join ' ')" }
  if ($pattern -and -not $out.Contains($pattern)) { Write-Host $out; Fail "output lacks '$pattern': install.ps1 $($argv -join ' ')" }
}

# --- 1. verify-only ---------------------------------------------------------
Run 0 'model set OK' @('-VerifyModels', '-ModelsDir', $Mv)
Pass 'MV verify-only accepts an intact set'
Run 0 'pixal3d-test-sv v1 (sv)' @('-VerifyModels', '-ModelsDir', $Mv, '-ModelsDirSv', $Sv)
if (-not $script:LastOut.Contains('pixal3d-test-mv v1 (mv)')) { Fail 'MV was not verified alongside SV' }
Pass 'verify-only checks both directories'

[System.IO.File]::AppendAllText((Join-Path $Sv 'tex_dec.gguf'), 'x')
Run 1 'does not verify: tex_dec.gguf' @('-VerifyModels', '-ModelsDir', $Mv, '-ModelsDirSv', $Sv)
if (-not $script:LastOut.Contains('pixal3d-test-mv v1 (mv)')) { Fail 'MV should still verify before the SV failure' }
Pass 'SV corruption is rejected while MV still verifies'
Copy-Item (Join-Path $SvSrc 'tex_dec.gguf') (Join-Path $Sv 'tex_dec.gguf') -Force

Remove-Item (Join-Path $Sv 'pixal3d_naf.gguf')
Run 1 'does not verify: pixal3d_naf.gguf' @('-VerifyModels', '-ModelsDir', $Mv, '-ModelsDirSv', $Sv)
Pass 'SV missing file is rejected'
Copy-Item (Join-Path $SvSrc 'pixal3d_naf.gguf') (Join-Path $Sv 'pixal3d_naf.gguf') -Force

# サイズ一致・内容違い
$p = Join-Path $Sv 'ss_dec.gguf'; $b = [System.IO.File]::ReadAllBytes($p); $b[0] = $b[0] -bxor 1; [System.IO.File]::WriteAllBytes($p, $b)
Run 1 'does not verify: ss_dec.gguf' @('-VerifyModels', '-ModelsDir', $Mv, '-ModelsDirSv', $Sv)
Pass 'same-size content change is rejected'
Copy-Item (Join-Path $SvSrc 'ss_dec.gguf') $p -Force

# family 取り違え
Run 1 'expects the sv set' @('-VerifyModels', '-ModelsDir', $Mv, '-ModelsDirSv', $Sv, '-ModelManifestSv', (Join-Path $Mv 'pixal3d-models.json'))
Pass 'an MV manifest is rejected for the SV directory'
Run 1 'expects the mv set' @('-VerifyModels', '-ModelsDir', $Mv, '-ModelManifest', (Join-Path $Sv 'pixal3d-models.json'))
Pass 'an SV manifest is rejected for the MV directory'

# トラバーサル名（既知 role に未知名 → manifest 段階で拒否）
$m = Get-Content -Raw (Join-Path $Mv 'pixal3d-models.json') | ConvertFrom-Json
$m.files[0].name = '../evil.gguf'
$trav = Join-Path $Work 'traversal.json'; $m | ConvertTo-Json -Depth 5 | Set-Content $trav
Run 1 '' @('-VerifyModels', '-ModelsDir', $Mv, '-ModelManifest', $trav)
if ($script:LastOut.Contains('does not verify')) { Fail 'traversal name reached the file stage' }
Pass 'traversal name is rejected before any file is read'

# 混在 manifest
$m = Get-Content -Raw (Join-Path $Sv 'pixal3d-models.json') | ConvertFrom-Json
$m.PSObject.Properties.Remove('model_family')
foreach ($f in $m.files) { if ($f.role -eq 'ss_flow') { $f.name = 'pixal3d_ss_flow_mv.gguf' } }
$mixed = Join-Path $Work 'mixed.json'; $m | ConvertTo-Json -Depth 5 | Set-Content $mixed
Run 1 '' @('-VerifyModels', '-ModelsDir', $Mv, '-ModelsDirSv', $Sv, '-ModelManifestSv', $mixed)
if ($script:LastOut.Contains('does not verify')) { Fail 'mixed manifest reached the file stage' }
Pass 'mixed-family manifest is rejected'

# --- 2. conformance vectors (JS/Python/C++/bash と同じ判定) ---------------------
$conf = Get-Content -Raw $Conf | ConvertFrom-Json
$n = 0
foreach ($c in $conf.cases) {
  $n++
  $case = Join-Path $Work 'case.json'
  $c.manifest | ConvertTo-Json -Depth 6 | Set-Content $case
  $out = (& pwsh -NoProfile -File $Install -VerifyModels -ModelsDir $Empty -ModelManifest $case -ConfigDir $Cfg 2>&1 | Out-String); $rc = $LASTEXITCODE
  $outSv = (& pwsh -NoProfile -File $Install -VerifyModels -ModelsDir $Mv -ModelsDirSv $Empty -ModelManifestSv $case -ConfigDir $Cfg 2>&1 | Out-String); $rcSv = $LASTEXITCODE
  $family = if ($null -eq $c.family) { 'null' } else { [string]$c.family }
  switch ($family) {
    'null' {
      if ($rc -eq 0 -or $rcSv -eq 0) { Fail "$($c.id): expected rejection (rc=$rc rcSv=$rcSv)" }
      if (($out + $outSv).Contains('does not verify')) { Write-Host $out; Write-Host $outSv; Fail "$($c.id): rejected at the file stage, not the manifest stage" }
    }
    'mv' {
      if (-not $out.Contains('does not verify')) { Write-Host $out; Fail "$($c.id): MV manifest should pass the manifest stage" }
      if (-not $outSv.Contains('expects the sv set')) { Write-Host $outSv; Fail "$($c.id): MV manifest should be refused as SV" }
    }
    'sv' {
      if (-not $outSv.Contains('does not verify')) { Write-Host $outSv; Fail "$($c.id): SV manifest should pass the manifest stage" }
      if (-not $out.Contains('expects the mv set')) { Write-Host $out; Fail "$($c.id): SV manifest should be refused as MV" }
    }
    default { Fail "$($c.id): unknown expected family '$family'" }
  }
}
if ($n -lt 18) { Fail "only $n conformance cases ran" }
Pass "conformance vectors: $n cases agree with the shared contract"

# --- 3. free-space precheck (size_bytes 1e15 の manifest、ダウンロード前に落ちる) -----
$m = Get-Content -Raw (Join-Path $Sv 'pixal3d-models.json') | ConvertFrom-Json
$m.files[0].size_bytes = [long]1e15
$huge = Join-Path $Work 'huge.json'; $m | ConvertTo-Json -Depth 5 | Set-Content $huge
$dest = Join-Path $Work 'dest-sv'
# -VerifyModels 無しの経路は release 解決が要るので、関数だけを取り出して直接呼ぶ。
$fnSrc = Get-Content -Raw $Install
$start = $fnSrc.IndexOf('# ---- Pixal3D model set (manifest-verified)')
$end = $fnSrc.IndexOf('function Fetch-Manifest(')
$body = $fnSrc.Substring($start, $end - $start)
$sb = [scriptblock]::Create(@"
`$ErrorActionPreference = 'Stop'
function Log(`$m)  { Write-Host "==> `$m" }
function Info(`$m) { Write-Host " -  `$m" }
function Warn(`$m) { Write-Host "warn: `$m" }
function Die(`$m)  { Write-Host "error: `$m"; exit 1 }
function Download(`$url, `$dest) { Write-Host "DOWNLOAD_ATTEMPTED `$url"; exit 99 }
`$VerifyModels = `$false
$body
Install-ModelSet '$huge' '$dest' 'http://127.0.0.1:9/sv' 'sv'
"@)
$out = (& pwsh -NoProfile -Command $sb 2>&1 | Out-String); $rc = $LASTEXITCODE
if ($rc -ne 1 -or -not $out.Contains('not enough free space')) { Write-Host $out; Fail "free-space precheck did not refuse (rc=$rc)" }
if ($out.Contains('DOWNLOAD_ATTEMPTED')) { Fail 'a download was attempted despite the free-space refusal' }
if (Get-ChildItem -Path $dest -Filter '*.part' -ErrorAction SilentlyContinue) { Fail '.part left after the free-space refusal' }
Pass 'free-space precheck refuses before downloading'

Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
Write-Host 'INSTALLER_MANIFEST_PS_OK'
exit 0
