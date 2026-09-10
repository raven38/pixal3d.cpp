# Release-workflow patch history

The macOS Desktop-alpha release change was originally stored here because the
previous OAuth token could not modify `.github/workflows/*` without GitHub's
`workflow` scope.

The change is now applied directly to `.github/workflows/release.yml` on this
PR. `macos-release-workflow.patch` is retained only as implementation history;
it is **not** an extra step that must be applied before release.

## macOS artifacts added to `release.yml`

The release workflow now adds macOS to both release jobs:

- runtime: `{ os: macos, backend: metal, runner: macos-14 }`, packaged as
  `trellis-metal-macos-arm64.tar.gz`. Packaging strips the absolute build rpath,
  requires `@loader_path`, and fails closed if the build-machine path remains.
- Studio: `{ os: macos, runner: macos-14 }`, built with
  `tauri build -- --bundles dmg,app` and uploaded as
  `trellis-studio-macos-arm64.dmg`. The bundle list stays command-line-only so
  Linux/Windows bundle targets are unchanged.

The asset names match the Darwin path in `install/install.sh`, so the installer
can consume the release artifacts directly. The current alpha is ad-hoc signed
and not notarized; the scripted installer clears quarantine, while direct dmg
distribution must still be described as unsigned/unnotarized.
