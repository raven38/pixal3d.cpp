# Release-workflow patches that need `workflow` scope

`.github/workflows/*` cannot be pushed by an OAuth token without GitHub's
`workflow` scope, so a workflow change prepared here is committed as a patch
instead of silently dropped.

## `macos-release-workflow.patch` — macOS artifacts for the Desktop alpha

Adds macOS to both jobs in `.github/workflows/release.yml`:

- runtime: `{ os: macos, backend: metal, runner: macos-14 }`, packaged as
  `trellis-metal-macos-arm64.tar.gz`. The packaging step strips the absolute
  build-directory rpath that CMake bakes in and fails if `@loader_path` is
  missing, so a released binary is self-contained and carries no path from the
  build machine (both verified locally — see the macOS section of
  `docs/PIXAL3D_RELEASE_CHECKLIST.md`).
- Studio: `{ os: macos, runner: macos-14 }`, built with
  `tauri build -- --bundles dmg,app` and uploaded as
  `trellis-studio-macos-arm64.dmg`. The bundle list is passed on the command
  line rather than added to `bundle.targets`, which would change the
  Linux/Windows artifact set.

Apply with:

```sh
gh auth refresh -h github.com -s workflow   # once, grants the missing scope
git apply patches/release/macos-release-workflow.patch
git commit -am "ci(release): build macOS artifacts for the Desktop alpha"
```

The asset names this patch produces are the ones `install/install.sh` already
expects on Darwin (`trellis-${BACKEND}-${ASSET_OS}.tar.gz` with
`ASSET_OS=macos-arm64`, and `trellis-studio-macos-arm64.dmg`), so the installer
works the moment a release carries them.
