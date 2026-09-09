# Pixal3D Upstream Policy

## Remotes

Use:

```text
origin    git@github.com:raven38/pixal3d.cpp.git
upstream  https://github.com/pwilkin/trellis.cpp.git
```

`origin` is the private development repository. `upstream` remains read-only and is used to inspect and selectively integrate trellis.cpp improvements.

## Baseline (recorded 2026-09-06)

| component | ref |
|---|---|
| trellis.cpp base (last upstream-shared commit) | `2516c48` — "Merge pull request #44 from fernandotonon/macos-metal" |
| `thirdparty/ggml` submodule | `737e88f25d4f62254f3b7a726fd9663036cc94da` (`pwilkin/ggml` branch `trellis-patches`, v0.15.1-10) |
| Pixal3D reference source | `TencentARC/Pixal3D` `f7cf384` ("feat: support multi-view image input") |
| Pixal3D weights | HF `TencentARC/Pixal3D` snapshot `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f` (MV DiTs + decoders) |

## Sync policy

Before native Pixal3D parity:

- keep upstream merges infrequent
- pin exact upstream and ggml SHAs
- avoid large refactors while implementing Pixal3D

After parity:

- review upstream changes periodically
- cherry-pick or merge changes only with validation fixtures
- avoid updating ggml and Pixal3D graph logic in the same debugging commit when possible
- local ggml changes go in `patches/ggml-webgpu/*.patch` (applied to the submodule working
  tree at configure time, see `docs/GGML_FORK_DIFF.md` "Local patches"), never as edits to the
  pinned submodule commit; keep each patch an `#ifndef`-guarded no-op by default so it can be
  offered upstream and dropped on the next sync

## License and attribution

Preserve upstream MIT license/copyright notices for copied or substantially derived code.

Before public release, add/maintain explicit attribution for trellis.cpp, TencentARC/Pixal3D, ggml/llama.cpp, and any model/runtime dependencies incorporated by the project.
