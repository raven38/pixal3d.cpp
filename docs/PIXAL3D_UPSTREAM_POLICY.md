# Pixal3D Upstream Policy

## Remotes

Use:

```text
origin    git@github.com:raven38/pixal3d.cpp.git
upstream  https://github.com/pwilkin/trellis.cpp.git
```

`origin` is the private development repository. `upstream` remains read-only and is used to inspect and selectively integrate trellis.cpp improvements.

## Sync policy

Before native Pixal3D parity:

- keep upstream merges infrequent
- pin exact upstream and ggml SHAs
- avoid large refactors while implementing Pixal3D

After parity:

- review upstream changes periodically
- cherry-pick or merge changes only with validation fixtures
- avoid updating ggml and Pixal3D graph logic in the same debugging commit when possible

## License and attribution

Preserve upstream MIT license/copyright notices for copied or substantially derived code.

Before public release, add/maintain explicit attribution for trellis.cpp, TencentARC/Pixal3D, ggml/llama.cpp, and any model/runtime dependencies incorporated by the project.
