# Trellis Studio

Desktop app (Tauri v2) for local image→3D generation with [trellis.cpp](../). It wraps the resident `trellis-server`, drives the image→3D pipeline, and previews results in an interactive `<model-viewer>` viewer with a locally-persisted gallery.

For end users: see [`docs/getting-started.md`](../docs/getting-started.md).

## Pixal3D multiview

Studio exposes the Pixal3D multiview pipeline through `/generate-mv`:

- drop/select pre-matted RGBA views plus `transforms.json`
- match views to `frames[].file_path`
- drag to reorder frames
- require an explicit positive `mesh_scale`; missing values fail closed
- validate matrices/FOV, matching and alpha mattes before generation
- choose 1024 or 1536
- follow resident-server progress and cancel a request
- preview/download the resulting textured GLB

Initial Desktop alpha support is **Windows x64 + Linux x86-64**. CUDA/Vulkan are advertised only on release-candidate surfaces actually validated. Native WebGPU is not an advertised Desktop alpha backend while issue #2 remains open.

Studio distinguishes a dedicated **managed model cache** from arbitrary external model folders. Only manifest-tracked regular files inside the managed root are deletable; external model directories are disconnect/reference-only. Traversal, symlink escape and symlinked managed-root cases are rejected.

Release gates and the exact support matrix are tracked in [`docs/PIXAL3D_RELEASE_CHECKLIST.md`](../docs/PIXAL3D_RELEASE_CHECKLIST.md).

## Architecture

```text
src/              Vite + TypeScript UI
  api.ts          POST /generate + /generate-mv, GET /health
  viewer.ts       <model-viewer> wrapper
  store.ts        IndexedDB gallery
  config.ts       server host/port/model config
  settings.ts     models/output/GPU/port + managed cache UI
  main.ts         generate flow/gallery/status
  mv_*.ts         Pixal3D MV calibration/preflight UI
src-tauri/
  src/server.rs   spawns/supervises trellis-server and forwards server-log events
  src/config.rs   config.json handling
  src/model_cache.rs managed-cache inspection/deletion safety boundary
  src/main.rs     Tauri commands and lifecycle
```

The app remains backend-agnostic and does not embed inference logic in the UI. Installer/model packaging supplies the native server and model set; Studio launches and supervises the configured server.

## CI

Two complementary Desktop UI gates are maintained:

- `Studio headless smoke`: TypeScript/Vite/Rust + Playwright browser-mode MV UI/preflight checks.
- `Studio Tauri Xvfb smoke`: builds the real Tauri executable, maps a WebKitGTK window, verifies Studio spawned the configured dummy server, checks `/health`, and verifies child cleanup on process exit.

Real Windows/Linux clean-install GPU generation remains a release-candidate gate rather than a hosted-CI substitute.

## Develop

```bash
npm install
npm run tauri dev
# or browser-only Studio UI against a manually started trellis-server:
npm run dev
```

Requires Node 20+, Rust, and the Tauri v2 platform dependencies.

## Build

```bash
npm run tauri build
```

Before the `v0.9.0-desktop-alpha` release, package/Tauri version metadata must be synchronized with the release tag and installer asset lookup must be verified against prerelease/tag-specific GitHub Releases.
