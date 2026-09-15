# Pixal3D Roadmap

> Current release acceptance criteria live in [`PIXAL3D_RELEASE_CHECKLIST.md`](PIXAL3D_RELEASE_CHECKLIST.md).
> Refreshed after PR #24/#25/#26/#27/#28/#29/#31/#33.

## M0 — Repository/bootstrap

- [x] Private `raven38/pixal3d.cpp` with preserved trellis.cpp history
- [x] Upstream policy / baseline SHAs / MIT notices
- [x] ggml WebGPU patch strategy documented

## M1 — Native baseline

- [x] Native Metal/CUDA builds and component tests
- [x] CUDA RTX 4090 validation
- [x] Linux/GCC build gate on `main` (#33)
- [ ] Known-good generic TRELLIS.2 product output (non-Pixal3D follow-up)

## M2 — Pixal3D single-view

- [x] model/config loading
- [x] projected conditioning + ProjectAttention
- [x] camera-aware projection
- [x] single-view golden tensor parity
- [ ] productized single-view Pixal3D path (MV is release focus)

## M3 — NAF

- [x] exact NAF feature extraction
- [x] neighborhood attention
- [x] high-resolution conditioning parity
- [ ] optional DINO-only debug fallback

## M4 — Pixal3D multiview

- [x] `transforms.json` parser / camera convention tests
- [x] per-view projection / average fusion / sequential-view memory path
- [x] MV end-to-end parity
- [x] `/generate-mv`
- [x] fail-closed explicit positive `mesh_scale`
- [x] Desktop client refuses missing `mesh_scale` before request (#19 / PR #27)
- [ ] automatic `mesh_scale` estimator: **not in release path**; PR #5 failed real calibration and is closed

## M5 — Native cascade / release candidate

- [x] SS / Shape-512 / Shape-1024 / Texture-1024
- [x] shape + texture decode
- [x] textured GLB export
- [x] native CUDA benchmark
- [x] CUDA `ss_decode` graph-support path exercised on NVIDIA L4 (#11)
- [ ] clean-install Windows release-candidate E2E (#18)
- [ ] clean-install Linux release-candidate E2E (#18)

## M6 — ggml WebGPU

- [x] WASM/Emscripten build
- [x] WebGPU op inventory and smoke tests
- [x] projection/MV accumulation
- [x] sparse gather/scatter / SparseConv3D
- [x] sparse texture/PBR decoder
- [x] NAF neighborhood attention
- [x] SS decoder routed to CPU in wasm because WebGPU Conv3D path was unsafe

## M7 — Browser memory/correctness hardening

- [x] sparse/chunked Texture-1024 conditioning
- [x] SS decoder silent-corruption workaround / support guard
- [x] output-head fusion / attention chunking / device-budget gate
- [x] DINO precision and Q8_0 fixes
- [x] MLP chunking implemented as **opt-in** (#12); measured WebGPU/NOFA peak did not improve, so it is not a release blocker
- [ ] native Dawn/WebGPU issue #2 remains under investigation; Chrome/browser path is separate

## M8 — Browser end-to-end

- [x] real RGBA image loading
- [x] `transforms.json` loading
- [x] SS → Shape-512 → Shape-1024 → Texture-1024
- [x] shape + texture decode
- [x] production postprocess
- [x] textured GLB returned to JS
- [x] official cyclops-class browser full E2E
- [x] browser-safe `remesh_res=512`
- [ ] browser 1536: explicitly outside initial Web alpha scope

## M9 — Apps

### Desktop / Trellis Studio

- [x] resident native `trellis-server` integration
- [x] single-image UI / viewer / gallery / GLB export
- [x] Pixal3D MV input / matching / reorder
- [x] explicit `mesh_scale` UI
- [x] 1024 / 1536 control
- [x] progress / cancel
- [x] managed model cache + safe delete (#16 / PR #25)
- [x] real Tauri/WebKitGTK Xvfb lifecycle smoke (#17 / PR #26)
- [x] MV preflight gate (#19 / PR #27)
- [ ] clean-install Windows/Linux E2E (#18)

### Web

- [x] production app shell + viewer/download (#20 / PR #28)
- [x] verified manifest/size/SHA256 OPFS installation (#15 / PR #28)
- [x] fast production browser UI/OPFS gate (#22 / PR #29)
- [ ] automatic persistent model delivery/version invalidation/safe delete (#9)
- [ ] storage quota + WebGPU/device-budget preflight UI (#21)
- [ ] real Chrome second-launch/no-retransfer/cache-delete release smoke

## M10 — Initial releases

### Desktop alpha

- [x] shared model manifest contract (#15 / PR #24)
- [x] safe managed model cache (#16 / PR #25)
- [x] Tauri Xvfb/WebKitGTK smoke (#17 / PR #26)
- [x] MV preflight (#19 / PR #27)
- [x] CUDA graph-support hardware validation (#11)
- [ ] generate final manifest from exact release model directory
- [ ] clean-install Windows/Linux E2E (#18)
- [ ] synchronize package/Tauri version with `v0.9.0-desktop-alpha`
- [ ] verify installer asset lookup for prerelease/tag-specific release
- [ ] tag `v0.9.0-desktop-alpha`

### Web alpha

- [x] production UI (#20 / PR #28)
- [x] verified local OPFS model install (#15 / PR #28)
- [x] production UI CI (#22 / PR #29)
- [x] automatic persistence/version lifecycle (#9 / PR #42)
- [x] storage/WebGPU preflight incl. device probe (#21)
- [x] real Chrome/WebGPU full generation + cache reuse/delete release gate (local 2026-09-10, public deployment 2026-09-11)
- [ ] tag `v0.9.0-web-alpha`

## M11 — Staged model delivery (first-run latency)

現状はモデルセット全量の取得と検証が終わるまで推論が始まらない。段ごとに必要なファイルが
揃った時点で先の段を開始し、残りを裏で取得し続ける（download/inference overlap）。

**根拠となる実測**（`pixal3d-q8_0 v1` / 公式 cyclops / 各段の生成時間は
`docs/PIXAL3D_E2E_STATUS.md` の browser 実測、合計 2879 s）:

| 段 | この段までに必要 | 割合 | この段の生成 |
|---|---:|---:|---:|
| conditioning (DINOv3+NAF) | 0.30 GiB | 4.0% | — |
| SS flow + decode | 1.77 GiB | 23.5% | 194.9 s |
| Shape-512 flow | 3.14 GiB | 41.7% | 165.7 s |
| Shape-1024 flow + decode | 5.34 GiB | 70.9% | 1476.3 s |
| Texture flow + decode | 7.54 GiB | 100% | 1042.4 s |

最初の段の開始に要るのは全体の 4% だけで、残りは生成時間の裏に隠せる。短縮量の上限は
ダウンロード時間そのもの。Web (7.54 GiB) では 40 MB/s で 6%、6.7 MB/s で 28% の短縮。

**Desktop の方が効果が大きい**。F16 は 12.72 GiB（Q8_0 の 1.69 倍）で、installer は
生成開始前に全量を取得する。回線別のダウンロード時間と、測れている生成時間の比較:

| 回線 | F16 DL | vs desktop Metal 実測 1987 s |
|---|---:|---|
| 1 Gbps | 109 s | 生成が支配項 |
| 300 Mbps | 364 s | 生成が支配項 |
| 100 Mbps | 1093 s | 生成が支配項 |
| 50 Mbps | 2186 s | **DL が支配項** |
| 20 Mbps | 5464 s | **DL が支配項** |

CUDA のフルパイプライン生成時間は未測定（L4 では `ss_dec` 単体のパリティのみ確認済み、
`docs/PIXAL3D_E2E_STATUS.md`）。CUDA が Metal より速いなら境界はさらに高速側へ動き、
より多くの回線環境で DL が支配項になる。**CUDA の実測を取るまで効果の断定はしない。**

- [ ] CUDA でのフル生成時間を実測し、上表の境界を確定する（#18 の実機作業と同時に取る）
- [ ] JS 側のみで済む先行改善: ダウンロード順を manifest 記載順から段の必要順へ固定し、
      進捗表示を「あと何段で開始できるか」に変える（C++ の変更を伴わない）
- [ ] `pixal3d_real_full_run()` の C ABI を段ごとに分割し、中間状態（SS active voxel /
      SLAT / coords）を呼び出しをまたいで保持する。共有 C++ の変更なので native/CUDA/
      Vulkan と同時に効く。Pixal3D parity 作業との衝突を避けるため着手時期を調整する
- [ ] WORKERFS のマウントを全ファイル検証後の 1 回から段ごとの増分マウントへ変更する
- [ ] desktop installer を「全量取得後に起動可」から「段が揃い次第起動可」へ変更する

## M12 — Linux / GPU pod での browser 検証

実機 Linux/Windows が無いため #18 の Linux 側が未達。共有 GPU クラスタの pod 上で headless Chrome +
実 WebGPU を動かし、Web 版の検証を Linux で回せるようにする。現在の browser 実測は全て
macOS / M4 Max / Metal であり、非 Apple の WebGPU 実装での数値が 1 つも無い。

**2026-09-11 に実測できたこと（GPU pod で Vulkan は動く）**:

```
deviceName = NVIDIA L4        driverName = NVIDIA
deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU    driverInfo = 580.178.04
```

`docs/PIXAL3D_VALIDATION.md` が要求する「CPU フォールバックでないこと」を満たす。成立条件は 2 つ:

1. pod spec の `NVIDIA_DRIVER_CAPABILITIES` に `graphics` を含める（既定の `compute,utility`
   では `/etc/vulkan/icd.d/` すら作られない）
2. **`nvidia/vulkan:1.3-470` イメージを使う**。`nvidia/cuda:*` に Vulkan ローダを自分で入れても
   `Could not get 'vkCreateInstance' ... for ICD libGLX_nvidia.so.0` で ICD がスキップされる。
   失敗時と成功時で ICD json・ローダ版・デバイスノードは同一だったので、原因は
   `nvidia/cuda` に欠けている何かであり、深追いせずイメージを替えるのが最短

**2026-09-11 夕、達成（L4 / driver 580.178.04 / Chromium 153.0.8010.12）**: 自作イメージ
`webgpu-gate:gate-20260911`（`docker/linux-webgpu-gate/`、内部レジストリ）で headless Chrome から
**実 NVIDIA L4 の WebGPU アダプタ**が取れ、ggml-webgpu 相当の `requestDevice` が通った。
実測 JSON: `docs/results/linux-webgpu-gate/2026-09-11-l4-probe.json`。

| | NVIDIA L4 (Vulkan/Dawn) | M4 Max (Metal) |
|---|---:|---:|
| adapter | `nvidia` / `lovelace` | `apple` / `metal-3` |
| shader-f16 | ✓（Dawn toggle 必須、下記） | ✓ |
| maxBufferSize | 4,294,967,292 | 4,294,967,292 |
| maxStorageBufferBindingSize | **2,147,483,644** | 4,294,967,292 |
| requestDevice / 3,332 MB 確保 / f16 dispatch | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ |

成立条件は 4 つ（詳細と根拠ログは `docker/linux-webgpu-gate/README.md`）:

1. `NVIDIA_DRIVER_CAPABILITIES=all`
2. イメージに libglvnd0/libgl1/libglx0。上表の「`nvidia/cuda` で ICD スキップ」の原因はこれの不足
   （`libGLX_nvidia.so.0` の dlopen 失敗）。`nvidia/cuda:12.8.0-base-ubuntu24.04` に足すだけで通ることを
   直接確認したので、apt が死んだ focal を使う必要は無かった。Mac 側で焼き込めば apt 問題も同時に消える
3. **xvfb + headed**（確認済み）。Playwright `headless: true`（chromium-headless-shell）は同 pod でも SwiftShader。
   フル Chromium の new headless は未検証
4. **`--enable-dawn-features=vulkan_enable_f16_on_nvidia`**。Dawn は NVIDIA+Vulkan で `shader-f16` を
   意図的に塞いでいる（`crbug.com/42251215`、Chromium 153 / Dawn main 2026-09 時点）。Vulkan 側は f16 機能を全て持つのに features に出ず、
   `Unsupported feature: shader-f16` で requestDevice が落ちる。**フラグ無しの一般ユーザーの Linux/NVIDIA
   Chrome では Web 版は動かない**ことを意味する。#21 preflight はこれを `webgpu-ready` と誤判定するので、
   `adapter.features.has('shader-f16')` の検査を preflight に足す必要がある（別タスク）

**フル生成の実測は上の項目参照。** `maxStorageBufferBindingSize` 2 GiB について: `ggml-alloc` はグラフバッファを `get_max_size`
（= この値）以下のチャンクに分割するので、browser 実測の per-view graph buffer 2837.8 MB は割れる。
制約になるのは単一テンソル > 2 GiB のときだけ（現行経路の最大は 1 GB。OP_GAP B6 の im2col 2.416 GB は
L4 では不可）。旧記述の「単一バッファ 3,332 MB」は grep で出典を確認できなかったため、上の判断は
`docs/spec/31-webgpu-bringup.md` §10.4 の実測（graph buffer 2837.8 MB / peak 3673.8 MB）に基づく。

- [x] カスタムイメージを内部レジストリへ push（`Dockerfile.vulkan` → `.chrome` → `.gate` の 3 層、pod 側 apt 不使用）
- [x] headless Chrome が実 WebGPU アダプタを返す（`nvidia/lovelace`、`shader-f16` あり、`requestDevice()` 成功）
- [x] L4 の `maxBufferSize` = 4,294,967,292 / `maxStorageBufferBindingSize` = 2,147,483,644 を実測
- [x] このイメージで `web/app/run_release_gate.mjs` を L4 pod で回した（同日夜、`Dockerfile.e2e`。モデルは
      HF から直接、app は公開 Worker から。`GATE_CHROMIUM_ARGS` 環境変数を gate に追加）。
      **Linux/NVIDIA での初のフル生成**: 取得+OPFS 検証 437 s（Mac 1210 s）、生成 **3209 s**
      （Mac Metal 2879 s の 1.11 倍）、GLB 32,050,460 B、再起動後の再転送 0、削除で 3.80 GB 解放。
      SS 段の `ss_coords@64 N=4438 bbox x[3..28] y[2..29] z[0..31]` は Mac と一致。
      段別: SS 169.6 s / Shape-512 179.1 s / Shape-1024 1279.0 s / Texture 768.2 s
      （Mac: 194.9 / 165.7 / 1476.3 / 1042.4）。この初回は最終の `shasum` がイメージに無く `ok=false`。
      成果物とレンダ: `docs/results/linux-webgpu-gate/l4-20260911/`（`render_l4.png` / `render_m4max.png` 4 視点並置）
- [x] **2026-09-14 再実行で `ok: true`**（manifest ハッシュを `node:crypto` に変更、`gate_commit fb13b24`、
      `app_url` を別記録）: 取得 289 s、生成 3286 s、GLB 32,050,460 B（初回と同サイズ）、再転送 0、
      削除 3.80 GB。`docs/results/linux-webgpu-gate/l4-20260914b/`（JSON / gate.log / `render_l4.png`）
- [ ] GPU が要らない工程（7.54 GiB 取得 / OPFS 検証 / 再起動後の再転送ゼロ / キャッシュ削除）は
      CPU pod でも回せる。#18 の Linux カバレッジをそこだけ先に埋める案も残す
