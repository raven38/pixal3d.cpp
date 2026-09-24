# 33 — Pixal3D sparse-structure resolution (`ss_res` 32 | 64) and the two 4 GiB limits

Epic #78 (children #79 oracle, #80 parameter, #81 quantizer, #82 GPU buffers, #83 wasm32 tail,
#84 release matrix). Branch `feat/ss-res-64` (from `main` = `d1b4926`, 2026-09-24).

Every number below is *measured* in this branch unless marked *derived*. Backends: native CUDA
(RTX 4090, `ssh win`, f16 GGUFs in `/mnt/hdd1/pixal3d/gguf` and the q8_0 desktop set), wasm32
(Emscripten 6.0.9, Node 26 and Chrome) — see §8 for what was *not* run.

## 1. What `ss_res` is, and which source this branch follows

The SS decoder always emits 64³ occupancy logits. `ss_res` is the grid the sparse structure is
read on: 32 max-pools them 2×2×2 (`sample_sparse_structure`, `ss_coords(logits, 64, 32)`), 64 keeps
them. The LR shape-512 flow then runs over those coords, the shape decoder upsamples them 16×
(`shape_decoder.upsample(..., upsample_times=4)`), and the cascade quantizes the upsampled coords
onto the HR flow grid `hr_res / 16`.

The three sources disagree, so the choice is explicit (W21):

| source | ss_res | LR projection grid | cascade quantizer |
|---|---|---|---|
| TencentARC Pixal3D `f7cf384`, `Pixal3DImageTo3DPipeline.run()` | hardcoded `ss_res = 32` | model config (32) | `round((c + 0.5) / 512 * (grid - 1))` |
| visualbruno/ComfyUI-Trellis2 (#193 "correct") | 32 or 64 | — | `int((c + 0.5) / (512 * ss_res/32) * grid)` (TRELLIS.2 truncation) |
| Aero-Ex/ComfyUI-Trellis2-GGUF (#193 "bug") | 32 or 64 | `coords.max() + 1` | `/ 512` literal → doubled coords at 64 |
| **this branch** | 32 (default) or 64 | **`R = ss_res`** | `round((c + 0.5) / (ss_res * 16) * (grid - 1))` |

We keep Pixal3D's own round / (grid − 1) convention (it is what the Pixal3D reference and our
existing fixtures use) and generalize only the span, which is `ss_res * 16` because the upsample
is 16×. The LR projection grid is the SS coordinate domain itself, never inferred from
`max(coord) + 1` (that is how the GGUF fork turned the doubled coords into an 8× dense `R³`
allocation). There is no upstream ss_res=64 Pixal3D run to compare against, so the oracle (§3)
is the reference *modules* driven with this formula.

## 2. Code

| where | change |
|---|---|
| `include/pixal3d_cascade.h`, `src/pixal3d_cascade.cpp` | `pixal3d_quantize_hr_coords(up, ss_res, hr_res)` and `pixal3d_cascade_select(...)` (run()'s 1024/1536 backoff). Throws when an upsampled coord is outside `[0, ss_res*16)` (wrong ss_res) or a quantized coord leaves `[0, grid)`. For ss_res=32 the float expression is the former one token for token (`/512.f`). |
| `trellis_args` | `--ss-res 32\|64` (strict), default 32; any other value, or use outside `--views`/`--sv-image`, fails at parse time. `trellis-server` has no `--views`, so the server cannot enable it. |
| `trellis_cli.cpp` `trellis_run_mv` | `ss_coords(logits, 64, ss_res)`; LR cond `Pixal3dSlatCondParams{512, ss_res, 512}`; cascade via `pixal3d_cascade_select`; logs `active voxels @res<ss_res> = N, max coord M`, `upsampled coords @res<span>=N (max M) -> ... tokens (max M)`; `--voxply` normalizes by ss_res. |
| `trellis_cli.cpp` TRELLIS.2 single-image cascade | unchanged (ss_res=32 fixed, truncation/grid formula) — commented why the Pixal3D helper must not be reused there. |
| `pixal3d_real_geometry_wasm.cpp` (browser real-input E2E + `trellis-test-pixal3d-real-e2e`) | `g_ss_res`, C ABI `pixal3d_real_set_ss_res(int)`, native `--ss-res N`; browser page/worker/Playwright take it (`web/real_e2e`). |
| `pixal3d_full_e2e_wasm.cpp` | fixture-driven, stays at 32 (`kSsRes`), but uses the helper. |
| `sparse.cpp` `sparse_c2s(..., coords_only)` | the cascade upsample's last C2S stops after the subdivision mask (identical coords; §7.2 — needed so ss_res=64 does not push the browser heap to 4 GiB). |
| `pixal3d_cond_gpu.cpp`, `pixal3d_postprocess.cpp` | §5 and §6. |

Audit of the remaining literal 32 / 512 in the Pixal3D path: SS latent `16³ × 8`, the SS decoder's
fixed 64³ output, the 32 latent channels, and `S = 512` image sizes are not ss_res; the partial
E2E's `cmax < 32 ? 512 : 1024` infers a *decode* resolution from HR coords (unchanged semantics).

## 3. Oracle (#79)

`tools/ref_pixal3d_ss_res.py` (win, Pixal3D `f7cf38429b0bd264f1995f0f8743a88b1c728b94`, weights
snapshot `b0cb2e1b…`, seed 42, `HR=1`):

```sh
ATTN_BACKEND=sdpa HF_HOME=/mnt/d/pixal3d/hf_home HF_HUB_OFFLINE=1 PIXAL3D_REPO=/mnt/hdd1/pixal3d/Pixal3D \
OUT=/mnt/hdd1/pixal3d/ref/pixal3d/ss_res HR=1 /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_ss_res.py
```

It reuses the pinned SS logits (`ss_sample/f32_occ_logits.npy` — the SS stage does not depend on
ss_res) and the 4-view `cond_slat` images/cameras, and writes `ss32/` and `ss64/` in the
`slat_sample` (LR) and `hr_sample` (`shape_hr`) layouts plus `upsampled_coords`, `hr1024_coords`,
`hr1536_coords` + `hr1536_meta` and `meta.json`. ~5 min per ss_res on the 4090, ~0.9 GB total;
not committed (regenerate with the command above).

| | ss_res=32 | ss_res=64 |
|---|---|---|
| SS active voxels (grid) | 4 377 (32³, coords 0..31) | 18 180 (64³, 0..63) |
| LR projected cond, reference dense / our sparse | 256 MiB / 34 MiB | **2 GiB** / 142 MiB |
| upsampled coords (span) | 1 188 442 (0..511, span 512) | 4 754 615 (0..1023, span 1024) |
| HR@1024 tokens (grid 64) | 17 489, coords 0..63 | 17 585, coords 0..63 |
| same with the literal `/512` quantizer | 17 489 (identical) | **71 130, coords 0..126** (the #193 bug) |
| 1536 cascade, `max_num_tokens` = its 1536 count | backs off to 1408: 33 423 tokens | backs off to 1408: 33 904 tokens |
| reference peak CUDA alloc, LR cond / LR flow / upsample / HR flow (MiB) | 4 690 / 10 669 / 2 037 / 10 896 | 8 422 / 10 908 / 5 397 / 10 956 |

ss_res=32 reproduces the older pinned fixtures bit for bit (`coords`, `noise`, `cond_proj`,
`f32_slat` of `slat_sample`, `hr_coords` of `hr_sample`: max |Δ| = 0).

## 4. Native parity (CUDA, f16)

| check | test | ss_res=32 | ss_res=64 |
|---|---|---|---|
| SS coords from the reference logits | `trellis-test-pixal3d-cascade --fixture` | exact (4 377) | exact (18 180) |
| quantizer on the reference upsample, 1024 | same | exact (17 489) | exact (17 585) |
| 1536 backoff (res, tokens) | same | exact (1408, 33 423) | exact (1408, 33 904) |
| native `shape_upsample` raw coords, symmetric diff | same (gate 1e-3) | 664 / 1.19 M = 5.6e-4 | 3 127 / 4.75 M = 6.6e-4 |
| HR tokens from the native upsample | same (gate 1e-3) | 2 differ (1.1e-4) | 0 differ |
| LR cond (sparse, split-graph NAF) rel max \|Δ\| global / lr / hr | `trellis-test-pixal3d-cond-slat --ss-res-dir` (tol 2e-2) | 2.7e-4 / 9.1e-4 / 3.5e-4 | 2.7e-4 / 1.2e-3 / 3.4e-4 |
| HR cond, same | same | 4.1e-4 / 2.0e-3 / 2.3e-4 | 4.1e-4 / 2.0e-3 / 2.2e-4 |
| LR shape flow, 12 steps: rel_final (cos) vs f32 ref; calibration f32↔bf16 | `trellis-test-pixal3d-slat-sample --stage shape512` | 0.191 (0.99965); 0.262 | 0.547 (0.99822); 0.722 |
| HR shape flow, same | `--stage shape_hr` | 0.280 (0.99932); 0.415 | 0.359 (0.99869); 0.428 |

The raw upsample difference is the fp16 decoder's subdivision threshold flipping a few near-zero
children; it is the same size at ss_res=32 on the unchanged code path, so it is not an ss_res
effect. The LR flow at 64³ is itself more sensitive (the reference's own f32↔bf16 spread is 0.72
vs 0.26 at 32³) — the 512 flow was trained on 32³ structures.

`trellis-test-pixal3d-cascade` without arguments runs the weight-free checks: ss_res=32 helper ==
literal-512 formula and the old backoff loop (all per-axis values at 1024…1536 + 200k random
coords), ss_res=64 maps the full span to 0..63 (the literal formula gives 126), 64-vs-32 cell
disagreement ≤ 1, and fail-closed inputs.

## 5. GPU / WebGPU per-buffer memory (#82)

A single tensor cannot be split across buffers, so what a WebGPU adapter bounds is the largest
*tensor* (`maxStorageBufferBindingSize`, reported by `ggml_backend_get_max_size`), not aggregate
VRAM. `pixal3d_cond_slat_plan()` computes the largest planned tensor in `uint64_t` before
anything is allocated; `pixal3d_cond_slat_gpu` logs it (`[cond-mem]`) and throws before
allocating when it exceeds the backend limit. `Pixal3dCondStats` carries
`largest_planned_bytes` / `backend_max_buffer`.

The plan counts the accumulators plus the known large intermediates: DINOv3 attention scores
(`soft_max(kᵀq)` is materialized, `[N, N, 16]` f32 — 1 026.5 MiB at S=1024), the NAF encoder concat
`[S, S, 256]`, pooled / q `[T, T, 256]`, and the neighborhood-attention value windows / scores /
output for the block chunk. It is the up-front estimate; the exact per-node gate already exists:
`check_graph_supported` throws before allocation when ggml-webgpu's `supports_op` rejects a node
or source larger than `maxStorageBufferBindingSize` (upstream ggml-webgpu behaviour).

Production plans at the worst case (every cell of the grid active), from the unit test:

| call | largest planned tensor | fits 2 047 MiB class | fits 4 095 MiB class |
|---|---|---|---|
| LR ss32 S512 T512 R32 | 256 MiB (NAF encoder concat) | yes | yes |
| **LR ss64 S512 T512 R64** | 1 024 MiB (sparse accumulator, 262 144 tokens) | yes | yes |
| HR 1024 S1024 T512 R64 (CLI, chunk 1024) | 1 026.5 MiB (DINOv3 scores) | yes | yes |
| HR 1536 S1024 T512 R96, token budget | 1 026.5 MiB (DINOv3 scores) | yes | yes |
| texture S1024 T1024 R64 (auto chunk) | 1 026.5 MiB (DINOv3 scores) | yes | yes |
| HR S1024 T512 R64 single graph (real-E2E driver) | 1 296 MiB (NAF value windows) | yes | yes |
| dense R96 accumulator (not a production path) | 3 456 MiB | **no** | yes |

So ss_res=64 does not move the largest tensor: at 64 the LR stage's worst case is the full 64³
grid as sparse tokens (1 GiB), below the S=1024 DINOv3 scores every run already has.

T=1024 always takes the split-graph NAF path, so the 4 GiB `[1024, 1024²]` NAF map is never one
tensor. Every production caller passes active coords (`trellis_cli.cpp` ×3, real/full E2E); only
fixture-comparison drivers use the dense path. Real ss_res=64 runs log e.g.
`[cond-mem] S=512 T=512 R=64 tokens=17881 (sparse gather, NAF chunked): largest planned tensor
256.0 MiB`.

## 6. wasm32 host heap: the bounded tail (#83)

Independent of §5: the browser build is wasm32 with `-sMAXIMUM_MEMORY=4294967296`, and the
postprocess tail (weld → TriBvh → narrow-band remesh → QEM → UV/bake) runs in that heap.

**Replay harness.** `trellis-test-pixal3d-real-e2e` with `PIXAL3D_DUMP_FIXTURE=<dir>` also writes
the decoded mesh + PBR voxels (`tail_*.npy`); `web/tail_replay/pixal3d_tail_replay.js`
(`pixal3d-tail-replay-node`, the same tail compiled for Node with the browser's 4 GiB ceiling and
allocator) and native `trellis-pixal3d-tail-replay` replay it in ~1.5–4 min.

**Measured** (cyclops ss_res=32, decode 9 444 178 faces; grown-heap high-water mark, MiB):

| remesh_res | remesh faces | tail peak from a clean heap | phase at peak |
|---|---|---|---|
| 512 | 4 476 320 | 1 982 | TriBvh build |
| 640 | 7 024 576 | 2 089 | QEM |
| 768 | 10 147 908 | 2 641 | QEM |
| 896 | 14 052 132 | 3 185 | QEM |
| 1024 | 18 392 028 | 3 809 | QEM |

So remesh 1024 *does* fit a clean wasm32 heap; the historical browser `std::bad_alloc` at 1024
(`PIXAL3D_E2E_STATUS.md` §4) came from the ~918 MiB already live when the tail started. The fixed
`remesh_res = 512` threw that fidelity away regardless of the heap.

**Policy** (`Pixal3dPostprocessOptions::host_budget_bytes`, browser drivers pass
`kPixal3dWasm32TailBudget` = 4 GiB − 128 MiB, `remesh_res = 0` = start at the decode resolution):
rungs `res, 3/4, 1/2, 3/8` (multiples of 16, ≥ 128); predicted peak = live heap at tail start +
max(180 B × F, 80 B × F + 160 B × 1.95 × F × (rung/res)²) (constants fitted to the table above,
F = decoded faces); the first rung that fits is taken and logged with every rung's prediction; a
`std::bad_alloc` inside remesh/QEM still steps down one rung. A start state that cannot hold
weld/BVH, or where even the smallest rung's prediction exceeds the budget, fails with
`HOST_HEAP_EXHAUSTED (preflight)` before building anything; a `std::bad_alloc` anywhere else in the
tail (or on the last rung) returns `HOST_HEAP_EXHAUSTED: std::bad_alloc during <stage>` instead of
an unhandled exception. Native passes no budget; the only native change is that last case
(previously the exception escaped).

Guard cases (wasm32 replay, cyclops ss32, budget 3 968 MiB; `--prefill-mib` emulates the heap the
flow stages leave behind):

| case | log | outcome |
|---|---|---|
| clean heap | 1024 predicted 3 853 → selected | OK, peak 3 809 |
| 918 MiB live | 1024 predicted 4 813 (over) → 768 predicted 3 584 selected | OK, peak 3 601, `BOUNDED_FALLBACK 1024 -> 768` |
| budget lied (100 000), 1 500 MiB live | 1024 selected, `remesh@1024 exhausted the host heap (std::bad_alloc); stepping down to 768` | OK |
| 3 000 MiB live | `HOST_HEAP_EXHAUSTED (preflight): live 3331 MiB + weld/BVH 1621 MiB` | clean FAIL |
| no budget, 1 500 MiB live | `HOST_HEAP_EXHAUSTED: std::bad_alloc during remesh` | clean FAIL |
| native replay, remesh_res 2048, budget 1 700 MiB | every rung over (smallest 768 needs 2 301) → `HOST_HEAP_EXHAUSTED (preflight): no remesh rung fits` | clean FAIL before weld |

ss_res=64 does not change the tail's input scale (cyclops seed 42: 9 348 986 vs 9 444 178 decoded
faces; tail peak 3 770 vs 3 809 MiB at 1024) and goes through the same guard.

**Geometry vs native** (surface-sampled, 2 M points, chamfer ×1e3 / F-score @ 1/1024; native CUDA
real-input run = remesh 1024, 1 M faces, same seed):

| | ss_res=32 | ss_res=64 |
|---|---|---|
| native vs itself, other sampling seed (noise floor) | 0.897 / 0.603 | 0.890 / 0.609 |
| browser policy (remesh 1024, 500 k faces) | 0.899 / 0.602 | 0.892 / 0.609 |
| old fixed remesh 512 | 1.435 / 0.018 | 1.381 / 0.020 |

## 7. Release matrix (#84)

### 7.1 Native CUDA (RTX 4090, `trellis-cli --views`, 18 runs)

Inputs: official cyclops 4-view (`transforms.json`, mesh_scale 1.0) and the synthetic `mvgen/dragon`
4-view set; f16 GGUFs and the shipped q8_0 desktop set; 1024 cascade (default `max_tokens` 49152)
plus one 1536 smoke per ss_res. VRAM = `nvidia-smi` sampled every 200 ms (includes ~1.3 GB held by
another process on that GPU; the true peak can be shorter than the sample interval). IoU =
`tools/silhouette_iou.py` against the input alphas.

| run | ss_res | SS vox (max) | LR R | upsample (span, max) | HR res / tokens (max) | decoded F | wall s | VRAM MiB | IoU |
|---|---|---|---|---|---|---|---|---|---|
| cyclops f16 s1 | 32 | 4 333 (31) | 32 | 1 170 961 (512, 511) | 1024 / 17 342 (63) | 9 224 068 | 157 | 7 895 | 0.9814 |
| | 64 | 17 881 (63) | 64 | 4 708 160 (1024, 1023) | 1024 / 17 266 (63) | 9 470 956 | 221 | 7 961 | 0.9827 |
| cyclops f16 s2 | 32 | 4 405 (31) | 32 | 1 198 432 | 1024 / 17 623 (63) | 9 387 588 | 177 | 7 939 | 0.9791 |
| | 64 | 17 855 (63) | 64 | 4 803 485 | 1024 / 17 537 (63) | 10 639 558 | 228 | 8 183 | 0.9840 |
| cyclops f16 s3 | 32 | 4 284 (31) | 32 | 1 155 415 | 1024 / 17 049 (63) | 9 001 104 | 184 | 7 823 | 0.9817 |
| | 64 | 17 720 (63) | 64 | 4 649 319 | 1024 / 17 193 (63) | 9 091 546 | 220 | 7 921 | 0.9827 |
| cyclops q8_0 s1 | 32 | 4 295 (31) | 32 | 1 182 827 | 1024 / 17 140 (63) | 10 166 954 | 215 | 7 553 | 0.9823 |
| | 64 | 17 754 (63) | 64 | 4 660 808 | 1024 / 17 040 (63) | 9 410 128 | 252 | 7 473 | 0.9828 |
| cyclops q8_0 s2 | 32 | 4 392 (31) | 32 | 1 191 433 | 1024 / 17 469 (63) | 9 334 100 | 219 | 7 481 | 0.9808 |
| | 64 | 17 812 (63) | 64 | 5 349 435 | 1024 / 17 460 (63) | 10 909 778 | 282 | 7 805 | 0.9833 |
| cyclops q8_0 s3 | 32 | 4 284 (31) | 32 | 1 160 336 | 1024 / 17 049 (63) | 9 027 252 | 193 | 7 385 | 0.9805 |
| | 64 | 17 698 (63) | 64 | 4 599 773 | 1024 / 17 138 (63) | 9 107 016 | 252 | 7 443 | 0.9831 |
| dragon f16 s1 | 32 | 2 788 (29) | 32 | 826 187 (512, 474) | 1024 / 11 454 (58) | 7 713 478 | 122 | 7 227 | 0.9486 |
| | 64 | 11 610 (59) | 64 | 3 276 855 (1024, 954) | 1024 / 11 421 (59) | 7 024 772 | 140 | 7 097 | 0.9409 |
| dragon f16 s2 | 32 | 2 777 (29) | 32 | 832 273 | 1024 / 11 435 (58) | 6 924 664 | 117 | 7 089 | 0.9445 |
| | 64 | 11 616 (59) | 64 | 3 324 358 | 1024 / 11 475 (59) | 6 960 670 | 142 | 7 093 | 0.9403 |
| cyclops f16 s1, `--res 1536` | 32 | 4 333 | 32 | 1 170 961 | 1536 / 39 748 (95) | 20 599 440 | 382 | 11 301 | 0.9870 |
| | 64 | 17 881 | 64 | 4 708 160 | 1536 / 40 113 (95) | 20 867 600 | 424 | 11 381 | 0.9835 |

These runs predate the coords-only upsample fix (§7.2); it changes no coordinate (verified on the
oracle fixture), only the upsample's time and memory. All 18 exit 0 with `nan/inf=0` in every SLAT, no backoff (all under 49 152 tokens), HR coords
always inside the model grid (max 63 / 95), and `[cond-mem]` largest planned conditioning tensor
1 024 MiB in every run (the texture stage's accumulator plan).

**Correctness gates:** ss_res=32 is numerically the old path (§4: exact coords, fixtures bit-identical);
ss_res=64 never produces doubled HR coords, no R≈128 conditioning, no dense R³ projection
(sparse gather logged in every run), no full 4 GiB NAF tensor (split-graph NAF logged).

**Cost of 64:** wall time +18 % to +40 % (the LR flow runs over ~4× the tokens: 17.7 k vs 4.3 k);
HR token count and decoded face count are unchanged within seed spread; VRAM peaks within the
sampling noise (−80 to +320 MiB).

**Quality (paired, same seed; IoU against the input silhouettes):** cyclops +0.0013 / +0.0049 /
+0.0010 (f16), +0.0005 / +0.0025 / +0.0026 (q8_0); dragon −0.0077 / −0.0042; 1536 smoke −0.0035.
The per-subject seed spread at ss_res=32 is 0.0026 (cyclops f16), 0.0018 (cyclops q8_0), 0.0041
(dragon). The sign flips between subjects and the magnitudes are at the seed spread, so **no
quality gain is claimed**; ss_res=64 is geometrically correct and memory-safe, not better. This
matches the upstream #193 measurement. `--ss-res` therefore stays opt-in with 32 as the default
(and the Studio/server do not expose it).

### 7.2 Browser WebGPU (Chrome 153 on Windows 11, RTX 4090 via D3D12, q8_0 set)

`web/real_e2e` real-input full E2E (live SS → Shape-512 → Shape-1024 → live texture conditioning →
Texture flow → decoders → bounded tail → textured GLB), cyclops, seed 42, driven by
`run_playwright.js` with `PIXAL3D_CHROME_CHANNEL=chrome`. The adapter reports **2 048 MB** — the
~2 GiB binding class; every `[cond-mem]` plan stayed below it (largest 1 296 MiB, the
single-graph NAF windows).

The pre-existing `DitRunner::check_device_budget` treats that adapter number as device memory and
refused the Shape flows at **both** ss_res (N ≈ 17.5 k needs 2 670 MB > 2 048 MB). On a 24 GB
discrete GPU that guard is wrong (it was built for Mac unified memory, where Chrome reported
~4 095 MB); these runs set its documented escape hatch `TRELLIS_ALLOW_OVER_BUDGET=1` through the
new harness hook `PIXAL3D_WASM_ENV`. Tracked as a follow-up; not an ss_res issue.

| | ss_res=32 | ss_res=64 (before the coords-only upsample fix) | ss_res=64 (after) |
|---|---|---|---|
| SS voxels / HR tokens | 4 369 / 17 423 | 18 015 / 17 563 | 18 015 / 17 563 (same) |
| wasm heap grown after Shape-512 | 1 324 MB | **3 972 MB** (of 4 096) | **2 250 MB** |
| tail: live at start → rung | 750 MiB → 1024 over (4 320), **768** (3 077) | 745 MiB → 1024 over (4 243), **768** (3 024) | 745 MiB → same choice, **768**; overall heap high-water 3 288 MB (was 3 972) |
| result | `REAL_FULL_E2E_RESULT: OK`, 1 297 s | OK, 1 775 s | OK, 1 759 s |
| silhouette IoU (native CUDA f16 same seed) | 0.9777 (0.9798) | 0.9844 (0.9833) | 0.9844 (geometry identical to the "before" GLB at the sampling noise floor) |

The ss_res=64 run exposed a host-heap risk *outside* the tail: `shape_upsample` computed and read
back the last C2S's `[64, M]` features (M ≈ 4.8 M at ss_res=64) only to discard them, growing the
wasm heap to 3 972 of 4 096 MB. `sparse_c2s(..., coords_only)` now stops after the subdivision
mask on that last stage (identical coords, native max RSS of the fixture test 3.43 → 1.78 GB).

## 8. Not run / known limits

- **Metal and native WebGPU (Dawn) were not run** in this branch: the Mac holds no GGUFs (disk
  at ~2 GB free). Both use the same C++ path as CUDA/wasm; per the probe-first rule they are not a
  gate here, but no Metal/Dawn number above exists.
- **Only the ~2 GiB WebGPU binding class was exercised on hardware** (Chrome 153 on the RTX 4090
  reports 2 048 MB, §7.2). The ~4 GiB class is covered by the plan unit test only.
- No per-stage browser-vs-native parity at ss_res=64 (the browser run is an E2E gate); the
  stage parity is native CUDA vs PyTorch (§4).
- The tail model (§6) is fitted on one subject at two ss_res. Another subject's remesh-to-decode
  face ratio can differ; the `std::bad_alloc` step-down and the explicit failure are the backstop,
  and the log shows every rung's prediction so a miss is visible.
- Quality evidence is two subjects × 2–3 seeds (§7.1); it supports "not worse beyond seed noise
  on cyclops, slightly worse on dragon", not a general statement.
- `DitRunner::check_device_budget` misreads a WebGPU adapter's per-buffer limit as device memory
  on discrete GPUs (§7.2); follow-up issue, unchanged here.
- `--ss-res` is CLI-only (Pixal3D `--views` / `--sv-image`); `trellis-server`/Studio do not expose it.
- Smaller WebGPU `maxBufferSize` adapters (C2S tiling, #42) are a separate issue and unchanged.
