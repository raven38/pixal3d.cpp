# Pixal3D WebGPU/WASM Memory Requirements

Analysis only — no code changed, pipeline not executed (GPU box busy). Derived from the native
CUDA run log (RTX 4090, 4 views, `--res 1024`, `docs/PIXAL3D_ROADMAP.md` M5 benchmark: 11.5 min
wall, 9.9 GB host RSS peak) plus arithmetic from `src/trellis_cli.cpp` (`trellis_run_mv`),
`src/flow_runner.cpp`, `src/dit.cpp`, `src/pixal3d_cond.cpp`, `src/naf.cpp`/`naf_gpu.cpp`,
`src/ss_decoder.cpp`, `src/shape_decoder.cpp`, `src/sparse.cpp`, `src/dual_grid.cpp`,
`src/remesh_dc.cpp`, `src/decimate_qem.cpp`, `src/uv_bake.cpp`, `src/mesh_glb.cpp`, and
`src/trellis_model.cpp` (`Model::load`). GPU peak was never measured natively; every GPU number
below is derived from tensor shapes in code, not profiled.

## 1. WebGPU runtime limits to query

**Measured 2026-09-06** in Google Chrome 152.0.7977.82 (macOS 26.5) on the Apple M1
Pro GPU (Metal 4, Dawn's Metal backend), via the browser WASM/WebGPU smoke harness
(`docs/spec/31-webgpu-bringup.md` §8): `web/smoke/index.html` → `pixal3d_smoke.wasm`'s exported
`webgpu_smoke_limits()` (its own independent `wgpu::Adapter`/`wgpu::Device` request, since the
ggml WebGPU backend keeps its own adapter/device private) plus a JS-side
`navigator.gpu.requestAdapter()` cross-check for the same fields. Both sides agree on every field
that exists on both APIs (`device.limits` in C++/emdawnwebgpu vs. `adapter.limits` in JS); the
`measured:` column below is the C++-side (`webgpu_smoke_limits()`) value, which is what the
actual ggml WebGPU backend will see.

| limit | spec default (guaranteed floor) | typical Chrome/Dawn desktop | measured: |
|---|---|---|---|
| `maxBufferSize` | 268,435,456 (256 MiB) | often ~2 GiB (2,147,483,648), backend-dependent | **4,294,967,292** (~4 GiB − 4B, i.e. `UINT32_MAX` rounded to a multiple of 4) |
| `maxStorageBufferBindingSize` | 134,217,728 (128 MiB) | often ~1–2 GiB, backend-dependent | **4,294,967,292** (same ~4 GiB ceiling as `maxBufferSize` on this adapter) |
| `maxComputeWorkgroupStorageSize` | 16,384 (16 KiB) | 32,768–49,152 (32–48 KiB) | **32,768** (32 KiB) |
| `maxComputeInvocationsPerWorkgroup` | 256 | 1024 | **1024** |
| `maxComputeWorkgroupSizeX` | 256 | 1024 | **1024** |
| `maxComputeWorkgroupSizeY` | 256 | 1024 | **1024** |
| `maxComputeWorkgroupSizeZ` | 64 | 64 | **64** |
| `maxComputeWorkgroupsPerDimension` | 65,535 | 65,535 (rarely higher) | **65,535** |
| `maxBindGroups` | 4 | 4 (rarely raised) | **4** |
| `maxStorageBuffersPerShaderStage` | 8 | 8–10 | **10** |
| `maxUniformBufferBindingSize` | 65,536 (64 KiB) | 65,536 (64 KiB), sometimes higher | **65,536** (64 KiB, exactly the spec floor) |

Both `maxBufferSize` and `maxStorageBufferBindingSize` land far above the "typical desktop"
estimate on this adapter (~4 GiB vs. the ~2 GiB anecdote) — **this single-adapter measurement
does not generalize**: it is Apple M1 Pro/Metal via Dawn in Chrome 152 specifically, not a cross-
vendor result. `shader-f16` and `subgroups` (needed by `ggml_backend_webgpu_init`'s hard
`ShaderF16` requirement and by the FlashAttention tile path, §4) are both **supported at the
adapter level and enabled on the created device** on this adapter/browser combination (confirmed
by `webgpu_smoke_limits()`'s device-feature check, not just the adapter-feature check — see
§8 for the full captured output). `maxUniformBufferBindingSize` is the one limit that sits
exactly at the spec-guaranteed floor rather than above it — code that assumes "typical desktop"
headroom for uniform buffers specifically should not, at least on this adapter.

`maxBufferSize`/`maxStorageBufferBindingSize` are the binding constraints (§3): even though this
adapter measured far above the "typical" ~2 GiB desktop figure, that is a single-vendor/single-OS
data point, not a spec guarantee — mobile and some Vulkan/ANGLE paths sit far closer to the
256 MiB / 128 MiB floor, so the browser target must still degrade gracefully to the spec floor,
not assume this measurement holds on other hardware.

## 2. Per-stage memory, derived from code

### 2.1 Weights (per `docs/PIXAL3D_ROADMAP.md`/task-given GGUF sizes; f32 = 2×f16)

| model | f16 | Q8_0 (est.) | Q4_0 (est.) |
|---|---|---|---|
| `dinov3.gguf` | 0.607 GB | ~0.33 GB | ~0.18 GB |
| `pixal3d_naf.gguf` | 1.3 MB | ~1 MB | ~0.6 MB |
| `pixal3d_ss_flow_mv.gguf` | 2.68 GB | ~1.5 GB | ~0.8 GB |
| `pixal3d_shape_flow_{512,1024}_mv.gguf` | 2.78 GB each | ~1.55 GB each | ~0.85 GB each |
| `pixal3d_tex_flow_1024_mv.gguf` | 2.78 GB | ~1.55 GB | ~0.85 GB |
| `ss_dec.gguf` | 0.147 GB | ~0.12 GB | ~0.09 GB |
| `shape_dec.gguf` / `tex_dec.gguf` | 0.949 GB each | ~0.78 GB each | ~0.62 GB each |
| **sum, all 8 stage models live at once** | **~14.1 GB** | **~7.6 GB** | **~4.4 GB** |

`tools/quantize_gguf.py` only requantizes 2D matmul weights whose ne0 is block-aligned
(`Q8_0`/`Q5_0`/`Q4_0`); conv/patch/norm tensors (4D/5D conv3d, GroupNorm, LayerNorm, biases)
stay F16/F32. The DiT flow checkpoints are almost entirely 2D linears (`to_qkv`, `to_out`,
`to_q`/`to_kv`, `mlp.0`/`mlp.2`, `proj_linear`) so they compress close to the block-size ratio
(Q8_0 ≈ 53%, Q4_0 ≈ 28% of f16 bytes); the decoders (`shape_dec`/`tex_dec`) carry more Conv3d
weight mass (`src/sparse.cpp`'s `sparse_submconv`/`sparse_convnext`, `[Ci,27,Co]` per tap) that
does *not* quantize, so their compression is shallower (est. ~80%/~65%). **Note: `src/trellis_model.cpp::Model::load`
allocates one `ggml_backend_alloc_ctx_tensors` buffer per GGUF and streams the whole file into
it before any compute runs** — there is no partial/lazy load today; "weight streaming" (§4) is a
new capability, not something already exposed by the loader.

Because `trellis_run_mv` frees each `Model` immediately after its stage (`m.free()`; see
`trellis_cli.cpp` lines 241–403), **peak weight residency is one stage's model, not the sum** —
at f16 that is 2.78 GB (a SLAT flow checkpoint), well inside even 8 GB of GPU memory.

### 2.2 Persistent activations per stage (host+GPU tensors that outlive the stage)

| stage | tokens (N) | d_model | persistent state |
|---|---|---|---|
| SS (dense) | 4096 (16³) | 1536 | latent `[8,4096]` = 131 KB; global cond `[5,1024]`; proj cond `[4096,1024]` dense = 16.8 MB |
| Shape 512 (LR) | 4394 voxels | 1536 | latent `[32,4394]` = 562 KB; proj gathered `[4394,2048]` = 36 MB (dense source 268 MB, R=32, freed after gather) |
| Shape 1024 (HR) | 17,484 tokens (grid 64; budget 49,152) | 1536 | latent `[32,17484]` = 2.2 MB; proj gathered `[17484,2048]` = 143 MB (dense source 2.15 GB, R=64, freed after gather) |
| Texture 1024 | 17,484 tokens | 1536 | input `[64,17484]` (32 tex + 32 shape-slat concat) = 4.5 MB; same 2.15 GB dense proj source |
| Shape/tex decode | up to ~4.6M voxels (measured final mesh density) | 1024→64 (4 ConvNeXt+C2S stages) | feats7 `[7,4.6M]` = 129 MB (shape); pbr6 `[6,~M]` similar (tex) |

The dense `[R³,2048]` proj grids (268 MB at R=32, 2.15 GB at R=64 — matches the task's given
numbers) are built once per view in `pixal3d_cond_slat` (host `std::vector<float>`, `src/pixal3d_cond.cpp`)
and only ever read at the sparse coords via `pixal3d_gather_proj`; the dense buffer is then
`.clear()`+`.shrink_to_fit()`'d in `trellis_cli.cpp` (lines 274–276, 330–331, 379–380) —
i.e. the code already treats the R=64 dense grid as a transient, not a stage-persistent tensor.

### 2.3 Single largest transient tensor per stage

- **DiT MLP hidden `[8192, N]` f32** (`src/dit.cpp::block`, `mlp.mlp.0`→GELU→`mlp.mlp.2`, **not
  chunked** — unlike the sparse decoders): SS N=4096 → 134 MB; shape-512 N≈4394 → 144 MB;
  **shape/tex-1024 N=17,484 → 573 MB**; at the 1536-cascade's `max_tokens=49,152` cap → **1.61 GB**.
  This is the single biggest per-op DiT tensor and the one most likely to blow `maxBufferSize`.
- **FA-padded K/V** (`src/dit.cpp::sdpa`, `prep_kv`): `[head_dim=128, Lk_pad, n_heads=12]` BF16,
  Lk_pad rounded to 256: HR self-attn Lk_pad=17,664 → 128×17,664×12×2 B ≈ **54 MB** each for K
  and V (small; FA's whole point is O(N), not O(N²)).
- **Exact (`--no-fa`) score chunk `[Lk, nq, nh]`** (`kAttnChunkBytes` in `dit.cpp`): deliberately
  capped at **1 GiB** by the existing chunking loop, regardless of N — already a WebGPU-shaped
  design, just tuned to native VRAM, not to a 256 MiB buffer cap.
- **NAF map `[1024, T, T]` f32** (`src/naf.cpp::naf_upsample`, `naf_na2d`'s `out`): **1 GB at
  T=512, 4 GB at T=1024** per view (given). The CPU path additionally holds `cat` (256×Sp×Sp,
  1.07 GB at Sp=1024), `pooled`/`k_up` (256×T×T, 1.07 GB each at T=1024) and `v_up` (1024×T×T,
  4.3 GB at T=1024) *simultaneously* — **~11 GB of coexisting host buffers for one NAF call at
  T=1024**, all in `naf.cpp`'s `std::vector<float>`s. `naf_gpu.cpp`'s CUDA path avoids materializing
  `k_up`/`v_up` (custom kernel indexes the pooled maps directly, per its own header comment and
  spec 30 §4's "on-demand evaluation... avoids materialising the 1024×T×T map"), but still
  round-trips `enc_pooled`/`k_pooled` through the host for the CPU-reused RoPE/pool ops.
- **Sparse decoder C2S stage `[Cout×8, N_in]`** (`src/sparse.cpp::sparse_c2s`, pre-gather conv1
  output): code's own comment cites **10.1 GB measured** for one stage on a denser reference
  object (M=32.7M final voxels); our measured run's final mesh (~4.6M voxels, ~14% of that
  density) scales (not measured) to roughly **1–1.5 GB** for the same stage — still one of the
  largest GPU allocations in the whole pipeline, and already chunk-mitigated by `kBlockChunkBytes`
  (1.5 GB budget) + the ggml_pad-into-one-buffer trick (avoiding a 2× concat-chain peak).
- **Dense proj grid `[R³, 2048]`** (§2.2): 2.15 GB at R=64, host-only, freed right after gather.

### 2.4 Host (WASM heap) needs: topology/geometry

Per `docs/PIXAL3D_ARCHITECTURE.md`'s CPU/WASM split, these stay on the host:

- Sparse coord hashing / neighbor tables: `build_neighbor_table` (`src/sparse.cpp`) is `[27, N]`
  int32 = 108 B/voxel; at the measured decode density (~4.6M final voxels) ≈ **497 MB**, and at
  the code's own worst-case "cottage" M=32.7M ≈ **3.5 GB** — this alone approaches the 4 GB
  wasm32 ceiling on a dense object.
- Mesh arrays: `dual_grid_to_mesh` verts `float[3,N]` + faces `int32[3,~2N]`; at the measured
  ~9.3M pre-remesh faces / ~4.6M voxels ≈ 4.6M×12 B (verts) + 9.3M×12 B (faces) ≈ **167 MB**.
  After `remesh_narrow_band_dc` (11.8M active voxels → V=9.15M, F=18.3M, per the task's given
  numbers): verts 9.15M×12 B + faces 18.3M×12 B ≈ **330 MB**, plus the `res³` narrow-band bitset
  (`cand`, `res=1024` → 1024³ bits = 128 MB) and the `acoord`/`vcoord`/`dual`/`owned` int/float
  arrays sized to `Na`≈11.8M (≈ 850 MB combined, freed before decimation).
- `decimate_qem` (`src/decimate_qem.cpp`, pure host/CPU, no GPU path referenced when built
  without `TRELLIS_HAVE_GPU_DECIMATE`/`_VK_DECIMATE`): per-round CSR adjacency + per-edge QEM
  (`float[10]` = 40 B/vertex) + edge/cost/prop arrays sized to F≈18.3M → **on the order of 1–2 GB**
  host RAM during simplification alone (not counted in the measured 9.9 GB RSS breakdown, since
  that run used the default decimation target and CUDA GPU decimate path natively).
- `uv_bake`/`mesh_glb`: atlas raster buffers `T×T×4` bytes (base+mr+mask+zbuf) at the MV default
  `T=4096` → base 64 MB + mr 64 MB + mask 16 MB + zbuf 64 MB ≈ **208 MB**; GLB serialization
  (`mesh_glb.cpp`) builds `pos`/`nrm`/`uv` f32 arrays sized to final V (post-decimation, ≤1M
  faces target) — small once decimated.

**Host RSS total**: the measured native run hit 9.9 GB peak; the topology/geometry pieces above
(neighbor tables, mesh arrays, narrow-band bitset, atlas buffers) are a meaningful fraction of
that and are exactly the "coordinates/topology" category `docs/PIXAL3D_ARCHITECTURE.md` assigns
to CPU/WASM — i.e. **the browser's WASM heap must budget ~1–2 GB even before any neural
activation**, on top of whatever stays resident from §2.2/§2.3 if it round-trips through host
memory (§5).

## 3. WebGPU constraint comparison

Using the "typical Chrome/Dawn desktop" figures from §1 (maxBufferSize ≈ 2 GiB,
maxStorageBufferBindingSize ≈ 1–2 GiB) as the optimistic case, and the **spec-guaranteed floor**
(256 MiB / 128 MiB) as the pessimistic case a browser target must not silently assume away:

| tensor | size | vs. spec floor (256/128 MiB) | vs. typical desktop (~2 GiB) |
|---|---|---|---|
| DiT MLP hidden, HR (N=17,484) | 573 MB | **exceeds both** floors (2.2×/4.5×) | fits |
| DiT MLP hidden, 1536-cascade cap (N=49,152) | 1.61 GB | **exceeds both** floors | fits, tight |
| NAF map, T=512 | 1 GB/view | **exceeds both** floors | fits |
| NAF map, T=1024 | 4 GB/view | **exceeds both floors, and the typical-desktop figure** | **exceeds** |
| NAF `v_up`/`out`, T=1024 (CPU-path host buffer) | 4.3 GB | N/A (host, not a GPU buffer) | — |
| Sparse decoder C2S stage (measured, dense object) | 10.1 GB | **exceeds everything** | **exceeds** |
| Sparse decoder C2S stage (our density, est.) | ~1–1.5 GB | **exceeds both** floors | fits |
| Dense proj grid R=64 | 2.15 GB | **exceeds both** floors | borderline/exceeds |
| Exact-path attn chunk (as tuned today) | 1 GiB | **exceeds both** floors | fits |

**Verdict**: at the spec-guaranteed floor (the only thing a conformant browser must offer), *every
transient above 256 MiB/128 MiB needs splitting* — that is nearly everything above except the
FA-padded K/V and the gathered per-token proj tensors. Even against the optimistic "typical
desktop" ~2 GiB figure, the NAF map at T=1024 and the C2S decoder stage on denser objects still
exceed it and must be tiled or evaluated on-demand regardless of adapter generosity.

**wasm32 (4 GiB heap) vs memory64**: the NAF CPU-path `v_up`/`out` buffers alone (4.3 GB at
T=1024) exceed a 4 GiB wasm32 linear memory by themselves, before any other allocation — this is
a **hard wasm32 blocker for the CPU NAF fallback at T=1024**, not a tuning problem. The GPU NAF
path (`naf_gpu.cpp`-style, keeping k/v un-materialized) avoids it, making **GPU-resident NAF
mandatory for wasm32**, or **memory64** mandatory if a host-materialized NAF map must ever exist
at T=1024. Neighbor tables (up to 3.5 GB on dense objects, §2.4) plus mesh/remesh arrays
(~1.2 GB combined) plus decimation's ~1–2 GB put the *topology-only* host budget within 2–3× of
the wasm32 ceiling even before decoder feature tensors — dense objects are at real risk of
exceeding wasm32's 4 GiB address space without memory64.

## 4. Required strategies — verdicts for an 8 GB / 16 GB GPU browser target

| strategy | verdict (8 GB GPU) | verdict (16 GB GPU) | why |
|---|---|---|---|
| Stage-wise model load/unload | **mandatory** | **mandatory** | Already how `trellis_run_mv` works natively (§2.1); without it the ~14 GB weight sum alone exceeds 8 GB and is close to 16 GB. No new design needed, just port the existing load/free pattern to WebGPU buffer lifetimes. |
| Weight streaming (partial/lazy load) | optional | optional | `Model::load` already streams sequentially from disk into one pre-allocated buffer (§2.1) — good enough once stage-wise unload is in place; true per-layer streaming only helps first-paint latency, not peak memory. |
| Buffer reuse (ggml_gallocr-style pooling) | **mandatory** | **mandatory** | ggml's own allocator already does this natively; the WebGPU backend must replicate it — without pooling, a 30-block DiT graph's per-block temporaries (§2.3) would each need a separate `GPUBuffer`, hitting `maxStorageBuffersPerShaderStage` (8, spec floor) almost immediately. |
| View-sequential conditioning | **mandatory** | **mandatory** | Already implemented (`pixal3d_cond_ss`/`pixal3d_cond_slat` loop views and average in place, `docs/PIXAL3D_ARCHITECTURE.md`); without it, V views' NAF maps/proj grids (§2.3) would need to coexist. |
| Temporary tensor lifetime optimization (the `.clear()+.shrink_to_fit()` pattern already in `trellis_cli.cpp`) | **mandatory** | **mandatory** | Directly determines whether the 2.15 GB dense proj grid (R=64) is transient or persistent; on 8 GB this is the difference between fitting and not. |
| On-demand NAF evaluation at grid points | **mandatory** | recommended | Spec 30 §4 already identifies this as exact (only R³×4 bilinear-corner points are ever read); current code still materializes the full map (§2.3, up to 4 GB/view at T=1024) — this is the single highest-value change for the browser target and is not optional at 8 GB. |
| Quantized weights (Q8_0/Q4_0) | **mandatory for full 1024 cascade at 8 GB**; optional at 16 GB | optional | Per-stage weight residency (2.78 GB f16 peak) already fits 8 GB alone, but stacked against §2.2/§2.3 activations plus §2.4's host-adjacent GPU staging, f16 leaves little headroom; Q8_0 (~1.55 GB/stage) gives real margin. ggml-WebGPU's Q8_0/Q4_0 dequant kernel support is unverified (`GGML_WEBGPU` doesn't exist on this fork yet) — treat as a dependency to confirm, not an assumption. |
| Reduced token budgets (`max_tokens`, lower `hr_res`) | optional (fallback) | optional | The existing backoff loop (`trellis_cli.cpp`'s `hr_res -= 128` cascade) already exists for VRAM reasons; reusing it as a browser-tier control (e.g. cap at res-512 shape only) is the cheapest lever if the above aren't enough, but degrades output quality/gate metrics. |

**Realistic first browser target**: **SS + shape-512 only, textured or untextured, single view**.
Rationale: shape-512's largest transients (144 MB MLP hidden, 268 MB dense proj grid, 36 MB
gathered proj) all fit comfortably under even the spec-floor 256 MiB/128 MiB limits without
splitting; the 1024 HR shape/tex stages need at least on-demand NAF (mandatory) and MLP-hidden
tiling (573 MB tensor split across `maxStorageBufferBindingSize`-sized bindings) before they are
WebGPU-safe at any adapter, let alone the spec floor. Full 1024 cascade with Q8_0 weights is a
plausible **second** milestone once NAF on-demand evaluation and MLP/attention tensor splitting
land — it is not realistic as the first working end-to-end path.

## 5. GPU→CPU→GPU round trips in the current native code

These are the places `trellis_run_mv`/`flow_runner.cpp`/`pixal3d_cond.cpp` read a GPU tensor back
to a host `std::vector<float>` and later re-upload it — each is a synchronization point that
stalls a WebGPU command queue (no async pipelining) and, in a browser, forces an `await` on
`mapAsync`/`readBuffer`:

1. **`sample_flow`'s Euler loop** (`src/flow_runner.cpp`): `sample` (the latent) lives in a host
   `std::vector<float>`; every step calls `fwd(sample, ...)` → `DitRunner::forward` uploads it
   (`ggml_backend_tensor_set`), runs the graph, then reads the *entire* output back to host
   (`tensor_to_f32(gout_)`) to do the Euler update (`sample[k] -= (t-tprev)*pred[k]`) in plain
   C++. For 12 steps × (1–2 forwards for CFG) this is up to 24 full latent round-trips per stage.
   **Must change**: the Euler update (elementwise, trivially a compute shader) should run as a
   ggml op on the latent buffer in place, keeping it GPU-resident for the whole sampling loop —
   only upload conditioning/noise once and read back once at the end.
2. **Conditioning built on host** (`src/pixal3d_cond.cpp`): DINOv3 output → `tensor_to_f32`
   (host) → `patch_tokens_to_chw` (host reshape) → `proj_grid_sample` (host CPU projection,
   `src/proj_grid.cpp`) → re-uploaded as `gcond_`/`gproj_` inputs to the DiT graph. Every view's
   global+proj condition takes a full DINOv3-latent GPU→host round trip, a host-side
   reshape/projection pass, then a host→GPU upload into the flow model. **Must change**: either
   port `proj_grid_sample`'s bilinear grid-sample to a WGSL compute kernel operating directly on
   the DINOv3 output buffer (keeping DINO features GPU-resident), or accept this one round trip
   per view as unavoidable if projection stays a CPU/WASM op per
   `docs/PIXAL3D_ARCHITECTURE.md`'s split — but then it must be pipelined across views (start
   view v+1's DINOv3 forward while view v's projection runs on the host) rather than serialized.
3. **NAF output read back** (`src/naf.cpp`/`naf_gpu.cpp`): the GPU path already keeps the
   encoder's conv/pool stack on-device but reads `enc_pooled` back to host (`tensor_to_f32(pooled)`)
   to run RoPE and the k-adaptive-pool on the CPU reference implementation, then hands the
   host-side `enc_pooled`/`k_pooled`/`lr` buffers into `naf_attn_cuda` (a *native CUDA* kernel
   call, not a ggml op) — this whole path has no WebGPU equivalent yet (`naf_attn.cu` is CUDA-only;
   Vulkan/Metal already fall back to the slow CPU path per spec 30 §4). **Must change**: this is
   the single largest porting gap for the browser target — RoPE, adaptive pooling, and the
   neighborhood-attention kernel all need WGSL implementations that never leave the GPU, both to
   avoid the round trip and (per §3) to avoid ever materializing the T×T map that motivated the
   CUDA kernel in the first place.
4. **`Model::load`/`Model::free()` per stage** (`src/trellis_model.cpp`): not a within-stage
   round trip, but each stage's `Model::load` creates a *new* `ggml_backend` and
   `Model::free()` destroys it (`ggml_backend_free`) — in WebGPU terms this maps to creating/
   destroying a `GPUDevice` (or at least losing its pipeline/buffer caches) eight times per
   generation. **Must change**: the browser runtime should hold one long-lived `GPUDevice` across
   stages and only free/reallocate the per-stage weight `GPUBuffer`s, not tear down the device
   context itself — recreating shader/pipeline objects per stage would add real latency that the
   native per-process model never paid.
5. **`ss_decode`/`shape_decode`/`tex_decode` segment chaining** (`src/ss_decoder.cpp`'s
   `run_seg`, `src/shape_decoder.cpp`'s `decode_unet`): each of SS-decoder's 3 segments and each
   of the 4 shape/tex ConvNeXt+C2S stages reads its output fully back to host
   (`tensor_to_f32(out)`) before the next segment/stage re-uploads it as input — `pixel_shuffle`
   between SS-decoder segments and the coordinate-dependent neighbor-table rebuild between C2S
   stages are genuinely host operations (topology changes, per
   `docs/PIXAL3D_ARCHITECTURE.md`'s CPU/WASM split), so some of these round trips are
   architecturally intentional, not incidental. **Must change**: only the *feature* tensors
   (`h`) need to survive the round trip today; once coords/neighbor-tables are computed on the
   WASM side, the feature buffer itself should stay GPU-resident and only the (much smaller)
   coordinate/neighbor int32 arrays should cross the WASM↔GPU boundary — currently the full f32
   feature buffer crosses every time.

## 6. Measured: SS stage on the ggml WebGPU backend (2026-09-06, `feat/webgpu-ss`)

Buffer-allocation accounting, not a live VRAM query (`ggml_backend_webgpu_device_get_memory()` is
a stub, §1/spec 31 §5): weights = `ggml_backend_buffer_get_size` of the model buffer, activations =
`ggml_gallocr_get_buffer_size` of the graph allocator (the size ggml actually requests from the
backend for all temporaries of one graph), conditioning = the persistent accumulator buffer.
Apple M1 Pro (32 GB unified), native Dawn 18eb229 / Chrome 152; same numbers on the RTX 4090 since
they are graph properties. Printed by `trellis-test-pixal3d-cond-ss … gpu` and
`trellis-test-pixal3d-ss-sample`.

| component | weights resident | activations / temporaries | conditioning tensors | peak (sum) | notes |
|---|---|---|---|---|---|
| DINOv3 @512, one view graph | 578.6 MB (f16 GGUF) | 88.4 MB (gallocr; largest tensor 67.8 MB = `[1029,1029,16]` f32 scores) | -- | 667 MB | `dinov3_encode`, freed after each call |
| projection + MV average (`pixal3d_cond_ss_gpu`), **1 view** | 578.6 MB | 88.4 MB (same graph + 4 `[1024,4096]` taps reuse the DINOv3 scratch) | 16.0 MB (`[1024,5]` + `[1024,4096]` f32 accumulators) | **683.0 MB** | 1.04 s/view native Dawn |
| same, **2 views** | 578.6 MB | 88.4 MB | 16.0 MB | **683.0 MB** | 2.07 s |
| same, **4 views** | 578.6 MB | 88.4 MB | 16.0 MB | **683.0 MB** | 4.14 s native; 5.6 s in Chrome |
| SS flow DiT, one forward, exact SDPA (`--no-fa`) | 2556.8 MB (f16 GGUF) | 930.1 MB (dominant: one `[4096,4096,12]` f32 score chunk = 805 MB; MLP hidden `[8192,4096]` f32 = 134 MB) | 16.0 MB (global + proj inputs, re-uploaded per forward) | **3503 MB** | 16.0 s/forward native Dawn (M1 Pro); CUDA 4090: 257 ms |
| SS flow DiT, FlashAttention (CUDA production path, for reference) | 2556.8 MB | 266.1 MB | 16.0 MB | 2839 MB | not available on WebGPU (BF16 K/V) |
| SS sampling (12 steps, 22 forwards) | as one forward | as one forward (graph reused) | 16.0 MB ×2 (cond + zero neg) | **3503 MB** | 351 s native Dawn; CUDA 5.7 s (exact) / 4.1 s (FA) |

Findings:

- **View-sequential conditioning verified**: the per-view graph is allocated and freed inside the
  view loop, so the 1/2/4-view peaks are identical (683.0 MB); only wall time scales with V. The
  alternative (all views' DINOv3 graphs alive at once) would add 88.4 MB per extra view -- small at
  S=512, but the same loop will carry S=1024's ~1.1 GB score tensors later.
- **Peak of the whole SS stage is the flow, not conditioning**: DINOv3 is freed before the flow
  weights load (`pixal3d_wasm.cpp`, mirroring `trellis_run_mv`'s per-stage `Model::free`), so the
  stage peak is 3.5 GB, of which 805 MB is the exact-attention score chunk. On WebGPU that single
  chunk is the first thing to shrink for a spec-floor (128 MiB binding) adapter: `kAttnChunkBytes`
  at 64-96 MiB would cut the activation buffer to ~250 MB with no numerical change (query chunking
  is exact), at the cost of ~40 more `CONCAT` nodes per attention (§5 B2).
- **Weights are the floor**: 2.56 GB of f16 flow weights sit in one `GPUBuffer` (under this
  adapter's 4 GiB `maxBufferSize`; a 256 MiB-floor adapter would need `Model::load` to split the
  weight buffer -- `ggml_backend_alloc_ctx_tensors` already does that per `max_size`, untested here).
- **Browser (Chrome 152, WORKERFS-mounted GGUFs)**: the 2.68 GB flow GGUF streams from the on-disk
  `File` into the WebGPU buffer in 4.7 s without a copy in the wasm heap (per-tensor `fread` through
  FileReaderSync into a 25 MB staging vector); wasm heap stays well under 1 GB for the SS stage.

## 7. Measured: Shape-512 stage on the ggml WebGPU backend (2026-09-06, `feat/webgpu-shape512-flow`)

Same accounting as §6 (buffer allocations, not a live VRAM query): weights =
`ggml_backend_buffer_get_size` of the model buffers, activations = the gallocr buffer of one graph,
conditioning = the persistent accumulators / the DiT's per-forward condition inputs. Apple M1 Pro,
native Dawn 18eb229 / Chrome 152. Printed by `trellis-test-naf --ggml`,
`trellis-test-pixal3d-cond-slat --stage gpu`, `trellis-test-pixal3d-slat-sample` and
`pixal3d_shape512_run`.

| component | weights resident | activations / temporaries | conditioning tensors | peak (sum) | notes |
|---|---|---|---|---|---|
| DINOv3 @512, one view | 578.6 MB | 88.4 MB | -- | 667 MB | as §6 |
| NAF alone, S=512 → T=128 (`naf_upsample_ggml`, test fixture) | 1.3 MB | 919.4 MB (S=512 encoder: `[512,512,128]` f32 conv outputs, 134 MB each, up to ~6 alive; direct conv, no im2col) | inputs 15.4 MB (image, RoPE tables `[16384,64]`×2, indices) | 936 MB | output `[1024,16384]` 64 MB; 4.4 s |
| NAF alone, S=512 → T=512 (inside the conditioning graph) | 1.3 MB | see next row | RoPE tables `[262144,64]` f32 × 2 = 134 MB, window index 0.3 MB, block index 1 MB | | output `[1024,262144]` = 1 GB, produced twice (attention output + its channel-major transpose) |
| **Shape-512 conditioning, one view graph** (`pixal3d_cond_slat_gpu`), V=1 and V=4 | 579.9 MB (DINOv3 + NAF) | **2837.8 MB** (gallocr; dominated by NAF at T=512: the two 1 GB `[1024,262144]` maps, the 340 MB `[81,256,4,1024]` logits/probabilities and gathered values; DINOv3's 88 MB reuses the same buffer) | 256.0 MB (`[1024,32768]` lr + hr accumulators + global) | **3673.8 MB** | flat in V (per-view buffer freed before the next view); 7.6 s (V=1) / 31.9 s (V=4, 11.9 s slowest view) native Dawn |
| host path for comparison (`pixal3d_cond_slat` on WebGPU: NAF on device, projection on host) | 579.9 MB | NAF graph alone | 268 MB host `[32768,2048]` + 1 GB host NAF map per view | | 56 s for V=4 (1 GB readback + host bilinear sampling per view) |
| Shape-512 flow DiT, one forward, exact SDPA (`--no-fa`), N=4377 | 2646.9 MB (f16 GGUF) | **1067.8 MB** (gallocr; dominant: one `[4377,4377,12]` f32 score chunk = 920 MB, `[8192,4377]` MLP hidden 143 MB) | 36.1 MB (`[2048,4377]` proj + `[1024,5]` global, re-uploaded per forward) | **3751 MB** | 21-24 s/forward; CUDA 4090: 0.11 s (FA) / 0.16 s (exact) |
| Shape-512 sampling (12 steps, 20 forwards) | as one forward | as one forward (graph reused) | 36.1 MB ×2 (cond + zero neg) | **3751 MB** | 423 s native Dawn, 473 s Chrome; CUDA 2.2 s (FA) / 3.2 s (exact) |

Findings:

- **The two stage peaks are close**: conditioning 3.67 GB (DINOv3+NAF weights 580 MB + 2.84 GB
  graph) and flow 3.75 GB (2.65 GB weights + 1.07 GB graph); DINOv3/NAF are freed before the flow
  weights load (`pixal3d_wasm.cpp`, mirroring `trellis_run_mv`), so they never coexist.
- **The conditioning graph is the NAF map**: 3.67 GB, of which 2 GB is the NAF
  map materialized twice (`[C/4, d², 4, blk]` mul_mat output and its `[C, T·T]` channel-major
  copy that the projection `get_rows` consumes). Spec 30 §4's on-demand evaluation (only the
  R³×4 = 131k tap pixels of the 262k are ever read at R=32) or a block-chunked attention with the
  taps gathered per chunk would cut this to ~1 GB with no numerical change; neither is needed on
  this 4 GiB-limit adapter and both are deferred.
- **Sequential view processing holds**: V=1 and V=4 have the same peak; only wall time scales.
- **Spec-floor adapters**: the 1 GB maps, the 920 MB attention score chunk and the 2.78 GB weight
  buffer all exceed the 128 MiB / 256 MiB floors; nothing in this phase changed that (§4's verdicts
  stand).
- **Browser (Chrome 152, WORKERFS-mounted GGUFs)**: identical buffer numbers to native (peak
  3673.8 MB cond / 1067.8 MB DiT activations, same graphs); the 2.78 GB flow GGUF streams into its
  `GPUBuffer` in 4.8 s; the wasm heap holds the fixture arrays (the 257 MB `s512_z_proj.npy`
  reference for the parity print is the largest) and stays under 1 GB. `-sMAXIMUM_MEMORY` is
  4 GiB, unchanged.

## 8. Measured: sparse shape-decoder allocation trace (2026-09-06, `feat/webgpu-sparse-backend-prep`)

Backend-preparation task, independent of the Shape-1024 model work; the decoder itself is
unchanged. `trellis_graph_alloc_trace` (`src/graph_dump.cpp`, declared in
`include/graph_dump.h`) is called from the two sparse-decoder graph runners
(`shape_decoder.cpp::run1`, `sparse.cpp::GraphRun::run`) right after `ggml_gallocr_alloc_graph`
and is a no-op unless env `TRELLIS_DBG_ALLOC_TRACE` is set. For every graph it prints the
graph-allocated tensors (input leafs and non-view nodes -- what gallocr places; weights and views
excluded) at or above `TRELLIS_DBG_ALLOC_TRACE_MIN_MB` (default 64; `=all` prints every one) as
`op / name / shape / dtype / bytes`, the ten largest, and a summary line: sum of allocations,
largest single allocation, a **simultaneously-live estimate** (interval sweep: each tensor is live
from its producing node to its last consumer, inputs and the output for the whole graph; gallocr's
in-place reuse is not modelled, so it is an upper bound) and gallocr's actual buffer size.
`trellis-shape-dec-alloc-trace <shape_dec.gguf> <coords.npy> <slat.npy> [gpu] [res]`
(`src/shape_dec_alloc_trace.cpp`) runs `shape_decode` on any coords/latent pair with the trace
on. No-behavior-change check: `trellis-test-pixal3d-shape-decode` on the `slat_sample` fixture
with and without the env gives byte-identical metrics (V=1188470 F=2383624, worst symmetric mean
2.4704e-06, PASS); the trace only adds stderr lines.

Run on the RTX 4090 (CUDA build) with the real Pixal3D `shape_dec.gguf` (0.95 GB f16) on the
**Shape-1024 latent** (`hr_sample`: `hr_coords.npy` N = 17,489 voxels at res 64 →
M = 4,649,837 voxels at res 1024, i.e. the density the op-gap trace saw), stage graphs in
execution order (`largest` = largest single allocation; `live est.` = the sweep; `gallocr` =
measured buffer):

| graph (tag) | nodes | allocs | largest single allocation | live est. | gallocr | Σ allocs |
|---|---|---|---|---|---|---|
| `from_latent` N=17489 | 2 | 3 | 71.6 MB `[1024,17489]` f32 (ADD) | 0.14 GB | 0.07 GB | 0.15 GB |
| ConvNeXt stage 0 (C=1024, 4 blocks) N=17489 | 812 | 590 | 287 MB `[4096,17489]` f32 (MLP hidden, SILU) | 0.65 GB | 0.51 GB | 29.0 GB |
| C2S stage 0 conv N=17489 → M=71261 | 402 | 291 | 287 MB `[4096,17489]` f32 (conv1 tap ADD) | 1.01 GB | 0.88 GB | 31.0 GB |
| ConvNeXt stage 1 (C=512, 16 blocks) N=71261 | 3248 | 2354 | 584 MB `[2048,71261]` f32 (MLP hidden) | 1.33 GB | 1.04 GB | 234 GB |
| C2S stage 1 conv N=71261 → M=288165 | 402 | 291 | 584 MB `[2048,71261]` f32 (conv1) | 2.08 GB | 1.79 GB | 62.5 GB |
| ConvNeXt stage 2 (C=256, 8 blocks) N=288165 | 1624 | 1178 | **1.18 GB `[1024,288165]` f32** (MLP hidden: MUL_MAT/ADD/SILU ×8 blocks, 24 allocations ≥ 1 GB) | 2.69 GB | 2.10 GB | 237 GB |
| C2S stage 2 conv N=288165 → M=1158936 | 602 | 430 | **1.18 GB `[1024,288165]` f32** (conv1 `[Cout·8, N]` tap MUL_MAT/ADD ×27, 54 allocations ≥ 1 GB) | 4.30 GB | 3.71 GB | 126 GB |
| ConvNeXt stage 3 (C=128, 4 blocks) N=1158936 | 1628 | 1178 | **1.57 GB `[512,768000]` f32** (MLP hidden chunk; 12 allocations ≥ 1 GB, plus `[512,390936]` 801 MB tail chunk) | 4.46 GB | 3.28 GB | 243 GB |
| C2S stage 3 subdiv N=1158936 | 8 | 7 | 593 MB `[128,1158936]` f32 (input) | 1.14 GB | 1.14 GB | 1.30 GB |
| **C2S stage 3 conv N=1158936 → M=4649837** | 1392 | 981 | **1.57 GB `[512,768000]` f32** (conv1 chunk MUL_MAT/ADD ×27 taps; 58 allocations ≥ 1 GB) | **6.97 GB** | **5.98 GB** | 254 GB |
| `output_layer` (5 × 1M-row chunks) | 3 | 4 | 256 MB `[64,1000000]` f32 (NORM) | 0.51 GB | 0.51 GB | 0.57 GB |

The tensors "estimated at ~1.2-1.6 GB" in §4/op-gap §5 are therefore exactly:

- **`[1024, 288165]` f32 = 1,180,323,840 B (1.18 GB)** -- stage 2's 4C = 1024-wide ConvNeXt MLP
  hidden (`mlp.0` MUL_MAT → bias ADD → SILU, then `mlp.2` input) in each of the 8 blocks, and
  stage-2 C2S `conv1`'s `[Cout·8 = 1024, N]` per-tap gather-matmul outputs and running-sum ADDs
  (27 taps). Both are single-chunk because `kBlockChunkBytes` (1.5 GB) / (1024 × 4 B) = 384k rows
  > N = 288,165.
- **`[512, 768000]` f32 = 1,572,864,000 B (1.57 GB)** -- the same two roles at stage 3
  (4C = 512 / `Cout·8` = 512), where N = 1,158,936 exceeds the budget and is chunked at
  1.5 GB / (512 × 4 B) = 768,000 rows (two chunks: 768,000 + 390,936, the latter 801 MB).
- **`[64, 4649838]` f32 = 1.19 GB** -- stage-3 C2S `hraw` (PAD seed, then NORM and SILU of it,
  three allocations) and **`[64, 4649837]` = 1.19 GB** the conv2 `out` PAD seed: the
  §5-B9 single-buffer pattern, sized by M, not by any chunk budget.
- Next tier: `[128, 1158937]` CONCAT (sentinel-row pad of the stage-3 input, 593 MB, five of
  them), the stage-3 input `[128, 1158936]` 566 MB, the conv2 neighbor table `[4649837, 27]` i32
  479 MB (input), 27 `[64, 1000000]` GET_ROWS gathers per conv2 chunk (256 MB), and the skip
  REPEAT `[4,16,1000000]` 256 MB.

Same run on the res-512 fixture (`slat_sample`, N = 4377 → M = 1,188,470): largest allocation
0.61 GB (`[512,295441]`, stage 3), C2S stage-3 gallocr 1.99 GB, everything else ≤ 1.09 GB.

Implications for the WebGPU port (decision input for tiling/chunking, nothing implemented here):

- **The 1.2-1.6 GB tensors are budget-driven, not structural**: every one of them is a
  `[4C, chunk]` / `[Cout·8, chunk]` intermediate whose row count comes from `kBlockChunkBytes`
  (1.5 GB, tuned for 24 GB VRAM; `TRELLIS_BLOCK_CHUNK_MB` / `TRELLIS_C2S_CHUNK_MB` already
  override it). Lowering the budget shrinks them linearly with no numerical change; the cost is
  more chunks (graph nodes: each ConvNeXt chunk ≈ 145 nodes, each C2S chunk ≈ 135, against the
  65,536 / 262,144-node contexts and the 24 / 64 chunk caps in `sparse.cpp`). On this 4 GiB-binding
  adapter they need no change at all.
- **What does scale with M and is not chunked**: the `[Cout, M+1]` PAD seed buffers (1.19 GB at
  M = 4.65M -- op-gap §10.3 verifies the PAD→CPY pattern bit-exact up to a 3 GiB seed), the
  `[C, N]` stage inputs/outputs, the `[Ci, N+1]` sentinel CONCATs and the `[M, 27]` i32 neighbor
  tables. These are the tensors that decide whether a buffer split is needed: on a spec-floor
  adapter (128 MiB binding) every one of them is over the limit from stage 2 on
  (`supports_op` rejects any node whose `ggml_nbytes` exceeds `maxStorageBufferBindingSize`), on
  this adapter none of them is.
- **Per-graph peaks**: the C2S stage-3 graph asks gallocr for 5.98 GB (live estimate 6.97 GB;
  the 14 % gap is gallocr's in-place reuse). ggml-alloc splits a graph buffer into chunks of
  `ggml_backend_buft_get_max_size` (= `maxStorageBufferBindingSize` on WebGPU), so 5.98 GB
  becomes two `GPUBuffer`s here; whether Chrome grants ~7 GB of device buffers per tab is the
  open question for the port and was not measured (the Shape-1024 sampling job holds the GPU).

## 9. Measured: sparse shape decoder on the ggml WebGPU backend (2026-09-06, `feat/webgpu-shape-decode`)

The §8 decoder, now executed on WebGPU (native Dawn, Apple M1 Pro 32 GB unified, `maxBufferSize`
= `maxStorageBufferBindingSize` = 4 GiB − 4 B; and Chrome 152) on the **real** Shape-1024 SLAT
(`hr_sample`, N = 17,489 → M = 4,649,809 on WebGPU) with the same `TRELLIS_DBG_ALLOC_TRACE`
instrumentation. Parity: `docs/spec/31-webgpu-bringup.md` §11. Process memory is
`footprint -f bytes` `phys_footprint` sampled every 2 s (it includes the Dawn/Metal
`IOAccelerator` buffers, which `ps` RSS does not: RSS reads 2.4 GB while the footprint is 10 GB).

### 9.1 Stage-by-stage trace (native WebGPU, `kBlockChunkBytes` default 1.5 GB)

Per decoder stage: active coordinates in / out, feature shape (channel-major `[C, N]`, all f32
activations, f16 weights), the largest graph-allocated tensor, and gallocr's buffer.

| stage | coords in → out | features | graph(s) | nodes | largest single allocation | largest temporary (op) | live est. | gallocr |
|---|---|---|---|---|---|---|---|---|
| from_latent | 17,489 | `[32,N]` → `[1024,N]` f32 | 1 | 2 | 71.6 MB `[1024,17489]` | same (MUL_MAT) | 0.14 GB | 0.07 GB |
| stage 0 ConvNeXt ×4 (C=1024) | 17,489 | `[1024,N]` | 1 | 812 | 287 MB `[4096,17489]` | MLP hidden (MUL_MAT/ADD/SILU) | 0.65 GB | 0.51 GB |
| stage 0 C2S (1024 → 512) | 17,489 → 71,262 | `[1024,N]` → `[512,M]` | subdiv + conv | 2 + 402 | 287 MB `[4096,17489]` | conv1 `[Cout·8, N]` per tap | 1.01 GB | 0.88 GB |
| stage 1 ConvNeXt ×16 (C=512) | 71,262 | `[512,N]` | 1 | 3248 | 584 MB `[2048,71262]` | MLP hidden ×16 blocks | 1.33 GB | 1.04 GB |
| stage 1 C2S (512 → 256) | 71,262 → 288,168 | `[512,N]` → `[256,M]` | subdiv + conv | 2 + 402 | 584 MB `[2048,71262]` | conv1 per tap ×27 | 2.08 GB | 1.79 GB |
| stage 2 ConvNeXt ×8 (C=256) | 288,168 | `[256,N]` | 1 | 1624 | **1.18 GB `[1024,288168]`** | MLP hidden ×8 | 2.69 GB | 2.10 GB |
| stage 2 C2S (256 → 128) | 288,168 → 1,158,925 | `[256,N]` → `[128,M]` | subdiv + conv | 2 + 602 | **1.18 GB `[1024,288168]`** | conv1 per tap ×27; `[128, M+1]` PAD/NORM/SILU 593 MB | 4.30 GB | 3.71 GB |
| stage 3 ConvNeXt ×4 (C=128) | 1,158,925 | `[128,N]` | 1 | 1628 | **1.57 GB `[512,768000]`** | MLP hidden chunk (2 chunks/block: 768,000 + 390,925 rows) | 4.46 GB | 3.28 GB |
| stage 3 C2S (128 → 64) | 1,158,925 → 4,649,809 | `[128,N]` → `[64,M]` | subdiv + conv | 8 + 1392 | **1.57 GB `[512,768000]`** | conv1 per tap ×27; `[64, M+1]` PAD/NORM/SILU + `[64, M]` out 1.19 GB each | 6.97 GB | **7.19 GB** |
| output_layer (64 → 7) | 4,649,809 | `[64,M]` → `[7,M]` | 5 × 1M-row chunks | 3 | 256 MB `[64,1000000]` (NORM) | same | 0.51 GB | 0.51 GB |

Host side (WASM in the browser) per stage: coords, the `[27, N]` / `[27, M]` int32 neighbour
tables (stage 3: 125 MB + 502 MB), the octant→gather index maps (37 MB), and the stage features
between graphs (`[64, 4.65M]` f32 = 1.19 GB at the end). Same graphs, node counts and shapes as
the CUDA trace of §8 up to the ±0.03 % voxel flips of spec 31 §11.3.

### 9.2 Every allocation ≥ 512 MB (Shape-1024, native WebGPU trace)

| tensor | bytes | source op | why it exists | lifetime overlap | tileable / avoidable exactly? |
|---|---|---|---|---|---|
| `[512, 768000]` f32 ×(4+4+27+27) | 1.57 GB | stage-3 ConvNeXt MLP hidden (`mlp.0` MUL_MAT → ADD → SILU) and stage-3 C2S conv1 per-tap MUL_MAT / running-sum ADD | `[4C, nr]` / `[Cout·8, nr]` intermediates of one voxel chunk; the chunk is `kBlockChunkBytes` (1.5 GB) / (512 × 4 B) = 768,000 rows | within one chunk 2-3 of them are live at once (gather → matmul → add); chunks are sequential, gallocr reuses their slots | **yes**: they are budget-driven; `TRELLIS_BLOCK_CHUNK_MB` / `TRELLIS_C2S_CHUNK_MB` shrink them linearly with no numerical change (§9.3) |
| `[512, 390925]` f32 ×(4+4+27+27) | 801 MB | same, the tail chunk | same | same | same |
| `[1024, 288168]` f32 ×(8+8+8+27+27) | 1.18 GB | stage-2 ConvNeXt MLP hidden ×8 blocks; stage-2 C2S conv1 per tap | single-chunk stage (N < budget/4 KB) | as above | yes (budget) |
| `[2048, 71262]` f32 ×(16·3+27·2) | 584 MB | stage-1 MLP hidden ×16; stage-1 C2S conv1 | single chunk | as above | yes (budget) |
| `[64, 4649810]` f32 ×3 | 1.19 GB | stage-3 C2S `hraw` (PAD seed written by chunked CPY), its NORM, its SILU (`hn`, conv2's padded input) | the channel→spatial result at M+1 rows (sentinel row) must exist whole: conv2's gather reads arbitrary rows of it | `hraw` dies at NORM, NORM at SILU; `hn` lives through all conv2 chunks, next to `out` | **M-scaled, not budget-scaled**: cannot shrink without a buffer split (§5 B9); NORM/SILU in place would save the two transient copies only in the estimate (tried, gallocr unchanged, op-gap §11.3) |
| `[64, 4649809]` f32 ×1 | 1.19 GB | stage-3 C2S conv2 output buffer (PAD seed + CPY per chunk) | the stage output, read back to the host whole | lives through every conv2 chunk, with `hn` | M-scaled; a split would only matter on a spec-floor adapter |
| `[64, 3082211]` f32 ×1 | 789 MB | stage-3 C2S: chunk-0 GET_ROWS (channel→spatial gather of the first conv1 chunk, 3.08M surviving octants) | the PAD seed's source | dies at the PAD | budget-driven (chunk of 768,000 input voxels × up to 8 octants) |
| `[128, 1158925/6]` f32 ×~12 | 593 MB | stage-3 inputs (`[C,N]` leaf), the `[Ci, N+1]` sentinel CONCAT per ConvNeXt block (×4) and in C2S, norm1 NORM/MUL/ADD/SILU, the stage-2 C2S output PAD | stage-scale `[C, N]` tensors | the CONCATs are one per block, sequential | M/N-scaled; the per-block CONCAT could be built once per stage instead of per block (not done: −593 MB × 3, only in the estimate for the same reason) |
| `[4649809, 27]` i32 (input) | 502 MB | conv2 neighbour table (host-built) | gather indices for the 27 taps at M rows | whole graph | M-scaled |

None of these exceeds the 4 GiB binding on this adapter; the C2S stage-3 graph's 7.19 GB gallocr
buffer (vs 5.98 GB measured on CUDA for the same graph in §8 -- ggml-alloc splits it into
`maxBufferSize`-sized chunks here, so the packing is looser) is allocated and computed by both
Dawn and Chrome (§9.4).

### 9.3 The tiling lever: chunk budget A/B

Same binary, same fixture, `TRELLIS_BLOCK_CHUNK_MB=512 TRELLIS_C2S_CHUNK_MB=512` (the existing
A/B overrides of `kBlockChunkBytes`, default 1.5 GB) versus the default:

| graph | default: nodes / largest / gallocr | 512 MB budget: nodes / largest / gallocr |
|---|---|---|
| stage 1 ConvNeXt ×16 | 3248 / 584 MB `[2048,71262]` / 1.04 GB | 6512 / 537 MB `[2048,65536]` / 0.98 GB |
| stage 2 ConvNeXt ×8 | 1624 / 1.18 GB `[1024,288168]` / 2.10 GB | 4872 / 537 MB `[1024,131072]` / 1.43 GB |
| stage 2 C2S conv | 602 / 1.18 GB / 3.71 GB | 1390 / 593 MB (`[128, M+1]` SILU) / 2.83 GB |
| stage 3 ConvNeXt ×4 | 1628 / 1.57 GB `[512,768000]` / 3.28 GB | 4052 / 593 MB (`[128, N+1]` CONCAT) / 2.79 GB |
| **stage 3 C2S conv** | 1392 / 1.57 GB / **7.19 GB** | 2376 / **1.19 GB** (`[64, M+1]` NORM) / **6.55 GB** |
| decode wall / process peak | 160-174 s / 10.4 GiB | 187 s / 10.2 GiB |
| mesh | V = 4,649,809, worst mean 6.3448e-7 | **identical** (V = 4,649,809, 6.3448e-7) |

So the budget removes every chunk-driven tensor above 600 MB exactly (largest allocation 1.57 →
1.19 GB) and takes 0.6-0.9 GB off each large graph, but the stage-3 C2S floor is set by the
M-scaled tensors -- `hn` `[64, M+1]` and `out` `[64, M]` (1.19 GB each, both live through every
conv2 chunk), the `[128, N+1]` sentinel input, the `[M, 27]` neighbour table (502 MB) and gallocr's
packing across the 4 GiB buffer chunks -- so the graph still needs 6.55 GB and the process ~10 GiB.
The default stays at 1.5 GB (graph node count is 1.7-3× at 512 MB; `kMaxChunksPerBlock`/`kGraphNodes`
would need retuning for smaller budgets); the knob is there for smaller adapters. The next real
step down is the §5 B9 split of the two `[Cout, M]` buffers, which is a graph-construction change
that only pays off on spec-floor adapters.

The other transient that was measured, on the host side: the WebGPU backend stages every
`get_tensor` through one MapRead buffer of the request size, and in the browser the mapped range
is a wasm-heap copy -- reading the 1.19 GB stage-3 C2S output back needed 2 × 1.19 GB on top of
~1.4 GB of live host vectors in a 4 GB heap and trapped (`memory access out of bounds` in
Chrome, first browser run). `tensor_to_f32` now reads F32 results in 256 MiB slices
(`ggml_backend_tensor_get` with offset), which bounds the staging buffer and the mapped copy to
256 MiB on every backend; exact.

### 9.4 Peak memory

| run | decode wall | process peak footprint | note |
|---|---|---|---|
| native WebGPU, Shape-1024, default budget | 160-174 s | **10.4 GiB** (`phys_footprint`, byte-exact; ≈ 7.19 GB stage-3 C2S buffer + 0.95 GB weights + ~2.5 GB host vectors) | the Mac GPU was shared with another session's flow probe during one run |
| native WebGPU, res-512 | 46 s | 3.5 GiB | |
| **Chrome 152, Shape-1024** (`web/shape_decode/`, sliced readback) | 288 s | **GPU process 9.93 GB**, all Chrome processes 16.0 GB (renderer/worker holds the wasm heap: fixture + host vectors + mesh) | first attempt (whole-tensor readback) trapped at the stage-3 C2S readback, §9.3 |
| Chrome 152, res-512 | 55 s | GPU process 4.03 GB, all Chrome 5.81 GB | |

So Chrome grants one tab the 6.55-7.19 GB stage-3 graph plus weights and staging (the §8 open
question); the remaining browser-side ceiling is the 4 GB wasm heap, which the decoder now stays
under at M = 4.65M (host vectors ≈ 2.6 GB at the stage-3 C2S: `[128,N]` input 0.59 GB, two
neighbour tables 0.63 GB, `[64,M]` output 1.19 GB, coords/index maps 0.15 GB, plus the 256 MiB
readback slice). Objects denser than ~1.5× this one would need the stage output read back in
pieces into a smaller host representation or the `[Cout, M]` split.

## 10. Measured: sparse texture (PBR) decoder on the ggml WebGPU backend (2026-09-07, `feat/webgpu-texture-decode`)

`tex_decode` (the §9 decoder's `decode_unet` with the shape decoder's masks as `guide_subs` and a
6-channel head) on native Dawn (Apple M1 Pro 32 GB, same adapter limits as §9) and Chrome 152, on
the **real** Texture-1024 SLAT (`hr_sample`, N = 17,489 → M = 4,649,809), with the §8
`TRELLIS_DBG_ALLOC_TRACE` instrumentation. Parity: `docs/spec/31-webgpu-bringup.md` §12. The
measurement is `trellis-test-pixal3d-tex-decode`, which runs `shape_decode` first (stage A, for
the masks) and `tex_decode` second (stage B) in one process; the footprint below is that process.

### 10.1 Stage-by-stage trace (native WebGPU, `kBlockChunkBytes` default 1.5 GB)

| stage | coords in → out | features (f32) | dtype (weights / staged mul_mat inputs) | graph nodes | largest single allocation | largest temporary (op) | live est. | gallocr | vs shape decoder (§9.1) |
|---|---|---|---|---|---|---|---|---|---|
| from_latent | 17,489 | `[32,N]` → `[1024,N]` | f16 / f16 | 2 | 71.6 MB `[1024,17489]` | same (MUL_MAT → ADD) | 0.14 GB | 0.07 GB | same |
| stage 0 ConvNeXt ×4 (C=1024) | 17,489 | `[1024,N]` | f16 / f16 | 812 | 287 MB `[4096,17489]` | MLP hidden (SILU) | 0.65 GB | 0.51 GB | same |
| stage 0 C2S (1024 → 512), mask-driven | 17,489 → 71,262 | `[1024,N]` → `[512,M]` | f16 / f16 | 402 (**no subdiv graph**) | 287 MB `[4096,17489]` | conv1 `[Cout·8, N]` per-tap ADD | 1.01 GB | 0.88 GB | same conv graph; the shape decoder's extra `_subdiv` graph (2 nodes, 0.07 GB) is absent |
| stage 1 ConvNeXt ×16 (C=512) | 71,262 | `[512,N]` | | 3248 | 584 MB `[2048,71262]` | MLP hidden ×16 | 1.33 GB | 1.04 GB | same |
| stage 1 C2S (512 → 256) | 71,262 → 288,168 | `[512,N]` → `[256,M]` | | 402 | 584 MB | conv1 per tap | 2.08 GB | 1.79 GB | same (no `_subdiv` 2 nodes / 0.07 GB) |
| stage 2 ConvNeXt ×8 (C=256) | 288,168 | `[256,N]` | | 1624 | 1.18 GB `[1024,288168]` | MLP hidden ×8 | 2.69 GB | 2.10 GB | same |
| stage 2 C2S (256 → 128) | 288,168 → 1,158,925 | `[256,N]` → `[128,M]` | | 602 | 1.18 GB | conv1 per tap; `[128, M+1]` PAD/NORM/SILU 593 MB | 4.30 GB | 3.71 GB | same (no `_subdiv` 2 nodes / 0.30 GB) |
| stage 3 ConvNeXt ×4 (C=128) | 1,158,925 | `[128,N]` | | 1628 | 1.57 GB `[512,768000]` | MLP hidden chunk (2 chunks/block) | 4.46 GB | 3.28 GB | same |
| stage 3 C2S (128 → 64) | 1,158,925 → 4,649,809 | `[128,N]` → `[64,M]` | | 1392 | **1.57 GB `[512,768000]`** (conv1 per-tap MUL_MAT/ADD) | `[64, M+1]` PAD/NORM/SILU + `[64, M]` out, 1.19 GB each | 6.97 GB | **7.19 GB** | same; the shape decoder's `_subdiv` graph here (8 nodes, gallocr 1.14 GB: the `[128,N]` leaf + `[8,N]` logits) is absent |
| output_layer (64 → **6**) | 4,649,809 | `[64,M]` → `[6,M]` | | 5 × 1M-row chunks, 3 nodes each | 256 MB `[64,1000000]` (NORM) | same | 0.51 GB | 0.51 GB (last chunk 0.33) | `[6,1000000]` output 22.9 MB vs `[7,1000000]` 26.7 MB |

Weights: 904.7 MB (`tex_dec.gguf`, 284 tensors: 106 f16 + 178 f32 biases/norms) vs 904.8 MB
(`shape_dec.gguf`, 292: the 4 × `to_subdiv` weight+bias pairs, 8 tensors, 0.1 MB). Host side per
stage: identical to §9.1 (coords, `[27,N]`/`[27,M]` neighbour tables 125 + 502 MB at stage 3,
gather maps, the `[64, M]` f32 stage output 1.19 GB), plus what the texture path carries across
the two decoders: the four octant masks (8 bytes per voxel per level, 12 MB) and the final
`[6, M]` attributes (112 MB, vs the `[7, M]` head 130 MB). Sparse feature buffers, neighbour /
index buffers and temporaries are therefore the §9 numbers to the byte -- the channel widths
that set them (torso `model_channels`) are the same in both checkpoints; the texture decoder
does **not** need less memory, only marginally less at the head and none for subdivision.

### 10.2 Largest allocations (texture decoder, native WebGPU trace)

Every allocation ≥ 512 MB is one of §9.2's, at the same node and size (`[512,768000]` 1.57 GB
×62 in stage-3 ConvNeXt/C2S conv1 -- budget-driven; `[512,390925]` 801 MB tail chunk;
`[1024,288168]` 1.18 GB ×~40 in stage 2; `[2048,71262]` 584 MB in stage 1; the M-scaled
`[64, 4649810]` ×3 and `[64, 4649809]` 1.19 GB in stage-3 C2S; the `[4649809, 27]` i32 table 502
MB). The texture head adds nothing above 256 MB (`[64,1000000]` NORM in the output layer). No
layer or op of the texture decoder exceeds its shape-decoder counterpart.

### 10.3 Peak memory

| run | decode wall (shape A + tex B) | process peak footprint | note |
|---|---|---|---|
| native WebGPU, `trellis-test-pixal3d-tex-decode`, Texture-1024 | 182 s + 219 s | **10.52 GB** (`phys_footprint`, sampled every 2 s; §9.4 shape-only: 10.4 GiB = 11.2 GB) | same largest graph (7.19 GB) + 0.9 GB weights + host vectors; the Mac GPU was shared with another session's flow probe for part of the run |
| Chrome 152, `web/tex_decode/` | 241 s + 268 s (browser wall 514 s) | GPU process 10.15 GB, all Chrome 16.3 GB (the sampler's own shell is counted in the total, a few MB) | §9.4 shape-only: 9.93 GB / 16.0 GB |

The 4 GB wasm heap ceiling of §9.4 is unchanged: the texture path's extra host state is the
masks (12 MB) and the `[6, M]` result (112 MB), read back in the §9.3 256 MiB slices.

## 11. Measured + fixed: Texture-1024 conditioning (2026-09-08, `feat/browser-partial-e2e`)

`tex_1024` は唯一 NAF target が 1024 の段（spec 30 §2 の表）。`pixal3d_cond_slat_gpu` の
単一グラフ版は、この段で **ggml gallocr が 1 本 11 344 MB のバッファ**を要求する。
§1 の実測 `maxBufferSize` は 4 294 967 292 B なので確保できない。内訳の主犯は NAF 出力
`[1024, 1024²]` f32 = 4 294 967 296 B —— **上限をちょうど 4 バイト超える**。

計測環境: M4 Max (64 GB), Metal (`MTL0`), `trellis-test-pixal3d-cond-tex`, 実入力 4 view
(`transforms.json` + pre-matted RGBA 1024²), sparse coords N=16 384。

| 経路 / backend | 単一グラフバッファ最大 | 永続 | 実時間 |
|---|---|---|---|
| 単一グラフ・dense R³ (従来) / Metal | **11 344 MB** | 2 196 MB | 40.7 s (4 view, N=16 384) |
| 分割グラフ・sparse coords / Metal | **3 332 MB** | 2 196 MB | 41.8 s (4 view, N=16 384) |
| 分割グラフ・sparse coords / **WebGPU (Chrome, M4 Max)** | **1 538 MB** | 2 733 MB | 47.3 s (4 view, N=10 901) |

WebGPU のほうが Metal より単一バッファが小さいのは、`naf_ggml_opts_for` が WebGPU では
`direct_conv=true` を選び、encoder の畳み込みが `[K*K*Ci, W*H]` の im2col バッファを作らない
ため（generic lowering で増えるぶんを上回って効く）。WebGPU 行は real-input full E2E
(`pixal3d_real_full_run`) の実行ログから。

分割の構成（`src/pixal3d_cond_gpu.cpp::cond_slat_gpu_chunked`、view ごとに 4 種のグラフ）:

| グラフ | 内容 | バッファ (S=1024, T=1024) |
|---|---|---|
| `..._dino` | DINOv3 → global 累積 + patch map を永続化 + lr projection tap 累積 | 1 120.7 MB |
| `..._naf_enc0` / `_enc1` | NAF encoder の 2 枝を別グラフで回し pooled の前半/後半へ直接書く | 1 042.0 / 3 332.0 MB |
| `..._naf_qk` | block 行 stripe ごとに RoPE → q_bm / k_rows | 224.2 MB |
| `..._naf_attn` | block chunk ごとに neighborhood attention → その chunk の hr tap を累積 | 564.4 MB |

永続バッファは 1 本ずつ別確保（pooled 1 024 MB、q_bm 1 024 MB、k_rows/patch/accumulator は小）。
NAF 出力を一度も materialize しないので、単一バッファは encoder 枝の 3 332 MB が最大になる。

### 数値の同値性

同一入力・同一 backend で、単一グラフ版（この修正前のコード）の出力と比較:

| 量 | max abs | L2 rel | cosine | bit identical |
|---|---|---|---|---|
| `global` | 0 | 0 | 1.0000000000 | **yes** |
| `proj` lr 半分（DINO 由来） | 0 | 0 | 1.0000000000 | **yes** |
| `proj` hr 半分（NAF 由来） | 3.815e-06 | 8.531e-09 | 1.0000000000 | no |

hr だけ差が出るのは、4 つの bilinear tap を chunk をまたいで足すため加算順序が変わるから
（chunk 外の tap は重み 0・索引 0 に潰しているので値そのものは同一）。

`coords` を渡す sparse 経路（dense R³ を作らず active voxel だけ蓄積）は、
S=512/R=32/T=512・V=2 で dense+`pixal3d_gather_proj` と **全成分ビット一致**を確認済み。
同条件で `--chunk 64` を強制すると単一バッファは 2 836 MB → 834 MB に下がる。

### PyTorch 参照との比較について

`tools/ref_pixal3d_cond_slat.py` の `tex_cond_global.npy` / `tex_cond_proj.npy` は
DINOv3+NAF(natten)+spconv が要るため CUDA 必須で、この Mac では生成できない。**本節の
検証は「検証済みの単一グラフ実装との同値性」であり、PyTorch 参照との突き合わせは未実施**。

## 12. Texture Flow の GPU 予算と Q8_0（2026-09-09）

ブラウザで Texture Flow を回すとマシン全体が固まっていた。原因は wasm ヒープではなく
**GPU 側の常駐量が WebGPU の予算を超えていた**こと。超えても確保は成功するが、
ユニファイドメモリ上で Metal がメモリを往復させ続け、1 forward が 40 秒から
139〜176 秒に伸びて帯域を WindowServer ごと奪う。

### 内訳は N（Texture Flow のトークン数 = grid-64 のアクティブボクセル数）に比例する

実測 2 点（native Metal, `TRELLIS_ATTN_CHUNK_MB=128`）から:

```
活性化 = 59.7 KB/token x N + 0.2 MB      （N=12083 -> 705.1 MB / N=17690 -> 1032.2 MB）
cond   = 16.0 KB/token x N               （proj [2048,N] を positive/negative 2 本）
合計   = 重み（固定）+ 75.7 KB/token x N
```

| 重み | 上限 N（予算 4095 MB） |
|---|---|
| f16 2647 MB | **19 600** |
| Q8_0 1408 MB | **約 36 400** |

N は被写体の grid-64 での**表面積**で決まる（体積ではない。cyclops のアクティブ集合は
内部ボクセルが 5.0% しかない厚さ 1 層のシェル）。細身の人物で 4 794、頭部の胸像で 17 690。

### 活性化を削る: attention のクエリ分割

活性化 1888 MB の過半は `kAttnChunkBytes = 1024 MB`（sdpa のクエリ分割 1 チャンクあたりの
スコア行列の予算）そのものだった。下げると:

| `TRELLIS_ATTN_CHUNK_MB` | 活性化 | 備考 |
|---|---|---|
| 1024（native の既定） | 1888.1 MB | |
| 512 | 1381.2 MB | |
| 256 | 1316.5 MB | 旧 `kMaxAttnChunks = 32` で頭打ち |
| **128（ブラウザの既定）** | **1032.2 MB** | 上限を 256 に広げた後 |
| 64 / 32 | 1032.2 MB | attention 以外が支配的になり頭打ち |

クエリ分割は softmax がクエリ行ごとに閉じているので結果が変わらない。**実測でも
12 step 全部の latent がビット一致**（chunk 1024 対 128、max abs = 0）、所要も
644.4 -> 602.5 秒で悪化しない。上限を縛っていたのは `flow_runner.cpp` のテンソル
メタデータ枠（1 本 368 B）で、広げても数十 MB。

### 重みを削る: Q8_0（ブラウザは採用、native は f16 のまま）

`tools/quantize_gguf.py` で 365 テンソルを Q8_0 化 → 2647 MB -> **1408 MB**。

| | f16 | Q8_0 |
|---|---|---|
| 合計常駐（N=17690） | 3956 MB（余裕 3.4%） | **2717 MB（余裕 34%）** |
| 上限 N | 19 600 | **約 36 400** |
| 12 step 所要 | 602.5 s | 594.5 s |
| latent の最終差 | — | L2 rel 2.7e-02（step ごとに増幅） |
| **baseColor 4096^2** | — | **max 7/255**・平均 0.36/255・差>2 の画素 0.019%・差>8 は 0% |
| metallicRoughness | — | max 1/255 |
| 4 視点レンダ | — | max 6/255、差>2 の画素 0.024% |
| 形状（V/F/bbox） | — | **完全一致**（texture flow は色にしか効かない） |

latent の 2.7e-02 は最終的な色では 1 画素単位の max 7/255 にしかならず、それが出るのは
全画素の 0.019%。**視覚的に区別できない。** 余裕 3.4% では被写体が 1 割大きいだけで
破綻するので、ブラウザは Q8_0 を使う。native はメモリ制約が無いので f16 のまま
（参照との一致を優先）。ブラウザ側はファイル名で選ぶだけなのでコード変更は不要
（`pixal3d_tex_flow_1024_mv.gguf` を Q8_0 版に差し替える）。

### 走らせる前に落とすゲート

`DitRunner` はグラフ確保の直後に weights + activations + cond を合計し、
`ggml_backend_dev_memory` の total と比べて超えていたら throw する。
`TRELLIS_DEVICE_BUDGET_MB=4095` でブラウザの予算を native から模擬でき、
**ブラウザを起動せずに「この入力はブラウザで通るか」を判定できる**。
`TRELLIS_DBG_BUDGET=1` で内訳表示、`TRELLIS_ALLOW_OVER_BUDGET=1` で意図的に踏める。

## 13. モデル一式の Q8_0 化（2026-09-09）

`tools/quantize_gguf.py` で全モデルを Q8_0 にした。**13 GB -> 7.8 GB（40% 減）**。

| モデル | f16 | Q8_0 | 量子化した / 全テンソル |
|---|---|---|---|
| `pixal3d_ss_flow_mv` | 2557 MB | **1360 MB** | 244 / 700 |
| `pixal3d_shape_flow_512_mv` | 2647 MB | **1408 MB** | 245 / 700 |
| `pixal3d_shape_flow_1024_mv` | 2647 MB | **1408 MB** | 245 / 700 |
| `pixal3d_tex_flow_1024_mv` | 2647 MB | **1408 MB** | 245 / 700 |
| `shape_dec` | 905 MB | 841 MB | 70 / 292 |
| `tex_dec` | 905 MB | 841 MB | 66 / 284 |
| `dinov3` | 579 MB | 579 MB | 対象外 |
| `ss_dec` / `pixal3d_naf` | 142 MB | 142 MB | 対象外 |

decoder がほとんど減らないのは、Conv3D の重みが 4D/5D で
「2D かつ ne0 がブロック整列」という対象条件から外れるため。さらに削るなら conv 重みの
扱いを別途決める必要があるが、そこは形状に直接効くので慎重にやること。

### 形状に効くモデルまで量子化して壊れないか

texture flow は色にしか効かないが、ss / shape flow と decoder は形状に効く。
全 Q8 と全 f16 で native E2E を回して `tools/silhouette_iou.py` で判定した
（views4_fixed、seed 1）:

| | 全 Q8 | f16 |
|---|---|---|
| mean silhouette IoU | **0.9105** | 0.9089 |
| scale_error | 0.0060 | 0.0030 |
| 最終 GLB | V=699 770 F=980 016 | V=694 278 F=945 776 |
| bbox | (0.6378, 0.9925, 0.3091) | (0.6320, 0.9858, 0.3115) |
| Shape-1024 tokens | 4 758 | 4 750 |
| ゲート判定 | **PASS** | PASS |

IoU は f16 と同等（差 0.0016 は SS decode の閾値が離散判定であることによる token 数の
ゆらぎ（4 758 対 4 750）の範囲で、Q8 が優れているという意味ではない）。

### 数値の実体は W8A16 + F32 蓄積

`Q8_0` は int8 + 32 要素ごとの f16 スケール。行列積では両バックエンドとも
**重みを f16 へ展開し、活性化も f16 タイルに落とし、積和は f32** で行う。

- Metal `kernel_mul_mm_q8_0_f32`: 重み・活性化とも `simdgroup_half8x8`、蓄積 `simdgroup_float8x8`
- WebGPU `mul_mat_reg_tile.wgsl`: `QUANT_OUT_TYPE f16`、`shmem: array<f16>`、`acc: array<f32>`

**活性化が f16 なのは量子化とは無関係**で、f16 の重みでも同じ経路（`kernel_mul_mm_f16_f32`
も活性化を half に落とす）。Q8_0 が変えたのは W16 -> W8 だけ。
なお小さい行列で選ばれる Metal の `mul_mv` は活性化を f32 のまま読むので W8A32 になる。

