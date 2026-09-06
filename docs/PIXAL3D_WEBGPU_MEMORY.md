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

**Measured 2026-09-06** in Google Chrome 152.0.7977.82 (macOS 26.5, Build 25F71) on the Apple M1
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
