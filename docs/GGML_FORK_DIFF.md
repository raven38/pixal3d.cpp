# GGML fork diff (`thirdparty/ggml`)

Source-level record of how the vendored ggml submodule differs from upstream
`ggml-org/ggml`. Compiled by reading `thirdparty/ggml` git history (read-only:
`log`/`diff`/`show`/`remote`/`describe`/`fetch`, no checkout of the submodule)
and a separate shallow-then-unshallowed clone of `ggml-org/ggml` in scratch
space, cross-referenced against `src/*.cpp`, `CMakeLists.txt`, and
`docs/spec/*.md` for how/why each patch is used.

## Header

| field | value |
|---|---|
| Fork URL | `https://github.com/pwilkin/ggml` |
| Fork branch | `trellis-patches` |
| Pinned commit (`.gitmodules` + submodule pointer) | `737e88f25d4f62254f3b7a726fd9663036cc94da` |
| `git describe --tags` | `v0.15.1-10-g737e88f2` |
| Upstream base tag | `v0.15.1` |
| Upstream base SHA | `4251bf0eba36a032b871038156fde8068da28062` — **identical** in `pwilkin/ggml` and `ggml-org/ggml` (verified by cloning `ggml-org/ggml` and comparing `git rev-parse v0.15.1` in both remotes); the fork is a genuine linear descendant of the real upstream tag, not a re-tagged/rewritten history |
| Upstream base date | 2026-06-12 |
| Current upstream master (for reference) | `e91ded1` / `v0.23.0`, 2026-09-04 — **517 commits** ahead of `v0.15.1` |
| Recorded | 2026-09-06 |

## Per-commit table

`git log --oneline v0.15.1..HEAD` lists 10 commits. The first is a genuine
upstream commit the fork branch picked up before diverging — confirmed with
`git merge-base --is-ancestor 3af5f576 origin/master` against the unshallowed
`ggml-org/ggml` clone, which returns true. The remaining 9 are fork-authored,
all CUDA-only, all with commit messages that explicitly cite a Pixal3D/TRELLIS.2
symptom (res-1024 shape decode, DiT FlashAttention on 15k–53k tokens).

| SHA | Subject | Author | Files | Backend | Purpose (tied to project usage) |
|---|---|---|---|---|---|
| `3af5f576` | scripts : fix sed command | Georgi Gerganov (upstream) | `scripts/release.sh` | none (tooling) | **Not a fork patch** — a real upstream commit inherited before the fork's own commits start; irrelevant to runtime behavior. |
| `9937a412` | cuda: 64-bit indexing in binbcast for >2^31-element tensors | Piotr Wilkin | `binbcast.cu` | CUDA | `k_bin_bcast`'s `ne0`/loop index were 32-bit `int`; a same-shape add collapses to `[C*N]` and TRELLIS.2's res-1024 sparse-conv output reaches ~3.7B elements, causing OOB writes / GPU page fault. |
| `6c9ba07b` | cuda: fix fastdiv UB for divisors > 2^31 | Piotr Wilkin | `common.cuh` | CUDA | `init_fastdiv_values`'s shift-by-32 UB, hit when binbcast's collapsed divisor exceeds 2^31 on a res-1024 goblin decode (N>524288 for a `[4096,N]` tensor). |
| `9c9bc0e0` | cuda: 64-bit element count in unary ops | Piotr Wilkin | `unary.cu` | CUDA | `unary_op_kernel`'s `int k`/`int i` silently truncate past 2^31 elements; reached by `silu` on the ConvNeXt MLP (4096-wide) at N>524288 in a res-1024 shape decode — **silent corruption**, not a crash. |
| `323c9f00` | cuda: 64-bit unravel path in binbcast for >2^32-element tensors | Piotr Wilkin | `binbcast.cu` | CUDA | Follow-up to `9937a412`: `k_bin_bcast_unravel`'s flat index/fastdiv products/`i_src0` still overflow past 2^32; same res-1024 ConvNeXt MLP at N>1.05M. Adds a 64-bit grid-stride kernel gated by an overflow check, uint32 fast path untouched. Commit notes at least one more >2^32 defect remains elsewhere in ggml-cuda (unchunked convnext). |
| `88c5c7d2` | cuda: FA kernel/prec diagnostics + reachable F32 tile accumulator (opt-in) | Piotr Wilkin | `fattn-tile.cuh`, `fattn.cu` | CUDA | Adds `GGML_FA_DEBUG=1` (prints selected FA kernel/prec/shapes) and opt-in `GGML_FA_TILE_FORCE_F32`; discovered that only `fattn-wmma-f16.cu` reads `ggml_flash_attn_ext_get_prec()` — TILE/VEC/MMA silently ignore a caller's `GGML_PREC_F32`. |
| `2d6d0b0c` | cuda: honour GGML_PREC_F32 in the TILE, VEC and MMA FlashAttention kernels | Piotr Wilkin | `fattn-mma-f16.cuh`, `fattn-tile.cuh`, `fattn-vec.cuh` | CUDA | Adds an `acc_f32` accumulator path to all three kernels. FP16 VKQ accumulation stagnates over many KV tiles (measured bias growing from d=-0.0034 at 8 tiles to d=-0.0174 at 59 tiles on a 30-block DiT) — directly the shape of `src/dit.cpp`'s SDPA at HR token counts. |
| `38d0059c` | cuda: honour GGML_PREC_F32 in the mma FlashAttention kernel on Turing+ | Piotr Wilkin | `fattn-mma-f16.cuh` | CUDA | Follow-up: `2d6d0b0c` fixed MMA only on AMD WMMA/MFMA; this extends the FP32 accumulator to mainstream NVIDIA (Turing/Ampere/Ada/Blackwell — sm_86, sm_120 verified), leaving only Volta unable to honour the request. |
| `f33ab068` | Fix CUDA PAD kernel launch limits | Stéphane Lenclud | `pad.cu` | CUDA | Rewrites `pad_f32` from a 3D grid (`blockIdx.y`=`i1`, `blockIdx.z`=`i3*ne2+i2`) to a flat grid-stride loop over `int64_t total`, removing CUDA's per-dimension grid limits. Backs `ggml_pad` calls in `src/dit.cpp` (zero-pad key dim for FlashAttention) and `src/sparse.cpp` (partial-tile padding) at res-1024 scale. |
| `737e88f2` | cuda: use 64-bit FlashAttention mask strides | Piotr Wilkin | `fattn-common.cuh` | CUDA | Widens `flash_attn_mask_to_KV_max`'s `s31`/`s33` KQ-mask strides from `int` to `int64_t`. **Byte-identical** to a later upstream fix (`6d2504d5`, merged 2026-06-30, ~2 weeks after this fork's base) — arrived at independently, not cherry-picked (see below). Needed by `src/dit.cpp`'s SDPA mask path at the 1024 cascade's ~53k HR tokens. |

`git diff --stat v0.15.1..HEAD` (fork-authored files only):

```
 src/ggml-cuda/binbcast.cu       | 108 +++++++++++++-
 src/ggml-cuda/common.cuh        |   6 +-
 src/ggml-cuda/fattn-common.cuh  |   6 +-
 src/ggml-cuda/fattn-mma-f16.cuh | 316 +++++++++++++++++++++++++++++++---------
 src/ggml-cuda/fattn-tile.cuh    | 235 +++++++++++++++++-------------
 src/ggml-cuda/fattn-vec.cuh     | 123 +++++++++-------
 src/ggml-cuda/fattn.cu          |  15 +-
 src/ggml-cuda/pad.cu            |  96 ++++++------
 src/ggml-cuda/unary.cu          |   8 +-
```

`include/` is untouched between `v0.15.1` and `HEAD` (`git diff --stat v0.15.1..HEAD -- include/` is empty) — **the fork adds no new public ggml API**; every patch fixes an existing op's CUDA implementation.

## Are these genuine patches, or cherry-picks?

- 9 of 10 are demonstrably fork-authored (author `Piotr Wilkin`/`Stéphane Lenclud`, commit messages that cite TRELLIS.2-specific tensor shapes and are `Co-Authored-By: Claude Opus 4.8`), not `git cherry-pick`s of upstream commits — their SHAs and full commit text do not appear in `ggml-org/ggml` history.
- One (`737e88f2`, mask strides) is **content-identical** to a commit upstream later made independently (`6d2504d5`, same 6-line diff, same before/after blob hashes `8dfa51ad`→`b76121fe`) — the fork and upstream fixed the same bug the same way, ~7 weeks apart, without one being a cherry-pick of the other (different commit metadata, different surrounding history).
- One (`9937a412`/`323c9f00` binbcast overflow) overlaps in *problem* with a later upstream commit (`6d1eb28b`, "address integer overflows in binary ops CUDA implementation") but the *fix* differs structurally: the fork widens only `ne0`/the loop index to `int64_t` and adds a separate 64-bit unravel kernel; upstream instead converts the whole shape/stride parameter set to `uint32_t` + `uint3` fastdiv and rewrites `k_bin_bcast`'s signature. These are **not** reconcilable by a trivial merge.
- `f33ab068` (PAD) and the FA-prec trio have no matching upstream commit at all as of `origin/master` (`v0.23.0`) — they remain fork-exclusive fixes.

## Per-file notes

**`src/ggml-cuda/fattn-common.cuh`** — `flash_attn_mask_to_KV_max`'s `s31`/`s33` params and the `launch_fattn` locals that feed them go from `int` to `int64_t` (see excerpt above). Prevents truncation of the KQ-mask row/batch stride once `mask->nb[1]`/`nb[3]` exceed 2^31 bytes, which the 1024-cascade's ~53k-token HR pass reaches.

**`src/ggml-cuda/fattn-tile.cuh` / `fattn-vec.cuh` / `fattn-mma-f16.cuh`** — all three gain an `acc_f32` (or equivalent) code path so `GGML_PREC_F32` widens the VKQ accumulator to `float2`/`tile<16,16,float>` instead of staying FP16, while K/V/Q stay FP16 in SRAM/registers (no extra shared-memory budget). `fattn-mma-f16.cuh` additionally reworks `mma_tile_sizes`, the VKQ_C rescale/sink indexing, and `shared_memory_limit_raised` (now indexed by `acc_f32`, fixing a latent OOB `__shared__` write once both accumulator variants can launch). `fattn.cu` gains the `GGML_FA_DEBUG` print and `GGML_FA_TILE_FORCE_F32` opt-in switch.

**`src/ggml-cuda/pad.cu`** — `pad_f32` changes from a `<blockIdx.x, blockIdx.y=i1, blockIdx.z=i3*ne2+i2>` grid (bounded by CUDA's per-dimension launch limits) to a flat grid-stride loop over `int64_t total = ne0*ne1*ne2*ne3` (see excerpt above under commit `f33ab068`). Same output, no dimension-count ceiling.

**`src/ggml-cuda/binbcast.cu`** — three commits: `9937a412` widens `k_bin_bcast`'s `ne0` and its loop variable to `int64_t`; `6c9ba07b` (in `common.cuh`) fixes the `fastdiv` shift-by-32 UB feeding it; `323c9f00` adds an entirely new ~98-line 64-bit `k_bin_bcast_unravel` variant, dispatched only when the element count or largest per-thread offset exceeds `UINT32_MAX` (plain 64-bit division, grid-stride, no fastdiv).

**`src/ggml-cuda/unary.cu`** — `unary_op_kernel`/`unary_cuda`'s `int k`/`int i` become `int64_t` (see excerpt above under commit `9c9bc0e0`).

**`scripts/release.sh`** — unrelated upstream tooling commit inherited incidentally (`3af5f576`); no runtime effect.

## Fork-only APIs used by pixal3d.cpp

**None found.** The submodule diff touches no header (`include/`), so it adds no new symbol. Every call site checked below resolves to an API already present in both `v0.15.1` and current upstream `master`:

| Symbol | Present in v0.15.1? | Present in current master? | Call sites |
|---|---|---|---|
| `ggml_flash_attn_ext_set_prec` / `_get_prec` | yes | yes | `src/dit.cpp:142`, `src/test_fa_mask_overflow.cpp:62`, `src/test_fa_bf16_range.cpp:54` |
| `ggml_pad` | yes | yes | `src/dit.cpp:129`, `src/sparse.cpp:314,352` |
| `ggml_pad_reflect_1d` | yes | yes | `src/naf_gpu.cpp:42,44` |
| `ggml_conv_2d`, `ggml_group_norm` | yes | yes | `src/naf_gpu.cpp` (ImageEncoder graph) |

What the fork *does* provide that upstream doesn't is **fixed CUDA kernel behavior** at the tensor sizes/precisions the project hits (>2^31/2^32-element tensors, FP32 FA accumulation on TILE/VEC/MMA, PAD past CUDA grid limits) — not new API surface. Two of the nine also add fork-only **diagnostic env vars** (`GGML_FA_DEBUG`, `GGML_FA_TILE_FORCE_F32`, from `88c5c7d2`); these are read inside ggml itself, not consumed by any `src/*.cpp` `getenv` call (only referenced in a `dit.cpp` comment telling a developer to check them) — i.e., a debugging aid, not a load-bearing dependency.

## CMake wiring (context for the diff)

`CMakeLists.txt`: `GGML_DIR=thirdparty/ggml`, `add_compile_definitions(GGML_MAX_NAME=128)` (both ggml and pixal3d.cpp sources need it — DiT/sparse tensor names exceed ggml's default 64-char limit; the header guards it with `#ifndef` so no submodule patch is needed), `add_subdirectory(thirdparty/ggml)` builds it in-tree, backend selected by the *upstream* flags (`-DGGML_CUDA=ON`/`-DGGML_VULKAN=ON`/`-DGGML_HIP=ON`; Metal auto-enables on Apple). Backends actually built by this project: **CUDA, HIP/ROCm, Vulkan, Metal, CPU** (`ggml-cpu` always linked as the base target); no BLAS backend is wired in. `GGML_WEBGPU` **does** exist on this fork (corrected 2026-09-06 -- see `docs/spec/31-webgpu-bringup.md` §3; the sentence above predates that audit).

## Custom ops outside ggml ("trellis/Pixal3D custom kernels")

These are hand-written GPU kernels in `src/`, dispatched independently of ggml's graph/backend-selection machinery — none of them take a ggml-managed CUDA stream; each does its own `cudaMalloc`/`cudaMemcpy`/kernel-launch (or raw Vulkan buffer/queue) on the default stream, so correctness relies on implicit stream-0 ordering relative to ggml's own CUDA backend, not the ggml scheduler.

| Op | Files | CUDA | Vulkan | CPU fallback | Dispatch |
|---|---|---|---|---|---|
| Deformable conv (BiRefNet background removal) | `deform_conv.cu`, `deform_conv_vk.cpp`, `deform_conv.comp`, `deform_conv_cpu.cpp` | raw `<<<blocks,threads>>>` launch, own host↔device transfers, default stream | compute shader compiled to SPIR-V at build time via `glslc` (embedded as a C header), driven with raw Vulkan calls independent of `ggml-vulkan`'s device/queue | `std::thread`-parallel over output channels, no OpenMP | **Compile-time** exclusive: CUDA/HIP branch always compiles `deform_conv.cu` (as CUDA or, per issue #20, as HIP language); Vulkan-only branch compiles `deform_conv_cpu.cpp`+`deform_conv_vk.cpp` with `TRELLIS_DEFORM_VULKAN` telling the CPU shim to defer; plain-CPU builds compile only `deform_conv_cpu.cpp`. |
| QEM mesh decimation (postprocess) | `decimate_qem.cu`, `decimate_qem_vk.cpp`, `decimate_qem.comp`, `decimate_qem.cpp` (host orchestration + CPU fallback) | self-contained CUDA/HIP port using `cub` for scan/sort/RLE | 4 per-round compute kernels (qem/cost/propagate/collapse) on a headless Vulkan device; needs 64-bit shader atomics (`GL_EXT_shader_atomic_int64`, vulkan1.2); host builds CSR adjacency/edges/compaction between rounds | pure C++ QEM loop (`simplify_round`), always compiled | **Runtime** fallthrough inside `decimate_qem()`: tries `decimate_qem_gpu()` (if `TRELLIS_HAVE_GPU_DECIMATE`) → `decimate_qem_vk()` (if `TRELLIS_HAVE_VK_DECIMATE`) → CPU loop, on any GPU failure (no device, wrong-arch kernel image, alloc/submit error) it prints a non-fatal message and falls through — output is unaffected, only speed. |
| NAF cross-scale neighborhood attention (HR texture feature upsampler) | `naf_attn.cu` (+ `naf_gpu.cpp` for the ggml-graph half) | custom kernel indexing the un-upsampled low-res k/v maps directly (avoids materializing up to 4 GB of `k_up`/`v_up` at T=1024) | none | `naf.cpp`'s original CPU implementation, unchanged | **Runtime**, gated by a compile flag *and* a Model backend check: only compiled when `TRELLIS_USE_CUDA`; at runtime `naf_upsample()` calls `naf_gpu_available(naf, S, T, h, w)` — which inspects the loaded `Model`'s actual backend plus shape preconditions — and only takes the GPU path if that passes and `TRELLIS_NAF_CPU` isn't set. The ImageEncoder (Conv2d/GroupNorm/SiLU/pool) itself runs as an ordinary `ggml` graph on the Model's backend in `naf_gpu.cpp`; only the O(T²·C) windowed-attention core is the raw CUDA kernel. |

## Implications for a WebGPU port

(Correction 2026-09-06: `GGML_WEBGPU` does exist on `trellis-patches` and is what this project builds -- `docs/spec/31-webgpu-bringup.md` §3. The rebase analysis below is kept for the record.) Upstream's `ggml-webgpu` backend has matured considerably since this fork's `v0.15.1` base (upstream is now 517 commits / ~7 minor versions ahead, at `v0.23.0`). Any WebGPU work means either (a) rebasing this fork's ggml onto a much newer upstream that carries `ggml-webgpu`, or (b) vendoring `ggml-webgpu` separately and re-applying only the patches that still matter. Per patch:

| Patch | Relevant to WebGPU? | Rebase effort onto newer upstream | Why |
|---|---|---|---|
| `737e88f2` mask strides | No direct CUDA relevance to WebGPU; the underlying *lesson* (32-bit mask strides truncate at scale) generalizes | **Trivial** — drop it; upstream already carries the identical fix (`6d2504d5`) since `v0.15.2`-ish | Superseded, not merely rebased |
| `9937a412` + `323c9f00` binbcast 64-bit | CUDA-only kernel, doesn't port to WGSL as-is; but WebGPU's own binary-broadcast shader will need an equivalent >2^31-element check (WebGPU has its own dispatch-count/buffer-size ceilings, e.g. 2^16 workgroups/dim on some implementations) | **Hard** — upstream's own overflow fix (`6d1eb28b`) rewrote `k_bin_bcast`'s whole parameter set to `uint32_t`+`uint3` fastdiv; the fork's minimal `int64_t` patch does not apply cleanly on top and the fix would need re-deriving against the new signature | Structural divergence from upstream's fix, confirmed by diff |
| `6c9ba07b` fastdiv UB | CPU-side header logic (`common.cuh`), no CUDA syntax; concept (avoid UB in a >2^31 fastdiv helper) could recur if a WebGPU fastdiv-style helper is added | **Trivial** — 6-line, self-contained, isolated function | No upstream churn found on this exact helper between v0.15.1 and master search window |
| `9c9bc0e0` unary 64-bit | CUDA-only; WebGPU's own unary shader would need the same `int64_t`/dispatch-size fix independently | **Trivial** — 4-line, isolated kernel signature change | No conflicting upstream commit on this exact kernel |
| `88c5c7d2`/`2d6d0b0c`/`38d0059c` FA prec (3 commits) | Irrelevant to WebGPU verbatim (CUDA MMA/TILE/VEC kernels don't exist in WGSL), but the *requirement* — SDPA needs FP32 VKQ accumulation for the 1024 cascade's long KV sequences — carries over and a WebGPU FA kernel will need its own FP32-accumulator path from day one | **Hard** — upstream has heavily reworked `fattn-mma-f16.cuh`/`fattn-tile.cuh`/`fattn-vec.cuh` since v0.15.1 (PDL, XOR-swizzle K/V smem, sparse-fa for DSV4/GLM, RDNA3 mma changes) — these patches touch the same functions upstream rewrote, so a mechanical rebase will conflict extensively; re-implementing the FP32-accumulator intent against the new kernel structure is closer to a fresh patch than a rebase | Confirmed via `git log` on `fattn-*.cuh` since v0.15.1 showing 6-8 intervening upstream rewrites per file |
| `f33ab068` PAD grid-stride | CUDA-only; WebGPU compute dispatch has its own per-dimension limits (workgroup count ceilings), so an equivalent grid-stride rewrite of any WebGPU PAD shader is likely needed anyway | **Moderate** — self-contained single-kernel rewrite, but must re-verify against upstream's own PAD changes since v0.15.1 (`b07ff832` non-contiguous-src0 support) which the fork's version predates | Function is small and isolated, but upstream moved the surrounding code |

Net: **none** of the 9 patches carry over mechanically to a WebGPU backend (different language/dispatch model entirely), but **all 9 identify real bug classes** (32-bit index/stride overflow at the 1024-cascade's tensor sizes, and FP16 accumulation bias in long-sequence attention) that a from-scratch WebGPU FA/binbcast/unary/pad shader will need to guard against independently — this document is the checklist for that, not a literal patch set to reapply.

## Local patches (`patches/ggml-webgpu/`, applied at configure time)

Added 2026-09-06 on `feat/webgpu-ss`. The submodule pin is **unchanged** (`737e88f2`); the root
`CMakeLists.txt` applies every `patches/ggml-webgpu/*.patch` to the submodule working tree with
`git apply` at configure time for `-DGGML_WEBGPU=ON` builds (idempotent: a patch that already
applies in reverse is skipped; `-DPIXAL3D_GGML_PATCHES=OFF` disables the step; `git -C
thirdparty/ggml checkout .` reverts). `git status` therefore shows `thirdparty/ggml` as
"modified content" in a configured WebGPU checkout -- expected, do not commit the submodule.
Patches 0001/0002 only add `#ifndef` guards around existing constants, so they are upstreamable
and are no-ops unless the build passes the define. Patch 0003 (added on
`feat/webgpu-shape512-flow`) changes encoder/shader behavior for every shape; it is a bug fix
worth upstreaming as-is, and 0004 (`feat/webgpu-sparse-backend-prep`) is the same fix for
`set_rows`. Patch 0005 (`feat/webgpu-shape-decode`) adds one dtype to the `cpy` shader.

| patch | what | why (measured) |
|---|---|---|
| `0001-webgpu-optional-subgroup-matrix-path.patch` | `GGML_WEBGPU_SUBGROUP_MATRIX` (default 1): when 0, never request `ChromiumExperimentalSubgroupMatrix`, so `mul_mat`/`flash_attn` take the `reg_tile` shaders (f16-staged inputs, **f32** accumulation) instead of the subgroup-matrix shaders (**f16** accumulation, per the shader's own TODO) | On the native Dawn/Metal build the f16-accumulating path overflowed DINOv3's attention scores (>65504) to NaN on 3 of 4 fixture views and gave `rel≈1.3e-2` on the 4th; with it off DINOv3 matches PyTorch at `rel 2.3e-4..8.7e-4` (`docs/spec/31-webgpu-bringup.md` §9). Emscripten builds never have the feature, so the browser was already on the good path. Root CMake defines it 0 unless `-DPIXAL3D_WEBGPU_SUBGROUP_MATRIX=ON`. |
| `0002-webgpu-overridable-queue-wait-timeout.patch` | `WEBGPU_RUNTIME_WAIT_TIMEOUT_MS` (default 30000u) becomes overridable | One SS DiT forward is a single graph submission; in Chrome it takes longer than 30 s, and the fixed ceiling aborted the browser run (`ggml_webgpu: Queue wait timed out after 30000 ms`). Root CMake and `web/ss/CMakeLists.txt` set 600000u. |
| `0004-webgpu-2d-dispatch-set-rows.patch` | The `set_rows` encoder dispatches on a 2D workgroup grid via `compute_2d_workgroups` (it launched `CEIL_DIV(threads, WG_SIZE)` workgroups on x only); `set_rows.wgsl` and `set_rows_quant.wgsl` take `@builtin(num_workgroups)` and linearize `gid.x + num_wg.x * WG_SIZE * gid.y`. Added on `feat/webgpu-sparse-backend-prep`. | The limit is on threads, not rows: `threads = rows × ne0/4` (f32/f16 dst, `ne0 % 4 == 0`), `rows × ne0` otherwise, `rows × blocks` (quantized); past `65535 × WG_SIZE` threads (67.1M at WG_SIZE 1024, 16.8M at the 256 spec floor) Dawn rejects the dispatch (`Dispatch workgroup count X (65536) exceeds max compute workgroups per dimension (65535)`) and the native build aborts in the uncaptured-error callback. Reached by 65536 rows of 4096 floats, 4M rows of 64, or the DiT RoPE scatter `[1,64,12,L]` past L = 87381. Regression: `trellis-webgpu-ops` set_rows cases (`docs/PIXAL3D_WEBGPU_OP_GAP.md` §10). |
| `0005-webgpu-cpy-i32-src.patch` | `cpy.wgsl` accepts `SRC_I32` (`SRC_TYPE i32`), `get_cpy_pipeline` maps `GGML_TYPE_I32` sources to it, and `supports_op` admits `CPY`/`CONT` with I32 source **and** I32 destination (F32→I32 was already there). Added on `feat/webgpu-shape-decode`. | The sparse decoder's submanifold conv takes tap `t`'s neighbour indices for an output range as `cont(view_1d(nbr_i32, nr, (t*N + r0)*4))` (`src/sparse.cpp::submconv_range`; the `cont` exists because the Vulkan `get_rows` asserts a zero index offset). The shader lib had F32/F16 sources only, so the very first ConvNeXt graph aborted with `Unsupported src type for cpy shader` (native Dawn) -- and `supports_op` said "unsupported", which in a browser is the §9 silent skip. Regression: `trellis-webgpu-ops` `cont(view) i32` cases at the decoder's own layouts (`docs/PIXAL3D_WEBGPU_OP_GAP.md` §11.1). |
| `0003-webgpu-2d-dispatch-row-ops.patch` | The `soft_max`, `sum_rows`, `row_norm` (NORM/RMS_NORM/L2_NORM), `get_rows`, `concat`, `pad` and `repeat` encoders dispatch on a 2D workgroup grid via the existing `compute_2d_workgroups`; their shaders take `@builtin(num_workgroups)` and linearize `wid.x + num_wg.x * wid.y` (workgroup-per-row shaders, with a row bound check; `sum_rows` gains an `n_rows` param) or `gid.x + num_wg.x * WG_SIZE * gid.y` (element shaders). The `cpy` shader, whose encoder was already 2D upstream, gets the same linearization (it read `gid.x` only). | Any of these ops with more than `maxComputeWorkgroupsPerDimension` (65535) rows / workgroups failed WebGPU validation and was silently skipped; `cont`/`cpy` over 64 Mi elements (256 MB f32) wrote only the first `1/wg_y` of the output (measured 50 % / 66.6 % / 79.9 % garbage at 256 / 512 / 1024 MB). The NAF attention graph needs a 1M-row softmax, a 4M-row `sum_rows`, 82944- and 262144-row `get_rows` and 1 GB `cont`s. Regression test: `trellis-webgpu-ops` (`docs/spec/31-webgpu-bringup.md` §10.2). |

## Summary

- Upstream base: `v0.15.1` @ `4251bf0eba36a032b871038156fde8068da28062` (verified identical between `pwilkin/ggml` and `ggml-org/ggml`).
- 10 commits between the tag and the pin; 9 are genuine fork-authored CUDA patches (1 is an inherited upstream commit, not a fork change).
- All 9 are CUDA-only bug fixes (32/64-bit index/stride overflow in `binbcast`/`unary`/`pad`/FA-mask, plus FP32 FlashAttention-accumulator support on TILE/VEC/MMA) driven directly by TRELLIS.2's res-1024 cascade tensor sizes (~3.7–4.9B-element tensors, ~53k HR FA tokens); every commit message names the triggering shape.
- **No fork-only ggml API** exists or is used — `include/` is unchanged between `v0.15.1` and `HEAD`; `pixal3d.cpp` only relies on ordinary upstream symbols (`ggml_flash_attn_ext_set_prec`, `ggml_pad`, `ggml_pad_reflect_1d`) whose *implementations* the fork fixed.
- One patch (`737e88f2`) is a byte-identical duplicate of a later independent upstream fix (`6d2504d5`) — trivially droppable on any future rebase.
- Estimated total rebase effort onto a newer (WebGPU-carrying) upstream: **skewed hard**, dominated by the 3 FlashAttention-precision commits, which collide with substantial upstream FA rewrites (PDL, XOR-swizzle, sparse-fa) since `v0.15.1`; the binbcast pair is also hard due to a structurally different upstream fix already in place; fastdiv/unary/PAD/mask-stride are trivial-to-moderate or simply drop (mask-stride is superseded outright).
- Custom non-ggml GPU kernels (`deform_conv.cu/.comp`, `decimate_qem.cu/.comp/_vk.cpp`, `naf_attn.cu`) are dispatched outside ggml's scheduler on the default CUDA stream / raw Vulkan queues, selected at compile time (deform_conv, decimate_qem GPU-vs-Vulkan-vs-CPU) or at runtime via a Model-backend check with CPU fallback (`naf_gpu_available()`, `decimate_qem_gpu()`/`_vk()` failure fallthrough) — these will need independent WebGPU/WGSL reimplementations regardless of the ggml fork question.
