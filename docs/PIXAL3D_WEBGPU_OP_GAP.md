# Pixal3D WebGPU operator gap — measured inventory (2026-09-06)

This document replaces `PIXAL3D_OP_SUPPORT_MATRIX.md`'s guesswork with a **measured** operator
inventory of one full `--views` multiview run on `feat/webgpu-bringup`, traced on the CUDA box
(`ssh win`, RTX 4090). Sections 1–4 are facts from the trace. **Sections 1–4's `class` columns are
now filled in** (this pass) by cross-referencing every op instance against the actual WebGPU
backend coverage — `docs/spec/31-webgpu-bringup.md` §4/§5 (verbatim `supports_op` conditions, current
upstream `ggml-org/llama.cpp` @ `74a7c897`) and, where a verdict is version-sensitive, a direct read
of the vendored fork's own `thirdparty/ggml/src/ggml-webgpu/ggml-webgpu.cpp` @ `737e88f2`. Section 5
gives the required per-instance record for every class-B/C/D op; §6 summarizes; §7 orders the
recommended kernel work.

## 0. Classification legend and global caveats

- **A** — already supported by ggml WebGPU: the `supports_op` condition holds for this exact
  dtype/contiguity/shape, **and** the tensor fits under the WebGPU spec-floor
  `maxStorageBufferBindingSize` (128 MiB) with headroom, so no tiling or validation is needed for a
  spec-minimum browser target.
- **B** — supported but needs validation or graph-side accommodation: dtype/op passes
  `supports_op`, but (i) the instance's tensor exceeds the 128 MiB storage-binding floor (and/or the
  256 MiB `maxBufferSize` floor) and needs tiling/chunking or a graph-construction change to be safe
  on a spec-minimum adapter, and/or (ii) semantics need runtime verification (non-contiguous/permuted
  srcs, F16/precision risk, subgroup-gated paths, broadcast rules `supports_op` doesn't itself check).
- **C** — missing: no `case` for the op in `ggml_backend_webgpu_device_supports_op`'s switch (falls
  to `default:` → `false`), or the dtype/condition fails outright for this use (e.g. BF16 anywhere).
- **D** — trellis/Pixal3D custom kernel outside ggml's graph (`naf_attn.cu`, `decimate_qem.cu`,
  `deform_conv.cu`) — needs an independent WGSL implementation or a deliberate CPU-keep decision.
- **E** — CPU/WASM by design, no GPU work needed (topology, camera math, GLB, mesh postprocess).

**Version note**: §4 of `docs/spec/31-webgpu-bringup.md` cites *current upstream* (`74a7c897`,
2026-09-05); the vendored fork (`737e88f2`, pinned in this repo, ~900 lines / 2 commits behind) is
additive-only relative to it (`docs/GGML_FORK_DIFF.md` §"Are these genuine patches"; confirmed here
directly by `grep`-ing the fork's own `ggml-webgpu.cpp`: zero hits for `GGML_OP_GROUP_NORM`,
`GGML_OP_PAD_REFLECT_1D`, `GGML_OP_POOL_2D`, `GGML_OP_CONV_3D`, `GGML_OP_IM2COL_3D`,
`GGML_OP_CONV_2D_DW`, `GGML_OP_ARANGE` — every op §4 lists as "no case in the switch" is confirmed
absent from the fork too, not just from upstream's historical state). The two upstream-only
*additions* the fork lacks (`CONV_2D_DW` support, `SWIGLU_CLAMP`) are not used anywhere in this
inventory, so they don't change any verdict below. One divergence *was* found and matters: the
fork's `RMS_NORM`/`NORM`/`L2_NORM` case (`ggml-webgpu.cpp:4265-4269`, this repo's vendored copy) is
`op->type==F32 && src0->type==F32` only — it does **not** carry the `ggml_is_contiguous_rows(src0)`
guard §4's upstream table lists. Every NORM/RMS_NORM instance in this pipeline already receives a
contiguous input (post-`ggml_cont`/post-`mul_mat` output), so no verdict below actually flips on
this — but a future `thirdparty/ggml` sync that pulls in the stricter upstream check must not
silently start rejecting (CPU-offloading) any norm that currently passes on the fork; call this out
explicitly rather than assume the two versions are interchangeable here.

**Global size-precheck caveat (applies to every row below)**: `ggml_backend_webgpu_device_supports_op`
rejects an op outright (`ggml-webgpu.cpp:4060-4076`) if `ggml_nbytes(dst)`, `src0`, or `src1` exceeds
`capabilities.limits.maxStorageBufferBindingSize` — spec floor 128 MiB (134,217,728 B), typical
Chrome/Dawn desktop ~1–2 GiB (`docs/PIXAL3D_WEBGPU_MEMORY.md` §1). This is a **per-tensor-buffer**
ceiling, not a per-op-family one: even a "trivially tileable" elementwise op (SILU, ADD, MUL) is
flatly rejected — and its whole node CPU-offloaded by the scheduler — if its *materialized* tensor
exceeds the limit; ggml does not auto-tile a single op's dispatch across multiple buffer bindings.
This is why many otherwise-trivial ops below are marked **B** purely on size: at native-CUDA
tensor sizes (multi-hundred-MB to multi-GB), most large intermediate tensors in this pipeline need a
**graph-construction** fix (smaller row/voxel/tile chunks — several chunking mechanisms already exist
in the C++ source, tuned for native VRAM budgets in the GB range, not the WebGPU spec floor) before
they satisfy this precheck at all, independent of whether the *op type* is otherwise fully supported.

## 1. Method

**Instrumentation** (additive, off by default, no behavior change when unset):

- `include/graph_dump.h` / `src/graph_dump.cpp`: `trellis_graph_dump(tag, ggml_cgraph*)`. Gated on
  env `TRELLIS_DUMP_OPS` (a file path, or unset stderr fallback if set-but-empty); a no-op
  `getenv` check otherwise. Walks every graph node and appends one `[gd] tag=... node=i/n op=...`
  line (op name, `GGML_OP_UNARY`/`GGML_OP_GLU` sub-op, every non-null `src[k]` and the `dst` as
  `type[ne0,ne1,ne2,ne3]{cont|noncont|view-cont|view-noncont}(bytes)`, and a best-effort decode of
  `op_params` for `FLASH_ATTN_EXT`/`SOFT_MAX`/`ROPE`/`PAD`/`IM2COL[_3D]`/`CONV_2D`/`CONV_3D`/
  `POOL_2D`/`NORM`/`RMS_NORM`/`GROUP_NORM`), then one `[gd-summary]` line per graph (op histogram,
  node count, largest single tensor in the graph).
- Call sites added (one line each, right after the last `ggml_build_forward_expand` and before
  `ggml_gallocr_alloc_graph`/`ggml_backend_graph_compute`), tagged descriptively:
  - `src/flow_runner.cpp` `DitRunner::DitRunner` (line ~57) → tag `dit_N<tokens>_dcond<Lc>_proj<0|1>`
    (covers both the dense SS DiT and every sparse SLAT DiT — dense vs. sparse is distinguished by
    the token count matching `R³` or an arbitrary sparse coord count).
  - `src/dinov3.cpp` `dinov3_encode` (line ~106) → `dinov3_S512` / `dinov3_S1024`.
  - `src/ss_decoder.cpp` `run_seg` (line ~68, now takes a `tag` param) → `ss_dec_seg{1,2,3}_res{16,32,64}`.
  - `src/shape_decoder.cpp` `run1` (line ~40, now takes a `tag` param), threaded through
    `linear_rows` and the ConvNeXt-stage graph in `decode_unet` → `{shape_dec,tex_dec,
    sparse_upsample}_{from_latent,convnext_stage<i>,output_layer}_N<N>`.
  - `src/sparse.cpp` `GraphRun::run` (line ~161, now takes a `tag` param), threaded through
    `sparse_c2s`'s two graphs (subdiv-logits graph, conv1→C2S→conv2 graph) → `{kind}_c2s_stage<i>_
    {subdiv,conv}_N<N>[_M<M>]`; `sparse_c2s`'s signature gained a trailing `const char* tag = "c2s"`
    parameter (`include/sparse.h`).
  - `src/naf_gpu.cpp` `naf_upsample_gpu` (line ~146) → `naf_encoder_S<Sp>_T<T>`.
- `birefnet.cpp` was **not** instrumented — confirmed dead on the `--views` path (see §3).

**Build** (remote, CUDA): files scp'd individually to
`/mnt/hdd1/pixal3d/pixal3d.cpp/{src,include}/`; `CMakeLists.txt`'s `add_library(trellis_core ...)`
source list amended remotely with `sed -i '/src\/transforms_json.cpp/a\  src/graph_dump.cpp'` (no
overwrite). `cmake --build build-cuda --target trellis-cli -j 16` — clean build, no warnings from
the new file.

**Run**:

```
cd /mnt/hdd1/pixal3d/out
TRELLIS_DUMP_OPS=/mnt/hdd1/pixal3d/out/ops_dump.txt \
  ../pixal3d.cpp/build-cuda/trellis-cli \
  --views /mnt/hdd1/pixal3d/Pixal3D/assets/mv_images/example \
  --models /mnt/hdd1/pixal3d/gguf --seed 42 --res 1024 ops_trace.glb \
  > ops_trace.log 2>&1
```

V=4 views, res=1024 (full HR cascade: SS → shape-512 → shape-1024 → texture-1024 → GLB). Wall time
429.3s. GPU was otherwise idle (one unrelated long-idle ComfyUI server holding 2.4 GB, not a
competing trellis job). Outputs: `/mnt/hdd1/pixal3d/out/ops_trace.log` (pipeline log),
`/mnt/hdd1/pixal3d/out/ops_dump.txt` (10.8 MB, 70,318 lines, 80 dumped graphs), `ops_trace.glb`/
`.ply`. Nothing was committed; the six touched source files plus `graph_dump.{h,cpp}` and the
`CMakeLists.txt` one-liner are the total diff (local repo).

## 2. Per-stage graph inventory

Each table lists the ops present in that stage's graph(s), the **largest instance** seen across
all invocations of that stage in this run (4 views for per-view stages), `count` = occurrences of
that op within **one** graph build. Shapes are ggml `ne` order (`ne0` fastest / innermost).
Classification (`class` column, A-E per §0's legend) is filled in per instance below; an op's class
can and does differ across rows when the same op is used with different dtypes/shapes in different
stages (see §0's global size-precheck caveat for why the same op/dtype pair often lands A at one
stage's tensor size and B at another's).

### DINOv3 @ 512 (`src/dinov3.cpp:57` `dinov3_encode`, tag `dinov3_S512`, 8 builds = 4 views × {SS cond, shape-512 cond})

1410 nodes/graph. Ntok=1029 (5 prefix + 32×32 patches).

| op (+sub-op) | file:line | count | largest shape (in→out) | dtype | layout | bytes | notes | class |
|---|---|---|---|---|---|---|---|---|
| IM2COL | dinov3.cpp:80 (`ggml_conv_2d` patch embed) | 1 | in f32[512,512,3,1] → f16[768,32,32,1] | f16/f32 | cont | 3.1 MB | patch_embed k=16 s=16 p=0 | **A** |
| MUL_MAT (linears: patch-embed GEMM, QKV/out proj, MLP fc1/fc2) | dinov3.cpp:17-21 `lin`, called :48,54,96,98 | 121 | [4096,1024,1,1]×[1024,1029,1,1]→[4096,1029,1,1] | f32 | cont | 16.9 MB (S512) / ~67 MB (S1024) | ordinary per-token linear, F32×F32 | **A** |
| MUL_MAT (attn QKᵀ score, non-FA) | dinov3.cpp:40 `sdpa` | 24 | [64,1029,16]×2 → [1029,1029,16,1] | f32 | cont | 67.8 MB (S512) / **1.076 GB (S1024)** | plain O(L²) attention, no FlashAttention | **A** at S512 (fits the 128 MiB floor); **B** at S1024 — 1.076 GB blows both spec floors, 4th-largest tensor in the run; needs query-chunking (dit.cpp's exact-SDPA pattern) or FA rerouting (§5/§7 item 1) |
| SOFT_MAX | dinov3.cpp:40 `sdpa` | 24 | [1029,1029,16,1] | f32 | cont | 67.8 MB (S512) / **1.076 GB (S1024)** | `scale=0.125,max_bias=0`; consumes the oversized score tensor above | **A** at S512; **B** at S1024 (same size issue/fix) |
| NORM | dinov3.cpp:22 `ln` | 49 | [1024,1029,1,1] | f32 | cont | 4.2 MB (S512) / 16.8 MB (S1024) | `eps=1e-5`; final LN non-affine; contiguous input throughout | **A** |
| UNARY:GELU_ERF | dinov3.cpp:97 (MLP) | 24 | [4096,1029,1,1] | f32 | cont | 16.9 MB (S512) / ~67 MB (S1024) | exact-erf GELU; `GELU_ERF` in supported UNARY list | **A** |
| CONCAT | dinov3.cpp:87 (cls‖reg‖patches), :33 `rope_half` (`[-x2,x1]`) | 50 | [1024,5,1,1]+[1024,1024,1,1] → [1024,1029,1,1] | f32 | cont | 4.2 MB (S512) / 16.8 MB (S1024) | prefix-token concat; F32 dtype supported | **A** |
| RESHAPE/VIEW/PERMUTE/CONT | dinov3.cpp:38-44 (head split), :31-32 `rope_half` | 126/168/98/266 | up to [3072,1029,1,1] | f32 | mixed (many `view-noncont`→`CONT`) | up to 12.6 MB (S512) / ~50 MB (S1024) | RESHAPE/VIEW/PERMUTE unconditionally `true`; CONT realized as CPY F32→F32 (supported) | **A** |
| ADD / MUL / SCALE / CPY | dinov3.cpp:19 (bias-add), 24 (LN affine), 33-34 (RoPE), 85-86 (cls/reg f16→f32 cast) | 217/192/48/2 | up to [4096,1029,1,1] | f32/f16 | cont | up to 16.9 MB (S512) / ~67 MB (S1024) | bias-add/LN-affine/RoPE all F32-F32; cls/reg cast is CPY F16→F32 (supported set) | **A** |

**Contrast with DiT's RoPE (below): DINOv3's cos/sin tables are host-precomputed** (`rcos`/`rsin`
built in plain C++, `dinov3.cpp:60-71`) **and uploaded as ordinary input tensors** (`gcos`/`gsin`,
lines 76-77) — no `ggml_arange` anywhere in this file. This RoPE path is already in the
WebGPU-friendly shape; only the attention score tensor at S=1024 needs work.

At S=1024 (tag `dinov3_S1024`, 8 builds = 4 views × {shape-1024 cond, tex-1024 cond}): identical
graph shape, Ntok=4101 (64×64 patches); the **SOFT_MAX/MUL_MAT attention score tensor scales to
[4101,4101,16,1] = 1.076 GB** — the single largest DINOv3 tensor in the run, and (see §headline)
the 4th-largest tensor overall. This is the plain quadratic-memory attention path, not FlashAttention.

### SS conditioning (host, `src/pixal3d_cond.cpp:39` `pixal3d_cond_ss` + `src/proj_grid.cpp`)

No ggml graph — pure CPU. Per view (V=4): ImageNet-normalize [3,512,512] (`pixal3d_cond.cpp:13`),
call `dinov3_encode`, reshape patch tokens to CHW (`patch_tokens_to_chw`, cond.cpp:27), compute the
4×4 relative camera matrix (`mv_calc_mats`, proj_grid.cpp:173, Gauss-Jordan double-precision 4×4
inverse), then `proj_grid_sample` (proj_grid.cpp:156): for R=16 → 4096 grid points × 1024 channels,
bilinear-sample (double-precision accumulate) the 32×32 DINOv3 feature map → `[4096,1024]` f32
(16.8 MB), averaged in-place across views. Classification: **CPU (host/WASM), by design** — no TBD.

### SS flow DiT, dense (`src/dit.cpp:289` `build_dit_dense` via `src/flow_runner.cpp:39`, tag `dit_N4096_dcond5_proj1`, N=4096=16³, 1 build, 12 sampler steps × ~1.8 fwd/step = 22 forwards but 1 graph)

4135 nodes/graph, 30 blocks, `d_model=1536`, `n_heads=12`, `head_dim=128`, proj_attn ON (`d_proj=1024`).

| op (+sub-op) | file:line | count | largest shape | dtype | layout | bytes | notes | class |
|---|---|---|---|---|---|---|---|---|
| FLASH_ATTN_EXT (default path) | dit.cpp:130,141 `sdpa` | 60 (30 self + 30 cross) | q f32[128,4096,12,1], **k/v bf16**[128,4096,12,1], mask f16[4096,4096,1,1] → f32[128,12,4096,1] | mixed f32/**bf16**/f16 | cont | 33.5 MB (mask) | `scale=0.0883883, prec=10 (F32 accum)`; `prep_kv` casts K/V to **BF16 by default** (`fa_fast` is false unless `TRELLIS_FA_FAST=1`) | **C** — confirmed identical in both upstream and the vendored fork (`ggml-webgpu.cpp:4208-4212`): `FLASH_ATTN_EXT` requires `src1`(K)/`src2`(V) ∈ {F32,F16,Q4_0,Q8_0}; **no BF16 anywhere in the WebGPU backend**. This is the default, load-bearing path — see §5 |
| FLASH_ATTN_EXT (`TRELLIS_FA_FAST=1`, F16 K/V) | dit.cpp:121,130 | same call sites | same shapes, K/V **f16** instead of bf16 | mixed f32/f16/f16 | cont | 33.5 MB (mask) | opt-in fallback; code comment: *"F16 K/V can overflow on HR activations (the reason BF16+F32 is the default)"* | **B** — dtype passes the FA gate, but gated further on `capabilities.supports_subgroups`+alignment (subgroup/tile path selection, `ggml-webgpu.cpp:4226-4262`) and on precision risk (F16 range) at HR — needs validation, not just a flag flip |
| FLASH_ATTN_EXT (`--no-fa` exact path: `MUL_MAT`+`SOFT_MAX_EXT`, chunked) | dit.cpp:153-182 `sdpa` | n/a (replaces FA node) | `[Lk,nq,nh]` score chunk, capped at `kAttnChunkBytes=1 GiB` | f32 | cont | up to 1 GiB/chunk | correctness oracle; already chunked, but tuned to native VRAM (1 GiB), not the WebGPU floor | **B** — MUL_MAT/SOFT_MAX both dtype-supported (F32), but the existing 1 GiB chunk budget (`kAttnChunkBytes`, `TRELLIS_ATTN_CHUNK_MB` override already exists) exceeds both spec floors and must be re-tuned to ~128 MiB for a spec-minimum target — a parameter change, no new kernel |
| REPEAT (FA pad mask) | dit.cpp:83 `build_pad_mask` | 2 | [4096,1,1,1] → [4096,4096,1,1] f16 | f16 | cont | 33.5 MB (SS) / ~38.6 MB (shape-512, est.) / **619.5 MB (HR, N=17484)** | materializes the whole FA pad mask by broadcast-repeat, shared by all 30 blocks | **A** at SS/shape-512 (fits 128 MiB floor); **B** at HR — 619.5 MB blows both floors; candidate for a WebGPU-side broadcast (no materialization) or tiling — §5/§7 item 1 |
| MUL_MAT | dit.cpp:24 `lin` (qkv/out/mlp/adaLN linears) | 245 | [1536,8192,1,1]×[1536,4096,1,1] → [8192,4096,1,1] | f16×f32→f32 | cont | 134.2 MB (SS) / 143.7 MB (shape-512) / **572.9 MB (HR)** / 1.61 GB (1536-token cascade cap) | MLP up-projection (1536→8192) is the largest GEMM | **B** everywhere — dtype passes (F16×F32 in the MUL_MAT rule), but (i) size is at/over the 128 MiB floor even at SS (134.2 MB vs. 134.22 MB floor — essentially exactly at the line) and far over at HR, needing tiling; (ii) the measured F16-staged F32 GEMM accuracy on WebGPU/Dawn is `rel≈1.3e-3` (`docs/spec/31-webgpu-bringup.md` §7, Apple M1 Pro), which fails the project's own `1e-3` tolerance and needs re-validating against the massive-activation sensitivity noted in spec 30 §5 |
| UNARY:GELU | dit.cpp:274 (MLP) | 30 | [8192,4096,1,1] | f32 | cont | 134.2 MB (SS) / **572.9 MB (HR)** / 1.61 GB (cascade cap) | tanh-approx GELU (`ggml_gelu`, distinct from DINOv3's `gelu_erf`); `GELU` in supported UNARY list | **A** dtype-wise; **B** on size — same MLP-hidden tensor as the MUL_MAT above, same tiling need |
| RMS_NORM | dit.cpp:39 `rms_gamma` (q/k norm) | 120 (4/block × 30) | [128,12,4096,1] | f32 | view-cont | 25.2 MB (SS) / ~100 MB (HR, est.) | `eps=1e-12`; MultiHeadRMSNorm ×γ; input already contiguous (post `ggml_view`+reshape) | **A** — fork's `RMS_NORM` case (`ggml-webgpu.cpp:4265-4269`) doesn't even carry upstream's `contiguous_rows` guard, and the input is contiguous either way (see §0 version note) |
| ARANGE | dit.cpp:57,58 `apply_rope`, :73 `build_pad_mask` | 122 | `[hd]`=[128] (rope idx) / `[Lk_pad]` up to ~17,664 | f32 | cont | trivial (≤70 KB) | builds the even/odd RoPE index arrays and the pad-mask ramp fresh on **every** call (30 blocks × 2 for self-attn RoPE + 2 for the two pad masks) | **C** — `GGML_OP_ARANGE` has **no case at all** in the switch (confirmed absent in both upstream §4 and the fork by direct grep); every one of the 122 occurrences forces a CPU fallback + graph split. Tensors are tiny, but this is 122 sync points per DiT graph build — see §5/§7 item 1 (host-precompute + upload, exactly like DINOv3 already does) |
| SET_ROWS / SUB / UNARY:STEP / CPY | dit.cpp:53-61 `apply_rope` (SET_ROWS,SUB), :74-75 `build_pad_mask` (STEP,CPY cast) | 120/60/2/362 | up to `[1,128,12,4096]` | f32/i32/f16 | cont/view-cont | up to 25.2 MB (SS) / ~100 MB (HR) | SET_ROWS scatters rotated even/odd pairs; STEP builds the mask ramp; CPY realizes the F32→I32 / F32→F16 casts | **A** — SET_ROWS (`op∈{F16,F32,Q8_0,Q4_0}`, `src0==F32`, `src1∈{I64,I32}`), SUB (F32/F32), STEP (UNARY, F32/F32), CPY (F32→I32 and F32→F16, both in the supported set) all pass dtype-wise; downstream of the CPU-forced ARANGE above but each node itself is fine |
| PAD | dit.cpp:129 `prep_kv` | 60 | [128,5,12,1]→[128,256,12,1] | f32 | cont | 1.6 MB | zero-pads K/V key dim to 256-multiple for FA tiling, **before** the BF16/F16 cast | **A** — `PAD` requires `op->type==F32 && src0->type==F32`; the current code order (pad-then-cast) already satisfies this. Porting note: padding *after* the cast (BF16/F16 tensor) would silently fail this gate — preserve the pad-before-cast order |
| NORM | dit.cpp:30 `layernorm` | 91 | [1536,4096,1,1] | f32 | cont | 25.2 MB (SS) / ~100 MB (HR) | `eps=1e-6`; affine on norm2 only, non-affine elsewhere (adaLN modulation) | **A** (see §0 version note on the fork's looser NORM guard — moot here, input already contiguous) |

At sparse N=4394 (shape-512 flow, tag `dit_N4394_dcond5_proj1`) and N=17484 (shape-1024 **and**
texture-1024 flow, tag `dit_N17484_dcond5_proj1`, built twice — same shape family, different
weights): identical op set/graph topology, larger tensors — the HR (N=17484) FA pad-mask REPEAT
alone reaches **619.5 MB** (`[17664,17536,1,1]` f16) and the MLP GEMM/GELU reach 572.9 MB. proj_attn
is ON for all three DiTs (`p.d_proj`=1024 for SS, 2048 for both SLAT stages per
`dit_detect_proj_attn`, dit.cpp:281) — `ProjectAttention`'s extra `proj_linear` (`lin`, dit.cpp:263)
folds into the same `MUL_MAT`/`ADD` op types, no separate ggml op.

### SS decoder (`src/ss_decoder.cpp`, 3 sequential graphs, tags `ss_dec_seg{1,2,3}_res{16,32,64}`)

Dense `ResBlock3d`/`Conv3d`/`pixel_shuffle` U-Net, channels-last. `ggml_conv_3d` (Apple: direct;
elsewhere: `IM2COL_3D`+`MUL_MAT`) at res16→32→64.

| stage | file:line | key ops | largest shape | bytes | notes | class |
|---|---|---|---|---|---|---|---|
| seg1 (res16, C:8→1024) | ss_decoder.cpp:27 `conv3d`, :48 `resblock` | IM2COL_3D×10, MUL_MAT×10, NORM/MUL/UNARY:SILU×8, ADD×22 | IM2COL_3D dst f16[13824,16,16,16] | 113.2 MB | `s=1,p=1,d=1` (submanifold-style dense 3×3×3), 2×`ResBlock3d(512)`+2×`Conv3d` | **IM2COL_3D: C** (non-Apple `ggml_conv_3d` path); **CONV_3D: C** (`#ifdef __APPLE__` `ggml_conv_3d_direct`, ss_decoder.cpp:39) — `GGML_OP_IM2COL_3D` and `GGML_OP_CONV_3D` both have **no case in the switch on either platform's lowering** (confirmed absent from the fork by grep); every Conv3d in this file is unsupported regardless of which branch compiles. Gated MUL_MAT is moot until one of these gets a kernel. **NORM/MUL/SILU/ADD (chln, resblock skip): A** — all F32, contiguous (post-`ggml_cont` permutes), small (≤4.2 MB) |
| seg2 (res32, C:128→256) | same, ch=128 | IM2COL_3D×5, MUL_MAT×5, UNARY:SILU×4 | IM2COL_3D dst f16[3456,32,32,32] | 226.5 MB | `pixel_shuffle_3d(scale=2)` (CPU, ss_decoder.cpp:80) between segments | same as seg1: **IM2COL_3D/CONV_3D: C**; other ops **A** |
| seg3 (res64, C:32→1) | same, ch=32 | IM2COL_3D×5, MUL_MAT×5 | IM2COL_3D dst f16[864,64,64,64] | **453.0 MB** | final `chln`+SiLU+`Conv3d(32→1)` → occupancy logits `[64,64,64,1]` | same as seg1: **IM2COL_3D/CONV_3D: C**; other ops **A** — see §5/§7 item 3 for the recommended Conv3D WGSL strategy |

`ss_coords` (ss_decoder.cpp:128, downsample res64→res32 occupancy→active-voxel-coord list) is CPU
host code (no ggml graph) — 4394 active voxels @res32 in this run.

### NAF encoder (ggml, `src/naf_gpu.cpp:115` `naf_upsample_gpu`, tags `naf_encoder_S{512,1024}_T{512,1024}`) + NAF attention (custom CUDA)

Two-branch `Conv2d`+`EncBlock` image encoder (`encoder` k=1, `sem_encoder` k=3 reflect-padded),
concat, optional `AvgPool2d` down to target T. 4 views × 3 stage-configs = 12 builds
(shape-512:S512→T512, shape-1024:S1024→T512, tex-1024:S1024→T1024).

| op (+sub-op) | file:line | count | largest shape (S=1024 case) | bytes | notes | class |
|---|---|---|---|---|---|---|
| PAD_REFLECT_1D | naf_gpu.cpp:43,45 `reflect_pad2d` | 10 | [1024,1026,128,1]→[1026,1026,128,1] | 538.97 MB | applied twice (W then H) per k=3 conv, pad=1 | **C** — `GGML_OP_PAD_REFLECT_1D` has no case in the switch (confirmed on the fork too); needs a small WGSL kernel (edge-mirror index remap into a larger buffer) — §5/§7 item 2 |
| IM2COL | naf_gpu.cpp:55 `conv_bias` (`ggml_conv_2d`) | 10 | in f32[1026,1026,128,1] → f16[1152,1024,1024,1] | **2.416 GB — the single largest tensor in the whole run** | k=3, 128→128, s=1,p=0 (padding pre-applied via reflect) | **B** — IM2COL is dtype-supported (F32/F16), but 2.416 GB exceeds even the "typical Chrome/Dawn desktop" ~2 GiB figure, let alone either spec floor; this must not be materialized as-is on WebGPU. §5/§7 item recommends replacing this im2col-then-GEMM lowering with a direct/tiled conv2d (avoids the [K²Ci,H,W] buffer entirely, mirroring ss_decoder.cpp's Apple `CONV_3D`-direct pattern for 2D) |
| MUL_MAT | naf_gpu.cpp:55 | 10 | same im2col output ×[1152,128,1,1] | 2.416 GB (src) | conv GEMM | **B** — gated on the same oversized IM2COL src above; same fix |
| GROUP_NORM | naf_gpu.cpp:62 `group_norm_affine` | 8 | [1024,1024,128,1] | 536.9 MB | `n_groups=8, eps=1e-5`; affine applied as separate MUL+ADD (ggml's group_norm has no built-in affine) | **C** — `GGML_OP_GROUP_NORM` has no case in the switch (confirmed absent from the fork too); the affine MUL/ADD that follow (naf_gpu.cpp:65-66) are independently **A**, but blocked on this. Small/well-understood kernel (group mean/var reduction) — §5/§7 item 2 |
| UNARY:SILU | naf_gpu.cpp:73,76 `enc_block` | 8 | [1024,1024,128,1] | 536.9 MB | | **A** dtype-wise; **B** on size — 536.9 MB exceeds both spec floors, hits the **global size precheck** (§0) even though SILU is a trivial elementwise op; needs the encoder graph tiled over spatial/channel blocks at the WebGPU spec-floor target |
| CONCAT | naf_gpu.cpp:137 (`e1‖e2`, channel dim) | 1 | 2×[1024,1024,128,1]→[1024,1024,256,1] | 1.074 GB | | **A** dtype-wise (F32); **B** on size — 1.074 GB; recommend restructuring to avoid a full-resolution two-branch concat (e.g. pool each branch first, or fuse the two branches' consumers instead of materializing the concatenated tensor) |
| POOL_2D | naf_gpu.cpp:139 (`pool_k=Sp/T` only when Sp≠T) | 0 or 1 | [1024,1024,256,1]→[512,512,256,1] | 1.074 GB | `pool_op=AVG(1), k=s=2, p=0`; absent when Sp==T (shape-512, tex-1024) | **C** — `GGML_OP_POOL_1D/POOL_2D` has no case in the switch (confirmed absent from the fork too); needs a small avg-pool WGSL kernel, plus tiling given the 1.074 GB size — §5/§7 item 2 |

**NAF cross-scale neighborhood attention** (the "heavy part" per `docs/spec/30-pixal3d-cond.md`
§4) is **not ggml** — see §3. RoPE-apply and the k/v adaptive-pool for the GPU path reuse the exact
CPU reference functions (`naf_rope_apply_inplace`, `naf_adaptive_avg_pool2d` in `src/naf.cpp`) —
these run on the **host**, between the ggml encoder graph and the CUDA attention kernel, operating
on the (small, ≤256×T×T) pooled tensors after CPU readback.

### Shape-512 flow (sparse), shape-1024 flow, texture-1024 flow

Covered under "SS flow DiT" above (same `build_dit_dense`/`sdpa` code path, `p.proj_attn=true`,
`d_proj=2048` for both SLAT stages) — tags `dit_N4394_dcond5_proj1` (shape-512, N=4394 active
voxels @res32) and `dit_N17484_dcond5_proj1` (shape-1024 **and** texture-1024, N=17484 tokens
@res1024-quantized-grid, 2 separate builds with the same shape).

### SparseTensor ops / SparseConv3D / C2S / S2C — shape decoder, texture decoder, and the LR→HR coordinate-upsample cascade

All three consumers (`shape_dec_*` = `src/shape_decoder.cpp:127` `shape_decode`, `tex_dec_*` =
`:141` `tex_decode`, `sparse_upsample_*` = `:134` `shape_upsample` for cascade coord growth only,
`coords_only=true`) share `decode_unet` (`:86`) → 4 `Stage`s of {ConvNeXt block(s) + C2S}, each a
separate ggml graph. Measured on the HR (`shape_dec`/`tex_dec`, N₀=17484→M₃=4,620,540) and the
coordinate-cascade (`sparse_upsample`, N₀=4394→M₃=1,188,765) paths:

| op (+sub-op) | file:line | count/graph | largest shape (shape_dec stage3) | bytes | notes | class |
|---|---|---|---|---|---|---|
| GET_ROWS | sparse.cpp:75 `submconv_range` (27-tap gather), :334 `xs` skip-gather | 108–432/stage (scales with chunk count × 27 taps) | src f32[64,6144000,1,1] (post-subdiv conv2, stage3) → f32[64,3080029,1,1] | 1.573 GB | submanifold-conv neighbor gather; **this is the op WebGPU support hinges on for sparse conv** — indices come from `build_neighbor_table` (CPU, sparse.cpp:23, `unordered_map`-based, tap-major `[27,N]` int32) | **B** — GET_ROWS dtype passes (`src0∈{F32,...}`⇒`op==F32`, index dtype not gated by `supports_op` at all); but (i) 1.573 GB needs chunking well below the existing `kBlockChunkBytes`≈1.5 GB (native-VRAM-tuned) budget to fit the WebGPU floor — a parameter change, already-existing chunk machinery (`TRELLIS_BLOCK_CHUNK_MB`/`TRELLIS_C2S_CHUNK_MB`); (ii) sparse.cpp:71-72's own comment flags a **Vulkan-specific** "index tensor must start at buffer offset 0" assertion (already worked around there via `ggml_cont`) — whether WebGPU's GET_ROWS shader has the same offset restriction is unverified and must be checked before assuming the existing `ggml_cont` guard is sufficient |
| MUL_MAT | sparse.cpp:77 `mul_mat_rows`, :128/130 (ConvNeXt MLP) | 81–232/stage | [128,512,1,1]×[128,768000,1,1]→[512,768000,1,1] | **1.573 GB** | ConvNeXt MLP widen (C→4C); row-chunked (`kBlockChunkBytes`≈1.5 GB budget) to bound peak memory — this chunking is host orchestration around otherwise-plain `MUL_MAT`s | **B** — dtype supported (F32/F16/quantized weight × F32 act all valid), same 1.5 GB→~128 MiB chunk-budget re-tuning need as GET_ROWS above |
| UNARY:SILU | sparse.cpp:129 (ConvNeXt MLP) | 2–16/stage | [512,768000,1,1] | 1.573 GB | | **B** — dtype **A**, size needs the same chunk re-tuning (global size precheck, §0) |
| NORM | sparse.cpp:126 (ConvNeXt rowLN), sparse_c2s norm1/norm2 (sparse.cpp:286-287,320) | 2–16/stage | [64,4620541,1,1] (post-subdiv, +1 sentinel row) | 1.183 GB | `eps=1e-6` throughout sparse ops | **B** — dtype **A** (F32, contiguous outputs of `submconv_range`/`mul_mat_rows`), size needs chunk re-tuning |
| ADD | sparse.cpp:80 (submconv bias+accumulate), :350 (skip-add) | 56–496/stage | [512,768000,1,1] | 1.573 GB | 27-tap accumulation is a chain of `ADD`s, not a fused reduce | **B** — dtype **A**; size needs chunk re-tuning; also subject to the global caveat (§0) that `supports_op` has no explicit broadcast-shape check for the per-channel bias-add pattern — worth a runtime check once a WebGPU build exists |
| PAD | sparse.cpp:316,354 (`hraw`/`out` zero-seed buffer) | 2/graph | [64,3080029,1,1]→[64,4620541,1,1] | 1.183 GB | seeds a big output buffer once, then `CPY`s each chunk in (see below) — avoids an O(chunks) concat-chain that would peak at ~2× the result | **B, structural** — PAD is F32-dtype-supported, but this pattern allocates **one** buffer sized to the full stage output (1.183 GB) and writes into *views* of it; a single WebGPU buffer binding cannot exceed `maxStorageBufferBindingSize` regardless of how many separate `CPY`s write into it — shrinking the per-chunk `CPY` size does **not** fix this, the seed buffer itself must become multiple real `GPUBuffer`s (a genuine graph-construction rework specific to WebGPU, not just a chunk-size parameter) |
| CPY | sparse.cpp:317,355 (chunk write-into-view) | varies (rooted via `roots`) | up to [64,1540511,1,1] | 394 MB | writes not reachable from the graph's single output root, so explicitly rooted via `ggml_build_forward_expand` on each `CPY` | **B** — dtype supported (F32→F32); the per-chunk view itself (394 MB) already exceeds the floor independent of the PAD structural issue above, so both need fixing together |
| REPEAT | sparse.cpp:349 (skip `repeat_interleave(R)` via `ggml_repeat_4d`) | 1–5/graph | [1,16,1000000,1]→[4,16,1000000,1] | 256 MB | S2C skip-connection channel interleave (R=Cout/K) | **B** — dtype **A** (F32 in REPEAT's supported set); 256 MB sits at/over the spec floor, same broadcast-materialization pattern as DiT's FA pad mask (§5/§7 item 1) |
| CONCAT | sparse.cpp:86 (`submconv_pad` sentinel row), :243-244 (C2S new-coord/gather-index construction is host-side; the actual second CONCAT instance is the per-stage sentinel-row append reused at each of the 4 stages) | 1/graph | [128,295876,1,1]+[128,1,1,1]→[128,295877,1,1] | 590.2 MB | appends the zero sentinel row absent-neighbor indices point at | **B** — dtype **A** (F32); 590.2 MB exceeds the floor; recommend allocating the sentinel row as part of the buffer from the start (graph-construction change, §7 item 1) instead of a per-stage CONCAT of the whole feature tensor |

`output_layer` (final LN + `Linear(64,out_ch)`, `shape_decoder.cpp:66-82` `linear_rows`) is chunked
at `kLinearRowChunk=1,000,000` rows — 5 graph builds per decode call (4×N=1,000,000 + 1×N=620,540),
each 3 nodes (NORM/MUL_MAT/ADD), largest `[64,1000000,1,1]` = 256 MB. **Class B** for all three ops
— NORM/MUL_MAT/ADD are dtype-supported (F32), but 256 MB is ~2× the 128 MiB storage-binding floor
and exactly at the 256 MiB `maxBufferSize` floor; `kLinearRowChunk` is native-VRAM-tuned (bounded by
Vulkan's ~2.1M-row dispatch-grid ceiling, per the code comment) and needs a much smaller row cap for
a WebGPU spec-floor target — a pure parameter change, no new kernel (the chunking loop already
exists).

**Host-side, between every stage** (not ggml): `build_neighbor_table` (CPU hash map, rebuilt at
every N), and `sparse_c2s`'s octant-mask→new-coords→gather-index construction (`sparse.cpp:225–273`,
CPU) — sizes at shape_dec stage3: N=1,152,830→M=4,620,540 coords, `nnbr` = 27×4,620,540 int32 ≈
499 MB host RAM. Classification: **CPU (host/WASM), by design.**

### Geometry / GLB (CPU) — see §4.

## 3. Custom / non-ggml operations

| kernel | file | what it computes | inputs/outputs (largest, this run) | CPU fallback? | GPU→CPU readback in this flow? | class |
|---|---|---|---|---|---|---|
| `naf_attn_kernel`/`naf_attn_cuda` | `src/naf_attn.cu:34,90` | Cross-scale 9×9 neighborhood attention (NATTEN-style), one thread per output pixel, 4 heads × 64-dim QK, softmax over 81 logits, weighted sum of 81 low-res V vectors. Indexes the **un-upsampled** pooled k/v maps directly (equivalence argument in `naf_attn.h`) instead of materializing `k_up`/`v_up` (would be up to 4 GB f32 at T=1024). | q f32[256,1024,1024] (RoPE'd), k_pooled f32[256,64,64], v f32[1024,64,64] → out f32[1024,1024,1024] (4 GB f32 at T=1024, largest attention output in the whole pipeline) | Yes — `src/naf.cpp`'s `naf_na2d`+CPU reference path (bit-exact vs. fixture at T=128/512, ~52 s/view @T=512/32-core; **not exercised in this run** since the CUDA path was available) | Yes — `naf_upsample_gpu` reads back `enc_pooled`/`cat` (ggml→host) to run RoPE+adaptive-pool on CPU (reusing `naf.cpp`'s reference functions), then re-uploads q/k_pooled/v into the CUDA kernel (`naf_attn_cuda`'s own `cudaMemcpy`s) | **D** — non-ggml; WGSL port is the highest-risk item per the porting-plan docs (native ggml has no neighborhood-attention op); CPU fallback exists but is far too slow for interactive use (~52 s/view @T=512) — see §5/§7 item 5 |
| `decimate_qem` (GPU) | `src/decimate_qem.cu` | CuMesh-style parallel QEM edge-collapse mesh simplification (Garland-Heckbert quadrics, atomicMin cost propagation, threshold-ladder driver). **Exercised in this run**: `decimate_qem_gpu(target=1000000): V 9,121,621→472,905, F 18,246,528→947,288`. | verts/faces host arrays in/out (mesh-sized: ~9.1M V / 18.2M F in) | Yes — `src/decimate_qem.cpp:203` `decimate_qem`, CPU port, same algorithm | Host↔device round-trip is the entire call (host mesh in, CUDA rounds, host mesh out) | **D, but "keep on CPU" is a legitimate first-pass answer** — non-ggml; the CPU port is already the designated WASM path (per `docs/GGML_FORK_DIFF.md`), decimation runs post-mesh-extraction (topology stage, already CPU/WASM territory), and it costs no *additional* round trip vs. the GPU path (which is itself a whole-mesh host↔device call). A WGSL port is a real but low-priority nice-to-have — §5/§7 item 7 |
| `deform_conv2d` (GPU: `.cu`; CPU: `_cpu.cpp`) | `src/deform_conv.cu`, `src/deform_conv_cpu.cpp` | Modulated deformable conv2d (torchvision `deform_conv2d` v2), used only by BiRefNet's `ASPPDeformable`. | n/a this run | Yes (`deform_conv_cpu.cpp`, std::thread-parallel; Vulkan/HIP compute-shader variants also exist) | n/a | **E** — off-path: `--views` mode never calls BiRefNet (confirmed — `ops_trace.log` shows no BiRefNet stage, and `trellis_run_mv` in `trellis_cli.cpp` has no birefnet call). Listed for completeness only; if a future entry point re-enables BiRefNet, reclassify as **D** with an existing CPU fallback (not a hard blocker even then) |

`GLU` (`GGML_OP_GLU`) and `ROPE` (`GGML_OP_ROPE`) appear **zero times** in the entire 70,318-line
dump — Pixal3D/TRELLIS.2 doesn't use gated-linear-unit MLPs, and RoPE (both DINOv3's 2D and the
DiT's 3D interleaved-pair variant) is hand-built from `RESHAPE`/`VIEW`/`MUL`/`SUB`/`ADD`/`SET_ROWS`/
`ARANGE` rather than `ggml_rope_ext` — worth knowing before assuming `GGML_OP_ROPE` WebGPU coverage
matters here at all.

## 4. Host-side (CPU) operations that would stay on WASM

| stage | file:line | what | size (this run) |
|---|---|---|---|
| Camera math | `proj_grid.cpp:110,173` | 4×4 c2w/w2c inverse (Gauss-Jordan, double), relative calc-mat per view | tiny (16 floats/view × 4 views) |
| ProjGrid point generation + bilinear sample | `proj_grid.cpp:86,156` | grid meshgrid + rotation (host loop, R³ points), `grid_sample`-equivalent bilinear fetch | up to 64³=262,144 points × 2048 ch (tex-1024 stage) |
| Pixal3D cond orchestration | `pixal3d_cond.cpp:13,27,39,83` | ImageNet-normalize, patch-token→CHW transpose, multiview average-fusion accumulation | `[1024,64,64]` per view (S=1024) |
| SS occupancy → coords | `ss_decoder.cpp:128` `ss_coords` | res64→res32 max-pool-like downsample + active-voxel list build | 4,394 coords this run |
| Sparse coordinate hashing / neighbor maps | `sparse.cpp:22` `build_neighbor_table` | `unordered_map`-based 3×3×3 neighbor lookup, rebuilt at every N transition | up to N=1,152,830 → 27×1,152,830 int32 ≈ 125 MB, per call (8 calls across shape+tex decode) |
| C2S octant mask / index construction | `sparse.cpp:225–273` | subdivision-mask→new-coords, gather-index (`gidx`/`gloc`), chunk rebasing | M up to 4,620,540 |
| FlexiDualGrid mesh extraction | `dual_grid.cpp:17` `dual_grid_to_mesh` | dense-voxel dual contouring → triangle mesh | decoded voxels 4,620,540 → V=4,620,540 F=9,249,484 |
| Narrow-band DC remesh | `remesh_dc.cpp:47` `remesh_narrow_band_dc` | UDF-based dual-contour remesh (BVH-snapped), `remesh_band=1` (MV default) | 11,765,360 active voxels → V=9,153,136 F=18,309,764 |
| Mesh cleanup | `uv_bake.cpp:269,319,467,505,613` (`weld_vertices`,`clean_mesh`,`drop_small_components`,`fill_holes`,`fill_small_holes`) | weld near-dupes, drop floater components, fill small/boundary holes | weld 4,623,407→4,617,751 V; dropped 8 floater comps; +11,838 faces from hole-fill |
| QEM decimate (CPU fallback) | `decimate_qem.cpp:203` | same algorithm as the CUDA kernel (§3), CPU port | not exercised this run (GPU path used); WASM's only option |
| UV chart / atlas bake | `uv_bake.cpp:885` `uv_bake` (xatlas) | chart unwrap + trilinear PBR texel bake with BVH texel-snap (`tri_bvh.cpp`) | 1,240 merge clusters, atlas 4096×4096 (xatlas packed 1148×1137), Vo=621,799 Fo=947,288 |
| GLB serialization | `mesh_glb.cpp:212,278` | glTF/GLB binary write, WebP PBR texture encode | final `ops_trace.glb` |

All rows above are CPU/WASM by design (topology, host orchestration, or established-portable mesh
code) — E, no further record needed.

## 5. Required records for every Class B/C/D instance

Fields: **op/function**; **file:line**; **shapes** (in→out); **dtype**; **layout**; **largest size**
(this run) **vs. limits** (128 MiB storage-binding spec floor / 256 MiB `maxBufferSize` spec floor /
~1–2 GiB typical Chrome-Dawn desktop, per `docs/PIXAL3D_WEBGPU_MEMORY.md` §1); **CPU fallback
possible?**; **would that fallback need GPU readback of a large feature tensor?**; **recommended
strategy**.

### C — missing ops (need a new WGSL kernel before anything downstream of them can run on WebGPU)

| # | op | file:line | shapes | dtype | layout | size vs. limits | CPU fallback? | GPU readback needed? | strategy |
|---|---|---|---|---|---|---|---|---|---|
| C1 | `FLASH_ATTN_EXT` w/ BF16 K/V (DiT default) | dit.cpp:130,141 | q[128,4096,12]/k,v[128,4096,12]bf16→out[128,12,4096] (SS); scales to [128,17664,12] at HR | f32/**bf16**/f16 | cont | mask 33.5 MB→619.5 MB; exceeds floor at HR only, but the op is rejected at **every** size (dtype gate fails first) | Yes — `--no-fa` exact path (dit.cpp:153-182) already exists and is bit-exact to 8e-5 | No — exact path stays GPU-resident, same as FA | Not a CPU-fallback problem: **switch WebGPU builds to F16 K/V** (`TRELLIS_FA_FAST` path) with the precision risk validated (§5 B-side entry below), or add a WGSL BF16-cast-to-F16-at-encode-time shim so FA's storage stays effectively F16 while upstream numerics keep BF16's dynamic-range intent as closely as possible; long-term, upstream has no BF16 WGSL story to inherit (`docs/spec/31-webgpu-bringup.md` §6) so this is either a permanent F16-with-validated-tolerance decision or new kernel work, not a quick port |
| C2 | `ARANGE` (RoPE index build, FA pad-mask ramp) | dit.cpp:57,58,73 | out `[128]` (RoPE idx) / `[Lk_pad]` up to ~17,664 | f32 | cont | trivial (≤70 KB) but 122 occurrences/graph | Yes, trivially — output is a pure function of `(hd, Lk_pad)`, no runtime data dependency | No | **Graph-construction fix, no kernel**: precompute the even/odd RoPE index arrays and the pad-mask ramp on the host once per `(hd, Lk_pad)` config and upload as ordinary input tensors — exactly the pattern DINOv3's RoPE already uses (dinov3.cpp:60-71). Eliminates 122 CPU-fallback graph splits per DiT build. §7 item 1 |
| C3 | `IM2COL_3D` / `CONV_3D` (SS decoder Conv3d, both platform branches) | ss_decoder.cpp:27,39,42 `conv3d` | in [16,16,16,512]→im2col dst f16[13824,16,16,16] (seg1) up to [864,64,64,64] f16 (seg3) | f32 in / f16 im2col dst | cont | 113 MB → 453 MB; exceeds the 128/256 MiB floor from seg1 onward | Yes — the whole SS decoder could stay CPU, but at res16-64 dense voxel grids (up to 262,144 output voxels) this would be materially slower than GPU convs and sits mid-pipeline (blocks everything downstream) | Yes if kept CPU — the seg1→seg2→seg3 chain feeds `ss_coords`, so a CPU decoder needs the SS latent read back once (small: [8,4096]) but keeps the whole ResBlock stack off-GPU | **New WGSL Conv3D kernel** (direct convolution, not im2col-materialize — mirrors the existing Apple/Metal `ggml_conv_3d_direct` code path already in this codebase, just needs a WebGPU implementation of the same op). §7 item 3 |
| C4 | `PAD_REFLECT_1D` (NAF reflect-pad) | naf_gpu.cpp:43,45 `reflect_pad2d` | [1024,1024,128]→[1026,1024,128] (W) then →[1026,1026,128] (H) | f32 | cont | 538.97 MB, exceeds both floors | Yes, cheap — reflect-pad is index remapping, not compute | Only if the whole NAF encoder falls back; in isolation, no | Small WGSL kernel: copy interior + mirror 1-pixel border into a pre-sized larger buffer. §7 item 2 |
| C5 | `GROUP_NORM` (NAF encoder GroupNorm(8)) | naf_gpu.cpp:62 `group_norm_affine` | [1024,1024,128,1] | f32 | cont | 536.9 MB, exceeds both floors | Yes, but would force the whole EncBlock chain off-GPU (GN sits between every conv pair) | Yes if CPU-fallback'd — the conv outputs feeding it are already GPU-resident and would need reading back | Small, well-understood WGSL kernel (per-group mean/var reduction + normalize; affine already split out as separate MUL/ADD, which are already A). §7 item 2 |
| C6 | `POOL_2D` (NAF encoder AvgPool, Sp≠T only) | naf_gpu.cpp:139 | [1024,1024,256]→[512,512,256] | f32 | cont | 1.074 GB, exceeds both floors | Yes, trivial op | Yes if CPU-fallback'd (large tensor) | Small WGSL avg-pool kernel (k=s=2, p=0, the only configuration used); tile given size. §7 item 2 |

### B — supported but needs validation, tiling, or a graph-construction change

| # | op | file:line | shapes | dtype | layout | size vs. limits | CPU fallback? | GPU readback needed? | strategy |
|---|---|---|---|---|---|---|---|---|---|
| B1 | `FLASH_ATTN_EXT` w/ F16 K/V (`TRELLIS_FA_FAST=1`) | dit.cpp:121,130 | same as C1, F16 not BF16 | f32/f16/f16 | cont | same as C1 | Yes — same `--no-fa` path | No | Validate: (i) `capabilities.supports_subgroups` + tile-path alignment (`ggml-webgpu.cpp:4226-4262`) on the target adapter; (ii) F16-range overflow risk at HR activations (the exact reason the CUDA path defaults to BF16, per dit.cpp:100-103 comment) — needs its own golden-tensor comparison before trusting it as the WebGPU FA path |
| B2 | `FLASH_ATTN_EXT` exact path (`--no-fa`, chunked MUL_MAT+SOFT_MAX) | dit.cpp:153-182 | `[Lk,nq,nh]` chunks, `kAttnChunkBytes=1 GiB` | f32 | cont | 1 GiB/chunk ≫ 128 MiB floor | N/A (this *is* the CPU-parity oracle) | No | Retune `kAttnChunkBytes` (already env-overridable via `TRELLIS_ATTN_CHUNK_MB`) to ~64-96 MiB for a spec-floor target — no new code |
| B3 | `REPEAT` (DiT FA pad mask, HR only) | dit.cpp:83 | [17664,1]→[17664,17536] f16 | f16 | cont | 619.5 MB ≫ both floors | N/A (mask is derived, not a fallback target) | No | Check whether `ggml_flash_attn_ext`'s mask argument tolerates a `[Lk_pad,1]` broadcast source instead of a materialized `[Lk_pad,Lq_pad]` tensor (would eliminate the REPEAT entirely, §7 item 1); if not, tile the mask build per Q-chunk |
| B4 | DiT `MUL_MAT`/`UNARY:GELU` (MLP hidden) | dit.cpp:24,274 | [8192,4096]→ up to [8192,49152] (cascade cap) | f16×f32→f32 | cont | 134 MB (SS, at the floor) → 573 MB (HR) → 1.61 GB (cascade cap) | No sensible CPU fallback (would serialize the whole DiT) | N/A | Tile the MLP block over token ranges (same query-chunk pattern already used for exact attention) to keep each `[8192,nr]` intermediate under ~100 MiB; separately validate F16-GEMM accuracy (measured `rel≈1.3e-3` on Apple/Dawn, `docs/spec/31-webgpu-bringup.md` §7) against the massive-activation sensitivity in spec 30 §5 before trusting F16 weights at HR |
| B5 | DINOv3 attn `MUL_MAT`+`SOFT_MAX` (S1024 only) | dinov3.cpp:40 | [64,4101,16]×2→[4101,4101,16] | f32 | cont | 1.076 GB ≫ both floors | Query-chunk like dit.cpp's exact path | No | Preferred: reroute through `FLASH_ATTN_EXT` (dtype trivially passes — DINOv3 is all-F32, no BF16 problem) instead of materializing the O(L²) score matrix at all; fallback: query-chunk |
| B6 | NAF `IM2COL`+`MUL_MAT` (k=3 conv) | naf_gpu.cpp:55 | [1026,1026,128]→[1152,1024,1024] | f32/f16 | cont | **2.416 GB — largest tensor in the run**, exceeds even typical desktop | Yes, but this conv sits inside the encoder chain feeding the attention kernel — a full CPU fallback here reintroduces the exact host round trip §5 of `PIXAL3D_WEBGPU_MEMORY.md` calls out as "must change" | Yes if CPU-fallback'd | Replace the im2col-materialize lowering with a **direct/tiled conv2d** (never build the full `[K²Ci,H,W]` buffer) — highest-value single fix in the NAF encoder, §7 item 2/3 boundary |
| B7 | NAF `UNARY:SILU`/`CONCAT` (post-conv/post-branch) | naf_gpu.cpp:73,76,137 | up to [1024,1024,256] | f32 | cont | 536.9 MB–1.074 GB | Same encoder-chain caveat as B6 | Yes if CPU-fallback'd | Tile the encoder graph over spatial blocks once IM2COL/CONV are fixed; for CONCAT specifically, consider restructuring to avoid a full-resolution two-branch concat |
| B8 | Sparse `GET_ROWS`/`MUL_MAT`/`SILU`/`NORM`/`ADD` (submconv/ConvNeXt) | sparse.cpp:75,77,80,126,129 | up to [512,768000] / [64,4620541] | f32 | cont | 1.18–1.57 GB | No (defeats the purpose of sparse GPU conv; CPU sparse conv at millions of voxels is the slow path this architecture exists to avoid) | N/A | Retune `kBlockChunkBytes`/`kMulMatRowChunk` (already exist, native-VRAM-tuned to ~1.5 GB) down to a WebGPU-floor-safe budget (~64-100 MiB); also confirm GET_ROWS index-tensor offset behavior on WebGPU (§ sparse table note) |
| B9 | Sparse `PAD`+`CPY` (chunk-into-one-buffer pattern) | sparse.cpp:316-317,354-355 | seed [64,4620541] / chunk-view up to [64,1540511] | f32 | cont/view | 1.18 GB seed, 394 MB/chunk | No | N/A | **Structural, not just a size fix**: the single-buffer-with-multiple-CPY-views pattern needs to become multiple real WebGPU buffers once any one chunk's seed buffer exceeds `maxStorageBufferBindingSize` — smaller `CPY` chunks alone don't fix the shared seed-buffer size |
| B10 | Sparse `REPEAT`/`CONCAT` (skip-interleave, sentinel row) | sparse.cpp:349,86 | [1,16,1e6]→[4,16,1e6] / +1 sentinel row | f32 | cont | 256 MB / 590.2 MB | No | N/A | Same broadcast-materialization concern as B3; for CONCAT, allocate the sentinel row as part of the buffer from graph-construction time instead of appending it every stage |
| B11 | `output_layer` chunked linear (NORM/MUL_MAT/ADD) | shape_decoder.cpp:66-82 `linear_rows` | [64,1000000]→[7-or-6,1000000] | f32 | cont | 256 MB/chunk | No | N/A | Shrink `kLinearRowChunk` (currently 1,000,000, Vulkan-dispatch-tuned) to a WebGPU-floor-safe row count — parameter change only |

### D — custom kernels needing an independent WebGPU implementation (or a deliberate CPU-keep decision)

| # | function | file | I/O (largest) | CPU fallback? | GPU readback if CPU? | strategy |
|---|---|---|---|---|---|---|
| D1 | `naf_attn_cuda` (cross-scale neighborhood attention) | naf_attn.cu:34,90 | q[256,1024,1024], k_pooled[256,64,64], v[1024,64,64]→out[1024,1024,1024] (4 GB f32 conceptual, never materialized) | Yes, `naf.cpp` CPU reference (~52 s/view @T=512) | Already round-trips (naf_gpu.cpp:156-180) | WGSL compute kernel, one invocation (or small tile) per output pixel, reading pooled k/v directly — same non-materializing design as the CUDA kernel, since a materialized k_up/v_up would itself blow every WebGPU size limit. §7 item 5 |
| D2 | `decimate_qem_gpu`/`_vk` (QEM mesh decimation) | decimate_qem.cu | mesh-sized (~9.1M V/18.2M F in, this run) | Yes, `decimate_qem.cpp`, same algorithm, already the WASM path | No extra round trip vs. the GPU path (whole-mesh in/out either way) | **Keep on CPU/WASM for the first browser target** — no porting urgency; revisit only if decimation throughput becomes the bottleneck. §7 item 7 |

**Global note for every B row above driven purely by size**: the fix in each case is either (a) a
pure parameter/chunk-size change against machinery that already exists in the C++ source (tuned for
native multi-GB VRAM budgets, not the ~128 MiB WebGPU floor), or (b) a graph-construction change
(avoid materializing a broadcast/concat/im2col buffer at all). None of the B rows need a new WGSL
kernel by themselves — they need the *graph* built differently for a WebGPU target. That is the
central practical distinction between B and C/D in this document.

## 6. Summary

**Counts** (per-instance, splitting rows that classify differently by stage/shape as done above;
custom kernels and CPU/WASM rows counted separately):

| class | count | where |
|---|---|---|
| A | ~34 op-instances | DINOv3 linears/norms/unary/concat/reshape (S512 fully, S1024 mostly); DiT RMS_NORM/NORM/SET_ROWS/SUB/STEP/CPY/PAD; SS decoder's norm/silu/add (not the conv itself); IM2COL/patch-embed |
| B | ~24 op-instances | DiT FA (F16 variant)/exact-path/pad-mask-REPEAT/MLP MUL_MAT+GELU at HR; DINOv3 attention at S1024; every NAF op past the reflect-pad (IM2COL/MUL_MAT/SILU/CONCAT); every sparse-decoder op (GET_ROWS/MUL_MAT/SILU/NORM/ADD/PAD/CPY/REPEAT/CONCAT); `output_layer` linear |
| C | 6 distinct missing ops | `FLASH_ATTN_EXT`-with-BF16 (DiT default), `ARANGE` (DiT RoPE/mask), `IM2COL_3D`/`CONV_3D` (SS decoder), `PAD_REFLECT_1D`, `GROUP_NORM`, `POOL_2D` (all three in NAF encoder) |
| D | 2 custom kernels (+1 off-path) | `naf_attn_cuda` (blocking), `decimate_qem_gpu` (deferrable); `deform_conv2d` off-path (E unless BiRefNet re-enabled) |
| E | ~13 stages/functions | SS conditioning, proj-grid, host RoPE-table math (DINOv3's, already E-shaped), sparse coord hashing/C2S index construction, all of §4 (camera math, dual-grid, remesh, mesh cleanup, UV bake, GLB) |

**C/D list grouped by stage** (the operations that block a from-scratch WebGPU run today):

- **DiT (SS + both SLAT flows)**: C — `FLASH_ATTN_EXT` w/ BF16 K/V (default path, all three DiTs); `ARANGE` (RoPE + pad-mask, 122×/graph).
- **SS decoder**: C — `IM2COL_3D`/`CONV_3D` (every Conv3d, both platform branches, all 3 segments).
- **NAF encoder**: C — `PAD_REFLECT_1D`, `GROUP_NORM`, `POOL_2D`.
- **NAF attention**: D — `naf_attn_cuda` (no ggml op exists for neighborhood attention at all).
- **Mesh postprocess**: D — `decimate_qem_gpu` (CPU-keep candidate, not blocking).
- **DINOv3, sparse decoder, shape_decoder's `output_layer`**: no C/D — all B (size/validation) or A.

**Minimal set for the realistic first browser target (SS + shape-512, per
`docs/PIXAL3D_WEBGPU_MEMORY.md` §4's "Realistic first browser target")**:

- Must fix: DiT `FLASH_ATTN_EXT` BF16 gap (C1/B1 — F16 K/V with validated tolerance, or the exact
  chunked path retuned to the spec floor); DiT `ARANGE` (C2, cheap graph fix); SS decoder's
  `IM2COL_3D`/`CONV_3D` (C3 — this stage runs unconditionally for every generation, at res16-64,
  where tensors are 113–453 MB, still over-floor but far smaller than the HR case); every
  size-driven B at SS/shape-512 scale (DiT MLP hidden 134–144 MB, DINOv3 attention only at S512 so
  already A, sparse decoder tensors scaled to shape-512's ~4,394→~1.19M voxel density rather than
  HR's 4.6M).
- Does **not** need: NAF's `PAD_REFLECT_1D`/`GROUP_NORM`/`POOL_2D`/`naf_attn_cuda` (NAF is a
  texture/HR-conditioning feature — shape-512 untextured needs no NAF pass at all per
  `docs/spec/30-pixal3d-cond.md`); DINOv3 attention tiling (only needed at S1024, and shape-512
  conditioning runs DINOv3 at S512, which is already A).
- **decimate_qem**: can stay CPU/WASM even for the full cascade (D2 verdict).

**Full 1024 cascade (shape-1024 + texture-1024) adds**: DiT MLP tiling at 573 MB→1.61 GB scale;
DiT FA pad-mask REPEAT at 619.5 MB (B3); DINOv3 attention tiling/FA-reroute at S1024 (B5, 1.076 GB);
the entire NAF stack (C4/C5/C6 + D1 `naf_attn_cuda`, since texture conditioning is HR-only); NAF's
2.416 GB im2col conv (B6, the single largest tensor in the whole pipeline); sparse decoder tensors
at full HR density (up to 1.57–1.19 GB, more chunking than shape-512 needs).

## 7. Recommended kernel implementation order

Ordered by dependency (what blocks what), blast radius (how many stages/tensors a fix unblocks),
and size (cheapest wins first):

1. **Graph-construction changes that need no new kernel** — highest value-per-effort, unblocks
   real testing immediately: host-precompute DiT's `ARANGE`-built RoPE index/pad-mask arrays (C2,
   mirrors DINOv3's already-working pattern); retune every existing chunk-budget constant
   (`kAttnChunkBytes`, `kBlockChunkBytes`, `kMulMatRowChunk`, `kLinearRowChunk`) from native-VRAM
   scale (GB) down to the WebGPU spec floor (~64-100 MiB) — B2/B4/B8/B11; investigate whether
   `ggml_flash_attn_ext`'s mask argument supports a broadcast (non-materialized) source to drop the
   FA pad-mask REPEAT (B3/B10) without a kernel. None of this touches WGSL at all, and it is a
   prerequisite for every size-driven B below regardless of what else ships.
2. **Small missing ops**: `GROUP_NORM` (C5), `POOL_2D` (C6), `PAD_REFLECT_1D` (C4). Justification:
   conceptually simple kernels (reduction, pooling, index-remap — no gather/scatter, no cross-voxel
   topology), each unblocks the entire NAF encoder chain (currently every downstream NAF op is
   forced CPU once any one of these three is hit), and none has a dependency on anything else in
   this list.
3. **IM2COL_3D / direct Conv3D for the SS decoder** (C3) — blocks the SS decoder unconditionally
   (every generation runs it), moderate complexity (3D convolution, not just elementwise), and the
   codebase already has a working *native* direct-conv3d code path (`ggml_conv_3d_direct`, Apple
   branch) to use as the reference implementation for a WGSL port, lowering risk relative to writing
   the numerics from scratch. Also fixes NAF's im2col-materialization problem (B6) by the same
   direct/tiled-conv technique in 2D.
4. **Sparse conv gather/scatter** (B8/B9's `GET_ROWS`/`PAD`+`CPY` structural fix) — highest
   remaining tensor sizes (up to 1.57 GB) and the most structurally involved fix (the
   single-buffer-multi-view pattern must become genuinely multiple WebGPU buffers), but is "only"
   parameter/graph work once GET_ROWS's WebGPU-specific offset behavior is confirmed (no unsupported
   op involved) — ordered after Conv3D since it's needed by both `shape_decoder`/`tex_decode` and the
   coordinate-upsample cascade, i.e. broad blast radius, but not needed until a full HR run is
   attempted.
5. **NAF neighborhood attention WGSL** (`naf_attn_cuda`, D1) — highest implementation risk (no
   existing ggml op to model it on, must be written from scratch, and correctness depends on
   replicating the non-materializing on-demand-evaluation design exactly, or every size limit in
   this document gets blown by a materialized 4 GB `k_up`/`v_up`) — ordered after the simpler
   kernels so the team has WGSL authoring experience from items 2-3 first, and because NAF is
   HR/texture-only (not needed for the first browser target per §6).
6. **ProjGrid sampling on GPU** — currently `E` (host bilinear sample,
   `proj_grid.cpp:156`) by design; promoting it to GPU-resident (per
   `docs/PIXAL3D_WEBGPU_MEMORY.md` §5 item 2's "port `proj_grid_sample` to a WGSL compute kernel
   operating directly on the DINOv3 output buffer") removes one GPU→host→GPU round trip per view but
   is optional — the memory doc itself accepts "one round trip per view" as tolerable if pipelined
   across views. Ordered last among the *GPU-porting* items because it's a latency optimization, not
   a correctness/capacity blocker (unlike items 1-5).
7. **Decimation** (D2) — recommend **keep on CPU/WASM** rather than port: the CPU fallback is
   already the designated WASM path, costs no extra round trip relative to the GPU variant (both are
   whole-mesh host↔device calls), and sits in the topology/postprocess stage the architecture docs
   already assign to CPU. Revisit only if decimation throughput measurably gates the browser
   experience after everything else above ships.

---

*Instrumentation: `include/graph_dump.h`, `src/graph_dump.cpp`. Trace log:
`/mnt/hdd1/pixal3d/out/ops_trace.log`. Raw per-node dump:
`/mnt/hdd1/pixal3d/out/ops_dump.txt` (remote, `ssh win`, not copied into the repo).*

## 8. Validated on WebGPU (2026-09-06, `feat/webgpu-ss`, SS stage)

"Validated" here means: the op **executed on the ggml WebGPU backend inside the real Pixal3D
graph** (not merely `supports_op == true`, not merely compiled), and the stage's output matched the
existing PyTorch/CUDA fixtures (`docs/spec/31-webgpu-bringup.md` §9 has the numbers). Every graph
below is also gated at build time by `check_graph_supported()` (`src/trellis_model.cpp`), which
throws on the WebGPU backend if any node fails `supports_op` -- necessary because the WebGPU graph
encoder **silently skips** nodes it has no kernel for (`ggml_webgpu_encode` returns `nullopt`,
leaving the output buffer uninitialized). Counts are per graph, from `TRELLIS_DUMP_OPS` on the
WebGPU build (native Dawn) -- the Emscripten build runs the identical graphs.

| graph (tag) | ops executed on WebGPU (count/graph) | parity result |
|---|---|---|
| DINOv3 @512 (`dinov3_S512`, 1410 nodes) | IM2COL 1 (f32→f16 patch embed), MUL_MAT 145 (f16×f32 and f32×f32, incl. the batched `[64,1029,16]` attention scores/values), SOFT_MAX 24, NORM 49, UNARY:GELU_ERF 24, CONCAT 50, ADD 217, MUL 192 (incl. `[1024]`→`[1024,1029]` broadcast), SCALE 48, CPY 2 (f16→f32 cls/reg), CONT 266, VIEW 168, PERMUTE 98, RESHAPE 126 | per-view tokens vs PyTorch `rel 2.3e-4 … 8.7e-4`, cos ≥ 0.9999998 (native Dawn and Chrome give identical numbers) |
| SS conditioning, device-resident (`pixal3d_cond_ss_gpu_S512_R16_v<i>`, 1429 nodes = DINOv3 + 19) | + GET_ROWS 4 (f32 `[1024,1024]` patch map, I32 `[4096]` indices, src0 is a **view** at token offset 5), MUL 4 (`[1,4096]` weight broadcast), ADD 5, SCALE 2, CPY 2 (`ggml_cpy` into the persistent accumulators, aliasing the ADD's own input), VIEW 2 | `z_proj`/`z_global` vs PyTorch `rel 2.2e-4 / 9.5e-5`, cos 1.0000000; vs the host projection on the same backend: `z_global` bit-exact, `z_proj` max\|d\| 5.7e-6 (f32 vs f64 tap accumulation); V=1/2/4 |
| SS flow DiT, exact SDPA (`dit_N4096_dcond5_proj1` with `--no-fa`, 30 blocks, 3943 nodes, largest tensor 805 MB) | MUL_MAT 365 (f16 weights × f32 activations for every linear incl. `proj_linear`; f32×f32 batched `[4096,4096,12]` scores and `[128,4096,12]` values per attention), SOFT_MAX 60 (scale 0.088, no mask), RMS_NORM 120, NORM 91, UNARY:GELU 30 (tanh) + UNARY:SILU 2 (t-embedder), SET_ROWS 120 (f32 dst, **I32 index views** of the host-built `rope_idx` input -- replaces ARANGE, C2), CPY 120 (f16→f32 RMSNorm gammas), SUB 60 / ADD 605 / MUL 510 (incl. `[1536]` adaLN view_1d broadcasts), SCALE 60, CONT 510 / VIEW 570 / PERMUTE 240 / RESHAPE 480; no CONCAT (one query chunk at N=4096) | block-0 intermediates `rel ≤ 6.0e-4` (msa 4.6e-4, global 6.0e-4, proj_linear 3.2e-4, mlp 4.4e-4); `output rel 9.0e-2, cos 0.99998` -- the same class as CPU/Metal/CUDA (spec 30 §5; CUDA f16-weights gives 0.53) |
| SS sampling (12 Euler/CFG steps, 22 forwards, same serialized noise) | the DiT graph above, re-run 22× with new inputs; sampler arithmetic on the host | final latent vs CUDA exact-SDPA run: `mean\|d\| 4.1e-3, cos 0.99984`; occupancy IoU vs f32 ref **0.9977**, vs bf16 ref **0.9932** (baseline f32↔bf16 0.9945), vs the CUDA production run **1.0000** (identical active-voxel set) |

Not exercised / still open after this stage:

- **FLASH_ATTN_EXT** (B1, F16 K/V tile path): not used -- the SS run takes the exact chunked SDPA.
  At N=4096 the score tensor is 805 MB in one chunk (fits this adapter's 4 GiB binding limit, not
  the 128 MiB spec floor: B2's `kAttnChunkBytes` retune is still pending).
- **BF16** anywhere (C1): still unsupported; the WebGPU build must run with `--no-fa` (or the
  F16-K/V FA variant, unvalidated).
- **ARANGE** (C2): resolved by graph construction for the DiT RoPE (host `rope_idx` input);
  `build_pad_mask` still uses it, which only matters once FA is enabled on WebGPU.
- **IM2COL_3D / CONV_3D** (C3): the SS decoder ran on the CPU backend for the IoU measurement.
- **New hazard found (not in §5)**: the native backend's subgroup-matrix `mul_mat`/`flash_attn`
  shaders accumulate in **f16** and overflowed DINOv3's attention scores to NaN; disabled by
  `patches/ggml-webgpu/0001` (`docs/GGML_FORK_DIFF.md` "Local patches"). The Emscripten build
  never selects that path.
- **Queue-wait ceiling**: upstream's fixed 30 s wait aborts a browser DiT forward;
  `patches/ggml-webgpu/0002` makes it a build-time define (600 s here).

## 9. Validated on WebGPU (2026-09-06, `feat/webgpu-shape512-flow`, Shape-512 stage)

Same meaning of "validated" as §8 (executed on the ggml WebGPU backend inside the real graph,
output matched the fixtures -- `docs/spec/31-webgpu-bringup.md` §10). Backend delta this phase:
`patches/ggml-webgpu/0003` (2D dispatch for row/element-parallel ops, `cpy` `gid.y` fix; without it
every op below with more than 65535 rows was silently skipped and every `CONT` over 256 MB was
partly garbage). No new op type was added to the backend.

| graph (tag) | ops executed on WebGPU (count/graph) | parity result |
|---|---|---|
| NAF (`naf_ggml_S512_T512`, 204 nodes; the same graph inside the conditioning graph) | CONV_2D 10 (**direct**, f16 kernel × f32 input, replaces IM2COL+MUL_MAT), CONCAT 22 (20 mirrored border views = the PAD_REFLECT_1D lowering, e1‖e2, RoPE rotate-half), NORM 8 (**GroupNorm lowering**: 8 rows × 4M elements), UNARY:SILU 8, UNARY:NEG 1, MUL 10 / ADD 19 (affine, bias, RoPE), GET_ROWS 3 (`[256,262144]` block reorder of q, `[256,1024]`/`[1024,1024]` window gathers with 82944 indices), SUM_ROWS 1 (k pooling, 262144 rows of 256; 3 at T=128 where the avg-pool lowering runs), MUL_MAT 2 (batched `[64,81,4,1024]ᵀ[64,256,4,1024]`, `[81,256,4,1024]ᵀ[81,256,4,1024]`), SOFT_MAX 1 (1M rows of 81), SCALE 2, CONT 30 (incl. two 1 GB transposes), PERMUTE 5, TRANSPOSE 3, RESHAPE 55, VIEW 22 -- 204 nodes | T=128: out rel 4.0e-4 vs PyTorch (tol 3e-3); T=512: 5.5e-4 (`s512_naf_hr_v0`) |
| Shape-512 conditioning, device-resident (`pixal3d_cond_slat_gpu_S512_R32_T512_v<i>` = DINOv3 + NAF + 8 GET_ROWS `[1024,32768]` taps + MUL/ADD/SCALE/CPY into three accumulators) | as above | V=4 `z_global` 9.5e-5, `z_proj` lr 2.4e-4 / hr 1.9e-4; V=1 7.2e-5 / 2.7e-4 / 4.1e-4 -- identical digits to the host path |
| Shape-512 flow DiT, sparse (`dit_N4377_dcond5_proj1`, `--no-fa`) | the §8 SS DiT op set at N=4377 (`proj_linear` on the 2048-wide `[lr‖hr]` condition, score tensor `[4377,4377,12]` f32 = 920 MB in one chunk) | block test N=2000: block-0 probes ≤ 8.2e-4, output rel 5.6e-3 cos 0.999998; sampling: spec 31 §10.6 |

Not exercised / still open after this stage: everything in §8's list (FLASH_ATTN_EXT F16 tile
path, BF16, IM2COL_3D/CONV_3D) plus the sparse-decoder ops (§5 B8-B11: no missing op type, but
1.2-1.6 GB tensors and the single-buffer `PAD`+`CPY` pattern). `SET_ROWS` was still 1D-dispatched at
this point (fixed by patch 0004, §10.1). NAF at S=1024 / T=512 (shape-1024 conditioning) would use the
`POOL_2D` lowering with k=2 and d = 8 blocks -- the lowering is validated at k=4 (T=128) and the
block formulation at d = 4 and 16; DINOv3's 1.08 GB score tensor at S=1024 is untiled.

## 10. Backend preparation for the sparse decoder (2026-09-06, `feat/webgpu-sparse-backend-prep`)

Independent of the Shape-1024 model work: no new op type, no decoder port. One backend limitation
fixed (`SET_ROWS` dispatch), regression coverage for the large tensor patterns the sparse decoder
will exercise, and the allocation trace of `docs/PIXAL3D_WEBGPU_MEMORY.md` §8. Adapter for every
number below: Apple M1 Pro, native Dawn 18eb229 (`build-webgpu`, `-DGGML_WEBGPU=ON -DGGML_METAL=OFF`)
and Chrome 152 (`web/ops/`), `maxComputeWorkgroupsPerDimension` 65535,
`maxComputeInvocationsPerWorkgroup` 1024, `maxStorageBufferBindingSize` = `maxBufferSize` = 4 GiB − 4 B.

### 10.1 `SET_ROWS`: root cause and fix (`patches/ggml-webgpu/0004`)

`ggml_webgpu_set_rows` computed `threads` = `rows × ne0/4` (f32/f16 dst with `ne0 % 4 == 0`, the
`VEC4` shader variant), `rows × ne0` (any other `ne0`, e.g. the DiT RoPE scatter with `ne0 = 1`) or
`rows × blocks_per_row[/2]` (quantized dst), then dispatched `CEIL_DIV(threads, WG_SIZE)` workgroups
on **x only** (`ggml_backend_webgpu_build(..., wg_x, 1)`), and both shaders (`set_rows.wgsl`,
`set_rows_quant.wgsl`) indexed by `gid.x` alone. `WG_SIZE` is `maxComputeInvocationsPerWorkgroup`
(1024 here, 256 on a spec-floor adapter), so the op broke once `threads > 65535 × WG_SIZE`:
**67.1M threads on this adapter, 16.8M on a 256-invocation one** -- it is a thread-count limit, not
a row-count one. In rows: 65536 rows of 4096 f32, 4M rows of 64, 22.4M rows of 3, or the DiT RoPE
`[1,64,12,L]` scatter past L = 87381 tokens (at the spec floor: L > 21845, i.e. *inside* the
Shape-1024 range of ~17.5k tokens only by a 20 % margin). Measured on the unpatched build
(`trellis-webgpu-ops` with 0001-0003 only): `[4096,65535]` (65535 workgroups) bit-exact,
`[4096,65536]` (65536 workgroups) →

```
ggml_webgpu: Device error! Reason: 2, Message: Dispatch workgroup count X (65536) exceeds max
compute workgroups per dimension (65535). - While encoding [ComputePassEncoder].DispatchWorkgroups(65536, 1, 1)
```

followed by `GGML_ABORT` from the backend's uncaptured-error callback (native Dawn aborts; in a
browser the same validation error drops the whole command buffer, i.e. the §9 silent skip).

Fix (same mechanism as 0003 and as the upstream `cpy`/binary encoders): the encoder calls
`compute_2d_workgroups(CEIL_DIV(threads, WG_SIZE), maxComputeWorkgroupsPerDimension, wg_x, wg_y)`
and both shaders take `@builtin(num_workgroups)` and linearize
`tid = gid.x + num_wg.x * WG_SIZE * gid.y`; the bound check, the row/column decomposition and the
I64 high-word check are unchanged (`tid` replaces `gid.x` verbatim), so semantics are identical.
`tid` stays in u32: the grid is over-provisioned by at most `wg_y − 1 < 65535` workgroups, so
`tid < threads + 65535 × WG_SIZE`, which cannot wrap while `threads` itself fits the encoder's
existing `uint32_t` (4.29 G threads = 17 GB of f32 rows -- beyond any buffer this backend can
bind). No CUDA/Vulkan/Metal file is touched (the patch is `src/ggml-webgpu/` only, applied at
configure time like 0001-0003); the shaders are shared by the native Dawn and Emscripten builds.

### 10.2 Regression coverage (`trellis-webgpu-ops`, WebGPU vs ggml-cpu, identical inputs)

`set_rows` cases write into a prefilled dst (`ggml_set_rows` returns a view of it), so the
comparison covers written **and untouched** rows; indices are a deterministic permutation with row
0 and row n−1 forced in (boundary), or, for the dup case, `hash(i) % n_dst` with every writer's
source row derived from its target row (ggml leaves the winner among duplicate indices
unspecified -- the CPU reference is thread-order dependent -- so the test makes all writers agree).
Native Dawn, patch 0004 applied; `--only`/`--budget-mb` select subsets:

| case (src shape, dst rows) | workgroups (WG 1024) | out | max\|d\| | mean\|d\| | before 0004 |
|---|---|---|---|---|---|
| `[1024,65534]` → 65534 rows | 16384 | 256 MB | 0 | 0 | pass |
| `[4096,65535]` → 65535 rows | **65535** (limit) | 1024 MB | 0 | 0 | pass |
| `[4096,65536]` → 65536 rows | **65536** | 1024 MB | 0 | 0 | **abort** (above) |
| `[64,4194304]` → 4M rows | 65536 | 1024 MB | 0 | 0 | abort |
| `[16,16777216]` → 16M rows | 65536 | 1024 MB | 0 | 0 | abort |
| `[3,22369622]` → **22.4M rows** (`ne0 = 3`, non-vec4 path) | 65536 | 256 MB | 0 | 0 | abort |
| `[64,4620541]` → 4620542 rows (sparse-decoder stage-3 `[Cout, M]` layout) | 72196 | 1128 MB | 0 | 0 | abort |
| RoPE `[1,64,12,87384]` → `[1,128,12,87384]`, idx `[64]` broadcast over (nh, L) | 65538 | 512 MB | 0 | 0 | abort |
| `[4096,65536]` → **f16** dst | 65536 | 512 MB | 0 | 0 | abort |
| `[4096,65536]`, **I64** idx (error-buffer variant) | 65536 | 1024 MB | 0 | 0 | abort |
| dup idx `[256,200000]` → 50000 rows (4 writers/row avg) | 13 | 49 MB | 0 | 0 | pass |
| **Q8_0** dst `[256,70000]` (quant shader, `PAIR_BLOCKS`) | 274 | 18 MB (dequantized) | 0 | 0 | pass |

Maximum row count exercised: 22,369,622 rows (dst 22.4M rows, `ne0 = 3`); largest workgroup
count 72,196 (`wg_y = 2`). Every case is bit-exact, including the f32→f16 conversion and the Q8_0
quantization (the WGSL and CPU quantizers round identically on these inputs). The quantized shader
cannot reach the limit on this adapter within memory (it needs 67M rows × ≥ 32 elements ≥ 8.6 GB of
f32 source), so its 2D path is exercised only at `wg_y = 1`; it is the same three-line change as
the f32 shader. Browser results: §10.4. The 17 pre-existing cases (0003 shapes, NAF mul_mats,
1 GB `cont`) still pass with identical numbers (spec 31 §10.2 table).

### 10.3 Sparse-decoder `PAD` → `CPY`(view) pattern at real size

`sparse_c2s` seeds one `[Cout, M(+1)]` buffer with `ggml_pad` from chunk 0 and writes every later
chunk with `ggml_cpy` into a row-offset view, the copies rooted explicitly (§5 B9). Reproduced
with the res-1024 stage-3 dims from the CUDA trace (§5: `[64,3080029]` → `[64,4620541]`, 394 MB
chunk views) and scaled up:

| case | seed / chunks | out | result |
|---|---|---|---|
| `[64,100000]`, 4 chunks | pad 30000 + 3 cpy | 24 MB | bit-exact |
| `[64,4620541]` (c2s `hraw`) | pad 3080029 + cpy 1540512 (394 MB view at +788 MB) | 1128 MB | bit-exact |
| `[64,4620541]` (conv2 `out`) | pad 1000000 + 4 cpy of 1M rows (256 MB each) | 1128 MB | bit-exact |
| `[64,8388608]` | pad + 7 cpy of 1048576 rows | **2048 MB** | bit-exact |
| `[64,12582912]` | pad + 11 cpy of 1048576 rows | **3072 MB** (6 GB resident with the chunk inputs) | bit-exact |
| `cont(view_2d)` `[64,1540512]` at byte offset 788 MB of a `[64,4620541]` input | | 376 MB | bit-exact |
| `cont(view_2d)` `[512,300000]` at byte offset 614 MB of a `[512,600000]` input (ConvNeXt chunk read) | | 586 MB | bit-exact |

Finding: with 0003's `cpy` fix the pattern is correct at every layout the decoder produces on this
adapter, up to a 3 GiB seed buffer; no new bug reproduced, nothing to fix. The remaining ceiling is
the one already recorded in §5 B9 and confirmed in code: `supports_op` rejects any node whose
`ggml_nbytes` exceeds `maxStorageBufferBindingSize` (`ggml-webgpu.cpp` ~L4099), so a `[64, M]` f32
seed buffer is unsupported from M = 16,777,216 (exactly 4 GiB > 4 GiB − 4 B) here, and from
M = 524,288 (128 MiB) on a spec-floor adapter -- the buffer split is a graph-construction change,
not a kernel bug, and stays deferred (`docs/PIXAL3D_WEBGPU_MEMORY.md` §8).

### 10.4 Browser (Chrome 152, `web/ops/`)

Same source file compiled to WASM (`web/smoke/CMakeLists.txt` target `pixal3d-webgpu-ops-wasm` →
`web/ops/pixal3d_ops.{js,wasm}`, entry `webgpu_ops_run(args)`, page `web/ops/index.html?args=…`,
driver `web/ops/run_playwright.js`, Chrome 152.0.7977.82, JSPI). The wasm heap holds the CPU
reference and both result vectors, so the browser run is budgeted (`--budget-mb 1200 --only
set_rows --only [64,100000]`; 12 s wall):

| case | workgroups | out | max\|d\| | mean\|d\| | result |
|---|---|---|---|---|---|
| `[1024,65534]` | 16384 | 256 MB | 0 | 0 | PASS |
| `[3,22369622]` (22.4M rows, `ne0 = 3`) | **65536** | 256 MB | 0 | 0 | PASS |
| RoPE `[1,64,12,87384]`, idx `[64]` broadcast | **65538** | 512 MB | 0 | 0 | PASS |
| dup idx `[256,200000]` → 50000 | 13 | 49 MB | 0 | 0 | PASS |
| Q8_0 dst `[256,70000]` | 274 | 18 MB | 1.2e-2 | 4.7e-8 | PASS (tol 2e-2) |
| pad+cpy `[64,100000]` 4 chunks | | 24 MB | 0 | 0 | PASS |
| `[4096,65535]`, `[4096,65536]`, `[64,4194304]`, `[16,16777216]`, `[64,4620541]`, f16, I64 (≥ 1.5 GB each) | | | | | SKIP (budget) |

So the two shapes that need the 2D dispatch and fit a browser tab (65536 and 65538 workgroups)
pass bit-exact in Chrome, on the same shader source the native run uses. The Q8_0 case is the one
non-exact row: 5 of 17.9M dequantized values differ by one quantization step (Chrome's Tint and
the CPU quantizer round ties differently; native Dawn was exact) -- a rounding, not a dispatch,
difference. The f16-dst case (1024 MB src + 512 MB dst) was tried at `--budget-mb 1600` and
failed with `std::bad_alloc` while staging its 1 GiB host vector in the wasm heap (the entry
catches and prints C++ exceptions); that is the harness's heap, not the backend, so the ≥ 1.5 GB
cases are native-only. Note for `?args=`: the entry splits on spaces, so `--only` patterns must
not contain one (`[64,100000]`, not `pad+cpy [64,100000]`).
The existing browser smoke (`web/smoke`, rebuilt from the same patched submodule tree) still
passes (`RESULT: PASS`, gelu/silu graphs rel ≤ 2.6e-4, 64 MiB round trip exact).

### 10.5 Remaining backend limits that would block the shape decoder

None of the ops the decoder uses is missing (§5 B8-B11 stand: GET_ROWS, MUL_MAT, NORM, SILU,
ADD, MUL, CONCAT, REPEAT, PAD, CPY, SET_ROWS are all dtype-supported and now all
2D-dispatched). What remains is size, on two adapter classes:

- **This adapter (4 GiB binding)**: nothing blocks. The largest decoder tensors at res 1024
  (`docs/PIXAL3D_WEBGPU_MEMORY.md` §8: 1.57 GB `[512,768000]`, 1.19 GB `[64,4649838]` PAD seeds)
  bind and compute; a 3 GiB seed buffer and a 22.4M-row SET_ROWS are verified. The only hard edge
  is `ggml_nbytes(node) > maxStorageBufferBindingSize` → `supports_op = false` (4 GiB − 4 B), first
  reached by a `[64, M]` f32 buffer at M = 16,777,216 voxels, 3.6× the densest object seen. The
  C2S stage-3 graph asks gallocr for 5.98 GB (two `GPUBuffer` chunks); whether Chrome grants that
  much device memory to one tab is unmeasured (the Shape-1024 sampling job holds the GPU).
- **Spec-floor adapter (128 MiB binding, WG_SIZE 256)**: every `[C, N]` stage tensor from stage 2
  on, the `[Cout, M+1]` seed buffers and the `[M, 27]` neighbor tables exceed the binding limit --
  the §5 B9 buffer split plus a `kBlockChunkBytes` retune to ≤ 100 MiB are needed before any
  decoder graph runs; and the DiT RoPE SET_ROWS hits the (now handled) 2D path at L > 21845
  tokens, i.e. the fix in §10.1 is required there for Shape-1024-sized token counts.
- **Harness**: `--budget-mb` skips oversized cases; the browser build cannot host the ≥ 1.5 GB
  cases (wasm heap), so those stay native-only.

## 11. Validated on WebGPU (2026-09-06, `feat/webgpu-shape-decode`, sparse shape decoder)

Same meaning of "validated" as §8/§9: executed on the ggml WebGPU backend inside the real
decoder graphs, on the **real** Shape-1024 SLAT fixture (`hr_sample`, `tools/ref_pixal3d_hr_sample.py`:
N = 17,489 voxels at res 64 → M ≈ 4.65M at res 1024) and on the res-512 one (`slat_sample`,
N = 4,377 → 1.19M), against the PyTorch reference decode of the same SLATs, natively (Dawn) and
in Chrome. Results: `docs/spec/31-webgpu-bringup.md` §11, memory: `docs/PIXAL3D_WEBGPU_MEMORY.md`
§9. Branch base: `feat/webgpu-sparse-backend-prep` (§10); patches 0003/0004, the ops regression
and the allocation trace are reused as-is.

### 11.1 The one backend gap: `CONT` of an I32 view (`patches/ggml-webgpu/0005`)

§10.5 said "none of the ops the decoder uses is missing" -- true for the op *types*, wrong for one
dtype. The first ConvNeXt graph (`shape_dec_convnext_stage0`) aborted on native Dawn with

```
ggml-webgpu-shader-lib.hpp:2923: Unsupported src type for cpy shader
```

on `CONT src0=i32[4377] view-cont → dst=i32[4377]`: `submconv_range` (`src/sparse.cpp`) takes tap
`t`'s neighbour indices for the output range `[r0, r0+nr)` of the tap-major `[27·N]` int32 table as
`ggml_cont(ggml_view_1d(nbr, nr, (t·N + r0)·4))` -- the `cont` is there because the Vulkan
`get_rows` kernel asserts a zero index offset. `cpy.wgsl` had `SRC_F32`/`SRC_F16` sources only
(`DST_I32` already existed for the F32→I32 conversion), and `supports_op` admitted `CPY`/`CONT`
only for F32/F16 sources or F32→I32; the graph runner did not consult `supports_op` (it does now,
§11.3), so natively the shader lib aborted and in a browser the node would have been the §9
silent skip.

Fix (`0005-webgpu-cpy-i32-src.patch`, 3 hunks): `cpy.wgsl` gains `#elif defined(SRC_I32)
#define SRC_TYPE i32`; `get_cpy_pipeline` maps `GGML_TYPE_I32` sources to `SRC_I32`
(variant `cpy_i32_i32`); `supports_op` accepts I32 source with I32 destination. The shader body is
untouched (`dst[j] = DST_TYPE(src[i])` is the identity for i32→i32). Every layout the decoder
produces is covered in `trellis-webgpu-ops` (WebGPU vs ggml-cpu, native Dawn, Apple M1 Pro):

| case | view byte offset | out | max\|d\| | result |
|---|---|---|---|---|
| `cont(view_1d)` i32 `[4377]` at `+17508 B` (tap 1 of the res-512 stage-0 table; offset ≢ 0 mod 256, the storage-binding alignment) | 17,508 | 17 KB | 0 | PASS |
| `cont(view_1d)` i32 `[768000]` at tap 13 + 390,936 rows of a `[27·1158936]` table (Shape-1024 stage-3 ConvNeXt chunk) | 62.1 MB | 2.9 MB | 0 | PASS |
| `cont(view_1d)` i32 `[4649837]` at tap 26 of a `[27·4649837]` table (stage-3 conv2, M rows) | 483.6 MB | 17.7 MB | 0 | PASS |

(`read_f32` in the test converts I32 outputs exactly -- indices are < 2^24.)

### 11.2 Ops executed on WebGPU by the decoder (Shape-1024 fixture, all graphs)

| graph (tag, `TRELLIS_DUMP_OPS`) | ops on WebGPU | notes |
|---|---|---|
| `shape_dec_from_latent_N17489` | MUL_MAT f16×f32 `[32,1024]ᵀ[32,N]`, ADD | |
| `shape_dec_convnext_stage{0..3}` (4/16/8/4 blocks, 812/3248/1624/1628 nodes) | per block and chunk: SCALE + CONCAT (sentinel zero row `[Ci, N+1]`), 27 × { CONT i32 idx (**0005**), GET_ROWS `[Ci, nr]`, CONT f16 tap weight `[Ci,Co]`, MUL_MAT, ADD_inplace }, ADD bias, NORM, MUL, ADD (affine), MUL_MAT `[C,4C]`, ADD_inplace bias, SILU_inplace, MUL_MAT `[4C,C]`, ADD bias, CONT (chunk view of the block input), ADD residual, CONCAT (chunk outputs) | stage 3 has 2 chunks per block (768,000 + 390,925 rows); every tensor ≤ 1.57 GB |
| `shape_dec_c2s_stage{0..3}_subdiv` | MUL_MAT `[Cin,8]`, ADD | host reads the `[8,N]` logits: octant mask → new coords + gather index (CPU/WASM topology) |
| `shape_dec_c2s_stage{0..3}_conv` (402/402/602/1392 nodes) | NORM, MUL, ADD, SILU (norm1), SCALE+CONCAT (pad), conv1 as above (27 taps, chunked), GET_ROWS `[Cout, m1−m0]` (channel→spatial gather), PAD (`[Cout, M+1]` seed) + CPY into row-offset views (later chunks), NORM, SILU_inplace (norm2), conv2 as above at M rows (chunked ≤ 1M), GET_ROWS `[K, M]` + REPEAT `[R,K,nr]` (skip), ADD, PAD + CPY (output buffer) | stage 3: M = 4,649,809, the `[64, M+1]` seed/NORM/SILU buffers are 1.19 GB each; `[M,27]` i32 neighbour table 502 MB as an input |
| `shape_dec_output_layer_N{1000000,649809}` | NORM (no affine), MUL_MAT `[64,7]`, ADD | 1M-row chunks |

No op type was added; besides 0005 the decoder needed nothing from the backend. `SET_ROWS` is not
used by the decoder (the C2S single-buffer pattern is `PAD` + `CPY`, §10.3); patch 0004 stays for the
DiT RoPE scatter.

### 11.3 Sparse-runtime changes reusable by the texture decoder

All in the shared runtime (`src/sparse.cpp`, `src/shape_decoder.cpp`, `src/trellis_model.cpp`),
none in texture-specific code; `tex_decode` goes through the same `decode_unet` / `sparse_c2s`
and inherits them:

- **Supported-op guard**: `run1` and `GraphRun::run` call `check_graph_supported` after building
  each graph (the flow/cond graphs already did). On WebGPU an unsupported node now throws
  `graph '<tag>' has ops the ggml WebGPU backend would silently skip` before any compute, instead
  of the browser producing wrong features silently -- this is what would have caught 11.1 in Chrome.
- **Tried and not kept -- in-place elementwise ops** (27-tap accumulation as `ggml_add_inplace`,
  MLP bias/SiLU and C2S norm2-SiLU in place): exact (identical mesh: V = 4,649,809, worst mean
  6.3448e-7) and it cuts the graph-allocated tensor count (stage-3 C2S 981 → 791, live estimate
  6.97 → 5.79 GB) but gallocr's actual buffers do not move at all (7.19 / 3.28 / 3.71 / 2.10 GB for
  the four largest graphs, before and after; process footprint 10.4 GiB either way) -- gallocr was
  already reusing those tensors' slots. Reverted to keep the validated graph identical to CUDA's;
  the number that does move is the chunk budget (`docs/PIXAL3D_WEBGPU_MEMORY.md` §9.3).
- **`tensor_to_f32` reads F32 outputs straight into the result vector, in 256 MiB slices** (it
  used to stage every readback through a second byte buffer of the same size, and the WebGPU
  backend maps a staging buffer of the whole request: +2 × 1.19 GB transient at the stage-3 C2S
  output, which trapped the first Chrome run -- memory doc §9.3). Isolated regression:
  `trellis-webgpu-ops`' `read_f32` now reads outputs the same way, so its ≥ 1 GB cases
  (`cont(permute) 1 GB`, `pad+cpy [64,4620541]` ×2, 5 slices each) exercise the sliced readback:
  bit-exact, native Dawn.
- **Per-stage debug dump** `TRELLIS_DBG_STAGE_DUMP=<dir>` (`stage_dump`, `shape_decoder.cpp`;
  `npy::save_i32` added to `include/npy.h`) and `tools/compare_sparse_stages.py` +
  `tools/ref_pixal3d_shape_dec_stages.py` -- the stage-by-stage parity method of spec 31 §11.3,
  applicable to `tex_dec` as-is (`--kind tex_dec` once the tex decoder dumps).
- `npy::save`/`save_i32` throw on a short write (a disk-full dump used to leave a 4 KB file
  silently).

Remaining for the texture decoder (not touched here): the tex decoder's own `guide_subs`-driven
C2S runs the same graphs at the same M with `Cout` channels 6 at the output, so nothing new is
expected from the backend; the open items are the ones in memory doc §9.4 (spec-floor adapters,
`kBlockChunkBytes` on a 128 MiB binding) and running `trellis-test-pixal3d-tex-decode` on WebGPU.
