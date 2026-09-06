# Pixal3D WebGPU operator gap — measured inventory (2026-09-06)

This document replaces `PIXAL3D_OP_SUPPORT_MATRIX.md`'s guesswork with a **measured** operator
inventory of one full `--views` multiview run on `feat/webgpu-bringup`, traced on the CUDA box
(`ssh win`, RTX 4090). Sections 1–3 (and 4) are facts from the trace; the **WebGPU support
classification column is intentionally left `TBD`** for every op except the handful that are
unambiguously CPU-only geometry — a follow-up pass cross-references these ops against the actual
WebGPU backend coverage in the vendored `thirdparty/ggml` commit.

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
Classification is `TBD` throughout — see the note at the top of this document.

### DINOv3 @ 512 (`src/dinov3.cpp:57` `dinov3_encode`, tag `dinov3_S512`, 8 builds = 4 views × {SS cond, shape-512 cond})

1410 nodes/graph. Ntok=1029 (5 prefix + 32×32 patches).

| op (+sub-op) | file:line | count | largest shape (in→out) | dtype | layout | bytes | notes | class |
|---|---|---|---|---|---|---|---|---|
| IM2COL | dinov3.cpp:79 (`ggml_conv_2d` patch embed) | 1 | in f32[512,512,3,1] → f16[768,32,32,1] | f16/f32 | cont | 3.1 MB | patch_embed k=16 s=16 p=0 | TBD |
| MUL_MAT | dinov3.cpp:79,17,46,95 (patch embed GEMM, QKV/out proj, MLP, attn score+ctx) | 145 | [64,1029,16,1]×2 → [1029,1029,16,1] | f32 | cont | 67.8 MB | QKᵀ, plain (non-FA) attention | TBD |
| SOFT_MAX | dinov3.cpp:46 `attn` | 24 | [1029,1029,16,1] | f32 | cont | 67.8 MB | `scale=0.125,max_bias=0`; full O(L²) score matrix materialized (no FlashAttention here) | TBD |
| NORM | dinov3.cpp:22 `ln` | 49 | [1024,1029,1,1] | f32 | cont | 4.2 MB | `eps=1e-5`; final LN is non-affine | TBD |
| UNARY:GELU_ERF | dinov3.cpp:96 (MLP) | 24 | [4096,1029,1,1] | f32 | cont | 16.9 MB | exact-erf GELU (not tanh-approx) | TBD |
| CONCAT | dinov3.cpp:86 (cls‖reg‖patches), attn qkv-split path | 50 | [1024,5,1,1]+[1024,1024,1,1] → [1024,1029,1,1] | f32 | cont | 4.2 MB | prefix-token concat | TBD |
| RESHAPE/VIEW/PERMUTE/CONT | dinov3.cpp:46 `attn` (head split, RoPE layout) | 126/168/98/266 | up to [3072,1029,1,1] | f32 | mixed (many `view-noncont`→`CONT`) | up to 12.6 MB | host-precomputed cos/sin RoPE tables applied via elementwise MUL/SUB/ADD (see dit RoPE note below — same pattern) | TBD |
| ADD / MUL / SCALE / CPY | dinov3.cpp:16,22,46 | 217/192/48/2 | up to [4096,1029,1,1] | f32/f16 | cont | up to 16.9 MB | bias-add, LN affine, attn scale (0.125), cls/reg f16→f32 cast | TBD |

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
| FLASH_ATTN_EXT | dit.cpp:141 `sdpa` | 60 (30 self + 30 cross) | q f32[128,4096,12,1], k/v bf16[128,4096,12,1], mask f16[4096,4096,1,1] → f32[128,12,4096,1] | mixed f32/bf16/f16 | cont | 33.5 MB (mask) | `scale=0.0883883 (1/√128), max_bias=0, logit_softcap=0, prec=10 (F32 accum)`; self-attn Lk=Lq=4096, cross-attn Lk=Lc=5(global)+... | TBD |
| REPEAT | dit.cpp:83 `build_pad_mask` | 2 | [4096,1,1,1] → [4096,4096,1,1] f16 | f16 | cont | 33.5 MB | **materializes the whole FA pad mask by broadcast-repeat** — built once per DiT graph and shared by all 30 blocks, but still a full [Lk_pad,Lq_pad] tensor; candidate for a WebGPU-side broadcast instead of materialization | TBD |
| MUL_MAT | dit.cpp:24 `lin` (qkv/out/mlp/adaLN linears) | 245 | [1536,8192,1,1]×[1536,4096,1,1] → [8192,4096,1,1] | f16×f32→f32 | cont | 134.2 MB | MLP up-projection (1536→8192, widen×~5.3) is the largest GEMM | TBD |
| UNARY:GELU | dit.cpp:274 (MLP) | 30 | [8192,4096,1,1] | f32 | cont | 134.2 MB | tanh-approx GELU (`ggml_gelu`, distinct from DINOv3's `gelu_erf`) | TBD |
| RMS_NORM | dit.cpp:39 `rms_gamma` (q/k norm) | 120 (4/block × 30) | [128,12,4096,1] | f32 | view-cont | 25.2 MB | `eps=1e-12`; MultiHeadRMSNorm ×γ | TBD |
| ARANGE / SET_ROWS / SUB / UNARY:STEP / CPY | dit.cpp:49–75 (`apply_rope`, `build_pad_mask`) | 122/120/60/2/362 | up to [1,128,12,4096] | f32/i32/f16 | cont/view-cont | up to 25.2 MB | **3D interleaved-pair RoPE is hand-built from primitives — `ggml_rope`/`GGML_OP_ROPE` is never called anywhere in this pipeline (0 occurrences in the whole run's dump)**; K/V padding to the FA `KQ_STRIDE=256` tile boundary via `PAD` | TBD |
| PAD | dit.cpp:129 `prep_kv` | 60 | [128,5,12,1]→[128,256,12,1] | f32 | cont | 1.6 MB | zero-pads K/V key dim to 256-multiple for FA tiling | TBD |
| NORM | dit.cpp:30 `layernorm` | 91 | [1536,4096,1,1] | f32 | cont | 25.2 MB | `eps=1e-6`; affine on norm2 only, non-affine elsewhere (adaLN modulation) | TBD |

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
| seg1 (res16, C:8→1024) | ss_decoder.cpp:27 `conv3d`, :48 `resblock` | IM2COL_3D×10, MUL_MAT×10, NORM/MUL/UNARY:SILU×8, ADD×22 | IM2COL_3D dst f16[13824,16,16,16] | 113.2 MB | `s=1,p=1,d=1` (submanifold-style dense 3×3×3), 2×`ResBlock3d(512)`+2×`Conv3d` | TBD |
| seg2 (res32, C:128→256) | same, ch=128 | IM2COL_3D×5, MUL_MAT×5, UNARY:SILU×4 | IM2COL_3D dst f16[3456,32,32,32] | 226.5 MB | `pixel_shuffle_3d(scale=2)` (CPU, ss_decoder.cpp:80) between segments | TBD |
| seg3 (res64, C:32→1) | same, ch=32 | IM2COL_3D×5, MUL_MAT×5 | IM2COL_3D dst f16[864,64,64,64] | **453.0 MB** | final `chln`+SiLU+`Conv3d(32→1)` → occupancy logits `[64,64,64,1]` | TBD |

`ss_coords` (ss_decoder.cpp:128, downsample res64→res32 occupancy→active-voxel-coord list) is CPU
host code (no ggml graph) — 4394 active voxels @res32 in this run.

### NAF encoder (ggml, `src/naf_gpu.cpp:115` `naf_upsample_gpu`, tags `naf_encoder_S{512,1024}_T{512,1024}`) + NAF attention (custom CUDA)

Two-branch `Conv2d`+`EncBlock` image encoder (`encoder` k=1, `sem_encoder` k=3 reflect-padded),
concat, optional `AvgPool2d` down to target T. 4 views × 3 stage-configs = 12 builds
(shape-512:S512→T512, shape-1024:S1024→T512, tex-1024:S1024→T1024).

| op (+sub-op) | file:line | count | largest shape (S=1024 case) | bytes | notes | class |
|---|---|---|---|---|---|---|
| PAD_REFLECT_1D | naf_gpu.cpp:40 `reflect_pad2d` | 10 | [1024,1026,128,1]→[1026,1026,128,1] | 538.97 MB | applied twice (W then H) per k=3 conv, pad=1 | TBD |
| IM2COL | naf_gpu.cpp:54 `conv_bias` (`ggml_conv_2d`) | 10 | in f32[1026,1026,128,1] → f16[1152,1024,1024,1] | **2.416 GB — the single largest tensor in the whole run** | k=3, 128→128, s=1,p=0 (padding pre-applied via reflect) | TBD |
| MUL_MAT | naf_gpu.cpp:54 | 10 | same im2col output ×[1152,128,1,1] | 2.416 GB (src) | conv GEMM | TBD |
| GROUP_NORM | naf_gpu.cpp:61 `group_norm_affine` | 8 | [1024,1024,128,1] | 536.9 MB | `n_groups=8, eps=1e-5`; affine applied as separate MUL+ADD (ggml's group_norm has no built-in affine) | TBD |
| UNARY:SILU | naf_gpu.cpp:72 `enc_block` | 8 | [1024,1024,128,1] | 536.9 MB | | TBD |
| CONCAT | naf_gpu.cpp:136 (`e1‖e2`, channel dim) | 1 | 2×[1024,1024,128,1]→[1024,1024,256,1] | 1.074 GB | | TBD |
| POOL_2D | naf_gpu.cpp:138 (`pool_k=Sp/T` only when Sp≠T) | 0 or 1 | [1024,1024,256,1]→[512,512,256,1] | 1.074 GB | `pool_op=AVG(1), k=s=2, p=0`; absent when Sp==T (shape-512, tex-1024) | TBD |

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
| GET_ROWS | sparse.cpp:74 `submconv_range` (27-tap gather), :332 `xs` skip-gather | 108–432/stage (scales with chunk count × 27 taps) | src f32[64,6144000,1,1] (post-subdiv conv2, stage3) → f32[64,3080029,1,1] | 1.573 GB | submanifold-conv neighbor gather; **this is the op WebGPU support hinges on for sparse conv** — indices come from `build_neighbor_table` (CPU, sparse.cpp:23, `unordered_map`-based, tap-major `[27,N]` int32) | TBD |
| MUL_MAT | sparse.cpp:76 `mul_mat_rows`, :127/129 (ConvNeXt MLP) | 81–232/stage | [128,512,1,1]×[128,768000,1,1]→[512,768000,1,1] | **1.573 GB** | ConvNeXt MLP widen (C→4C); row-chunked (`kBlockChunkBytes`≈1.5 GB budget) to bound peak memory — this chunking is host orchestration around otherwise-plain `MUL_MAT`s | TBD |
| UNARY:SILU | sparse.cpp:128 (ConvNeXt MLP) | 2–16/stage | [512,768000,1,1] | 1.573 GB | | TBD |
| NORM | sparse.cpp:125 (ConvNeXt rowLN), sparse_c2s norm1/norm2 | 2–16/stage | [64,4620541,1,1] (post-subdiv, +1 sentinel row) | 1.183 GB | `eps=1e-6` throughout sparse ops | TBD |
| ADD | sparse.cpp:79 (submconv bias+accumulate), :348 (skip-add) | 56–496/stage | [512,768000,1,1] | 1.573 GB | 27-tap accumulation is a chain of `ADD`s, not a fused reduce | TBD |
| PAD | sparse.cpp:314,352 (`hraw`/`out` zero-seed buffer) | 2/graph | [64,3080029,1,1]→[64,4620541,1,1] | 1.183 GB | seeds a big output buffer once, then `CPY`s each chunk in (see below) — avoids an O(chunks) concat-chain that would peak at ~2× the result | TBD |
| CPY | sparse.cpp:315,353 (chunk write-into-view) | varies (rooted via `roots`) | up to [64,1540511,1,1] | 394 MB | writes not reachable from the graph's single output root, so explicitly rooted via `ggml_build_forward_expand` on each `CPY` | TBD |
| REPEAT | sparse.cpp:347 (skip `repeat_interleave(R)`) | 1–5/graph | [1,16,1000000,1]→[4,16,1000000,1] | 256 MB | S2C skip-connection channel interleave (R=Cout/K) | TBD |
| CONCAT | sparse.cpp:85 (`submconv_pad` sentinel row), :242 | 1/graph | [128,295876,1,1]+[128,1,1,1]→[128,295877,1,1] | 590.2 MB | appends the zero sentinel row absent-neighbor indices point at | TBD |

`output_layer` (final LN + `Linear(64,out_ch)`) is chunked at `kLinearRowChunk=1,000,000` rows —
5 graph builds per decode call (4×N=1,000,000 + 1×N=620,540), each 3 nodes (NORM/MUL_MAT/ADD),
largest `[64,1000000,1,1]` = 256 MB.

**Host-side, between every stage** (not ggml): `build_neighbor_table` (CPU hash map, rebuilt at
every N), and `sparse_c2s`'s octant-mask→new-coords→gather-index construction (`sparse.cpp:225–273`,
CPU) — sizes at shape_dec stage3: N=1,152,830→M=4,620,540 coords, `nnbr` = 27×4,620,540 int32 ≈
499 MB host RAM. Classification: **CPU (host/WASM), by design.**

### Geometry / GLB (CPU) — see §4.

## 3. Custom / non-ggml operations

| kernel | file | what it computes | inputs/outputs (largest, this run) | CPU fallback? | GPU→CPU readback in this flow? | class |
|---|---|---|---|---|---|---|
| `naf_attn_kernel`/`naf_attn_cuda` | `src/naf_attn.cu:34,90` | Cross-scale 9×9 neighborhood attention (NATTEN-style), one thread per output pixel, 4 heads × 64-dim QK, softmax over 81 logits, weighted sum of 81 low-res V vectors. Indexes the **un-upsampled** pooled k/v maps directly (equivalence argument in `naf_attn.h`) instead of materializing `k_up`/`v_up` (would be up to 4 GB f32 at T=1024). | q f32[256,1024,1024] (RoPE'd), k_pooled f32[256,64,64], v f32[1024,64,64] → out f32[1024,1024,1024] (4 GB f32 at T=1024, largest attention output in the whole pipeline) | Yes — `src/naf.cpp`'s `naf_na2d`+CPU reference path (bit-exact vs. fixture at T=128/512, ~52 s/view @T=512/32-core; **not exercised in this run** since the CUDA path was available) | Yes — `naf_upsample_gpu` reads back `enc_pooled`/`cat` (ggml→host) to run RoPE+adaptive-pool on CPU (reusing `naf.cpp`'s reference functions), then re-uploads q/k_pooled/v into the CUDA kernel (`naf_attn_cuda`'s own `cudaMemcpy`s) | non-ggml; WGSL port is the highest-risk item per the porting-plan docs (native ggml has no neighborhood-attention op) |
| `decimate_qem` (GPU) | `src/decimate_qem.cu` | CuMesh-style parallel QEM edge-collapse mesh simplification (Garland-Heckbert quadrics, atomicMin cost propagation, threshold-ladder driver). **Exercised in this run**: `decimate_qem_gpu(target=1000000): V 9,121,621→472,905, F 18,246,528→947,288`. | verts/faces host arrays in/out (mesh-sized: ~9.1M V / 18.2M F in) | Yes — `src/decimate_qem.cpp:203` `decimate_qem`, CPU port, same algorithm | Host↔device round-trip is the entire call (host mesh in, CUDA rounds, host mesh out) | non-ggml; CPU fallback exists and is the WASM path |
| `deform_conv2d` (GPU: `.cu`; CPU: `_cpu.cpp`) | `src/deform_conv.cu`, `src/deform_conv_cpu.cpp` | Modulated deformable conv2d (torchvision `deform_conv2d` v2), used only by BiRefNet's `ASPPDeformable`. | n/a this run | Yes (`deform_conv_cpu.cpp`, std::thread-parallel; Vulkan/HIP compute-shader variants also exist) | n/a | **off-path**: `--views` mode never calls BiRefNet (confirmed — `ops_trace.log` shows no BiRefNet stage, and `trellis_run_mv` in `trellis_cli.cpp` has no birefnet call). Listed for completeness only. |

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
code) — no TBD classification needed.

---

*Instrumentation: `include/graph_dump.h`, `src/graph_dump.cpp`. Trace log:
`/mnt/hdd1/pixal3d/out/ops_trace.log`. Raw per-node dump:
`/mnt/hdd1/pixal3d/out/ops_dump.txt` (remote, `ssh win`, not copied into the repo).*
