# Pixal3D desktop VRAM budget (CUDA, multiview 1024)

Measured on the RTX 4090 (24 GB) with `trellis-cli` (4-view cyclops, q8_0 MV set, `--res 1024`,
seed 2), sampling `nvidia-smi memory.used` every ~200 ms next to the CLI's own debug toggles
(`TRELLIS_DBG_ALLOC_TRACE=1 TRELLIS_DBG_BUDGET=1 TRELLIS_DBG_COND=1 TRELLIS_DBG_MEM=1`). All
figures are process peaks above the idle baseline (CUDA context included), per stage window.

## Baseline (main @ ec464fe, 2026-09-19)

| stage | peak | what it is |
|---|---:|---|
| HR-1024 shape flow | +3.35 GB | `[budget]` weights 1408 + activations 1825 + cond 271 MB |
| shape decode | **+7.36 GB** | `c2s_stage3_conv` gallocr 5.98 GB (ten `[512,768000]` f32 = 1.5 GB chunk tensors) + shape_dec 0.84 GB |
| texture cond (DINO + NAF@1024 x4) | **+5.66 GB** | one spike per view: `naf_attn_cuda` `cudaMalloc`s the whole NAF map (out 4 GiB + q 1 GiB) |
| texture flow | +3.35 GB | `[budget]` 1408 + 1827 + 271 MB |
| texture (PBR) decode | **+7.36 GB** | same C2S stage-3 graph as the shape decoder + tex_dec 0.84 GB |

The peak is the sparse decoder's stage-3 C2S graph, identical for shape and texture, so
`--no-texture` does not lower the requirement.

## Changes (branch `feat/desktop-vram-8gb`)

1. **Decoder chunk budget scales with the device** (`src/sparse.cpp::chunk_budget_bytes`).
   `kBlockChunkBytes` (1500 MiB, tuned on 24 GB) becomes `min(1500 MiB, max(256 MiB, total/16))`
   on GPU backends other than WebGPU: 24 GB → 1500 (unchanged), 16 GB → 1024, 12 GB → 768,
   8 GB → 512. `TRELLIS_BLOCK_CHUNK_MB` / `TRELLIS_C2S_CHUNK_MB` still override per site, and
   `TRELLIS_DEVICE_BUDGET_MB` (the DiT gate's simulated-device override) applies here too, so a
   24 GB card can rehearse the 8 GB chunking. The browser path keeps the constant (ggml-webgpu
   reports `maxBufferSize`, not device memory; `docs/PIXAL3D_WEBGPU_MEMORY.md` gates it).
   C2S norm2/silu on the `[Cout, M+1]` buffer run in place (−0.14 GB gallocr at M = 4.7M).
2. **1024-res conditioning uses the device-resident, sparse-coords path**
   (`pixal3d_cond_slat_gpu(..., &coords)`, the browser path) for both the HR shape stage
   (T=512, `naf_block_chunk = 1024` = the split-graph auto value) and the texture stage (T=1024,
   `cond_slat_gpu_chunked`). No NAF map is materialized on either side; `proj` comes back
   already gathered at the active tokens.
3. **NAF encoder graph uses direct conv on CUDA** (`naf_ggml_opts_for`): the sem_encoder's k=3
   im2col at S=1024 made the encoder graph 3332 MB; direct is 1541 MB. `TRELLIS_DBG_NAF_DIRECT=0/1`
   forces either graph on any backend for A/B.

Parity (`trellis-test-pixal3d-cond-tex --host`, 4090, cyclops, stride-4 coords):

| config | vs | global | proj max abs / L2rel | time (device path) |
|---|---|---|---|---|
| T=1024 chunked, im2col | CUDA host path (`naf_attn_cuda`) | exact | 7.6e-6 / 7.3e-8 | 6.7 s (host 68.3 s) |
| T=1024 chunked, direct conv | CPU reference (`TRELLIS_NAF_CPU=1`, 1 view) | exact | 1.1e-4 / 1.0e-6 | 4.3 s (host 377 s) |
| T=512 chunked (`--chunk 1024`) | CUDA host path | exact | 4.0e-3 / 8.6e-5 (same as the single-graph device path) | 11.1 s (single-graph 16.9 s) |

## After (same input, `TRELLIS_DEVICE_BUDGET_MB=8192`, 2026-09-20)

Per-stage peaks from the clean run `wsl_final3_8g_s2` (started only after 60 s of 0 % GPU
utilisation, idle baseline 1651 MiB, no other compute process; the box is shared and
contaminated runs were discarded). Stage timings are from the WSL build reading the GGUFs over
9p and are only comparable with each other.

| stage | main | after | note |
|---|---:|---:|---|
| HR-1024 cond (DINO + NAF@512 x4) | +4.78 GB | **+3.76 GB** | split graph: per-view graph 3789 → 2065 MB |
| HR-1024 shape flow | +3.35 GB | +3.5 GB | unchanged (`[budget]` 3540 MB) |
| shape decode | +7.36 GB | **+6.23 GB** | C2S stage-3 gallocr 6.02 → 4.76 GB at 512 MiB chunks |
| texture cond (DINO + NAF@1024 x4) | +5.66 GB | **+4.42 GB** | weights 0.31 + persistent 2.2 + encoder graph 1.54 GB |
| texture flow | +3.35 GB | +3.76 GB | unchanged (`[budget]` 3543 MB) |
| texture (PBR) decode | +7.36 GB | **+6.07 GB** | as shape decode |
| whole run | +7.36 GB | **+6.23 GB** | |
| texture stage wall (WSL) | 116 s | 55 s | cond 68 → 6 s; decode 17 → 20 s at 512 MiB chunks |

The remaining decoder floor at M = 4.7M voxels is the M-scaled set of the stage-3 C2S graph:
inputs (feats 574 MB, neighbour tables 487 + 121 MB), the padded `[128, N+1]` conv1 input
(574 MB), `hraw/hn [64, M+1]` (1154 MB) and the conv1 working set (2 × chunk + gather). Going
below ~5 GB for this object needs the M-split (B9) work tracked for the browser, not a knob.
An actual 8 GB card has not been tried: WSL CUDA oversubscribes into system RAM, so a
VRAM-pinning co-tenant on the 4090 does not prove a fit -- only the peak trace does.

Output drift vs main on this input: HR SLAT mean −0.1065 → −0.1066 (std 5.2386 → 5.2372),
decoded voxels 4,722,792 → 4,725,304 (+0.05 %), final GLB 970,092 → 964,238 faces. The
Windows-native release build of the same main differs from the WSL build by 2 % voxels on the
same seed, so this is within build-to-build variation.

## Round 2 (#38: B9 split, cond buffer scoping, conv1 estimate, one cond path), 2026-09-20

Branch `feat/vram-b9-c2s-split` (stacked on PR #34). Same input, same harness, clean windows
(`wsl_b9all_8g_s2` / `wsl_b9all_24g_s2`, baseline 2432 MiB, no other compute process).

| change | what | effect (4090) |
|---|---|---|
| #39 `sparse_c2s` graph 2 → 2a (conv1 → persistent `hn`) + 2b (conv2 + skip + head) | only `hn [Cout, M+1]` is bridged between graphs, in its own backend buffer; the conv1 input, its pad and both neighbour tables no longer co-reside with it; the skip input is gathered on the host | stage-3 C2S gallocr 4.71 → 2a 2.46 + 2b 1.53 GB (+ `hn` 1.15 GB) at 512 MiB chunks; outputs byte-identical (shape verts/faces/coords, tex attrs/coords) |
| #43 conv1 chunk estimate `Cout*8*4` → `2*Cout*8*4 + Cin*4` | matches the live set the trace shows (accumulator + tap output + gather) | 2a 2.46 → 1.92 GB at 512 MiB; 3.32 GB at the 1500 MiB default (4 chunks instead of 2); byte-identical at both budgets |
| #40 cond `pooled` / `q_bm` (1 GiB each at T=1024) allocated per view around their producer/consumer | not resident during the encoder graph (G2) or the attention graphs (G4) | cond-tex stats peak 5838 → 3023 MB (T=1024), 3035 → 2779 (T=512); unchanged numerics (im2col A/B: 7.3e-8 vs the host path) |
| #43 LR-512 cond → `pixal3d_cond_slat_gpu` split graph (`naf_block_chunk = 256`) | one cond path in the CLI | per-view peak 1199 MB (single graph 3210, host path ~2.1 GB), 2.5 s vs 9.4 s |

| stage | main | PR #34 (8 GB sim) | round 2, 8 GB sim | round 2, 24 GB default |
|---|---:|---:|---:|---:|
| LR-512 cond + flow | +2.14 GB | +2.07 GB | +1.80 GB | +1.80 GB |
| HR-1024 cond | +4.78 GB | +3.76 GB | +3.76 GB | +3.76 GB |
| HR-1024 shape flow | +3.35 GB | +3.5 GB | +3.5 GB | +3.5 GB |
| shape decode | +7.36 GB | +6.23 GB | **+4.73 GB** | +6.03 GB |
| texture cond | +5.66 GB | +4.42 GB | **+3.42 GB** | +3.42 GB |
| texture flow | +3.35 GB | +3.76 GB | +3.77 GB | +3.77 GB |
| texture decode | +7.36 GB | +6.07 GB | **+4.36 GB** | +5.97 GB |
| whole run | +7.36 GB | +6.23 GB | **+4.73 GB** | +6.03 GB |
| shape decode / texture stage wall (WSL) | 24.2 s / 116 s | 27.2 s / 55 s | 27.2 s / 56.5 s | 24.0 s / 53.2 s |

The 24 GB default keeps its speed (shape decode 24.0 s vs main 24.2 s) and drops 1.3 GB from
the B9 split alone. Under the 8 GB simulation the whole run stays under 5 GB; the decoder's
remaining max is the stage-3 ConvNeXt graph (gallocr 2.80 GB, of which ~0.57 GB is the per-block
`ggml_concat` output chain -- the next lever if more headroom is needed) and the C2S 2a graph
(1.95 GB + the 1.15 GB `hn`).

Output drift from switching the LR-512 cond to the device path (direct conv; the CUDA host
path pins im2col, L2rel 8e-5 between them): LR SLAT mean −0.2133 → −0.2146, HR tokens
17,582 → 17,585, decoded voxels 4,725,304 → 4,706,509 (−0.4 %), final GLB 964k → 972k faces --
the same class of drift as the HR cond switch in PR #34, all parity tests PASS.

Still not done (#41, #42): no real 8 GB card, no Metal/Vulkan run (the three files
syntax-compile with the Metal flags), `hn` is still one 1.15 GB buffer.
