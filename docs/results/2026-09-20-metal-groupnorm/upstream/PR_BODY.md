# metal : size the GROUP_NORM threadgroup to the group (was 32 threads fixed)

`ggml_metal_op_group_norm` dispatches one threadgroup per group with `nth = 32` — the loop that scales
`nth` with the tensor size is commented out (`ggml-metal-ops.cpp`, "int nth = 32; // SIMD width"). Every
group is therefore reduced and normalized by a single SIMD group, three passes over the group, regardless
of its size. For the group sizes stable-diffusion-style encoders produce this is the slowest op in the graph:

| `GROUP_NORM` on `[1024,1024,128,1]` f32 (8 groups, 64 MB each), M4 Max | ms/op | max\|d\| vs float64 |
|---|---:|---:|
| `nth = 32` (current) | 453 | 2.2e-3 |
| this PR (`nth` from the group size, up to 1024) | **16.1** | **1.2e-5** |
| same tensor as `ggml_norm` over a `[n/8, 8]` reshape (reference for what the hardware can do; 8–16 ms across runs, 10.8 ms in the same run as the patched figure) | 8–16 | 9.5e-7 |

Changes:

1. `ggml_metal_op_group_norm`: `nth` doubles from 32 while `nth < gs/4` (`gs = ne00*ne01*ceil(ne02/ngrp)`, the
   elements one threadgroup reduces) up to the pipeline's max threads per threadgroup, capped at 1024 because
   the kernel's cross-SIMD-group buffer holds 32 partial sums. The commented-out loop was based on `ne00`
   (copied from `norm`, where a threadgroup handles one row); for `group_norm` the threadgroup handles a whole
   group, so the bound has to be the group size.
2. `kernel_group_norm_f32`: a `threadgroup_barrier` before the second zero-fill of `buf`, on the safe side. With
   `ntg > 32` every SIMD group reads `buf[tiisg]` after the mean reduction and SIMD group 0 then re-zeroes `buf`
   for the variance reduction; nothing orders the two, and `kernel_norm_fuse_impl` has a barrier at the same
   point. I have not reproduced a race (the `ntg > 32` path was unreachable at `ntg = 32`; `test-backend-ops`
   was run with both hunks applied, not with the barrier hunk isolated), so treat it as a hygiene fix rather
   than a demonstrated bug.

The `max|d|` column is against a float64 two-pass reference: the 32-thread version's error is a systematic
overestimate of `1/std` (each thread accumulates 524k values in f32); the wider reduction shortens every
partial sum. Validated with `test-backend-ops -o GROUP_NORM` (2/2 OK) and `-o GROUP_NORM_MUL_ADD` (2/2 OK) on Metal (M4 Max, macOS 26.5).

Measured on pixal3d.cpp (TRELLIS.2 / Pixal3D image→3D, NAF image encoder: 8 GroupNorm(8) on
`[1024,1024,128]`), where this op was 58 % of the encoder graph (3.66 s of 6.37 s per view): with the patch the
graph runs in 2.78 s and `GROUP_NORM` is 0.13 s (16 ms/op), within 3 % of expressing the op as `ggml_norm` over a
`[n/8, 8]` reshape (2.71 s).
