# Pixal3D Validation

## Principle

Use PyTorch as the golden implementation and identify the **first divergent tensor**. Final mesh-only comparison is insufficient.

## Three-way validation

On an RTX 4090 development machine:

```text
                 PyTorch/CUDA
                     |
               golden fixtures
                     |
          +----------+----------+
          |                     |
 pixal3d.cpp CUDA      pixal3d.cpp WebGPU
          |                     |
          +------ compare -------+
```

Run CUDA and WebGPU sequentially when VRAM contention would distort results.

## Required metrics

For floating tensors:

- max absolute error
- mean absolute error
- relative error
- cosine similarity

For sparse topology:

- coordinate-set equality
- order-sensitive equality when required
- neighbor-map equality
- active voxel count

For geometry:

- vertex count
- face count
- bounds
- nearest-surface error / Chamfer where appropriate
- normal agreement
- attribute comparison

## Fixture hierarchy

```text
tests/fixtures/
  projection/
  project_attention/
  rope3d/
  flow_block/
  ss_decoder/
  sparse_neighbor_map/
  sparse_conv3d/
  c2s/
  s2c/
  shape_decoder/
  texture_decoder/
  e2e/
```

Each fixture should record inputs, expected outputs, shape, dtype, source commit, model/stage, and tolerance.

## Determinism

Do not depend on two RNG implementations matching bit-for-bit. Generate deterministic noise once, serialize it, and feed identical tensors to all implementations.

Sampling parity is checked per step, not only at the end: the SLAT sampling tests replay any single
Euler step from a serialized step-k latent through the production sampler (`SamplerParams::step_begin/
step_end`) and score it against the reference's step-(k+1) latent -- `trellis-test-pixal3d-slat-sample
--start-step k [--num-steps n] [--input-latent] [--expected-latent] [--repeat R]`, the same entry point
in the browser via `pixal3d_shape_step_run` / `web/shape1024/run_playwright.js --step k`. `--repeat R`
re-runs the identical step from the same immutable input and reports unique output hashes, the first
differing index and the max/mean run-to-run difference: the primary probe for backend nondeterminism
(`docs/spec/31-webgpu-bringup.md` §11.5). The per-step reference latents are the existing
`{f32,bf16,cuda,cuda_nofa}_*_x_step<k>.npy` dumps; they are not regenerated for this and stay external.

## Suggested CLI

```text
pixal3d-cli validate projection --fixture tests/fixtures/projection/basic
pixal3d-cli validate flow-block --fixture ...
pixal3d-cli validate e2e --reference pytorch_dump/
```

Headless browser tests should invoke the real browser WebGPU implementation, not a Node CPU fallback.
