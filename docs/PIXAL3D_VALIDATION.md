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

## Suggested CLI

```text
pixal3d-cli validate projection --fixture tests/fixtures/projection/basic
pixal3d-cli validate flow-block --fixture ...
pixal3d-cli validate e2e --reference pytorch_dump/
```

Headless browser tests should invoke the real browser WebGPU implementation, not a Node CPU fallback.
