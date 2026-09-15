# canonical rig（transforms.json 無し）の同値検証 — 2026-09-15, macOS 26 / M4 Max (Metal)

同一バイナリ・同一 seed(1)・同一 4 枚。対照 = transforms.json あり、合成 = transforms.json を除いたコピー + --mesh-scale 1.0。

## 段階ごとの数値（完全一致）
```
== 対照/transforms.json
      active voxels @res32 = 4438
      upsampled coords @res512=1211860 -> quantized @res1024 (grid 64, Pixal3D round/grid-1 formula) = 17612 tokens
      mesh V=4739865 F=9493442
      filled 3368 small holes -> V=4743233 F=9513876
  fill_holes: 3841 boundary loops filled
  remesh_dc: 11834789 active voxels -> V=9265912 F=18539204 (eps=0.0009794, project_back=0.00)
== 合成/canonical rig
      active voxels @res32 = 4438
      upsampled coords @res512=1211860 -> quantized @res1024 (grid 64, Pixal3D round/grid-1 formula) = 17612 tokens
      mesh V=4739865 F=9493442
      filled 3368 small holes -> V=4743233 F=9513876
```

## GLB 比較
```
json.glb  32899848 bytes  sha256 28acc43bdfd6035149584ab12f6401e6cb8b28b9efd4ced49b40987bf761438c
synth.glb 32899848 bytes  sha256 176784f4ad96e82435d2c1ddd442515b19ec657687270927bc5d868169534f1a
glTF JSON chunk の差分: asset.extras.generated（生成時刻）のみ
BIN chunk の差分: 0 / 32,898,048 バイト（完全一致）
accessors: POSITION/NORMAL VEC3 639713, TEXCOORD_0 VEC2 639713, indices SCALAR 2850510
```
