# 30 — Pixal3D conditioning: ProjectAttention, ProjGrid, multiview fusion, NAF

Reverse-engineered from `TencentARC/Pixal3D` @ `f7cf384` (`pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`,
`pixal3d/modules/{attention,sparse/attention}/proj_attention.py`, `pixal3d/pipelines/pixal3d_mv_image_to_3d.py`,
`inference_mv.py`) and `valeoai/NAF` (`src/model/naf.py`, `src/layers/{attentions,convolutions,rope}.py`,
checkpoint `naf_release.pth`). Everything else in the cascade (DINOv3, samplers, SS/shape/tex decoders, postprocess)
is identical to TRELLIS.2 — see specs 01–29.

## 1. DiT delta (`image_attn_mode = "proj"`)

All four MV flow checkpoints (`ss_flow_img_dit_1_3B_64_bf16_mv`, `slat_flow_img2shape_dit_1_3B_{512,1024}_bf16_mv`,
`slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv`) set `image_attn_mode: "proj"`; the SLAT ones add
`proj_in_channels: 2048`, the SS one omits it (→ `proj_in = cond_channels = 1024`).

Per block the cross-attention sub-layer becomes

```
global_out = cross_attn_block(norm2(h), global_cond)   # unchanged MHA, weights under blocks.i.cross_attn.cross_attn_block.*
proj_out   = proj_linear(proj_cond)                    # nn.Linear(proj_in, 1536, bias) — per latent token
h          = h + global_out + proj_out                  # same residual slot as TRELLIS.2's cross-attn
```

Tensor names: `blocks.i.cross_attn.cross_attn_block.{to_q,to_kv,to_out}.{weight,bias}`,
`blocks.i.cross_attn.cross_attn_block.{q,k}_rms_norm.gamma [12,128]`, `blocks.i.cross_attn.proj_linear.{weight [1536,proj_in], bias}`.
Nothing else differs from TRELLIS.2 (701 tensors SS / 700 SLAT). `gated_proj` (DINO+VAE) exists in code but no
released checkpoint uses it.

Conditioning tensors:

| name | shape | source |
|---|---|---|
| `global` | `[B, 5, 1024]` | CLS + 4 register DINOv3 tokens (after final non-affine LayerNorm), mean over views |
| `proj` (SS) | `[B, R³=4096, 1024]` | ProjGrid samples, R=16, token order `ix*R*R + iy*R + iz` (same as latent) |
| `proj` (SLAT) | `SparseTensor feats [N, 2048]` | dense `[B, R, R, R, 2048]` gathered at active coords `(b,x,y,z)`; R = 32 (512 stage) / 64 (1024 stage) / `hr_res//16` for the 1536 cascade |
| negative (CFG) | zeros for both | `proj_linear(0) = bias` |

## 2. ProjGrid (pixel-aligned projection)

```
one = linspace(-1, 1, R); (x,y,z) = meshgrid(one, one, one, 'ij')
p = stack(x,y,z) @ Rot^T,  Rot = [[1,0,0],[0,0,-1],[0,1,0]]        # → [R³, 3], z fastest
p = p / mesh_scale / 2
F = [[1,0,0,0],[0,0,-1,-2],[0,1,0,0],[0,0,0,1]]; F[1,3] = -distance   # canonical front view c2w (Blender, camera looks -Z, +Y up)
cam = (p_h @ inv(c2w)^T)[:3];  depth = -z_cam
focal_px = (16 / tan(fov_x/2)) * S / 32                              # S = image_resolution (512 or 1024), 32 mm sensor
x_pix =  focal_px * x_cam / (-z_cam + 1e-8) + S/2
y_pix = -focal_px * y_cam / (-z_cam + 1e-8) + S/2
norm  = (pix + 0.5) / S * 2 - 1
out[k] = grid_sample(fmap[C,H,W], norm, bilinear, align_corners=False, padding_mode='border')   # → [R³, C]
```

`valid_mask` (inside image and depth>0) is computed but **not applied**; border padding clamps out-of-view points.
`fmap` is the 32×32 (S=512) / 64×64 (S=1024) DINOv3 patch map (BHWC) for the LR branch, or the NAF map
(`[C, S_naf, S_naf]`, BCHW) for the HR branch; H,W ≠ S is fine because sampling uses normalized coordinates.

Stage configs (`inference_mv.IMAGE_COND_CONFIGS`):

| stage | image_size S | grid R | NAF target | proj channels |
|---|---|---|---|---|
| ss | 512 | 16 | — | 1024 |
| shape_512 | 512 | 32 | 512 | 2048 = [lr ‖ hr] |
| shape_1024 | 1024 | 64 | 512 | 2048 |
| tex_1024 | 1024 | 64 | 1024 | 2048 |

Image → tensor: LANCZOS resize to S, RGB premultiplied by alpha (black background), then ImageNet normalize for
DINOv3 only (NAF receives the **un-normalized** [0,1] image).

## 3. Multiview (`DinoV3ProjMultiViewFeatureExtractor`, fusion = "average")

`transforms.json` frames: `transform_matrix` = 4×4 c2w (Blender/NeRF, Z-up world), `camera_angle_x` (rad, per frame or
top-level), `mesh_scale`; `distance_i = ‖c2w_i[:3,3]‖`. Frame 0 = main view, expected to be the canonical front view.

When the input directory has no `transforms.json`, the runtime synthesizes the canonical turntable rig instead
(`src/transforms_json.cpp::synthesize_canonical_rig`, identical to `web/real_e2e/calibration.js`): exactly 4 images in
natural filename order → front / right / back / left, elevation 0, `distance = 3.1192049980163574`,
`camera_angle_x = 0.3490658503988659`. `mesh_scale` has no default there and must be given explicitly
(`--mesh-scale`, or the `mesh_scale` form field on `POST /generate-mv`).

```
F' = F with F'[1,3] = -distance_0
calc_mat_i = F' @ inv(C_0) @ C_i                  # calc_mat_0 == F' → V=1 degenerates to single-view
z_proj   = mean_i ProjGrid(fmap_i, fov_i, calc_mat_i)   # views processed sequentially, accumulated in place
z_global = mean_i [CLS_i ‖ reg_i]                        # [B, 5, 1024]
```

Fused shapes are independent of V, which is why the denoisers need no MV-specific weights.

## 4. NAF (`valeoai/NAF`, `naf_release.pth`, 2.6 MB, dim=256, heads=4, kernel 9×9)

Inputs: `image [B,3,S,S]` in [0,1], `features [B,1024,h,w]` (DINO patch map, h=w=S/16), `output_size (T,T)`.

```
# ImageEncoder
if S > 4T: image = bilinear_resize(image, min(S, 4T))                      # not hit for S∈{512,1024}, T∈{512,1024}
e1 = encoder(image):     Conv2d(3→128, k1) → 2× EncBlock(k1)               # EncBlock: GN(8)→SiLU→conv1→GN(8)→SiLU→conv2, no residual
e2 = sem_encoder(image): Conv2d(3→128, k3, reflect pad) → 2× EncBlock(k3, reflect)
x  = adaptive_avg_pool2d(cat[e1,e2], (T,T))                                # [B,256,T,T]  (identity when S==T)
x  = RoPE(x): 4 heads × 64; periods_j = 100^(2j/32), j<16; coords (i+0.5)/T*2-1 per axis;
              angles = 2π·[u/periods ‖ v/periods] tiled ×2 (64); rope_apply(x, sin, cos) with rotate-half
# cross-scale neighborhood attention
q = x                                            # [B,256,T,T] → heads 4×64
k = nearest_exact_resize(adaptive_avg_pool2d(x, (h,w)), (T,T))   # pool 16×16 (RoPE'd) blocks, then replicate back
v = nearest_exact_resize(features, (T,T))        # [B,1024,T,T] → heads 4×256
out = na2d(q, k, v, kernel=(9,9), dilation=(T/h, T/w), stride=1, scale=64^-0.5)  # → [B,1024,T,T]
```

Neighborhood-attention semantics (NATTEN): with dilation d every query pixel attends only to pixels congruent to it
mod d; within that sub-lattice (length L_r = ⌈(L−r)/d⌉ per axis) the 9-wide window is centered on the query and
**shifted (clamped) to stay in bounds**, never zero-padded. Because k and v are 16× nearest-upsampled and d = 16,
each query effectively attends to a 9×9 block of DINO patches; softmax over 81 logits `q·k/8`, output = weighted
sum of the 81 patch features. Output map `[1024, T, T]` is then sampled by ProjGrid (HR branch) and concatenated
after the LR branch: `proj = [lr ‖ hr]` (2048).

Implementation note: only the R³ projected points (× 4 bilinear corners) are ever read from the NAF map, so an
on-demand evaluation at those pixels is exact and avoids materialising the 1024×T×T map (1 GB f32 at T=512 per view).

Implementation status: `src/naf.cpp` is the CPU reference (threaded, bit-exact vs the fixture at T=128/512; ~52 s per
view at T=512 on 32 cores). CUDA builds dispatch to `src/naf_gpu.cpp` (ggml graph for the encoder) + `src/naf_attn.cu`
(custom neighborhood-attention kernel that indexes the pooled 32×32 key/value maps directly — with an integer
upsample factor d the dilated window position `start+i` *is* the low-res index); output rel 6.5e-4 vs the fixture,
~11 s wall at T=512 on a loaded box. `TRELLIS_NAF_CPU=1` forces the CPU path; Vulkan/Metal kernels are still TODO
(they fall back to CPU).

## 5. Numerical conditioning of the Pixal3D SS flow (measured 2026-09-06)

The MV SS flow develops "massive activations" (|h| up to ~6.5e4 at token 2066 / channels 671, 1425 from
block 3 on; the SLAT flows sit at 1–5e4), so every per-op rounding error is amplified through the 30
blocks and the final LayerNorm:

| implementation | per-op error | after_block0 | after_block29 | output |
|---|---|---|---|---|
| PyTorch f32 vs f64 (`ss_flow_proj_f64`) | ~6e-8 | 1.3e-6 | 1.7e-4 | 8.7e-4 |
| C++ ggml CPU, f32 weights, exact SDPA | GELU f16 table ~1e-4 | 1.1e-4 | 8.6e-3 | 8.3e-2 |
| C++ Metal / CUDA, f32 weights, exact SDPA | F32 GEMM stages inputs as f16 (~3e-4) | 3.5e-4 | 1.6e-2 / 2.9e-2 | 6.3e-2 / 5.6e-2 |
| C++ CUDA, f16 weights, FlashAttention | + bf16 K/V | 6e-3 | 0.19 | 0.53 |

Block-0 attention / cross-attention / `proj_linear` match to 1–3e-6 on the CPU backend, i.e. the
wiring is exact; the drift is precision, not logic. Note the checkpoints are stored as **F32**
(despite `bf16` in the file names), so an f16 GGUF rounds the weights (visible as 2e-4 on `t_emb_mod`).
Consequence: block-level golden comparisons are only meaningful on the CPU backend; the production
gate must be end-to-end sampling parity (SS occupancy / SLAT samples) against the reference run in
its own production dtype (`bfloat16`), with reference-f32-vs-bf16 as the calibration baseline.

## 6. Validation fixtures

`tools/ref_pixal3d_{proj_grid,ss_flow,slat_flow,cond_ss}.py` dump golden tensors (win: `/mnt/hdd1/pixal3d/ref/pixal3d/`);
C++ tests `trellis-test-proj-grid`, `trellis-test-pixal3d-ss-flow`, `trellis-test-pixal3d-slat-flow`.
