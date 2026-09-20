# Official Pixal3D assets through pixal3d.cpp — SV and MV (2026-09-20)

Inputs: `TencentARC/Pixal3D/assets/images/*` (13 pre-matted RGBA + 6 `s_*` RGB scene photos) and
`assets/mv_images/example` (4 views + transforms.json, mesh_scale 1.0, FOV 20°).
Runtime: pixal3d.cpp commit `ec464fe`, CUDA build (WSL2, RTX 4090), seed 42, res-1024 cascade,
Pixal3D default postprocess (QEM 1M faces, 4096² atlas).
Weights: SV = `pixal3d-sv-q8_0 v1`, MV = `pixal3d-q8_0 v1` (Studio desktop-alpha install), `birefnet.gguf` (ilintar/trellis2-gguf).
SV path = `trellis-cli --bg-only` (RGB inputs only) → `tools/run_pixal3d_estimated.py` (crop, MoGe-2 FOV, canonical front camera, `--pixal3d-weights sv`).
All 21 runs exit 0 (20 SV + 1 MV). Another session shared the GPU during the batch (the script waited between assets), so wall times are indicative only.

## MV example: pixal3d.cpp vs official PyTorch (`inference_mv.py`, seed 42, bf16, run 2026-09-06 on the same box)

| | pixal3d.cpp MV (4 views) | official PyTorch MV | pixal3d.cpp SV (view00 only) |
|---|---:|---:|---:|
| silhouette IoU vs the 4 input views (`tools/silhouette_iou.py --res=256`) | 0.9831 | 0.9830 | 0.8781 |
| mesh_scale recovered from projection (truth 1.0) | 0.999 | 0.996 | 0.983 |
| bbox extents | 0.756 × 0.846 × 0.991 | 0.757 × 0.845 × 0.987 | — |
| faces / GLB MB | 985K / 35.1 | 976K / 34.3 | 957K / 34.5 |
| symmetric Chamfer to official PyTorch MV (300K surface samples) | 0.0038 (0.25 % of bbox diag; 95 % of points within 0.01) | — | 0.0267 (1.8 %) |
| symmetric Chamfer to pixal3d.cpp MV | — | 0.0038 | 0.0263 |

Caveat: C++ (mt19937) and torch RNG streams differ, so seed 42 is not the same noise — this is
same-input / same-weights (Q8_0 vs bf16) agreement, not bit parity (bit parity is covered by the
serialized-noise stage tests). A res-1024 voxel is ≈0.001, so 0.0038 is a few voxels.
Renders: `mv_example_ours_vs_pytorch.webp` (ours | PyTorch), `mv_example_sv_vs_mv_vs_pytorch.webp` (SV | MV | PyTorch).

## SV on the 19 single images (+ MV example front view)

Silhouette IoU is against the *staged input crop* used for generation (single view, 256²), with the
same manual FOV the run used; no official GLBs exist for these images, so there is no reference comparison.

| image | matte | MoGe-2 FOV | faces | GLB MB | flow+decode+post s | IoU vs input | mesh_scale est. |
|---|---|---:|---:|---:|---:|---:|---:|
| 0_img | pre-matted | 29.5° | 985K | 38.0 | 235 | 0.9730 | 0.996 |
| 1_img | pre-matted | 34.6° | 979K | 48.6 | 232 | 0.9573 | 1.000 |
| 3_img | pre-matted | 34.1° | 987K | 43.0 | 178 | 0.8643 | 1.000 |
| 4_img | pre-matted | 29.3° | 985K | 38.6 | 214 | 0.9777 | 0.996 |
| 5_img | pre-matted | 19.9° | 940K | 40.6 | 291 | 0.9735 | 0.991 |
| 6_img | pre-matted | 52.1° | 963K | 40.7 | 291 | 0.9473 | 0.991 |
| 7_img | pre-matted | 41.9° | 920K | 37.0 | 172 | 0.9054 | 0.996 |
| 9_img | pre-matted | 18.7° | 960K | 40.5 | 308 | 0.9477 | 0.987 |
| 10_img | pre-matted | 30.6° | 970K | 37.8 | 224 | 0.9711 | 0.996 |
| 11_img | pre-matted | 43.6° | 981K | 40.4 | 235 | 0.9156 | 0.987 |
| 12_img | pre-matted | 31.0° | 910K | 35.4 | 282 | 0.9579 | 0.983 |
| 17_img | pre-matted | 37.1° | 993K | 47.2 | 249 | 0.9647 | 1.000 |
| 21_img | pre-matted | 34.8° | 932K | 39.0 | 180 | 0.9047 | 0.987 |
| s_13_img | BiRefNet | 40.2° | 983K | 36.6 | 177 | 0.8686 | 0.991 |
| s_14_img | BiRefNet | 60.8° | 961K | 38.5 | 154 | 0.9140 | 0.945 |
| s_15_img | BiRefNet | 38.7° | 975K | 38.5 | 244 | 0.9673 | 0.987 |
| s_16_img | BiRefNet | 30.6° | 945K | 42.4 | 177 | 0.9586 | 0.987 |
| s_18_img | BiRefNet | 44.9° | 966K | 39.1 | 194 | 0.6798 | 0.992 |
| s_20_img | BiRefNet | 31.0° | 970K | 39.6 | 124 | 0.9316 | 0.992 |
| mv_example_view00 | pre-matted | 27.4° | 957K | 34.5 | 226 | 0.9774 | 0.983 |

Observations (gallery: `official_sv_gallery.webp`, per-asset 4-view `quad4k/<name>_quad4k.webp`, tiles = orbit 0/90/180/270° — the 180° tile is the input view):
- 17/20 have IoU ≥ 0.90; all recovered mesh_scale within 0.945–1.000 of the 1.0 gauge.
- Low-IoU cases are matte-driven, not port failures: **s_18** (0.68) — BiRefNet kept only two corners of the rug, the model generated the whole rug; **s_13** (0.87) — the matte's detached lampshade was dropped as a floater (2 %/3 % component filter); **3_img** (0.86) — thin palm fronds.
- **10_img** and **5_img** (elevated / isometric inputs) come out pitched: the SV camera model has FOV but no elevation, so the object is generated relative to a horizontal front camera. Same behaviour as the official single-image path.
- `s_*` scene photos are matted by BiRefNet in the official pipeline too (`ZhengPeng7/BiRefNet`), so fragmented results such as s_14 (coffee machine + floating cups/plates) are input properties.
- SV vs MV on the cyclops: SV hallucinates the unseen cheek (yellow-green blotch) and rounds the head; MV matches the PyTorch reference.

Files: `glb/<name>.glb` + `glb/<name>.run.log` (stdout of every run), `glb/ref_pytorch_mv_example.glb`, `sv_iou.tsv`.
Remote originals: `ssh win:/mnt/hdd1/pixal3d/out/official/` (also `.ply`, `_base.png`, cutouts); scripts `/mnt/hdd1/pixal3d/official_assets.sh` (MV) and `official_assets_sv.sh`.
