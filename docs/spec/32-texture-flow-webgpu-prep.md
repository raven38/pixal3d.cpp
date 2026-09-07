# 32 — Texture-1024 flow on WebGPU: integration plumbing (prep phase)

Branch `feat/webgpu-texture-flow-prep` (from `v0.4.0-webgpu-shape512-flow` = `1d907a5`,
2026-09-06). Scope: everything needed so that, once the Shape-1024 stage is validated
(`feat/webgpu-shape1024-flow`, spec 31 §11), the texture stage can be *executed and debugged*
immediately -- entry points, fixtures, deterministic noise, reporting, memory accounting and the
browser harness. **No texture flow was executed on WebGPU in this phase, no backend/WGSL change
was made, no golden tensor was created, and Shape-512 behaviour is unchanged** (§8). The texture
stage was run on CUDA through the new plumbing (§8.2) to establish the `cuda_tex_x_*` reference.

Every stage number below is either measured in this phase (marked *measured*), copied from the
sibling branches' measurements of the identical graph (marked *Shape-1024 branch*), or derived
from tensor shapes (marked *derived*). Nothing here is a texture-parity result.

## 1. Texture flow model and config

| item | value | source |
|---|---|---|
| checkpoint | `slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv` (stored F32 despite the name, spec 30 §5) | `hr_sample/meta.json` |
| GGUF | `pixal3d_tex_flow_1024_mv.gguf`, 2,775,670,944 B (f16); win `/mnt/hdd1/pixal3d/gguf/`. **Not on the Mac** (§7) | win listing |
| vs `pixal3d_shape_flow_1024_mv.gguf` | +98,304 B = `input_layer.weight` `[1536, 64]` vs `[1536, 32]` in f16 -- the only weight-shape delta between the two flows | *derived* from the two GGUF sizes |
| `ElasticSLatFlowModel` args | resolution 64, **in_channels 64, out_channels 32**, model 1536, cond 1024, 30 blocks, 12 heads, mlp_ratio 5.3334, rope, share_mod, qk_rms_norm (+cross), `image_attn_mode "proj"`, `proj_in_channels 2048`, dtype bfloat16 | `meta.json` `tex_flow_config` |
| C++ params | `DiTParams{in_ch = 64, out_ch = 32, d_cond = 1024}` + `dit_detect_proj_attn` (d_proj 2048); `make_sparse_runner(m, p, hr_coords, 5)` | `trellis_cli.cpp` [5/6], `test_pixal3d_slat_sample.cpp --stage tex` |
| sampler | `FlowEulerGuidanceIntervalSampler`: steps 12, **guidance_strength 1.0**, guidance_rescale 0.0, interval [0.6, 0.9], rescale_t 3.0, sigma_min 1e-5 | `tex_sampler_params.npy` = `[12, 1, 0, 0.6, 0.9, 3]` (*measured*), `trellis_cli.cpp` |
| consequence of gs = 1.0 | `sample_flow` issues **one forward per step, 12 forwards total** (the `gs == 1.0f` branch; no CFG pair, no guidance rescale) -- vs 20 forwards for the shape stages | `flow_runner.cpp` |
| normalization | `tex_slat_normalization` mean/std (`TEX_MEAN`/`TEX_STD` in `trellis_cli.cpp`, `tex_norm_mean/std.npy`) applied to the final latent only | `pipeline_mv.json` |

### Native entry paths (the authoritative C++)

- Production: `trellis_cli.cpp` `trellis_run_mv` stage "[5/6] texture SLAT flow": condition
  `pixal3d_cond_slat(dino, naf, views1024, {S=1024, R=64, T=1024})` → `pixal3d_gather_proj(…, shc)`
  → `Model::load(pixal3d_tex_flow_1024_mv.gguf)` → `make_sparse_runner(in_ch 64)` → the `fwdp`
  lambda rebuilding `x64 = [state(0:32) ; slat_norm(32:64)]` every call → `sample_flow` →
  denormalize with `TEX_MEAN/TEX_STD` → `tex_decode`.
- Parity test: `trellis-test-pixal3d-slat-sample <tex_flow.gguf> <hr_sample> [gpu] --stage tex`
  (alias `--stage texture`), the same runner/lambda over the fixture (§4).
- Browser/WASM: `pixal3d_texture_run` in `src/pixal3d_wasm.cpp` (§5), same runner/lambda.

## 2. Input tensors (what the texture stage takes from Shape-1024)

All three come out of the Shape-1024 stage; nothing else is stage-specific.

| tensor | shape / dtype | origin | in the fixture |
|---|---|---|---|
| token list `hr_coords` | `[N, 4]` int32 `(b, x, y, z)`, N = **17,489** on the fixture object, grid 64 | LR→HR transition: `shape_upsample` + `round((c+0.5)/512·63)` + lexicographic `unique` -- identical for the Shape-1024 and texture flows (`trellis_run_mv`: `shc` is reused verbatim) | `hr_sample/hr_coords.npy` (shared) |
| concat half | `[N, 32]` f32, NORMALIZED shape SLAT, **== the Shape-1024 sampler's `x_final`** | `sample_tex_slat` re-normalizes the denormalized HR SLAT with the *shape* stats; `trellis_cli.cpp` passes `slat_norm` directly. *Measured* on the fixture: max\|`f32_tex_concat_cond` − `f32_shape_x_final`\| = 2.4e-7 (f32 and bf16 runs alike), i.e. a Shape-1024 run's own `x_final` dump is this tensor | `hr_sample/f32_tex_concat_cond.npy` (`bf16_` for the bf16 cascade) |
| DiT input per forward | `[N, 64]` = `[state(0:32) ; concat(32:64)]` (channel order per `structured_latent_flow.py` `sparse_cat([x, concat_cond])`), rebuilt on the host every forward; the sampler integrates only the 32 state channels | built by the test / WASM lambda |
| initial noise | `[N, 32]` f32, seed 43 `torch.Generator` | `hr_sample/tex_noise.npy` (`--noise FILE` to substitute) | |

Dependency on a *validated* Shape-1024 output: only the concat half. Until the WebGPU
Shape-1024 latent passes its gate, the fixture's `f32_tex_concat_cond.npy` (PyTorch's) is the
input, and the test measures the texture flow alone; afterwards `--concat-cond
<cpp_|cuda_|browser_>shape_x_final.npy` chains the two stages, and the test prints how far that
input sits from the reference's concat (the reference's own f32-vs-bf16 concat gap is rel 0.41,
*measured* -- the texture stage inherits whatever its shape input carries).

## 3. Conditioning tensors (`tex_1024` = `IMAGE_COND_CONFIGS["tex_1024"]`, spec 30 §2)

| tensor | shape | notes |
|---|---|---|
| `global` | `[1, 5, 1024]` f32 (host token-major == ggml `[1024, 5]`) | DINOv3 CLS + 4 registers at S = 1024, mean over views -- **the same DINOv3 pass as Shape-1024** (S=1024 both) |
| `proj` at tokens | `[N, 2048]` f32 = `[lr ‖ hr]` | lr: bilinear taps of the DINOv3 64×64 patch map at grid R = 64 (same as Shape-1024); hr: taps of the **NAF map at T = 1024** (Shape-1024 uses T = 512) -- the only conditioning difference between the two stages |
| negatives | zeros for both (`proj_linear(0)` = bias) | unused by the tex sampler (gs = 1.0 → no negative forward), still passed for interface parity |
| fixture | `hr_sample/tex_cond_global.npy` `[1,5,1024]`, `tex_cond_proj.npy` `[17489, 2048]` (143.3 MB) -- PyTorch's condition gathered at `hr_coords` | `tools/ref_pixal3d_hr_sample.py` |
| host/native producer | `pixal3d_cond_slat(dino, naf, views1024, Pixal3dSlatCondParams{1024, 64, 1024, mesh_scale})` + `pixal3d_gather_proj` | `trellis_cli.cpp` [5/6] |
| device producer (WebGPU) | `pixal3d_cond_slat_gpu(…, {1024, 64, 1024})` -- exists at this tag with dense `[64³, 2048]` accumulators; the coords-gathered overload (`coords` argument) is on the Shape-1024 branch | §7 |

NAF geometry at tex_1024: image 1024², DINOv3 patch grid h = 64, T = 1024 → block size d = T/h =
**16** (the same d Shape-512 validated at T=512/h=32), 64×64 = 4096 blocks of 16×16 pixels, 81
taps per block; encoder at 1024² (S ≤ 4T, no input resize), **no `POOL_2D` on the encoder output**
(`adaptive_avg_pool2d(x, (T,T))` is the identity when S == T -- Shape-1024's k=2 pooling lowering
is not used here).

## 4. Test plumbing added (`trellis-test-pixal3d-slat-sample`)

```
trellis-test-pixal3d-slat-sample <pixal3d_tex_flow_1024_mv.gguf> <hr_sample_dir> [gpu] \
    --stage texture --backend webgpu [--noise FILE] [--concat-cond FILE] [--dump DIR] [--ext PREFIX]
```

| flag | behaviour |
|---|---|
| `--stage texture` / `shape1024` | aliases of the existing `tex` / `shape_hr` (fixture names, sampler params, TEX_MEAN/STD unchanged) |
| `--backend cpu\|gpu\|cuda\|metal\|vulkan\|webgpu\|hip` | maps onto the `[gpu]` index every Pixal3D test and the WASM entry already use (`cpu` → −1, else 0 unless `[gpu]` is given), then **verifies the backend ggml picked by name** after `Model::load` and aborts on mismatch (`gpu` = any non-CPU). A `--backend webgpu` run on a Metal/CUDA build therefore fails at load instead of measuring the wrong device. Conflicting `[gpu]`/`--backend` pairs are rejected at parse time |
| `--noise FILE` | deterministic initial latent `[N,32]` from any `.npy` (default: the stage's fixture file) -- one serialized tensor for PyTorch / native / WebGPU |
| `--concat-cond FILE` (tex only) | the Shape-1024 latent to concatenate (default `f32_tex_concat_cond.npy`, fallback `bf16_`); shape-checked against `[N,32]`; its rel / cosine vs the fixture's f32 and bf16 concat are printed |
| per-step stats | each step line now carries `t_scaled`, the forward count and wall ms (forwards are grouped by `t_scaled`; the tex stage shows `forwards=1` on every step), plus the existing latent mean/std/max, rel / cos vs f32, rel vs bf16 / cuda / ext and the f32-vs-bf16 calibration |
| final | rel + mean per-token cosine vs f32 (existing), verdict (unchanged threshold `max(2·rel(f32,bf16), 5e-2)`), and a one-line `summary: stage= backend= N= in_ch= steps= forwards= fwd_ms= rel_final= cos_final= calib_rel=` |
| memory | `memory (<backend>): weights \| dit activations \| cond per forward (+ zero neg) \| state [N,32] \| tex concat half + [N,64] input per forward \| device-resident sum`, `device free/total` after weights and after the DiT alloc when the backend reports it (`ggml_backend_dev_memory`; CUDA/Metal do, WebGPU/CPU report 0 → "not reported"), and the host-side per-step trace size after sampling |

Existing fixtures wired (no new goldens): `hr_sample/` from `tools/ref_pixal3d_hr_sample.py` --
`hr_coords`, `tex_noise`, `tex_cond_global/proj`, `tex_norm_mean/std`, `tex_sampler_params`,
`f32_/bf16_tex_concat_cond`, `f32_/bf16_tex_x_step0..12`, `tex_x_final`, `tex_slat`. The 38
texture-stage files (217 MB) were copied from win to the Mac's `~/pixal3d_assets/ref/pixal3d/hr_sample/`
this phase; the shape-stage files were already there.

Expected run of the next phase (the deliverable command), and what it needs:

```
# native WebGPU (build-webgpu, Dawn): exact SDPA, f16 weights
TRELLIS_NOFA=1 ./build-webgpu/trellis-test-pixal3d-slat-sample \
    ~/pixal3d_assets/gguf/pixal3d_tex_flow_1024_mv.gguf ~/pixal3d_assets/ref/pixal3d/hr_sample \
    --stage texture --backend webgpu --dump <dir>            # fixture concat (PyTorch's Shape-1024)
    … --concat-cond <dir>/cpp_shape_x_final.npy              # chained to the WebGPU Shape-1024 latent
# CUDA reference of the same commit (win): --backend cuda, then copy cpp_tex_* -> cuda_tex_*
```

## 5. Browser / WASM plumbing (`pixal3d_texture_run`, `web/texture/`)

Same structure as `web/shape512/` (spec 31 §10.7): `run_playwright.js` / `index.html` → `main.js`
→ `worker.js` (WORKERFS mounts, one `ccall` under JSPI) → `web/ss/pixal3d_ss.wasm` (one module
for SS / Shape-512 / texture; `-sEXPORTED_FUNCTIONS` and `-sJSPI_EXPORTS` extended in
`web/ss/CMakeLists.txt`) → `run_texture_impl` → the same sparse `DitRunner` / `sample_flow` as
native.

```
const char* pixal3d_texture_run(dinov3_gguf, naf_gguf, tex_flow_gguf, cond_dir, sample_dir,
                                n_views, own_cond, concat_cond_npy);
pixal3d_texture_latent / _latent_size / _steps / _n_steps       // [N,32] row-major, per-step x n_steps
```

- `own_cond = 0` (default): fixture condition (`tex_cond_global/proj`), DINOv3/NAF **not
  loaded** (`dinov3_gguf`/`naf_gguf`/`cond_dir` may be empty). Memory = the flow alone (§6).
- `concat_cond_npy = ""`: `sample_dir/f32_tex_concat_cond.npy`; otherwise the file given -- the
  page mounts it under its own `/concat/` so a `browser_shape_x_final.npy` from `web/shape1024/`
  keeps its name. Its distance to the fixture's f32 concat is reported.
- `own_cond = 1`: computes the tex_1024 condition on the device with
  `pixal3d_cond_slat_gpu(…, {1024, 64, 1024})` and reports `z_global` / `proj_lr` / `proj_hr` at
  the tokens vs `tex_cond_*` -- wired, **expected to fail on a 4 GiB adapter** (§7); the entry
  prints the expected buffer sizes before loading anything.
- Report: the same lines as the native test (`memory (...)`, one `forward k: t_scaled ms` line
  per forward, per-step `latent` / `f32` / `bf16` parity with cos, final verdict under the
  unchanged criterion, `summary:`), then `RESULT: OK|FAIL`.
- Driver: `node texture/run_playwright.js <tex_flow.gguf> <hr_sample> <out_dir> [concat|-] [own]
  [dinov3] [naf] [cond_slat] [views]` (serve `web/` on 8199); saves `browser_tex_x_step<k>.npy`
  / `browser_tex_x_final.npy` for `--stage texture --ext <out_dir>/browser_`. Only the `tex_*`,
  `hr_coords`, `tex_norm_*`, `tex_sampler_params` files are handed to the page (not the 100+ MB
  decoder dumps or the shape references).
- Native debug build of the same entry: `pixal3d-ss-run --texture <tex_flow.gguf> <hr_sample>
  [concat|-] [own] [dinov3] [naf] [cond_slat] [views] [dump_prefix]`.

*Measured this phase*: the module builds with the new entry (`scripts/build_wasm_ss.sh`, emcc,
`web/ss/pixal3d_ss.wasm` 2.89 MB) and the native `pixal3d-ss-run` / CUDA `build-cuda` targets
compile; the page/worker/driver pass `node --check`. Not run in a browser (§8).

## 6. WebGPU ops and the largest tensors

### 6.1 Ops required (nothing new)

The texture DiT graph is `build_dit_dense` with `proj_attn`, exactly the Shape-512 / Shape-1024
graph; op-gap doc §2 already lists the texture flow under the same tag family
(`dit_N17484_dcond5_proj1`, "shape-1024 **and** texture-1024"). The op set executed on WebGPU
for the Shape-512 sampling (op-gap §9, spec 31 §10.5/§10.6) -- `MUL_MAT`, `ADD`, `MUL`, `SCALE`,
`NORM`, `RMS_NORM`, `SOFT_MAX`, `UNARY:GELU/SILU`, `GET_ROWS`, `SET_ROWS` (RoPE scatter with the
host index input), `CONCAT` (query-chunked exact SDPA), `CONT`/`CPY`, `PERMUTE`/`TRANSPOSE`/
`VIEW`/`RESHAPE` -- is the whole list; the only shape delta is `input_layer`'s
`mul_mat([1536,64] f16, [64,N] f32)` (K = 64 instead of 32). The dimensional thresholds at
N = 17,489 (13-14 attention query chunks, `SET_ROWS` of 13.4 M rows ≤ 65,535 workgroups,
`[8192, N]` MLP hidden) are the ones the Shape-1024 branch is exercising right now; the texture
flow crosses no threshold Shape-1024 does not. `FLASH_ATTN_EXT` stays unused (`TRELLIS_NOFA=1`
/ `g_no_fa` in the WASM entry).

The conditioning graph (own-cond path only) is the Shape-512 NAF/DINOv3 op set at larger sizes
(§6.2); `POOL_2D` lowering is *not* needed (S == T).

### 6.2 Expected largest tensors

Flow (identical graph to Shape-1024 except the input row; *Shape-1024 branch* measurements at
N = 17,489, `docs/PIXAL3D_WEBGPU_MEMORY.md` §8 there):

| tensor | size | note |
|---|---|---|
| weights (f16 GGUF on device) | 2647.3 MB | +0.1 MB vs the shape flow (`input_layer`) |
| DiT gallocr buffer, one forward, exact SDPA | ≈ 1861 MB | score chunk `[17489, 1279, 12]` f32 1024 MB (`kAttnChunkBytes`), MLP hidden `[8192, 17489]` 546.5 MB |
| condition per forward | 136.7 MB | `[2048, 17489]` proj + `[1024, 5]` global; zero negatives allocated but never uploaded (gs = 1.0) |
| DiT input `[64, 17489]` f32 | 4.5 MB | rebuilt per forward on the host (*derived*) |
| sampler state / concat half | 2.2 MB each | (*derived*) |
| **device-resident sum** | **≈ 4650 MB** (CUDA *measured* 4777.7 MB incl. the zero negative, §8.2) | same as Shape-1024's 4645 MB; fits the M1 Pro adapter (`maxBufferSize` 4,294,967,292 B per buffer) |
| per-step host trace | 12 × 2.2 MB = 27 MB | test / WASM report only |

Time (*Shape-1024 branch*, same graph): one exact-SDPA forward at N = 17,489 is minutes in
Chrome and ≈ 1-2 min natively; the texture stage needs **12** of them (vs 20 for shape), so
expect ≈ 60 % of the Shape-1024 sampling wall time.

Conditioning at tex_1024 (own-cond path, *derived* from the Shape-512/1024 measurements, spec 31
§10.8 / §11.4 and memory doc §7/§8):

| tensor | size at T = 1024 (per view) | Shape-1024 (T = 512) |
|---|---|---|
| NAF output map `[1024, T²]` f32 | **4,294,967,296 B -- 4 bytes over `maxBufferSize` (4,294,967,292 B)**, produced twice (block-major attention output + its channel-major transpose) | 1024 MB × 2 |
| NAF window-gathered values `[1024, 81·4096]` + transpose | 1296 MB × 2 (same block count: 64×64 blocks) | 1296 MB × 2 |
| NAF logits / probabilities `[81, 256, 4, 4096]` | 1359 MB (d² = 256 queries per block vs 64) | 340 MB |
| RoPE tables `[T², 64]` f32 × 2 | 512 MB | 134 MB |
| encoder concat `[1024, 1024, 256]` | 1024 MB | 1024 MB |
| DINOv3 @1024 attention scores `[4101, 4101, 16]` | 1026.5 MB | same |
| dense proj accumulators `[64³, 2048]` (this tag) | 2147 MB | 137 MB gathered (Shape-1024 branch) |
| **sum, one view graph** | **> 12 GB** | 3.79 GB |

## 7. What Shape-512 / Shape-1024 do not exercise, and the real blockers

New relative to the two shape stages:

1. `in_ch = 64` -- `input_layer` `[1536, 64]` and the host-side `[N, 64]` concat rebuilt every
   forward (validated on CUDA, §8.2, with both FA and the exact SDPA the WebGPU path uses; not
   yet on WebGPU).
2. A sampler with `guidance_strength = 1.0`: the single-forward branch of `sample_flow`, 12
   forwards, no negative condition upload, no guidance rescale (CUDA-validated, §8.2; not WebGPU).
3. `tex_slat_normalization` for the denormalized comparison (`tex_norm_mean/std.npy`).
4. NAF at T = 1024 (4 M pixels): d = 16 blocks (validated size), 4096 blocks (validated count),
   but the `[1024, 1024²]` map and the `[81, 256, 4, 4096]` logits are 4× Shape-512's -- only a
   memory problem, no new op.
5. Chaining a stage output into the next stage's input (`--concat-cond`, `/concat/` mount).

Blockers that need the Shape-1024 phase to finish (in order):

| blocker | why it gates the texture phase | resolution |
|---|---|---|
| **Shape-1024 sampling gate on WebGPU** (spec 31 §11.5 "TBD") | the texture flow is the same graph at the same N; if the exact-SDPA chunking / `SET_ROWS` at N = 17,489 fail or drift there, they fail here identically. The chained input (`--concat-cond`) only exists once that latent passes | wait; then run §4's command twice (fixture concat, then the WebGPU `cpp_shape_x_final.npy`) |
| `WEBGPU_RUNTIME_WAIT_TIMEOUT_MS` 600 s → 7200 s | one N = 17,489 forward exceeds the 10-minute ceiling in Chrome; the bump lives on the Shape-1024 branch (root `CMakeLists.txt`, `web/ss/CMakeLists.txt`) and was deliberately **not** duplicated here (no backend/build-config change in this phase) | merge Shape-1024 first, or cherry-pick that one-line change |
| `pixal3d_cond_slat_gpu(…, coords)` (gathered accumulators) | the dense `[64³, 2048]` accumulators (2.15 GB) at this tag do not fit next to the NAF graph; the Shape-1024 branch's overload accumulates only the N tokens | merge Shape-1024 |
| **NAF map at T = 1024 exceeds `maxBufferSize` by 4 bytes** and the own-cond graph is > 12 GB | independent of Shape-1024: the T = 1024 map cannot be materialized on this adapter at all. Options (spec 30 §4 implementation note, memory doc §7 findings): on-demand NAF evaluated only at the taps of the N active tokens (N × 4 = 70 k pixels of the 1 M → the hr half becomes `[1024, 70k]` ≈ 287 MB), or block-chunked attention with the taps gathered per chunk. Both are graph-side changes in `naf_gpu.cpp` / `pixal3d_cond_gpu.cpp`, no new kernel | a follow-up phase; until then the texture stage on WebGPU runs on the fixture condition (PyTorch's), which is what the gate in §4 measures anyway |
| the texture GGUF is not on the Mac | `pixal3d_tex_flow_1024_mv.gguf` is 2.78 GB; the Mac has ≈ 1.2-1.5 GiB free (`df`, this session). Copying it requires freeing space (e.g. one of the three local flow GGUFs) -- a user decision | before the first native/browser run |

Not blockers: the sparse decoders (texture latent → PBR needs `tex_decode`, sparse conv -- out of
scope of every WebGPU phase so far; the latent is the gate), the mesh.

## 8. Verification done in this phase (and what was not)

| check | result |
|---|---|
| `trellis-test-pixal3d-slat-sample` builds: Mac default (Metal) build, win CUDA build (`build-cuda`, sm_89) | OK / OK (*measured*) |
| `pixal3d-ss-run` (native build of the WASM entry) builds: Mac, win CUDA | OK / OK |
| `scripts/build_wasm_ss.sh` (emcc 6.0.9, emdawnwebgpu, JSPI) with `pixal3d_texture_run` exported | OK, `web/ss/pixal3d_ss.wasm` |
| Shape-512 regression: the pre-change binary and the new binary on the same 256-token subset of `slat_sample` (CPU backend, `--backend cpu`) | **identical**: all 72 numeric parity lines (per-step latent mean/std/max, max\|d\|/mean\|d\|/rel vs f32/bf16, cos, final verdict) match byte-for-byte (§8.1) |
| argument / fixture error paths of the new flags (unknown stage or backend, `--backend`/`[gpu]` conflicts, `--concat-cond` on a shape stage, noise or concat file with the wrong token count) | each rejected with a one-line message and exit 1 before any model load (*measured*) |
| `--concat-cond f32_shape_x_final.npy` on the texture fixture | reported `rel 6.75e-8, cos 1.0000000` vs `f32_tex_concat_cond` -- the Shape-1024 latent is the concat half (*measured*, fixture-loading path; the model load that follows was not run) |
| Shape-512 full fixture, CUDA (RTX 4090, `--backend cuda`, FA) | `rel(mine, f32) = 2.1947e-01`, cos 0.996954, PASS -- the digits of record (spec 31 §10.6: 0.2195) (*measured*) |
| **Texture stage, CUDA** (RTX 4090): `--stage texture --backend cuda`, fixture concat; `--concat-cond f32_shape_x_final.npy`; `pixal3d-ss-run --texture` (the WASM entry natively, exact SDPA); `--backend webgpu` on the CUDA build | all PASS / the mismatch aborts as designed -- §8.2 (*measured*, run at the user's request after the branch was pushed) |
| texture stage on WebGPU (native Dawn or browser) | **not run** -- the Mac GPU was occupied by the Shape-1024 validation and the texture GGUF is not on the Mac |

### 8.1 Shape-512 subset regression

A 256-token prefix of the `slat_sample` fixture (coords, noise, proj, references sliced;
`cond_global`, norm tables unchanged) sampled on the ggml CPU backend with the binary built
from `v0.4.0-webgpu-shape512-flow` and with this branch's binary. The parity numbers are
meaningless as a gate (the attention context is truncated, both runs FAIL the threshold), the
point is that both binaries produce the same latents: the 72 numeric lines of the two logs
(`grep -E "latent mean|max\|d\||cos\(mine|rel\(mine"`) are identical (`diff` empty). 20 forwards
(the shape sampler's CFG pairs), 9.6 s per forward on the M1 Pro CPU backend at N = 256, both
runs. New lines in the same run, for reference:

```
memory (CPU): weights 2646.9 MB | dit activations 16.6 MB | cond per forward 2.0 MB (+ zero neg 2.0 MB) | state [N=256,32] 0.03 MB | device-resident sum 2667.6 MB
memory (host): per-step trace 12 x 0.03 MB = 0.4 MB
timing: 20 forwards over 12 steps, 9567.6 ms per forward, 191.4 s sampling wall (191.4 s inside forwards)
  step  1: t_scaled=1000.000 forwards=2 45930.8 ms
  step 12: t_scaled=214.286 forwards=1 4788.8 ms
summary: stage=shape512 backend=CPU N=256 in_ch=32 steps=12 forwards=20 fwd_ms=9567.6 rel_final=1.6665e+00 cos_final=0.557965 calib_rel=1.1384e-01
```

(The subset's `rel_final` 1.67 / FAIL is the truncated-context artefact described above, equal in
both binaries; the full-fixture Shape-512 numbers of record remain spec 31 §10.6.)

### 8.2 Texture stage on CUDA (RTX 4090, `hr_sample`, N = 17,489, 12 forwards)

Three runs of this branch's binaries on win (`build-cuda`, sm_89), the same fixture noise and
condition, PyTorch f32 as the target and the reference's f32-vs-bf16 run as calibration.
`cuda_tex_x_*` (run A) and `cuda_nofa_tex_x_*` (run C) are now in the fixture directory (win and
Mac) as the third reference for the WebGPU runs, exactly like the Shape-512 phase's `cuda_*`.

| run | concat half | attention | ms / forward | final rel vs f32 | cos | verdict |
|---|---|---|---|---|---|---|
| A `trellis-test-pixal3d-slat-sample --stage texture --backend cuda` | fixture `f32_tex_concat_cond` | FlashAttention (production) | 1655 | **3.93e-3** | 0.999993 | PASS (threshold max(2·0.345, 0.05) = 0.690) |
| B same, `--concat-cond f32_shape_x_final.npy` | the Shape-1024 latent (rel 5.5e-8 / cos 1.0000000 vs the fixture concat, reported by the flag) | FA | 1626 | 4.71e-3 | 0.999994 | PASS, `concat_cond=external` |
| C `pixal3d-ss-run --texture` (the WASM entry) | fixture | exact SDPA (`g_no_fa`, what WebGPU will run) | 3428 | **3.66e-3** | 0.999999 | PASS, `RESULT: OK` |
| D `--backend webgpu` on the CUDA build | -- | -- | -- | -- | -- | exit 1: `--backend webgpu requested but the model loaded on 'CUDA0'` |

| per step, run A (rel / cos vs f32) | 1 | 4 | 8 | 12 |
|---|---|---|---|---|
| CUDA FA | 2.5e-4 / 1.0000000 | 1.1e-3 / 1.0000000 | 4.6e-3 / 0.9999994 | 3.9e-3 / 0.9999935 |
| CUDA exact SDPA (run C) | 1.8e-4 / 1.0000000 | 3.5e-4 / 1.0000000 | 1.3e-3 / 0.9999999 | 3.7e-3 / 0.9999988 |
| (PyTorch f32 vs bf16) | 1.0e-2 | 5.6e-2 | 2.07e-1 | 3.45e-1 |

Final latent statistics: mean −0.019715, std 0.543219, max 4.2583 (A); −0.020013 / 0.543457 /
4.2583 (C); no non-finite values. Denormalized texture SLAT (run A): rel 3.3e-3 vs f32
(calibration 0.290). The texture stage sits two orders of magnitude closer to the f32 reference
than the shape stages (Shape-512: 0.22): with guidance 1.0 there is no CFG difference to amplify
the per-op rounding through the 12 steps, and both `forwards=1` per step and the 12-forward total
are visible in the new per-step lines.

Memory (run A, `memory (CUDA0)` lines): weights 2647.0 MB, DiT activation buffer 1850.9 MB (FA)
/ 1863.2 MB (exact SDPA, run C), condition 136.7 MB per forward (+136.7 MB zero negative,
never uploaded), state 2.13 MB, concat half 2.13 MB + `[N,64]` input 4.27 MB per forward,
device-resident sum **4777.7 MB**; `ggml_backend_dev_memory` free 20340 → 18488 MB across the
DiT alloc (total 24563.5 MB). This is the §6.2 estimate (≈ 4650 MB) plus the texture-only
inputs.

---

# Texture-1024 flow on WebGPU: execution and parity (`feat/webgpu-texture-flow`, 2026-09-07)

Branch `feat/webgpu-texture-flow` (from the prep branch's `8972329`). This part of the document
is the record of the phase that *ran* the texture stage on WebGPU: every number below is
measured on the Apple M1 Pro (32 GB), native Dawn (the same prebuilt Dawn as spec 31 §7) or
Chrome 152 with the JSPI/WORKERFS module, on the `hr_sample` fixture (N = 17,489 tokens,
`f32_tex_concat_cond` = PyTorch's Shape-1024 latent as the concat half, fixture `tex_cond_*`
condition, `tex_noise` seed 43). **Dependency stated explicitly: this milestone consumes a
validated Shape-1024 latent from the fixture; it does not depend on, and was not chained to,
the live Shape-1024 WebGPU sampling of `feat/webgpu-shape1024-flow` (`v0.5.0-webgpu-shape1024-flow`
unresolved at the time of writing).** `--concat-cond <webgpu shape_x_final>` is the one-flag
step that chains them once that latent passes its own gate.

## 9. What changed on this branch

| change | why | regression |
|---|---|---|
| `patches/ggml-webgpu/0003`: 2D dispatch for the fused `rms_norm_mul` encoder + `rms_norm_mul.wgsl` row index from `(wid.x, wid.y)` (taken from the Shape-1024 branch; the only backend change) | the q/k `MultiHeadRMSNorm × gamma` at N = 17,489 is 12 × 17,489 = 209,868 rows > 65,535 workgroups; without it the fused op is silently truncated | `trellis-webgpu-ops --only rms_norm`: `rms_norm x gamma [128,12,17489]` rel 1.5e-7, `rms_norm alone` 1.3e-7 vs CPU, PASS |
| `WEBGPU_RUNTIME_WAIT_TIMEOUT_MS` 600 s → 7200 s (root and `web/ss/CMakeLists.txt`) | one texture forward is 245 s in Chrome (§11); the old ceiling would abort a slow adapter | -- |
| `trellis-test-pixal3d-slat-sample --probe-steps k1,k2,..` | single-step gates (§10): integrate sampler step k once from the f32 reference's `x_step{k-1}`, score against `x_step{k}` of f32 / `cuda_` / `cuda_nofa_`; exits before any full run | the full-run path is untouched (flag absent → identical control flow) |
| `--backend metal` accepts ggml-metal's device name `MTL0` | the name check rejected every Metal run | -- |
| `web/texture/run_playwright.js`: `PIXAL3D_WEB_PORT` | port 8199 was held by another session's server | -- |
| `pixal3d_wasm.cpp`: the texture `device-resident sum` is summed in `double` | under wasm32 the byte total (4790 MB) overflowed 32-bit `size_t` and printed 693.9 MB (= 4789.9 − 4096) in the first browser report | print-only |
| `trellis-webgpu-ops --only <substr>` + the two `rms_norm` cases | the regression for the patch | -- |

No change to `dit.cpp`, `flow_runner.cpp`, `naf_gpu.cpp`, `pixal3d_cond_gpu.cpp`, the decoders,
or any Shape-1024 code.

## 10. Single-step probes (the development regression)

`--probe-steps` runs one forward per probed step (the tex sampler has gs = 1.0 everywhere) from
the PyTorch f32 latent of the previous step, so each number is the error of ONE forward + one
Euler update, with no trajectory accumulation. Threshold 5e-2 (the full-run floor). Exact SDPA
(`TRELLIS_NOFA=1`) on both backends.

| step (t → t_prev) | native WebGPU, GPU alone: rel vs f32 / vs `cuda_nofa` | Metal (portability control): rel vs f32 / vs `cuda_nofa` | ms / forward WebGPU / Metal |
|---|---|---|---|
| 1 (1.0000 → 0.9706) | **2.62e-5** / 1.60e-4 | 5.92e-5 / 1.16e-4 | 208,045 / 136,016 |
| 6 (0.8077 → 0.7500) | **1.74e-4** / 5.58e-4 | 1.33e-4 / 5.56e-4 | 383,161 (shared GPU, see below) / 135,928 |
| 11 (0.3750 → 0.2143) | **1.45e-4** / 3.22e-3 | 1.56e-4 / 3.22e-3 | 198,119 / 135,469 |
| 12 (0.2143 → 0) | **3.31e-4** / 3.65e-3, three consecutive runs bit-identical | 3.67e-4 / 3.65e-3 | 198,402-204,450 / 132,778 |

cos(mine, f32) = 1.0000000 at every probe on both backends. The `cuda_nofa` column is the
distance to CUDA's *own trajectory* at that step (its `x_step{k}` starts from its own
`x_step{k-1}`), so it carries CUDA's accumulated 3.2-3.7e-3 -- the two backends agree with each
other to that level and both sit an order of magnitude closer to f32 than the reference's bf16
run (1.0e-2 at step 1, 3.45e-1 at step 12).

**The shared-GPU corruption, measured.** The first probe runs overlapped with another session's
Shape-1024 sampler on the same M1 Pro (`~/pixal3d.cpp/build-webgpu`, spec 31 §11's
nondeterministic ggml-webgpu/Dawn-Metal fault). Under that contention the same binary and inputs
gave: step 12 rel **6.27e-2** (cos 0.9986) and, on the next run, **2.90e-1** (cos 0.934) --
the two results differ from each other by rel 0.35, every token affected; step 11 **3.89e-2**;
forwards 358-416 s instead of ~200 s. Steps 1 and 6 in the same runs stayed at 2.6e-5 / 1.7e-4.
Serialized behind a lock (`mkdir /tmp/pixal3d_gpu.lock` + a `pgrep` guard on the other
worktrees' binaries; the sibling session honors the same directory) the fault did not
recur in any of the 7 native, 4 Metal and 12 browser forwards that followed. Contended numbers
are therefore excluded from every table here; a contended run is not evidence about the graph.

## 11. Full 12-step sampling (one run each, taken after the probes passed)

Verdict criterion unchanged: `rel(mine, f32 x_final) <= max(2·rel(f32, bf16), 5e-2)` =
max(0.690, 0.05) = 0.690 (the test's `rel` is the mean-absolute ratio; the L2-norm ratios of
the numpy cross-check are given separately and are ~2.5× smaller).

| run | forwards | ms / forward | wall | final rel vs f32 (test def.) | cos(mine, f32) | verdict |
|---|---|---|---|---|---|---|
| Chrome 152 / WASM (`web/texture/run_playwright.js`, lock held 05:07:58-05:57:28) | 12 | 244,812 (first 297,415, last 273,049) | 2937.7 s inside forwards, 2944.0 s in the module, 2944.2 s playwright | **1.6072e-3** | 0.9999998 | PASS, `RESULT: OK` |
| native Dawn, run 1 (lock held 05:57:36-06:39:52, no other GPU tenant known: the sibling session's next job was waiting on the same lock) | 12 | 210,927 | 2531.1 s | 1.7118e-1 (**invalid**: the step-3 forward was corrupted, see below) | 0.998723 | formally PASS (< 0.690), **not accepted** |
| native Dawn, run 2 (lock held 06:54:08-07:34:33) | 12 | 201,876 (steps 3-4: 237,228 / 228,573) | 2422.5 s | 1.1250e-1 (**invalid**: step 3 again, see below) | 0.998309 | formally PASS, **not accepted** |
| native Dawn, **run 3** (lock held 08:08:27-08:47:15; `trellis-test-pixal3d-slat-sample --backend webgpu --dump`) | 12 | **193,744** (192,618-195,549, uniform) | 2324.9 s | **1.4616e-3** | 1.000000 | **PASS** -- the milestone's native run |
| CUDA FA, RTX 4090 (prep phase §8.2, reference of record) | 12 | 1,655 | 20 s | 3.93e-3 | 0.999993 | PASS |
| CUDA exact SDPA (`cuda_nofa_`, what WebGPU computes) | 12 | 3,428 | 41 s | 3.66e-3 | 0.999999 | PASS |

Per-step browser latent statistics and distances (numpy L2-norm ratios; `cuda` = FA run A,
`cuda_nofa` = exact-SDPA run C of the prep phase):

| step | rel vs f32 | rel vs cuda | rel vs cuda_nofa | cos vs cuda_nofa | latent mean / std / max\|x\| |
|---|---|---|---|---|---|
| 1 | 1.07e-5 | 5.93e-5 | 5.75e-5 | 1.0000000 | −0.001435 / 0.969469 / 4.6272 |
| 2 | 1.97e-5 | 1.01e-4 | 9.11e-5 | 1.0000000 | −0.001970 / 0.936994 / 4.4813 |
| 3 | 3.11e-5 | 1.90e-4 | 1.29e-4 | 1.0000000 | −0.002579 / 0.900838 / 4.3180 |
| 4 | 4.43e-5 | 2.83e-4 | 1.72e-4 | 1.0000000 | −0.003312 / 0.860429 / 4.1335 |
| 5 | 6.35e-5 | 3.90e-4 | 2.26e-4 | 1.0000000 | −0.004207 / 0.815147 / 3.9228 |
| 6 | 8.58e-5 | 5.39e-4 | 2.91e-4 | 1.0000000 | −0.005287 / 0.764457 / 3.6807 |
| 7 | 1.20e-4 | 7.48e-4 | 3.76e-4 | 1.0000000 | −0.006590 / 0.708060 / 3.4003 |
| 8 | 1.67e-4 | 1.05e-3 | 4.89e-4 | 1.0000000 | −0.008183 / 0.646489 / 3.0749 |
| 9 | 2.38e-4 | 1.46e-3 | 6.50e-4 | 0.9999998 | −0.010141 / 0.582546 / 2.8518 |
| 10 | 3.43e-4 | 2.02e-3 | 8.83e-4 | 0.9999996 | −0.012603 / 0.525081 / 2.9494 |
| 11 | 4.86e-4 | 2.74e-3 | 1.18e-3 | 0.9999993 | −0.015769 / 0.497399 / 3.4581 |
| 12 = final | **6.17e-4** | 3.28e-3 | **1.48e-3** | 0.9999989 | −0.019989 / 0.543361 / 4.2620 |

(Test-definition rel of the browser run vs f32 per step: 2.62e-5, 1.02e-4, 1.21e-4, 1.56e-4,
2.41e-4, 3.14e-4, 4.39e-4, 5.68e-4, 6.43e-4, 2.66e-3, 2.06e-3, 1.61e-3; vs bf16 1.0e-2 → 3.44e-1,
i.e. the reference's own bf16 run is 200× farther from f32 than the browser is. No non-finite
value at any step.)

**Native run 1 was corrupted by the nondeterministic fault, without a co-tenant.** Its steps 1
and 2 agree with the browser run to rel 1.1e-8 and 5.1e-6 (numpy L2; the same bits up to the
float rounding of the host-side Euler update), then step 3 jumps to rel 3.2e-3 vs f32 (and vs
the browser) and the error grows monotonically to 5.8e-2 (test def. 1.71e-1, cos 0.9987) --
exactly the "step jump" signature of spec 31 §11's fault, one bad forward in 12, with the lock
held for the whole run and the sibling session's next job (`web/shape1024/run_playwright.js`)
queued behind the same lock since 05:48. **Run 2 reproduced it at the same step**: steps 1-2
bit-level with the browser again, step 3 rel 1.19e-2 (four times run 1's), final 6.4e-2 (L2) /
1.125e-1 (test def.). In both corrupted runs the affected forwards were also *slower*
(run 1: steps 2-5 213-240 s; run 2: steps 3-4 237 / 229 s) against 193-200 s for every clean
forward of run 3 and of the probes -- a timing fingerprint of the fault worth checking in the
Shape-1024 investigation. Run 3, the same binary and inputs, was clean at every step: its
trajectory follows the browser's to L2 rel 1.1e-8 (step 1) … 2.0e-4 (step 12), cos 1.0000000
throughout, and sits at 1.5e-3 from the CUDA exact-SDPA run at the end. A follow-up probe of
steps 1, 2, 3, 4 in ONE process (four consecutive forwards, lock held, 09:00) was clean at every
step (rel 2.6e-5 / 1.1e-4 / 6.8e-5 / 4.6e-5, 193-195 s each), so the fault is not tied to "the
third forward of a process" or to t = 937.5; it is intermittent (2 of 3 full runs, 0 of 5 probe
processes with 1-4 forwards, 0 of 12 Chrome forwards this session). The test's calibrated verdict still says PASS
(0.171 < 0.690) because the threshold is derived from the reference's own bf16 drift; that is
why the per-step tables and the CUDA/browser distances are reported alongside, and why run 1 is
not the milestone's native run. The dumps are kept as `tex_webgpu_full_run1_corrupt/` in the
session scratchpad. This is the first observation of the fault with no other WebGPU process on
the GPU; it is reported to the Shape-1024 investigation and not pursued here (out of scope).

Browser vs native WebGPU (run 3), numpy L2 rel per step 1 … 12: 1.1e-8, 5.1e-6, 9.1e-6,
1.3e-5, 1.8e-5, 2.6e-5, 3.4e-5, 4.7e-5, 6.8e-5, 9.9e-5, 1.5e-4, **2.0e-4**; cos 1.0000000 at every
step. The two Dawn builds (native prebuilt vs Chrome's) agree with each other 3× more tightly
than either agrees with f32, i.e. the residual is the f16-weight arithmetic, not the platform.
Native run 3 vs CUDA exact SDPA: 5.7e-5 → 1.48e-3 (L2); test-definition final rel vs `cuda_`
(FA) 3.3e-3.

## 12. Texture conditioning (`tex_1024`: S = 1024, R = 64, NAF T = 1024)

The production producer is `pixal3d_cond_slat(dino, naf, views1024, {1024, 64, 1024, mesh_scale})`
+ `pixal3d_gather_proj` at `hr_coords` (`trellis_cli.cpp` [5/6]); the device producer is the
view-sequential `pixal3d_cond_slat_gpu` validated for Shape-512 (spec 31 §10.4) with
`{1024, 64, 1024}`. The PyTorch fixture is `tex_cond_global` `[1,5,1024]` / `tex_cond_proj`
`[17489, 2048]` = `[lr ‖ hr]` gathered at the tokens.

| producer / backend | z_global | proj lr (DINOv3 taps at R = 64) | proj hr (NAF T = 1024 taps) | result |
|---|---|---|---|---|
| `pixal3d_cond_slat_gpu`, ggml WebGPU, V = 1 (`pixal3d-ss-run --texture … own_cond=1`) | -- | -- | -- | **cannot run**: `check_graph_supported` rejects the view graph before allocation -- 8 unsupported nodes, all `GET_ROWS` with the `[1024, 1048576]` f32 NAF map as source (4,294,967,296 B > `maxBufferSize` 4,294,967,292 B; ggml-webgpu's `supports_op` refuses tensors above the buffer limit). DINOv3 + NAF weights loaded (578.6 + 1.3 MB, 299 ms); the graph is never submitted. *Measured.* |
| `pixal3d_cond_slat_gpu`, Metal (`build-metal`, same C++ producer, V = 4) | **9.31e-5** (max\|d\| 2.7e-3, cos 0.9999999) | **5.82e-4** (max\|d\| 1.4e-2, cos 1.0000000) | **1.45e-4** (max\|d\| 3.5e-3, cos 1.0000000) | the Shape-1024 digits (8.9e-5 / 4.8e-4 / 1.0e-4 on the sibling branch); 75.5 s for V = 4 (slowest view 20.2 s), weights 579.9 MB + accumulators 2048.0 MB + view graph 11,359 MB = **13,987 MB** peak on Metal. The same process then sampled 12 steps on Metal with this own condition: final rel 1.38e-3 vs f32 (test def.), cos 1.000000, 133 s / forward -- not a gate, the portability data point |
| (fixture-condition path used by every flow/sampling number in §10/§11) | 0 | 0 | 0 | the gate input |

The WebGPU own-condition path therefore needs the on-demand / block-chunked NAF (spec 30 §4
note, §7 above) before z_global / lr / hr can be measured on WebGPU at T = 1024; on a
`maxBufferSize` ≥ 4 GiB + 4 B adapter the same graph would also need the coords-gathered
accumulators of the Shape-1024 branch (dense `[64³, 2048]` = 2.15 GB) to fit next to the 12 GB
NAF graph. Conditioning parity on WebGPU is thus established for the *Shape-512 / Shape-1024
producers* (T = 512) only; for T = 1024 the shared C++ producer is checked on Metal here.

## 13. Memory

Buffer-allocation accounting as in `docs/PIXAL3D_WEBGPU_MEMORY.md` §6/§7 (`memory (...)` lines
of the test and of the module), plus the process footprint of the native run.

| component | texture flow (this phase, N = 17,489) | Shape-1024 flow (sibling branch, same N) | Shape-512 flow (N = 4,377) |
|---|---|---|---|
| weights (f16 GGUF on device) | 2647.0 MB | 2646.9 MB | 2646.9 MB |
| DiT gallocr buffer, one forward, exact SDPA (activations + temporaries) | **1863.2 MB** (largest tensor inside: one `[17489,1279,12]` f32 score chunk = 1024 MB `kAttnChunkBytes`; `[8192,17489]` MLP hidden 546.5 MB) | 1861 MB | 1067.8 MB |
| conditioning per forward (`[2048,N]` proj + `[1024,5]` global) | 136.7 MB (+ 136.7 MB zero negative allocated on the host, never uploaded: gs = 1.0) | 136.7 MB | 36.1 MB |
| sampler state `[N,32]` | 2.13 MB | 2.13 MB | 0.53 MB |
| shape concat half `[N,32]` + rebuilt `[N,64]` DiT input per forward | 2.13 MB + 4.27 MB (texture only) | -- | -- |
| **device-resident sum** | **4789.9 MB** (native and browser identical; the browser's first report printed 693.9 MB = the wasm32 overflow, fixed in §9) | 4645 MB | 3751 MB |
| largest single allocation | the 2647 MB weight buffer; largest graph tensor 1024 MB | same | 2647 MB / 920 MB |
| native process `phys_footprint` during sampling (`footprint`, 43 s into the run) | **5.29 GB**, peak 5.43 GB (MALLOC_LARGE 467 MB host-side: fixture, references, `[N,64]` input) | -- | -- |
| host per-step trace (test / module report only) | 12 × 2.13 MB = 25.6 MB | | |
| Chrome | GPU-process RSS is not meaningful for Metal-backed buffers (98 MB by `ps`); the device-resident sum above is what the module allocates | | |

`ggml_backend_dev_memory` reports 4096 / 4096 MB on WebGPU (not a live query). The texture
stage is therefore +145 MB over Shape-1024 (the extra `[N,64]` input, the concat half and the
`input_layer` K = 64) and 1.04 GB over Shape-512, entirely from the N-scaled activations and the
2048-wide condition; nothing new was allocated for the texture path.

## 14. Regressions

All on the native Dawn build of this branch, through the GPU lock, after the changes of §9.

| path | command | result |
|---|---|---|
| SS stage on WebGPU (flow on WebGPU, SS decoder on the CPU backend) | `trellis-test-pixal3d-ss-sample pixal3d_ss_flow_mv.gguf ss_dec.gguf ss_sample 0 cond_ss -1` (`TRELLIS_NOFA=1`) | active-voxel IoU(mine, CUDA run) **1.0000**, IoU(mine, f32) 0.9977, IoU(mine, bf16) 0.9932 (calibration IoU(f32, bf16) 0.9945) -- PASS, the spec 31 §9 digits |
| Shape-512 stage on WebGPU | `trellis-test-pixal3d-slat-sample pixal3d_shape_flow_512_mv.gguf slat_sample --stage shape512 --backend webgpu` | rel(mine, f32) **2.1009e-1**, cos 0.997361, threshold 0.524 -- PASS (20 forwards, 17.3 s each, 346 s; the spec 31 §10.6 number of record) |
| texture conditioning | §12 | see there |
| texture flow (single-step probes) | §10 | 4/4 PASS native WebGPU (GPU alone), 4/4 Metal |
| texture sampling | §11 | Chrome PASS 1.61e-3; native run 3 PASS 1.46e-3 (runs 1-2 discarded: fault) |
| `trellis-webgpu-ops --only rms_norm` (patch 0003 update) | | 2/2 PASS, rel 1.5e-7 / 1.3e-7 |

(The first Shape-512 attempt aborted at model load: the Mac's `pixal3d_shape_flow_512_mv.gguf`
mirror had been removed to free the disk earlier in the session; it was re-copied from win and
the run repeated.)

## 15. Blockers remaining for a live end-to-end run

| blocker | status after this phase | what it takes |
|---|---|---|
| live Shape-1024 latent on WebGPU (`v0.5.0-webgpu-shape1024-flow`) | still open on its own branch; this milestone used the fixture's PyTorch latent as the concat half (rel 5.5e-8 from a Shape-1024 sampler's `x_final`, §2) | when that latent passes its gate: `--concat-cond <dir>/cpp_shape_x_final.npy` (native) or the `/concat/` mount (browser); no code change |
| nondeterministic ggml-webgpu/Dawn-Metal fault at N = 17,489 | reproduced here under GPU sharing (§10) **and once without a co-tenant** (native run 1, §11: 1 corrupted forward in 12, step-3 jump); Chrome 12/12 and the solo probes 7/7 were clean | the Shape-1024 branch's investigation; until then every long run is serialized through `/tmp/pixal3d_gpu.lock`, and a full run is accepted only when its per-step trajectory stays at the probe level (no step jump), never on the calibrated verdict alone |
| own-condition at T = 1024 on WebGPU | **解決済み (2026-09-08, `feat/browser-partial-e2e`)**: block-chunk 化した NAF と sparse coords 蓄積で、`[1024, 1048576]` の NAF map を一度も materialize しなくなった。単一グラフバッファ 11 344 MB → 3 332 MB (Metal) / 1 538 MB (Chrome WebGPU)。数値は単一グラフ版と global/lr がビット一致、hr が L2 rel 8.5e-09 | 実測と内訳は `docs/PIXAL3D_WEBGPU_MEMORY.md` §11、到達点は `docs/PIXAL3D_E2E_STATUS.md` |
| time | 200 s / forward native, 245 s in Chrome (12 forwards: 42 min / 49 min) vs 1.7-3.4 s on the RTX 4090 | FlashAttention-class attention on WebGPU (BF16/F16 FA tile path, spec 31 §5) or a smaller score budget with more chunks; out of scope here |
| texture decoder, mesh, GLB | validated separately on their own branches (`v0.7.0-webgpu-shape-decode`, `feat/webgpu-texture-decode`); not chained | browser E2E orchestration (roadmap M8 "full generation") |
| Mac disk | 2.78 GB per flow GGUF; the disk hit 0 bytes once this session (§9 of the memory note) | keep one flow GGUF mirror at a time or run the CUDA references on win |
