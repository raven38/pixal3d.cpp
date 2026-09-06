# 32 — Texture-1024 flow on WebGPU: integration plumbing (prep phase)

Branch `feat/webgpu-texture-flow-prep` (from `v0.4.0-webgpu-shape512-flow` = `1d907a5`,
2026-09-06). Scope: everything needed so that, once the Shape-1024 stage is validated
(`feat/webgpu-shape1024-flow`, spec 31 §11), the texture stage can be *executed and debugged*
immediately -- entry points, fixtures, deterministic noise, reporting, memory accounting and the
browser harness. **No texture flow was executed on WebGPU in this phase, no backend/WGSL change
was made, no golden tensor was created, and Shape-512 behaviour is unchanged** (§8).

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
| **device-resident sum** | **≈ 4650 MB** | same as Shape-1024's 4645 MB; fits the M1 Pro adapter (`maxBufferSize` 4,294,967,292 B per buffer) |
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
   forward (validated on CUDA by the existing `--stage tex` parity run and the production CLI;
   not yet on WebGPU).
2. A sampler with `guidance_strength = 1.0`: the single-forward branch of `sample_flow`, 12
   forwards, no negative condition upload, no guidance rescale (CUDA-validated, not WebGPU).
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
| Shape-512 full fixture on a GPU, texture stage on any backend | **not run** -- the Mac GPU was occupied by the Shape-1024 validation, GPU jobs on win were not permitted for this phase, and the texture GGUF is not on the Mac. The texture-stage code path is therefore compile-checked only; the `--stage tex` numerics themselves are unchanged from the tag |

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
