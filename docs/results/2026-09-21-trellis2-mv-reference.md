# TRELLIS.2 multi-image pinned reference / golden-fixture protocol (#59)

Date: 2026-09-21  
Parent: #58  
Primary reference: microsoft/TRELLIS.2 PR #104, pinned to fork commit

```text
dongjinlee123/TRELLIS.2
fe7b90c376cb2ed2fb48582d1b376b56e1e16c44
```

PR #104 base at the time:

```text
microsoft/TRELLIS.2
5565d240c4a494caaf9ece7a554542b76ffa36d3
```

Secondary semantic reference:

```text
microsoft/TRELLIS
442aa1e1afb9014e80681d3bf604e8d728a86ee7
```

The secondary reference is useful because original TRELLIS already contains the same
stochastic / multidiffusion sampler idea. The TRELLIS.2 PR is authoritative for this
project because it includes the current FlowEuler guidance-strength and
guidance-rescale conventions.

## Frozen fusion semantics

### stochastic

At sampler step `k`:

```text
positive condition = cond[k % V]
```

Then the unmodified `FlowEulerGuidanceIntervalSampler` handles the selected positive
condition and the single negative condition.

Consequences:

- the view counter advances once per sampler step;
- the counter is created when PR #104's injection context is entered;
- SS and texture each get a fresh counter;
- **cascade shape LR + HR share one injection context**, so HR continues after the LR
  12-step counter. For V=2/3/4/6/12 this happens to start at view 0 again; for
  V=5/7/8 it starts at `12 % V`;
- outside the guidance interval the selected positive prediction is used directly;
- when guidance strength is exactly 1, the original CFG mixin skips the negative model call.

### multidiffusion

At every sampler step:

```text
pred_pos[v] = raw model prediction for view v
pred_pos    = mean_v(pred_pos[v])

inside guidance interval:
    pred_neg = one raw negative prediction
    pred_cfg = guidance_strength * pred_pos
             + (1 - guidance_strength) * pred_neg
    optional guidance-rescale is applied to pred_cfg

outside guidance interval:
    pred = pred_pos
```

Important details copied from PR #104:

- averaging happens on raw model predictions, before CFG;
- the negative model is evaluated once per guided step, not once per view;
- PR #104 computes that negative prediction inside the interval even for
  `guidance_strength == 1`; this is intentionally frozen as reference behavior;
- guidance interval endpoints are inclusive;
- per-view positive forwards are sequential in the reference.

## Weightless oracle

`tools/ref_trellis2_mv.py --selftest` is stdlib-only and runs in Linux CI.

The expected values live in:

```text
tests/fixtures/trellis2_mv/synthetic_oracle.json
```

It freezes:

- rescaled timestep sequence;
- stochastic view schedule;
- guidance interval boundaries;
- negative-call behavior;
- final stochastic state;
- final multidiffusion state;
- V=1 equivalence;
- steps < views behavior;
- the PR #104 multidiffusion `guidance_strength == 1` negative-call quirk.

Run:

```bash
python3 tools/ref_trellis2_mv.py --selftest
```

Expected terminal marker:

```text
TRELLIS2_MV_ORACLE_OK
```

## Full reference fixture generator

The same script can generate full per-stage tensors from the pinned PR #104 checkout.
It refuses any other git HEAD by default and also rejects tracked local modifications.

Example checkout:

```bash
git clone https://github.com/dongjinlee123/TRELLIS.2.git /tmp/TRELLIS.2-mv-ref
git -C /tmp/TRELLIS.2-mv-ref checkout fe7b90c376cb2ed2fb48582d1b376b56e1e16c44
```

### 1-view control

```bash
python3 tools/ref_trellis2_mv.py \
  --repo /tmp/TRELLIS.2-mv-ref \
  --model /path/to/TRELLIS.2-4B \
  --images /path/to/view0.png \
  --mode stochastic \
  --pipeline-type 1024_cascade \
  --seed 42 \
  --device cuda:0 \
  --out /tmp/trellis2-mv-ref/v1-stochastic
```

Repeat with `--mode multidiffusion`. These two fixtures are the full-model V=1
control for the native sampler.

### 2-view and 4-view fixtures

```bash
python3 tools/ref_trellis2_mv.py \
  --repo /tmp/TRELLIS.2-mv-ref \
  --model /path/to/TRELLIS.2-4B \
  --images view0.png view1.png view2.png view3.png \
  --mode stochastic \
  --pipeline-type 1024_cascade \
  --seed 42 \
  --device cuda:0 \
  --out /tmp/trellis2-mv-ref/v4-stochastic
```

Run both `stochastic` and `multidiffusion`.

For the 512 fixture:

```bash
... --pipeline-type 512 ...
```

For optional 1536 metadata:

```bash
... --pipeline-type 1536_cascade ...
```

Use `--decode-summary` only when the final decoded mesh/voxel hashes are needed; the
stage-level fixture does not require it.

## Generated data contract

Each output directory contains `manifest.json` plus local `.npy` arrays.

The manifest records:

- pinned source SHA;
- input hashes and deterministic view order;
- preprocessed-image hashes;
- mode / pipeline type / seed;
- final cascade resolution;
- per-stage step counts;
- stochastic view schedules, including the shared LR→HR cascade counter;
- array shape / dtype / SHA256;
- one stable aggregate `fixture_hash`.

Stage tensors include:

### Conditioning

```text
cond512.npy       [V, L512, 1024]
neg512.npy        [1, L512, 1024]
cond1024.npy      [V, L1024, 1024]   (cascade)
neg1024.npy       [1, L1024, 1024]
```

### Sparse structure

```text
ss_noise.npy
ss_latent.npy
ss_coords.npy
```

### Shape SLat

```text
shape_lr_noise_{coords,feats}.npy
shape_lr_norm_{coords,feats}.npy
shape_lr_denorm_{coords,feats}.npy
shape_hr_upsampled_coords.npy
shape_hr_quant_coords.npy
shape_hr_noise_{coords,feats}.npy
shape_hr_norm_{coords,feats}.npy
shape_hr_denorm_{coords,feats}.npy
```

The HR entries are absent for the 512 pipeline.

### Texture SLat

```text
tex_noise_{coords,feats}.npy
tex_norm_{coords,feats}.npy
tex_denorm_{coords,feats}.npy
```

Large numeric arrays are **not** committed to this repository. Their SHA256/shape/dtype
metadata belong in the generated manifest and later result docs.

## Reproducibility rules

A fixture is comparable only when all of these match:

- PR #104 pinned SHA;
- input-image SHA256 and order;
- model checkpoint set;
- pipeline type;
- mode;
- seed;
- backend/device implementation;
- sampler parameters.

The script deliberately uses the pinned PR implementation's
`inject_sampler_multi_image()` rather than a locally rewritten full-model sampler.
The independent stdlib synthetic oracle exists to protect against accidentally
misreading that implementation.

## #59 completion gate

Code/protocol status:

- [x] PR #104 SHA pinned in code/docs.
- [x] original TRELLIS semantic reference pinned.
- [x] weightless exact synthetic oracle.
- [x] stochastic ordering and guidance boundary tests.
- [x] multidiffusion average-before-CFG semantics frozen.
- [x] full 512 / 1024-cascade fixture generator implemented.
- [x] per-stage tensor/hash manifest implemented.
- [ ] actual 1-view full fixtures generated and compared.
- [ ] actual 2-view full fixtures generated.
- [ ] actual 4-view full fixtures generated.
- [ ] repeat-run fixture hashes confirmed stable on the reference GPU environment.

Therefore #59 should remain open until the GPU fixture runs above are recorded. Once
those are done, #60 and #61 can proceed in parallel using the same manifests.
