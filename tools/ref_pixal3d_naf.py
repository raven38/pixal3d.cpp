#!/usr/bin/env python3
"""Golden dump for valeoai/NAF (neighborhood-attention feature upsampler), as used by
TencentARC Pixal3D's HR conditioning branch (docs/spec/30-pixal3d-cond.md section 4).

Loads the real `naf(pretrained=True)` model (dim=256, heads_attn=4, heads_rope=4,
kernel_size=9, rope_base=100.0, img_layers=2) from the torch.hub cache and runs it on
a real image + real DINOv3 patch map (view 0 of the existing SS conditioning fixture
at tools/ref_pixal3d_cond_ss.py's OUT dir), dumping every intermediate the C++ port
needs to reproduce dilated neighborhood attention bit-for-bit: the two encoder
branches before pooling, the pooled/RoPE'd query map, the pooled/upsampled key map,
the upsampled value map, the RoPE coordinate/angle tables, and — critically — the
exact 9x9 dilated-window neighbor pixel lists NATTEN attends to at 5 probe query
pixels (corners, an interior point, near-border points), derived analytically from
NATTEN's clamped-window algorithm and then verified by reconstructing `out` at those
5 pixels via manual softmax(q.k*scale)-weighted sum over exactly those neighbors.

    OUT=/mnt/hdd1/pixal3d/ref/pixal3d/naf COND_SS_DIR=/mnt/hdd1/pixal3d/ref/pixal3d/cond_ss \
        /mnt/hdd1/conda_envs/pixal3d/bin/python tools/ref_pixal3d_naf.py
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F

OUT = os.environ.get("OUT", "/mnt/hdd1/pixal3d/ref/pixal3d/naf")
os.makedirs(OUT, exist_ok=True)
COND_SS_DIR = os.environ.get("COND_SS_DIR", "/mnt/hdd1/pixal3d/ref/pixal3d/cond_ss")
NAF_HUB_DIR = os.path.expanduser(os.environ.get("NAF_HUB_DIR", "~/.cache/torch/hub/valeoai_NAF_main"))
NAF_CKPT = os.path.expanduser(os.environ.get("NAF_CKPT", "~/.cache/torch/hub/checkpoints/naf_release.pth"))
DEV = os.environ.get("REF_DEV", "cuda")

shapes = {}
def save(name, t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", a)
    shapes[name] = list(a.shape)
    print(f"  {name:24s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")
    assert not np.isnan(a).any(), f"{name} has NaNs"
    assert not np.isinf(a).any(), f"{name} has Infs"
    return a


# ---------------- load model ----------------
print("[NAF] loading model...")
try:
    model = torch.hub.load("valeoai/NAF", "naf", pretrained=True, device=DEV, trust_repo=True)
    load_path = "torch.hub.load(valeoai/NAF, naf, pretrained=True)"
except Exception as e:
    print(f"[warn] torch.hub.load failed ({e}); falling back to manual import + state_dict load")
    sys.path.insert(0, NAF_HUB_DIR)
    from src.model.naf import NAF
    model = NAF()
    sd = torch.load(NAF_CKPT, map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(DEV)
    load_path = f"manual NAF() + load_state_dict({NAF_CKPT})"
model.eval()
print("  ->", load_path)

import natten
print("  natten version:", getattr(natten, "__version__", "?"))
try:
    from natten.functional import na2d_av, na2d_qk  # noqa: F401
    natten_recent = False
except Exception:
    natten_recent = True
print("  natten_recent (legacy na2d_qk/na2d_av unavailable) =", natten_recent)
assert natten_recent, "expected the modern na2d() codepath (matches CrossAttention.forward's NATTEN_RECENT branch)"

# ---------------- inputs: view 0 of the real cond_ss fixture ----------------
images_512 = np.load(f"{COND_SS_DIR}/images_512.npy")   # [1,V,3,512,512], [0,1], premultiplied, un-normalized
dino_tokens = np.load(f"{COND_SS_DIR}/dino_tokens.npy")  # [V,1029,1024]  CLS + 4 reg + 32*32 patches
image = torch.from_numpy(images_512[:, 0]).to(DEV).float()          # [1,3,512,512]
patch_tokens = dino_tokens[0, 5:]                                     # [1024,1024] drop CLS+4 reg
patch_tokens = patch_tokens.reshape(32, 32, 1024)[None]               # [1,32,32,1024]
lr_features = torch.from_numpy(patch_tokens).to(DEV).float().permute(0, 3, 1, 2).contiguous()  # [1,1024,32,32]
print("[Inputs]")
save("image", image)
save("lr_features", lr_features)
S = image.shape[-1]
assert S == 512

ie = model.image_encoder
rope = ie.rope
upsampler = model.upsampler
scale = upsampler.scale
print(f"[NAF] scale={scale} (expect head_dim**-0.5 = {(256 // 4) ** -0.5})")
assert abs(scale - (256 // 4) ** -0.5) < 1e-9


def axis_window(pos: int, length: int, dilation: int, k: int):
    """NATTEN's clamped dilated-window neighbor indices along one axis: pixels
    congruent to `pos` mod `dilation` form a sub-lattice of length
    L_sub = ceil((length - r)/dilation); the k-wide window is centered on pos's
    sub-lattice index and clamped (shifted, never padded) to stay in [0, L_sub)."""
    r = pos % dilation
    i_sub = pos // dilation
    L_sub = -(-(length - r) // dilation)  # ceil division
    start = i_sub - k // 2
    start = max(start, 0)
    start = min(start, L_sub - k)
    return [r + (start + w) * dilation for w in range(k)]


meta = {
    "source_commit_naf": "valeoai/NAF (torch.hub cache, naf_release.pth, 2.6 MB)",
    "script": "tools/ref_pixal3d_naf.py",
    "seed": "n/a (deterministic: no RNG, real image + real DINO map + frozen weights)",
    "dtype": "float32",
    "device": DEV,
    "load_path": load_path,
    "natten_version": getattr(natten, "__version__", "?"),
    "natten_codepath": "modern na2d() (legacy na2d_qk/na2d_av unavailable in installed natten)",
    "na2d_default_scale": f"scale=None -> natten resolves query.shape[-1]**-0.5; head_dim=256/4heads=64 -> scale={scale}",
    "config": {"dim": 256, "heads_attn": 4, "heads_rope": 4, "kernel_size": 9, "rope_base": 100.0,
               "rope_rescale": 2.0, "img_layers": 2},
    "rope_rescale_note": "rope_rescale=2.0 is only applied when RoPE.training is True (see rope.py create_coordinate); "
                         "model.eval() -> not applied here, matching inference.",
    "shapes": {},
    "per_T": {},
}

runs = {}
for T in (128, 512):
    pfx = f"t{T}_"
    print(f"\n=== T={T} ===")
    o_size = (T, T)
    downsample = image.shape[-2] > 4 * o_size[0] or image.shape[-1] > 4 * o_size[1]
    print(f"  S={S} 4T={4*T} S>4T -> downsample_triggered={downsample}")
    x_img = image
    if downsample:
        # mirrors ImageEncoder.forward's F.interpolate branch exactly (not hit for our S,T)
        x_img = F.interpolate(
            image,
            size=(min(image.shape[-2], 4 * o_size[0], 4 * o_size[1]), min(image.shape[-1], 4 * o_size[1], 4 * o_size[0])),
            mode="bilinear", align_corners=False,
        )

    with torch.no_grad():
        e1 = ie.encoder(x_img)
        e2 = ie.sem_encoder(x_img)
        enc_cat = torch.cat([e1, e2], dim=1)                          # [1,256,H,W]
        enc_pooled = F.adaptive_avg_pool2d(enc_cat, output_size=o_size)  # [1,256,T,T]
        q_rope = rope(enc_pooled)                                     # [1,256,T,T]  == queries
        k_pooled = F.adaptive_avg_pool2d(q_rope, output_size=lr_features.shape[-2:])  # [1,256,32,32]
        k_up = F.interpolate(k_pooled, size=o_size, mode="nearest-exact")  # [1,256,T,T] before head split
        out = upsampler(q_rope, k_pooled, lr_features, x_img)          # [1,1024,T,T]  real CrossAttention path

    dilation = T // lr_features.shape[-1]
    assert upsampler.dilation == (dilation, dilation)
    print(f"  dilation={dilation}")

    save(pfx + "enc_cat", enc_cat)
    save(pfx + "enc_pooled", enc_pooled)
    save(pfx + "q_rope", q_rope)
    save(pfx + "k_pooled", k_pooled)
    save(pfx + "k_up", k_up)
    save(pfx + "out", out)

    if T == 128:
        with torch.no_grad():
            v_up = F.interpolate(lr_features, size=o_size, mode="nearest-exact")  # [1,1024,T,T]
        save(pfx + "v_up", v_up)

        coords = rope.create_coordinate(H=T, W=T)                     # [T*T,2]
        periods = rope.periods                                        # [16]
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]  # [T*T,2,16]
        angles = angles.flatten(1, 2).tile(2)                         # [T*T,64]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        save(pfx + "rope_periods", periods)
        save(pfx + "coords", coords)
        save(pfx + "angles", angles)
        save(pfx + "cos", cos)
        save(pfx + "sin", sin)

        # self-check: rope_apply(x, sin, cos) on enc_pooled reproduces q_rope exactly
        from einops import rearrange
        x_flat = rearrange(enc_pooled, "b (n d) h w -> b n (h w) d", n=4)
        x1, x2 = x_flat.chunk(2, dim=-1)
        rot_half = torch.cat([-x2, x1], dim=-1)
        x_roped = x_flat * cos + rot_half * sin
        x_roped = rearrange(x_roped, "b n (h w) d -> b (n d) h w", h=T, w=T)
        d_rope = (x_roped - q_rope).abs().max().item()
        print(f"  rope self-check (manual rope_apply vs model rope module) max|d|={d_rope:.3e}")
        meta["per_T"].setdefault("t128", {})["rope_self_check_max_abs_diff"] = d_rope
        assert d_rope < 1e-4

        # ---- na_probe: 5 query pixels, dilated-window clamped-neighbor derivation ----
        probe_q = [(0, 0), (127, 127), (64, 64), (1, 3), (126, 120)]
        neighbors = []
        for (qy, qx) in probe_q:
            ys = axis_window(qy, T, dilation, 9)
            xs = axis_window(qx, T, dilation, 9)
            nb = [(y, x) for y in ys for x in xs]  # 9x9=81, row-major (y outer, x inner)
            assert len(nb) == 81
            neighbors.append(nb)
        probe_q_arr = np.array(probe_q, dtype=np.float32)             # [5,2]
        probe_nb_arr = np.array(neighbors, dtype=np.float32)          # [5,81,2]
        save(pfx + "probe_queries", torch.from_numpy(probe_q_arr))
        save(pfx + "probe_neighbors", torch.from_numpy(probe_nb_arr))

        # ---- sanity: manual softmax(q.k*scale) reconstruction at the 5 probe pixels ----
        heads, d_qk, d_v = 4, 64, 256
        q_np = q_rope[0].cpu().numpy()      # [256,T,T]
        k_np = k_up[0].cpu().numpy()        # [256,T,T]
        v_np = v_up[0].cpu().numpy()        # [1024,T,T]
        out_np = out[0].cpu().numpy()       # [1024,T,T]
        max_abs_d = 0.0
        for pi, (qy, qx) in enumerate(probe_q):
            qvec = q_np[:, qy, qx].reshape(heads, d_qk)                      # [4,64]
            nb = neighbors[pi]
            kmat = np.stack([k_np[:, y, x] for (y, x) in nb], axis=0).reshape(81, heads, d_qk)  # [81,4,64]
            vmat = np.stack([v_np[:, y, x] for (y, x) in nb], axis=0).reshape(81, heads, d_v)    # [81,4,256]
            logits = np.einsum("hd,nhd->hn", qvec, kmat) * scale           # [4,81]
            logits = logits - logits.max(axis=-1, keepdims=True)
            w = np.exp(logits); w = w / w.sum(axis=-1, keepdims=True)      # [4,81]
            recon = np.einsum("hn,nhd->hd", w, vmat).reshape(1024)         # [1024]
            real = out_np[:, qy, qx]
            d = np.abs(recon - real).max()
            max_abs_d = max(max_abs_d, float(d))
            print(f"  probe[{pi}] q=({qy},{qx}) manual-vs-real max|d|={d:.3e}")
        meta["per_T"]["t128"]["probe_reconstruction_max_abs_diff"] = max_abs_d
        assert max_abs_d < 1e-2, f"probe reconstruction mismatch too large: {max_abs_d}"

    meta["per_T"].setdefault(f"t{T}", {})["downsample_triggered"] = bool(downsample)
    meta["per_T"].setdefault(f"t{T}", {})["dilation"] = dilation

meta["shapes"] = shapes

# ---------------- weights export ----------------
print("\n[NAF] exporting weights...")
from safetensors.torch import save_file
sd = {k: v.detach().to(torch.float32).cpu().contiguous() for k, v in model.state_dict().items()}
save_file(sd, f"{OUT}/naf.safetensors")
naf_cfg = {"name": "NAF", "args": {"dim": 256, "heads_attn": 4, "heads_rope": 4, "kernel_size": 9,
                                    "rope_base": 100.0, "img_layers": 2}}
json.dump(naf_cfg, open(f"{OUT}/naf.json", "w"), indent=2)
print(f"  {len(sd)} tensors -> {OUT}/naf.safetensors, config -> {OUT}/naf.json")
meta["naf_state_dict_tensor_count"] = len(sd)
meta["naf_state_dict_keys"] = sorted(sd.keys())

json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("\nDONE ->", OUT)
