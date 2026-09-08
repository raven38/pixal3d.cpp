#!/usr/bin/env python3
"""Convert TRELLIS.2 (and helper) safetensors checkpoints to GGUF.

Run with the project's uv venv:
    /media/ilintar/D_SSD/trellis2-venv/bin/python tools/convert.py [component ...]

Design:
  * safetensors is parsed by hand (the numpy backend can't read bf16), so we
    control the bf16 -> f32 -> f16 path exactly (bf16 = high 16 bits of f32).
  * Quantization policy: weight matrices / convs (ndim >= 2) -> f16; all 1-D
    params (norms, biases, gammas, the per-block `modulation` vectors) -> f32.
  * Tensor names are preserved verbatim (all core models are <= 37 chars,
    under GGML_MAX_NAME=64). The model config JSON is embedded as metadata.
"""
import json, struct, sys, os
import numpy as np
import gguf

MODELS = os.environ.get("TRELLIS_MODELS", "/media/ilintar/D_SSD/models/trellis2")
OUT = os.environ.get("TRELLIS_GGUF_OUT", f"{MODELS}/gguf")
# ss_dec and dinov3 live under directory layouts that don't line up with MODELS on
# every host (e.g. the Pixal3D snapshot has no tilarge/ subfolder, and the DINOv3
# timm checkpoint lives in the HF cache) -> dedicated overrides, same defaults as before.
SS_DEC_CKPT = os.environ.get("TRELLIS_SS_DEC_CKPT", f"{MODELS}/tilarge/ckpts/ss_dec_conv3d_16l8_fp16")
DINOV3_CKPT = os.environ.get("TRELLIS_DINOV3_CKPT", f"{MODELS}/dinov3")
NAF_CKPT = os.environ.get("TRELLIS_NAF_CKPT", f"{MODELS}/naf/naf")

# component -> (safetensors path, config json path or None, gguf arch tag)
MANIFEST = {
    "ss_flow":        (f"{MODELS}/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
                       f"{MODELS}/ckpts/ss_flow_img_dit_1_3B_64_bf16.json",        "trellis2-ss-flow"),
    "shape_flow_512": (f"{MODELS}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
                       f"{MODELS}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json", "trellis2-slat-flow"),
    "tex_flow_512":   (f"{MODELS}/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
                       f"{MODELS}/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json", "trellis2-slat-flow"),
    "shape_flow_1024":(f"{MODELS}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
                       f"{MODELS}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json", "trellis2-slat-flow"),
    "tex_flow_1024":  (f"{MODELS}/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
                       f"{MODELS}/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json", "trellis2-slat-flow"),
    "shape_dec":      (f"{MODELS}/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
                       f"{MODELS}/ckpts/shape_dec_next_dc_f16c32_fp16.json",       "trellis2-shape-dec"),
    "tex_dec":        (f"{MODELS}/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
                       f"{MODELS}/ckpts/tex_dec_next_dc_f16c32_fp16.json",         "trellis2-tex-dec"),
    "ss_dec":         (f"{SS_DEC_CKPT}.safetensors",
                       f"{SS_DEC_CKPT}.json",                                      "trellis2-ss-dec"),
    "dinov3":         (f"{DINOV3_CKPT}/model.safetensors",
                       f"{DINOV3_CKPT}/config.json",                               "dinov3-vitl16"),
    "birefnet":       (f"{MODELS}/birefnet/model.safetensors",
                       f"{MODELS}/birefnet/config.json",                           "birefnet-swinl"),
    # Pixal3D multiview flow DiTs (reuse TRELLIS.2's ckpt dir layout under MODELS/ckpts).
    "pixal3d_ss_flow_mv":         (f"{MODELS}/ckpts/ss_flow_img_dit_1_3B_64_bf16_mv.safetensors",
                       f"{MODELS}/ckpts/ss_flow_img_dit_1_3B_64_bf16_mv.json",     "pixal3d-ss-flow"),
    "pixal3d_shape_flow_512_mv":  (f"{MODELS}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv.safetensors",
                       f"{MODELS}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv.json", "pixal3d-slat-flow"),
    "pixal3d_shape_flow_1024_mv": (f"{MODELS}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv.safetensors",
                       f"{MODELS}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv.json", "pixal3d-slat-flow"),
    "pixal3d_tex_flow_1024_mv":   (f"{MODELS}/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv.safetensors",
                       f"{MODELS}/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv.json", "pixal3d-slat-flow"),
    "pixal3d_naf": (f"{NAF_CKPT}.safetensors", f"{NAF_CKPT}.json", "pixal3d-naf"),
}



# 学習済みの「トークンそのもの」は f16 に落とさない。cls_token / reg_token /
# pos_embed 類は行列積の重みではなく残差ストリームに直接足される値なので、丸め誤差が
# そのままトークンの誤差になる。実測（2026-09-09, DINOv3 ViT-L）: f16 に落とすと
# 埋め込みが PyTorch 参照に対して L2 相対 2.128e-04 ずれ、それが 24 ブロックを素通りして
# 最終トークンの 6.2e-04 になり、conditioning の z_global の残差の主因になっていた。
# f32 に戻しても cls 1024 + reg 4096 要素で 10 KB しか増えない。
EMBED_TOKEN_SUFFIXES = ("cls_token", "reg_token", "mask_token", "pos_embed",
                        "position_embeddings", "storage_tokens")

def keep_f32(name):
    return name.endswith(EMBED_TOKEN_SUFFIXES) or ".cls_token" in name or ".reg_token" in name


def read_safetensors(path):
    """Yield (name, numpy_f32_or_f16_array) preserving natural (torch) shape."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        items = [(k, v) for k, v in hdr.items() if k != "__metadata__"]
        for name, v in items:
            dt = v["dtype"]; shape = v["shape"]
            o0, o1 = v["data_offsets"]
            fh.seek(base + o0)
            buf = fh.read(o1 - o0)
            if dt == "BF16":
                u16 = np.frombuffer(buf, dtype="<u2")
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(buf, dtype="<f2").astype(np.float32)
            elif dt == "F32":
                arr = np.frombuffer(buf, dtype="<f4")
            elif dt == "F64":
                arr = np.frombuffer(buf, dtype="<f8").astype(np.float32)
            elif dt == "I64":
                arr = np.frombuffer(buf, dtype="<i8").astype(np.int64)
            elif dt == "I32":
                arr = np.frombuffer(buf, dtype="<i4").astype(np.int32)
            elif dt == "C64":
                # complex RoPE phase buffers (Pixal3D stores rope_phases persistently);
                # the C++ side recomputes RoPE tables on the host, so skip them.
                print(f"  skipping {name} (dtype {dt}, shape {shape})")
                continue
            else:
                raise ValueError(f"unhandled dtype {dt} for {name}")
            arr = arr.reshape(shape) if shape else arr.reshape(())
            yield name, arr


def convert_birefnet(w, src):
    """BiRefNet: fold BatchNorm (running_mean/var + weight/bias) into per-channel scale/shift,
    store relative_position_index as int32, keep relative_position_bias_table in f32, everything
    else f16 (>=2D) / f32 (1D). eps=1e-5."""
    EPS = 1e-5
    f16ok = os.environ.get("FORCE_F32") != "1"   # f16 matmul (cublasLt HHH) is broken on sm_120 -> allow F32
    T = {name: arr for name, arr in read_safetensors(src)}
    # find BatchNorm prefixes (have both running_mean and running_var)
    bn_pref = set()
    for k in T:
        if k.endswith(".running_mean") and (k[:-len(".running_mean")] + ".running_var") in T:
            bn_pref.add(k[:-len(".running_mean")])
    skip = set()
    for p in bn_pref:
        skip.update([p+".running_mean", p+".running_var", p+".weight", p+".bias", p+".num_batches_tracked"])
    n_f16 = n_f32 = total = 0
    # emit folded BN scale/shift
    for p in sorted(bn_pref):
        g = T[p+".weight"].astype(np.float32); b = T[p+".bias"].astype(np.float32)
        mu = T[p+".running_mean"].astype(np.float32); var = T[p+".running_var"].astype(np.float32)
        scale = g / np.sqrt(var + EPS); shift = b - mu * scale
        w.add_tensor(p+".scale", np.ascontiguousarray(scale.astype(np.float32)))
        w.add_tensor(p+".shift", np.ascontiguousarray(shift.astype(np.float32)))
        n_f32 += 2; total += 2
    # emit the rest
    for name, arr in T.items():
        if name in skip or name.endswith(".num_batches_tracked"):
            continue
        if name.endswith("relative_position_index") or name.endswith("attn_mask") or "relative_position_index" in name:
            data = np.ascontiguousarray(arr.astype(np.int32)); n_f32 += 1
        elif name.endswith("relative_position_bias_table"):
            data = np.ascontiguousarray(arr.astype(np.float32)); n_f32 += 1   # keep precision for the bias gather
        elif keep_f32(name):
            data = np.ascontiguousarray(arr.astype(np.float32)); n_f32 += 1   # 埋め込みトークンは丸めない
        elif arr.ndim >= 2 and f16ok:
            data = np.ascontiguousarray(arr.astype(np.float16)); n_f16 += 1
        else:
            data = np.ascontiguousarray(arr.astype(np.float32)); n_f32 += 1
        w.add_tensor(name, data); total += 1
    return n_f16, n_f32, total


def convert(component):
    src, cfg, arch = MANIFEST[component]
    os.makedirs(OUT, exist_ok=True)
    dst = f"{OUT}/{component}.gguf"
    w = gguf.GGUFWriter(dst, arch)
    if cfg and os.path.exists(cfg):
        with open(cfg) as f:
            w.add_string("trellis.config_json", f.read())
    w.add_string("general.name", component)
    n_f16 = n_f32 = 0
    total = 0
    force_f32 = os.environ.get("FORCE_F32") == "1"
    sparse_conv = component in ("shape_dec", "tex_dec", "shape_enc")

    if component == "birefnet":
        n_f16, n_f32, total = convert_birefnet(w, src)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        sz = os.path.getsize(dst)
        print(f"  {component:14s} -> {os.path.basename(dst):22s} {total:4d} tensors (f16={n_f16}, f32={n_f32})  {sz/1e9:.2f} GB")
        return

    for name, arr in read_safetensors(src):
        # ggml is max 4-D. Two distinct 5-D conv layouts:
        if arr.ndim == 5:
            if sparse_conv:
                # sparse submanifold conv weight [Co,Kd,Kh,Kw,Ci] -> [Co,27,Ci]
                # (== ggml ne [Ci,27,Co]; 27 taps t = kd*9+kh*3+kw)
                co = arr.shape[0]; ci = arr.shape[4]
                arr = arr.reshape(co, 27, ci)
            else:
                # dense Conv3d weight [OC,IC,KD,KH,KW] -> [OC*IC,KD,KH,KW] (ggml_conv_3d)
                oc, ic = arr.shape[0], arr.shape[1]
                arr = arr.reshape(oc * ic, arr.shape[2], arr.shape[3], arr.shape[4])
        if arr.ndim >= 2 and not force_f32 and not keep_f32(name):
            data = np.ascontiguousarray(arr.astype(np.float16)); n_f16 += 1
        else:
            data = np.ascontiguousarray(arr.astype(np.float32)); n_f32 += 1
        w.add_tensor(name, data)
        total += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    sz = os.path.getsize(dst)
    print(f"  {component:14s} -> {os.path.basename(dst):22s} {total:4d} tensors "
          f"(f16={n_f16}, f32={n_f32})  {sz/1e9:.2f} GB")


if __name__ == "__main__":
    comps = sys.argv[1:] or list(MANIFEST.keys())
    print(f"converting {len(comps)} component(s) -> {OUT}")
    for c in comps:
        convert(c)
    print("done")
