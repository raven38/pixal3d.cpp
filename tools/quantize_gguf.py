#!/usr/bin/env python3
"""Selectively quantize a trellis.cpp GGUF: F16 2D matmul weights -> Q8_0/Q5_0/Q4_0,
everything else (F32 norms/biases, conv/patch weights, 1D/3D/4D tensors) kept as-is.
Only 2D tensors whose contraction dim (ne0) is block-aligned are quantized, since those
are the ggml_mul_mat weights that ggml dequantizes on the fly on every backend."""
import sys, os
import numpy as np
import gguf
from gguf import GGMLQuantizationType as GT, GGUFValueType as VT

def block_size(qtype):
    return gguf.GGML_QUANT_SIZES[qtype][0]

def quantize_file(src, dst, qname):
    qtype = getattr(GT, qname)
    bs = block_size(qtype)
    r = gguf.GGUFReader(src)
    arch = r.fields['general.architecture'].contents()
    w = gguf.GGUFWriter(dst, arch)

    reserved = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count', 'general.architecture'}
    for k, f in r.fields.items():
        if k in reserved:
            continue
        vt = f.types[0]
        val = f.contents()
        if vt == VT.ARRAY:
            w.add_key_value(k, val, VT.ARRAY, sub_type=f.types[1])
        else:
            w.add_key_value(k, val, vt)

    nq = nk = 0
    q_bytes = keep_bytes = 0
    for t in r.tensors:
        d = t.data
        if t.tensor_type == GT.F16 and d.ndim == 2 and d.shape[-1] % bs == 0:
            q = gguf.quants.quantize(d.astype(np.float32), qtype)
            w.add_tensor(t.name, q, raw_dtype=qtype)
            nq += 1; q_bytes += q.nbytes
        else:
            w.add_tensor(t.name, d)          # keep original dtype (F16/F32)
            nk += 1; keep_bytes += d.nbytes

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"  {os.path.basename(src)} -> {qname}: quantized {nq} tensors, kept {nk}; "
          f"data {(q_bytes+keep_bytes)/1e9:.2f} GB (was {sum(t.data.nbytes for t in r.tensors)/1e9:.2f} GB)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: quantize_gguf.py <src.gguf> <dst.gguf> <Q8_0|Q5_0|Q4_0>"); sys.exit(1)
    quantize_file(sys.argv[1], sys.argv[2], sys.argv[3])
