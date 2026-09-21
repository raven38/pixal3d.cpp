#!/usr/bin/env python3
"""issue #55 §2.4 / U1: ggml-metal の group_norm を (1) threadgroup を group サイズに合わせて 32 → 最大 1024 に拡張、
(2) 2 回目の smem zero-fill の前に barrier を足す、の 2 ハンクで書き換える / 戻す（submodule はコミットしない）。
`int nth = 32; // SIMD width` は 9 つの dispatcher に同じ行があるので、直前の `get_pipeline_group_norm` を含む一意な文脈で置換する。
使い方: nth_patch.py apply|revert|check [ggml_dir]"""
import sys, subprocess
G = sys.argv[2] if len(sys.argv) > 2 else "/Users/s25705/Downloads/pixal3d-metal-groupnorm/thirdparty/ggml"
OPS = G + "/src/ggml-metal/ggml-metal-ops.cpp"
KER = G + "/src/ggml-metal/ggml-metal.metal"
ops_old = ("    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);\n\n"
           "    int nth = 32; // SIMD width\n"
           "    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {\n"
           "    //    nth *= 2;\n"
           "    //}\n\n"
           "    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));\n"
           "    //nth = std::min(nth, ne00/4);\n")
ops_new = ("    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);\n\n"
           "    // one threadgroup per group: size it to the group (ne00*ne01*ceil(ne02/ngrp) elements), not\n"
           "    // to ne00 -- the kernel reduces across SIMD groups through a 32-float threadgroup buffer,\n"
           "    // so up to 32 SIMD groups (1024 threads) per group\n"
           "    const int64_t gs = (int64_t) ne00*ne01*((ne02 + ngrp - 1)/ngrp);\n\n"
           "    int nth = 32; // SIMD width\n"
           "    while (nth < gs/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {\n"
           "        nth *= 2;\n"
           "    }\n"
           "    nth = std::min(nth, 1024);\n")
ker_old = ("    const float mean = tmp / gs;\n"
           "    tmp = 0.0f;\n\n"
           "    for (int j = start; j < end; j += ntg) {\n"
           "        float xi = src0[j] - mean;\n"
           "        dst[j] = xi;\n"
           "        tmp += xi * xi;\n"
           "    }\n\n"
           "    tmp = simd_sum(tmp);\n"
           "    if (ntg > N_SIMDWIDTH) {\n"
           "        if (sgitg == 0) {\n"
           "            buf[tiisg] = 0.0f;\n"
           "        }\n")
ker_new = ("    const float mean = tmp / gs;\n"
           "    tmp = 0.0f;\n\n"
           "    for (int j = start; j < end; j += ntg) {\n"
           "        float xi = src0[j] - mean;\n"
           "        dst[j] = xi;\n"
           "        tmp += xi * xi;\n"
           "    }\n\n"
           "    tmp = simd_sum(tmp);\n"
           "    if (ntg > N_SIMDWIDTH) {\n"
           "        // every SIMD group has to finish reading buf[] from the mean reduction before it is re-zeroed\n"
           "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
           "        if (sgitg == 0) {\n"
           "            buf[tiisg] = 0.0f;\n"
           "        }\n")
def rw(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, f"{path}: context not unique/found ({s.count(old)})"
    open(path, "w").write(s.replace(old, new))
cmd = sys.argv[1]
if cmd == "apply":
    rw(OPS, ops_old, ops_new); rw(KER, ker_old, ker_new); print("apply ok")
elif cmd == "revert":
    rw(OPS, ops_new, ops_old); rw(KER, ker_new, ker_old); print("revert ok")
elif cmd == "check":
    d = subprocess.run(["git", "-C", G, "diff", "--", "src/ggml-metal/ggml-metal-ops.cpp", "src/ggml-metal/ggml-metal.metal"], capture_output=True, text=True).stdout
    print(d); print("hunks:", d.count("\n@@ "))
