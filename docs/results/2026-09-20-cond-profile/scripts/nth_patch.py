#!/usr/bin/env python3
"""診断用: ggml-metal の group_norm dispatch **だけ** を nth=32 固定 → 最大 1024 に変える / 戻す（submodule はコミットしない）。
`int nth = 32; // SIMD width` は 9 つの op（sum / sum_rows / set_rows / soft_max / l2_norm / group_norm / norm / argmax / tri）に
同じ行があるので、直前の `ggml_metal_library_get_pipeline_group_norm` を含めた一意な文脈で置換し、置換数が 1 であることを assert する。
使い方: nth_patch.py apply|revert|check"""
import sys, subprocess
p = "/Users/s25705/Downloads/pixal3d-cond-profile/thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp"
s = open(p).read()
ctx = "    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);\n\n    int nth = 32; // SIMD width\n"
new = ("    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);\n\n    int nth = 32; // SIMD width\n"
       "    // DIAGNOSTIC (pixal3d #33, group_norm only): scale the threadgroup up to the pipeline max; the kernel\n"
       "    // already reduces across SIMD groups through the 32-float smem buffer.\n"
       "    while (nth < 1024 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) nth *= 2;\n")
assert s.count(ctx) == 1, "group_norm context must be unique"  # new は ctx を先頭に含むので両状態で 1
if sys.argv[1] == "apply":
    assert ctx in s; s = s.replace(ctx, new)
elif sys.argv[1] == "revert":
    assert new in s; s = s.replace(new, ctx)
elif sys.argv[1] == "check":
    d = subprocess.run(["git", "-C", "/Users/s25705/Downloads/pixal3d-cond-profile/thirdparty/ggml", "diff", "--", "src/ggml-metal/ggml-metal-ops.cpp"], capture_output=True, text=True).stdout
    print(d); print("hunks:", d.count("@@") // 2); sys.exit(0)
open(p, "w").write(s); print(sys.argv[1], "ok")
