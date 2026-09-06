// Debug-only allocation trace of the native sparse shape decoder (docs/PIXAL3D_WEBGPU_MEMORY.md).
// Runs the unchanged shape_decode (from_latent -> 4x (ConvNeXt stage + C2S) -> output_layer)
// with TRELLIS_DBG_ALLOC_TRACE set, so every stage graph prints its graph-allocated tensors
// (op, shape, dtype, bytes), the ten largest, the largest single allocation, a
// simultaneously-live estimate and gallocr's actual buffer size (src/graph_dump.cpp,
// trellis_graph_alloc_trace). Nothing about the decode itself changes.
//
//   trellis-shape-dec-alloc-trace <shape_dec.gguf> <coords.npy> <slat.npy> [gpu] [res]
//     coords.npy  int32 [N,4] (b,x,y,z) or [N,3] (x,y,z) -- the SS-stage active voxels
//     slat.npy    float32 [N,32] denormalized shape SLAT (row-major == ggml [32,N])
//     gpu         -1 = CPU backend, else the build's GPU device (default 0)
//     res         final voxel resolution written into ShapeOut (default 512; decode is res-free)
//   env: TRELLIS_DBG_ALLOC_TRACE_MIN_MB=<MB> (per-tensor print threshold, default 64),
//        TRELLIS_DBG_ALLOC_TRACE=all (print every allocation)
#include "trellis_model.h"
#include "shape_decoder.h"
#include "npy.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal <i4 (int32) .npy reader -- npy.h is f32-only (same as src/test_pixal3d_shape_decode.cpp).
struct I32Array {
    std::vector<int64_t> shape;
    std::vector<int32_t> data;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};
static I32Array load_npy_i32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); throw std::runtime_error("npy: bad magic " + path); }
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fclose(f); throw std::runtime_error("npy: header len " + path); }
    std::string hdr(hlen, '\0');
    if (fread(hdr.data(), 1, hlen, f) != hlen) { fclose(f); throw std::runtime_error("npy: header " + path); }
    if (hdr.find("'<i4'") == std::string::npos && hdr.find("\"<i4\"") == std::string::npos) {
        fclose(f); throw std::runtime_error("npy: only <i4 supported here: " + path + " hdr=" + hdr);
    }
    I32Array a;
    size_t p = hdr.find("'shape':"); p = hdr.find('(', p); size_t q = hdr.find(')', p);
    std::string s = hdr.substr(p + 1, q - p - 1);
    for (size_t i = 0; i < s.size();) {
        if (isdigit((unsigned char)s[i])) { int64_t v = 0; while (i < s.size() && isdigit((unsigned char)s[i])) v = v * 10 + (s[i++] - '0'); a.shape.push_back(v); }
        else ++i;
    }
    if (a.shape.empty()) a.shape.push_back(1);
    a.data.resize(a.numel());
    if ((int64_t)fread(a.data.data(), sizeof(int32_t), a.numel(), f) != a.numel()) { fclose(f); throw std::runtime_error("npy: short data " + path); }
    fclose(f);
    return a;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <shape_dec.gguf> <coords.npy> <slat.npy> [gpu] [res]\n", argv[0]);
        return 1;
    }
    const std::string gguf = argv[1], coords_path = argv[2], slat_path = argv[3];
    const int gpu = argc > 4 ? atoi(argv[4]) : 0;
    const int res = argc > 5 ? atoi(argv[5]) : 512;
    if (!getenv("TRELLIS_DBG_ALLOC_TRACE")) setenv("TRELLIS_DBG_ALLOC_TRACE", "1", 1);

    I32Array co = load_npy_i32(coords_path);
    npy::Array slat = npy::load(slat_path);
    const int64_t N = co.shape.empty() ? 0 : co.shape[0];
    if (co.shape.size() != 2 || (co.shape[1] != 3 && co.shape[1] != 4)) { fprintf(stderr, "coords must be [N,3] or [N,4]\n"); return 1; }
    if (slat.shape.size() != 2 || slat.shape[0] != N || slat.shape[1] != 32) {
        fprintf(stderr, "slat must be [N,32] with N=%lld (got [%lld,%lld])\n", (long long)N,
                (long long)(slat.shape.size() > 0 ? slat.shape[0] : 0), (long long)(slat.shape.size() > 1 ? slat.shape[1] : 0));
        return 1;
    }
    const int cw = (int)co.shape[1], c0 = cw - 3;
    std::vector<std::array<int,3>> coords0((size_t)N);
    for (int64_t i = 0; i < N; ++i)
        coords0[(size_t)i] = { co.data[i*cw + c0], co.data[i*cw + c0 + 1], co.data[i*cw + c0 + 2] };
    std::vector<float> latent(slat.data.begin(), slat.data.begin() + (size_t)32 * N);

    printf("[alloc-trace] N=%lld input voxels, gpu=%d, res=%d, trace=%s min_mb=%s\n", (long long)N, gpu, res,
           getenv("TRELLIS_DBG_ALLOC_TRACE"), getenv("TRELLIS_DBG_ALLOC_TRACE_MIN_MB") ? getenv("TRELLIS_DBG_ALLOC_TRACE_MIN_MB") : "64");
    fflush(stdout);
    trellis::Model m = trellis::Model::load(gguf, gpu);
    printf("[alloc-trace] loaded %s (%zu tensors, %.2f GB)\n", m.arch.c_str(), m.tensors.size(), m.total_bytes() / 1e9);
    fflush(stdout);
    trellis::ShapeOut so = trellis::shape_decode(m, latent, coords0, res);
    printf("[alloc-trace] decoded voxels @res%d = %zu, feats7 = %zu floats\n", so.res, so.coords.size(), so.feats7.size());
    m.free();
    return 0;
}
