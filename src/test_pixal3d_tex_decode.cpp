// REAL tex-SLAT -> 6ch PBR parity test for the texture (PBR) decoder (trellis-cli's
// "[5/6] texture SLAT flow + PBR decode" stage). Mirrors src/test_pixal3d_shape_decode.cpp's
// style/harness but for tex_decode() instead of shape_decode()+dual_grid_to_mesh, and drives
// shape_decode() first (at res 1024, from the SAME hr_coords + a real denormalized HR shape
// SLAT) purely to obtain the `subs` (per-C2S binarized subdivision masks) that tex_decode
// needs as guide_subs -- exactly how trellis_cli.cpp wires shape decode -> tex decode.
//
// Fixture (produced by tools/ref_pixal3d_hr_sample.py, see its header comment):
//   hr_coords.npy        [Nh,4]  int32   (b,x,y,z) at grid-64 -- HR flow's sparse-token coords
//   f32_shape_slat.npy    [Nh,32] float32 denormalized HR shape SLAT (f32-torso calibration run)
//   f32_tex_slat.npy      [Nh,32] float32 denormalized HR texture SLAT (same run)
//   f32_tex_attrs.npy     [Nt,6]  float32 reference tex_decoder output, ALREADY *0.5+0.5'd
//                                 (SparseUnetVaeDecoder(tex_slat, guide_subs=subs).feats*0.5+0.5)
//   f32_tex_coords.npy    [Nt,3]  int32   voxel coords (batch column already dropped) for the
//                                 above, at the final decoded resolution (1024)
//
// tex_decode()'s return is channel-major [6*M], PRE *0.5+0.5 (see shape_decoder.h) -- this test
// applies *0.5+0.5 before comparing, to match f32_tex_attrs.npy's convention.
//
// Channel layout (docs/spec/11-tex_dec.md): [0:3]=base_color RGB, [3]=metallic, [4]=roughness,
// [5]=alpha.
//
//   trellis-test-pixal3d-tex-decode <shape_dec.gguf> <tex_dec.gguf> <fixture_dir> [gpu]
#include "trellis_model.h"
#include "shape_decoder.h"
#include "npy.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using std::array; using std::string; using std::vector;

namespace trellis { extern bool g_sparse_cast_f32; }

// Minimal <i4 (int32) .npy reader -- npy.h (shared, not to be touched here) is f32-only.
// Mirrors the loader in src/test_pixal3d_shape_decode.cpp / src/test_pixal3d_slat_sample.cpp.
struct I32Array {
    std::vector<int64_t> shape;
    std::vector<int32_t> data;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};
static I32Array load_npy_i32(const string& path) {
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

static uint64_t coord_key(int x, int y, int z) {
    return ((uint64_t)(uint32_t)(x + (1 << 20)) << 42) | ((uint64_t)(uint32_t)(y + (1 << 20)) << 21) | (uint32_t)(z + (1 << 20));
}

struct ChanStats { double max_abs = 0, mean_abs = 0, ref_mean = 0, ref_var = 0; };

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <shape_dec.gguf> <tex_dec.gguf> <fixture_dir> [gpu]\n", argv[0]);
        return 1;
    }
    const string shape_gguf = argv[1], tex_gguf = argv[2], fdir = argv[3];
    const int gpu = argc > 4 ? atoi(argv[4]) : 0;
    printf("fixture_dir=%s gpu=%d\n", fdir.c_str(), gpu);

    // production default: fp16 torso (no --f32, no TRELLIS_F32W) -- both shape_dec and tex_dec
    // checkpoints declare use_fp16=True and the fixture leaves the decoders at that default.
    if (getenv("TRELLIS_F32W")) { trellis::g_sparse_cast_f32 = true; printf("(TRELLIS_F32W: f32 weight compute -- NOT production default)\n"); }

    I32Array co = load_npy_i32(fdir + "/hr_coords.npy");                // [Nh,4] (b,x,y,z)
    npy::Array shape_slat = npy::load(fdir + "/f32_shape_slat.npy");    // [Nh,32] denormalized
    npy::Array tex_slat   = npy::load(fdir + "/f32_tex_slat.npy");      // [Nh,32] denormalized
    npy::Array ref_attrs  = npy::load(fdir + "/f32_tex_attrs.npy");     // [Nt,6] already *0.5+0.5
    I32Array   ref_coords = load_npy_i32(fdir + "/f32_tex_coords.npy"); // [Nt,3]

    const int64_t Nh = co.shape.empty() ? 0 : co.shape[0];
    if (shape_slat.shape[0] != Nh || tex_slat.shape[0] != Nh) {
        fprintf(stderr, "slat rows (shape=%lld tex=%lld) != hr_coords Nh=%lld\n",
                (long long)shape_slat.shape[0], (long long)tex_slat.shape[0], (long long)Nh);
        return 1;
    }
    const int cw = (int)co.shape[1], c0 = cw - 3;
    vector<array<int,3>> coords0(Nh);
    for (int64_t i = 0; i < Nh; ++i)
        coords0[i] = { co.data[i*cw + c0], co.data[i*cw + c0 + 1], co.data[i*cw + c0 + 2] };
    // [N,32] row-major == ggml [32,N] channel-major (ne0=32): direct flat copy, no transpose.
    vector<float> shape_latent(shape_slat.data.begin(), shape_slat.data.begin() + (size_t)32 * Nh);
    vector<float> tex_latent(tex_slat.data.begin(), tex_slat.data.begin() + (size_t)32 * Nh);

    printf("input: Nh=%lld HR sparse-token voxels @grid-64\n", (long long)Nh);

    // ---- Stage A: shape_decode @res1024 -- only to obtain so.subs (guide_subs for tex_decode)
    // and so.coords (which must equal the tex decoder's own upsampled coords, since both share
    // the identical subdivision masks starting from the identical input coords0). ----
    trellis::ShapeOut so;
    {
        trellis::Model m = trellis::Model::load(shape_gguf, gpu);
        printf("loaded shape_dec: %s (%zu tensors)\n", m.arch.c_str(), m.tensors.size());
        so = trellis::shape_decode(m, shape_latent, coords0, /*resolution=*/1024);
        m.free();
    }
    printf("shape_decode: subs stages=%zu, final coords M=%d @res%d\n", so.subs.size(), (int)so.coords.size(), so.res);

    // ---- Stage B: tex_decode driven by so.subs, at the SAME coords0 input. ----
    vector<float> pbr;   // [6*M] channel-major, PRE *0.5+0.5
    {
        trellis::Model m = trellis::Model::load(tex_gguf, gpu);
        printf("loaded tex_dec: %s (%zu tensors)\n", m.arch.c_str(), m.tensors.size());
        pbr = trellis::tex_decode(m, tex_latent, coords0, so.subs);
        m.free();
    }
    const int64_t M = (int64_t)so.coords.size();
    if ((int64_t)pbr.size() != 6 * M) {
        fprintf(stderr, "tex_decode returned %zu floats, expected 6*M=%lld\n", pbr.size(), (long long)(6*M));
        return 1;
    }
    vector<float> mine(6 * M);
    for (int64_t i = 0; i < 6 * M; ++i) mine[i] = pbr[i] * 0.5f + 0.5f;   // match reference scale

    printf("\n=== voxel-count comparison ===\n");
    printf("  mine (shape_decode/tex_decode M): %lld\n", (long long)M);
    printf("  reference (f32_tex_attrs Nt):     %lld\n", (long long)ref_attrs.shape[0]);

    // ---- match by integer voxel coords (order between the two decoders need not agree) ----
    std::unordered_map<uint64_t, int64_t> mine_by_coord;
    mine_by_coord.reserve((size_t)M * 2);
    for (int64_t i = 0; i < M; ++i) {
        const auto& c = so.coords[i];
        mine_by_coord[coord_key(c[0], c[1], c[2])] = i;
    }
    const int64_t Nt = ref_attrs.shape[0];
    int64_t matched = 0;
    ChanStats ch[6];
    vector<double> ref_sum(6, 0.0), ref_sumsq(6, 0.0);
    for (int64_t t = 0; t < Nt; ++t) {
        const int x = ref_coords.data[3*t], y = ref_coords.data[3*t+1], z = ref_coords.data[3*t+2];
        for (int k = 0; k < 6; ++k) { double v = ref_attrs.data[6*t+k]; ref_sum[k] += v; ref_sumsq[k] += v*v; }
        auto it = mine_by_coord.find(coord_key(x, y, z));
        if (it == mine_by_coord.end()) continue;
        ++matched;
        const int64_t mi = it->second;
        for (int k = 0; k < 6; ++k) {
            const double d = std::fabs((double)mine[6*mi + k] - (double)ref_attrs.data[6*t + k]);
            ch[k].max_abs = std::max(ch[k].max_abs, d);
            ch[k].mean_abs += d;
        }
    }
    for (int k = 0; k < 6; ++k) {
        if (matched) ch[k].mean_abs /= matched;
        ch[k].ref_mean = ref_sum[k] / std::max<int64_t>(Nt, 1);
        ch[k].ref_var = ref_sumsq[k] / std::max<int64_t>(Nt, 1) - ch[k].ref_mean * ch[k].ref_mean;
    }
    const double iou = (matched > 0) ? (double)matched / (double)(M + Nt - matched) : 0.0;

    printf("\n=== coordinate-set match ===\n");
    printf("  matched voxels: %lld / mine=%lld ref=%lld   IoU=%.4f\n",
           (long long)matched, (long long)M, (long long)Nt, iou);

    static const char* names[6] = {"base_R", "base_G", "base_B", "metallic", "roughness", "alpha"};
    printf("\n=== per-channel diff on matched voxels (mine vs reference, both in [0,1] post *0.5+0.5) ===\n");
    printf("  %-10s %10s %10s %12s %12s\n", "channel", "max|d|", "mean|d|", "ref_mean", "ref_var");
    for (int k = 0; k < 6; ++k)
        printf("  %-10s %10.4f %10.4f %12.4f %12.6f\n", names[k], ch[k].max_abs, ch[k].mean_abs, ch[k].ref_mean, ch[k].ref_var);

    printf("\n=== verdict ===\n");
    const bool coord_ok = iou > 0.90;
    // Parity is judged per channel against the reference decode of the same SLAT. A channel that
    // is flat in BOTH (e.g. metallic/roughness on assets without metals) is a match, not a bug.
    bool attrs_ok = true;
    for (int k = 0; k < 6; ++k) attrs_ok = attrs_ok && ch[k].max_abs < 0.15 && ch[k].mean_abs < 5e-3;
    double mine_var[6] = {0,0,0,0,0,0};
    if (matched > 0) {
        vector<double> msum(6,0.0), msumsq(6,0.0);
        for (int64_t t = 0; t < Nt; ++t) {
            const int x = ref_coords.data[3*t], y = ref_coords.data[3*t+1], z = ref_coords.data[3*t+2];
            auto it = mine_by_coord.find(coord_key(x, y, z));
            if (it == mine_by_coord.end()) continue;
            for (int k = 0; k < 6; ++k) { double v = mine[6*it->second+k]; msum[k]+=v; msumsq[k]+=v*v; }
        }
        for (int k = 0; k < 6; ++k) { double mm = msum[k]/matched; mine_var[k] = msumsq[k]/matched - mm*mm; }
    }
    printf("per-channel variance mine vs ref:");
    for (int k = 0; k < 6; ++k) printf("  %s %.2e/%.2e", names[k], mine_var[k], ch[k].ref_var);
    printf("\ncoord_ok(IoU>0.90)=%s  attrs_ok(max|d|<0.15, mean|d|<5e-3 all channels)=%s\n",
           coord_ok ? "yes" : "no", attrs_ok ? "yes" : "no");
    printf(coord_ok && attrs_ok ? "PASS (tex_decode reproduces the reference per-voxel PBR attributes)\n"
                                : "FAIL\n");
    return coord_ok && attrs_ok ? 0 : 1;
}
