// End-to-end shape-SLAT (res-512 stage) sampling parity test for the Pixal3D MV shape flow: runs
// the full 12-step FlowEuler/CFG sampler (sparse ProjectAttention DiT, cond+proj) from fixture
// noise over the fixture's active coords, then denormalizes with the shape SLAT mean/std and
// compares both the normalized per-step latents and the denormalized SLAT against f32 and bf16
// PyTorch references, using the ref-f32-vs-bf16 delta as the calibration baseline. See
// docs/spec/30-pixal3d-cond.md §5/6: block-level goldens are only meaningful on the CPU backend;
// the production parity gate is end-to-end sampling in the reference's own dtype (bf16). Mirrors
// test_pixal3d_ss_sample.cpp (dense SS version) but drives the SPARSE runner (test_pixal3d_slat_flow.cpp
// coords/layout conventions: coords [N,4] int32 (b,x,y,z), x/proj [N,C] row-major == ggml [C,N]
// channel-major, same bytes -- no dense-grid remap needed here) and the shape_flow() sampler params
// from trellis_cli.cpp (guidance_rescale=0.5, gi0=0.6, gi1=1.0, rescale_t=3.0).
// Fixture layout mirrors tools/ref_pixal3d_slat_sample.py (<fixture_dir>):
//   coords.npy [N,4] int32 (b,x,y,z); noise.npy [N,32]; cond_global.npy [1,5,1024];
//   cond_proj.npy [N,2048]; t_seq.npy; guidance_strength.npy; norm_mean.npy/norm_std.npy [32];
//   f32_/bf16_ prefixed x_step0..12.npy [N,32] (step0 = initial noise, step1..12 = post-step),
//   x_final.npy [N,32], slat.npy [N,32] (denormalized).
//   trellis-test-pixal3d-slat-sample <pixal3d_shape_flow_512_mv.gguf> <fixture_dir> [gpu]
#include "trellis_model.h"
#include "flow_runner.h"
#include "dit.h"
#include "trellis_args.h"
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
#include <vector>

using std::string; using std::vector;

// shape SLAT mean/std (trellis_cli.cpp SHAPE_MEAN/SHAPE_STD) used to denormalize the sampled
// latent for the final SLAT comparison; kept in sync by hand (see docs/spec note there).
static const float SHAPE_MEAN[32]={0.781296f,0.018091f,-0.495192f,-0.558457f,1.060530f,0.093252f,1.518149f,-0.933218f,-0.732996f,2.604095f,-0.118341f,-2.143904f,0.495076f,-2.179512f,-2.130751f,-0.996944f,0.261421f,-2.217463f,1.260067f,-0.150213f,3.790713f,1.481266f,-1.046058f,-1.523667f,-0.059621f,2.220780f,1.621212f,0.877230f,0.567247f,-3.175944f,-3.186688f,1.578665f};
static const float SHAPE_STD[32]={5.972266f,4.706852f,5.445010f,5.209927f,5.320220f,4.547237f,5.020802f,5.444004f,5.226681f,5.683095f,4.831436f,5.286469f,5.652043f,5.367606f,5.525084f,4.730578f,4.805265f,5.124013f,5.530808f,5.619001f,5.103930f,5.417670f,5.269677f,5.547194f,5.634698f,5.235274f,6.110351f,5.511298f,6.237273f,4.879207f,5.347008f,5.405691f};

// Loads a fixture .npy if present; returns false (leaving `out` untouched) if missing, so
// callers can skip gracefully.
static bool load_opt(const string& path, npy::Array& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    out = npy::load(path);
    return true;
}

// Minimal <i4 (int32) .npy reader -- coords.npy is saved via save_int() (matches
// test_pixal3d_ss_sample.cpp / test_pixal3d_slat_flow.cpp conventions); npy::load() is f32-only
// and npy.h is not to be touched by this test.
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

struct Stats { double maxabs, meanabs, rel; };
// mine (vector) vs ref (npy::Array): same byte layout for all tensors in this test (see header
// comment -- [N,C] row-major == ggml [C,N] channel-major, so no reindexing needed).
static Stats diff_stats(const vector<float>& mine, const npy::Array& ref) {
    double maxabs = 0, sumabs = 0, refmax = 0;
    size_t n = std::min(mine.size(), (size_t)ref.numel());
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)mine[i] - ref.data[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)ref.data[i]));
    }
    return { maxabs, sumabs / (n ? n : 1), refmax > 0 ? maxabs / refmax : maxabs };
}
// f32 ref vs bf16 ref (both npy::Array), refmax normalized against `a` (treated as ground truth
// -- i.e. this is "rel(a, b)" per the PASS-criterion wording: how far b drifts from a).
static Stats diff_stats2(const npy::Array& a, const npy::Array& b) {
    double maxabs = 0, sumabs = 0, refmax = 0;
    size_t n = std::min((size_t)a.numel(), (size_t)b.numel());
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a.data[i] - b.data[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)a.data[i]));
    }
    return { maxabs, sumabs / (n ? n : 1), refmax > 0 ? maxabs / refmax : maxabs };
}
static void print_stats(const char* tag, const Stats& s) {
    printf("    %-4s max|d|=%.4e mean|d|=%.4e rel=%.4e\n", tag, s.maxabs, s.meanabs, s.rel);
}

// token-major [1,Ntok,D] npy -> ggml channel-major [D,Ntok] (global cond only; x/proj tensors
// here are already [N,C] == ggml [C,N] by byte layout, see header comment).
static vector<float> to_channel_major(const npy::Array& a, int64_t Ntok, int64_t D) {
    vector<float> out((size_t)D * Ntok);
    for (int64_t tok = 0; tok < Ntok; ++tok) for (int64_t d = 0; d < D; ++d) out[d + D * tok] = a.data[tok * D + d];
    return out;
}

// Mean per-token cosine similarity between mine [C,N] (ggml channel-major) and ref [N,C]
// (row-major, same bytes) -- both indexed at c + C*n.
static double mean_cosine(const vector<float>& mine, const npy::Array& ref, int C, int64_t N) {
    double sum = 0; int64_t cnt = 0;
    size_t lim = std::min(mine.size(), (size_t)ref.numel());
    for (int64_t n = 0; n < N; ++n) {
        if ((size_t)(C * (n + 1)) > lim) break;
        double dot = 0, ma = 0, mb = 0;
        for (int c = 0; c < C; ++c) {
            double a = mine[c + C * n], b = ref.data[c + C * n];
            dot += a * b; ma += a * a; mb += b * b;
        }
        if (ma > 0 && mb > 0) { sum += dot / (std::sqrt(ma) * std::sqrt(mb)); ++cnt; }
    }
    return cnt ? sum / cnt : 0.0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <pixal3d_shape_flow_512_mv.gguf> <fixture_dir> [gpu]\n", argv[0]);
        return 1;
    }
    const string gguf_flow = argv[1], fdir = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;
    printf("fixture_dir=%s gpu=%d\n", fdir.c_str(), gpu);

    if (getenv("TRELLIS_NOFA")) { trellis::g_no_fa = true; printf("(TRELLIS_NOFA: soft_max path)\n"); }

    // ---- fixture inputs ----
    I32Array co = load_npy_i32(fdir + "/coords.npy");            // [N,4] (b,x,y,z)
    npy::Array noise = npy::load(fdir + "/noise.npy");           // [N,32]
    npy::Array zg = npy::load(fdir + "/cond_global.npy");        // [1,5,1024]
    npy::Array zp = npy::load(fdir + "/cond_proj.npy");          // [N,2048]
    npy::Array nmean = npy::load(fdir + "/norm_mean.npy");       // [32]
    npy::Array nstd  = npy::load(fdir + "/norm_std.npy");        // [32]
    npy::Array tseq, gstr;
    bool has_tseq = load_opt(fdir + "/t_seq.npy", tseq);
    bool has_gstr = load_opt(fdir + "/guidance_strength.npy", gstr);

    const int64_t N = co.shape.empty() ? 0 : co.shape[0];
    const int64_t Cin = noise.shape[1];
    const int64_t Lc = zg.shape[1], Dc = zg.shape[2], Dp = zp.shape[1];
    printf("fixture: N=%lld Cin=%lld Lc=%lld Dc=%lld Dp=%lld", (long long)N, (long long)Cin, (long long)Lc, (long long)Dc, (long long)Dp);
    if (has_gstr && gstr.numel() > 0) printf(" guidance_strength(fixture)=%.3f", gstr.data[0]);
    printf("\n");
    if (has_tseq) printf("fixture t_seq: %lld entries (informational; sampler recomputes its own schedule from SamplerParams)\n", (long long)tseq.numel());
    if (zp.shape[0] != N) { fprintf(stderr, "cond_proj tokens=%lld != N=%lld\n", (long long)zp.shape[0], (long long)N); return 1; }
    if (noise.shape[0] != N) { fprintf(stderr, "noise tokens=%lld != N=%lld\n", (long long)noise.shape[0], (long long)N); return 1; }

    // coords [N,4] (b,x,y,z) -> sparse-runner coords [N][3] (x,y,z)
    vector<std::array<int,3>> coords3(N);
    const int cw = (int)co.shape[1], c0 = cw - 3;
    for (int64_t i = 0; i < N; ++i)
        coords3[i] = { co.data[i * cw + c0], co.data[i * cw + c0 + 1], co.data[i * cw + c0 + 2] };

    // x/proj [N,C] row-major == ggml [C,N] channel-major: same bytes, direct copy (see header).
    vector<float> sample(noise.data.begin(), noise.data.begin() + (size_t)Cin * N);
    vector<float> cond = to_channel_major(zg, Lc, Dc);
    vector<float> proj(zp.data.begin(), zp.data.begin() + (size_t)Dp * N);
    vector<float> neg_cond(cond.size(), 0.0f);
    vector<float> neg_proj(proj.size(), 0.0f);   // proj_linear(0) = bias

    // ---- sparse SLAT flow sampler ----
    trellis::Model mf = trellis::Model::load(gguf_flow, gpu);
    printf("loaded %s (%zu tensors)\n", mf.arch.c_str(), mf.tensors.size());
    trellis::DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(TRELLIS_F32W: f32 weight compute)\n"); }
    bool detected = trellis::dit_detect_proj_attn(mf, p);
    printf("proj_attn detected=%d d_proj=%d (fixture Dp=%lld)\n", detected, p.d_proj, (long long)Dp);
    if (!detected) { fprintf(stderr, "expected a Pixal3D checkpoint (blocks.0.cross_attn.proj_linear.weight missing)\n"); return 1; }
    if (p.d_proj != (int)Dp) { fprintf(stderr, "d_proj mismatch: checkpoint=%d fixture=%lld\n", p.d_proj, (long long)Dp); return 1; }

    trellis::DitRunner* run = trellis::make_sparse_runner(mf, p, coords3, (int)Lc);
    trellis::FlowFwdProj fwd = [&](const vector<float>& x, float ts, const float* c, const float* pj) {
        return run->forward(x, ts, c, pj);
    };

    trellis::SamplerParams sp;   // shape_flow() params, trellis_cli.cpp (res-512 / non-cascade branch)
    sp.steps = 12; sp.guidance_strength = 7.5f; sp.guidance_rescale = 0.5f;
    sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 3.0f; sp.sigma_min = 1e-5f;

    vector<vector<float>> trace;
    vector<float> out = trellis::sample_flow(fwd, sample, cond.data(), neg_cond.data(),
                                              proj.data(), neg_proj.data(), sp, &trace);
    delete run; mf.free();

    // ---- per-step latent parity vs f32 and bf16 refs, plus f32-vs-bf16 calibration ----
    printf("\nper-step latent parity ([N=%lld,C=%lld], ggml channel-major == npy row-major, no remap):\n",
           (long long)N, (long long)Cin);
    for (int k = 0; k <= sp.steps; ++k) {
        const vector<float>& mine = (k == 0) ? sample : trace[k - 1];   // trace[i] = sample after step i+1
        npy::Array rf, rb;
        bool hf = load_opt(fdir + "/f32_x_step" + std::to_string(k) + ".npy", rf);
        bool hb = load_opt(fdir + "/bf16_x_step" + std::to_string(k) + ".npy", rb);
        printf("  step %2d:\n", k);
        if (hf) print_stats("f32", diff_stats(mine, rf)); else printf("    f32  (missing)\n");
        if (hb) print_stats("bf16", diff_stats(mine, rb)); else printf("    bf16 (missing)\n");
        if (hf && hb) print_stats("f32v.bf16[calib]", diff_stats2(rf, rb));
    }

    // Explicit final-latent check (redundant with trace.back(), ships as its own file).
    Stats final_f32{}, final_calib{};
    bool has_final_f32 = false, has_final_calib = false;
    double cos_f32 = -1.0;
    {
        npy::Array rf, rb;
        printf("  final  :\n");
        bool hf = load_opt(fdir + "/f32_x_final.npy", rf);
        bool hb = load_opt(fdir + "/bf16_x_final.npy", rb);
        if (hf) { final_f32 = diff_stats(out, rf); has_final_f32 = true; print_stats("f32", final_f32); cos_f32 = mean_cosine(out, rf, (int)Cin, N); }
        else printf("    f32  (missing)\n");
        if (hb) print_stats("bf16", diff_stats(out, rb)); else printf("    bf16 (missing)\n");
        if (hf && hb) { final_calib = diff_stats2(rf, rb); has_final_calib = true; print_stats("f32v.bf16[calib]", final_calib); }
    }
    if (cos_f32 >= 0) printf("  mean per-token cosine-similarity(mine, f32 ref) = %.6f\n", cos_f32);

    // ---- denormalize -> compare to reference (already-denormalized) SLAT ----
    printf("\ndenormalized SLAT parity (mine denorm with norm_mean/std vs fixture's own *_slat.npy):\n");
    vector<float> slat_dn((size_t)Cin * N);
    for (int64_t n = 0; n < N; ++n) for (int c = 0; c < (int)Cin; ++c) {
        const float mean = (c < (int)nmean.numel()) ? nmean.data[c] : SHAPE_MEAN[c];
        const float std_  = (c < (int)nstd.numel())  ? nstd.data[c]  : SHAPE_STD[c];
        slat_dn[(size_t)c + Cin * n] = out[(size_t)c + Cin * n] * std_ + mean;
    }
    {
        npy::Array sf, sb;
        bool hf = load_opt(fdir + "/f32_slat.npy", sf);
        bool hb = load_opt(fdir + "/bf16_slat.npy", sb);
        if (hf) print_stats("f32", diff_stats(slat_dn, sf)); else printf("    f32  (missing)\n");
        if (hb) print_stats("bf16", diff_stats(slat_dn, sb)); else printf("    bf16 (missing)\n");
        if (hf && hb) print_stats("f32v.bf16[calib]", diff_stats2(sf, sb));
    }

    // ---- verdict ----
    // PASS: rel(mine, f32 ref) on x_final <= max(2 * rel(f32 ref, bf16 ref), 5e-2) -- i.e. we must
    // be at least as close to the f32 reference as the production bf16 run is (with a floor).
    printf("\n=== verdict ===\n");
    if (!has_final_f32) {
        printf("FAIL: no f32 x_final reference found in fixture -- cannot evaluate\n");
    } else {
        const double calib_rel = has_final_calib ? final_calib.rel : 0.0;
        const double threshold = std::max(2.0 * calib_rel, 5e-2);
        const bool pass = final_f32.rel <= threshold;
        printf("rel(mine, f32 ref) = %.4e ; threshold = max(2*rel(f32,bf16)=%.4e, 5e-2) = %.4e%s\n",
               final_f32.rel, has_final_calib ? 2.0 * calib_rel : -1.0, threshold,
               has_final_calib ? "" : "  [no bf16 calibration -- floor only]");
        printf("%s\n", pass ? "PASS" : "FAIL");
    }
    return 0;
}
