// End-to-end SLAT sampling parity test for the Pixal3D MV flow stages: runs the full 12-step
// FlowEuler/CFG sampler (sparse ProjectAttention DiT, cond+proj) from fixture noise over the
// fixture's active coords, then denormalizes with the stage's mean/std and compares both the
// normalized per-step latents and the denormalized SLAT against f32 and bf16 PyTorch references,
// using the ref-f32-vs-bf16 delta as the calibration baseline. See docs/spec/30-pixal3d-cond.md
// §5/6: block-level goldens are only meaningful on the CPU backend; the production parity gate
// is end-to-end sampling in the reference's own dtype (bf16). Mirrors test_pixal3d_ss_sample.cpp
// (dense SS version) but drives the SPARSE runner (test_pixal3d_slat_flow.cpp coords/layout
// conventions: coords [N,4] int32 (b,x,y,z), x/proj [N,C] row-major == ggml [C,N] channel-major,
// same bytes -- no dense-grid remap needed here).
//
// Three stages, selected by --stage (default shape512, preserved byte-for-byte):
//   shape512 : res-512 shape-SLAT stage, in_ch=out_ch=32, fixture from tools/ref_pixal3d_slat_sample.py.
//              coords.npy [N,4] i32; noise.npy [N,32]; cond_global.npy [1,5,1024]; cond_proj.npy
//              [N,2048]; norm_mean.npy/norm_std.npy [32]; f32_/bf16_ x_step0..12.npy [N,32],
//              x_final.npy [N,32], slat.npy [N,32] (denormalized). Sampler params hardcoded
//              (trellis_cli.cpp shape_flow(): steps=12 gs=7.5 gr=0.5 gi=[0.6,1.0] rescale_t=3.0).
//   shape_hr : res-1024 (grid 64) shape-SLAT stage, same in_ch=out_ch=32 as shape512, but the
//              HR cascade's cond (2048-ch proj at grid 64) and coords. Same sampler params as
//              shape512 (trellis_cli.cpp's shape_flow() lambda is reused verbatim for LR/HR).
//   tex      : texture-SLAT stage, in_ch=64 (concat of the 32-ch noise/state and the 32-ch
//              NORMALIZED shape SLAT -- verified against trellis_cli.cpp:296-322: `tslat` there
//              is `slat_norm`/`lr_norm`, never denormalized, and is re-concatenated into a fresh
//              64-ch buffer [state(0:32) ; tslat(32:64)] on EVERY forward call inside the FlowFwd
//              lambda, since the sampler only integrates the first 32 (state) channels -- the
//              concat half is step-invariant. This test mirrors that: the DitRunner is built with
//              p.in_ch=64/p.out_ch=32, and the FlowFwdProj wrapper below rebuilds the 64-ch input
//              from the current 32-ch sampler state plus a fixture-provided fixed 32-ch
//              concat_cond every call. out_ch=32, and normalization uses TEX_MEAN/TEX_STD.
// Fixture dirs: shape512 from tools/ref_pixal3d_slat_sample.py's <fixture_dir>; shape_hr/tex from
// tools/ref_pixal3d_hr_sample.py's <fixture_dir> (planned name /mnt/hdd1/pixal3d/ref/pixal3d/hr_sample):
//   hr_coords.npy [Nh,4] i32 (shared by shape_hr and tex); hr_cond_global.npy/hr_cond_proj.npy
//   (shape_hr); tex_cond_global.npy/tex_cond_proj.npy (tex); shape_noise.npy/tex_noise.npy [Nh,32];
//   shape_norm_mean.npy/std.npy, tex_norm_mean.npy/std.npy [32]; shape_sampler_params.npy/
//   tex_sampler_params.npy [6] = (steps, guidance_strength, guidance_rescale, gi0, gi1, rescale_t)
//   -- read when present, else the hardcoded defaults above; f32_/bf16_ prefixed shape_x_step{k},
//   shape_x_final, shape_slat (shape_hr) / tex_x_step{k}, tex_x_final, tex_slat (tex).
//   tex_concat_cond.npy [Nh,32]: the fixed second half of the tex DiT's 64-ch input (this test
//   tries the unprefixed shared name first, then f32_tex_concat_cond.npy -- the task spec listed
//   this file ambiguously under the "per f32_/bf16_ prefix" group, which would imply per-dtype-run
//   values from each dtype's own shape-stage output; UNVERIFIED until tools/ref_pixal3d_hr_sample.py
//   is [Nh,32] float32, saved per-dtype-run as f32_tex_concat_cond.npy / bf16_tex_concat_cond.npy
//   (confirmed with the fixture author 2026-09-06): each dtype's own HR-shape cascade produces a
//   slightly different concat_cond (denormalized shape SLAT, renormalized with
//   shape_slat_normalization), so there is no shared unprefixed file. This test uses the f32
//   run's concat_cond as "mine"'s tex-DiT input (f32 is the primary comparison target
//   throughout), falling back to bf16's only if f32's file is absent. Channel order confirmed
//   against structured_latent_flow.py: `x = sp.sparse_cat([x, concat_cond], dim=-1)` -- noise
//   channels first (0:32), concat_cond appended after (32:64) -- matching trellis_cli.cpp.
//   trellis-test-pixal3d-slat-sample <flow.gguf> <fixture_dir> [gpu] [--stage shape512|shape_hr|tex]
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
// tex SLAT mean/std (trellis_cli.cpp TEX_MEAN/TEX_STD), used only for --stage tex.
static const float TEX_MEAN[32]={3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,-1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293000f,-1.925608f,-1.217717f,1.213905f,0.971588f,-0.023631f,0.106750f,2.021786f,0.250524f,-0.662387f,-0.768862f};
static const float TEX_STD[32]={2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,2.295916f,3.234985f,3.233086f,2.260140f,2.874801f,2.810596f,3.292720f,2.674999f,2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.775190f};

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

// Per-stage fixture file names + defaults. shape512 reproduces the original hardcoded names
// exactly (byte-for-byte behavior preserved); shape_hr/tex use the hr_sample fixture layout.
struct StageFiles {
    string coords, noise, cond_global, cond_proj, norm_mean, norm_std;
    string sampler_params;         // "" => no optional sampler-params file for this stage
    string x_step_prefix, x_final, slat;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <flow.gguf> <fixture_dir> [gpu] [--stage shape512|shape_hr|tex]\n", argv[0]);
        return 1;
    }
    const string gguf_flow = argv[1], fdir = argv[2];
    int gpu = 0;
    string stage_name = "shape512";
    for (int i = 3; i < argc; ++i) {
        string a = argv[i];
        if (a == "--stage" && i + 1 < argc) stage_name = argv[++i];
        else gpu = atoi(a.c_str());
    }
    enum class Stage { SHAPE512, SHAPE_HR, TEX } stage;
    if (stage_name == "shape512") stage = Stage::SHAPE512;
    else if (stage_name == "shape_hr") stage = Stage::SHAPE_HR;
    else if (stage_name == "tex") stage = Stage::TEX;
    else { fprintf(stderr, "unknown --stage '%s' (want shape512|shape_hr|tex)\n", stage_name.c_str()); return 1; }
    const bool is_tex = (stage == Stage::TEX);

    StageFiles sf;
    switch (stage) {
    case Stage::SHAPE512:
        sf = { "coords.npy", "noise.npy", "cond_global.npy", "cond_proj.npy", "norm_mean.npy", "norm_std.npy",
               "", "x_step", "x_final", "slat" };
        break;
    case Stage::SHAPE_HR:
        sf = { "hr_coords.npy", "shape_noise.npy", "hr_cond_global.npy", "hr_cond_proj.npy",
               "shape_norm_mean.npy", "shape_norm_std.npy",
               "shape_sampler_params.npy", "shape_x_step", "shape_x_final", "shape_slat" };
        break;
    case Stage::TEX:
        sf = { "hr_coords.npy", "tex_noise.npy", "tex_cond_global.npy", "tex_cond_proj.npy",
               "tex_norm_mean.npy", "tex_norm_std.npy",
               "tex_sampler_params.npy", "tex_x_step", "tex_x_final", "tex_slat" };
        break;
    }

    printf("stage=%s fixture_dir=%s gpu=%d\n", stage_name.c_str(), fdir.c_str(), gpu);

    if (getenv("TRELLIS_NOFA")) { trellis::g_no_fa = true; printf("(TRELLIS_NOFA: soft_max path)\n"); }

    // ---- fixture inputs ----
    I32Array co = load_npy_i32(fdir + "/" + sf.coords);           // [N,4] (b,x,y,z)
    npy::Array noise = npy::load(fdir + "/" + sf.noise);          // [N,32] (state/output channels)
    npy::Array zg = npy::load(fdir + "/" + sf.cond_global);       // [1,5,1024]
    npy::Array zp = npy::load(fdir + "/" + sf.cond_proj);         // [N,2048]
    npy::Array nmean = npy::load(fdir + "/" + sf.norm_mean);      // [32]
    npy::Array nstd  = npy::load(fdir + "/" + sf.norm_std);       // [32]
    npy::Array tseq, gstr;
    bool has_tseq = load_opt(fdir + "/t_seq.npy", tseq);
    bool has_gstr = load_opt(fdir + "/guidance_strength.npy", gstr);

    const int64_t N = co.shape.empty() ? 0 : co.shape[0];
    const int64_t Cin = noise.shape[1];        // sampler state width: 32 for every stage (tex concats
                                                // to 64 only inside the DiT forward, see below)
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

    // tex stage only: fixed 32-ch second half of the DiT's 64-ch input (NORMALIZED shape SLAT,
    // per trellis_cli.cpp:296-322 and confirmed against structured_latent_flow.py's
    // `x = sp.sparse_cat([x, concat_cond], dim=-1)` -- noise/state channels first, concat_cond
    // (renormalized HR shape SLAT) appended after). Loaded once, re-concatenated every forward
    // call below (mirrors trellis_cli.cpp's tex FlowFwd lambda rebuilding x64 each step).
    // Per-dtype-run file (confirmed with the fixture author): each dtype's own HR-shape cascade
    // produces a slightly different concat_cond (denormalized shape_slat -> renormalized with
    // shape_slat_normalization), so there is no shared unprefixed file -- use the f32 run's own
    // concat_cond as "mine"'s input (consistent with f32 being the primary comparison target
    // throughout this test); bf16's is a fallback only if f32's is somehow absent.
    vector<float> concat_cond;
    if (is_tex) {
        npy::Array cc;
        string used;
        if (load_opt(fdir + "/f32_tex_concat_cond.npy", cc)) used = "f32_tex_concat_cond.npy";
        else if (load_opt(fdir + "/bf16_tex_concat_cond.npy", cc)) used = "bf16_tex_concat_cond.npy";
        else { fprintf(stderr, "tex stage: no f32_/bf16_ tex_concat_cond.npy found in %s\n", fdir.c_str()); return 1; }
        if (cc.shape[0] != N) { fprintf(stderr, "tex_concat_cond tokens=%lld != N=%lld\n", (long long)cc.shape[0], (long long)N); return 1; }
        concat_cond.assign(cc.data.begin(), cc.data.begin() + (size_t)Cin * N);
        printf("tex concat_cond loaded from %s\n", used.c_str());
    }

    // ---- sparse SLAT flow sampler ----
    trellis::Model mf = trellis::Model::load(gguf_flow, gpu);
    printf("loaded %s (%zu tensors)\n", mf.arch.c_str(), mf.tensors.size());
    trellis::DiTParams p;
    p.in_ch = is_tex ? (int)(2 * Cin) : (int)Cin;   // tex DiT sees [state(32) ; shape_slat_norm(32)]
    p.out_ch = (int)Cin;
    p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(TRELLIS_F32W: f32 weight compute)\n"); }
    bool detected = trellis::dit_detect_proj_attn(mf, p);
    printf("proj_attn detected=%d d_proj=%d (fixture Dp=%lld)\n", detected, p.d_proj, (long long)Dp);
    if (!detected) { fprintf(stderr, "expected a Pixal3D checkpoint (blocks.0.cross_attn.proj_linear.weight missing)\n"); return 1; }
    if (p.d_proj != (int)Dp) { fprintf(stderr, "d_proj mismatch: checkpoint=%d fixture=%lld\n", p.d_proj, (long long)Dp); return 1; }

    trellis::DitRunner* run = trellis::make_sparse_runner(mf, p, coords3, (int)Lc);
    trellis::FlowFwdProj fwd = [&](const vector<float>& x, float ts, const float* c, const float* pj) {
        if (!is_tex) return run->forward(x, ts, c, pj);
        vector<float> x64((size_t)2 * Cin * N);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < Cin; ++k) x64[(size_t)k + 2 * Cin * n]        = x[(size_t)k + Cin * n];
            for (int64_t k = 0; k < Cin; ++k) x64[(size_t)Cin + k + 2 * Cin * n]  = concat_cond[(size_t)k + Cin * n];
        }
        return run->forward(x64, ts, c, pj);
    };

    // Sampler params: hardcoded stage default, overridden by the fixture's own sampler_params.npy
    // [6] = (steps, guidance_strength, guidance_rescale, gi0, gi1, rescale_t) when present. shape512
    // never has one (sf.sampler_params == ""), so it keeps the original hardcoded values verbatim.
    trellis::SamplerParams sp;
    if (is_tex) { sp.steps = 12; sp.guidance_strength = 1.0f; sp.guidance_rescale = 0.0f; sp.gi0 = 0.6f; sp.gi1 = 0.9f; sp.rescale_t = 3.0f; }
    else        { sp.steps = 12; sp.guidance_strength = 7.5f; sp.guidance_rescale = 0.5f; sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 3.0f; }
    sp.sigma_min = 1e-5f;
    if (!sf.sampler_params.empty()) {
        npy::Array spa;
        if (load_opt(fdir + "/" + sf.sampler_params, spa) && spa.numel() >= 6) {
            sp.steps = (int)std::lround((double)spa.data[0]);
            sp.guidance_strength = spa.data[1];
            sp.guidance_rescale  = spa.data[2];
            sp.gi0 = spa.data[3]; sp.gi1 = spa.data[4]; sp.rescale_t = spa.data[5];
            printf("sampler params loaded from %s: steps=%d gs=%.3f gr=%.3f gi=[%.2f,%.2f] rescale_t=%.2f\n",
                   sf.sampler_params.c_str(), sp.steps, sp.guidance_strength, sp.guidance_rescale, sp.gi0, sp.gi1, sp.rescale_t);
        } else {
            printf("no %s in fixture -- using hardcoded %s default sampler params\n",
                   sf.sampler_params.c_str(), is_tex ? "tex" : "shape");
        }
    }

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
        bool hf = load_opt(fdir + "/f32_" + sf.x_step_prefix + std::to_string(k) + ".npy", rf);
        bool hb = load_opt(fdir + "/bf16_" + sf.x_step_prefix + std::to_string(k) + ".npy", rb);
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
        bool hf = load_opt(fdir + "/f32_" + sf.x_final + ".npy", rf);
        bool hb = load_opt(fdir + "/bf16_" + sf.x_final + ".npy", rb);
        if (hf) { final_f32 = diff_stats(out, rf); has_final_f32 = true; print_stats("f32", final_f32); cos_f32 = mean_cosine(out, rf, (int)Cin, N); }
        else printf("    f32  (missing)\n");
        if (hb) print_stats("bf16", diff_stats(out, rb)); else printf("    bf16 (missing)\n");
        if (hf && hb) { final_calib = diff_stats2(rf, rb); has_final_calib = true; print_stats("f32v.bf16[calib]", final_calib); }
    }
    if (cos_f32 >= 0) printf("  mean per-token cosine-similarity(mine, f32 ref) = %.6f\n", cos_f32);

    // ---- denormalize -> compare to reference (already-denormalized) SLAT ----
    printf("\ndenormalized SLAT parity (mine denorm with norm_mean/std vs fixture's own *_%s.npy):\n", sf.slat.c_str());
    const float* fallback_mean = is_tex ? TEX_MEAN : SHAPE_MEAN;
    const float* fallback_std  = is_tex ? TEX_STD  : SHAPE_STD;
    vector<float> slat_dn((size_t)Cin * N);
    for (int64_t n = 0; n < N; ++n) for (int c = 0; c < (int)Cin; ++c) {
        const float mean = (c < (int)nmean.numel()) ? nmean.data[c] : fallback_mean[c];
        const float std_  = (c < (int)nstd.numel())  ? nstd.data[c]  : fallback_std[c];
        slat_dn[(size_t)c + Cin * n] = out[(size_t)c + Cin * n] * std_ + mean;
    }
    {
        npy::Array sf32, sb16;
        bool hf = load_opt(fdir + "/f32_" + sf.slat + ".npy", sf32);
        bool hb = load_opt(fdir + "/bf16_" + sf.slat + ".npy", sb16);
        if (hf) print_stats("f32", diff_stats(slat_dn, sf32)); else printf("    f32  (missing)\n");
        if (hb) print_stats("bf16", diff_stats(slat_dn, sb16)); else printf("    bf16 (missing)\n");
        if (hf && hb) print_stats("f32v.bf16[calib]", diff_stats2(sf32, sb16));
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
