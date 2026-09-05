// Validate include/naf.h (valeoai/NAF neighborhood-attention feature upsampler,
// docs/spec/30-pixal3d-cond.md section 4) two ways:
//   trellis-test-naf --selftest                       hand-computed unit checks, no GGUF/fixture needed
//   trellis-test-naf <naf.gguf> <fixture_dir> [gpu] [--full]
//                                                       PyTorch parity (t128_* always; t512_* only with --full)
//
// Reference: valeoai/NAF src/model/naf.py + src/layers/{attentions,convolutions,rope}.py
#include "naf.h"
#include "trellis_model.h"
#include "npy.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using std::string; using std::vector;
using namespace trellis;

// ---------------------------------------------------------------------------
// shared compare helper (same shape as the other trellis-test-* binaries)
// ---------------------------------------------------------------------------
static bool compare(const char* name, const vector<float>& mine, const npy::Array& ref, double tol = 2e-3) {
    if ((int64_t)mine.size() != ref.numel()) {
        printf("  %-24s SIZE MISMATCH mine=%zu ref=%lld\n", name, mine.size(), (long long)ref.numel());
        return false;
    }
    double maxabs = 0, sumabs = 0, refmax = 0;
    for (size_t i = 0; i < mine.size(); ++i) {
        double d = std::fabs((double)mine[i] - ref.data[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)ref.data[i]));
    }
    double rel = refmax > 0 ? maxabs / refmax : maxabs;
    bool ok = rel < tol;
    printf("  %-24s max|d|=%.4e mean|d|=%.4e rel=%.4e tol=%.1e  %s\n",
           name, maxabs, sumabs / mine.size(), rel, tol, ok ? "PASS" : "FAIL");
    return ok;
}

static bool try_load(const string& path, npy::Array& out) {
    try { out = npy::load(path); return true; } catch (...) { return false; }
}

// ===========================================================================
// --selftest: hand-computable cases, no PyTorch / GGUF needed
// ===========================================================================
static bool test_groupnorm() {
    // 4 channels, 2 groups (2 ch/group), gamma=1 beta=0 -> each group's output
    // must have mean~0, var~1 (GroupNorm's defining property, independent of
    // the arbitrary input values).
    const int C = 4, H = 2, W = 2, G = 2;
    vector<float> x = {1, 2, 3, 4,   5, 6, 7, 8,   -1, 0, 1, 2,   10, 20, 30, 40};
    vector<float> gamma(C, 1.f), beta(C, 0.f);
    vector<float> out = naf_group_norm(x, C, H, W, G, gamma.data(), beta.data(), 1e-5f);
    bool ok = true;
    const int cg = C / G;
    for (int g = 0; g < G; ++g) {
        double sum = 0, sumsq = 0; int n = cg * H * W;
        for (int cc = 0; cc < cg; ++cc)
            for (int i = 0; i < H * W; ++i) { float v = out[(size_t)(g * cg + cc) * H * W + i]; sum += v; sumsq += (double)v * v; }
        double mean = sum / n, var = sumsq / n - mean * mean;
        bool g_ok = std::fabs(mean) < 1e-3 && std::fabs(var - 1.0) < 5e-2;
        printf("  groupnorm group %d: mean=%.5f var=%.5f  %s\n", g, mean, var, g_ok ? "PASS" : "FAIL");
        ok &= g_ok;
    }
    return ok;
}

static bool test_nearest_exact() {
    const int Hin = 4, Hout = 7; // upsample, PyTorch 'nearest-exact': src = floor((dst+0.5)*In/Out)
    vector<float> x(Hin); for (int i = 0; i < Hin; ++i) x[i] = (float)i;
    vector<float> out = naf_nearest_exact_resize(x, 1, Hin, 1, Hout, 1);
    bool ok = true;
    for (int oy = 0; oy < Hout; ++oy) {
        int expect = std::min(Hin - 1, (int)std::floor(((double)oy + 0.5) * Hin / Hout));
        int got = (int)std::lround(out[oy]);
        bool o = got == expect;
        printf("  nearest-exact oy=%d -> src=%d (want %d)  %s\n", oy, got, expect, o ? "PASS" : "FAIL");
        ok &= o;
    }
    return ok;
}

static bool test_adaptive_pool() {
    const int Hin = 10, Hout = 3; // PyTorch adaptive_avg_pool2d bin formula
    vector<float> x(Hin); for (int i = 0; i < Hin; ++i) x[i] = (float)i;
    vector<float> out = naf_adaptive_avg_pool2d(x, 1, Hin, 1, Hout, 1);
    const int starts[3] = {0, 3, 6}, ends[3] = {4, 7, 10};
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
        double s = 0; for (int k = starts[i]; k < ends[i]; ++k) s += k;
        double expect = s / (ends[i] - starts[i]);
        bool o = std::fabs(out[i] - expect) < 1e-4;
        printf("  adaptive_pool bin %d: got=%.4f want=%.4f (range [%d,%d))  %s\n",
               i, out[i], expect, starts[i], ends[i], o ? "PASS" : "FAIL");
        ok &= o;
    }
    return ok;
}

static bool test_na_window() {
    bool ok = true;
    int r, p, Lr, start;
    naf_na_window(0, 128, 4, 9, r, p, Lr, start);
    bool o1 = r == 0 && p == 0 && Lr == 32 && start == 0;
    printf("  na_window q=0   L=128 d=4 K=9: r=%d p=%d Lr=%d start=%d (want r=0 p=0 Lr=32 start=0)  %s\n",
           r, p, Lr, start, o1 ? "PASS" : "FAIL");
    ok &= o1;
    naf_na_window(127, 128, 4, 9, r, p, Lr, start);
    bool o2 = r == 3 && p == 31 && Lr == 32 && start == 23;
    printf("  na_window q=127 L=128 d=4 K=9: r=%d p=%d Lr=%d start=%d (want r=3 p=31 Lr=32 start=23)  %s\n",
           r, p, Lr, start, o2 ? "PASS" : "FAIL");
    ok &= o2;
    return ok;
}

static bool test_rope() {
    // Unit-vector probe of rotate_half + tiling: with only head0-channel0 set
    // to 1 (rest 0), rotate_half([x1||x2])=[-x2||x1] over the two 32-wide
    // halves means out[0]=cos(angle0) and out[32]=sin(angle0) exactly.
    const int T = 4;
    vector<float> periods(16);
    for (int j = 0; j < 16; ++j) periods[j] = std::pow(100.0, 2.0 * j / 32.0);
    vector<float> x((size_t)256 * T * T, 0.f);
    const int iy = 1, ix = 2;
    x[(size_t)0 * T * T + (size_t)iy * T + ix] = 1.f;
    naf_rope_apply_inplace(x, T, periods);
    const float u = ((float)iy + 0.5f) / T * 2.f - 1.f;
    const float angle0 = 2.f * (float)M_PI * u / periods[0];
    const float want_c0 = std::cos(angle0), want_c32 = std::sin(angle0);
    const float got_c0 = x[(size_t)0 * T * T + (size_t)iy * T + ix];
    const float got_c32 = x[(size_t)32 * T * T + (size_t)iy * T + ix];
    bool ok = std::fabs(got_c0 - want_c0) < 1e-4 && std::fabs(got_c32 - want_c32) < 1e-4;
    printf("  rope unit-vector: c0=%.5f (want %.5f)  c32=%.5f (want %.5f)  %s\n",
           got_c0, want_c0, got_c32, want_c32, ok ? "PASS" : "FAIL");
    return ok;
}

static int run_selftest() {
    printf("=== naf --selftest ===\n");
    bool all_ok = true;
    all_ok &= test_groupnorm();
    all_ok &= test_nearest_exact();
    all_ok &= test_adaptive_pool();
    all_ok &= test_na_window();
    all_ok &= test_rope();
    printf("=== selftest %s ===\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

// ===========================================================================
// fixture comparison
// ===========================================================================

// t128_probe_neighbors' axis order and 81-entry enumeration order aren't
// pinned by the spec, so this checks the SET of 81 (row,col) neighbor pixels
// each probe query attends to, under both (query axis order) x (neighbor axis
// order) interpretations, and reports whichever one the fixture matches.
static bool check_probe_neighbors(const string& dir, int T, int h, int w) {
    npy::Array pq, pn;
    if (!try_load(dir + "/t128_probe_queries.npy", pq) || !try_load(dir + "/t128_probe_neighbors.npy", pn)) {
        printf("  t128_probe_queries/neighbors.npy not found -- probe check SKIPPED\n");
        return true; // not a failure, just unverified
    }
    const int64_t Nq = pq.shape[0];
    const int dy = T / h, dx = T / w;
    bool all_ok = true;
    for (int64_t qi = 0; qi < Nq; ++qi) {
        int a = (int)std::lround(pq.data[qi * 2 + 0]);
        int b = (int)std::lround(pq.data[qi * 2 + 1]);
        bool matched = false;
        const char* which = "none";
        for (int conv = 0; conv < 2 && !matched; ++conv) {
            // conv==0: query=(a,b)=(y,x); conv==1: query=(a,b)=(x,y)
            int y = conv == 0 ? a : b;
            int x = conv == 0 ? b : a;
            int ry, py, Lry, starty, rx, px, Lrx, startx;
            naf_na_window(y, T, dy, 9, ry, py, Lry, starty);
            naf_na_window(x, T, dx, 9, rx, px, Lrx, startx);
            vector<std::pair<int,int>> mine;
            for (int wy = 0; wy < 9; ++wy)
                for (int wx = 0; wx < 9; ++wx)
                    mine.push_back({(starty + wy) * dy + ry, (startx + wx) * dx + rx});
            vector<std::pair<int,int>> ref_yx, ref_xy;
            for (int k = 0; k < 81; ++k) {
                int r0 = (int)std::lround(pn.data[(qi * 81 + k) * 2 + 0]);
                int r1 = (int)std::lround(pn.data[(qi * 81 + k) * 2 + 1]);
                ref_yx.push_back({r0, r1});
                ref_xy.push_back({r1, r0});
            }
            std::sort(mine.begin(), mine.end());
            std::sort(ref_yx.begin(), ref_yx.end());
            std::sort(ref_xy.begin(), ref_xy.end());
            if (mine == ref_yx) { matched = true; which = conv == 0 ? "query=(y,x) neighbor=(row,col)" : "query=(x,y) neighbor=(row,col)"; }
            else if (mine == ref_xy) { matched = true; which = conv == 0 ? "query=(y,x) neighbor=(col,row)" : "query=(x,y) neighbor=(col,row)"; }
        }
        printf("  probe[%lld] query=(%d,%d): %-38s %s\n",
               (long long)qi, a, b, matched ? which : "no interpretation matched", matched ? "PASS" : "FAIL");
        all_ok &= matched;
    }
    return all_ok;
}

// t128_{coords,angles,cos,sin} are dumped alongside t128_rope_periods -- an
// independent re-derivation of the RoPE tables (recomputed here from the
// checkpoint's periods, not by calling naf_rope_apply_inplace) cross-checks
// the formula itself, on top of q_rope already exercising the applied rotation.
static bool check_rope_tables(const string& dir, int T, const vector<float>& periods) {
    const string pfx = "t" + std::to_string(T) + "_";
    npy::Array coords_ref, angles_ref, cos_ref, sin_ref;
    bool have = try_load(dir + "/" + pfx + "coords.npy", coords_ref) &&
                try_load(dir + "/" + pfx + "angles.npy", angles_ref) &&
                try_load(dir + "/" + pfx + "cos.npy", cos_ref) &&
                try_load(dir + "/" + pfx + "sin.npy", sin_ref);
    if (!have) {
        printf("  %s{coords,angles,cos,sin}.npy not found -- rope-table check SKIPPED\n", pfx.c_str());
        return true;
    }
    vector<float> coords((size_t)T * T * 2), angles((size_t)T * T * 64), cosv((size_t)T * T * 64), sinv((size_t)T * T * 64);
    for (int i = 0; i < T; ++i) {
        const float u = ((float)i + 0.5f) / T * 2.f - 1.f;
        for (int j = 0; j < T; ++j) {
            const float v = ((float)j + 0.5f) / T * 2.f - 1.f;
            const size_t pix = (size_t)i * T + j;
            coords[pix * 2 + 0] = u; coords[pix * 2 + 1] = v;
            float ang[64];
            for (int k = 0; k < 16; ++k) {
                ang[k] = 2.f * (float)M_PI * u / periods[k];
                ang[16 + k] = 2.f * (float)M_PI * v / periods[k];
            }
            for (int d = 0; d < 32; ++d) ang[32 + d] = ang[d];
            for (int d = 0; d < 64; ++d) {
                angles[pix * 64 + d] = ang[d];
                cosv[pix * 64 + d] = std::cos(ang[d]);
                sinv[pix * 64 + d] = std::sin(ang[d]);
            }
        }
    }
    bool ok = true;
    ok &= compare("coords", coords, coords_ref, 1e-4);
    ok &= compare("angles", angles, angles_ref, 2e-3);
    ok &= compare("cos", cosv, cos_ref, 2e-3);
    ok &= compare("sin", sinv, sin_ref, 2e-3);
    return ok;
}

static bool run_size(const Model& m, int T, const vector<float>& image, int S,
                      const vector<float>& lr, int C, int h, int w, const string& dir, bool have_periods_check) {
    const string pfx = "t" + std::to_string(T) + "_";
    npy::Array enc_cat_ref, enc_pooled_ref, q_rope_ref, k_pooled_ref, k_up_ref, out_ref, v_up_ref;
    bool have = try_load(dir + "/" + pfx + "enc_cat.npy", enc_cat_ref) &&
                try_load(dir + "/" + pfx + "enc_pooled.npy", enc_pooled_ref) &&
                try_load(dir + "/" + pfx + "q_rope.npy", q_rope_ref) &&
                try_load(dir + "/" + pfx + "k_pooled.npy", k_pooled_ref) &&
                try_load(dir + "/" + pfx + "k_up.npy", k_up_ref) &&
                try_load(dir + "/" + pfx + "out.npy", out_ref);
    if (!have) {
        printf("%sfixture files not found under %s -- T=%d parity UNVERIFIED\n", pfx.c_str(), dir.c_str(), T);
        return false;
    }
    bool have_v_up = try_load(dir + "/" + pfx + "v_up.npy", v_up_ref);

    printf("\n=== T=%d ===\n", T);
    NafDebug dbg;
    auto t0 = std::chrono::steady_clock::now();
    vector<float> out = naf_upsample(m, image.data(), S, lr.data(), C, h, w, T, &dbg);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    printf("  naf_upsample(T=%d) runtime: %.2fs\n", T, secs);

    // The GPU encoder path (CUDA builds, unless TRELLIS_NAF_CPU=1) runs its GEMMs at the
    // backend's f16-staged precision (docs/spec/30 section 5), so its intermediates sit
    // around 3e-3..7e-3; the final output is still held to 3e-3.
    const bool gpu_path = std::string(ggml_backend_name(m.backend)).find("CUDA") != std::string::npos
                          && !std::getenv("TRELLIS_NAF_CPU");
    const double enc_tol = gpu_path ? 1e-2 : 2e-3;
    if (gpu_path) printf("  (GPU NAF path: encoder-stage tolerance %.0e)\n", enc_tol);
    bool ok = true;
    ok &= compare("enc_cat", dbg.enc_cat, enc_cat_ref, enc_tol);
    ok &= compare("enc_pooled", dbg.enc_pooled, enc_pooled_ref, enc_tol);
    ok &= compare("q_rope", dbg.q_rope, q_rope_ref, enc_tol);
    ok &= compare("k_pooled", dbg.k_pooled, k_pooled_ref, enc_tol);
    ok &= compare("k_up", dbg.k_up, k_up_ref, enc_tol);
    if (have_v_up) ok &= compare("v_up", dbg.v_up, v_up_ref, enc_tol);
    else printf("  %-24s v_up.npy not found -- SKIPPED\n", pfx.c_str());
    ok &= compare("out", out, out_ref, 3e-3);

    (void)have_periods_check;
    return ok;
}

static int run_fixture(const string& gguf, const string& dir, int gpu, bool full) {
    printf("=== naf fixture: %s ===\n", dir.c_str());
    npy::Array image_a, lr_a;
    if (!try_load(dir + "/image.npy", image_a) || !try_load(dir + "/lr_features.npy", lr_a)) {
        printf("image.npy/lr_features.npy not found under %s -- parity UNVERIFIED\n", dir.c_str());
        return -1;
    }
    const int S = (int)image_a.shape[image_a.shape.size() - 1];
    const int C = (int)lr_a.shape[lr_a.shape.size() - 3];
    const int h = (int)lr_a.shape[lr_a.shape.size() - 2];
    const int w = (int)lr_a.shape[lr_a.shape.size() - 1];
    printf("fixture: S=%d C=%d h=%d w=%d\n", S, C, h, w);

    Model m = Model::load(gguf, gpu);
    printf("loaded %s (%zu tensors)\n", m.arch.c_str(), m.tensors.size());

    bool all_ok = true;
    vector<float> periods = tensor_to_f32(m.get("image_encoder.rope.periods"));
    npy::Array periods_ref;
    bool have_periods = try_load(dir + "/t128_rope_periods.npy", periods_ref);
    if (have_periods) all_ok &= compare("rope_periods", periods, periods_ref, 1e-4);
    else printf("t128_rope_periods.npy not found -- periods-vs-checkpoint check SKIPPED (still used internally)\n");

    all_ok &= run_size(m, 128, image_a.data, S, lr_a.data, C, h, w, dir, have_periods);
    all_ok &= check_probe_neighbors(dir, 128, h, w);
    all_ok &= check_rope_tables(dir, 128, periods);

    if (full) {
        all_ok &= run_size(m, 512, image_a.data, S, lr_a.data, C, h, w, dir, have_periods);
    } else {
        printf("\n(T=512 skipped -- pass --full to run it; its out.npy is ~1GB)\n");
    }

    m.free();
    printf("\n=== overall %s ===\n", all_ok ? "PASS" : "FAIL (see above)");
    return all_ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s --selftest | <naf.gguf> <fixture_dir> [gpu] [--full]\n", argv[0]);
        return 1;
    }
    if (string(argv[1]) == "--selftest") return run_selftest();

    vector<string> pos;
    bool full = false;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--full") full = true;
        else pos.push_back(argv[i]);
    }
    if (pos.size() < 2) {
        fprintf(stderr, "usage: %s --selftest | <naf.gguf> <fixture_dir> [gpu] [--full]\n", argv[0]);
        return 1;
    }
    const string gguf = pos[0], dir = pos[1];
    const int gpu = pos.size() > 2 ? atoi(pos[2].c_str()) : 0;

    int rc = run_fixture(gguf, dir, gpu, full);
    if (rc < 0) return 0; // fixture absent: not a build/test failure, just unverified
    return rc;
}
