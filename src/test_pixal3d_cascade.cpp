// Pixal3D sparse-structure resolution (ss_res 32|64) cascade coordinate tests (#78-#81).
//
//   trellis-test-pixal3d-cascade
//       Synthetic, weight-free checks of pixal3d_quantize_hr_coords / pixal3d_cascade_select:
//       ss_res=32 is bit-identical to the former literal-512 formula (and backoff loop),
//       ss_res=64 maps the 1024 span into the 64-cell grid (never the doubled ~128 domain),
//       mismatched ss_res / unsupported values fail closed.
//   trellis-test-pixal3d-cascade --fixture DIR --models MDIR [--gpu N]
//       Real-fixture oracle from tools/ref_pixal3d_ss_res.py: for each ss<R>/ subdir present,
//       occ_logits -> ss_coords(64, R) == ss_coords.npy; shape_upsample(lr_slat, ss_coords) ==
//       upsampled_coords.npy (sorted multiset); quantization @1024 == hr1024_coords.npy and the
//       1536 backoff == hr1536_coords.npy + hr1536_meta.npy. Exact integer equality throughout.
#include "pixal3d_cascade.h"
#include "pixal3d_cond.h"
#include "npy.h"
#include "shape_decoder.h"
#include "ss_decoder.h"
#include "trellis_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

using namespace trellis;
using C3 = std::array<int,3>;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

// The pre-#81 trellis_cli.cpp expression and backoff loop, kept verbatim as the ss_res=32 oracle.
static std::vector<C3> old_select(const std::vector<C3>& up, int hr_res, int max_tokens, int* out_res) {
    for (;;) {
        const int gi = hr_res / 16;
        const float gm1 = (float)(gi - 1);
        std::set<C3> q;
        for (auto& c : up)
            q.insert({ (int)std::lround((c[0]+0.5f)/512.f*gm1),
                       (int)std::lround((c[1]+0.5f)/512.f*gm1),
                       (int)std::lround((c[2]+0.5f)/512.f*gm1) });
        if ((int)q.size() < max_tokens || hr_res == 1024) { *out_res = hr_res; return {q.begin(), q.end()}; }
        hr_res -= 128;
    }
}

static int max_coord(const std::vector<C3>& v) {
    int m = -1;
    for (auto& c : v) m = std::max({m, c[0], c[1], c[2]});
    return m;
}

static std::vector<C3> random_coords(int span, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, span - 1);
    std::vector<C3> v(n);
    for (auto& c : v) c = {d(rng), d(rng), d(rng)};
    return v;
}

static void synthetic() {
    printf("[synthetic] ss_res=32 == literal-512 formula\n");
    for (int hr : {1024, 1152, 1280, 1408, 1536}) {
        std::vector<C3> axis;                         // every per-axis value 0..511 on each axis
        for (int c = 0; c < 512; ++c) { axis.push_back({c, 0, 511 - c}); axis.push_back({511 - c, c, 0}); axis.push_back({0, 511 - c, c}); }
        auto rnd = random_coords(512, 200000, 1234u + hr);
        axis.insert(axis.end(), rnd.begin(), rnd.end());
        int res_old = 0;
        auto a = old_select(axis, hr, 1 << 30, &res_old);
        auto b = pixal3d_quantize_hr_coords(axis, 32, hr);
        CHECK(a == b, "hr=%d: ss_res=32 helper differs from the literal-512 formula (%zu vs %zu)", hr, a.size(), b.size());
        CHECK(max_coord(b) == hr / 16 - 1, "hr=%d: max coord %d != grid-1 %d", hr, max_coord(b), hr / 16 - 1);
    }

    printf("[synthetic] ss_res=32 backoff == old loop\n");
    {
        auto up = random_coords(512, 400000, 7u);
        int res_old = 0;
        const int n1536 = (int)pixal3d_quantize_hr_coords(up, 32, 1536).size();
        const int n1408 = (int)pixal3d_quantize_hr_coords(up, 32, 1408).size();
        for (int mt : {n1536 + 1, n1536, (n1536 + n1408) / 2, n1408, 1}) {
            auto a = old_select(up, 1536, mt, &res_old);
            auto r = pixal3d_cascade_select(up, 32, 1536, mt);
            CHECK(a == r.coords && res_old == r.hr_res && r.grid == r.hr_res / 16,
                  "max_tokens=%d: select (res %d, %zu) != old (res %d, %zu)", mt, r.hr_res, r.coords.size(), res_old, a.size());
        }
        auto r = pixal3d_cascade_select(up, 32, 1024, 1);
        CHECK(r.hr_res == 1024, "1024 must never back off (got %d)", r.hr_res);
        printf("  1536 tokens=%d 1408 tokens=%d\n", n1536, n1408);
    }

    printf("[synthetic] ss_res=64 stays in the model grid\n");
    {
        std::vector<C3> all;
        for (int c = 0; c < 1024; ++c) all.push_back({c, c, c});
        auto q = pixal3d_quantize_hr_coords(all, 64, 1024);
        CHECK(max_coord(q) == 63, "ss_res=64 @1024: max coord %d != 63", max_coord(q));
        CHECK(q.front() == (C3{0, 0, 0}), "ss_res=64 @1024: min coord not 0");
        int res_old = 0;
        auto doubled = old_select(all, 1024, 1 << 30, &res_old);   // what #193's bug produces
        CHECK(max_coord(doubled) >= 125, "literal-512 at span 1024 should double (max %d)", max_coord(doubled));
        printf("  ss_res=64 @1024: max coord %d (literal-512 formula would give %d)\n", max_coord(q), max_coord(doubled));
        auto q15 = pixal3d_quantize_hr_coords(all, 64, 1536);
        CHECK(max_coord(q15) == 95, "ss_res=64 @1536: max coord %d != 95", max_coord(q15));
        // Same physical position at the two ss_res: a res-64 child c sits inside res-32 child c/2;
        // their normalized centers differ by 0.25/512, so quantized cells differ by at most 1.
        int worst = 0;
        for (int c = 0; c < 1024; ++c) {
            const int a = pixal3d_quantize_hr_coords({{c, 0, 0}}, 64, 1024)[0][0];
            const int b = pixal3d_quantize_hr_coords({{c / 2, 0, 0}}, 32, 1024)[0][0];
            worst = std::max(worst, std::abs(a - b));
        }
        CHECK(worst <= 1, "ss_res 64 vs 32 cell disagreement %d > 1", worst);
    }

    printf("[synthetic] fail closed\n");
    auto throws = [](auto fn) { try { fn(); } catch (const std::exception&) { return true; } return false; };
    CHECK(throws([] { pixal3d_quantize_hr_coords({{600, 0, 0}}, 32, 1024); }),
          "ss_res=32 with a span-1024 coord must throw (ss_res mismatch)");
    CHECK(throws([] { pixal3d_quantize_hr_coords({{-1, 0, 0}}, 64, 1024); }), "negative coord must throw");
    CHECK(throws([] { pixal3d_quantize_hr_coords({{0, 0, 0}}, 48, 1024); }), "ss_res=48 must throw");
    CHECK(throws([] { pixal3d_quantize_hr_coords({{0, 0, 0}}, 16, 1024); }), "ss_res=16 must throw");
    CHECK(!pixal3d_ss_res_supported(128) && pixal3d_ss_res_supported(32) && pixal3d_ss_res_supported(64),
          "supported set must be exactly {32, 64}");
    CHECK(pixal3d_quantize_hr_coords({}, 64, 1024).empty(), "empty input -> empty output");
}

// #82: every production conditioning call must fit the measured WebGPU binding-limit classes
// (~4 GiB maxBufferSize adapters report 4095 MiB; ~2 GiB maxStorageBufferBindingSize adapters
// 2047 MiB) even at the worst-case token count (the full grid), while the dense R^3 path the
// production code must not use trips the smaller class once R grows.
static void memory_plan() {
    printf("[synthetic] conditioning memory plan vs per-buffer limit classes\n");
    const uint64_t MiB = 1 << 20, lim4 = 4095 * MiB, lim2 = 2047 * MiB;
    struct Case { const char* name; Pixal3dSlatCondParams prm; int64_t tokens; bool sparse; bool fits2; };
    auto P = [](int S, int R, int T, int chunk) { Pixal3dSlatCondParams p{S, R, T, 1.0f}; p.naf_block_chunk = chunk; return p; };
    const Case cases[] = {
        { "LR ss32 S512 T512 R32, full grid",        P(512, 32, 512, 256),   32LL * 32 * 32, true, true },
        { "LR ss64 S512 T512 R64, full grid",        P(512, 64, 512, 256),   64LL * 64 * 64, true, true },
        { "HR 1024 S1024 T512 R64, full grid",       P(1024, 64, 512, 1024), 64LL * 64 * 64, true, true },
        { "HR 1536 S1024 T512 R96, token budget",    P(1024, 96, 512, 1024), 49152,          true, true },
        { "tex S1024 T1024 R64 (auto chunk), full",  P(1024, 64, 1024, 0),   64LL * 64 * 64, true, true },
        { "HR S1024 T512 R64 single graph (no chunk)", P(1024, 64, 512, 0), 64LL * 64 * 64, true, true },
        { "DENSE R96 (1536 grid) accumulator",       P(1024, 96, 512, 1024), 96LL * 96 * 96, false, false },
    };
    for (const Case& c : cases) {
        const Pixal3dCondSlatPlan p = pixal3d_cond_slat_plan(c.prm, c.tokens, c.sparse);
        printf("  %-40s largest %7.1f MiB (%s)%s\n", c.name, p.largest_bytes / (double)MiB, p.what, p.chunked ? " chunked" : "");
        CHECK(p.largest_bytes <= lim4 || !c.fits2, "%s: %llu MiB exceeds the 4 GiB class", c.name, (unsigned long long)(p.largest_bytes / MiB));
        CHECK((p.largest_bytes <= lim2) == c.fits2, "%s: 2 GiB class verdict wrong (%llu MiB)", c.name, (unsigned long long)(p.largest_bytes / MiB));
    }
    // Known intermediates must be in the plan: S=1024 DINOv3 scores [4101, 4101, 16] f32 = 1026.5 MiB,
    // and the unchunked S=1024/T=512 NAF value windows [1024, 81 * 4096] f32 = 1296 MiB.
    CHECK(pixal3d_cond_slat_plan(P(1024, 64, 512, 1024), 1000, true).largest_bytes == 4101ULL * 4101 * 16 * 4,
          "S=1024 plan must count the DINOv3 attention scores");
    CHECK(pixal3d_cond_slat_plan(P(1024, 64, 512, 0), 1000, true).largest_bytes == 1024ULL * 81 * 4096 * 4,
          "unchunked S=1024 plan must count the NAF value windows");
    const Pixal3dCondSlatPlan t = pixal3d_cond_slat_plan(P(1024, 64, 1024, 0), 1000, true);
    CHECK(t.chunked, "T=1024 must always take the chunked NAF path (a full [1024, 1024^2] map is 4 GiB)");
    CHECK(pixal3d_cond_slat_plan(P(1024, 64, 1024, 0), 1LL << 31, true).largest_bytes == (uint64_t)1024 * (1ULL << 31) * 4,
          "plan arithmetic must stay in uint64 (no 32-bit wrap)");
}

static bool exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

static std::vector<C3> load_coords(const std::string& path) {   // [N,4] (b,x,y,z) or [N,3]
    auto a = npy::load_i32(path);
    const int w = (int)a.shape.back();
    if (a.shape.size() != 2 || (w != 3 && w != 4)) throw std::runtime_error(path + ": expected [N,3|4]");
    std::vector<C3> v((size_t)a.shape[0]);
    for (size_t i = 0; i < v.size(); ++i)
        for (int k = 0; k < 3; ++k) v[i][k] = a.data[i * w + (w - 3) + k];
    return v;
}

static void fixture(const std::string& dir, const std::string& models, int gpu) {
    const std::string occ = dir + "/occ_logits.npy";
    for (int R : {32, 64}) {
        const std::string sd = dir + "/ss" + std::to_string(R);
        if (!exists(sd + "/ss_coords.npy")) { printf("[fixture] %s: absent, skipped\n", sd.c_str()); continue; }
        printf("[fixture] ss_res=%d (%s)\n", R, sd.c_str());
        auto ref_ss = load_coords(sd + "/ss_coords.npy");
        if (exists(occ)) {
            auto l = npy::load(occ);
            auto mine = ss_coords(l.data, 64, R);
            CHECK(mine == ref_ss, "ss_res=%d: ss_coords %zu != reference %zu", R, mine.size(), ref_ss.size());
            printf("  ss_coords: %zu voxels, max %d (reference %zu)\n", mine.size(), max_coord(mine), ref_ss.size());
        }
        CHECK(max_coord(ref_ss) < R, "ss_res=%d: reference SS coord %d outside the grid", R, max_coord(ref_ss));

        auto slat = npy::load(sd + "/lr_slat.npy");   // [N,32] denormalized == ggml [32,N]
        Model m = Model::load(models + "/shape_dec.gguf", gpu);
        auto up = shape_upsample(m, slat.data, ref_ss);
        m.free();
        auto ref_up = load_coords(sd + "/upsampled_coords.npy");
        // shape_upsample's subdivision masks threshold fp16-decoder logits, so a handful of
        // near-zero children flip between backends (also at ss_res=32, i.e. pre-existing): the raw
        // multiset is held to a relative symmetric difference, the domain/quantizer exactly.
        // Measured on CUDA/q8_0 2026-09-24: 5.6e-4 at ss_res=32 (unchanged pre-#80 path), 6.6e-4
        // at 64; the resulting HR token sets differ by 2 / 0 tokens.
        auto su = up, sr = ref_up;
        std::sort(su.begin(), su.end()); std::sort(sr.begin(), sr.end());
        std::vector<C3> diff;
        std::set_symmetric_difference(su.begin(), su.end(), sr.begin(), sr.end(), std::back_inserter(diff));
        const double rel_up = (double)diff.size() / (double)sr.size();
        CHECK(rel_up <= 1e-3, "ss_res=%d: upsampled coords differ in %zu of %zu (rel %.2e > 1e-3)", R, diff.size(), sr.size(), rel_up);
        CHECK(max_coord(up) < pixal3d_ss_source_span(R), "ss_res=%d: upsampled coord outside the span", R);
        printf("  upsampled: %zu coords, max %d, span %d (reference %zu, symmetric diff %zu = %.2e)\n",
               up.size(), max_coord(up), pixal3d_ss_source_span(R), ref_up.size(), diff.size(), rel_up);
        {
            auto qn = pixal3d_quantize_hr_coords(up, R, 1024);
            auto ref_q = load_coords(sd + "/hr1024_coords.npy");
            std::vector<C3> dq;
            std::set_symmetric_difference(qn.begin(), qn.end(), ref_q.begin(), ref_q.end(), std::back_inserter(dq));
            const double rel_q = (double)dq.size() / (double)ref_q.size();
            CHECK(rel_q <= 1e-3, "ss_res=%d: native-upsample hr1024 tokens differ in %zu of %zu", R, dq.size(), ref_q.size());
            printf("  hr1024 from native upsample: %zu tokens (reference %zu, symmetric diff %zu = %.2e)\n",
                   qn.size(), ref_q.size(), dq.size(), rel_q);
        }

        auto q = pixal3d_quantize_hr_coords(ref_up, R, 1024);
        auto ref_q = load_coords(sd + "/hr1024_coords.npy");
        CHECK(q == ref_q, "ss_res=%d: hr1024 coords %zu != reference %zu", R, q.size(), ref_q.size());
        CHECK(max_coord(q) <= 63, "ss_res=%d: hr1024 max coord %d > 63", R, max_coord(q));
        printf("  hr1024: %zu tokens, max %d (reference %zu)\n", q.size(), max_coord(q), ref_q.size());

        if (exists(sd + "/hr1536_coords.npy")) {
            auto meta = npy::load_i32(sd + "/hr1536_meta.npy");   // (max_tokens, selected_res)
            auto r = pixal3d_cascade_select(ref_up, R, 1536, meta.data[0], true);
            auto ref_r = load_coords(sd + "/hr1536_coords.npy");
            CHECK(r.hr_res == meta.data[1] && r.coords == ref_r, "ss_res=%d: 1536 backoff res %d/%zu != reference %d/%zu",
                  R, r.hr_res, r.coords.size(), meta.data[1], ref_r.size());
            printf("  1536 cascade (max_tokens %d): res %d, %zu tokens (reference res %d, %zu)\n",
                   meta.data[0], r.hr_res, r.coords.size(), meta.data[1], ref_r.size());
        }
    }
}

int main(int argc, char** argv) {
    std::string dir, models;
    int gpu = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--fixture" && i + 1 < argc) dir = argv[++i];
        else if (a == "--models" && i + 1 < argc) models = argv[++i];
        else if (a == "--gpu" && i + 1 < argc) gpu = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--fixture DIR --models MDIR [--gpu N]]\n", argv[0]); return 2; }
    }
    try {
        synthetic();
        memory_plan();
        if (!dir.empty()) {
            if (models.empty()) { fprintf(stderr, "--fixture needs --models\n"); return 2; }
            fixture(dir, models, gpu);
        }
    } catch (const std::exception& e) {
        printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
    printf(g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
