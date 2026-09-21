// trellis2-mv (TASK-PORT P4) SS MV stage tests.
//   ./build/trellis-test-trellis2-mv-ss [models_dir] [fixture_root] [gpu]
//
// Ports the old track's src/test_trellis2_mv_ss.cpp (~/Downloads/pixal3d-mv-62) onto PR #71's
// sample_flow_multi API. See docs/design/2026-09-21-trellis2-mv-port-hardening.md §P4 for the
// scoping decision this file implements:
//
// The old track's include/trellis2_mv_ss.h declared 5 functions/types; only 4 of them
// (trellis2_mv_ss_transpose_for_decode / trellis2_mv_sample_ss_flow / trellis2_mv_decode_ss /
// trellis2_mv_sample_ss_core) are ever called by its test file -- the 5th, the models_dir
// production entry point trellis2_mv_sample_ss() that depends on the old track's
// Trellis2MvCondBank type, is unused here. #71 has no equivalent trellis2_mv_ss.h/.cpp module:
// its own SS stage (trellis_cli.cpp's `trellis_run()`, ~line 906-920) is a plain sequence of
// existing shared calls (Model::load, make_dense_runner, sample_flow_multi via a local
// `sample_bank` lambda, ss_decode, ss_coords) inlined directly in that function, not behind a
// reusable API. Rather than add a new, unused-in-production "trellis2_mv_ss" module that would
// only be a parallel reimplementation of #71's own SS stage (and therefore not actually validate
// it), the 4 functions this test needs are kept as file-local `static` helpers here, built only
// from #71's already-shared building blocks (sample_flow_multi, ss_decode, ss_coords).
// trellis_cli.cpp is NOT modified by this commit.
//
// Consequence (documented, not hidden): this test exercises the SAME sampler math and decoder
// #71's production SS stage uses, but NOT #71's own CLI wiring, its real DitRunner/ss_flow.gguf
// forward pass, or its cond-bank construction -- every run below drives sample_flow_multi with
// either a synthetic FakeModel or a ReplayModel that plays back pre-recorded per-step tensors
// from a real-weight fixture (never a live DiT forward). This machine has ss_dec.gguf but not a
// plain (non-Pixal3D) ss_flow.gguf checkpoint under either GGUF set
// (~/nfs/weights/pixal3d/gguf-q8_0 only has pixal3d_ss_flow_mv.gguf, a different,
// proj_attn-bearing checkpoint) -- the same constraint the old track's test file documented for
// its own environment -- so no test here can drive a real DitRunner either way.
//
// This machine has ss_dec.gguf (SS decoder, shared between Pixal3D/TRELLIS.2) -- every test here
// drives the 4 static helpers with a fake or real-fixture *replay* FlowFwd, never a real
// DitRunner.
//
// fixture_root defaults to ~/nfs/pixal3d_trellis2mv_ref_v2 -- the v2 golden fixture (confirmed
// present, all 6 run directories, this session). A missing run directory or required file is a
// hard FAIL here (not a silent skip): TASK-PORT's parity claim is "6 runs all pass", so a test
// that can report ALL PASS while having run 0 of them would defeat that claim (codex exec review
// finding, 2026-09-21).
#include "flow_runner.h"
#include "trellis_model.h"
#include "ss_decoder.h"
#include "npy.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trellis;
namespace fs = std::filesystem;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// =================================================================================================
// File-local SS-stage helpers (see the file header comment for why these are static here, not a
// new production module). TRELLIS.2 SS flow is always a dense 16^3 grid, 8 channels -- fixed, not
// a stage parameter (unlike shape/texture SLat, which are sparse and view/coord-count dependent).
// =================================================================================================
namespace {
constexpr int kSsInCh       = 8;
constexpr int kSsRes        = 16;
constexpr int kSsL          = kSsRes * kSsRes * kSsRes;  // 4096
constexpr int kSsLatentSize = kSsInCh * kSsL;            // 32768

struct Trellis2MvSsResult {
    std::vector<float> latent;              // fused SS latent, native layout [8,L] (c + 8*sp)
    std::vector<std::array<int,3>> coords;  // ss_coords() output, unmodified
};

// Reorders the sampler's native layout [8,L] (c + 8*sp) into the SS decoder's expected torch
// memory layout [8,16,16,16] (c*L+sp) -- the same transpose trellis_cli.cpp's SS block performs
// inline (`zdec[c*4096+sp] = z[c+8*sp]`, trellis_cli.cpp ~:914). Pure/allocation-only, no model
// dependency. Throws std::runtime_error if `latent.size() != kSsLatentSize`.
std::vector<float> trellis2_mv_ss_transpose_for_decode(const std::vector<float>& latent) {
    if ((int)latent.size() != kSsLatentSize)
        throw std::runtime_error("trellis2_mv_ss_transpose_for_decode: expected " +
                                 std::to_string(kSsLatentSize) + " elements, got " +
                                 std::to_string(latent.size()));
    std::vector<float> zdec(kSsLatentSize);
    for (int c = 0; c < kSsInCh; ++c)
        for (int sp = 0; sp < kSsL; ++sp) zdec[(size_t)c * kSsL + sp] = latent[(size_t)c + kSsInCh * sp];
    return zdec;
}

// Fuses `conds` (unaveraged -- issue #58's prohibition on averaging DINO tokens/latents/coords/
// meshes as a fusion substitute) via `mode`, returns the fused SS latent in native [8,L] layout
// (c + 8*sp), pre-transpose/pre-decode. `fwd` is an already-constructed forward functor (a
// DitRunner::forward wrapper in production, a fake/replay model in tests).
//
// `noise` must be exactly kSsLatentSize elements, already in that same native layout. Throws
// std::runtime_error if `noise.size() != kSsLatentSize`.
//
// SS never shares a stochastic-counter with another stage (unlike shape SLat's LR->HR cascade),
// so this always starts stochastic mode at view 0 via a fresh local counter closed over this call.
std::vector<float> trellis2_mv_sample_ss_flow(const FlowFwd& fwd,
                                              const std::vector<const float*>& conds,
                                              const float* neg_cond, MultiCondMode mode,
                                              const SamplerParams& sp, std::vector<float> noise,
                                              std::vector<std::vector<float>>* trace = nullptr) {
    if ((int)noise.size() != kSsLatentSize)
        throw std::runtime_error("trellis2_mv_sample_ss_flow: noise must have " +
                                 std::to_string(kSsLatentSize) + " elements, got " +
                                 std::to_string(noise.size()));
    int counter = 0;
    return sample_flow_multi(fwd, std::move(noise), conds, neg_cond, sp, mode, trace, &counter);
}

// Calls the existing, unmodified ss_decode()/ss_coords() -- adds no new decode logic.
Trellis2MvSsResult trellis2_mv_decode_ss(const Model& ss_dec_model, const std::vector<float>& zdec) {
    Trellis2MvSsResult r;
    std::vector<float> logits = ss_decode(ss_dec_model, zdec);
    r.coords = ss_coords(logits, 64, 32);
    return r;
}

// Composes sample -> transpose -> decode in one call.
Trellis2MvSsResult trellis2_mv_sample_ss_core(const FlowFwd& fwd, const Model& ss_dec_model,
                                              const std::vector<const float*>& conds,
                                              const float* neg_cond, MultiCondMode mode,
                                              const SamplerParams& sp, std::vector<float> noise,
                                              std::vector<std::vector<float>>* trace = nullptr) {
    std::vector<float> latent = trellis2_mv_sample_ss_flow(fwd, conds, neg_cond, mode, sp, std::move(noise), trace);
    std::vector<float> zdec = trellis2_mv_ss_transpose_for_decode(latent);
    Trellis2MvSsResult r = trellis2_mv_decode_ss(ss_dec_model, zdec);
    r.latent = std::move(latent);
    return r;
}
} // namespace

// ---------------------------------------------------------------------------------------------
// Deterministic fake model, same formula as P3's src/test_flow_multi.cpp FakeModel /
// tools/gen_trellis2_mv_sampler_fixture.py make_fake_model, sized to the SS latent. Structural/
// wiring tests only -- no fixture data involved.
// ---------------------------------------------------------------------------------------------
struct CallEntry { int view; float t; };   // view == -1 means the call used neg_cond

struct FakeModel {
    int dim = kSsLatentSize;
    std::vector<CallEntry> log;

    static int decode_tag(const float* cond) {
        const float val = cond[0];
        if (std::fabs(val) < 1e-6f) return -1;
        return (int)std::lround(val / 10.0f) - 1;
    }
    std::vector<float> operator()(const std::vector<float>&, float tscaled, const float* cond) {
        const float t_val = tscaled / 1000.0f;
        const int view = decode_tag(cond);
        log.push_back({ view, t_val });
        float scale, offset;
        if (view < 0) { scale = 0.2f; offset = 0.0f; }
        else { scale = 1.0f + 0.4f * view; offset = 0.5f * (view + 1); }
        std::vector<float> out(dim);
        for (int k = 0; k < dim; ++k) out[k] = offset + scale * std::sin(3.0f * t_val + (float)k);
        return out;
    }
};
static std::vector<float> make_cond_buf(int view) { return { (float)((view + 1) * 10) }; }
static std::vector<float> make_neg_buf() { return { 0.0f }; }
static std::vector<int> positive_views(const std::vector<CallEntry>& log) {
    std::vector<int> v; for (const auto& e : log) if (e.view >= 0) v.push_back(e.view); return v;
}
static std::vector<float> make_deterministic_noise(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = std::sin(0.017f * i + 0.3f) * 0.6f;
    return v;
}
// Production sparse_structure_sampler params (docs/spec/33-trellis2-mv-reference.md §3 in the old
// track; this tree has not ported that catalog doc, values kept verbatim from it).
static SamplerParams ss_production_params(int steps = 12) {
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 7.5f; sp.guidance_rescale = 0.7f;
    sp.gi0 = 0.6; sp.gi1 = 1.0; sp.rescale_t = 5.0;
    return sp;
}

// ---------------------------------------------------------------------------------------------
// §1: pure / structural tests (no fixture data)
// ---------------------------------------------------------------------------------------------
static void test_transpose_pure() {
    printf("-- trellis2_mv_ss_transpose_for_decode: index-pattern round trip --\n");
    std::vector<float> latent(kSsLatentSize);
    for (int c = 0; c < kSsInCh; ++c)
        for (int sp = 0; sp < kSsL; ++sp) latent[c + kSsInCh * sp] = (float)(c * 100000 + sp);
    std::vector<float> zdec = trellis2_mv_ss_transpose_for_decode(latent);
    bool ok = true;
    for (int c = 0; c < kSsInCh && ok; ++c)
        for (int sp = 0; sp < kSsL && ok; ++sp)
            if (zdec[(size_t)c * kSsL + sp] != (float)(c * 100000 + sp)) ok = false;
    check(ok, "every (channel, spatial) element lands at torch [c*L+sp] unchanged");

    bool threw = false;
    try { trellis2_mv_ss_transpose_for_decode(std::vector<float>(10)); }
    catch (const std::exception&) { threw = true; }
    check(threw, "wrong-size latent throws instead of reading/writing out of bounds");
}

static void test_noise_size_validation() {
    printf("-- trellis2_mv_sample_ss_flow: noise size validation --\n");
    FakeModel fm;
    auto cond = make_cond_buf(0), neg = make_neg_buf();
    std::vector<const float*> conds = { cond.data() };
    FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* c) { return fm(x, t, c); };
    bool threw = false;
    try {
        trellis2_mv_sample_ss_flow(fwd, conds, neg.data(), MultiCondMode::Stochastic,
                                   ss_production_params(), std::vector<float>(10));
    } catch (const std::exception&) { threw = true; }
    check(threw, "a 10-element noise vector (not kSsLatentSize) is rejected at the entrance");
}

// "V=1 MV stage == current single-image stage": this machine has no ss_flow.gguf either way, so
// the comparison target is the real, unmodified `sample_flow` (not a reimplementation) driven by
// the same fake FlowFwd/noise/SamplerParams -- checks that trellis2_mv_sample_ss_core's own
// transpose/decode wiring doesn't diverge from what the single-image path would compute, at
// production SS sampler parameters.
static void test_v1_bit_exact_vs_sample_flow(const Model& ss_dec) {
    printf("-- V=1 core vs real sample_flow(): bit-exact latent + exact coords --\n");
    auto noise = make_deterministic_noise(kSsLatentSize);
    auto cond = make_cond_buf(0), neg = make_neg_buf();
    std::vector<const float*> conds = { cond.data() };
    const SamplerParams sp = ss_production_params();

    for (MultiCondMode mode : { MultiCondMode::Stochastic, MultiCondMode::MultiDiffusion }) {
        FakeModel fm_core, fm_ref;
        FlowFwd fwd_core = [&](const std::vector<float>& x, float t, const float* c) { return fm_core(x, t, c); };
        FlowFwd fwd_ref  = [&](const std::vector<float>& x, float t, const float* c) { return fm_ref(x, t, c); };

        Trellis2MvSsResult core = trellis2_mv_sample_ss_core(fwd_core, ss_dec, conds, neg.data(),
                                                              mode, sp, noise);
        std::vector<float> ref_latent = sample_flow(fwd_ref, noise, cond.data(), neg.data(), sp);
        std::vector<float> ref_zdec = trellis2_mv_ss_transpose_for_decode(ref_latent);
        std::vector<float> ref_logits = ss_decode(ss_dec, ref_zdec);
        std::vector<std::array<int,3>> ref_coords = ss_coords(ref_logits, 64, 32);

        const bool latent_eq = core.latent.size() == ref_latent.size() &&
            std::memcmp(core.latent.data(), ref_latent.data(), core.latent.size() * sizeof(float)) == 0;
        check(latent_eq, std::string("mode=") + (mode == MultiCondMode::Stochastic ? "stochastic" : "multidiffusion") +
              ": V=1 latent bit-exact vs sample_flow()");
        check(core.coords == ref_coords, std::string("mode=") + (mode == MultiCondMode::Stochastic ? "stochastic" : "multidiffusion") +
              ": V=1 decoded coords match sample_flow()+ss_decode()+ss_coords()");
    }
}

static void test_view_reorder() {
    printf("-- swapping cond order reverses the stochastic view schedule --\n");
    auto noise = make_deterministic_noise(kSsLatentSize);
    auto neg = make_neg_buf();
    const SamplerParams sp = ss_production_params(6);   // 6 steps is enough to see the pattern

    auto run = [&](int tagA, int tagB) {
        FakeModel fm;
        auto bufA = make_cond_buf(tagA), bufB = make_cond_buf(tagB);
        std::vector<const float*> conds = { bufA.data(), bufB.data() };
        FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* c) { return fm(x, t, c); };
        trellis2_mv_sample_ss_flow(fwd, conds, neg.data(), MultiCondMode::Stochastic, sp, noise);
        return positive_views(fm.log);
    };
    const std::vector<int> order1 = run(0, 1);   // conds = [tag(view0), tag(view1)]
    const std::vector<int> order2 = run(1, 0);   // conds = [tag(view1), tag(view0)]  -- swapped
    check(order1 == std::vector<int>({0,1,0,1,0,1}), "conds=[A,B]: schedule is A,B,A,B,...");
    check(order2 == std::vector<int>({1,0,1,0,1,0}), "conds=[B,A]: schedule is B,A,B,A,... (reversed)");
}

static void test_duplicate_views() {
    printf("-- multidiffusion: duplicate views collapse to the V=1 result --\n");
    auto noise = make_deterministic_noise(kSsLatentSize);
    auto cond = make_cond_buf(0), neg = make_neg_buf();
    const SamplerParams sp = ss_production_params();

    FakeModel fm1;
    std::vector<const float*> conds1 = { cond.data() };
    FlowFwd fwd1 = [&](const std::vector<float>& x, float t, const float* c) { return fm1(x, t, c); };
    std::vector<float> single = trellis2_mv_sample_ss_flow(fwd1, conds1, neg.data(),
                                                            MultiCondMode::MultiDiffusion, sp, noise);

    FakeModel fm2;
    std::vector<const float*> conds2 = { cond.data(), cond.data() };
    FlowFwd fwd2 = [&](const std::vector<float>& x, float t, const float* c) { return fm2(x, t, c); };
    std::vector<float> dup = trellis2_mv_sample_ss_flow(fwd2, conds2, neg.data(),
                                                         MultiCondMode::MultiDiffusion, sp, noise);

    const bool eq = single.size() == dup.size() &&
        std::memcmp(single.data(), dup.data(), single.size() * sizeof(float)) == 0;
    check(eq, "conds=[A,A] multidiffusion latent bit-exact vs conds=[A] alone");
}

// "no active coords remains a normal explicit failure": ss_coords()'s own contract (empty vector,
// no throw, for all-non-positive logits) is pre-existing/unmodified; this pins that
// trellis2_mv_decode_ss's pass-through doesn't add an exception in between, using the REAL
// ss_dec.gguf (a handful of far-out-of-distribution constant latents empirically probed until one
// actually decodes to zero active voxels).
static void test_explicit_empty_coords(const Model& ss_dec) {
    printf("-- explicit empty-coords path (no active coords is not an exception) --\n");
    const std::vector<float> candidates = { 0.0f, -1.0f, -5.0f, -20.0f, -100.0f, 100.0f };
    bool found_empty = false;
    float which = 0.0f;
    for (float v : candidates) {
        std::vector<float> zdec(kSsLatentSize, v);
        Trellis2MvSsResult r;
        bool threw = false;
        try { r = trellis2_mv_decode_ss(ss_dec, zdec); }
        catch (const std::exception&) { threw = true; }
        check(!threw, "constant latent (" + std::to_string(v) + "): trellis2_mv_decode_ss does not throw");
        if (!threw && r.coords.empty() && !found_empty) { found_empty = true; which = v; }
    }
    check(found_empty, found_empty
          ? ("at least one candidate constant latent decodes to coords.empty()==true (found: " + std::to_string(which) + ")")
          : "at least one candidate constant latent decodes to coords.empty()==true (none of the probed candidates did -- widen the candidate list)");
}

// SS's own NaN/Inf safety net is sample_flow_multi's; this pins that the static helpers above
// don't defeat it.
static void test_nan_inf_safety(const Model& ss_dec) {
    printf("-- Core propagates sample_flow_multi's NaN/Inf safety net --\n");
    if (getenv("TRELLIS_NOFIX")) {
        printf("SKIP (TRELLIS_NOFIX=1 explicitly disables this exact safety net for A/B testing --\n"
               "      this test's premise doesn't hold under it; run without TRELLIS_NOFIX to check it)\n");
        return;
    }
    auto noise = make_deterministic_noise(kSsLatentSize);
    auto cond = make_cond_buf(0), neg = make_neg_buf();
    std::vector<const float*> conds = { cond.data() };
    const SamplerParams sp = ss_production_params();
    FlowFwd fwd_bad = [&](const std::vector<float>&, float tscaled, const float*) {
        const float v = (tscaled / 1000.0f < 0.5f) ? std::numeric_limits<float>::infinity() : 1.0f;
        return std::vector<float>(kSsLatentSize, v);
    };
    Trellis2MvSsResult result = trellis2_mv_sample_ss_core(fwd_bad, ss_dec, conds, neg.data(),
                                                            MultiCondMode::MultiDiffusion, sp, noise);
    bool all_finite = true;
    for (float v : result.latent) if (!std::isfinite(v)) { all_finite = false; break; }
    check(all_finite, "an Inf-returning fwd() still yields a fully finite latent");
}

// ---------------------------------------------------------------------------------------------
// §2: real v2 fixture replay tests. No ss_flow.gguf is needed: a ReplayModel returns the exact
// recorded per-view/negative model predictions in the exact order the fixture's own manifest
// shows they were called, so sample_flow_multi's CFG/rescale/Euler-integration code runs on real
// TRELLIS.2 numbers at production SS parameters. This validates:
//   (a) the call pattern (how many positive calls, whether/when negative is called) matches the
//       real reference exactly -- not just synthetic self-consistency;
//   (b) the per-step CFG-blended prediction sample_flow_multi computes (recovered from
//       consecutive trace entries) matches the fixture's recorded ss_step*_cfg.npy;
//   (c) the final decoded coordinate set matches ss_coords.npy exactly.
// ---------------------------------------------------------------------------------------------
static std::vector<float> to_native_ss(const std::vector<float>& torch_layout) {
    if ((int)torch_layout.size() != kSsLatentSize)
        throw std::runtime_error("to_native_ss: expected " + std::to_string(kSsLatentSize) +
                                 " elements, got " + std::to_string(torch_layout.size()));
    std::vector<float> native(kSsLatentSize);
    for (int c = 0; c < kSsInCh; ++c)
        for (int sp = 0; sp < kSsL; ++sp) native[c + kSsInCh * sp] = torch_layout[(size_t)c * kSsL + sp];
    return native;
}

// Replays the fixture's recorded per-call tensors in the exact order a filesystem scan of
// ss_step{k}_pos{i}.npy / ss_step{k}_neg.npy finds them (positive calls first, in ordinal order,
// then the negative call if present). Throws if the sampler calls more times than recorded; a
// post-run size/mismatch check (in the caller) catches under-calling and pos/neg branch mismatches.
struct ReplayModel {
    std::vector<std::vector<float>> tensors;    // native layout, aligned with is_neg_expected
    std::vector<bool> is_neg_expected;
    const float* neg_ptr = nullptr;
    size_t next = 0;
    int mismatches = 0;
    std::vector<float> operator()(const std::vector<float>&, float, const float* cond) {
        if (next >= tensors.size())
            throw std::runtime_error("ReplayModel: sampler called more times than the " +
                                     std::to_string(tensors.size()) + " calls recorded in the fixture");
        if ((cond == neg_ptr) != is_neg_expected[next]) ++mismatches;
        return tensors[next++];
    }
};

static bool build_replay(const std::string& dir, int steps, ReplayModel& rm, std::string& err) {
    for (int k = 0; k < steps; ++k) {
        int ord = 0;
        for (;; ++ord) {
            const std::string path = dir + "/ss_step" + std::to_string(k) + "_pos" + std::to_string(ord) + ".npy";
            if (!fs::exists(path)) break;
            npy::Array a = npy::load(path);
            if ((int)a.numel() != kSsLatentSize) { err = "unexpected shape: " + path; return false; }
            rm.tensors.push_back(to_native_ss(a.data));
            rm.is_neg_expected.push_back(false);
        }
        const std::string neg_path = dir + "/ss_step" + std::to_string(k) + "_neg.npy";
        if (fs::exists(neg_path)) {
            npy::Array a = npy::load(neg_path);
            if ((int)a.numel() != kSsLatentSize) { err = "unexpected shape: " + neg_path; return false; }
            rm.tensors.push_back(to_native_ss(a.data));
            rm.is_neg_expected.push_back(true);
        }
    }
    if (rm.tensors.empty()) { err = "no ss_step*_pos*/neg.npy files found in " + dir; return false; }
    return true;
}

// Returns true iff the run actually executed against present fixture data (a missing run
// directory or required file is a hard FAIL via check(), not a silent skip -- see the file header
// comment on why "6 runs all present" is asserted, not assumed).
static bool test_fixture_replay(const std::string& fixture_root, const std::string& run_name,
                                int V, MultiCondMode mode, const Model& ss_dec) {
    const std::string dir = fixture_root + "/" + run_name;
    const std::string tag = run_name + " (V=" + std::to_string(V) + ", " +
        (mode == MultiCondMode::Stochastic ? "stochastic" : "multidiffusion") + ")";
    printf("-- fixture replay: %s --\n", tag.c_str());
    if (!fs::exists(dir)) {
        check(false, tag + ": required fixture run directory is present (" + dir + ")");
        return false;
    }
    if (!fs::exists(dir + "/noise_ss_0.npy") || !fs::exists(dir + "/ss_coords.npy")) {
        check(false, tag + ": required noise_ss_0.npy and ss_coords.npy are present in " + dir);
        return false;
    }

    npy::Array noise = npy::load(dir + "/noise_ss_0.npy");
    if ((int)noise.numel() != kSsLatentSize) { check(false, tag + ": noise_ss_0.npy has the SS latent size"); return false; }
    const std::vector<float> noise_native = to_native_ss(noise.data);

    npy::ArrayI32 coords_raw = npy::load_i32(dir + "/ss_coords.npy");
    if (!(coords_raw.shape.size() == 2 && coords_raw.shape[1] == 4)) {
        check(false, tag + ": ss_coords.npy has shape [N,4] (torch SparseTensor [batch,x,y,z])");
        return false;
    }
    std::set<std::array<int,3>> coords_gold;
    for (int64_t i = 0; i < coords_raw.shape[0]; ++i)
        coords_gold.insert({ coords_raw.data[i*4+1], coords_raw.data[i*4+2], coords_raw.data[i*4+3] });

    const int steps = 12;   // SS captures all 12 production steps
    ReplayModel rm;
    std::string err;
    if (!build_replay(dir, steps, rm, err)) { check(false, tag + ": " + err); return false; }

    std::vector<float> neg_buf{0.0f};
    std::vector<std::vector<float>> cond_bufs(V, std::vector<float>{0.0f});   // content unused by ReplayModel
    std::vector<const float*> conds;
    for (auto& b : cond_bufs) conds.push_back(b.data());
    rm.neg_ptr = neg_buf.data();

    FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* c) { return rm(x, t, c); };
    const SamplerParams sp = ss_production_params(steps);

    std::vector<std::vector<float>> trace;
    Trellis2MvSsResult result;
    try {
        result = trellis2_mv_sample_ss_core(fwd, ss_dec, conds, neg_buf.data(), mode, sp, noise_native, &trace);
    } catch (const std::exception& e) {
        check(false, tag + ": replay run threw: " + e.what());
        return false;
    }

    check(rm.next == rm.tensors.size(), tag + ": consumed every recorded pos/neg call, none left over (" +
          std::to_string(rm.next) + "/" + std::to_string(rm.tensors.size()) + ")");
    check(rm.mismatches == 0, tag + ": every call's pos/neg branch matches the recorded call pattern");
    check((int)trace.size() == steps, tag + ": trace has one entry per step");

    std::set<std::array<int,3>> coords_mine;
    for (auto& c : result.coords) coords_mine.insert(c);
    // Duplicate coords in result.coords would hide behind the set-conversion above; catch that
    // structurally regardless of strict/production mode (codex exec review finding).
    check(result.coords.size() == coords_mine.size(),
          tag + ": decoded coords has no duplicates (raw=" + std::to_string(result.coords.size()) +
          ", set=" + std::to_string(coords_mine.size()) + ")");

    // sample_flow_multi shares apply_guidance_rescale() (src/flow_runner.cpp) with the existing
    // single-image sample_flow(); by default it clamps the CFG-rescale variance ratio to
    // [0.2, 5.0] as an OOD/NaN robustness guard that predates this stage and is NOT part of the
    // pinned TRELLIS.2 reference algorithm. Under TRELLIS_NOFIX=1 that clamp is disabled and every
    // run below is expected to match the pinned reference to ~1e-7 relative (per-step pred) and an
    // exact coordinate set. That is this test's actual "passes the #59/v2 numerical/coord gates"
    // claim (issue #77 in the old track's numbering; this tree has not ported that issue): run
    // with TRELLIS_NOFIX=1 for the strict gate. Without it, the two checks below report the same
    // real, expected, root-caused divergence as informational only (not a stage bug -- this clamp
    // is shared, pre-existing infrastructure) so default runs of this binary don't spuriously fail
    // on inherited behavior.
    const bool strict = getenv("TRELLIS_NOFIX") != nullptr;

    const std::vector<double> ts = flow_t_schedule(sp.steps, sp.rescale_t);
    double max_rel = 0.0;
    int cfg_files_checked = 0;
    if ((int)trace.size() == steps) {
        for (int k = 0; k < steps; ++k) {
            const std::string cfg_path = dir + "/ss_step" + std::to_string(k) + "_cfg.npy";
            if (!fs::exists(cfg_path)) continue;
            npy::Array cfg = npy::load(cfg_path);
            if ((int)cfg.numel() != kSsLatentSize) continue;
            const std::vector<float>& prev = (k == 0) ? noise_native : trace[k - 1];
            const double dt = ts[k] - ts[k + 1];
            std::vector<float> pred_native(kSsLatentSize);
            for (int i = 0; i < kSsLatentSize; ++i)
                pred_native[i] = (float)(((double)prev[i] - (double)trace[k][i]) / dt);
            std::vector<float> pred_torch = trellis2_mv_ss_transpose_for_decode(pred_native);
            double maxd = 0, gmax = 0;
            for (int i = 0; i < kSsLatentSize; ++i) {
                const double d = std::fabs((double)pred_torch[i] - (double)cfg.data[i]);
                maxd = std::max(maxd, d); gmax = std::max(gmax, std::fabs((double)cfg.data[i]));
            }
            max_rel = std::max(max_rel, maxd / std::max(gmax, 1e-12));
            if (getenv("TRELLIS_DBG_SS_PRED"))
                fprintf(stderr, "  [predcheck] step %2d dt=%.6f rel=%.4e maxd=%.4e gmax=%.4e\n",
                        k, dt, maxd / std::max(gmax, 1e-12), maxd, gmax);
            ++cfg_files_checked;
        }
    }
    // Meta-check (always asserted, strict or not): did this run actually exercise all 12 recorded
    // CFG files. This is not a numerical parity claim -- it's "did the test do what it says it did"
    // (codex exec review finding: without this, a fixture missing ss_step*_cfg.npy files would
    // silently under-check and still print ALL PASS).
    check(cfg_files_checked == steps,
          tag + ": all " + std::to_string(steps) + " ss_step*_cfg.npy files were found and checked (checked=" +
          std::to_string(cfg_files_checked) + ")");

    char relbuf[64]; snprintf(relbuf, sizeof relbuf, "%.4e", max_rel);
    const std::string cfg_msg = tag + ": per-step CFG-blended pred matches ss_step*_cfg.npy (max_rel=" +
        relbuf + ")" + (strict ? "" : " [informational -- rerun with TRELLIS_NOFIX=1 for the strict gate]");
    if (strict) check(max_rel < 1e-3, cfg_msg);
    else printf("%s %s\n", (max_rel < 1e-3) ? "ok  " : "info", cfg_msg.c_str());

    const std::string coords_msg = tag + ": decoded coord set matches ss_coords.npy exactly (mine=" +
        std::to_string(coords_mine.size()) + ", gold=" + std::to_string(coords_gold.size()) + ")" +
        (strict ? "" : " [informational -- rerun with TRELLIS_NOFIX=1 for the strict gate]");
    if (strict) check(coords_mine == coords_gold, coords_msg);
    else printf("%s %s\n", (coords_mine == coords_gold) ? "ok  " : "info", coords_msg.c_str());

    // Production (clamp ON) regression pin: the two checks above are informational under default
    // settings because the divergence they'd otherwise flag is real, expected, and caused by
    // shared infrastructure (see the `strict` comment above) -- but "informational" must not mean
    // "untested". These asserted checks quantify that same known divergence as a genuine
    // regression guard: if the clamp stops firing (min_ratio no longer < 0.2), the divergence it
    // causes vanishes or changes magnitude, which is worth re-investigating, not a silent pass. If
    // they instead blow WAY past these bounds, something is newly and more badly broken.
    //
    // NOTE (TASK-PORT P4 scope, codex exec review): these bounds are a PIN of a divergence the old
    // track measured on its own machine/run, not a failure-mode detector validated against known
    // bad examples -- see docs/design/2026-09-21-trellis2-mv-port-hardening.md §P4 ("codex追加指摘
    // のうち採用しないもの"). They are not re-derived or re-validated here.
    if (!strict) {
        int sym_diff = 0;
        for (auto& c : coords_mine) if (!coords_gold.count(c)) ++sym_diff;
        for (auto& c : coords_gold) if (!coords_mine.count(c)) ++sym_diff;

        double min_ratio = std::numeric_limits<double>::infinity();
        const double gs = sp.guidance_strength, sm = sp.sigma_min;
        for (int k = 0; k < steps; ++k) {
            if (!flow_in_guidance_interval(ts[k], sp.gi0, sp.gi1)) continue;
            const std::string avg_path = dir + "/ss_step" + std::to_string(k) + "_avg.npy";
            const std::string pos0_path = dir + "/ss_step" + std::to_string(k) + "_pos0.npy";
            const std::string neg_path = dir + "/ss_step" + std::to_string(k) + "_neg.npy";
            const bool use_avg = (V > 1 && mode == MultiCondMode::MultiDiffusion && fs::exists(avg_path));
            const std::string src_path = use_avg ? avg_path : pos0_path;
            if (!fs::exists(src_path) || !fs::exists(neg_path)) continue;
            npy::Array pv = npy::load(src_path), nv = npy::load(neg_path);
            if ((int)pv.numel() != kSsLatentSize || (int)nv.numel() != kSsLatentSize) continue;
            const std::vector<float>& prev_native = (k == 0) ? noise_native : trace[k - 1];
            const std::vector<float> prev = trellis2_mv_ss_transpose_for_decode(prev_native);   // torch layout, matches pv/nv
            const double a = 1.0 - sm, b = sm + (1.0 - sm) * ts[k];
            double mp = 0, mc = 0;
            std::vector<double> x0p(kSsLatentSize), x0c(kSsLatentSize);
            for (int i = 0; i < kSsLatentSize; ++i) {
                const double raw = gs * (double)pv.data[i] + (1.0 - gs) * (double)nv.data[i];
                x0p[i] = a * prev[i] - b * (double)pv.data[i];
                x0c[i] = a * prev[i] - b * raw;
                mp += x0p[i]; mc += x0c[i];
            }
            mp /= kSsLatentSize; mc /= kSsLatentSize;
            double vp = 0, vc = 0;
            for (int i = 0; i < kSsLatentSize; ++i) { vp += (x0p[i]-mp)*(x0p[i]-mp); vc += (x0c[i]-mc)*(x0c[i]-mc); }
            const double ratio = vc > 0 ? std::sqrt(vp / (kSsLatentSize - 1)) / std::sqrt(vc / (kSsLatentSize - 1)) : 1.0;
            min_ratio = std::min(min_ratio, ratio);
        }

        printf("info %s: production diagnostics (sym_diff=%d max_rel=%.4e min_ratio=%.4f)\n",
               tag.c_str(), sym_diff, max_rel, min_ratio);

        // Generic bounds across all 6 real runs (old track observation, 2026-09-21, that machine,
        // CPU backend): min_ratio 0.136-0.19, max_rel 0.099-0.116, sym_diff small (single-digit to
        // low tens). Not re-derived on this machine; ported verbatim as a pin.
        check(min_ratio < 0.2,
              tag + ": production clamp actually fires: min_ratio=" + std::to_string(min_ratio) + " < 0.2");
        check(max_rel > 0.05 && max_rel < 0.20,
              tag + ": production per-step divergence stays in the known band (max_rel=" + std::to_string(max_rel) + ")");
        check(sym_diff >= 1 && sym_diff <= 50,
              tag + ": production coord-set divergence stays in the known band (sym_diff=" + std::to_string(sym_diff) + ")");

        // TASK-PORT F1 (verifier finding, 2026-09-21): the old track additionally pinned a tighter
        // band for this one specific run (old track's own machine, 2026-09-21). Restored verbatim
        // -- this machine's measured values (min_ratio=0.1363, max_rel=0.099183, sym_diff=4) fall
        // inside it.
        if (run_name == "run_1img_baseline_512") {
            check(min_ratio >= 0.13 && min_ratio <= 0.16,
                  tag + ": pinned case -- min_ratio in [0.13,0.16] (got " + std::to_string(min_ratio) + ")");
            check(max_rel >= 0.09 && max_rel <= 0.12,
                  tag + ": pinned case -- max_rel in [0.09,0.12] (got " + std::to_string(max_rel) + ")");
            check(sym_diff >= 2 && sym_diff <= 8,
                  tag + ": pinned case -- coord-set sym_diff near 4 (got " + std::to_string(sym_diff) + ")");
        }
    }
    return true;
}

static int run_all_fixture_replays(const std::string& fixture_root, const Model& ss_dec) {
    int ran = 0;
    ran += test_fixture_replay(fixture_root, "run_1img_baseline_512", 1, MultiCondMode::Stochastic, ss_dec) ? 1 : 0;
    ran += test_fixture_replay(fixture_root, "run_1img_injected_stochastic_512", 1, MultiCondMode::Stochastic, ss_dec) ? 1 : 0;
    ran += test_fixture_replay(fixture_root, "run_2img_real_stochastic_512", 2, MultiCondMode::Stochastic, ss_dec) ? 1 : 0;
    ran += test_fixture_replay(fixture_root, "run_2img_real_multidiffusion_512", 2, MultiCondMode::MultiDiffusion, ss_dec) ? 1 : 0;
    ran += test_fixture_replay(fixture_root, "run_4img_real_stochastic_512", 4, MultiCondMode::Stochastic, ss_dec) ? 1 : 0;
    ran += test_fixture_replay(fixture_root, "run_4img_real_multidiffusion_512", 4, MultiCondMode::MultiDiffusion, ss_dec) ? 1 : 0;
    check(ran == 6, "all 6 v2 fixture runs were present and executed (ran=" + std::to_string(ran) + "/6)");
    return ran;
}

// Internal-only re-exec mode: main() below spawns this binary again with this sentinel argv[1] to
// run the strict parity gate. TRELLIS_NOFIX is force-enabled HERE, in-process, before any
// sample_flow/sample_flow_multi call -- flow_runner.cpp caches it in a function-local
// `static const bool no_fix` on first call, so it cannot be toggled mid-process, and this must not
// depend on whatever environment the outer caller happened to invoke the binary with. A subprocess
// is the only way to guarantee this in the same binary as the production (clamp-ON) checks below,
// which equally must not have TRELLIS_NOFIX set.
static int run_parity_subprocess_mode(int argc, char** argv) {
    setenv("TRELLIS_NOFIX", "1", 1);
    const std::string models_dir = argc > 2 ? argv[2] : "/Users/<redacted-user>/nfs/weights/pixal3d/gguf-q8_0";
    const char* home = getenv("HOME");
    const std::string fixture_root = argc > 3 ? argv[3]
        : (std::string(home ? home : "") + "/nfs/pixal3d_trellis2mv_ref_v2");
    const int gpu = argc > 4 ? atoi(argv[4]) : -1;
    printf("== parity (clamp off): SS MV stage vs pinned reference ==\n");
    Model ss_dec = Model::load(models_dir + "/ss_dec.gguf", gpu);
    run_all_fixture_replays(fixture_root, ss_dec);
    ss_dec.free();
    printf(g_fail == 0 ? "\n[parity subprocess] ALL PASS\n" : "\n[parity subprocess] %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--internal-parity-subprocess")
        return run_parity_subprocess_mode(argc, argv);

    const std::string models_dir = argc > 1 ? argv[1] : "/Users/<redacted-user>/nfs/weights/pixal3d/gguf-q8_0";
    const char* home = getenv("HOME");
    const std::string fixture_root = argc > 2 ? argv[2]
        : (std::string(home ? home : "") + "/nfs/pixal3d_trellis2mv_ref_v2");
    const int gpu = argc > 3 ? atoi(argv[3]) : -1;

    Model ss_dec = Model::load(models_dir + "/ss_dec.gguf", gpu);

    test_transpose_pure();
    test_noise_size_validation();
    test_v1_bit_exact_vs_sample_flow(ss_dec);
    test_view_reorder();
    test_duplicate_views();
    test_nan_inf_safety(ss_dec);
    test_explicit_empty_coords(ss_dec);

    printf("== production (clamp on, default settings): known divergence asserted, not just printed ==\n");
    run_all_fixture_replays(fixture_root, ss_dec);
    ss_dec.free();

    printf("== parity (clamp off): re-exec'ing self with TRELLIS_NOFIX forced internally ==\n");
    {
        const std::string cmd = "\"" + std::string(argv[0]) + "\" --internal-parity-subprocess \"" +
            models_dir + "\" \"" + fixture_root + "\" " + std::to_string(gpu);
        const int rc = system(cmd.c_str());
        check(rc == 0, "parity (clamp off) subprocess exited 0 -- SS MV stage matches the pinned "
              "reference exactly (coords + ~1e-7 per-step) across all 6 real fixture runs");
    }

    printf(g_fail == 0 ? "\nALL PASS\n" : "\n%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
