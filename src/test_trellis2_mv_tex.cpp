// trellis2-mv (TASK-STAGES) texture SLat MV stage tests: step {0, 8, 11}, 512 + 1024cascade.
//   ./build/trellis-test-trellis2-mv-tex [fixture_root] [gpu(未使用)]
//
// docs/design/2026-09-21-trellis2-mv-stages-shape-tex.md 参照。texture の production パラメータ
// (`gs=1.0, gr=0.0, gi=[0.6,0.9]`, trellis_cli.cpp:1091) は CFG 出力が `sample`/`t` に一切
// 依存しない特殊ケース（`gr=0.0` で `apply_guidance_rescale` 自体が呼ばれず -- flow_runner.cpp
// :560/:584 の `if (sp.guidance_rescale > 0.0f)` ガード。かつ `gs=1.0` 固定なので CFG混合は
// 常に `pred = 1.0*pos + 0.0*neg = pos` に代数的に潰れる）。このため shape と違い、
// step0 だけでなく step8/step11 も #71 の本番 `sample_flow_multi` を `sp.steps=1` で直接呼んで
// 検証できる（設計ドキュメント「追記」節）。ただし step8 は実際には guided (t=0.6 相当) だが
// `sp.steps=1` は `t=1.0`（not guided）を強制するため**実際とは異なる分岐**（neg呼び出し無し）
// を通る -- gs=1固定のためその分岐差は出力値に影響しない、という代数的事実に依拠した簡略化で
// あることを明記する（実際の "guided分岐でneg項がゼロ化される" というコード行そのものの
// カバレッジは、この2ファイルのテストでは持たない）。
//
// step0/11 は実際に not-guided (t=1.0 / t≈0 はいずれも [0.6,0.9] 外) なので `sp.steps=1` は
// 分岐まで完全に実際と一致する（tex_step{0,11}に neg.npy が存在しないことで確認済み）。
//
// 64ch concat (trellis_cli.cpp:1082-1090, [state(32);shape_slat_norm(32)]) は ReplayModel が
// forward の入力 x を無視する契約上、fixture replay からは検証できない。純粋な配線の
// structural test を別途行う（test_concat_structure）。
//
// trellis_cli.cpp は変更しない。
#include "flow_runner.h"
#include "npy.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

// Production texture-SLAT sampler params (trellis_cli.cpp:1091).
static SamplerParams tex_production_params(int steps = 12) {
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 1.0f; sp.guidance_rescale = 0.0f;
    sp.gi0 = 0.6; sp.gi1 = 0.9; sp.rescale_t = 3.0;
    return sp;
}

// =================================================================================================
// §1: 64ch concat の structural test（trellis_cli.cpp:1083-1090 の複製、pure data movement）。
// =================================================================================================
namespace {
std::vector<float> build_tex_input_concat(const std::vector<float>& state32, const std::vector<float>& shape_norm32, int N) {
    std::vector<float> x64((size_t)64 * N);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < 32; ++k) x64[(size_t)k + 64*n]      = state32[(size_t)k + 32*n];
        for (int k = 0; k < 32; ++k) x64[(size_t)32 + k + 64*n] = shape_norm32[(size_t)k + 32*n];
    }
    return x64;
}
} // namespace

static void test_concat_structure() {
    printf("-- tex input concat: x64[0:32]=state, x64[32:64]=fixed shape_slat_norm, per token --\n");
    const int N = 5;
    std::vector<float> state(32 * N), shape(32 * N);
    for (size_t i = 0; i < state.size(); ++i) state[i] = (float)(1000 + i);
    for (size_t i = 0; i < shape.size(); ++i) shape[i] = (float)(2000 + i);
    auto x64 = build_tex_input_concat(state, shape, N);
    check((int)x64.size() == 64 * N, "concat output size is 64*N");
    bool ok = true;
    for (int n = 0; n < N && ok; ++n)
        for (int k = 0; k < 32 && ok; ++k) {
            if (x64[(size_t)k + 64*n] != state[(size_t)k + 32*n]) ok = false;
            if (x64[(size_t)32 + k + 64*n] != shape[(size_t)k + 32*n]) ok = false;
        }
    check(ok, "every token's first 32ch == current state, next 32ch == fixed shape_slat_norm");
}

// step0/11 の guidance-interval 外境界の pure な裏付け（fixture 不要）。
static void test_gi_boundaries() {
    printf("-- tex sampler: step0/8/11 guidance-interval membership --\n");
    const SamplerParams sp = tex_production_params();
    const std::vector<double> ts = flow_t_schedule(sp.steps, sp.rescale_t);
    check(!flow_in_guidance_interval(ts[0], sp.gi0, sp.gi1), "step0 (t=" + std::to_string(ts[0]) + ") is outside [0.6,0.9]");
    check(flow_in_guidance_interval(ts[8], sp.gi0, sp.gi1), "step8 (t=" + std::to_string(ts[8]) + ") is inside [0.6,0.9]");
    check(!flow_in_guidance_interval(ts[11], sp.gi0, sp.gi1), "step11 (t=" + std::to_string(ts[11]) + ") is outside [0.6,0.9]");
}

// =================================================================================================
// §2: step{0,8,11} CFG 検証。#71 の本番 sample_flow_multi を sp.steps=1 で直接呼ぶ。
// =================================================================================================
namespace {
struct ReplayModel {
    std::vector<std::vector<float>> tensors;
    std::vector<bool> is_neg_expected;
    const float* neg_ptr = nullptr;
    size_t next = 0;
    int mismatches = 0;
    std::vector<float> operator()(const std::vector<float>&, float, const float* cond) {
        if (next >= tensors.size())
            throw std::runtime_error("ReplayModel: sampler called more times than the " +
                                     std::to_string(tensors.size()) + " calls recorded for this step");
        if ((cond == neg_ptr) != is_neg_expected[next]) ++mismatches;
        return tensors[next++];
    }
};

// `include_neg=false`: skip loading the recorded neg tensor even if the file exists. Needed for
// the real_branch_guided==true case (see check_step_cfg's header comment): the real step WAS
// guided and recorded an extra neg.npy (multidiffusion's "always evaluate neg" quirk even at
// gs=1), but our sp.steps=1 replay forces the not-guided branch, which never calls neg -- feeding
// that unused tensor into the ReplayModel would leave it uncomsumed and fail the "consumed every
// call" check for the wrong reason (a scope mismatch we already documented, not a real bug).
bool build_replay_one_step(const std::string& prefix, int N, ReplayModel& rm, std::string& err,
                           bool include_neg = true) {
    for (int ord = 0;; ++ord) {
        const std::string p = prefix + "_pos" + std::to_string(ord) + ".npy";
        if (!fs::exists(p)) break;
        npy::Array a = npy::load(p);
        if (!(a.shape.size() == 2 && a.shape[1] == 32 && a.shape[0] == N)) { err = "unexpected shape: " + p; return false; }
        rm.tensors.push_back(a.data); rm.is_neg_expected.push_back(false);
    }
    const std::string negp = prefix + "_neg.npy";
    if (include_neg && fs::exists(negp)) {
        npy::Array a = npy::load(negp);
        if (!(a.shape.size() == 2 && a.shape[1] == 32 && a.shape[0] == N)) { err = "unexpected shape: " + negp; return false; }
        rm.tensors.push_back(a.data); rm.is_neg_expected.push_back(true);
    }
    if (rm.tensors.empty()) { err = "no " + prefix + "_pos*/neg.npy files found"; return false; }
    return true;
}

bool load_native32(const std::string& path, int expectN, npy::Array& out, std::string& err) {
    if (!fs::exists(path)) { err = "missing: " + path; return false; }
    out = npy::load(path);
    if (!(out.shape.size() == 2 && out.shape[1] == 32 && (expectN < 0 || out.shape[0] == expectN))) {
        err = "unexpected shape: " + path; return false;
    }
    return true;
}
} // namespace

// `real_branch`: 実際の step が guided かどうか（fixture のneg.npy有無から判定、報告文に使うだけ）。
static bool check_step_cfg(const std::string& tag, const std::string& prefix,
                           const SamplerParams& sp_full, int V, MultiCondMode mode,
                           const std::vector<float>& sample0, const std::string& cfg_path, int N,
                           bool real_branch_guided) {
    std::string err;

    // 2回、独立に構築した ReplayModel + 同一入力で sample_flow_multi を呼び、bit-identical を
    // assert する（TASK-STAGES.md 項目7「決定性」。native側のみの決定性確認）。
    auto run_once = [&](std::vector<float>& out_result, ReplayModel& rm_out, std::string& err_out) -> bool {
        if (!build_replay_one_step(prefix, N, rm_out, err_out, /*include_neg=*/!real_branch_guided)) return false;
        std::vector<float> negbuf(32, 0.0f);
        std::vector<std::vector<float>> condbufs(V, std::vector<float>(32, 0.0f));
        std::vector<const float*> conds; conds.reserve(V);
        for (auto& b : condbufs) conds.push_back(b.data());
        rm_out.neg_ptr = negbuf.data();
        FlowFwd fwd = [&rm_out](const std::vector<float>& x, float t, const float* c) { return rm_out(x, t, c); };
        SamplerParams sp1 = sp_full; sp1.steps = 1;   // t[0]==1.0 は steps に関わらず不変 -> not-guided (tex gi1=0.9)
        std::vector<float> sample = sample0;
        try {
            out_result = sample_flow_multi(fwd, sample, conds, negbuf.data(), sp1, mode);
        } catch (const std::exception& e) {
            err_out = std::string("sample_flow_multi(steps=1) threw: ") + e.what();
            return false;
        }
        return true;
    };

    ReplayModel rm, rm2;
    std::vector<float> result, result2;
    std::string err2;
    if (!run_once(result, rm, err)) { check(false, tag + ": " + err); return false; }
    const bool ok2 = run_once(result2, rm2, err2);
    check(ok2 && result.size() == result2.size() &&
          std::memcmp(result.data(), result2.data(), result.size() * sizeof(float)) == 0,
          tag + ": determinism -- same replay input run twice natively is bit-identical" +
          (ok2 ? "" : (" (2nd run failed: " + err2 + ")")));

    check(rm.next == rm.tensors.size(), tag + ": consumed every recorded pos/neg call for this step (" +
          std::to_string(rm.next) + "/" + std::to_string(rm.tensors.size()) + ")");
    if (real_branch_guided)
        printf("info %s: real step was guided but sp.steps=1 forces not-guided (no neg call) -- valid here only "
               "because gs=1.0 makes both branches algebraically identical (pred=pos); see file header comment\n",
               tag.c_str());
    else
        check(rm.mismatches == 0, tag + ": pos/neg call pattern matches the recorded (not-guided) branch exactly");

    const std::vector<double> ts1 = flow_t_schedule(1, sp_full.rescale_t);
    const double dt1 = ts1[0] - ts1[1];
    std::vector<float> pred(sample0.size());
    for (size_t i = 0; i < pred.size(); ++i)
        pred[i] = (float)(((double)sample0[i] - (double)result[i]) / dt1);

    npy::Array cfg;
    if (!load_native32(cfg_path, N, cfg, err)) { check(false, tag + ": " + err); return false; }

    double maxd = 0, gmax = 0;
    for (size_t i = 0; i < pred.size(); ++i) {
        const double d = std::fabs((double)pred[i] - (double)cfg.data[i]);
        maxd = std::max(maxd, d); gmax = std::max(gmax, std::fabs((double)cfg.data[i]));
    }
    const double rel = maxd / std::max(gmax, 1e-12);
    char relbuf[64]; snprintf(relbuf, sizeof relbuf, "%.4e", rel);
    check(rel < 1e-3, tag + ": CFG pred (via real sample_flow_multi, sp.steps=1) matches recorded cfg.npy "
          "(rel=" + std::string(relbuf) + ", N=" + std::to_string(N) + ")");
    return true;
}

// =================================================================================================
// §3: fixture run table + driver
// =================================================================================================
namespace {
struct RunSpec { const char* name; const char* resolution; int V; MultiCondMode mode; };
const RunSpec RUNS[] = {
    { "run_1img_baseline_512",              "512",  1, MultiCondMode::Stochastic },
    { "run_1img_injected_stochastic_512",   "512",  1, MultiCondMode::Stochastic },
    { "run_2img_real_stochastic_512",       "512",  2, MultiCondMode::Stochastic },
    { "run_2img_real_multidiffusion_512",   "512",  2, MultiCondMode::MultiDiffusion },
    { "run_4img_real_stochastic_512",       "512",  4, MultiCondMode::Stochastic },
    { "run_4img_real_multidiffusion_512",   "512",  4, MultiCondMode::MultiDiffusion },
    { "run_1img_baseline_1024c",            "1024c", 1, MultiCondMode::Stochastic },
    { "run_1img_injected_stochastic_1024c", "1024c", 1, MultiCondMode::Stochastic },
    { "run_2img_real_stochastic_1024c",     "1024c", 2, MultiCondMode::Stochastic },
    { "run_2img_real_multidiffusion_1024c", "1024c", 2, MultiCondMode::MultiDiffusion },
    { "run_4img_real_stochastic_1024c",     "1024c", 4, MultiCondMode::Stochastic },
    { "run_4img_real_multidiffusion_1024c", "1024c", 4, MultiCondMode::MultiDiffusion },
};
} // namespace

static int check_coords_rows(const std::string& tag, const std::string& path) {
    if (!fs::exists(path)) { check(false, tag + ": coords file present (" + path + ")"); return -1; }
    npy::ArrayI32 c = npy::load_i32(path);
    if (!(c.shape.size() == 2 && c.shape[1] == 4)) { check(false, tag + ": coords shape is [N,4]"); return -1; }
    const int N = (int)c.shape[0];
    std::set<std::array<int,4>> uniq;
    for (int64_t i = 0; i < N; ++i) uniq.insert({ c.data[i*4], c.data[i*4+1], c.data[i*4+2], c.data[i*4+3] });
    check((int)uniq.size() == N, tag + ": coords has no duplicate rows (N=" + std::to_string(N) + ")");
    return N;
}

static int run_all(const std::string& fixture_root) {
    int ran = 0;
    for (const auto& r : RUNS) {
        const std::string dir = fixture_root + "/" + r.name;
        const std::string tag0 = std::string(r.name) + " (V=" + std::to_string(r.V) + ", " +
            (r.mode == MultiCondMode::Stochastic ? "stochastic" : "multidiffusion") + ")";
        if (!fs::exists(dir)) { check(false, tag0 + ": fixture run directory present (" + dir + ")"); continue; }

        // texture の coords は「その run が最終的に確定した shape 座標」そのもの:
        // 512 run は ss_coords.npy、1024cascade run は shape_slat_coords.npy (HR final)。
        const std::string coords_path = dir + (std::string(r.resolution) == "512" ? "/ss_coords.npy" : "/shape_slat_coords.npy");
        const int N = check_coords_rows(tag0, coords_path);
        if (N < 0) continue;

        std::string err;
        npy::Array noise;
        if (!load_native32(dir + "/noise_tex_0.npy", N, noise, err)) { check(false, tag0 + ": " + err); continue; }

        bool ok = true;
        ok &= check_step_cfg(tag0 + " step0", dir + "/tex_step0", tex_production_params(), r.V, r.mode,
                             noise.data, dir + "/tex_step0_cfg.npy", N, /*real_branch_guided=*/false);
        ok &= check_step_cfg(tag0 + " step8", dir + "/tex_step8", tex_production_params(), r.V, r.mode,
                             noise.data, dir + "/tex_step8_cfg.npy", N, /*real_branch_guided=*/true);
        ok &= check_step_cfg(tag0 + " step11", dir + "/tex_step11", tex_production_params(), r.V, r.mode,
                             noise.data, dir + "/tex_step11_cfg.npy", N, /*real_branch_guided=*/false);
        if (ok) ++ran;
    }
    check(ran == 12, "all 12 texture fixture runs (512 x6 + 1024cascade x6) were present and checked (ran=" +
          std::to_string(ran) + "/12)");
    return ran;
}

int main(int argc, char** argv) {
    const char* home = getenv("HOME");
    const std::string fixture_root = argc > 1 ? argv[1]
        : (std::string(home ? home : "") + "/nfs/pixal3d_trellis2mv_ref_v2");

    test_concat_structure();
    test_gi_boundaries();

    printf("== texture step{0,8,11} CFG vs pinned reference (gr=0.0 -> sample-independent, no clamp path) ==\n");
    run_all(fixture_root);

    printf(g_fail == 0 ? "\nALL PASS\n" : "\n%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
