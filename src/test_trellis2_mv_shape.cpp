// trellis2-mv (TASK-STAGES) shape SLat MV stage tests: 512 単独解像度 + 1024 cascade (LR/HR)。
//   ./build/trellis-test-trellis2-mv-shape [fixture_root] [gpu(未使用)]
//
// docs/design/2026-09-21-trellis2-mv-stages-shape-tex.md の設計・codex exec レビューの
// 反映結果を実装したもの。P4 (`src/test_trellis2_mv_ss.cpp`) との最大の違い:
//
// v2 fixture (~/nfs/pixal3d_trellis2mv_ref_v2) は shape/texture の予測テンソルを
// step {0, 8, 11} の3点だけ保存している（SS は全12点）。中間 step の速度場が無いため、
// 12ステップ Euler 積分を通しで replay して最終 shape_slat と比較することは出来ない
// （設計ドキュメント「レビュー反映」1番）。本ファイルが実際に検証するのは:
//
//   - step0 の CFG 後出力（LR・HR 両方）: step0 の直前状態は常に既知の初期ノイズ
//     （`noise_shape_0.npy` / `noise_shape_cascade_{0,1}.npy`）なので、#71 の本番
//     `sample_flow_multi` を `sp.steps=1` で直接呼び、返り値から速度を逆算して
//     `shape_{,lr_,hr_}step0_cfg.npy` と比較する（設計ドキュメント「追記」節）。
//     CFG混合・rescale・clampは複製せず、本番コードをそのまま呼ぶ。
//   - step8 の guidance-interval inside 判定（純粋な境界確認、fixtureのneg有無で裏付け）。
//   - coords/feats の行数整合（重複なし、noise/予測/最終featsの行数がcoordsと一致）。
//   - quantize backoff ループ（trellis_cli.cpp:982-997 の複製、model依存なしの純算術）の
//     structural test。実 fixture では常に1回で抜けるため synthetic データで backoff 自体を
//     exercise する（「実fixtureでbackoffが起きた」とは主張しない）。
//
// 削除した検証（設計ドキュメント「レビュー反映」1・2番、実行不可能と判明）:
//   - 最終 shape_slat_{feats,coords} が参照と一致すること（中間9ステップの速度場が無い）。
//   - HRコード集合を自己導出（upsample+quantize）して参照 shape_slat_coords.npy と
//     一致することの検証（参照のLR最終SLatが無いため自己導出できない）。
// これらは統括への FINDING として報告する（fixture再生成の要否は統括判断）。
//
// trellis_cli.cpp は変更しない。ここに複製したのは quantize backoff ループのみ
// （denorm・CFG混合は複製しない — 本番 sample_flow_multi を直接呼ぶため不要）。
#include "flow_runner.h"
#include "npy.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
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

// Production shape-SLAT sampler params (trellis_cli.cpp:939, cfg.gsh の既定値 7.5f を使用)。
static SamplerParams shape_production_params(int steps = 12) {
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 7.5f; sp.guidance_rescale = 0.5f;
    sp.gi0 = 0.6; sp.gi1 = 1.0; sp.rescale_t = 3.0;
    return sp;
}

// =================================================================================================
// §1: quantize backoff ループの structural test（trellis_cli.cpp:982-997 の複製、pure arithmetic）。
// model/fixture不要。実fixtureでは常にhr_res<=1024で1回目に抜けるので、backoffが実際に周回する
//経路はここでしか確認できない（設計ドキュメント「レビュー反映」8番）。
// =================================================================================================
namespace {
std::vector<std::array<int,3>> quantize_hr_backoff(const std::vector<std::array<int,3>>& hr_coords512,
                                                     int hr_target, int max_tokens, int& out_hr_res) {
    int hr_res = hr_target;
    for (;;) {
        const int gi = hr_res / 16;
        const float g = (float)gi;
        std::set<std::array<int,3>> q;
        for (auto& c : hr_coords512)
            q.insert({ (int)((c[0]+0.5f)/512.f*g), (int)((c[1]+0.5f)/512.f*g), (int)((c[2]+0.5f)/512.f*g) });
        if ((int)q.size() < max_tokens || hr_res <= 1024) {
            out_hr_res = hr_res;
            return std::vector<std::array<int,3>>(q.begin(), q.end());
        }
        hr_res -= 128;
    }
}
} // namespace

static void test_quantize_backoff_synthetic() {
    printf("-- quantize_hr_backoff: structural test (synthetic coords, model/fixture unused) --\n");
    // 高解像度側で確実に max_tokens を超えるよう、res512 空間に一様分布する多数の座標を作る。
    std::vector<std::array<int,3>> coords;
    for (int x = 0; x < 128; ++x) for (int y = 0; y < 128; ++y) for (int z = 0; z < 8; ++z)
        coords.push_back({ x * 4, y * 4, z * 64 });   // 128*128*8 = 131072 点、grid96量子化でも重複少数
    int hr_res = 0;
    auto shc = quantize_hr_backoff(coords, /*hr_target=*/1536, /*max_tokens=*/2000, hr_res);
    check(hr_res < 1536, "backoff actually reduces hr_res below the 1536 target when over max_tokens (hr_res=" +
          std::to_string(hr_res) + ")");
    check(hr_res >= 1024 && (hr_res % 128 == 1536 % 128), "hr_res stays on the -128 grid and floors at/above 1024 (hr_res=" +
          std::to_string(hr_res) + ")");
    check(!shc.empty(), "backoff still returns a non-empty coord set");

    // floor 側: max_tokens が十分大きければ1回目 (grid=96) で抜ける。
    int hr_res2 = 0;
    auto shc2 = quantize_hr_backoff(coords, 1536, /*max_tokens=*/1000000, hr_res2);
    check(hr_res2 == 1536, "large max_tokens: no backoff needed, hr_res stays at the target (hr_res=" +
          std::to_string(hr_res2) + ")");

    // 1024 floor: hr_res<=1024 は件数に関わらず即確定（本番の実 fixture 6run が常にこの分岐を通る所以）。
    int hr_res3 = 0;
    auto shc3 = quantize_hr_backoff(coords, 1024, /*max_tokens=*/1, hr_res3);
    check(hr_res3 == 1024, "hr_res<=1024 floor: accepted on the first pass even with max_tokens=1 (hr_res=" +
          std::to_string(hr_res3) + ")");
}

// i=8 の guidance-interval 境界（TASK-STAGES.md 項目3）。純粋な境界確認、fixture不要。
static void test_i8_boundary() {
    printf("-- shape sampler: step i=8 guidance-interval boundary --\n");
    const SamplerParams sp = shape_production_params();
    const std::vector<double> ts = flow_t_schedule(sp.steps, sp.rescale_t);
    check(ts.size() == 13, "flow_t_schedule(12, rescale_t=3.0) returns 13 knots");
    // 設計ドキュメントで python 実測済み: ts[8] == 0.6 (gi0 の境界そのもの)。
    check(std::fabs(ts[8] - 0.6) < 1e-9, "ts[8] == 0.6 (shape gi0, exact boundary; ts[8]=" +
          std::to_string(ts[8]) + ")");
    check(flow_in_guidance_interval(ts[8], sp.gi0, sp.gi1),
          "flow_in_guidance_interval(ts[8], 0.6, 1.0) == true (inside, boundary is inclusive)");
}

// =================================================================================================
// §2: step0 CFG 検証。#71 の本番 sample_flow_multi を sp.steps=1 で直接呼ぶ（設計ドキュメント
// 「追記」節）。denorm・CFG混合・rescale・clamp は一切複製しない。
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

bool build_replay_one_step(const std::string& prefix, int N, ReplayModel& rm, std::string& err) {
    for (int ord = 0;; ++ord) {
        const std::string p = prefix + "_pos" + std::to_string(ord) + ".npy";
        if (!fs::exists(p)) break;
        npy::Array a = npy::load(p);
        if (!(a.shape.size() == 2 && a.shape[1] == 32 && a.shape[0] == N)) { err = "unexpected shape: " + p; return false; }
        rm.tensors.push_back(a.data); rm.is_neg_expected.push_back(false);
    }
    const std::string negp = prefix + "_neg.npy";
    if (fs::exists(negp)) {
        npy::Array a = npy::load(negp);
        if (!(a.shape.size() == 2 && a.shape[1] == 32 && a.shape[0] == N)) { err = "unexpected shape: " + negp; return false; }
        rm.tensors.push_back(a.data); rm.is_neg_expected.push_back(true);
    }
    if (rm.tensors.empty()) { err = "no " + prefix + "_pos*/neg.npy files found"; return false; }
    return true;
}

// [N,32] torch row-major (index n*32+c) と native channel-major (index c+32*n) は代数的に同一
// (n*32+c == c+32*n) なので transpose 不要（設計ドキュメント「fixtureのレイアウト確認」参照）。
bool load_native32(const std::string& path, int expectN, npy::Array& out, std::string& err) {
    if (!fs::exists(path)) { err = "missing: " + path; return false; }
    out = npy::load(path);
    if (!(out.shape.size() == 2 && out.shape[1] == 32 && (expectN < 0 || out.shape[0] == expectN))) {
        err = "unexpected shape: " + path; return false;
    }
    return true;
}
} // namespace

// Issue #77 の OOD clamp ratio を step0 の記録済みテンソルから直接計算する（前方宣言、定義は
// check_step0_cfg の後）。pure arithmetic, apply_guidance_rescale は呼ばない -- 本番実装の検証
// ではなく、この step のデータが [0.2,5.0] guard に触れるかどうかの診断のみ。欠損時は NaN。
static double compute_step0_clamp_ratio(const std::string& prefix, const SamplerParams& sp, int N,
                                        const std::vector<float>& noise, bool multidiffusion);

// Returns false (and reports via check()) on any structural failure. `strict`: TRELLIS_NOFIX=1
// (parity gate, clamp disabled) vs default (production gate, clamp on -- known/expected
// divergence from the un-clamped reference is informational only, matching P4's precedent).
static bool check_step0_cfg(const std::string& tag, const std::string& prefix,
                            const SamplerParams& sp_full, int V, MultiCondMode mode,
                            const std::string& noise_path, const std::string& cfg_path, int N) {
    std::string err;
    npy::Array noise;
    if (!load_native32(noise_path, N, noise, err)) { check(false, tag + ": " + err); return false; }

    const bool multidiffusion = (mode == MultiCondMode::MultiDiffusion);

    // 2回、独立に構築した ReplayModel + 同一入力で sample_flow_multi を呼び、bit-identical を
    // assert する（TASK-STAGES.md 項目7「決定性」。native側のみの決定性確認、上の追記コメント参照）。
    auto run_once = [&](std::vector<float>& out_result, ReplayModel& rm_out, std::string& err_out) -> bool {
        if (!build_replay_one_step(prefix, N, rm_out, err_out)) return false;
        // ローカルの neg/cond バッファは呼び出しごとに作る（ReplayModel の neg_ptr 比較用）。
        auto negbuf = std::make_shared<std::vector<float>>(32, 0.0f);
        auto condbufs = std::make_shared<std::vector<std::vector<float>>>(V, std::vector<float>(32, 0.0f));
        std::vector<const float*> conds; conds.reserve(V);
        for (auto& b : *condbufs) conds.push_back(b.data());
        rm_out.neg_ptr = negbuf->data();
        FlowFwd fwd = [&rm_out](const std::vector<float>& x, float t, const float* c) { return rm_out(x, t, c); };
        SamplerParams sp1 = sp_full; sp1.steps = 1;   // t[0]==1.0 は steps に関わらず不変（設計ドキュメント参照）
        try {
            out_result = sample_flow_multi(fwd, noise.data, conds, negbuf->data(), sp1, mode);
        } catch (const std::exception& e) {
            err_out = std::string("sample_flow_multi(steps=1) threw: ") + e.what();
            return false;
        }
        return true;
    };

    ReplayModel rm1, rm2;
    std::vector<float> result, result2;
    std::string err1, err2;
    const bool ok1 = run_once(result, rm1, err1);
    if (!ok1) { check(false, tag + ": " + err1); return false; }
    const bool ok2 = run_once(result2, rm2, err2);
    check(ok2 && result.size() == result2.size() &&
          std::memcmp(result.data(), result2.data(), result.size() * sizeof(float)) == 0,
          tag + ": determinism -- same replay input run twice natively is bit-identical" +
          (ok2 ? "" : (" (2nd run failed: " + err2 + ")")));

    check(rm1.next == rm1.tensors.size(), tag + ": consumed every recorded pos/neg call for this step (" +
          std::to_string(rm1.next) + "/" + std::to_string(rm1.tensors.size()) + ")");
    check(rm1.mismatches == 0, tag + ": every call's pos/neg branch matches the recorded call pattern");

    const std::vector<double> ts1 = flow_t_schedule(1, sp_full.rescale_t);
    const double dt1 = ts1[0] - ts1[1];
    std::vector<float> pred(noise.data.size());
    for (size_t i = 0; i < pred.size(); ++i)
        pred[i] = (float)(((double)noise.data[i] - (double)result[i]) / dt1);

    npy::Array cfg;
    if (!load_native32(cfg_path, N, cfg, err)) { check(false, tag + ": " + err); return false; }

    double maxd = 0, gmax = 0;
    for (size_t i = 0; i < pred.size(); ++i) {
        const double d = std::fabs((double)pred[i] - (double)cfg.data[i]);
        maxd = std::max(maxd, d); gmax = std::max(gmax, std::fabs((double)cfg.data[i]));
    }
    const double rel = maxd / std::max(gmax, 1e-12);
    const bool strict = getenv("TRELLIS_NOFIX") != nullptr;
    char relbuf[64]; snprintf(relbuf, sizeof relbuf, "%.4e", rel);

    if (strict) {
        check(rel < 1e-3, tag + ": step0 CFG pred (via real sample_flow_multi, sp.steps=1) matches recorded "
              "cfg.npy (parity gate, clamp OFF, rel=" + std::string(relbuf) + ", N=" + std::to_string(N) + ")");
    } else {
        // production gate (clamp ON): 2026-09-21 差し戻し対応 (3) -- min_ratio を info printf のみに
        // せず、実測帯（0.2462-0.5918、本ファイルの全run）を回帰ガードとして assert する。
        // ratio>=0.2 が成立する限り clamp は no-op なので、その場合は rel も hard assert にできる
        // （parity gate と数値的に同じ結果になるはず）。ratio が求まらない/0.2未満の場合のみ
        // rel を info に落とす（clamp由来の既知の乖離、SS版と同じ扱い）。
        const double ratio = compute_step0_clamp_ratio(prefix, sp_full, N, noise.data, multidiffusion);
        const bool ratio_known = std::isfinite(ratio);
        if (ratio_known) {
            check(ratio >= 0.2, tag + ": OOD clamp regression guard -- step0 ratio stays >=0.2 (observed "
                  "band 0.2462-0.5918 across all runs, 2026-09-21; ratio=" + std::to_string(ratio) + ")");
        } else {
            printf("info %s: OOD clamp ratio unavailable (missing pos/neg files) -- rel check left informational\n",
                   tag.c_str());
        }
        const std::string msg = tag + ": step0 CFG pred (via real sample_flow_multi, sp.steps=1) matches recorded "
            "cfg.npy (production gate, clamp ON, rel=" + std::string(relbuf) + ", N=" + std::to_string(N) + ")";
        if (ratio_known && ratio >= 0.2) check(rel < 1e-3, msg);   // ratio>=0.2 -> clamp no-op -> hard assert
        else printf("%s %s [informational -- clamp may be active]\n", (rel < 1e-3) ? "ok  " : "info", msg.c_str());
    }
    return true;
}

// Issue #77 の OOD clamp ratio を step0 の記録済みテンソルから直接計算する（pure arithmetic,
// apply_guidance_rescale は呼ばない -- 本番実装の検証ではなく、この step のデータが [0.2,5.0]
// guard に触れるかどうかの診断のみ）。ファイル欠如時は NaN を返す。
static double compute_step0_clamp_ratio(const std::string& prefix, const SamplerParams& sp, int N,
                                        const std::vector<float>& noise, bool multidiffusion) {
    std::string err;
    npy::Array pos0, neg;
    const std::string pos_path = multidiffusion ? (prefix + "_avg.npy") : (prefix + "_pos0.npy");
    if (!load_native32(pos_path, N, pos0, err) || !load_native32(prefix + "_neg.npy", N, neg, err))
        return std::numeric_limits<double>::quiet_NaN();
    const double a = 1.0 - sp.sigma_min, b = sp.sigma_min + (1.0 - sp.sigma_min) * 1.0;   // t==1.0 at step0
    double mp = 0, mc = 0;
    std::vector<double> x0p(noise.size()), x0c(noise.size());
    const double gs = sp.guidance_strength;
    for (size_t i = 0; i < noise.size(); ++i) {
        const double raw = gs * (double)pos0.data[i] + (1.0 - gs) * (double)neg.data[i];
        x0p[i] = a * noise[i] - b * (double)pos0.data[i];
        x0c[i] = a * noise[i] - b * raw;
        mp += x0p[i]; mc += x0c[i];
    }
    mp /= noise.size(); mc /= noise.size();
    double vp = 0, vc = 0;
    for (size_t i = 0; i < noise.size(); ++i) { vp += (x0p[i]-mp)*(x0p[i]-mp); vc += (x0c[i]-mc)*(x0c[i]-mc); }
    return vc > 0 ? std::sqrt(vp / (noise.size()-1)) / std::sqrt(vc / (noise.size()-1)) : 1.0;
}

// =================================================================================================
// §3: fixture run table + driver
// =================================================================================================
namespace {
struct RunSpec { const char* name; int V; MultiCondMode mode; };
const RunSpec RUNS_512[] = {
    { "run_1img_baseline_512",              1, MultiCondMode::Stochastic },
    { "run_1img_injected_stochastic_512",   1, MultiCondMode::Stochastic },
    { "run_2img_real_stochastic_512",       2, MultiCondMode::Stochastic },
    { "run_2img_real_multidiffusion_512",   2, MultiCondMode::MultiDiffusion },
    { "run_4img_real_stochastic_512",       4, MultiCondMode::Stochastic },
    { "run_4img_real_multidiffusion_512",   4, MultiCondMode::MultiDiffusion },
};
const RunSpec RUNS_1024C[] = {
    { "run_1img_baseline_1024c",            1, MultiCondMode::Stochastic },
    { "run_1img_injected_stochastic_1024c", 1, MultiCondMode::Stochastic },
    { "run_2img_real_stochastic_1024c",     2, MultiCondMode::Stochastic },
    { "run_2img_real_multidiffusion_1024c", 2, MultiCondMode::MultiDiffusion },
    { "run_4img_real_stochastic_1024c",     4, MultiCondMode::Stochastic },
    { "run_4img_real_multidiffusion_1024c", 4, MultiCondMode::MultiDiffusion },
};
} // namespace

// coords ファイルの行数・重複なしを確認し、行数を返す（失敗時 -1）。
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

static int run_512(const std::string& fixture_root) {
    int ran = 0;
    for (const auto& r : RUNS_512) {
        const std::string dir = fixture_root + "/" + r.name;
        const std::string tag = std::string(r.name) + " (V=" + std::to_string(r.V) + ", " +
            (r.mode == MultiCondMode::Stochastic ? "stochastic" : "multidiffusion") + ")";
        if (!fs::exists(dir)) { check(false, tag + ": fixture run directory present (" + dir + ")"); continue; }
        const int N = check_coords_rows(tag, dir + "/ss_coords.npy");
        if (N < 0) continue;
        if (check_step0_cfg(tag, dir + "/shape_step0", shape_production_params(), r.V, r.mode,
                            dir + "/noise_shape_0.npy", dir + "/shape_step0_cfg.npy", N)) {
            ++ran;
        }
    }
    check(ran == 6, "all 6 shape-512 (LR-equivalent) fixture runs were present and checked (ran=" +
          std::to_string(ran) + "/6)");
    return ran;
}

static int run_1024c(const std::string& fixture_root) {
    int ran = 0;
    for (const auto& r : RUNS_1024C) {
        const std::string dir = fixture_root + "/" + r.name;
        const std::string base_tag = std::string(r.name) + " (V=" + std::to_string(r.V) + ", " +
            (r.mode == MultiCondMode::Stochastic ? "stochastic" : "multidiffusion") + ")";
        if (!fs::exists(dir)) { check(false, base_tag + ": fixture run directory present (" + dir + ")"); continue; }

        bool ok = true;
        const int N_lr = check_coords_rows(base_tag + " LR", dir + "/ss_coords.npy");
        ok &= (N_lr >= 0);
        if (ok) {
            const std::string tag = base_tag + " LR";
            ok &= check_step0_cfg(tag, dir + "/shape_lr_step0", shape_production_params(), r.V, r.mode,
                                  dir + "/noise_shape_cascade_0.npy", dir + "/shape_lr_step0_cfg.npy", N_lr);
        }

        const int N_hr = check_coords_rows(base_tag + " HR", dir + "/shape_slat_coords.npy");
        ok &= (N_hr >= 0);
        if (N_hr >= 0) {
            // HR の初期ノイズは LR に依存しない独立ノイズ（設計ドキュメント「レビュー反映」2番、
            // noise_shape_cascade_1.npy の行数が shape_slat_coords.npy と一致することで実測確認済み）。
            const std::string tag = base_tag + " HR";
            const bool ok_hr = check_step0_cfg(tag, dir + "/shape_hr_step0", shape_production_params(), r.V, r.mode,
                                               dir + "/noise_shape_cascade_1.npy", dir + "/shape_hr_step0_cfg.npy", N_hr);
            ok &= ok_hr;
        }
        if (ok) ++ran;
    }
    check(ran == 6, "all 6 shape-1024cascade (LR+HR) fixture runs were present and checked (ran=" +
          std::to_string(ran) + "/6)");
    return ran;
}

// =================================================================================================
// §4: LR->HR 共有 stochastic カウンタ配線の real-fixture-scale structural test。
// 2026-09-21 差し戻し対応 (1) -- 設計ドキュメント「レビュー反映」2番・「検証項目」2番が約束した
// 「continued カウンタを渡しても実行が破綻せず」を、それまで一度も stochastic_counter を渡して
// いなかった (sample_flow_multi へのデフォルト nullptr のまま) のを修正して実装する。
//
// trellis_cli.cpp:925 (`int shape_stochastic_counter = 0;`) / :940-942
// (`t2mv_mode == Stochastic ? &shape_stochastic_counter : nullptr`) と同じ配線で、
// 実 1024cascade stochastic run の実 N_lr/N_hr/V を使い、LR呼び出し(steps=12)完了後に
// counter==12、それを継続して渡した HR呼び出し(steps=12)完了後に counter==24 になることを
// assert する。呼び出し値そのもの（forward の戻り値）はこの検証には無関係なので、実重み
// fixture の記録テンソルではなく決定的なダミー forward を使う（V が12を割り切るため
// continued/reset の数値差はどのみち観測できない -- 設計ドキュメント参照。この test の目的は
// 「API経路が例外なく通ること」と「counter値の遷移そのもの」だけ）。continued/reset の判別自体は
// test_trellis2_mv_sampler.cpp:75-97 と test_flow_multi.cpp の synthetic fixture が担保済み
// （重複実装しない）。
namespace {
struct RunScaleSpec { const char* name; int V; };
const RunScaleSpec STOCHASTIC_1024C_SCALES[] = {
    { "run_1img_baseline_1024c",            1 },
    { "run_1img_injected_stochastic_1024c", 1 },
    { "run_2img_real_stochastic_1024c",     2 },
    { "run_4img_real_stochastic_1024c",     4 },
};
} // namespace

static void test_cascade_stochastic_counter_real_scale(const std::string& fixture_root) {
    printf("-- shape LR->HR shared stochastic counter: real-fixture-scale structural test --\n");
    for (const auto& r : STOCHASTIC_1024C_SCALES) {
        const std::string dir = fixture_root + "/" + r.name;
        const std::string tag = std::string(r.name) + " (V=" + std::to_string(r.V) + ")";
        if (!fs::exists(dir)) { check(false, tag + ": fixture run directory present (" + dir + ")"); continue; }

        npy::Array noise_lr, noise_hr;
        std::string err;
        if (!load_native32(dir + "/noise_shape_cascade_0.npy", -1, noise_lr, err) ||
            !load_native32(dir + "/noise_shape_cascade_1.npy", -1, noise_hr, err)) {
            check(false, tag + ": " + err); continue;
        }
        const int N_lr = (int)noise_lr.shape[0], N_hr = (int)noise_hr.shape[0];

        std::vector<float> negbuf(32, 0.0f);
        std::vector<std::vector<float>> condbufs(r.V, std::vector<float>(32, 0.0f));
        std::vector<const float*> conds; conds.reserve(r.V);
        for (auto& b : condbufs) conds.push_back(b.data());
        // 戻り値は無関係（このテストは counter の遷移だけを見る）。有限なゼロ速度を返すだけの
        // ダミー forward -- 実 DiT/fixture データは使わない。
        FlowFwd dummy_fwd = [](const std::vector<float>& x, float, const float*) {
            return std::vector<float>(x.size(), 0.0f);
        };
        const SamplerParams sp = shape_production_params();   // steps=12 (LR/HRとも本番どおり)

        int counter = 0;   // trellis_cli.cpp:925 の shape_stochastic_counter に対応
        bool threw = false; std::string what;
        try {
            sample_flow_multi(dummy_fwd, noise_lr.data, conds, negbuf.data(), sp,
                              MultiCondMode::Stochastic, nullptr, &counter);   // :940-942 相当（LR）
        } catch (const std::exception& e) { threw = true; what = e.what(); }
        check(!threw, tag + ": LR call (steps=12, real N=" + std::to_string(N_lr) +
              ") does not throw" + (threw ? (" (" + what + ")") : ""));
        check(counter == 12, tag + ": shared counter == 12 after the LR call (counter=" +
              std::to_string(counter) + ")");

        threw = false;
        try {
            sample_flow_multi(dummy_fwd, noise_hr.data, conds, negbuf.data(), sp,
                              MultiCondMode::Stochastic, nullptr, &counter);   // 同じ counter を継続（HR）
        } catch (const std::exception& e) { threw = true; what = e.what(); }
        check(!threw, tag + ": HR call (steps=12, real N=" + std::to_string(N_hr) +
              "), continuing the LR counter, does not throw" + (threw ? (" (" + what + ")") : ""));
        check(counter == 24, tag + ": shared counter == 24 after the HR call continues the LR "
              "counter (counter=" + std::to_string(counter) + ")");
    }
}

static int run_parity_subprocess_mode(int argc, char** argv) {
    setenv("TRELLIS_NOFIX", "1", 1);
    const char* home = getenv("HOME");
    const std::string fixture_root = argc > 2 ? argv[2]
        : (std::string(home ? home : "") + "/nfs/pixal3d_trellis2mv_ref_v2");
    printf("== parity (clamp off): shape step0 CFG vs pinned reference ==\n");
    run_512(fixture_root);
    run_1024c(fixture_root);
    printf(g_fail == 0 ? "\n[parity subprocess] ALL PASS\n" : "\n[parity subprocess] %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--internal-parity-subprocess")
        return run_parity_subprocess_mode(argc, argv);

    const char* home = getenv("HOME");
    const std::string fixture_root = argc > 1 ? argv[1]
        : (std::string(home ? home : "") + "/nfs/pixal3d_trellis2mv_ref_v2");

    test_quantize_backoff_synthetic();
    test_i8_boundary();
    test_cascade_stochastic_counter_real_scale(fixture_root);

    printf("== production (clamp on, default settings): step0 CFG, informational + diagnostics ==\n");
    run_512(fixture_root);
    run_1024c(fixture_root);

    printf("== parity (clamp off): re-exec'ing self with TRELLIS_NOFIX forced internally ==\n");
    {
        const std::string cmd = "\"" + std::string(argv[0]) + "\" --internal-parity-subprocess \"" + fixture_root + "\"";
        const int rc = system(cmd.c_str());
        check(rc == 0, "parity (clamp off) subprocess exited 0 -- shape step0 CFG matches the pinned "
              "reference (~1e-6-1e-3 rel) across all 12 real fixture runs (512 x6 + 1024cascade LR/HR x6)");
    }

    printf(g_fail == 0 ? "\nALL PASS\n" : "\n%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
