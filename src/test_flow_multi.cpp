// trellis-test-flow-multi: issue #61 の synthetic sampler oracle（native側）。
// TASK-PORT P3: 旧トラック（~/Downloads/pixal3d-mv-61/src/test_flow_multi.cpp）を
// #71 の sample_flow_multi API へ移植したもの。docs/design/2026-09-21-trellis2-mv-port-hardening.md
// §P3参照。
//
// tools/gen_trellis2_mv_sampler_fixture.py（#59, PASS確認済み）と同じ検証観点を、決定的な
// fake FlowFwd と呼び出しログで native `sample_flow_multi` に対して直接検証する。4B重み・GPU・
// ネットワーク一切不要。
//
// 数値の突き合わせは2種類に分ける:
//   1. 呼び出し回数・順序などの構造的性質は自己完結の assert。
//   2. CFG/guidance_rescale を経た実数値は tests/fixtures/trellis2_mv_sampler/sampler_oracle.txt
//      （tools/gen_trellis2_mv_sampler_fixture.py が pin済み Python 実装を実際に動かして
//      生成した独立fixture）と突き合わせる。
//
// #71 の sample_flow_multi は旧トラックと2点でAPI/挙動が異なる（移植時に判明、直さず報告）:
//   - StochasticCounter型ではなく生の int* を7番目でなく8番目の引数として取る
//     （include/flow_runner.h: ..., trace = nullptr, stochastic_counter = nullptr)。
//   - 入力検証で std::runtime_error でなく std::invalid_argument を投げる（catchはstd::exceptionへ）。
// TASK-PORT F3（2026-09-21）: P3移植時点では conds 要素の null 検証と、stochastic 分岐の forward
// 戻り値サイズ検証が #71 に欠けており、それぞれ SKIP / 既知FAIL として扱っていた。flow_runner.cpp
// 側に両検証を追加したため、現在はどちらも通常の check() で 74/74 (0 FAIL, 0 SKIP) になる。
//
// Usage: trellis-test-flow-multi [fixture_path]
//   既定の fixture_path は "tests/fixtures/trellis2_mv_sampler/sampler_oracle.txt"
//   （リポジトリルートで実行することを前提。trellis-test-sv-input と同じ規約）。
#include "flow_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using trellis::FlowFwd;
using trellis::MultiCondMode;
using trellis::SamplerParams;

static int g_fail = 0;

static void check(const char* name, bool cond, const std::string& detail = "") {
    printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name,
           (!cond && !detail.empty()) ? " -- " : "", (!cond && !detail.empty()) ? detail.c_str() : "");
    if (!cond) ++g_fail;
}

// ---------------------------------------------------------------------------
// 決定的 fake model。tools/gen_trellis2_mv_sampler_fixture.py の make_fake_model と同一の式。
// cond の先頭要素を "tag" として使う: (view+1)*10 で view をエンコード、negは0.0。
// ---------------------------------------------------------------------------
struct CallEntry { int view; float t; };   // view == -1 は negative

struct FakeModel {
    int dim;
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
    std::vector<int> v;
    for (const auto& e : log) if (e.view >= 0) v.push_back(e.view);
    return v;
}
static int negative_count(const std::vector<CallEntry>& log) {
    int n = 0; for (const auto& e : log) if (e.view < 0) ++n; return n;
}

// ---------------------------------------------------------------------------
// 4.1-1: stochastic 呼び出し順序
// ---------------------------------------------------------------------------
static void test_stochastic_order() {
    const int V = 3, steps = 12, dim = 6;
    std::vector<std::vector<float>> condbuf; for (int v = 0; v < V; ++v) condbuf.push_back(make_cond_buf(v));
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds; for (auto& b : condbuf) conds.push_back(b.data());

    FakeModel fm{ dim, {} };
    FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* c) { return fm(x, t, c); };
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 2.0f; sp.gi0 = 0.0; sp.gi1 = 1.0; sp.rescale_t = 1.0;
    std::vector<float> noise(dim, 0.f);
    trellis::sample_flow_multi(fwd, noise, conds, negbuf.data(), sp, MultiCondMode::Stochastic);

    std::vector<int> pv = positive_views(fm.log);
    std::vector<int> expected; for (int k = 0; k < steps; ++k) expected.push_back(k % V);
    check("stochastic: positive view sequence == [k % V]", pv == expected);
    check("stochastic: negative call count == steps (always-inside interval)", negative_count(fm.log) == steps);
}

// ---------------------------------------------------------------------------
// 4.1-2: stochastic guidance_strength=0 でも counter は進む
// ---------------------------------------------------------------------------
static void test_stochastic_g0_counter_advances() {
    const int V = 2, steps = 6, dim = 6;
    std::vector<std::vector<float>> condbuf; for (int v = 0; v < V; ++v) condbuf.push_back(make_cond_buf(v));
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds; for (auto& b : condbuf) conds.push_back(b.data());
    std::vector<float> noise(dim, 0.f);

    // g=0 の run: positive forward は1回も呼ばれない (counterは無条件に進むはず)
    FakeModel fm0{ dim, {} };
    FlowFwd fwd0 = [&](const std::vector<float>& x, float t, const float* c) { return fm0(x, t, c); };
    SamplerParams sp0; sp0.steps = steps; sp0.guidance_strength = 0.0f; sp0.gi0 = 0.0; sp0.gi1 = 1.0; sp0.rescale_t = 1.0;
    int ctr = 0;
    trellis::sample_flow_multi(fwd0, noise, conds, negbuf.data(), sp0, MultiCondMode::Stochastic, nullptr, &ctr);
    check("stochastic g=0: no positive forward is issued", positive_views(fm0.log).empty());
    check("stochastic g=0: negative called every step", negative_count(fm0.log) == steps);

    // 同じ counter を継続し、g=1 に切り替えて次の呼び出しを可視化する。
    FakeModel fm1{ dim, {} };
    FlowFwd fwd1 = [&](const std::vector<float>& x, float t, const float* c) { return fm1(x, t, c); };
    SamplerParams sp1 = sp0; sp1.guidance_strength = 1.0f; sp1.steps = 1;
    trellis::sample_flow_multi(fwd1, noise, conds, negbuf.data(), sp1, MultiCondMode::Stochastic, nullptr, &ctr);
    std::vector<int> pv1 = positive_views(fm1.log);
    check("stochastic g=0->g=1: counter continued across the unused steps",
          !pv1.empty() && pv1[0] == (steps % V),
          pv1.empty() ? "no positive call" : ("got view " + std::to_string(pv1[0]) + " expected " + std::to_string(steps % V)));
}

// ---------------------------------------------------------------------------
// 4.1-3: steps < views
// ---------------------------------------------------------------------------
static void test_stochastic_steps_lt_views() {
    const int V = 4, steps = 2, dim = 6;
    std::vector<std::vector<float>> condbuf; for (int v = 0; v < V; ++v) condbuf.push_back(make_cond_buf(v));
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds; for (auto& b : condbuf) conds.push_back(b.data());
    FakeModel fm{ dim, {} };
    FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* c) { return fm(x, t, c); };
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 1.0f; sp.gi0 = 0.0; sp.gi1 = 1.0; sp.rescale_t = 1.0;
    std::vector<float> noise(dim, 0.f);
    trellis::sample_flow_multi(fwd, noise, conds, negbuf.data(), sp, MultiCondMode::Stochastic);
    std::vector<int> pv = positive_views(fm.log);
    std::vector<int> expected = { 0, 1 };
    check("stochastic steps<views: view sequence truncates cleanly", pv == expected);
}

// ---------------------------------------------------------------------------
// 4.1-4/5: interval外でもcounterは進み選択condが使われる／t==gi0,gi1の両端inclusive
// ---------------------------------------------------------------------------
static void test_stochastic_interval_edges() {
    const int V = 3, steps = 6, dim = 6;
    std::vector<std::vector<float>> condbuf; for (int v = 0; v < V; ++v) condbuf.push_back(make_cond_buf(v));
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds; for (auto& b : condbuf) conds.push_back(b.data());
    std::vector<float> noise(dim, 0.f);

    FakeModel fm{ dim, {} };
    FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* c) { return fm(x, t, c); };
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 5.0f; sp.gi0 = 0.4; sp.gi1 = 0.6; sp.rescale_t = 1.0;
    trellis::sample_flow_multi(fwd, noise, conds, negbuf.data(), sp, MultiCondMode::Stochastic);
    std::vector<int> pv = positive_views(fm.log);
    std::vector<int> expected; for (int k = 0; k < steps; ++k) expected.push_back(k % V);
    check("stochastic: interval-outside steps still advance & use the round-robin cond", pv == expected);

    // t==gi0/gi1 の両端inclusive: 実スケジュールの ts[i] を1つ取り、そこだけを区間にする。
    SamplerParams sp2; sp2.steps = 4; sp2.guidance_strength = 9.0f; sp2.rescale_t = 1.0;
    // rescale_t=1.0 の場合 flow_t_schedule は 1 - i/steps と恒等。i=2 の t を境界にする。
    const std::vector<double> ts2 = trellis::flow_t_schedule(sp2.steps, sp2.rescale_t);
    sp2.gi0 = ts2[2]; sp2.gi1 = ts2[2];
    FakeModel fm2{ dim, {} };
    FlowFwd fwd2 = [&](const std::vector<float>& x, float t, const float* c) { return fm2(x, t, c); };
    trellis::sample_flow_multi(fwd2, noise, conds, negbuf.data(), sp2, MultiCondMode::Stochastic);
    check("stochastic: gi0==gi1==ts[2] hits exactly one step (inclusive both ends)",
          negative_count(fm2.log) == 1,
          "got " + std::to_string(negative_count(fm2.log)) + " negative calls");
}

// ---------------------------------------------------------------------------
// 4.1-6: multidiffusion 呼び出し回数（実パラメータ表）
// ---------------------------------------------------------------------------
// expect_inside/outside は #59 の一次資料読み下し（docs/spec/33-trellis2-mv-reference.md、
// 旧トラック側。この#71ツリーには未移植）通り SS=10/2・shape=9/3・texture=6/6。
// shape/texture は rescale_t=3.0, steps=12 で ts[8] がちょうど t=0.6（guidance_interval下端）に
// 乗る境界ケースを含む -- TASK-PORT P1 で移植した flow_t_schedule() の double 演算が
// 正しく効いていることの直接的な回帰ガード（P1 が無いと shape=8/4・tex=5/7 になる）。
struct StageParams { const char* name; int steps; float gs; double gi0, gi1; float gr; double rescale_t; int expect_inside, expect_outside; };
static const StageParams STAGES[] = {
    { "sparse_structure_sampler", 12, 7.5f, 0.6, 1.0, 0.7f, 5.0, 10, 2 },
    { "shape_slat_sampler",       12, 7.5f, 0.6, 1.0, 0.5f, 3.0, 9, 3 },
    { "tex_slat_sampler",         12, 1.0f, 0.6, 0.9, 0.0f, 3.0, 6, 6 },
};

static void test_multidiffusion_call_counts() {
    const int V = 4, dim = 6;
    std::vector<std::vector<float>> condbuf; for (int v = 0; v < V; ++v) condbuf.push_back(make_cond_buf(v));
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds; for (auto& b : condbuf) conds.push_back(b.data());
    std::vector<float> noise(dim, 0.f);

    for (const auto& st : STAGES) {
        FakeModel fm{ dim, {} };
        FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* c) { return fm(x, t, c); };
        SamplerParams sp; sp.steps = st.steps; sp.guidance_strength = st.gs; sp.gi0 = st.gi0; sp.gi1 = st.gi1;
        sp.guidance_rescale = st.gr; sp.rescale_t = st.rescale_t;
        trellis::sample_flow_multi(fwd, noise, conds, negbuf.data(), sp, MultiCondMode::MultiDiffusion);
        const int pos_calls = (int)positive_views(fm.log).size();
        const int neg_calls = negative_count(fm.log);
        std::string tag = std::string(st.name) + ": ";
        check((tag + "positive call count == steps*V").c_str(), pos_calls == st.steps * V,
              "got " + std::to_string(pos_calls) + " expected " + std::to_string(st.steps * V));
        check((tag + "negative call count == #inside steps").c_str(), neg_calls == st.expect_inside,
              "got " + std::to_string(neg_calls) + " expected " + std::to_string(st.expect_inside));
        if (st.gs == 1.0f)
            check((tag + "guidance_strength==1.0 does not short-circuit negative eval").c_str(),
                  neg_calls == st.expect_inside && st.expect_inside > 0);
    }
}

// ---------------------------------------------------------------------------
// 4.1-7: guidance_strength in {0,1,0.5} で短絡が起きない
// ---------------------------------------------------------------------------
static void test_no_shortcircuit_strengths() {
    const int V = 2, steps = 8, dim = 6;
    std::vector<std::vector<float>> condbuf; for (int v = 0; v < V; ++v) condbuf.push_back(make_cond_buf(v));
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds; for (auto& b : condbuf) conds.push_back(b.data());
    std::vector<float> noise(dim, 0.f);
    for (float g : { 0.0f, 1.0f, 0.5f }) {
        FakeModel fm{ dim, {} };
        FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* c) { return fm(x, t, c); };
        SamplerParams sp; sp.steps = steps; sp.guidance_strength = g; sp.gi0 = 0.0; sp.gi1 = 1.0; sp.guidance_rescale = 0.0f; sp.rescale_t = 1.0;
        trellis::sample_flow_multi(fwd, noise, conds, negbuf.data(), sp, MultiCondMode::MultiDiffusion);
        check(("multidiffusion g=" + std::to_string(g) + ": negative called every step").c_str(),
              negative_count(fm.log) == steps, "got " + std::to_string(negative_count(fm.log)));
    }
}

// ---------------------------------------------------------------------------
// 4.1-8: zero/duplicate conditions -- 重複視点の平均は単一視点と同値
// ---------------------------------------------------------------------------
static void test_duplicate_conditions() {
    const int dim = 6, steps = 6;
    std::vector<float> negbuf = make_neg_buf();
    std::vector<float> noise(dim, 0.f);
    std::vector<float> dup_cond = { 70.0f };   // view "6" 相当の値を2視点で共有
    std::vector<const float*> conds2 = { dup_cond.data(), dup_cond.data() };
    std::vector<const float*> conds1 = { dup_cond.data() };

    FakeModel fm1{ dim, {} };
    FlowFwd fwd1 = [&](const std::vector<float>& x, float t, const float* c) { return fm1(x, t, c); };
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 3.0f; sp.gi0 = 0.0; sp.gi1 = 1.0; sp.guidance_rescale = 0.6f; sp.rescale_t = 1.4;
    std::vector<float> out1 = trellis::sample_flow_multi(fwd1, noise, conds1, negbuf.data(), sp, MultiCondMode::MultiDiffusion);

    FakeModel fm2{ dim, {} };
    FlowFwd fwd2 = [&](const std::vector<float>& x, float t, const float* c) { return fm2(x, t, c); };
    std::vector<float> out2 = trellis::sample_flow_multi(fwd2, noise, conds2, negbuf.data(), sp, MultiCondMode::MultiDiffusion);

    bool eq = out1.size() == out2.size();
    for (size_t k = 0; eq && k < out1.size(); ++k) if (out1[k] != out2[k]) eq = false;
    check("multidiffusion: duplicate-view average == single-view (identity)", eq);
}

// ---------------------------------------------------------------------------
// 4.1-9/11: V=1 は本番 sample_flow（無改変）とベースライン比較
// ---------------------------------------------------------------------------
static bool vec_bitexact(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k) if (std::memcmp(&a[k], &b[k], sizeof(float)) != 0) return false;
    return true;
}

static bool vec_close_tol(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k) {
        const float d = std::fabs(a[k] - b[k]);
        if (d > tol * std::max(1.0f, std::fabs(b[k]))) return false;
    }
    return true;
}

static void test_v1_collapse() {
    const int dim = 6, steps = 8;
    std::vector<float> cond1 = make_cond_buf(0);
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds1 = { cond1.data() };
    std::vector<float> noise(dim, 0.f);

    for (MultiCondMode mode : { MultiCondMode::Stochastic, MultiCondMode::MultiDiffusion }) {
        for (float g : { 0.0f, 1.0f, 3.0f }) {
            for (float gr : { 0.0f, 0.4f }) {
                for (auto interval : { std::pair<double,double>{0.0,1.0}, std::pair<double,double>{0.3,0.6} }) {
                    // multidiffusion + g==0 + rescale有効の組み合わせは仕様上値の一致を主張しない
                    // （ベースラインはg==0短絡でrescaleを一切通らないが、multidiffusionは
                    // raw positive平均からrescaleを計算するため構造的に異なる値になる）。
                    if (mode == MultiCondMode::MultiDiffusion && g == 0.0f && gr > 0.0f) continue;

                    FakeModel fm_base{ dim, {} };
                    FlowFwd fwd_base = [&](const std::vector<float>& x, float t, const float* c) { return fm_base(x, t, c); };
                    SamplerParams sp; sp.steps = steps; sp.guidance_strength = g; sp.gi0 = interval.first; sp.gi1 = interval.second;
                    sp.guidance_rescale = gr; sp.rescale_t = 1.0;
                    std::vector<float> baseline = trellis::sample_flow(fwd_base, noise, cond1.data(), negbuf.data(), sp);

                    FakeModel fm_multi{ dim, {} };
                    FlowFwd fwd_multi = [&](const std::vector<float>& x, float t, const float* c) { return fm_multi(x, t, c); };
                    std::vector<float> multi = trellis::sample_flow_multi(fwd_multi, noise, conds1, negbuf.data(), sp, mode);

                    std::string label = std::string("V=1 collapse (") + (mode == MultiCondMode::Stochastic ? "stochastic" : "multidiffusion")
                        + ", g=" + std::to_string(g) + ", gr=" + std::to_string(gr)
                        + ", gi=[" + std::to_string(interval.first) + "," + std::to_string(interval.second) + "])";
                    // stochasticは常にbit完全一致（式が同一）、multidiffusionは許容誤差で比較
                    // （g==1+rescale>0のULP非結合性を吸収するため。tol=1e-4は他の許容誤差と揃えた）。
                    const bool ok = (mode == MultiCondMode::Stochastic)
                        ? vec_bitexact(baseline, multi)
                        : vec_close_tol(baseline, multi, 1e-4f);
                    check(label.c_str(), ok);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4.1-9 (NaN/Inf): fake model が Inf を返しても sample_flow と同じ安全化を共有する
// ---------------------------------------------------------------------------
static void test_nan_inf_consistency() {
    const int dim = 4, steps = 5;
    std::vector<float> cond1 = make_cond_buf(0);
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds1 = { cond1.data() };
    std::vector<float> noise(dim, 0.f);
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 4.0f; sp.gi0 = 0.0; sp.gi1 = 1.0; sp.rescale_t = 1.0;

    auto inf_model = [&](const std::vector<float>&, float tscaled, const float*) {
        std::vector<float> out(dim, 1.0f);
        if (tscaled > 500.0f) out[0] = std::numeric_limits<float>::infinity();   // poison one late step
        return out;
    };
    FlowFwd fwd_base = inf_model;
    std::vector<float> baseline = trellis::sample_flow(fwd_base, noise, cond1.data(), negbuf.data(), sp);
    FlowFwd fwd_multi = inf_model;
    std::vector<float> multi_sto = trellis::sample_flow_multi(fwd_multi, noise, conds1, negbuf.data(), sp, MultiCondMode::Stochastic);
    std::vector<float> multi_md  = trellis::sample_flow_multi(fwd_multi, noise, conds1, negbuf.data(), sp, MultiCondMode::MultiDiffusion);
    check("NaN/Inf: stochastic V=1 matches sample_flow's safety net", vec_bitexact(baseline, multi_sto));
    check("NaN/Inf: multidiffusion V=1 matches sample_flow's safety net", vec_bitexact(baseline, multi_md));
    bool all_finite = true; for (float v : multi_sto) if (!std::isfinite(v)) all_finite = false;
    check("NaN/Inf: output is fully finite (safety net engaged)", all_finite);
}

// ---------------------------------------------------------------------------
// 4.1-10: cascade counter継続（構造: view列）
// ---------------------------------------------------------------------------
static void test_cascade_counter_continuation() {
    const int V = 3, steps = 5, dim = 6;
    std::vector<std::vector<float>> condbuf; for (int v = 0; v < V; ++v) condbuf.push_back(make_cond_buf(v));
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds; for (auto& b : condbuf) conds.push_back(b.data());
    std::vector<float> noise(dim, 0.f);
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 1.0f; sp.gi0 = 0.0; sp.gi1 = 1.0; sp.rescale_t = 1.0;

    int ctr = 0;
    FakeModel fm_lr{ dim, {} };
    FlowFwd fwd_lr = [&](const std::vector<float>& x, float t, const float* c) { return fm_lr(x, t, c); };
    trellis::sample_flow_multi(fwd_lr, noise, conds, negbuf.data(), sp, MultiCondMode::Stochastic, nullptr, &ctr);
    FakeModel fm_hr{ dim, {} };
    FlowFwd fwd_hr = [&](const std::vector<float>& x, float t, const float* c) { return fm_hr(x, t, c); };
    trellis::sample_flow_multi(fwd_hr, noise, conds, negbuf.data(), sp, MultiCondMode::Stochastic, nullptr, &ctr);

    std::vector<int> lr_pv = positive_views(fm_lr.log), hr_pv = positive_views(fm_hr.log);
    std::vector<int> expected_lr; for (int k = 0; k < steps; ++k) expected_lr.push_back(k % V);
    check("cascade: LR view sequence == [k % V]", lr_pv == expected_lr);
    check("cascade: HR first view continues the counter (not reset)",
          !hr_pv.empty() && hr_pv[0] == (steps % V));

    // 対照: 新しい (共有しない) counter なら 0 から再開する。
    FakeModel fm_reset{ dim, {} };
    FlowFwd fwd_reset = [&](const std::vector<float>& x, float t, const float* c) { return fm_reset(x, t, c); };
    trellis::sample_flow_multi(fwd_reset, noise, conds, negbuf.data(), sp, MultiCondMode::Stochastic);   // counter省略 = nullptr
    std::vector<int> reset_pv = positive_views(fm_reset.log);
    check("cascade control: a fresh (unshared) counter restarts at view 0", !reset_pv.empty() && reset_pv[0] == 0);
}

// ---------------------------------------------------------------------------
// 4.1-12: 決定性
// ---------------------------------------------------------------------------
static void test_determinism() {
    const int V = 4, steps = 12, dim = 6;
    std::vector<std::vector<float>> condbuf; for (int v = 0; v < V; ++v) condbuf.push_back(make_cond_buf(v));
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds; for (auto& b : condbuf) conds.push_back(b.data());
    std::vector<float> noise(dim, 0.f);
    SamplerParams sp; sp.steps = steps; sp.guidance_strength = 7.5f; sp.gi0 = 0.6; sp.gi1 = 1.0; sp.guidance_rescale = 0.5f; sp.rescale_t = 3.0;

    FakeModel fm1{ dim, {} }; FlowFwd fwd1 = [&](const std::vector<float>& x, float t, const float* c) { return fm1(x, t, c); };
    std::vector<float> out1 = trellis::sample_flow_multi(fwd1, noise, conds, negbuf.data(), sp, MultiCondMode::MultiDiffusion);
    FakeModel fm2{ dim, {} }; FlowFwd fwd2 = [&](const std::vector<float>& x, float t, const float* c) { return fm2(x, t, c); };
    std::vector<float> out2 = trellis::sample_flow_multi(fwd2, noise, conds, negbuf.data(), sp, MultiCondMode::MultiDiffusion);
    check("determinism: identical input run twice -> bit-exact output", vec_bitexact(out1, out2));

    // スレッド数変化: sample_flow_multi 自体は単一スレッド関数で内部並列化を一切行わないため、
    // これは「関数内部の並行実行」を突く検証ではない。ここで確認できるのは「他スレッドが
    // 同時に走っているプロセス内で、各スレッドが独立に呼んだ sample_flow_multi の結果が
    // スレッド数に関わらず自分自身の単独実行結果と bit-exact に一致する」ことだけ（誤って
    // 静的/グローバル可変状態を共有していれば検出できる）。真のマルチスレッドconcurrencyを
    // exerciseするテストではない。
    for (int nthreads : { 1, 8 }) {
        std::vector<std::vector<float>> results(nthreads);
        std::vector<std::thread> threads;
        for (int th = 0; th < nthreads; ++th) {
            threads.emplace_back([&, th]() {
                FakeModel fm{ dim, {} };
                FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* c) { return fm(x, t, c); };
                results[th] = trellis::sample_flow_multi(fwd, noise, conds, negbuf.data(), sp, MultiCondMode::MultiDiffusion);
            });
        }
        for (auto& th : threads) th.join();
        bool all_match = true;
        for (int th = 0; th < nthreads; ++th) if (!vec_bitexact(results[th], out1)) all_match = false;
        check(("determinism: " + std::to_string(nthreads) + " concurrent threads each -> bit-exact match with the single-threaded baseline").c_str(), all_match);
    }
}

// ---------------------------------------------------------------------------
// 4.1-13: 入力検証
// #71 は std::invalid_argument を投げる（旧トラックの std::runtime_error ではない）ため
// catch を std::exception に変更（例外が飛ぶこと自体の期待値は変えていない）。
// TASK-PORT F3 で flow_runner.cpp に2件の検証を追加済み: conds 要素の null チェック（null なら
// std::invalid_argument）と、stochastic 分岐の forward 戻り値サイズチェック（multidiffusion
// 分岐と同じ std::runtime_error）。これにより以下は SKIP/既知FAIL でなく通常の check() になる。
// ---------------------------------------------------------------------------
static void test_input_validation() {
    const int dim = 4;
    std::vector<float> cond1 = make_cond_buf(0);
    std::vector<float> negbuf = make_neg_buf();
    std::vector<float> noise(dim, 0.f);
    SamplerParams sp; sp.steps = 4;
    FlowFwd fwd = [&](const std::vector<float>&, float, const float*) { return std::vector<float>(dim, 0.f); };

    auto threw = [&](std::vector<const float*> conds, const float* neg, int steps) {
        SamplerParams s = sp; s.steps = steps;
        try { trellis::sample_flow_multi(fwd, noise, conds, neg, s, MultiCondMode::Stochastic); return false; }
        catch (const std::exception&) { return true; }
    };
    check("validation: empty conds throws", threw({}, negbuf.data(), 4));
    check("validation: null cond element throws", threw({ nullptr }, negbuf.data(), 4));
    check("validation: null neg_cond throws", threw({ cond1.data() }, nullptr, 4));
    check("validation: steps<=0 throws", threw({ cond1.data() }, negbuf.data(), 0));

    // fwd_bad_size returns dim+1 (LARGER than Nst=dim) -- stochastic 分岐は今 tmp.size() を
    // チェックするので std::runtime_error が飛ぶ（F3 で multidiffusion 分岐と揃えた）。
    FlowFwd fwd_bad_size = [&](const std::vector<float>&, float, const float*) { return std::vector<float>(dim + 1, 0.f); };
    bool threw_size = false;
    try { trellis::sample_flow_multi(fwd_bad_size, noise, { cond1.data() }, negbuf.data(), sp, MultiCondMode::Stochastic); }
    catch (const std::exception&) { threw_size = true; }
    check("validation: forward return-size mismatch throws (stochastic)", threw_size);
}

// ---------------------------------------------------------------------------
// 4.2: 独立fixture（tools/gen_trellis2_mv_sampler_fixture.py 生成）との突き合わせ
// ---------------------------------------------------------------------------
struct FixtureCase {
    std::string name, mode;
    int V = 0, steps = 0, dim = 0;
    double rescale_t = 1, gi0 = 0, gi1 = 1;
    float gs = 1, gr = 0;
    std::vector<CallEntry> calls;               // single-run cases
    std::vector<float> final_sample;
    // cascade-only fields
    std::vector<CallEntry> lr_calls, hr_cont_calls, hr_reset_calls;
    std::vector<float> lr_final, hr_cont_final, hr_reset_final;
    bool is_cascade = false;
};

static bool parse_fixture(const std::string& path, std::vector<FixtureCase>& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::string line;
    FixtureCase cur; bool in_case = false;
    auto push_floats = [](std::istringstream& iss, std::vector<float>& v) {
        float x; while (iss >> x) v.push_back(x);
    };
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key; iss >> key;
        if (key == "CASE") { cur = FixtureCase{}; in_case = true; iss >> cur.name; cur.is_cascade = (cur.name == "stochastic_cascade"); }
        else if (!in_case) { continue; }
        else if (key == "MODE") iss >> cur.mode;
        else if (key == "V") iss >> cur.V;
        else if (key == "STEPS") iss >> cur.steps;
        else if (key == "DIM") iss >> cur.dim;
        else if (key == "RESCALE_T") iss >> cur.rescale_t;
        else if (key == "GI0") iss >> cur.gi0;
        else if (key == "GI1") iss >> cur.gi1;
        else if (key == "GUIDANCE_STRENGTH") iss >> cur.gs;
        else if (key == "GUIDANCE_RESCALE") iss >> cur.gr;
        else if (key == "NCALLS" || key == "LR_NCALLS" || key == "HR_CONTINUED_NCALLS" || key == "HR_RESET_NCALLS") { /* count only, not needed */ }
        else if (key == "CALL") { int v; float t; iss >> v >> t; cur.calls.push_back({ v, t }); }
        else if (key == "LR_CALL") { int v; float t; iss >> v >> t; cur.lr_calls.push_back({ v, t }); }
        else if (key == "HR_CONTINUED_CALL") { int v; float t; iss >> v >> t; cur.hr_cont_calls.push_back({ v, t }); }
        else if (key == "HR_RESET_CALL") { int v; float t; iss >> v >> t; cur.hr_reset_calls.push_back({ v, t }); }
        else if (key == "FINAL") push_floats(iss, cur.final_sample);
        else if (key == "LR_FINAL") push_floats(iss, cur.lr_final);
        else if (key == "HR_CONTINUED_FINAL") push_floats(iss, cur.hr_cont_final);
        else if (key == "HR_RESET_FINAL") push_floats(iss, cur.hr_reset_final);
        else if (key == "END") { out.push_back(cur); in_case = false; }
    }
    return true;
}

// TSEQ blocks (raw t-schedule dumps, independent of any sampler run) live in the same fixture
// file as the CASE blocks but use a separate "TSEQ <name>" / "END" framing.
struct TSeqCase { std::string name; int steps = 0; double rescale_t = 0; std::vector<double> values; };

static bool parse_tseq_cases(const std::string& path, std::vector<TSeqCase>& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::string line;
    TSeqCase cur; bool in_tseq = false;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key; iss >> key;
        if (key == "TSEQ") { cur = TSeqCase{}; in_tseq = true; iss >> cur.name; }
        else if (!in_tseq) { continue; }
        else if (key == "TSEQ_STEPS") iss >> cur.steps;
        else if (key == "TSEQ_RESCALE_T") iss >> cur.rescale_t;
        else if (key == "TSEQ_VALUES") { double v; while (iss >> v) cur.values.push_back(v); }
        else if (key == "END") { out.push_back(cur); in_tseq = false; }
    }
    return true;
}

// 4.2b: flow_t_schedule() を、その式を一切参照しない独立生成の double 列と bit-exact 比較する。
// CASE fixtureのfinal-sample比較は許容誤差1e-4を使うため、境界(t==0.6)のようなexact-decimal
// 性質を検出できない -- この比較だけがそれを検証する。加えて shape_slat_sampler の
// 本番パラメータで「t==0.6 の step が inside と判定される」ことを明示的に assert する
// (TASK-PORT P1 が移植した境界が将来のリファクタで黙って反転しないためのガード)。
static void test_tseq_bitexact(const std::vector<TSeqCase>& tseqs) {
    for (const auto& tc : tseqs) {
        std::vector<double> got = trellis::flow_t_schedule(tc.steps, tc.rescale_t);
        bool exact = (got.size() == tc.values.size());
        std::string detail;
        if (exact) {
            for (size_t i = 0; i < got.size(); ++i) {
                if (got[i] != tc.values[i]) {
                    exact = false;
                    char buf[160];
                    snprintf(buf, sizeof buf, "index %zu: flow_t_schedule=%.17g fixture=%.17g", i, got[i], tc.values[i]);
                    detail = buf;
                    break;
                }
            }
        } else {
            detail = "size mismatch";
        }
        check(("tseq " + tc.name + ": flow_t_schedule() bit-exact vs independent numpy dump").c_str(), exact, detail);
    }

    // shape_slat_sampler_production の唯一の境界 step (index 8, t==0.6 ちょうど) が inside と
    // 判定されること、かつ inside/outside の合計が仕様通り 9/3 になることを明示的に assert する。
    for (const auto& tc : tseqs) {
        if (tc.name != "shape_slat_sampler_production") continue;
        int inside = 0, outside = 0;
        for (int i = 0; i < tc.steps; ++i) {
            if (trellis::flow_in_guidance_interval(tc.values[i], 0.6, 1.0)) ++inside; else ++outside;
        }
        check("tseq shape_slat_sampler_production: step 8 (t==0.6) is classified inside the guidance interval",
              trellis::flow_in_guidance_interval(tc.values[8], 0.6, 1.0));
        check("tseq shape_slat_sampler_production: inside==9 (regression guard for the P1 boundary fix)",
              inside == 9);
        check("tseq shape_slat_sampler_production: outside==3", outside == 3);
    }
}

static bool vec_close(const std::vector<float>& a, const std::vector<float>& b, float tol, std::string& detail) {
    if (a.size() != b.size()) { detail = "size mismatch"; return false; }
    for (size_t k = 0; k < a.size(); ++k) {
        const float d = std::fabs(a[k] - b[k]);
        const float scale = std::max(1.0f, std::fabs(b[k]));
        if (d > tol * scale) {
            char buf[160]; snprintf(buf, sizeof buf, "index %zu: got %.9g expected %.9g (|diff|=%.3g)", k, a[k], b[k], d);
            detail = buf; return false;
        }
    }
    return true;
}

static void run_fixture_case(const FixtureCase& c) {
    const float TOL = 1e-4f;
    std::vector<std::vector<float>> condbuf; for (int v = 0; v < c.V; ++v) condbuf.push_back(make_cond_buf(v));
    std::vector<float> negbuf = make_neg_buf();
    std::vector<const float*> conds; for (auto& b : condbuf) conds.push_back(b.data());
    std::vector<float> noise(c.dim, 0.f);
    SamplerParams sp; sp.steps = c.steps; sp.guidance_strength = c.gs; sp.gi0 = c.gi0; sp.gi1 = c.gi1;
    sp.guidance_rescale = c.gr; sp.rescale_t = c.rescale_t;
    MultiCondMode mode = (c.mode == "stochastic") ? MultiCondMode::Stochastic : MultiCondMode::MultiDiffusion;

    if (!c.is_cascade) {
        FakeModel fm{ c.dim, {} };
        FlowFwd fwd = [&](const std::vector<float>& x, float t, const float* cc) { return fm(x, t, cc); };
        std::vector<float> out = trellis::sample_flow_multi(fwd, noise, conds, negbuf.data(), sp, mode);
        std::string detail;
        check(("fixture " + c.name + ": final sample matches independent oracle").c_str(),
              vec_close(out, c.final_sample, TOL, detail), detail);
        std::vector<int> got_views; for (auto& e : fm.log) got_views.push_back(e.view);
        std::vector<int> exp_views; for (auto& e : c.calls) exp_views.push_back(e.view);
        check(("fixture " + c.name + ": call view sequence matches independent oracle").c_str(), got_views == exp_views);
    } else {
        int ctr = 0;
        FakeModel fm_lr{ c.dim, {} };
        FlowFwd fwd_lr = [&](const std::vector<float>& x, float t, const float* cc) { return fm_lr(x, t, cc); };
        std::vector<float> lr_out = trellis::sample_flow_multi(fwd_lr, noise, conds, negbuf.data(), sp, mode, nullptr, &ctr);
        FakeModel fm_hr{ c.dim, {} };
        FlowFwd fwd_hr = [&](const std::vector<float>& x, float t, const float* cc) { return fm_hr(x, t, cc); };
        std::vector<float> hr_cont = trellis::sample_flow_multi(fwd_hr, noise, conds, negbuf.data(), sp, mode, nullptr, &ctr);
        FakeModel fm_reset{ c.dim, {} };
        FlowFwd fwd_reset = [&](const std::vector<float>& x, float t, const float* cc) { return fm_reset(x, t, cc); };
        std::vector<float> hr_reset = trellis::sample_flow_multi(fwd_reset, noise, conds, negbuf.data(), sp, mode);

        std::string d1, d2, d3;
        check("fixture cascade: LR final matches independent oracle", vec_close(lr_out, c.lr_final, TOL, d1), d1);
        check("fixture cascade: HR (continued counter) final matches independent oracle", vec_close(hr_cont, c.hr_cont_final, TOL, d2), d2);
        check("fixture cascade: HR (reset counter) final matches independent oracle", vec_close(hr_reset, c.hr_reset_final, TOL, d3), d3);
    }
}

int main(int argc, char** argv) {
    const std::string fixture_path = argc > 1 ? argv[1] : "tests/fixtures/trellis2_mv_sampler/sampler_oracle.txt";

    test_stochastic_order();
    test_stochastic_g0_counter_advances();
    test_stochastic_steps_lt_views();
    test_stochastic_interval_edges();
    test_multidiffusion_call_counts();
    test_no_shortcircuit_strengths();
    test_duplicate_conditions();
    test_v1_collapse();
    test_nan_inf_consistency();
    test_cascade_counter_continuation();
    test_determinism();
    test_input_validation();

    std::vector<FixtureCase> cases; std::string err;
    if (!parse_fixture(fixture_path, cases, err)) {
        printf("[FAIL] fixture: %s\n", err.c_str());
        ++g_fail;
    } else {
        check("fixture: file parsed with the expected number of cases", cases.size() == 5,
              "got " + std::to_string(cases.size()));
        for (const auto& c : cases) run_fixture_case(c);
    }

    std::vector<TSeqCase> tseqs; std::string tseq_err;
    if (!parse_tseq_cases(fixture_path, tseqs, tseq_err)) {
        printf("[FAIL] tseq: %s\n", tseq_err.c_str());
        ++g_fail;
    } else {
        check("tseq: file parsed with the expected number of raw-schedule dumps", tseqs.size() == 3,
              "got " + std::to_string(tseqs.size()));
        test_tseq_bitexact(tseqs);
    }

    printf("\n");
    printf("%d FAIL (trellis-test-flow-multi, CPU, no model weights)\n", g_fail);
    // TASK-PORT F3 (2026-09-21): flow_runner.cpp now validates both null cond elements and the
    // stochastic branch's forward return size, so g_fail==0 (74/74) is the expected steady state.
    return g_fail ? 1 : 0;
}
