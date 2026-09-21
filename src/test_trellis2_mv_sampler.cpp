#include "flow_runner.h"
#include <cmath>
#include <cstdio>
#include <vector>

using std::vector;

static int fail = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); ++fail; }
}

static bool close_vec(const vector<float>& a, const vector<float>& b, float atol = 2e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > atol) return false;
    return true;
}

int main() {
    const vector<float> noise = {0.5f, -1.0f, 0.25f, 1.5f};
    float c0 = 0.25f, c1 = -0.5f, c2 = 1.0f, neg = -0.75f;
    const vector<const float*> conds = {&c0, &c1, &c2};

    trellis::SamplerParams sp;
    sp.steps = 4;
    sp.rescale_t = 3.0f;
    sp.guidance_strength = 2.5f;
    sp.guidance_rescale = 0.4f;
    sp.gi0 = 0.35f;
    sp.gi1 = 0.9f;
    sp.sigma_min = 1e-5f;

    int pos_calls[3] = {0,0,0}, neg_calls = 0;
    auto fwd = [&](const vector<float>& x, float tscaled, const float* c) {
        int ci = -1;
        if (c == &c0) ci = 0;
        else if (c == &c1) ci = 1;
        else if (c == &c2) ci = 2;
        else if (c == &neg) ++neg_calls;
        if (ci >= 0) ++pos_calls[ci];
        vector<float> y(x.size());
        const float t = tscaled / 1000.0f;
        for (size_t i = 0; i < x.size(); ++i)
            y[i] = 0.15f*x[i] + (float)(i+1)*0.2f*(*c) +
                   0.0003f*tscaled*((i&1) ? -1.0f : 1.0f);
        return y;
    };

    auto st = trellis::sample_flow_multi(
        fwd, noise, conds, &neg, sp, trellis::MultiCondMode::Stochastic);
    check(close_vec(st, {-0.11309146f,-1.43191948f,-1.06779335f,0.09845656f}),
          "stochastic final matches pinned Python oracle");
    check(pos_calls[0] == 2 && pos_calls[1] == 1 && pos_calls[2] == 1,
          "stochastic cycles views 0,1,2,0 exactly");
    check(neg_calls == 3, "stochastic negative forward only on guided steps");

    pos_calls[0]=pos_calls[1]=pos_calls[2]=0; neg_calls=0;
    auto md = trellis::sample_flow_multi(
        fwd, noise, conds, &neg, sp, trellis::MultiCondMode::MultiDiffusion);
    check(close_vec(md, {-0.07511912f,-1.37591274f,-0.96573399f,0.26588240f}),
          "multidiffusion final matches pinned Python oracle");
    check(pos_calls[0] == 4 && pos_calls[1] == 4 && pos_calls[2] == 4,
          "multidiffusion evaluates every view every step");
    check(neg_calls == 3, "multidiffusion evaluates one negative on guided steps");

    // V=1: both formulations produce the same math.
    float only = 0.375f;
    vector<const float*> one = {&only};
    auto one_st = trellis::sample_flow_multi(
        fwd, noise, one, &neg, sp, trellis::MultiCondMode::Stochastic);
    auto one_md = trellis::sample_flow_multi(
        fwd, noise, one, &neg, sp, trellis::MultiCondMode::MultiDiffusion);
    check(close_vec(one_st, one_md, 2e-5f), "V=1 stochastic == multidiffusion");

    // PR #104 cascade quirk/contract: one shape injection context spans LR + HR.
    // With 12 steps and V=5, LR consumes indices 0..4,0..4,0,1 and HR starts at view2.
    {
        float v0=0.f, v1=1.f, v2=2.f, v3=3.f, v4=4.f;
        vector<const float*> five = {&v0,&v1,&v2,&v3,&v4};
        int calls[5] = {0,0,0,0,0};
        int counter = 0;
        trellis::SamplerParams cp = sp;
        cp.steps = 12;
        cp.guidance_strength = 1.0f;
        cp.guidance_rescale = 0.0f;
        auto count_fwd = [&](const vector<float>& x, float, const float* c) {
            const float* vv[5] = {&v0,&v1,&v2,&v3,&v4};
            for (int q=0;q<5;++q) if (c==vv[q]) calls[q]++;
            return vector<float>(x.size(), 0.0f);
        };
        (void)trellis::sample_flow_multi(count_fwd, noise, five, &neg, cp,
                                          trellis::MultiCondMode::Stochastic, nullptr, &counter);
        check(counter == 12, "shared stochastic counter advances by LR step count");
        int before[5]; for(int q=0;q<5;++q) before[q]=calls[q];
        (void)trellis::sample_flow_multi(count_fwd, noise, five, &neg, cp,
                                          trellis::MultiCondMode::Stochastic, nullptr, &counter);
        check(counter == 24, "shared stochastic counter advances through HR");
        check(calls[2] == before[2] + 3, "HR starts at view2 for V=5 after 12 LR steps");
        check(calls[0] == before[0] + 2 && calls[1] == before[1] + 2,
              "HR shared schedule wraps after starting at view2");
    }

    // PR #104 quirk: multidiffusion still does one negative forward at gs=1;
    // stochastic delegates to the existing CFG mixin semantics and skips it.
    sp.guidance_strength = 1.0f;
    neg_calls = 0;
    (void)trellis::sample_flow_multi(
        fwd, noise, one, &neg, sp, trellis::MultiCondMode::Stochastic);
    check(neg_calls == 0, "stochastic gs=1 skips negative forward");
    neg_calls = 0;
    (void)trellis::sample_flow_multi(
        fwd, noise, one, &neg, sp, trellis::MultiCondMode::MultiDiffusion);
    check(neg_calls == 3, "multidiffusion gs=1 preserves PR104 negative calls");

    std::printf(fail ? "FAIL: %d check(s)\n" : "PASS\n", fail);
    return fail ? 1 : 0;
}
