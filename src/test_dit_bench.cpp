// trellis-test-dit-bench: time forwards of one sparse Pixal3D flow DiT at a chosen token count
// without the rest of the pipeline (no conditioning, no decoder, no postprocess). Random
// coords / latent / cond -- this measures speed only, never quality. With --profile it prints the
// same [prof] breakdown the CLI does (flow_runner.cpp::DitRunner::profile_forward).
//
// Usage: trellis-test-dit-bench <flow.gguf> [N=17612] [forwards=4] [gpu=0] [--profile] [--no-fa]
//   gpu: -1 = ggml CPU backend, 0 = the build's GPU device.
// Prints one line per forward and a final "bench: N=... mean of the last (forwards-1) = X s".
#include "flow_runner.h"
#include "trellis_model.h"
#include "trellis_args.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <flow.gguf> [N] [forwards] [gpu] [--profile] [--no-fa]\n", argv[0]); return 2; }
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--profile")) trellis::g_profile = true;
        else if (!strcmp(argv[i], "--no-fa")) trellis::g_no_fa = true;
        else pos.push_back(argv[i]);
    }
    const std::string gguf = pos[0];
    const int N = pos.size() > 1 ? atoi(pos[1].c_str()) : 17612;
    const int n_fwd = pos.size() > 2 ? atoi(pos[2].c_str()) : 4;
    const int gpu = pos.size() > 3 ? atoi(pos[3].c_str()) : 0;
    setvbuf(stdout, nullptr, _IONBF, 0);

    trellis::Model m = trellis::Model::load(gguf, gpu);
    trellis::DiTParams p;
    p.in_ch = 32; p.out_ch = 32; p.d_cond = 1024;     // shape flow; the tex flow uses in_ch 64
    if (gguf.find("tex_flow") != std::string::npos) p.in_ch = 64;
    if (!trellis::dit_detect_proj_attn(m, p)) { fprintf(stderr, "not a Pixal3D ProjectAttention checkpoint\n"); return 1; }
    const int Lc = 5;

    // Random distinct voxel coords in a 64^3 grid (RoPE tables come from them), random inputs.
    std::mt19937 rng(1);
    std::vector<std::array<int, 3>> coords;
    {
        std::vector<int> cells(64 * 64 * 64);
        for (int i = 0; i < (int)cells.size(); ++i) cells[i] = i;
        std::shuffle(cells.begin(), cells.end(), rng);
        for (int i = 0; i < N; ++i) coords.push_back({ cells[i] / 4096, (cells[i] / 64) % 64, cells[i] % 64 });
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t)p.in_ch * N), cond((size_t)p.d_cond * Lc), proj((size_t)p.d_proj * N);
    for (auto& v : x) v = nd(rng);
    for (auto& v : cond) v = nd(rng);
    for (auto& v : proj) v = nd(rng);

    trellis::DitRunner* run = trellis::make_sparse_runner(m, p, coords, Lc);
    printf("bench: %s  N=%d  d_proj=%d  activations %.0f MB  %s%s\n", gguf.c_str(), N, p.d_proj,
           run->alloc_bytes() / 1048576.0, trellis::g_no_fa ? "exact SDPA" : "FlashAttention",
           trellis::g_profile ? "  --profile" : "");
    double sum = 0; int cnt = 0;
    for (int i = 0; i < n_fwd; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> out = run->forward(x, 1000.0f * (1.0f - i / 12.0f), cond.data(), proj.data());
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        size_t bad = 0; for (float v : out) if (!std::isfinite(v)) bad++;
        printf("bench: forward #%d  %.2fs  (nonfinite %zu/%zu)%s\n", i, dt, bad, out.size(),
               trellis::g_profile && i == 1 ? "  [includes the profiling passes]" : "");
        if (i >= 1 && !(trellis::g_profile && i == 1)) { sum += dt; cnt++; }
    }
    if (cnt) printf("bench: N=%d  mean whole forward (excluding #0%s) = %.2fs over %d\n", N,
                    trellis::g_profile ? " and the profiled #1" : "", sum / cnt, cnt);
    delete run;
    return 0;
}
