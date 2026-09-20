// Validate the C++/GGML Pixal3D SS-flow DiT (dense, ProjectAttention) against PyTorch golden
// tensors. Mirrors test_ss_flow.cpp; the only differences are two cross-attn conditioning
// inputs (global_cond, proj_cond) instead of one, and the extra blk0_global_out/blk0_proj_out/
// blk0_cross_out intermediates ProjectAttention exposes.
//   trellis-test-pixal3d-ss-flow <pixal3d_ss_flow.gguf> <ref_dir> [gpu]
#include "trellis_model.h"
#include "dit.h"
#include "trellis_args.h"
#include "npy.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using std::vector; using std::string;

static void compare(const char* name, const vector<float>& mine, const npy::Array& ref) {
    if ((int64_t)mine.size() != ref.numel()) {
        printf("  %-20s SIZE MISMATCH mine=%zu ref=%lld\n", name, mine.size(), (long long)ref.numel());
        return;
    }
    double maxabs = 0, sumabs = 0, refmax = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < mine.size(); ++i) {
        double d = std::fabs((double)mine[i] - ref.data[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)ref.data[i]));
        dot += (double)mine[i] * ref.data[i]; na += (double)mine[i] * mine[i]; nb += (double)ref.data[i] * ref.data[i];
    }
    double rel = refmax > 0 ? maxabs / refmax : maxabs;
    double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    printf("  %-20s max|d|=%.4e mean|d|=%.4e  rel=%.4e cos=%.7f  %s\n",
           name, maxabs, sumabs / mine.size(), rel, cos, rel < 2e-2 ? "OK" : "**");
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <pixal3d_ss_flow.gguf> <ref_dir> [gpu]\n", argv[0]); return 1; }
    const string gguf = argv[1], ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;

    trellis::Model m = trellis::Model::load(gguf, gpu);
    printf("loaded %s (%zu tensors) on backend %s\n", m.arch.c_str(), m.tensors.size(), ggml_backend_name(m.backend));

    // ---- reference inputs ----
    npy::Array x  = npy::load(ref + "/input_x.npy");        // [1,8,16,16,16]
    npy::Array tA = npy::load(ref + "/input_t.npy");        // [1]
    npy::Array gA = npy::load(ref + "/input_global.npy");   // [1,5,1024]  global (CLS + 4 register) cond
    npy::Array jA = npy::load(ref + "/input_proj.npy");     // [1,4096,1024]  proj_cond, one per latent token
    const int64_t Cin = x.shape[1], R = x.shape[2], L = R * R * R;
    const int64_t Lc = gA.shape[1], Dc = gA.shape[2];
    const int64_t Dp = jA.shape[2];
    const float t = tA.data[0];
    printf("inputs: Cin=%lld L=%lld Lc=%lld Dc=%lld Dp=%lld t=%.1f\n",
           (long long)Cin, (long long)L, (long long)Lc, (long long)Dc, (long long)Dp, t);

    trellis::DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(f32 weight compute)\n"); }
    if (getenv("TRELLIS_NOFA")) { trellis::g_no_fa = true; printf("(no-fa: soft_max path)\n"); }
    bool detected = trellis::dit_detect_proj_attn(m, p);
    printf("proj_attn detected=%d d_proj=%d (npy Dp=%lld)\n", detected, p.d_proj, (long long)Dp);
    if (!detected) { fprintf(stderr, "expected a Pixal3D checkpoint (blocks.0.cross_attn.proj_linear.weight missing)\n"); return 1; }
    if (p.d_proj != (int)Dp) { fprintf(stderr, "d_proj mismatch: checkpoint=%d fixture=%lld\n", p.d_proj, (long long)Dp); return 1; }

    // ---- host-prepared inputs ----
    // h0[c + Cin*sp] = x[c*L + sp]
    vector<float> h0(Cin * L);
    for (int64_t c = 0; c < Cin; ++c) for (int64_t sp = 0; sp < L; ++sp) h0[c + Cin * sp] = x.data[c * L + sp];
    // timestep embedding (cos|sin), 256
    vector<float> tfreq(256);
    for (int j = 0; j < 128; ++j) { float f = std::exp(-std::log(10000.f) * j / 128.f); tfreq[j] = std::cos(t * f); tfreq[128 + j] = std::sin(t * f); }
    // 3D rope cos/sin: data[token*64 + pair]
    const int half = p.head_dim / 2;       // 64
    vector<float> rcos(L * half), rsin(L * half);
    float freqs[21]; for (int j = 0; j < 21; ++j) freqs[j] = 1.0f / std::pow(10000.f, j / 21.0f);
    for (int64_t tok = 0; tok < L; ++tok) {
        int cx = tok / (R * R), cy = (tok / R) % R, cz = tok % R;
        for (int pp = 0; pp < half; ++pp) {
            float ang = 0;
            if (pp < 21) ang = cx * freqs[pp];
            else if (pp < 42) ang = cy * freqs[pp - 21];
            else if (pp < 63) ang = cz * freqs[pp - 42];
            rcos[tok * half + pp] = std::cos(ang); rsin[tok * half + pp] = std::sin(ang);
        }
    }
    { // sanity-check rope against reference (same fixture as test_ss_flow.cpp, if present)
        FILE* f = fopen((ref + "/rope_cos.npy").c_str(), "rb");
        if (f) { fclose(f);
            npy::Array rc = npy::load(ref + "/rope_cos.npy");
            double md = 0; for (size_t i = 0; i < rcos.size(); ++i) md = std::max(md, std::fabs((double)rcos[i] - rc.data[i]));
            printf("rope_cos vs ref max|d|=%.3e\n", md);
        }
    }
    // proj_cond: fixture is [1, L, Dp] row-major (token-major); ggml wants [Dp, L] channel-major.
    vector<float> pj(Dp * L);
    for (int64_t tok = 0; tok < L; ++tok) for (int64_t d = 0; d < Dp; ++d) pj[d + Dp * tok] = jA.data[tok * Dp + d];

    // ---- build graph ----
    size_t meta = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(32768, false) + (1 << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ggml_context* c = ggml_init(ip);
    ggml_tensor* gh0  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, L);          ggml_set_input(gh0);
    ggml_tensor* gtf  = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);             ggml_set_input(gtf);
    ggml_tensor* gcd  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Dc, Lc);          ggml_set_input(gcd);
    ggml_tensor* gcos = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, L);   ggml_set_input(gcos);
    ggml_tensor* gsin = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, L);   ggml_set_input(gsin);
    ggml_tensor* gpj  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Dp, L);           ggml_set_input(gpj);

    std::map<string, ggml_tensor*> inter;
    ggml_tensor* out = trellis::build_dit_dense(c, m, p, gh0, gtf, gcd, gcos, gsin, &inter, gpj);

    ggml_cgraph* g = ggml_new_graph_custom(c, 32768, false);
    ggml_build_forward_expand(g, out);
    for (auto& [k, v] : inter) ggml_set_output(v);
    trellis::check_graph_supported(m.backend, g, "test_pixal3d_ss_flow");

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) { fprintf(stderr, "alloc failed\n"); return 1; }

    ggml_backend_tensor_set(gh0,  h0.data(),    0, h0.size()    * 4);
    ggml_backend_tensor_set(gtf,  tfreq.data(), 0, tfreq.size() * 4);
    ggml_backend_tensor_set(gcd,  gA.data.data(), 0, gA.numel()  * 4);
    ggml_backend_tensor_set(gcos, rcos.data(),  0, rcos.size()  * 4);
    ggml_backend_tensor_set(gsin, rsin.data(),  0, rsin.size()  * 4);
    ggml_backend_tensor_set(gpj,  pj.data(),    0, pj.size()    * 4);

    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }

    printf("\ncomparisons (rel<2e-2 = OK):\n");
    // {fixture name, graph intermediate name}; missing fixture files are skipped silently.
    std::vector<std::pair<string, string>> names = {
        {"after_input_layer", "after_input_layer"}, {"t_emb_mod", "t_emb_mod"},
        {"blk0_msa_out", "blk0_msa"}, {"blk0_global_out", "blk0_global_out"}, {"blk0_proj_out", "blk0_proj_out"},
        {"blk0_cross_out", "blk0_cross_out"}, {"blk0_mlp_out", "blk0_mlp"}};
    for (int i = 0; i < p.n_blocks; ++i) {
        const string n = "after_block" + std::to_string(i);
        names.push_back({n, n});
        if (i == 14)
            for (const char* s : {"msa_out", "global_out", "proj_out", "cross_out", "mlp_out"}) {
                const string f = string("blk15_") + s;
                const string g = (string(s) == "msa_out" || string(s) == "mlp_out") ? f.substr(0, f.size() - 4) : f;
                names.push_back({f, g});
            }
    }
    names.push_back({"prefinal", "prefinal"});
    for (const auto& [fn, gn] : names) {
        FILE* f = fopen((ref + "/" + fn + ".npy").c_str(), "rb");
        if (!f) { if (fn.rfind("after_block", 0) != 0 || fn == "after_block29") printf("  %-20s (no golden dump)\n", fn.c_str()); continue; }
        fclose(f);
        auto it = inter.find(gn);
        if (it == inter.end()) { printf("  %-20s (no graph intermediate %s)\n", fn.c_str(), gn.c_str()); continue; }
        compare(fn.c_str(), trellis::tensor_to_f32(it->second), npy::load(ref + "/" + fn + ".npy"));
    }
    // output: remap mine[c+Cin*sp] -> ref[c*L+sp]
    {
        vector<float> mine = trellis::tensor_to_f32(out);   // [Cin, L] ne0=Cin
        vector<float> remap(Cin * L);
        for (int64_t cc = 0; cc < Cin; ++cc) for (int64_t sp = 0; sp < L; ++sp) remap[cc * L + sp] = mine[cc + Cin * sp];
        compare("output", remap, npy::load(ref + "/output.npy"));
    }

    ggml_gallocr_free(alloc); ggml_free(c); m.free();
    return 0;
}
