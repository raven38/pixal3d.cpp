// Validate the C++/GGML Pixal3D sparse SLAT flow DiT (ElasticSLatFlowModel, ProjectAttention)
// against PyTorch golden tensors, layer by layer. Mirrors test_shape_flow.cpp (same build_dit_dense
// builder, RoPE filled from sparse coords) with the extra proj_cond input and the
// blk0_global_out/blk0_proj_out/blk0_cross_out intermediates ProjectAttention exposes.
//   trellis-test-pixal3d-slat-flow <pixal3d_shape_flow_512.gguf> <ref_dir> [gpu]
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

static bool compare(const char* name, const vector<float>& mine, const npy::Array& ref) {
    if ((int64_t)mine.size() != ref.numel()) {
        printf("  %-20s SIZE MISMATCH mine=%zu ref=%lld\n", name, mine.size(), (long long)ref.numel());
        return false;
    }
    double maxabs = 0, sumabs = 0, refmax = 0;
    for (size_t i = 0; i < mine.size(); ++i) {
        double d = std::fabs((double)mine[i] - ref.data[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)ref.data[i]));
    }
    const double rel = refmax > 0 ? maxabs / refmax : maxabs;
    const bool ok = rel < 2e-2;
    printf("  %-20s max|d|=%.4e mean|d|=%.4e  rel=%.4e  %s\n",
           name, maxabs, sumabs / mine.size(), rel, ok ? "OK" : "**");
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <pixal3d_shape_flow_512.gguf> <ref_dir> [gpu]\n", argv[0]); return 1; }
    const string gguf = argv[1], ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;

    if (getenv("TRELLIS_NOFA")) { trellis::g_no_fa = true; printf("(no-fa: soft_max path)\n"); }
    trellis::Model m = trellis::Model::load(gguf, gpu);
    printf("loaded %s (%zu tensors)\n", m.arch.c_str(), m.tensors.size());

    npy::Array co = npy::load(ref + "/coords.npy");         // [N,3] or [N,4] with leading batch column
    npy::Array x  = npy::load(ref + "/input_x.npy");        // [N,32]
    npy::Array tA = npy::load(ref + "/input_t.npy");        // [1]
    npy::Array gA = npy::load(ref + "/input_global.npy");   // [1,5,1024]  global (CLS + 4 register) cond
    npy::Array jA = npy::load(ref + "/input_proj.npy");     // [N,2048]  proj_cond, one per active voxel
    const int64_t L = x.shape[0], Cin = x.shape[1];
    const int64_t Lc = gA.shape[1], Dc = gA.shape[2];
    const int64_t Dp = jA.shape[1];
    const float t = tA.data[0];
    printf("inputs: L=%lld Cin=%lld Lc=%lld Dc=%lld Dp=%lld t=%.1f\n",
           (long long)L, (long long)Cin, (long long)Lc, (long long)Dc, (long long)Dp, t);

    trellis::DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(f32 weight compute)\n"); }
    bool detected = trellis::dit_detect_proj_attn(m, p);
    printf("proj_attn detected=%d d_proj=%d (npy Dp=%lld)\n", detected, p.d_proj, (long long)Dp);
    if (!detected) { fprintf(stderr, "expected a Pixal3D checkpoint (blocks.0.cross_attn.proj_linear.weight missing)\n"); return 1; }
    if (p.d_proj != (int)Dp) { fprintf(stderr, "d_proj mismatch: checkpoint=%d fixture=%lld\n", p.d_proj, (long long)Dp); return 1; }

    // x is already [N,32] row-major = ggml [32, N] channel-major: same bytes.
    vector<float> h0(x.data.begin(), x.data.begin() + (size_t)Cin * L);
    // proj_cond [N, Dp] row-major -> ggml [Dp, N] channel-major: same bytes too.
    vector<float> pj(jA.data.begin(), jA.data.begin() + (size_t)Dp * L);

    vector<float> tfreq(256);
    for (int j = 0; j < 128; ++j) {
        const float f = std::exp(-std::log(10000.f) * j / 128.f);
        tfreq[j] = std::cos(t * f); tfreq[128 + j] = std::sin(t * f);
    }

    // 3D RoPE over the SPARSE coords (the only thing that differs from the dense SS path)
    const int half = p.head_dim / 2;   // 64
    vector<float> rcos((size_t)L * half), rsin((size_t)L * half);
    float freqs[21]; for (int j = 0; j < 21; ++j) freqs[j] = 1.0f / std::pow(10000.f, j / 21.0f);
    const float* cd = co.data.data();
    const int cw = (int)co.shape[1], c0 = cw - 3;   // [N,3] (x,y,z) or [N,4] (batch,x,y,z)
    for (int64_t tok = 0; tok < L; ++tok) {
        const int cx = (int)cd[tok*cw + c0], cy = (int)cd[tok*cw + c0 + 1], cz = (int)cd[tok*cw + c0 + 2];
        for (int pp = 0; pp < half; ++pp) {
            float ang = 0;
            if (pp < 21)      ang = cx * freqs[pp];
            else if (pp < 42) ang = cy * freqs[pp - 21];
            else if (pp < 63) ang = cz * freqs[pp - 42];
            rcos[tok * half + pp] = std::cos(ang); rsin[tok * half + pp] = std::sin(ang);
        }
    }

    // Generous: chunked attention adds ~8 tensors per chunk per attention, and a 30-block DiT
    // with 2 attentions each is one graph. Meta is host RAM for structs only (~370 B each).
    size_t meta = ggml_tensor_overhead() * 131072 + ggml_graph_overhead_custom(262144, false) + (1 << 20);
    ggml_context* c = ggml_init({ meta, nullptr, true });
    ggml_tensor* gh0  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, L);         ggml_set_input(gh0);
    ggml_tensor* gtf  = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);            ggml_set_input(gtf);
    ggml_tensor* gcd  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Dc, Lc);         ggml_set_input(gcd);
    ggml_tensor* gcos = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, L);  ggml_set_input(gcos);
    ggml_tensor* gsin = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, L);  ggml_set_input(gsin);
    ggml_tensor* gpj  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Dp, L);          ggml_set_input(gpj);

    std::map<string, ggml_tensor*> inter;
    ggml_tensor* out = trellis::build_dit_dense(c, m, p, gh0, gtf, gcd, gcos, gsin, &inter, gpj);

    ggml_cgraph* g = ggml_new_graph_custom(c, 262144, false);
    ggml_build_forward_expand(g, out);
    for (auto& [k, v] : inter) ggml_set_output(v);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) { fprintf(stderr, "alloc failed\n"); return 1; }

    ggml_backend_tensor_set(gh0,  h0.data(),      0, h0.size()   * 4);
    ggml_backend_tensor_set(gtf,  tfreq.data(),   0, tfreq.size()* 4);
    ggml_backend_tensor_set(gcd,  gA.data.data(), 0, gA.numel()  * 4);
    ggml_backend_tensor_set(gcos, rcos.data(),    0, rcos.size() * 4);
    ggml_backend_tensor_set(gsin, rsin.data(),    0, rsin.size() * 4);
    ggml_backend_tensor_set(gpj,  pj.data(),      0, pj.size()   * 4);

    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }

    printf("\nwalking the DiT (rel<2e-2 = OK); the FIRST ** is where it breaks:\n");
    const char* names[] = { "after_input_layer", "t_emb_mod", "blk0_global_out", "blk0_proj_out", "blk0_cross_out",
                            "after_block0", "after_block1", "after_block29", "prefinal" };
    for (const char* n : names) {
        if (!inter.count(n)) continue;
        FILE* f = fopen((ref + "/" + n + ".npy").c_str(), "rb");
        if (!f) { printf("  %-20s (no golden dump)\n", n); continue; }
        fclose(f);
        compare(n, trellis::tensor_to_f32(inter[n]), npy::load(ref + "/" + n + ".npy"));
    }
    compare("output", trellis::tensor_to_f32(out), npy::load(ref + "/output.npy"));

    ggml_gallocr_free(alloc); ggml_free(c); m.free();
    return 0;
}
