// Backend-vs-backend probe for one sparse Pixal3D SLAT DiT forward: runs the shared graph on the
// chosen backend for a given latent / timestep / condition (hr_sample or slat_sample fixture
// layout), prints the magnitude and finiteness of every named intermediate, and saves a token
// subsample of each so two backends' dumps can be diffed offline (first divergent tensor). Used
// to localize the Shape-1024 WebGPU sampling divergence (spec 31 §11.5).
//
//   trellis-test-pixal3d-slat-probe <flow.gguf> <fixture_dir> <x.npy> <t_scaled> <out_dir> [gpu] [n_tokens_saved]
//     fixture_dir: hr_sample (hr_coords/hr_cond_*) or slat_sample (coords/cond_*) layout (auto)
//     x.npy:       [N,32] latent to feed (e.g. a per-step dump cpp_shape_x_step1.npy)
#include "trellis_model.h"
#include "dit.h"
#include "trellis_args.h"
#include "npy.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using std::string; using std::vector;

static bool exists(const string& p) { FILE* f = fopen(p.c_str(), "rb"); if (!f) return false; fclose(f); return true; }

static vector<int32_t> load_i32(const string& path, int64_t& rows, int64_t& cols) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    unsigned char magic[8]; if (fread(magic, 1, 8, f) != 8) throw std::runtime_error("npy " + path);
    uint16_t hlen; if (fread(&hlen, 2, 1, f) != 1) throw std::runtime_error("npy " + path);
    string hdr(hlen, '\0'); if (fread(hdr.data(), 1, hlen, f) != hlen) throw std::runtime_error("npy " + path);
    vector<int64_t> shape;
    size_t p = hdr.find("'shape':"); p = hdr.find('(', p); size_t q = hdr.find(')', p);
    string sh = hdr.substr(p + 1, q - p - 1);
    for (size_t i = 0; i < sh.size();) {
        if (isdigit((unsigned char)sh[i])) { int64_t v = 0; while (i < sh.size() && isdigit((unsigned char)sh[i])) v = v * 10 + (sh[i++] - '0'); shape.push_back(v); }
        else ++i;
    }
    rows = shape[0]; cols = shape.size() > 1 ? shape[1] : 1;
    vector<int32_t> d((size_t)rows * cols);
    if ((int64_t)fread(d.data(), 4, d.size(), f) != (int64_t)d.size()) throw std::runtime_error("short " + path);
    fclose(f);
    return d;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <flow.gguf> <fixture_dir> <x.npy> <t_scaled> <out_dir> [gpu] [n_tokens_saved]\n", argv[0]);
        return 1;
    }
    const string gguf = argv[1], fdir = argv[2], xpath = argv[3], odir = argv[5];
    const float t = (float)atof(argv[4]);
    const int gpu = argc > 6 ? atoi(argv[6]) : 0;
    const int64_t nsave = argc > 7 ? atoll(argv[7]) : 2000;
    if (getenv("TRELLIS_NOFA")) { trellis::g_no_fa = true; printf("(no-fa: soft_max path)\n"); }

    const bool hr = exists(fdir + "/hr_coords.npy");
    const string cpath = fdir + (hr ? "/hr_coords.npy" : "/coords.npy");
    const string gpath = fdir + (hr ? "/hr_cond_global.npy" : "/cond_global.npy");
    const string ppath = fdir + (hr ? "/hr_cond_proj.npy" : "/cond_proj.npy");
    int64_t N, cw;
    vector<int32_t> co = load_i32(cpath, N, cw);
    npy::Array x = npy::load(xpath), gA = npy::load(gpath), jA = npy::load(ppath);
    const int64_t Cin = x.shape[1], Lc = gA.shape[1], Dc = gA.shape[2], Dp = jA.shape[1];
    if (x.shape[0] != N || jA.shape[0] != N) { fprintf(stderr, "token count mismatch\n"); return 1; }
    printf("N=%lld Cin=%lld Lc=%lld Dc=%lld Dp=%lld t_scaled=%.3f backend gpu=%d\n", (long long)N, (long long)Cin, (long long)Lc, (long long)Dc, (long long)Dp, t, gpu);

    trellis::Model m = trellis::Model::load(gguf, gpu);
    trellis::DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(f32 weight compute)\n"); }
    if (!trellis::dit_detect_proj_attn(m, p) || p.d_proj != (int)Dp) { fprintf(stderr, "proj_attn mismatch\n"); return 1; }

    vector<float> tfreq(256);
    for (int j = 0; j < 128; ++j) { const float f = std::exp(-std::log(10000.f) * j / 128.f); tfreq[j] = std::cos(t * f); tfreq[128 + j] = std::sin(t * f); }
    const int half = p.head_dim / 2;
    vector<float> rcos((size_t)N * half), rsin((size_t)N * half);
    float freqs[21]; for (int j = 0; j < 21; ++j) freqs[j] = 1.0f / std::pow(10000.f, j / 21.0f);
    for (int64_t tok = 0; tok < N; ++tok) {
        const int cx = co[tok * cw + cw - 3], cy = co[tok * cw + cw - 2], cz = co[tok * cw + cw - 1];
        for (int pp = 0; pp < half; ++pp) {
            float ang = 0;
            if (pp < 21) ang = cx * freqs[pp]; else if (pp < 42) ang = cy * freqs[pp - 21]; else if (pp < 63) ang = cz * freqs[pp - 42];
            rcos[tok * half + pp] = std::cos(ang); rsin[tok * half + pp] = std::sin(ang);
        }
    }
    vector<float> cond((size_t)Dc * Lc);
    for (int64_t k = 0; k < Lc; ++k) for (int64_t d = 0; d < Dc; ++d) cond[d + Dc * k] = gA.data[k * Dc + d];

    size_t meta = ggml_tensor_overhead() * 131072 + ggml_graph_overhead_custom(262144, false) + (1 << 20);
    ggml_context* c = ggml_init({ meta, nullptr, true });
    ggml_tensor* gh0  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, N);         ggml_set_input(gh0);
    ggml_tensor* gtf  = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);            ggml_set_input(gtf);
    ggml_tensor* gcd  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Dc, Lc);         ggml_set_input(gcd);
    ggml_tensor* gcos = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, N);  ggml_set_input(gcos);
    ggml_tensor* gsin = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, N);  ggml_set_input(gsin);
    ggml_tensor* gpj  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Dp, N);          ggml_set_input(gpj);
    ggml_tensor* gidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, p.head_dim);      ggml_set_input(gidx);
    vector<int32_t> ridx; trellis::dit_rope_index(p.head_dim, ridx);
    std::map<string, ggml_tensor*> inter;
    ggml_tensor* out = trellis::build_dit_dense(c, m, p, gh0, gtf, gcd, gcos, gsin, &inter, gpj, gidx);
    ggml_cgraph* g = ggml_new_graph_custom(c, 262144, false);
    ggml_build_forward_expand(g, out);
    for (auto& [k, v] : inter) { ggml_build_forward_expand(g, v); ggml_set_output(v); }
    trellis::check_graph_supported(m.backend, g, "slat_probe");
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) { fprintf(stderr, "alloc failed\n"); return 1; }
    printf("graph: %d nodes, activation buffer %.1f MB (all intermediates kept)\n", ggml_graph_n_nodes(g), ggml_gallocr_get_buffer_size(alloc, 0) / 1048576.0);
    ggml_backend_tensor_set(gh0, x.data.data(), 0, (size_t)Cin * N * 4);
    ggml_backend_tensor_set(gtf, tfreq.data(), 0, tfreq.size() * 4);
    ggml_backend_tensor_set(gcd, cond.data(), 0, cond.size() * 4);
    ggml_backend_tensor_set(gcos, rcos.data(), 0, rcos.size() * 4);
    ggml_backend_tensor_set(gsin, rsin.data(), 0, rsin.size() * 4);
    ggml_backend_tensor_set(gpj, jA.data.data(), 0, (size_t)Dp * N * 4);
    ggml_backend_tensor_set(gidx, ridx.data(), 0, ridx.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
    ggml_backend_synchronize(m.backend);

    // deterministic order: input layer, t_emb, block-0 probes, after_block0..29, prefinal, output
    vector<string> order = { "after_input_layer", "t_emb_mod", "blk0_msa", "blk0_global_out", "blk0_proj_out", "blk0_cross_out", "blk0_mlp" };
    for (int i = 0; i < p.n_blocks; ++i) order.push_back("after_block" + std::to_string(i));
    for (const char* b : { "blk1_", "blk15_" }) for (const char* n : { "msa", "global_out", "proj_out", "cross_out", "cross", "mlp" }) order.push_back(std::string(b) + n);
    order.push_back("blk15_msa"); order.push_back("blk15_global_out"); order.push_back("blk15_proj_out"); order.push_back("blk15_cross_out"); order.push_back("blk15_mlp");
    order.push_back("prefinal"); order.push_back("output");
    printf("%-20s %14s %12s %10s %8s\n", "tensor", "max|x|", "mean|x|", ">65504", "nonfin");
    for (const string& nm : order) {
        auto it = inter.find(nm);
        if (it == inter.end()) continue;
        ggml_tensor* tsr = it->second;
        vector<float> v = trellis::tensor_to_f32(tsr);
        double mx = 0, sum = 0; size_t over = 0, bad = 0;
        for (float f : v) { if (!std::isfinite(f)) { ++bad; continue; } const double a = std::fabs(f); sum += a; if (a > mx) mx = a; if (a > 65504.0) ++over; }
        printf("%-20s %14.3f %12.5f %10zu %8zu\n", nm.c_str(), mx, sum / v.size(), over, bad);
        // subsample: the first nsave tokens (ne1) of a [C, N] tensor, saved [nsave, C] row-major
        const int64_t C0 = tsr->ne[0], NT = tsr->ne[1];
        const int64_t ns = std::min<int64_t>(nsave, NT);
        vector<float> sub((size_t)ns * C0);
        std::memcpy(sub.data(), v.data(), (size_t)ns * C0 * sizeof(float));
        npy::save(odir + "/" + nm + ".npy", sub.data(), { ns, C0 });
    }
    ggml_gallocr_free(alloc); ggml_free(c); m.free();
    return 0;
}
