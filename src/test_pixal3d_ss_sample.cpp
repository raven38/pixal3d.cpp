// End-to-end sparse-structure (SS) sampling parity test for the Pixal3D MV SS flow: runs the
// full 12-step FlowEuler/CFG sampler (ProjectAttention DiT, cond+proj) from fixture noise, then
// decodes with the SS decoder and compares occupancy logits + active-voxel IoU against f32 and
// bf16 PyTorch references, using the ref-f32-vs-bf16 IoU as the calibration baseline. See
// docs/spec/30-pixal3d-cond.md §5/6: block-level goldens are only meaningful on the CPU backend;
// the production parity gate is end-to-end sampling in the reference's own dtype (bf16).
// Fixture layout mirrors tools/ref_pixal3d_ss_sample.py (<fixture_dir>):
//   noise.npy [1,8,16,16,16]; cond_global.npy [1,5,1024], cond_proj.npy [1,4096,1024],
//   neg_global.npy/neg_proj.npy (zeros); f32_/bf16_ prefixed x_step0..12.npy, x_final.npy,
//   occ_logits.npy [1,1,64,64,64], coords.npy [N,4] int32 (b,x,y,z) at ss_res=32 (hardcoded
//   max-pool target in Pixal3DImageTo3DPipeline.run(), see the script).
// Back-compat: if cond_global/cond_proj.npy aren't in fixture_dir, falls back to a separate
// cond fixture dir (5th arg, or $PIXAL3D_COND_DIR, default <fixture_dir>/../cond_ss) with the
// tools/ref_pixal3d_cond_ss.py names z_global.npy/z_proj.npy (neg = zeros).
//   trellis-test-pixal3d-ss-sample <pixal3d_ss_flow_mv.gguf> <ss_dec.gguf> <fixture_dir> [gpu] [cond_dir] [dec_gpu] [dump_dir]
// dec_gpu (default = gpu): backend for the SS decoder alone -- pass -1 to decode on the CPU when
// the flow ran on a backend without Conv3D kernels (ggml WebGPU). dump_dir: if given, writes this
// run's per-step latents (`cpp_x_step<k>.npy`, torch [1,8,16,16,16] layout), `cpp_x_final.npy`,
// `cpp_occ_logits.npy` and `cpp_coords.npy` so another backend's run can be compared against it;
// files named `cuda_x_step<k>.npy` / `cuda_x_final.npy` / `cuda_coords.npy` found in fixture_dir
// are loaded as a third (native CUDA) reference next to the f32/bf16 PyTorch ones.
#include "trellis_model.h"
#include "flow_runner.h"
#include "dit.h"
#include "ss_decoder.h"
#include "trellis_args.h"
#include "npy.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using std::string; using std::vector;

// Loads a fixture .npy if present; returns false (leaving `out` untouched) if missing, so
// callers can skip gracefully.
static bool load_opt(const string& path, npy::Array& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    out = npy::load(path);
    return true;
}

// Minimal <i4 (int32) .npy reader -- ref_pixal3d_ss_sample.py saves coords via save_int(),
// which npy::load() (f32-only) can't parse. Self-contained here since npy.h is not to be
// touched by this test.
struct I32Array {
    std::vector<int64_t> shape;
    std::vector<int32_t> data;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};
static I32Array load_npy_i32(const string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); throw std::runtime_error("npy: bad magic " + path); }
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fclose(f); throw std::runtime_error("npy: header len " + path); }
    std::string hdr(hlen, '\0');
    if (fread(hdr.data(), 1, hlen, f) != hlen) { fclose(f); throw std::runtime_error("npy: header " + path); }
    if (hdr.find("'<i4'") == std::string::npos && hdr.find("\"<i4\"") == std::string::npos) {
        fclose(f); throw std::runtime_error("npy: only <i4 supported here: " + path + " hdr=" + hdr);
    }
    I32Array a;
    size_t p = hdr.find("'shape':"); p = hdr.find('(', p); size_t q = hdr.find(')', p);
    std::string s = hdr.substr(p + 1, q - p - 1);
    for (size_t i = 0; i < s.size();) {
        if (isdigit((unsigned char)s[i])) { int64_t v = 0; while (i < s.size() && isdigit((unsigned char)s[i])) v = v * 10 + (s[i++] - '0'); a.shape.push_back(v); }
        else ++i;
    }
    if (a.shape.empty()) a.shape.push_back(1);
    a.data.resize(a.numel());
    if ((int64_t)fread(a.data.data(), sizeof(int32_t), a.numel(), f) != a.numel()) { fclose(f); throw std::runtime_error("npy: short data " + path); }
    fclose(f);
    return a;
}
static bool load_opt_i32(const string& path, I32Array& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    out = load_npy_i32(path);
    return true;
}

struct Stats { double maxabs, meanabs, rel, cos; };
static Stats diff_stats(const vector<float>& mine, const npy::Array& ref) {
    double maxabs = 0, sumabs = 0, refmax = 0, dot = 0, na = 0, nb = 0;
    size_t n = std::min(mine.size(), (size_t)ref.numel());
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)mine[i] - ref.data[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)ref.data[i]));
        dot += (double)mine[i] * ref.data[i]; na += (double)mine[i] * mine[i]; nb += (double)ref.data[i] * ref.data[i];
    }
    return { maxabs, sumabs / (n ? n : 1), refmax > 0 ? maxabs / refmax : maxabs,
             (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0 };
}
static void print_stats(const char* tag, const Stats& s) {
    printf("    %-4s max|d|=%.4e mean|d|=%.4e rel=%.4e cos=%.7f\n", tag, s.maxabs, s.meanabs, s.rel, s.cos);
}
// Latent summary (mean / std / max|x| / non-finite count) -- the per-step statistics the WebGPU
// milestone reports side by side across backends.
static void print_latent(const vector<float>& v) {
    double m = 0, mx = 0; size_t bad = 0;
    for (float x : v) { if (!std::isfinite(x)) { ++bad; continue; } m += x; mx = std::max(mx, (double)std::fabs(x)); }
    m /= v.size();
    double var = 0; for (float x : v) if (std::isfinite(x)) var += (x - m) * (x - m);
    printf("    latent mean=%+.6f std=%.6f max|x|=%.4f nonfinite=%zu\n", m, std::sqrt(var / v.size()), mx, bad);
}

// ggml channel-major [C, L] (c + C*sp) -> torch [C,16,16,16] memory (c*L + sp), as test_ss_flow.cpp.
static vector<float> remap_c_major_to_torch(const vector<float>& mine, int C, int64_t L) {
    vector<float> out((size_t)C * L);
    for (int c = 0; c < C; ++c) for (int64_t sp = 0; sp < L; ++sp) out[(size_t)c * L + sp] = mine[c + C * sp];
    return out;
}

// Active-voxel coord set at resolution R (drops the batch column), encoded x*R*R+y*R+z.
static std::set<uint64_t> coord_set(const I32Array& coords, int R) {
    std::set<uint64_t> s;
    const int64_t N = coords.shape.empty() ? 0 : coords.shape[0];
    for (int64_t i = 0; i < N; ++i) {
        int x = coords.data[i * 4 + 1], y = coords.data[i * 4 + 2], z = coords.data[i * 4 + 3];
        s.insert(((uint64_t)x * R + y) * R + z);
    }
    return s;
}
static std::set<uint64_t> coord_set(const vector<std::array<int,3>>& coords, int R) {
    std::set<uint64_t> s;
    for (auto& c : coords) s.insert(((uint64_t)c[0] * R + c[1]) * R + c[2]);
    return s;
}
static double iou(const std::set<uint64_t>& a, const std::set<uint64_t>& b) {
    size_t inter = 0; for (auto v : a) if (b.count(v)) ++inter;
    size_t uni = a.size() + b.size() - inter;
    return uni ? (double)inter / uni : 1.0;
}
// Sanity check only: ref_pixal3d_ss_sample.py hardcodes SS_RES=32 (max-pooled from the raw
// 64^3 decode), matching trellis_cli.cpp's ss_coords(logits,64,32). Returns the max coordinate
// seen so callers can warn if a fixture ever uses a different resolution.
static int max_coord(const I32Array& coords) {
    int mx = 0;
    const int64_t N = coords.shape.empty() ? 0 : coords.shape[0];
    for (int64_t i = 0; i < N; ++i) for (int k = 1; k <= 3; ++k) mx = std::max(mx, coords.data[i * 4 + k]);
    return mx;
}

// token-major [1,Ntok,D] -> ggml channel-major [D,Ntok]
static vector<float> to_channel_major(const npy::Array& a, int64_t Ntok, int64_t D) {
    vector<float> out((size_t)D * Ntok);
    for (int64_t tok = 0; tok < Ntok; ++tok) for (int64_t d = 0; d < D; ++d) out[d + D * tok] = a.data[tok * D + d];
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <pixal3d_ss_flow_mv.gguf> <ss_dec.gguf> <fixture_dir> [gpu] [cond_dir] [dec_gpu] [dump_dir]\n", argv[0]);
        return 1;
    }
    const string gguf_flow = argv[1], gguf_dec = argv[2], fdir = argv[3];
    const int gpu = argc > 4 ? atoi(argv[4]) : 0;
    string cond_dir;
    if (argc > 5 && argv[5][0]) cond_dir = argv[5];
    else if (const char* e = getenv("PIXAL3D_COND_DIR")) cond_dir = e;
    else cond_dir = fdir + "/../cond_ss";
    const int dec_gpu = argc > 6 ? atoi(argv[6]) : gpu;
    const string dump_dir = argc > 7 ? argv[7] : "";
    printf("fixture_dir=%s gpu=%d dec_gpu=%d (cond fallback dir=%s)%s%s\n", fdir.c_str(), gpu, dec_gpu, cond_dir.c_str(),
           dump_dir.empty() ? "" : " dump_dir=", dump_dir.c_str());

    if (getenv("TRELLIS_NOFA")) { trellis::g_no_fa = true; printf("(TRELLIS_NOFA: soft_max path)\n"); }

    // ---- fixture inputs ----
    npy::Array noise = npy::load(fdir + "/noise.npy");          // [1,8,16,16,16]
    npy::Array zg, zp, negg, negp;
    bool has_neg = false;
    if (load_opt(fdir + "/cond_global.npy", zg) && load_opt(fdir + "/cond_proj.npy", zp)) {
        printf("cond: cond_global.npy/cond_proj.npy from fixture_dir\n");
        has_neg = load_opt(fdir + "/neg_global.npy", negg) && load_opt(fdir + "/neg_proj.npy", negp);
    } else {
        printf("cond: z_global.npy/z_proj.npy from cond_dir=%s\n", cond_dir.c_str());
        zg = npy::load(cond_dir + "/z_global.npy");
        zp = npy::load(cond_dir + "/z_proj.npy");
    }
    const int64_t Cin = noise.shape[1], R = noise.shape[2], L = R * R * R;
    const int64_t Lc = zg.shape[1], Dc = zg.shape[2], Dp = zp.shape[2];
    printf("fixture: Cin=%lld R=%lld L=%lld Lc=%lld Dc=%lld Dp=%lld\n",
           (long long)Cin, (long long)R, (long long)L, (long long)Lc, (long long)Dc, (long long)Dp);
    if (zp.shape[1] != L) { fprintf(stderr, "cond_proj tokens=%lld != L=%lld\n", (long long)zp.shape[1], (long long)L); return 1; }

    // noise torch [c*L+sp] -> ggml [c + Cin*sp]; cond/proj token-major -> channel-major.
    vector<float> sample(Cin * L);
    for (int64_t c = 0; c < Cin; ++c) for (int64_t sp = 0; sp < L; ++sp) sample[c + Cin * sp] = noise.data[c * L + sp];
    vector<float> cond = to_channel_major(zg, Lc, Dc);
    vector<float> proj = to_channel_major(zp, L, Dp);
    vector<float> neg_cond = has_neg ? to_channel_major(negg, Lc, Dc) : vector<float>(cond.size(), 0.0f);
    vector<float> neg_proj = has_neg ? to_channel_major(negp, L, Dp) : vector<float>(proj.size(), 0.0f);   // proj_linear(0) = bias

    // ---- SS flow sampler ----
    trellis::Model mf = trellis::Model::load(gguf_flow, gpu);
    printf("loaded %s (%zu tensors) on backend %s, weights %.1f MB\n", mf.arch.c_str(), mf.tensors.size(),
           ggml_backend_name(mf.backend), mf.total_bytes() / 1048576.0);
    trellis::DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(TRELLIS_F32W: f32 weight compute)\n"); }
    bool detected = trellis::dit_detect_proj_attn(mf, p);
    printf("proj_attn detected=%d d_proj=%d (fixture Dp=%lld)\n", detected, p.d_proj, (long long)Dp);
    if (!detected) { fprintf(stderr, "expected a Pixal3D checkpoint (blocks.0.cross_attn.proj_linear.weight missing)\n"); return 1; }
    if (p.d_proj != (int)Dp) { fprintf(stderr, "d_proj mismatch: checkpoint=%d fixture=%lld\n", p.d_proj, (long long)Dp); return 1; }

    trellis::DitRunner* run = trellis::make_dense_runner(mf, p, (int)R, (int)Lc);
    printf("DiT graph activation buffer: %.1f MB (weights %.1f MB, cond inputs %.1f MB)\n",
           run->alloc_bytes() / 1048576.0, mf.total_bytes() / 1048576.0, (cond.size() + proj.size()) * 4 / 1048576.0);
    int n_fwd = 0; double fwd_ms = 0;
    trellis::FlowFwdProj fwd = [&](const vector<float>& x, float ts, const float* c, const float* pj) {
        const auto t0 = std::chrono::steady_clock::now();
        vector<float> r = run->forward(x, ts, c, pj);
        fwd_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++n_fwd;
        return r;
    };

    trellis::SamplerParams sp;   // pipeline_mv.json sparse_structure_sampler.params
    sp.steps = 12; sp.guidance_strength = 7.5f; sp.guidance_rescale = 0.7f;
    sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 5.0f; sp.sigma_min = 1e-5f;

    vector<vector<float>> trace;
    vector<float> out = trellis::sample_flow(fwd, sample, cond.data(), neg_cond.data(),
                                              proj.data(), neg_proj.data(), sp, &trace);
    printf("DiT forwards: %d, %.1f ms each (%.1f s total)\n", n_fwd, n_fwd ? fwd_ms / n_fwd : 0.0, fwd_ms / 1000.0);
    delete run; mf.free();

    // ---- per-step latent parity vs f32 / bf16 PyTorch refs (+ a native CUDA run if dumped) ----
    printf("\nper-step latent parity (ggml [%lld,L] remapped to torch [%lld,16,16,16]):\n",
           (long long)Cin, (long long)Cin);
    const std::vector<int64_t> lat_shape = {1, Cin, R, R, R};
    for (size_t k = 0; k < trace.size(); ++k) {
        vector<float> mine = remap_c_major_to_torch(trace[k], (int)Cin, L);
        npy::Array rf, rb, rc;
        bool hf = load_opt(fdir + "/f32_x_step" + std::to_string(k + 1) + ".npy", rf);
        bool hb = load_opt(fdir + "/bf16_x_step" + std::to_string(k + 1) + ".npy", rb);
        bool hc = load_opt(fdir + "/cuda_x_step" + std::to_string(k + 1) + ".npy", rc);
        printf("  step %2zu:\n", k + 1);
        print_latent(mine);
        if (hf) print_stats("f32", diff_stats(mine, rf)); else printf("    f32  (missing)\n");
        if (hb) print_stats("bf16", diff_stats(mine, rb)); else printf("    bf16 (missing)\n");
        if (hc) print_stats("cuda", diff_stats(mine, rc));
        if (!dump_dir.empty()) npy::save(dump_dir + "/cpp_x_step" + std::to_string(k + 1) + ".npy", mine.data(), lat_shape);
    }
    // Explicit final-latent check -- redundant with the last trace step (trace.back() == out,
    // and the fixture script asserts x_final == x_step12) but ships as its own file, so check it.
    {
        vector<float> mine = remap_c_major_to_torch(out, (int)Cin, L);
        npy::Array rf, rb, rc;
        printf("  final  :\n");
        print_latent(mine);
        if (load_opt(fdir + "/f32_x_final.npy", rf)) print_stats("f32", diff_stats(mine, rf)); else printf("    f32  (missing)\n");
        if (load_opt(fdir + "/bf16_x_final.npy", rb)) print_stats("bf16", diff_stats(mine, rb)); else printf("    bf16 (missing)\n");
        if (load_opt(fdir + "/cuda_x_final.npy", rc)) print_stats("cuda", diff_stats(mine, rc));
        if (!dump_dir.empty()) npy::save(dump_dir + "/cpp_x_final.npy", mine.data(), lat_shape);
    }

    // ---- SS decode + active-voxel extraction (mirrors trellis_cli.cpp [3/6]) ----
    vector<float> zdec((size_t)Cin * L);
    for (int64_t c = 0; c < Cin; ++c) for (int64_t sp2 = 0; sp2 < L; ++sp2) zdec[(size_t)c * L + sp2] = out[c + Cin * sp2];
    trellis::Model md = trellis::Model::load(gguf_dec, dec_gpu);
    printf("\nSS decoder on backend %s\n", ggml_backend_name(md.backend));
    vector<float> logits = trellis::ss_decode(md, zdec);   // torch [1,1,64,64,64] memory
    md.free();
    if (!dump_dir.empty()) npy::save(dump_dir + "/cpp_occ_logits.npy", logits.data(), {1, 1, 64, 64, 64});

    printf("\nSS decoder occupancy logits parity:\n");
    npy::Array logf, logb, logc;
    bool has_logf = load_opt(fdir + "/f32_occ_logits.npy", logf);
    bool has_logb = load_opt(fdir + "/bf16_occ_logits.npy", logb);
    if (has_logf) print_stats("f32", diff_stats(logits, logf)); else printf("    f32  (missing)\n");
    if (has_logb) print_stats("bf16", diff_stats(logits, logb)); else printf("    bf16 (missing)\n");
    if (load_opt(fdir + "/cuda_occ_logits.npy", logc)) print_stats("cuda", diff_stats(logits, logc));

    I32Array cf, cb, cc;
    bool has_cf = load_opt_i32(fdir + "/f32_coords.npy", cf);
    bool has_cb = load_opt_i32(fdir + "/bf16_coords.npy", cb);
    bool has_cc = load_opt_i32(fdir + "/cuda_coords.npy", cc);
    const int out_res = 32;   // Pixal3DImageTo3DPipeline.run() hardcodes ss_res=32 (max-pool 64->32)
    if (has_cf && max_coord(cf) >= out_res) printf("WARNING: f32 ref coords exceed res%d -- resolution mismatch?\n", out_res);
    if (has_cb && max_coord(cb) >= out_res) printf("WARNING: bf16 ref coords exceed res%d -- resolution mismatch?\n", out_res);
    vector<std::array<int,3>> mine_coords = trellis::ss_coords(logits, 64, out_res);
    std::set<uint64_t> S_mine = coord_set(mine_coords, out_res);
    if (!dump_dir.empty()) {   // (b,x,y,z) int32 rows, same layout as the fixture's coords.npy
        vector<int32_t> rows; rows.reserve(mine_coords.size() * 4);
        for (auto& cxyz : mine_coords) { rows.push_back(0); rows.push_back(cxyz[0]); rows.push_back(cxyz[1]); rows.push_back(cxyz[2]); }
        std::string hdr = "{'descr': '<i4', 'fortran_order': False, 'shape': (" + std::to_string(mine_coords.size()) + ", 4), }";
        size_t pad = (64 - ((10 + hdr.size() + 1) % 64)) % 64; hdr.append(pad, ' '); hdr += '\n';
        uint16_t hlen = (uint16_t)hdr.size();
        FILE* f = fopen((dump_dir + "/cpp_coords.npy").c_str(), "wb");
        if (f) { fwrite("\x93NUMPY\x01\x00", 1, 8, f); fwrite(&hlen, 2, 1, f); fwrite(hdr.data(), 1, hdr.size(), f);
                 fwrite(rows.data(), sizeof(int32_t), rows.size(), f); fclose(f); }
    }

    printf("\nactive-voxel IoU @res%d:\n", out_res);
    printf("  N_mine=%zu", S_mine.size());
    if (has_cf) printf(" N_ref_f32=%lld", (long long)cf.shape[0]);
    if (has_cb) printf(" N_ref_bf16=%lld", (long long)cb.shape[0]);
    if (has_cc) printf(" N_ref_cuda=%lld", (long long)cc.shape[0]);
    printf("\n");
    if (has_cc) printf("  IoU(mine, cuda run)  = %.4f\n", iou(S_mine, coord_set(cc, out_res)));

    double iou_mine_f32 = -1, iou_mine_bf16 = -1, iou_f32_bf16 = -1;
    std::set<uint64_t> S_f32, S_bf16;
    if (has_cf) { S_f32 = coord_set(cf, out_res); iou_mine_f32 = iou(S_mine, S_f32); printf("  IoU(mine, f32 ref)  = %.4f\n", iou_mine_f32); }
    if (has_cb) { S_bf16 = coord_set(cb, out_res); iou_mine_bf16 = iou(S_mine, S_bf16); printf("  IoU(mine, bf16 ref) = %.4f\n", iou_mine_bf16); }
    if (has_cf && has_cb) { iou_f32_bf16 = iou(S_f32, S_bf16); printf("  IoU(f32 ref, bf16 ref) [calibration baseline] = %.4f\n", iou_f32_bf16); }

    // ---- verdict ----
    printf("\n=== verdict ===\n");
    bool crit_a = has_cf && has_cb && iou_mine_bf16 >= iou_f32_bf16 - 0.05;
    bool crit_b = has_cf && iou_mine_f32 >= 0.9;
    if (!has_cf && !has_cb) {
        printf("FAIL: no reference coords found in fixture -- cannot evaluate\n");
    } else if (crit_a || crit_b) {
        printf("PASS: %s\n", crit_a
               ? "IoU(mine,bf16) >= IoU(f32,bf16) - 0.05  (matches bf16-production calibration)"
               : "IoU(mine,f32) >= 0.9  (near-exact vs f32 reference)");
    } else {
        printf("FAIL: IoU(mine,bf16)=%.4f < IoU(f32,bf16)-0.05=%.4f  AND  IoU(mine,f32)=%.4f < 0.9\n",
               iou_mine_bf16, (has_cf && has_cb) ? iou_f32_bf16 - 0.05 : -1.0, iou_mine_f32);
    }
    return 0;
}
