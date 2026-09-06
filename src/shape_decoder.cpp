#include "shape_decoder.h"
#include "sparse.h"
#include "trellis_model.h"
#include "graph_dump.h"
#include "npy.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>

namespace trellis {
using T = ggml_tensor;

// A whole ConvNeXt stage (up to 16 blocks) is built as one graph, and each block is now
// chunked over voxels (~145 nodes per chunk), so the node count scales with object density.
// This is host-side metadata only (~350 B/tensor), so headroom is cheap -- running out
// aborts in ggml_new_object, which is a much worse failure than a few MB of address space.
static constexpr size_t kGraphNodes = 262144;

// Live GTT (whole-device, so it shows anything the driver has NOT handed back) and this
// process's RSS. Probed either side of every graph: if GTT does not drop back after a
// gallocr_free, memory is being retained between stages rather than merely used by one.
void mem_probe(const char* tag) {
    if (!getenv("TRELLIS_DBG_MEM")) return;
    long long gtt = 0, rss = 0;
    if (FILE* f = fopen("/sys/class/drm/card1/device/mem_info_gtt_used", "r")) { if (fscanf(f, "%lld", &gtt) != 1) gtt = 0; fclose(f); }
    if (FILE* f = fopen("/proc/self/statm", "r")) { long long tot; if (fscanf(f, "%lld %lld", &tot, &rss) != 2) rss = 0; fclose(f); }
    fprintf(stderr, "      [mem] %-34s GTT=%6.2f GB  RSS=%6.2f GB\n",
            tag, gtt / 1e9, rss * 4096.0 / 1e9);
}

// run a graph producing `out`, with the given input tensors set from host data
static std::vector<float> run1(const Model& m, ggml_context* c, T* out,
                               std::vector<std::pair<T*, const void*>> ins,
                               const char* tag = "shape_dec") {
    ggml_set_output(out);
    ggml_cgraph* g = ggml_new_graph_custom(c, kGraphNodes, false);
    ggml_build_forward_expand(g, out);
    trellis_graph_dump(tag, g);
    check_graph_supported(m.backend, g, tag);   // WebGPU: throw rather than silently skip a node
    ggml_gallocr_t a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(a, g)) throw std::runtime_error("shape_dec alloc");
    if (getenv("TRELLIS_DBG_ALLOC"))
        fprintf(stderr, "      [stage-alloc] nodes=%d  gallocr buffer = %.2f GB\n",
                ggml_graph_n_nodes(g), ggml_gallocr_get_buffer_size(a, 0) / 1e9);
    trellis_graph_alloc_trace(tag, g, ggml_gallocr_get_buffer_size(a, 0));   // TRELLIS_DBG_ALLOC_TRACE only
    for (auto& [t, d] : ins) ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t));
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) throw std::runtime_error("shape_dec compute");
    mem_probe("run1 computed (pre-readback)");
    std::vector<float> r = tensor_to_f32(out);
    ggml_gallocr_free(a);
    mem_probe("run1 freed (post-readback)");
    return r;
}

static ggml_context* mkctx() {
    size_t meta = ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false) + (1 << 20);
    return ggml_init({ meta, nullptr, true });
}

// Per-voxel linear (optional per-row LN first), chunked so a single mul_mat
// never exceeds backend dispatch-grid limits (Vulkan caps grid dim 1 at 65535
// workgroups = ~2.1M rows; the res-1024 cascade reaches 2.6M voxels).
static constexpr int64_t kLinearRowChunk = 1000000;

static std::vector<float> linear_rows(const Model& m, const std::vector<float>& in, int Cin,
                                      int64_t N, const std::string& prefix, int Cout, bool pre_norm,
                                      const char* kind = "shape_dec") {
    std::vector<float> out((size_t)Cout * N);
    for (int64_t r0 = 0; r0 < N; r0 += kLinearRowChunk) {
        const int64_t n = std::min(kLinearRowChunk, N - r0);
        ggml_context* c = mkctx();
        T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, n); ggml_set_input(gh);
        T* x = pre_norm ? ggml_norm(c, gh, 1e-5f) : gh;
        x = ggml_add(c, ggml_mul_mat(c, m.get(prefix + ".weight"), x), m.get(prefix + ".bias"));
        std::string tag = std::string(kind) + "_" + prefix + "_N" + std::to_string(n);
        std::vector<float> r = run1(m, c, x, { {gh, in.data() + (size_t)Cin * r0} }, tag.c_str());
        std::copy(r.begin(), r.end(), out.begin() + (size_t)Cout * r0);
        ggml_free(c);
    }
    return out;
}

// Debug-only per-stage dump (TRELLIS_DBG_STAGE_DUMP=<dir>; docs/PIXAL3D_WEBGPU_MEMORY.md §9):
// every stage's coords [N,3] i32 in full, plus a coordinate-hashed row subset (~4096 rows) of
// its features [n,C] f32 with those rows' coords, and [n, mean, std, absmax] of the whole
// tensor. The subset rule (uint32 (x*73856093 ^ y*19349663 ^ z*83492791) % K == 0,
// K = max(1, N/4096)) depends only on the coordinate, so two backends / the PyTorch reference
// pick the same rows whenever their coordinate sets agree -- no [N,C] tensor is written.
static void stage_dump(const char* kind, const std::string& stage, const std::vector<float>& h, int C,
                       const std::vector<std::array<int,3>>& coords) {
    const char* dir = getenv("TRELLIS_DBG_STAGE_DUMP");
    if (!dir || !*dir) return;
    const int64_t N = (int64_t)coords.size();
    const std::string base = std::string(dir) + "/" + kind + "_" + stage + "_";
    std::vector<int32_t> ci((size_t)N * 3);
    for (int64_t i = 0; i < N; ++i) for (int a = 0; a < 3; ++a) ci[(size_t)i * 3 + a] = coords[(size_t)i][a];
    npy::save_i32(base + "coords.npy", ci.data(), { N, 3 });
    const uint32_t K = (uint32_t)std::max<int64_t>(1, N / 4096);
    std::vector<int32_t> sc; std::vector<float> sf;
    for (int64_t i = 0; i < N; ++i) {
        const uint32_t x = (uint32_t)coords[(size_t)i][0], y = (uint32_t)coords[(size_t)i][1], z = (uint32_t)coords[(size_t)i][2];
        if ((x * 73856093u ^ y * 19349663u ^ z * 83492791u) % K != 0) continue;
        for (int a = 0; a < 3; ++a) sc.push_back(coords[(size_t)i][a]);
        sf.insert(sf.end(), h.begin() + (size_t)C * i, h.begin() + (size_t)C * (i + 1));
    }
    const int64_t n = (int64_t)sc.size() / 3;
    npy::save_i32(base + "sub_coords.npy", sc.data(), { n, 3 });
    npy::save(base + "sub_feats.npy", sf.data(), { n, C });
    double sum = 0, sq = 0, mx = 0;
    for (float v : h) { sum += v; sq += (double)v * v; mx = std::max(mx, (double)std::fabs(v)); }
    const double mean = h.empty() ? 0 : sum / (double)h.size();
    const float st[4] = { (float)h.size(), (float)mean, (float)std::sqrt(std::max(0.0, sq / (double)std::max<size_t>(1, h.size()) - mean * mean)), (float)mx };
    npy::save(base + "stats.npy", st, { 4 });
    fprintf(stderr, "      [stage-dump] %s_%s: N=%lld C=%d subset=%lld (K=%u) mean=%.5f std=%.5f absmax=%.4f\n",
            kind, stage.c_str(), (long long)N, C, (long long)n, K, mean, st[2], mx);
}

// Generic SparseUnetVaeDecoder: from_latent -> 4×(ConvNeXt stage + C2S) -> final LN -> output_layer.
// guide_subs!=null -> tex (C2S driven by provided masks); subs_out!=null -> shape (collect masks).
static std::vector<float> decode_unet(const Model& m, const std::vector<float>& latent, int out_ch,
                                      std::vector<std::array<int,3>>& coords,
                                      const std::vector<std::vector<uint8_t>>* guide_subs,
                                      std::vector<std::vector<uint8_t>>* subs_out,
                                      bool coords_only = false) {
    const char* kind = coords_only ? "sparse_upsample" : (guide_subs ? "tex_dec" : "shape_dec");
    int N = (int)coords.size();
    // from_latent [32,N] -> [1024,N]
    std::vector<float> h = linear_rows(m, latent, 32, N, "from_latent", 1024, false, kind);
    stage_dump(kind, "from_latent", h, 1024, coords);
    struct Stage { int C, nblk, Cout, c2si; const char* s; };
    const Stage stages[4] = { {1024,4,512,4,"0"}, {512,16,256,16,"1"}, {256,8,128,8,"2"}, {128,4,64,4,"3"} };
    for (int si = 0; si < 4; ++si) {
        const Stage& st = stages[si];
        std::vector<int32_t> nbr = build_neighbor_table(coords);
        if (getenv("TRELLIS_DBG_MEM"))
            fprintf(stderr, "    == stage %d: C=%d nblk=%d N=%d | host h=%.2f GB nbr=%.2f GB\n",
                    si, st.C, st.nblk, N, (double)h.size() * 4 / 1e9, (double)nbr.size() * 4 / 1e9);
        {   // ConvNeXt stage
            ggml_context* c = mkctx();
            T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, st.C, N); ggml_set_input(gh);
            T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, N, 27);   ggml_set_input(gn);
            T* x = gh;
            for (int j = 0; j < st.nblk; ++j)
                x = sparse_convnext(c, m, std::string("blocks.") + st.s + "." + std::to_string(j), x, gn, N);
            std::string tag = std::string(kind) + "_convnext_stage" + std::to_string(si) + "_N" + std::to_string(N);
            h = run1(m, c, x, { {gh, h.data()}, {gn, nbr.data()} }, tag.c_str());
            ggml_free(c);
        }
        mem_probe("after ConvNeXt stage");
        stage_dump(kind, "stage" + std::to_string(si) + "_convnext", h, st.C, coords);
        const std::vector<uint8_t>* ext = guide_subs ? &(*guide_subs)[si] : nullptr;
        C2SResult r = sparse_c2s(m, std::string("blocks.") + st.s + "." + std::to_string(st.c2si), h, st.C, coords, st.Cout, ext,
                                 (std::string(kind) + "_c2s_stage" + std::to_string(si)).c_str());
        if (subs_out) subs_out->push_back(r.subdiv);
        h = std::move(r.feats); coords = std::move(r.coords); N = (int)coords.size();
        mem_probe("after c2s");
        stage_dump(kind, "stage" + std::to_string(si) + "_c2s", h, st.Cout, coords);
    }
    if (coords_only) return {};   // cascade upsample: just need the grown coords
    // final LN(no affine) + output_layer -> [out_ch, M]
    std::vector<float> out = linear_rows(m, h, 64, N, "output_layer", out_ch, true, kind);
    stage_dump(kind, "output", out, out_ch, coords);
    return out;
}

ShapeOut shape_decode(const Model& m, const std::vector<float>& latent,
                      const std::vector<std::array<int,3>>& coords0, int resolution) {
    ShapeOut so; so.coords = coords0; so.res = resolution;
    so.feats7 = decode_unet(m, latent, 7, so.coords, nullptr, &so.subs);
    return so;
}

std::vector<std::array<int,3>> shape_upsample(const Model& m, const std::vector<float>& latent,
                                              const std::vector<std::array<int,3>>& coords0) {
    std::vector<std::array<int,3>> coords = coords0;
    decode_unet(m, latent, 7, coords, nullptr, nullptr, /*coords_only=*/true);
    return coords;
}

std::vector<float> tex_decode(const Model& m, const std::vector<float>& tex_latent,
                              const std::vector<std::array<int,3>>& coords0,
                              const std::vector<std::vector<uint8_t>>& subs) {
    std::vector<std::array<int,3>> coords = coords0;
    return decode_unet(m, tex_latent, 6, coords, &subs, nullptr);   // 6-ch PBR at the same final coords
}

} // namespace trellis
