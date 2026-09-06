// ggml WebGPU backend op regression: large-dispatch shapes (patches/ggml-webgpu/0003, 0004), the
// batched mul_mat shapes the Pixal3D NAF attention graph uses (src/naf_gpu.cpp), and the
// sparse-decoder buffer patterns (src/sparse.cpp). Every case runs one small graph on the WebGPU
// device and on ggml-cpu with identical inputs and compares.
//
// The upstream encoders for soft_max / sum_rows / norm / get_rows / concat / pad / repeat
// dispatched one workgroup per row (or per WG_SIZE elements) on the x axis only, so any op with
// more than maxComputeWorkgroupsPerDimension (65535) rows failed WebGPU validation and the node
// was silently dropped. Patch 0003 spreads those dispatches over a 2D grid; these shapes sit
// well past the old limit (up to 4M rows / 67M elements).
//
// SET_ROWS (patch 0004) had the same 1D dispatch: its encoder launched CEIL_DIV(threads, WG_SIZE)
// workgroups on x, where threads = rows * ne0/4 (f32/f16 dst with ne0 % 4 == 0), rows * ne0
// (other ne0) or rows * blocks (quantized dst), and the shaders read gid.x only. The limit is
// therefore on THREADS, not rows: with WG_SIZE = maxComputeInvocationsPerWorkgroup (1024 on
// Apple/Dawn, 256 on a spec-floor adapter) the op broke past 67.1M / 16.8M threads -- 65536 rows
// of 4096 floats, 4M rows of 64, 22.4M rows of 3, or the DiT RoPE scatter ([1,64,12,L], ne0 = 1)
// past L = 87381 tokens. The set_rows cases below bracket that: rows below / at / just above
// 65535 workgroups and far beyond, every dst dtype variant, I32 and I64 indices, duplicate and
// boundary indices, untouched rows, and the sparse-decoder [64, 4.6M] layout.
//
// The pad+cpy cases reproduce sparse_c2s's write-into-one-buffer pattern: ggml_pad seeds a
// [Cout, M] buffer from chunk 0 and every later chunk is a ggml_cpy into a row-offset view of it,
// with the copies rooted explicitly (nothing reads them). The dims are the res-1024 stage-3 C2S
// ones from docs/PIXAL3D_WEBGPU_OP_GAP.md §5 plus scaled-up variants.
//
//   ./build-webgpu/trellis-webgpu-ops [--small] [--only <substr>] [--budget-mb <MB>]
//     --small        shrink the NAF-derived shapes (quick check)
//     --only s       run only the cases whose name contains s (repeatable)
//     --budget-mb N  skip cases whose inputs + output exceed N MB (browser / small adapters)
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>

using T = ggml_tensor;

namespace {

float detval(size_t i) { return ((i * 2654435761u) & 1023) / 512.0f - 1.0f; }

ggml_context* mkctx(size_t n_tensors) {
    size_t meta = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(256, false) + (1 << 16);
    return ggml_init({ meta, nullptr, true });
}

struct Diff { double max_abs = 0, mean_abs = 0, rel = 0, cos = 0; };

Diff compare(const std::vector<float>& a, const std::vector<float>& ref) {
    Diff d;
    double sum_abs = 0, refmax = 0, dot = 0, na = 0, nb = 0;
    const size_t n = std::min(a.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        const double diff = std::fabs((double)a[i] - (double)ref[i]);
        d.max_abs = std::max(d.max_abs, diff);
        sum_abs += diff;
        refmax = std::max(refmax, std::fabs((double)ref[i]));
        dot += (double)a[i] * ref[i]; na += (double)a[i] * a[i]; nb += (double)ref[i] * ref[i];
    }
    d.mean_abs = n ? sum_abs / n : 0;
    d.rel = refmax > 0 ? d.max_abs / refmax : d.max_abs;
    d.cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0;
    return d;
}

// One test case: `build` creates the input tensors (flagged ggml_set_input) and returns the
// output node. `build_roots` is the same but may also return extra graph roots (nodes nothing
// reads, e.g. ggml_cpy into a view of the output). `fill` overrides the generic input filler.
struct Case {
    const char* name;
    double tol;
    std::function<T*(ggml_context*, std::vector<T*>&)> build;
    std::function<T*(ggml_context*, std::vector<T*>&, std::vector<T*>&)> build_roots;
    std::function<void(std::vector<T*>&)> fill;
};

// Read any tensor back as f32 (f16 / quantized rows are dequantized on the host).
std::vector<float> read_f32(T* out) {
    const size_t ne = (size_t)ggml_nelements(out);
    std::vector<float> r(ne);
    if (out->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(out, r.data(), 0, ggml_nbytes(out));
        return r;
    }
    std::vector<uint8_t> raw(ggml_nbytes(out));
    ggml_backend_tensor_get(out, raw.data(), 0, raw.size());
    ggml_get_type_traits(out->type)->to_float(raw.data(), r.data(), (int64_t)ne);
    return r;
}

void generic_fill(std::vector<T*>& inputs) {
    size_t seed = 1;
    for (T* in : inputs) {
        const size_t ne = (size_t)ggml_nelements(in);
        if (in->type == GGML_TYPE_F32) {
            std::vector<float> h(ne);
            // not f16-representable (detval alone is k/512), so f16 input staging shows up
            for (size_t i = 0; i < ne; ++i) h[i] = detval(seed + i) * (1.0f + 0.37f * detval(seed * 3 + i));
            ggml_backend_tensor_set(in, h.data(), 0, ggml_nbytes(in));
        } else if (in->type == GGML_TYPE_I32) {
            // index inputs: the row count to index is stored in op_params[0] by the builder
            const int32_t nrows = in->op_params[0];
            std::vector<int32_t> h(ne);
            for (size_t i = 0; i < ne; ++i) h[i] = (int32_t)((i * 7919u + seed) % (size_t)nrows);
            ggml_backend_tensor_set(in, h.data(), 0, ggml_nbytes(in));
        }
        seed += ne;
    }
}

uint64_t g_budget_bytes = 0;   // 0 = unlimited

bool run_case(const Case& cs, ggml_backend_t gpu, ggml_backend_t cpu) {
    std::vector<float> res[2];
    size_t out_bytes = 0;
    for (int which = 0; which < 2; ++which) {
        ggml_backend_t be = which == 0 ? gpu : cpu;
        ggml_context* c = mkctx(256);
        std::vector<T*> inputs, roots;
        T* out = cs.build ? cs.build(c, inputs) : cs.build_roots(c, inputs, roots);
        ggml_set_output(out);
        ggml_cgraph* g = ggml_new_graph_custom(c, 256, false);
        for (T* r : roots) ggml_build_forward_expand(g, r);
        ggml_build_forward_expand(g, out);
        if (which == 0) {
            uint64_t bytes = out->view_src ? 0 : ggml_nbytes(out);   // 64-bit: wasm32 size_t wraps at 4 GiB
            for (T* in : inputs) bytes += ggml_nbytes(in);
            if (g_budget_bytes && bytes > g_budget_bytes) {
                printf("  %-40s SKIP (needs %.0f MB > budget)\n", cs.name, bytes / 1048576.0);
                ggml_free(c);
                return true;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(be);
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                T* n = ggml_graph_node(g, i);
                if (!ggml_backend_dev_supports_op(dev, n)) {
                    printf("  %-40s node %d %s unsupported on %s -> FAIL\n", cs.name, i, ggml_op_name(n->op), ggml_backend_name(be));
                    ggml_free(c);
                    return false;
                }
            }
        }
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(alloc, g)) {
            printf("  %-40s alloc failed on %s -> FAIL\n", cs.name, ggml_backend_name(be));
            ggml_free(c);
            return false;
        }
        if (cs.fill) cs.fill(inputs); else generic_fill(inputs);
        if (ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) {
            printf("  %-40s compute failed on %s -> FAIL\n", cs.name, ggml_backend_name(be));
            ggml_gallocr_free(alloc); ggml_free(c);
            return false;
        }
        ggml_backend_synchronize(be);
        res[which] = read_f32(out);
        out_bytes = ggml_nbytes(out);
        ggml_gallocr_free(alloc);
        ggml_free(c);
    }
    size_t bad = 0;
    for (float v : res[0]) if (!std::isfinite(v)) ++bad;
    Diff d = compare(res[0], res[1]);
    const bool ok = bad == 0 && d.rel < cs.tol;
    if (!ok) {
        size_t first = res[0].size(), nbad = 0;
        for (size_t i = 0; i < res[0].size(); ++i)
            if (std::fabs(res[0][i] - res[1][i]) > 1e-6) { if (first == res[0].size()) first = i; ++nbad; }
        printf("    first mismatch at %zu of %zu, %zu mismatching (%.1f%%)\n", first, res[0].size(), nbad, 100.0 * nbad / res[0].size());
    }
    printf("  %-40s out %7.1f MB  max|d|=%.3e mean|d|=%.3e rel=%.3e cos=%.7f nonfinite=%zu (tol %.0e) -> %s\n",
           cs.name, out_bytes / 1048576.0, d.max_abs, d.mean_abs, d.rel, d.cos, bad, cs.tol, ok ? "PASS" : "FAIL");
    return ok;
}

T* in_f32(ggml_context* c, std::vector<T*>& ins, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    T* t = ggml_new_tensor_4d(c, GGML_TYPE_F32, n0, n1, n2, n3);
    ggml_set_input(t); ins.push_back(t);
    return t;
}
T* in_idx(ggml_context* c, std::vector<T*>& ins, int64_t n, int32_t nrows) {
    T* t = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    t->op_params[0] = nrows;   // consumed by run_case's filler
    ggml_set_input(t); ins.push_back(t);
    return t;
}

// ---- SET_ROWS cases -------------------------------------------------------------------------
//
// dst = set_rows(dst_prefill [ne0, n_dst, ne2, ne3], src [ne0, n_src, ne2, ne3], idx [n_src, i1, i2])
// The op writes in place into the prefilled dst (ggml_set_rows returns a view of it), so the
// comparison covers written rows AND untouched rows. Index modes:
//   unique: a deterministic permutation of [0, n_dst) truncated to n_src, with 0 forced into
//           idx[0] and n_dst-1 into idx[n_src-1] (boundary rows), no duplicates;
//   dup:    idx[i] = hash(i) % n_dst (many writers per row); every writer's source row is a
//           function of its TARGET row, so the result is well defined whichever writer lands
//           last (ggml leaves the winner unspecified; the CPU reference is thread-order dependent).
enum class IdxMode { unique, dup };

std::vector<int32_t> permutation(int64_t n, uint64_t seed) {
    std::vector<int32_t> p((size_t)n);
    for (int64_t i = 0; i < n; ++i) p[(size_t)i] = (int32_t)i;
    uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
    for (int64_t i = n - 1; i > 0; --i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const int64_t j = (int64_t)((s >> 33) % (uint64_t)(i + 1));
        std::swap(p[(size_t)i], p[(size_t)j]);
    }
    return p;
}

std::vector<int32_t> make_indices(int64_t n_src, int64_t n_dst, IdxMode mode) {
    std::vector<int32_t> idx((size_t)n_src);
    if (mode == IdxMode::dup) {
        for (int64_t i = 0; i < n_src; ++i) idx[(size_t)i] = (int32_t)(((uint64_t)i * 2654435761ull + 17) % (uint64_t)n_dst);
        idx[0] = 0; idx[(size_t)n_src - 1] = (int32_t)(n_dst - 1);
        return idx;
    }
    std::vector<int32_t> p = permutation(n_dst, 0x5EED + (uint64_t)n_src);
    // force the boundary rows into the used prefix
    auto move_to = [&](int32_t value, size_t pos) {
        size_t at = 0;
        for (size_t i = 0; i < p.size(); ++i) if (p[i] == value) { at = i; break; }
        std::swap(p[at], p[pos]);
    };
    move_to(0, 0);
    move_to((int32_t)(n_dst - 1), (size_t)n_src - 1);
    for (int64_t i = 0; i < n_src; ++i) idx[(size_t)i] = p[(size_t)i];
    return idx;
}

struct SetRowsSpec {
    int64_t ne0, n_src, n_dst, ne2 = 1, ne3 = 1;
    ggml_type dst_type = GGML_TYPE_F32;
    ggml_type idx_type = GGML_TYPE_I32;
    IdxMode mode = IdxMode::unique;
    bool idx_bcast = false;   // idx [n_src] shared by all ne2*ne3 slices (the DiT RoPE layout)
};

Case set_rows_case(const char* name, double tol, SetRowsSpec s) {
    Case cs{ name, tol, nullptr, nullptr, nullptr };
    cs.build = [s](ggml_context* c, std::vector<T*>& ins) {
        T* dst = ggml_new_tensor_4d(c, s.dst_type, s.ne0, s.n_dst, s.ne2, s.ne3);
        ggml_set_input(dst); ins.push_back(dst);
        T* src = in_f32(c, ins, s.ne0, s.n_src, s.ne2, s.ne3);
        T* idx = s.idx_bcast ? ggml_new_tensor_3d(c, s.idx_type, s.n_src, 1, 1)
                             : ggml_new_tensor_3d(c, s.idx_type, s.n_src, s.ne2, s.ne3);
        ggml_set_input(idx); ins.push_back(idx);
        return ggml_set_rows(c, dst, src, idx);
    };
    cs.fill = [s](std::vector<T*>& ins) {
        T *dst = ins[0], *src = ins[1], *idx = ins[2];
        // indices (one table per (i2,i3) slice unless broadcast)
        const int64_t n_tab = s.idx_bcast ? 1 : s.ne2 * s.ne3;
        std::vector<int32_t> tab = make_indices(s.n_src, s.n_dst, s.mode);
        std::vector<int32_t> all((size_t)(s.n_src * n_tab));
        for (int64_t t = 0; t < n_tab; ++t)
            for (int64_t i = 0; i < s.n_src; ++i)   // dup mode: rotate per slice so slices differ
                all[(size_t)(t * s.n_src + i)] = s.mode == IdxMode::dup ? tab[(size_t)((i + t) % s.n_src)] : tab[(size_t)i];
        if (s.idx_type == GGML_TYPE_I64) {
            std::vector<int64_t> h(all.begin(), all.end());
            ggml_backend_tensor_set(idx, h.data(), 0, ggml_nbytes(idx));
        } else {
            ggml_backend_tensor_set(idx, all.data(), 0, ggml_nbytes(idx));
        }
        // source rows: dup mode derives the row from its target so concurrent writers agree
        {
            std::vector<float> h((size_t)ggml_nelements(src));
            for (int64_t i3 = 0; i3 < s.ne3; ++i3)
              for (int64_t i2 = 0; i2 < s.ne2; ++i2)
                for (int64_t i = 0; i < s.n_src; ++i) {
                    const int64_t t = s.idx_bcast ? 0 : (i3 * s.ne2 + i2);
                    const int64_t key = s.mode == IdxMode::dup ? (int64_t)all[(size_t)(t * s.n_src + i)] : i;
                    float* row = h.data() + (size_t)(((i3 * s.ne2 + i2) * s.n_src + i) * s.ne0);
                    for (int64_t k = 0; k < s.ne0; ++k)
                        row[k] = detval((size_t)(key * 131 + k * 7 + (i3 * s.ne2 + i2) * 9973 + 5)) * 1.37f + 0.001f * (float)k;
                }
            ggml_backend_tensor_set(src, h.data(), 0, ggml_nbytes(src));
        }
        // dst prefill (the untouched rows keep these values)
        {
            const size_t ne = (size_t)ggml_nelements(dst);
            std::vector<float> h(ne);
            for (size_t i = 0; i < ne; ++i) h[i] = detval(i * 3 + 11) * 0.5f;
            if (dst->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(dst, h.data(), 0, ggml_nbytes(dst));
            } else if (dst->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> h16(ne);
                ggml_fp32_to_fp16_row(h.data(), h16.data(), (int64_t)ne);
                ggml_backend_tensor_set(dst, h16.data(), 0, ggml_nbytes(dst));
            } else {
                std::vector<uint8_t> q(ggml_nbytes(dst));
                ggml_get_type_traits_cpu(dst->type)->from_float(h.data(), q.data(), (int64_t)ne);
                ggml_backend_tensor_set(dst, q.data(), 0, ggml_nbytes(dst));
            }
        }
    };
    return cs;
}

// ---- PAD -> CPY(view) cases (sparse_c2s's write-into-one-buffer pattern) --------------------
//
// out = ggml_pad(chunk0 [Cout, nr0], 0, M - nr0)  -> [Cout, M], zero tail
// for every later chunk k: ggml_cpy(chunk_k, view_2d(out, Cout, nr_k, out->nb[1], m0_k * out->nb[1]))
// The copies are extra graph roots (nothing consumes them), exactly as in src/sparse.cpp.
Case pad_cpy_case(const char* name, int64_t Cout, int64_t M, std::vector<int64_t> chunk_rows) {
    Case cs{ name, 1e-12, nullptr, nullptr, nullptr };
    cs.build_roots = [Cout, M, chunk_rows](ggml_context* c, std::vector<T*>& ins, std::vector<T*>& roots) {
        T* out = nullptr;
        int64_t m0 = 0;
        for (size_t k = 0; k < chunk_rows.size(); ++k) {
            const int64_t nr = chunk_rows[k];
            T* chunk = in_f32(c, ins, Cout, nr);
            if (!out) { out = ggml_pad(c, chunk, 0, (int)(M - nr), 0, 0); }
            else roots.push_back(ggml_cpy(c, chunk, ggml_view_2d(c, out, Cout, nr, out->nb[1], (size_t)m0 * out->nb[1])));
            m0 += nr;
        }
        return out;
    };
    return cs;
}

int ops_main(int argc, char** argv) {
    bool small = false;
    std::vector<std::string> only;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--small") small = true;
        else if (a == "--only" && i + 1 < argc) only.push_back(argv[++i]);
        else if (a == "--budget-mb" && i + 1 < argc) g_budget_bytes = (uint64_t)atoll(argv[++i]) * 1048576;
    }

    ggml_backend_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "WebGPU", 6) == 0) { gpu = ggml_backend_dev_init(d, nullptr); break; }
    }
    if (!gpu) { fprintf(stderr, "no WebGPU device\n"); return 1; }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    printf("=== trellis-webgpu-ops (%s vs %s)%s ===\n", ggml_backend_name(gpu), ggml_backend_name(cpu), small ? " [--small]" : "");

    // NAF attention shapes at T=512, h=w=32: 1024 blocks of 256 pixels, 4 heads, 81 taps.
    const int64_t BLK = small ? 64 : 1024, PX = 256, HD = 4, TAP = 81;
    const int64_t T2 = small ? 262144 / 4 : 262144;   // T*T
    std::vector<Case> cases = {
        { "soft_max 1M rows", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_soft_max_ext(c, in_f32(c, ins, TAP, PX, HD, BLK), nullptr, 0.125f, 0.0f); } },
        { "sum_rows 4M rows", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_sum_rows(c, in_f32(c, ins, 16, T2 * 16)); } },
        { "norm 8 rows x 4M", 1e-4, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_norm(c, in_f32(c, ins, T2 * 16, 8), 1e-5f); } },
        { "norm 1M rows x 81", 1e-5, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_norm(c, in_f32(c, ins, TAP, PX, HD, BLK), 1e-5f); } },
        { "get_rows 82944 x 1024", 0.0 + 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* src = in_f32(c, ins, 1024, 1024);
              return ggml_get_rows(c, src, in_idx(c, ins, TAP * BLK, 1024)); } },
        { "get_rows 262144 x 256", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* src = in_f32(c, ins, 256, T2);
              return ggml_get_rows(c, src, in_idx(c, ins, T2, (int32_t)T2)); } },
        { "concat 4M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_concat(c, in_f32(c, ins, T2 / 16, 32, 4), in_f32(c, ins, T2 / 16, 32, 4), 1); } },
        { "concat 16M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_concat(c, in_f32(c, ins, T2 / 4, 32, 4), in_f32(c, ins, T2 / 4, 32, 4), 1); } },
        { "concat 67M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_concat(c, in_f32(c, ins, T2, 32, 4), in_f32(c, ins, T2, 32, 4), 1); } },
        { "pad 33M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_pad(c, in_f32(c, ins, 512, 512, small ? 32 : 128), 1, 1, 0, 0); } },
        { "repeat 16M elems", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* src = in_f32(c, ins, 1, 4096);
              return ggml_repeat(c, src, ggml_new_tensor_2d(c, GGML_TYPE_F32, small ? 1024 : 4096, 4096)); } },
        { "mul_mat [64,81]x[64,256] b4096", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_mul_mat(c, in_f32(c, ins, 64, TAP, HD, BLK), in_f32(c, ins, 64, PX, HD, BLK)); } },
        { "mul_mat [81,256]x[81,256] b4096", 5e-3, [&](ggml_context* c, std::vector<T*>& ins) {
              return ggml_mul_mat(c, in_f32(c, ins, TAP, 256, HD, BLK), in_f32(c, ins, TAP, PX, HD, BLK)); } },
        { "cont(permute) 64 MB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 256, PX, HD, BLK / 16);
              return ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); } },
        { "cont(permute) 256 MB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 256, PX, HD, BLK / 4);
              return ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); } },
        { "cont(permute) 512 MB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 256, PX, HD, BLK / 2);
              return ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); } },
        { "cont(permute) 1 GB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 256, PX, HD, BLK);
              return ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); } },
    };

    // SET_ROWS (patch 0004). Workgroup counts assume WG_SIZE = 1024 (Apple/Dawn); on a
    // 256-invocation adapter every count below is 4x larger, i.e. still past the limit.
    const int64_t R64K = 65536;
    cases.push_back(set_rows_case("set_rows [1024,65534] 16384 wg", 1e-12, { 1024, 65534, 65534 }));
    cases.push_back(set_rows_case("set_rows [4096,65535] 65535 wg (limit)", 1e-12, { 4096, 65535, 65535 }));
    cases.push_back(set_rows_case("set_rows [4096,65536] 65536 wg", 1e-12, { 4096, R64K, R64K }));
    cases.push_back(set_rows_case("set_rows [64,4194304] 4M rows 65536 wg", 1e-12, { 64, 4194304, 4194304 }));
    cases.push_back(set_rows_case("set_rows [16,16777216] 16M rows 65536 wg", 1e-12, { 16, 16777216, 16777216 }));
    cases.push_back(set_rows_case("set_rows [3,22369622] ne0=3 65536 wg", 1e-12, { 3, 22369622, 22369622 }));
    cases.push_back(set_rows_case("set_rows [64,4620541] sparse-dec M 72196 wg", 1e-12, { 64, 4620541, 4620542 }));
    {   // DiT RoPE scatter: src [1,64,12,L] -> dst [1,128,12,L], idx [64] broadcast over (nh, L)
        SetRowsSpec s{ 1, 64, 128, 12, 87384 }; s.idx_bcast = true;
        cases.push_back(set_rows_case("set_rows rope [1,64,12,87384] 65538 wg", 1e-12, s));
    }
    {   SetRowsSpec s{ 4096, R64K, R64K }; s.dst_type = GGML_TYPE_F16;
        cases.push_back(set_rows_case("set_rows f16 dst [4096,65536] 65536 wg", 1e-3, s)); }
    {   SetRowsSpec s{ 4096, R64K, R64K }; s.idx_type = GGML_TYPE_I64;
        cases.push_back(set_rows_case("set_rows i64 idx [4096,65536] 65536 wg", 1e-12, s)); }
    {   SetRowsSpec s{ 256, 200000, 50000 }; s.mode = IdxMode::dup;
        cases.push_back(set_rows_case("set_rows dup idx [256,200000]->50000", 1e-12, s)); }
    {   SetRowsSpec s{ 256, 70000, 70000 }; s.dst_type = GGML_TYPE_Q8_0;
        cases.push_back(set_rows_case("set_rows q8_0 dst [256,70000]", 2e-2, s)); }

    // sparse decoder PAD -> CPY(view) pattern (src/sparse.cpp hraw / out buffers)
    cases.push_back(pad_cpy_case("pad+cpy [64,100000] 4 chunks", 64, 100000, { 30000, 30000, 30000, 10000 }));
    cases.push_back(pad_cpy_case("pad+cpy [64,4620541] c2s hraw 2 chunks", 64, 4620541, { 3080029, 1540512 }));
    cases.push_back(pad_cpy_case("pad+cpy [64,4620541] conv2 out 1M chunks", 64, 4620541,
                                 { 1000000, 1000000, 1000000, 1000000, 620541 }));
    cases.push_back(pad_cpy_case("pad+cpy [64,8388608] 2 GiB 1M chunks", 64, 8388608,
                                 { 1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576 }));
    cases.push_back(pad_cpy_case("pad+cpy [64,12582912] 3 GiB 1M chunks", 64, 12582912,   // ~12 GB host: --budget-mb to skip
                                 std::vector<int64_t>(12, 1048576)));
    cases.push_back({ "cont(view) [64,1540512] @ +788 MB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 64, 4620541);
              return ggml_cont(c, ggml_view_2d(c, x, 64, 1540512, x->nb[1], (size_t)3080029 * x->nb[1])); }, nullptr, nullptr });
    cases.push_back({ "cont(view) [512,300000] @ +614 MB", 1e-12, [&](ggml_context* c, std::vector<T*>& ins) {
              T* x = in_f32(c, ins, 512, 600000);
              return ggml_cont(c, ggml_view_2d(c, x, 512, 300000, x->nb[1], (size_t)300000 * x->nb[1])); }, nullptr, nullptr });

    bool all = true;
    for (const Case& cs : cases) {
        if (!only.empty()) {
            bool hit = false;
            for (const std::string& s : only) if (strstr(cs.name, s.c_str())) hit = true;
            if (!hit) continue;
        }
        all &= run_case(cs, gpu, cpu);
    }
    printf("=== %s ===\n", all ? "ALL PASS" : "SOME FAILED");
    ggml_backend_free(gpu); ggml_backend_free(cpu);
    return all ? 0 : 1;
}

} // namespace

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
// Browser entry (web/ops): the same cases, driven by a space-separated argument string.
extern "C" EMSCRIPTEN_KEEPALIVE int webgpu_ops_run(const char* args) {
    std::vector<std::string> toks = { "trellis-webgpu-ops" };
    std::string cur;
    for (const char* p = args ? args : ""; ; ++p) {
        if (*p == ' ' || *p == '\0') { if (!cur.empty()) toks.push_back(cur); cur.clear(); if (!*p) break; }
        else cur.push_back(*p);
    }
    std::vector<char*> argv;
    for (std::string& s : toks) argv.push_back(s.data());
    int rc = 1;
    try {
        rc = ops_main((int)argv.size(), argv.data());
    } catch (const std::exception& e) {   // e.g. std::bad_alloc when the wasm heap cannot grow
        printf("webgpu_ops_run: C++ exception: %s -> FAIL\n", e.what());
    }
    fflush(stdout);
    return rc;
}
#else
int main(int argc, char** argv) { return ops_main(argc, argv); }
#endif
