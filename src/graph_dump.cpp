// Diagnostic-only ggml graph dumper -- see include/graph_dump.h. Off unless env
// TRELLIS_DUMP_OPS is set; zero-cost (a single getenv check) otherwise.
#include "graph_dump.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace trellis {
namespace {

FILE* dump_file() {
    static FILE* f = nullptr;
    static bool opened = false;
    if (!opened) {
        opened = true;
        const char* path = std::getenv("TRELLIS_DUMP_OPS");
        f = (path && *path) ? std::fopen(path, "a") : stderr;
    }
    return f;
}

const char* layout_str(const ggml_tensor* t) {
    const bool cont = ggml_is_contiguous(t);
    if (t->view_src) return cont ? "view-cont" : "view-noncont";
    return cont ? "cont" : "noncont";
}

void print_tensor(FILE* f, const char* label, const ggml_tensor* t) {
    std::fprintf(f, " %s=%s[%lld,%lld,%lld,%lld]%s(%zuB)", label, ggml_type_name(t->type),
                 (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                 layout_str(t), ggml_nbytes(t));
}

// Best-effort decode of op_params for the ops the WebGPU porting plan cares most about
// (FlashAttention/softmax/RoPE/pad/conv/pool/norm); layouts mirror ggml.c's
// ggml_{flash_attn_ext,soft_max_ext,rope_impl,pad_ext,im2col,im2col_3d,conv_2d_direct,
// conv_3d_direct,pool_2d,norm_impl,group_norm_impl}. Everything else is left to the
// generic src/dst shapes printed by the caller.
void print_params(FILE* f, const ggml_tensor* t) {
    auto i32 = [&](int i) { int32_t v; std::memcpy(&v, &t->op_params[i], 4); return v; };
    auto f32 = [&](int i) { float v; std::memcpy(&v, &t->op_params[i], 4); return v; };
    switch (t->op) {
    case GGML_OP_FLASH_ATTN_EXT:
        std::fprintf(f, " params={scale=%g,max_bias=%g,logit_softcap=%g,prec=%d}",
                     f32(0), f32(1), f32(2), i32(3));
        break;
    case GGML_OP_SOFT_MAX:
        std::fprintf(f, " params={scale=%g,max_bias=%g}", f32(0), f32(1));
        break;
    case GGML_OP_ROPE:
        std::fprintf(f, " params={n_dims=%d,mode=%d,n_ctx_orig=%d,freq_base=%g,freq_scale=%g,"
                        "ext_factor=%g,attn_factor=%g,beta_fast=%g,beta_slow=%g}",
                     i32(1), i32(2), i32(4), f32(5), f32(6), f32(7), f32(8), f32(9), f32(10));
        break;
    case GGML_OP_PAD:
        std::fprintf(f, " params={lp0=%d,rp0=%d,lp1=%d,rp1=%d,lp2=%d,rp2=%d,lp3=%d,rp3=%d,circular=%d}",
                     i32(0), i32(1), i32(2), i32(3), i32(4), i32(5), i32(6), i32(7), i32(8));
        break;
    case GGML_OP_IM2COL:
        std::fprintf(f, " params={s0=%d,s1=%d,p0=%d,p1=%d,d0=%d,d1=%d,is_2D=%d}",
                     i32(0), i32(1), i32(2), i32(3), i32(4), i32(5), i32(6));
        break;
    case GGML_OP_IM2COL_3D:
        std::fprintf(f, " params={s0=%d,s1=%d,s2=%d,p0=%d,p1=%d,p2=%d,d0=%d,d1=%d,d2=%d,IC=%d}",
                     i32(0), i32(1), i32(2), i32(3), i32(4), i32(5), i32(6), i32(7), i32(8), i32(9));
        break;
    case GGML_OP_CONV_2D:
        std::fprintf(f, " params={s0=%d,s1=%d,p0=%d,p1=%d,d0=%d,d1=%d}",
                     i32(0), i32(1), i32(2), i32(3), i32(4), i32(5));
        break;
    case GGML_OP_CONV_3D:
        std::fprintf(f, " params={s0=%d,s1=%d,s2=%d,p0=%d,p1=%d,p2=%d,d0=%d,d1=%d,d2=%d,c=%d,n=%d,oc=%d}",
                     i32(0), i32(1), i32(2), i32(3), i32(4), i32(5), i32(6), i32(7), i32(8), i32(9), i32(10), i32(11));
        break;
    case GGML_OP_POOL_2D:
        std::fprintf(f, " params={pool_op=%d,k0=%d,k1=%d,s0=%d,s1=%d,p0=%d,p1=%d}",
                     i32(0), i32(1), i32(2), i32(3), i32(4), i32(5), i32(6));
        break;
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
        std::fprintf(f, " params={eps=%g}", f32(0));
        break;
    case GGML_OP_GROUP_NORM:
        std::fprintf(f, " params={n_groups=%d,eps=%g}", i32(0), f32(1));
        break;
    case GGML_OP_UNARY:
        std::fprintf(f, " sub_op=%s", ggml_unary_op_name(ggml_get_unary_op(t)));
        break;
    case GGML_OP_GLU:
        std::fprintf(f, " sub_op=%s", ggml_glu_op_name(ggml_get_glu_op(t)));
        break;
    default:
        break;
    }
}

std::string hist_key(const ggml_tensor* t) {
    if (t->op == GGML_OP_UNARY) return std::string("UNARY:") + ggml_unary_op_name(ggml_get_unary_op(t));
    if (t->op == GGML_OP_GLU)   return std::string("GLU:") + ggml_glu_op_name(ggml_get_glu_op(t));
    return ggml_op_name(t->op);
}

} // namespace

void trellis_graph_dump(const char* tag, ggml_cgraph* g) {
    if (!std::getenv("TRELLIS_DUMP_OPS")) return;   // off by default -- zero behavior change
    FILE* f = dump_file();
    if (!f) return;

    const int n = ggml_graph_n_nodes(g);
    std::unordered_map<std::string, int> hist;
    size_t max_tensor_bytes = 0;   // largest single tensor (src or dst) seen in this graph

    for (int i = 0; i < n; ++i) {
        ggml_tensor* t = ggml_graph_node(g, i);
        hist[hist_key(t)]++;

        std::fprintf(f, "[gd] tag=%s node=%d/%d op=%s", tag, i, n, ggml_op_name(t->op));
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!t->src[s]) continue;
            char lbl[8]; std::snprintf(lbl, sizeof lbl, "src%d", s);
            print_tensor(f, lbl, t->src[s]);
            size_t b = ggml_nbytes(t->src[s]);
            if (b > max_tensor_bytes) max_tensor_bytes = b;
        }
        print_tensor(f, "dst", t);
        print_params(f, t);
        size_t db = ggml_nbytes(t);
        if (db > max_tensor_bytes) max_tensor_bytes = db;
        std::fprintf(f, "\n");
    }

    std::fprintf(f, "[gd-summary] tag=%s nodes=%d max_tensor_bytes=%zu histogram={", tag, n, max_tensor_bytes);
    bool first = true;
    for (auto& kv : hist) {
        std::fprintf(f, "%s%s:%d", first ? "" : ",", kv.first.c_str(), kv.second);
        first = false;
    }
    std::fprintf(f, "}\n");
    std::fflush(f);
}

// ---- allocation trace (env TRELLIS_DBG_ALLOC_TRACE) ------------------------------------------
namespace {

const ggml_tensor* alloc_root(const ggml_tensor* t) {
    while (t->view_src) t = t->view_src;
    return t;
}

// A graph-allocated tensor: an INPUT leaf, or a node that is not a view. Weights (leafs without
// the input flag) live in the model buffer; views alias their root.
bool graph_allocated(const ggml_tensor* t) {
    if (t->view_src) return false;
    if (t->op == GGML_OP_NONE) return (t->flags & GGML_TENSOR_FLAG_INPUT) != 0;
    return true;
}

} // namespace

void trellis_graph_alloc_trace(const char* tag, ggml_cgraph* g, size_t gallocr_bytes) {
    const char* env = std::getenv("TRELLIS_DBG_ALLOC_TRACE");
    if (!env || !*env) return;   // off by default -- zero behavior change
    const bool verbose = std::strcmp(env, "all") == 0;
    double min_mb = 64.0;
    if (const char* m = std::getenv("TRELLIS_DBG_ALLOC_TRACE_MIN_MB")) min_mb = std::atof(m);
    const double min_bytes = min_mb * 1048576.0;

    const int n = ggml_graph_n_nodes(g);
    // producer index of every graph-allocated root (-1 = input leaf), last consumer index
    struct Info { int first, last; };
    std::unordered_map<const ggml_tensor*, Info> live;
    live.reserve((size_t)n * 2);
    auto touch = [&](const ggml_tensor* t, int i) {
        const ggml_tensor* r = alloc_root(t);
        if (!graph_allocated(r)) return;
        auto it = live.find(r);
        if (it == live.end()) live[r] = { r->op == GGML_OP_NONE ? -1 : i, i };
        else it->second.last = std::max(it->second.last, i);
    };
    for (int i = 0; i < n; ++i) {
        const ggml_tensor* t = ggml_graph_node(g, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) if (t->src[s]) touch(t->src[s], i);
        touch(t, i);   // the node's own output (or the root it writes into, for view nodes)
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) { auto it = live.find(alloc_root(t)); if (it != live.end()) it->second.last = n; }
    }
    // interval sweep: +bytes at first, -bytes after last
    std::vector<double> delta((size_t)n + 3, 0.0);
    double total = 0; size_t largest = 0; const ggml_tensor* largest_t = nullptr;
    std::vector<std::pair<size_t, const ggml_tensor*>> top;
    for (auto& kv : live) {
        const size_t b = ggml_nbytes(kv.first);
        delta[(size_t)(kv.second.first + 1)] += (double)b;
        delta[(size_t)(kv.second.last + 2)] -= (double)b;
        total += (double)b;
        if (b > largest) { largest = b; largest_t = kv.first; }
        top.push_back({ b, kv.first });
    }
    double cur = 0, peak = 0; int peak_at = -1;
    for (size_t i = 0; i + 1 < delta.size(); ++i) {
        cur += delta[i];
        if (cur > peak) { peak = cur; peak_at = (int)i - 1; }
    }
    std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });

    auto line = [&](const char* kind, int idx, const ggml_tensor* t) {
        std::fprintf(stderr, "      [at] %s %s node=%d op=%s%s%s name=%s shape=[%lld,%lld,%lld,%lld] dtype=%s bytes=%zu (%.1f MB)\n",
                     tag, kind, idx, ggml_op_name(t->op),
                     t->op == GGML_OP_UNARY ? ":" : "", t->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(t)) : "",
                     t->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                     ggml_type_name(t->type), ggml_nbytes(t), ggml_nbytes(t) / 1048576.0);
    };
    std::fprintf(stderr, "      [at] ==== %s: %d nodes, %zu graph-allocated tensors ====\n", tag, n, live.size());
    for (int i = 0; i < n; ++i) {
        const ggml_tensor* t = ggml_graph_node(g, i);
        if (!graph_allocated(t)) continue;
        if (verbose || (double)ggml_nbytes(t) >= min_bytes) line("alloc", i, t);
    }
    for (auto& kv : live)
        if (kv.first->op == GGML_OP_NONE && (verbose || (double)ggml_nbytes(kv.first) >= min_bytes)) line("input", -1, kv.first);
    std::fprintf(stderr, "      [at] -- %s: largest 10 --\n", tag);
    for (size_t k = 0; k < top.size() && k < 10; ++k) line("top", live[top[k].second].first, top[k].second);
    const ggml_tensor* pk = peak_at >= 0 && peak_at < n ? ggml_graph_node(g, peak_at) : nullptr;
    std::fprintf(stderr, "      [at-summary] %s nodes=%d allocs=%zu sum=%.2f GB largest=%.2f GB (%s %s [%lld,%lld,%lld,%lld] %s)"
                         " live_peak_est=%.2f GB at node %d (%s) gallocr_buffer=%.2f GB\n",
                 tag, n, live.size(), total / 1e9, largest / 1e9,
                 largest_t ? ggml_op_name(largest_t->op) : "-", largest_t ? largest_t->name : "-",
                 largest_t ? (long long)largest_t->ne[0] : 0, largest_t ? (long long)largest_t->ne[1] : 0,
                 largest_t ? (long long)largest_t->ne[2] : 0, largest_t ? (long long)largest_t->ne[3] : 0,
                 largest_t ? ggml_type_name(largest_t->type) : "-",
                 peak / 1e9, peak_at, pk ? ggml_op_name(pk->op) : "input", gallocr_bytes / 1e9);
    std::fflush(stderr);
}

} // namespace trellis
