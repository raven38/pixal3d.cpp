// Diagnostic-only ggml graph dumper -- see include/graph_dump.h. Off unless env
// TRELLIS_DUMP_OPS is set; zero-cost (a single getenv check) otherwise.
#include "graph_dump.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

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

} // namespace trellis
