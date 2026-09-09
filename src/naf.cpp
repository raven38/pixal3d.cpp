#include "naf.h"
#include "trellis_model.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

namespace trellis {

// ---------------------------------------------------------------------------
// weight containers (private to this file — the public surface for callers is
// just naf_upsample(Model, ...); see include/naf.h)
// ---------------------------------------------------------------------------
namespace {

// Minimal internal parallel-for: splits [0,n) into contiguous, statically
// sized ranges (one per thread, hardware_concurrency()-capped) and runs
// fn(begin,end) for each range on its own std::thread; the caller's fn
// processes indices in that range in the same order it would serially.
// This is used only where the callee's writes for distinct indices are to
// disjoint memory and any internal reduction (e.g. a dot-product/softmax
// accumulation) happens fully within one index's iteration -- so results are
// bit-identical to the single-threaded loop regardless of how work is split
// across threads/cores. No OpenMP dependency (trellis_core doesn't link it).
template <typename Fn>
static void naf_parallel_for(int n, Fn&& fn) {
    if (n <= 0) return;
#ifdef __EMSCRIPTEN__
    fn(0, n);   // pthread 無しでリンクしているので std::thread は生成できない
#else
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    int nthreads = (int)std::min<unsigned>(hw, (unsigned)n);
    if (nthreads <= 1) { fn(0, n); return; }
    const int chunk = (n + nthreads - 1) / nthreads;
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        const int begin = t * chunk;
        const int end = std::min(n, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([&fn, begin, end]() { fn(begin, end); });
    }
    for (auto& th : pool) th.join();
#endif
}

struct ConvW { int Co = 0, Ci = 0, K = 0; std::vector<float> w, b; };
struct GNW   { int C = 0; std::vector<float> w, b; };
struct BlockW { GNW norm1, norm2; ConvW conv1, conv2; };
struct BranchW { ConvW conv0; BlockW blk1, blk2; };

ConvW load_conv(const Model& m, const std::string& name) {
    ggml_tensor* wt = m.get(name + ".weight");
    ConvW c;
    // conv weights are torch [Co,Ci,kh,kw] == ggml ne reversed: ne0=kw,ne1=kh,ne2=Ci,ne3=Co.
    c.K  = (int)wt->ne[0];
    c.Ci = (int)wt->ne[2];
    c.Co = (int)wt->ne[3];
    c.w  = tensor_to_f32(wt);
    c.b  = tensor_to_f32(m.get(name + ".bias"));
    return c;
}

GNW load_gn(const Model& m, const std::string& name) {
    GNW g;
    g.w = tensor_to_f32(m.get(name + ".weight"));
    g.b = tensor_to_f32(m.get(name + ".bias"));
    g.C = (int)g.w.size();
    return g;
}

BlockW load_block(const Model& m, const std::string& prefix) {
    BlockW b;
    b.norm1 = load_gn(m, prefix + ".norm1");
    b.conv1 = load_conv(m, prefix + ".conv1");
    b.norm2 = load_gn(m, prefix + ".norm2");
    b.conv2 = load_conv(m, prefix + ".conv2");
    return b;
}

BranchW load_branch(const Model& m, const std::string& prefix) {
    BranchW br;
    br.conv0 = load_conv(m, prefix + ".0");
    br.blk1  = load_block(m, prefix + ".1");
    br.blk2  = load_block(m, prefix + ".2");
    return br;
}

// EncBlock(x) = conv2(SiLU(GN2(conv1(SiLU(GN1(x)))))) — no residual.
std::vector<float> enc_block_forward(const std::vector<float>& x, int H, int W, const BlockW& blk) {
    std::vector<float> h = naf_group_norm(x, blk.norm1.C, H, W, 8, blk.norm1.w.data(), blk.norm1.b.data(), 1e-5f);
    naf_silu_inplace(h);
    h = naf_conv2d(h, blk.conv1.Ci, H, W, blk.conv1.w.data(), blk.conv1.b.data(), blk.conv1.Co, blk.conv1.K);
    h = naf_group_norm(h, blk.norm2.C, H, W, 8, blk.norm2.w.data(), blk.norm2.b.data(), 1e-5f);
    naf_silu_inplace(h);
    h = naf_conv2d(h, blk.conv2.Ci, H, W, blk.conv2.w.data(), blk.conv2.b.data(), blk.conv2.Co, blk.conv2.K);
    return h;
}

// branch(x) = EncBlock(EncBlock(conv0(x))) — encoder (k=1) or sem_encoder (k=3, reflect).
std::vector<float> branch_forward(const std::vector<float>& img, int H, int W, const BranchW& br) {
    std::vector<float> e = naf_conv2d(img, br.conv0.Ci, H, W, br.conv0.w.data(), br.conv0.b.data(), br.conv0.Co, br.conv0.K);
    e = enc_block_forward(e, H, W, br.blk1);
    e = enc_block_forward(e, H, W, br.blk2);
    return e;
}

inline int reflect_idx(int idx, int N) {
    if (N <= 1) return 0;
    while (idx < 0 || idx >= N) {
        if (idx < 0) idx = -idx;
        if (idx >= N) idx = 2 * (N - 1) - idx;
    }
    return idx;
}

} // namespace

// ---------------------------------------------------------------------------
// building blocks (declared in naf.h so test_naf.cpp can unit-check them)
// ---------------------------------------------------------------------------

std::vector<float> naf_group_norm(const std::vector<float>& x, int C, int H, int W,
                                   int num_groups, const float* gamma, const float* beta, float eps) {
    std::vector<float> out(x.size());
    const int cg = C / num_groups;
    const size_t hw = (size_t)H * W;
    // Groups are independent (each group's mean/var/output depends only on
    // its own channels), so splitting the outer group loop across threads
    // doesn't touch the summation order *within* a group (still cc=0..cg-1,
    // i=0..hw-1 in order on whichever thread runs it) -- bit-identical.
    naf_parallel_for(num_groups, [&](int g0, int g1) {
        for (int g = g0; g < g1; ++g) {
            double sum = 0, sumsq = 0;
            const size_t n = (size_t)cg * hw;
            for (int cc = 0; cc < cg; ++cc) {
                const float* p = x.data() + (size_t)(g * cg + cc) * hw;
                for (size_t i = 0; i < hw; ++i) { sum += p[i]; sumsq += (double)p[i] * p[i]; }
            }
            const double mean = sum / (double)n;
            const double var = sumsq / (double)n - mean * mean;
            const double inv_std = 1.0 / std::sqrt(var + (double)eps);
            for (int cc = 0; cc < cg; ++cc) {
                const int c = g * cg + cc;
                const float* p = x.data() + (size_t)c * hw;
                float* o = out.data() + (size_t)c * hw;
                const float sc = (float)(gamma[c] * inv_std);
                const float sh = (float)(beta[c] - gamma[c] * mean * inv_std);
                for (size_t i = 0; i < hw; ++i) o[i] = p[i] * sc + sh;
            }
        }
    });
    return out;
}

void naf_silu_inplace(std::vector<float>& x) {
    // Elementwise, no shared reduction -> any partitioning is safe/identical.
    naf_parallel_for((int)x.size(), [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
    });
}

// Reflect-padded conv2d (pad = K/2; a no-op copy when K==1). Builds one
// reflect-padded [Ci,H+2p,W+2p] buffer, then accumulates per (co,ci,ky,kx) as
// a row-major AXPY so the inner loop is contiguous/vectorizable.
std::vector<float> naf_conv2d(const std::vector<float>& x, int Ci, int H, int W,
                               const float* w, const float* b, int Co, int K) {
    const int pad = K / 2;
    const int Hp = H + 2 * pad, Wp = W + 2 * pad;
    std::vector<float> padded((size_t)Ci * Hp * Wp);
    // Per-channel reflect-index fill: each ci writes its own disjoint slab,
    // no cross-channel state -> parallel over ci is a plain copy, order-free.
    naf_parallel_for(Ci, [&](int ci0, int ci1) {
        for (int ci = ci0; ci < ci1; ++ci) {
            const float* src = x.data() + (size_t)ci * H * W;
            float* dst = padded.data() + (size_t)ci * Hp * Wp;
            for (int y = 0; y < Hp; ++y) {
                const int sy = reflect_idx(y - pad, H);
                for (int xx = 0; xx < Wp; ++xx) {
                    const int sx = reflect_idx(xx - pad, W);
                    dst[(size_t)y * Wp + xx] = src[(size_t)sy * W + sx];
                }
            }
        }
    });
    std::vector<float> out((size_t)Co * H * W);
    // Each output channel co accumulates into its own acc[] buffer; the
    // summation order over (ci,ky,kx) for a given co is unchanged by which
    // thread computes it (still ci=0..Ci-1, ky=0..K-1, kx=0..K-1 in order)
    // -> parallelizing over co is bit-identical to the serial version.
    naf_parallel_for(Co, [&](int co0, int co1) {
        for (int co = co0; co < co1; ++co) {
            float* acc = out.data() + (size_t)co * H * W;
            std::fill(acc, acc + (size_t)H * W, b[co]);
            for (int ci = 0; ci < Ci; ++ci) {
                const float* P = padded.data() + (size_t)ci * Hp * Wp;
                const float* wp = w + ((size_t)co * Ci + ci) * K * K;
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        const float wv = wp[ky * K + kx];
                        for (int y = 0; y < H; ++y) {
                            const float* row = P + (size_t)(y + ky) * Wp + kx;
                            float* arow = acc + (size_t)y * W;
                            for (int xx = 0; xx < W; ++xx) arow[xx] += wv * row[xx];
                        }
                    }
                }
            }
        }
    });
    return out;
}

std::vector<float> naf_adaptive_avg_pool2d(const std::vector<float>& x, int C,
                                            int Hin, int Win, int Hout, int Wout) {
    std::vector<float> out((size_t)C * Hout * Wout);
    // Each channel's bins are computed independently (own sum/cnt), so
    // splitting over c leaves the per-bin summation order (yy,xx nested,
    // increasing) untouched -> bit-identical.
    naf_parallel_for(C, [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            const float* xc = x.data() + (size_t)c * Hin * Win;
            float* oc = out.data() + (size_t)c * Hout * Wout;
            for (int oy = 0; oy < Hout; ++oy) {
                const int y0 = (int)((int64_t)oy * Hin / Hout);
                const int y1 = (int)(((int64_t)(oy + 1) * Hin + Hout - 1) / Hout);
                for (int ox = 0; ox < Wout; ++ox) {
                    const int x0 = (int)((int64_t)ox * Win / Wout);
                    const int x1 = (int)(((int64_t)(ox + 1) * Win + Wout - 1) / Wout);
                    double sum = 0; int cnt = 0;
                    for (int yy = y0; yy < y1; ++yy)
                        for (int xx = x0; xx < x1; ++xx) { sum += xc[(size_t)yy * Win + xx]; ++cnt; }
                    oc[(size_t)oy * Wout + ox] = (float)(sum / std::max(cnt, 1));
                }
            }
        }
    });
    return out;
}

std::vector<float> naf_nearest_exact_resize(const std::vector<float>& x, int C,
                                             int Hin, int Win, int Hout, int Wout) {
    std::vector<int> ys(Hout), xs(Wout);
    for (int oy = 0; oy < Hout; ++oy)
        ys[oy] = std::min(Hin - 1, (int)std::floor(((double)oy + 0.5) * Hin / Hout));
    for (int ox = 0; ox < Wout; ++ox)
        xs[ox] = std::min(Win - 1, (int)std::floor(((double)ox + 0.5) * Win / Wout));
    std::vector<float> out((size_t)C * Hout * Wout);
    // Pure gather (no reduction) -> parallel over channels is trivially safe.
    naf_parallel_for(C, [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            const float* xc = x.data() + (size_t)c * Hin * Win;
            float* oc = out.data() + (size_t)c * Hout * Wout;
            for (int oy = 0; oy < Hout; ++oy)
                for (int ox = 0; ox < Wout; ++ox)
                    oc[(size_t)oy * Wout + ox] = xc[(size_t)ys[oy] * Win + xs[ox]];
        }
    });
    return out;
}

std::vector<float> naf_bilinear_resize(const std::vector<float>& x, int C,
                                        int Hin, int Win, int Hout, int Wout) {
    std::vector<float> out((size_t)C * Hout * Wout);
    auto src_coord = [](int dst, int Din, int Dout) -> float {
        if (Din == Dout) return (float)dst;
        const float scale = (float)Din / (float)Dout;
        const float s = ((float)dst + 0.5f) * scale - 0.5f;
        return s < 0.f ? 0.f : s;
    };
    // Elementwise (no reduction) -> parallel over output rows is safe.
    naf_parallel_for(Hout, [&](int oy0, int oy1) {
        for (int oy = oy0; oy < oy1; ++oy) {
            const float sy = src_coord(oy, Hin, Hout);
            int y0 = (int)std::floor(sy); if (y0 > Hin - 1) y0 = Hin - 1;
            int y1 = std::min(y0 + 1, Hin - 1);
            const float fy = sy - std::floor(sy);
            for (int ox = 0; ox < Wout; ++ox) {
                const float sx = src_coord(ox, Win, Wout);
                int x0 = (int)std::floor(sx); if (x0 > Win - 1) x0 = Win - 1;
                int x1 = std::min(x0 + 1, Win - 1);
                const float fx = sx - std::floor(sx);
                for (int c = 0; c < C; ++c) {
                    const float* xc = x.data() + (size_t)c * Hin * Win;
                    const float v00 = xc[(size_t)y0 * Win + x0], v01 = xc[(size_t)y0 * Win + x1];
                    const float v10 = xc[(size_t)y1 * Win + x0], v11 = xc[(size_t)y1 * Win + x1];
                    const float top = v00 * (1 - fx) + v01 * fx;
                    const float bot = v10 * (1 - fx) + v11 * fx;
                    out[(size_t)c * Hout * Wout + (size_t)oy * Wout + ox] = top * (1 - fy) + bot * fy;
                }
            }
        }
    });
    return out;
}

// periods_j = 100^(2j/32), j=0..15 (also present in the checkpoint as
// image_encoder.rope.periods — that's what naf_upsample passes in here).
// coords (i+0.5)/T*2-1 per axis (h=u then w=v); angles[32] = 2pi*[u/periods ||
// v/periods], tiled x2 to 64; rotate_half([x1||x2]) = [-x2||x1] over the two
// 32-wide halves of each 64-channel head.
void naf_rope_apply_inplace(std::vector<float>& x, int T, const std::vector<float>& periods) {
    if ((int)periods.size() != 16) throw std::runtime_error("naf_rope_apply_inplace: periods must have 16 entries");
    const int heads = 4, hd = 64;
    // Each (i,j) pixel's rotation reads/writes only its own `pix` column
    // across all (n,d) -> no cross-pixel state, so splitting the outer i
    // loop across threads (each with its own cosb/sinb/tmp scratch) is
    // elementwise-safe and bit-identical.
    naf_parallel_for(T, [&](int i0, int i1) {
        std::vector<float> cosb(64), sinb(64);
        std::vector<float> tmp(64);
        for (int i = i0; i < i1; ++i) {
            const float u = ((float)i + 0.5f) / (float)T * 2.f - 1.f;
            for (int j = 0; j < T; ++j) {
                const float v = ((float)j + 0.5f) / (float)T * 2.f - 1.f;
                for (int k = 0; k < 16; ++k) {
                    const float au = 2.f * (float)M_PI * u / periods[k];
                    const float av = 2.f * (float)M_PI * v / periods[k];
                    cosb[k] = std::cos(au); sinb[k] = std::sin(au);
                    cosb[16 + k] = std::cos(av); sinb[16 + k] = std::sin(av);
                }
                for (int d = 0; d < 32; ++d) { cosb[32 + d] = cosb[d]; sinb[32 + d] = sinb[d]; }
                const size_t pix = (size_t)i * T + j;
                for (int n = 0; n < heads; ++n) {
                    for (int d = 0; d < hd; ++d) tmp[d] = x[(size_t)(n * hd + d) * T * T + pix];
                    for (int d = 0; d < hd; ++d) {
                        const float rot = (d < 32) ? -tmp[d + 32] : tmp[d - 32];
                        x[(size_t)(n * hd + d) * T * T + pix] = tmp[d] * cosb[d] + rot * sinb[d];
                    }
                }
            }
        }
    });
}

void naf_na_window(int q, int L, int d, int K, int& r, int& p, int& Lr, int& start) {
    r = q % d;
    p = q / d;
    Lr = (L - r + d - 1) / d; // ceil((L-r)/d)
    start = p - K / 2;
    if (start < 0) start = 0;
    const int maxstart = Lr - K;
    if (start > maxstart) start = maxstart;
    if (start < 0) start = 0; // safety net if Lr < K (shouldn't happen for our inputs)
}

// NATTEN: query (y,x) attends only to pixels congruent to it mod (dy,dx); the
// 9x9 window is centered on the query's sub-lattice index and clamped
// (shifted) to stay in bounds — never zero-padded. scale = head_dim^-0.5 = 1/8
// for the 64-wide q/k heads.
void naf_na2d(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
              int T, int dy, int dx, int Cv, float scale, std::vector<float>& out) {
    const int heads = 4, qk_hd = 64, K = 9;
    const int v_hd = Cv / heads;
    const size_t TT = (size_t)T * T;
    out.assign((size_t)Cv * TT, 0.f);

    std::vector<int> ry(T), py(T), Lry(T), starty(T);
    std::vector<int> rx(T), px(T), Lrx(T), startx(T);
    for (int y = 0; y < T; ++y) naf_na_window(y, T, dy, K, ry[y], py[y], Lry[y], starty[y]);
    for (int x = 0; x < T; ++x) naf_na_window(x, T, dx, K, rx[x], px[x], Lrx[x], startx[x]);

    // Each query pixel (y,x) reads only q/k/v and writes only its own qpix
    // column of out[] (op[...+qpix]); distinct y rows never touch the same
    // output element. So splitting the outer y loop across threads (each
    // with its own qv/logits/probs/nyi/nxi scratch) leaves every per-pixel
    // softmax accumulation in the exact same order it ran serially in
    // (n, then wy, wx for the max/sumexp passes, then wy, wx, c for the
    // weighted-sum pass) -> bit-identical to the single-threaded version.
    naf_parallel_for(T, [&](int y0, int y1) {
        std::vector<float> qv(qk_hd), logits(K * K), probs(K * K);
        int nyi[K], nxi[K];
        for (int y = y0; y < y1; ++y) {
            for (int i = 0; i < K; ++i) nyi[i] = (starty[y] + i) * dy + ry[y];
            for (int x = 0; x < T; ++x) {
                for (int i = 0; i < K; ++i) nxi[i] = (startx[x] + i) * dx + rx[x];
                const size_t qpix = (size_t)y * T + x;
                for (int n = 0; n < heads; ++n) {
                    for (int c = 0; c < qk_hd; ++c) qv[c] = q[(size_t)(n * qk_hd + c) * TT + qpix];

                    float maxlog = -std::numeric_limits<float>::infinity();
                    for (int wy = 0; wy < K; ++wy) {
                        for (int wx = 0; wx < K; ++wx) {
                            const size_t kpix = (size_t)nyi[wy] * T + nxi[wx];
                            double dot = 0;
                            for (int c = 0; c < qk_hd; ++c) dot += (double)qv[c] * k[(size_t)(n * qk_hd + c) * TT + kpix];
                            const float lg = (float)(dot * (double)scale);
                            logits[wy * K + wx] = lg;
                            if (lg > maxlog) maxlog = lg;
                        }
                    }
                    double sumexp = 0;
                    for (int i = 0; i < K * K; ++i) { probs[i] = std::exp(logits[i] - maxlog); sumexp += probs[i]; }
                    const double invsum = 1.0 / sumexp;

                    float* op = out.data() + (size_t)n * v_hd * TT;
                    for (int wy = 0; wy < K; ++wy) {
                        for (int wx = 0; wx < K; ++wx) {
                            const float p = (float)(probs[wy * K + wx] * invsum);
                            const size_t vpix = (size_t)nyi[wy] * T + nxi[wx];
                            for (int c = 0; c < v_hd; ++c)
                                op[(size_t)c * TT + qpix] += p * v[(size_t)(n * v_hd + c) * TT + vpix];
                        }
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// full forward
// ---------------------------------------------------------------------------

std::vector<float> naf_upsample(const Model& naf, const float* image, int S,
                                 const float* lr, int C, int h, int w, int T,
                                 NafDebug* dbg) {
#if defined(TRELLIS_USE_CUDA)
    // GPU dispatch: ggml encoder + custom neighborhood-attention CUDA kernel
    // (src/naf_gpu.cpp, src/naf_attn.cu). TRELLIS_NAF_CPU=1 forces this CPU path
    // for A/B against the GPU one. Falls through to the unchanged CPU body below
    // whenever the GPU path isn't available (non-CUDA backend, or a config outside
    // naf_gpu_available()'s preconditions) -- see naf_gpu_available() for exactly
    // which configs qualify.
    {
        const char* force_cpu = std::getenv("TRELLIS_NAF_CPU");
        const bool cpu_forced = force_cpu && *force_cpu && *force_cpu != '0';
        if (!cpu_forced && naf_gpu_available(naf, S, T, h, w))
            return naf_upsample_gpu(naf, image, S, lr, C, h, w, T, dbg);
    }
#endif
    // Every other GPU backend (WebGPU, Metal, Vulkan): the all-ggml graph path
    // (src/naf_gpu.cpp, naf_upsample_ggml). The CPU backend keeps the reference
    // loops below (trellis-test-naf --ggml drives the graph path on it directly).
    {
        const char* force_cpu = std::getenv("TRELLIS_NAF_CPU");
        const bool cpu_forced = force_cpu && *force_cpu && *force_cpu != '0';
        if (!cpu_forced && naf_ggml_available(naf, S, T, h, w))
            return naf_upsample_ggml(naf, image, S, lr, C, h, w, T, dbg);
    }
    // 1. ImageEncoder input: bilinear-downsample only if S > 4T (not hit for
    // S=512, T in {128,512}; implemented per spec anyway).
    int Sp = S;
    std::vector<float> img_vec(image, image + (size_t)3 * S * S);
    if (S > 4 * T) {
        Sp = std::min(S, 4 * T);
        img_vec = naf_bilinear_resize(img_vec, 3, S, S, Sp, Sp);
    }

    BranchW encoder     = load_branch(naf, "image_encoder.encoder");
    BranchW sem_encoder = load_branch(naf, "image_encoder.sem_encoder");
    std::vector<float> periods = tensor_to_f32(naf.get("image_encoder.rope.periods"));

    std::vector<float> e1 = branch_forward(img_vec, Sp, Sp, encoder);      // [128,Sp,Sp]
    std::vector<float> e2 = branch_forward(img_vec, Sp, Sp, sem_encoder);  // [128,Sp,Sp]

    std::vector<float> cat((size_t)256 * Sp * Sp);
    std::memcpy(cat.data(), e1.data(), sizeof(float) * e1.size());
    std::memcpy(cat.data() + e1.size(), e2.data(), sizeof(float) * e2.size());
    if (dbg) { dbg->S_prime = Sp; dbg->enc_cat = cat; }

    std::vector<float> pooled = naf_adaptive_avg_pool2d(cat, 256, Sp, Sp, T, T); // [256,T,T]
    if (dbg) dbg->enc_pooled = pooled;

    naf_rope_apply_inplace(pooled, T, periods); // pooled is now q_rope == q
    if (dbg) dbg->q_rope = pooled;

    std::vector<float> k_pooled = naf_adaptive_avg_pool2d(pooled, 256, T, T, h, w); // [256,h,w]
    if (dbg) dbg->k_pooled = k_pooled;
    std::vector<float> k_up = naf_nearest_exact_resize(k_pooled, 256, h, w, T, T); // [256,T,T]
    if (dbg) dbg->k_up = k_up;

    std::vector<float> lr_vec(lr, lr + (size_t)C * h * w);
    std::vector<float> v_up = naf_nearest_exact_resize(lr_vec, C, h, w, T, T); // [C,T,T]
    if (dbg) dbg->v_up = v_up;

    const int dy = T / h, dx = T / w;
    std::vector<float> out;
    naf_na2d(pooled, k_up, v_up, T, dy, dx, C, 1.0f / 8.0f, out); // scale = 64^-0.5
    return out;
}

} // namespace trellis
