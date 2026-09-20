// CPU port of CuMesh's QEM edge-collapse simplifier (refs/CuMesh/src/simplify.cu +
// dtypes.cuh QEM). Faithful to the reference: Garland-Heckbert vertex quadrics, an
// edge-collapse cost of QEM(v_new) + lambda_edge_length*|e|^2 + lambda_skinny*skinny*|e|^2
// (skinny = mean over incident tris of 1 - normalized-triangle-quality), flip rejection,
// boundary-weighted collapse target, parallel independent-set collapse per round, and the
// threshold-ladder driver (thresh=1e-8, x10 when a round removes <1% of faces, until the
// face target is met). This replaces the meshopt/FQMS stand-in whose non-adaptive, sliny
// output was the geometry-quality gap vs the reference (issues #1/#2).
// CPU 経路は issue #29 でエッジ抽出を CSR 化し、頂点 / エッジ独立な段を parallel_for で
// 並列化した（Metal ビルドは GPU decimate を持たないので常にこの経路。18.4M 面 → 1M 面で
// 単スレッド 118 s だった）。アルゴリズムと float の累積順は変えていない。
#include "uv_bake.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <memory>
#include <utility>
#include "parallel_for.h"

namespace trellis {

// Self-contained CUDA/HIP port (src/decimate_qem.cu). Guarded prototype: only present when
// a GPU backend was compiled in; falls through to the CPU path below on false / failure.
#ifdef TRELLIS_HAVE_GPU_DECIMATE
bool decimate_qem_gpu(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);
#endif

// Self-contained Vulkan compute port (src/decimate_qem_vk.cpp). Runs the same four per-round
// kernels (qem/cost/propagate/collapse) on a headless Vulkan device; the host builds CSR
// adjacency/edges/boundary and does the stream compaction between rounds. Only present when a
// Vulkan backend was compiled in; requires 64-bit shader atomics and falls through to the CPU
// path below on false / failure (no device, missing atomics, alloc/submit error).
#ifdef TRELLIS_HAVE_VK_DECIMATE
bool decimate_qem_vk(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                     int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);
#endif

namespace {

struct QEM {
    float e[10];
    void zero() { for (int i = 0; i < 10; ++i) e[i] = 0.f; }
    // add plane (a,b,c,d) as outer product p p^T (upper triangle of symmetric 4x4)
    void add_plane(float a, float b, float c, float d) {
        e[0]+=a*a; e[1]+=a*b; e[2]+=a*c; e[3]+=a*d;
        e[4]+=b*b; e[5]+=b*c; e[6]+=b*d;
        e[7]+=c*c; e[8]+=c*d; e[9]+=d*d;
    }
    // v^T Q v for v=(x,y,z,1)
    float evaluate(float x, float y, float z) const {
        return e[0]*x*x + 2*e[1]*x*y + 2*e[2]*x*z + 2*e[3]*x
             + e[4]*y*y + 2*e[5]*y*z + 2*e[6]*y
             + e[7]*z*z + 2*e[8]*z    + e[9];
    }
};

struct V3 { float x, y, z; };
inline V3   sub(const V3& a, const V3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline V3   cross(const V3& a, const V3& b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
inline float dot(const V3& a, const V3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float nrm2(const V3& a) { return dot(a, a); }

inline uint64_t pack_cost(int id, float c) { uint32_t b; std::memcpy(&b, &c, 4); return ((uint64_t)b << 32) | (uint32_t)id; }

// TRELLIS_DBG_DECIM=1: ラウンドごとの段別時間（adjacency / edges / qem / cost / propagate /
// collapse / compact）を stderr に印字する。デバッグ用のログトグルで挙動は変えない。
inline bool dbg_decim() { static const bool on = std::getenv("TRELLIS_DBG_DECIM") != nullptr; return on; }
// TRELLIS_DBG_DECIM_THREADS=N: 並列段のスレッド数を固定する（テスト用。結果はスレッド数に依らず同一で、
// src/test_decimate_cpu.cpp が 1/2/3/8/16 で bit 一致を確認する）。未設定なら hardware_concurrency()。
inline int decim_threads() {
    static const int nt = [] {
#ifdef __EMSCRIPTEN__
        return 1;   // pthread 無し: env の上書きも無視して直列（prop も非 atomic なので必須）
#else
        const char* s = std::getenv("TRELLIS_DBG_DECIM_THREADS");
        int n = s ? std::atoi(s) : 0;
        return n > 0 ? n : parallel_threads();
#endif
    }();
    return nt;
}
template <class Fn> inline void decim_for(int64_t n, Fn&& fn) { parallel_for(n, std::forward<Fn>(fn), decim_threads()); }
inline double dbg_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// (a,b) 昇順に並んだ一意エッジを、頂点 a の隣接面から局所的に取り出す。faces に含まれる各面
// (x,y,z) の 3 辺は (x,y),(y,z),(z,x) なので、a を含む面ごとに「a 以外の 2 頂点」が a の相手
// になる。相手 > a のものだけ集めれば、各無向エッジは小さい方の端点でちょうど 1 回列挙される。
// 相手の多重度 = そのエッジを含む面数（1 なら境界辺、3 以上なら非多様体辺）。集めた相手を
// 挿入ソート + RLE して emit(b, count) を b 昇順に呼ぶ。unordered_map 版（変更前）と同じ
// 集合・同じ面数を返し、ID の付番順だけが「(a,b) 昇順」に固定される。
// 前提: 各面は相異なる有効な 3 頂点（decimate_qem_cpu の入口で縮退面・範囲外 index を落として
// 保証する。変更前は縮退面 (a,a,b) から自己ループ辺 (a,a) を作っていた）。
template <class Emit>
inline void local_edges(int a, const std::vector<int32_t>& faces, const std::vector<int>& off,
                        const std::vector<int>& v2f, std::vector<int>& scratch, Emit&& emit) {
    scratch.clear();
    for (int j = off[a]; j < off[a+1]; ++j) {
        const int32_t* ff = &faces[3 * (size_t)v2f[j]];
        for (int k = 0; k < 3; ++k) if (ff[k] > a) scratch.push_back(ff[k]);
    }
    const int n = (int)scratch.size();
    if (n == 0) return;
    int* p = scratch.data();
    if (n <= 32) {   // 典型は 2×deg ≈ 12 本: 挿入ソートが最速
        for (int i = 1; i < n; ++i) { int x = p[i], j = i - 1; while (j >= 0 && p[j] > x) { p[j+1] = p[j]; --j; } p[j+1] = x; }
    } else {
        std::sort(p, p + n);
    }
    for (int i = 0; i < n;) {
        int b = p[i], c = 1;
        while (i + c < n && p[i+c] == b) ++c;
        emit(b, c);
        i += c;
    }
}

// pack した (cost,id) の min を CAS で取る（CUDA k_propagate の atomicMin と同じ）。
inline void atomic_min_u64(std::atomic<uint64_t>& slot, uint64_t p) {
    uint64_t cur = slot.load(std::memory_order_relaxed);
    while (p < cur && !slot.compare_exchange_weak(cur, p, std::memory_order_relaxed)) {}
}

// One simplify round. Mutates verts (positions of kept vertices) + faces (compacted),
// and V/F. Returns nothing; caller reads V/F. Independent-set collapse: an edge collapses
// only if it is the strict minimum-cost edge over every face incident to either endpoint,
// so no two collapses in a round share a face or vertex (matches the CUDA propagate+own check).
//
// 並列化（issue #29）: 頂点独立な段（qem）とエッジ独立な段（cost / propagate / collapse）を
// parallel_for で回す。float の累積順（qem の面順、skinny の面順）は v2f の CSR 順で決まり、
// v2f は直列に面順で組むので変更前と同じ。propagate の min-reduce は順序非依存、collapse の
// 書き込みは勝者ごとに互いに素（勝者は両端点の全隣接面を「所有」し、面や頂点を共有する
// 2 本の勝者は存在しない）。したがって出力は、変更前の実装のエッジ ID を (a,b) 昇順に
// 固定したものと bit 一致する（src/test_decimate_cpu.cpp で判定）。
void simplify_round(std::vector<float>& verts, int& V, std::vector<int32_t>& faces, int& F,
                    float lam_len, float lam_skinny, float thresh) {
    auto vat = [&](int i) -> V3 { return {verts[3*i], verts[3*i+1], verts[3*i+2]}; };
    const bool dbg = dbg_decim();
    double tp[8] = {0}; double tq = dbg ? dbg_now() : 0.0;
    auto phase = [&](int k) { if (dbg) { double t = dbg_now(); tp[k] = t - tq; tq = t; } };
    const int V_in = V, F_in = F;

    // vertex -> incident face adjacency (CSR)。面順の直列 counting sort（qem / skinny の累積順を
    // 変更前と同じに保つため並列化しない。18M 面で ~0.1 s）。
    std::vector<int> off(V + 1, 0);
    for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) off[faces[3*f+k] + 1]++;
    for (int i = 0; i < V; ++i) off[i+1] += off[i];
    std::vector<int> v2f(off[V]);
    { std::vector<int> cur(off.begin(), off.end() - 1);
      for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) { int v = faces[3*f+k]; v2f[cur[v]++] = f; } }
    phase(0);

    // unique undirected edges (sorted by (a,b), CSR by the smaller endpoint) + boundary vertices
    // (edge used by a single face). 2 パス: 頂点ごとの一意数 → prefix sum → 書き込み。
    std::vector<int> eoff((size_t)V + 1, 0);
    decim_for(V, [&](int64_t b, int64_t e) {
        std::vector<int> scratch; scratch.reserve(64);
        for (int a = (int)b; a < (int)e; ++a) {
            int n = 0;
            local_edges(a, faces, off, v2f, scratch, [&](int, int) { ++n; });
            eoff[a + 1] = n;
        }
    });
    for (int i = 0; i < V; ++i) eoff[i+1] += eoff[i];
    const int E = eoff[V];
    std::vector<uint64_t> edges((size_t)E);
    std::vector<uint8_t> eopen((size_t)E, 0);
    decim_for(V, [&](int64_t b, int64_t e) {
        std::vector<int> scratch; scratch.reserve(64);
        for (int a = (int)b; a < (int)e; ++a) {
            int t = eoff[a];
            local_edges(a, faces, off, v2f, scratch, [&](int bb, int c) {
                edges[t] = ((uint64_t)(uint32_t)a << 32) | (uint32_t)bb;
                eopen[t] = (c == 1);
                ++t;
            });
        }
    });
    std::vector<uint8_t> boundary((size_t)V, 0);
    for (int t = 0; t < E; ++t) if (eopen[t]) {
        boundary[(int)(edges[t] >> 32)] = 1; boundary[(int)(edges[t] & 0xffffffffu)] = 1;
    }
    phase(1);

    // per-vertex QEM = sum of incident face plane quadrics (normalized normals)
    std::vector<QEM> qem((size_t)V);
    decim_for(V, [&](int64_t b, int64_t e) {
        for (int i = (int)b; i < (int)e; ++i) {
            qem[i].zero();
            for (int j = off[i]; j < off[i+1]; ++j) {
                int f = v2f[j];
                V3 a = vat(faces[3*f]), bb = vat(faces[3*f+1]), c = vat(faces[3*f+2]);
                V3 n = cross(sub(bb,a), sub(c,a));
                float ln = std::sqrt(nrm2(n));
                if (ln > 1e-20f) { n.x/=ln; n.y/=ln; n.z/=ln; }
                qem[i].add_plane(n.x, n.y, n.z, -dot(n, a));
            }
        }
    });
    phase(2);

    // shape-metric accumulation over the faces incident to `keep` (skip those that will be
    // removed with `other`); returns false if any face flips normal.
    auto process = [&](int keep, int other, const V3& vn, float& skinny, int& ntri) -> bool {
        for (int j = off[keep]; j < off[keep+1]; ++j) {
            int f = v2f[j]; int x = faces[3*f], y = faces[3*f+1], z = faces[3*f+2];
            if (x == other || y == other || z == other) continue;
            V3 a = vat(x), b = vat(y), c = vat(z);
            V3 na = (x==keep)?vn:a, nb = (y==keep)?vn:b, nc = (z==keep)?vn:c;
            V3 on = cross(sub(b,a), sub(c,a));
            V3 ne1 = sub(nb,na), ne2 = sub(nc,na), nn = cross(ne1, ne2);
            if (dot(on, nn) < 0.f) return false;               // flip
            float narea = 0.5f * std::sqrt(nrm2(nn));
            float denom = nrm2(sub(nc,nb)) + nrm2(ne1) + nrm2(ne2);
            if (denom < 1e-12f) denom = 1e-12f;
            float sm = 4.0f * 1.7320508f * narea / denom;       // 1 = equilateral, 0 = degenerate
            skinny += 1.0f - std::min(std::max(sm, 0.0f), 1.0f);
            ntri++;
        }
        return true;
    };

    // edge collapse cost + collapse target（エッジ独立）
    std::vector<float> cost((size_t)E);
    std::vector<V3> vnew((size_t)E);
    decim_for(E, [&](int64_t b, int64_t e) {
        for (int t = (int)b; t < (int)e; ++t) {
            int e0 = (int)(edges[t] >> 32), e1 = (int)(edges[t] & 0xffffffffu);
            V3 v0 = vat(e0), v1 = vat(e1);
            float w0 = 0.5f;
            if (boundary[e0] && !boundary[e1]) w0 = 1.0f;
            else if (!boundary[e0] && boundary[e1]) w0 = 0.0f;
            V3 v = { v0.x*w0 + v1.x*(1-w0), v0.y*w0 + v1.y*(1-w0), v0.z*w0 + v1.z*(1-w0) };
            vnew[t] = v;
            float el2 = nrm2(sub(v1, v0));
            float c = (QEM{{qem[e0].e[0]+qem[e1].e[0], qem[e0].e[1]+qem[e1].e[1], qem[e0].e[2]+qem[e1].e[2],
                            qem[e0].e[3]+qem[e1].e[3], qem[e0].e[4]+qem[e1].e[4], qem[e0].e[5]+qem[e1].e[5],
                            qem[e0].e[6]+qem[e1].e[6], qem[e0].e[7]+qem[e1].e[7], qem[e0].e[8]+qem[e1].e[8],
                            qem[e0].e[9]+qem[e1].e[9]}}).evaluate(v.x, v.y, v.z);
            c += lam_len * el2;
            float skinny = 0.f; int ntri = 0; bool ok = true;
            ok = process(e0, e1, v, skinny, ntri) && process(e1, e0, v, skinny, ntri);
            if (!ok) { cost[t] = std::numeric_limits<float>::infinity(); continue; }
            if (ntri > 0) skinny /= ntri;
            c += lam_skinny * skinny * el2;
            cost[t] = c;
        }
    });
    phase(3);

    // propagate each edge's (cost,id) to all faces touching either endpoint (min-reduce)。
    // 比較は pack_cost の符号なし 64 bit キー（cost の float ビット列 << 32 | id）で行う: 非負有限の
    // cost では数値順と一致し、同コストは id の小さい方が勝つ（丸めで負になった cost は符号ビットで
    // キーが大きくなり負けるだけ、INF / NaN は collapse 側の `cost <= thresh` で落ちる。変更前と同じ）。min は順序非依存なので並列の interleaving は結果に
    // 現れない。CAS ループは relaxed で足りる（値の min-reduce だけで、後段は join 後に読む）。
    // Emscripten（pthread 無し、直列）では atomic 型を使わず素の配列に min を書く。
#ifdef __EMSCRIPTEN__
    std::vector<uint64_t> prop((size_t)F, UINT64_MAX);
    auto prop_min = [&](int f, uint64_t p) { if (p < prop[f]) prop[f] = p; };
    auto prop_get = [&](int f) { return prop[f]; };
#else
    // 主要ターゲット（x86-64 / arm64 / wasm）では lock-free。そうでない環境でも mutex 実装で正しさは保たれる。
    std::unique_ptr<std::atomic<uint64_t>[]> prop(new std::atomic<uint64_t>[(size_t)std::max(F, 1)]);
    decim_for(F, [&](int64_t b, int64_t e) {
        for (int64_t f = b; f < e; ++f) prop[f].store(UINT64_MAX, std::memory_order_relaxed);
    });
    auto prop_min = [&](int f, uint64_t p) { atomic_min_u64(prop[f], p); };
    auto prop_get = [&](int f) { return prop[f].load(std::memory_order_relaxed); };
#endif
    decim_for(E, [&](int64_t b, int64_t e) {
        for (int t = (int)b; t < (int)e; ++t) {
            uint64_t p = pack_cost(t, cost[t]);
            int e0 = (int)(edges[t] >> 32), e1 = (int)(edges[t] & 0xffffffffu);
            for (int j = off[e0]; j < off[e0+1]; ++j) prop_min(v2f[j], p);
            for (int j = off[e1]; j < off[e1+1]; ++j) prop_min(v2f[j], p);
        }
    });
    phase(4);

    // collapse edges that win every incident face and are under threshold。勝者は e0 / e1 の
    // 全隣接面を所有するので、verts[e0] / vdead[e1] / fdead / faces への書き込みは勝者間で互いに素。
    std::vector<uint8_t> vdead((size_t)V, 0), fdead((size_t)F, 0);
    decim_for(E, [&](int64_t b, int64_t e) {
        for (int t = (int)b; t < (int)e; ++t) {
            if (!(cost[t] <= thresh)) continue;
            uint64_t p = pack_cost(t, cost[t]);
            int e0 = (int)(edges[t] >> 32), e1 = (int)(edges[t] & 0xffffffffu);
            bool own = true;
            for (int j = off[e0]; j < off[e0+1] && own; ++j) if (prop_get(v2f[j]) != p) own = false;
            for (int j = off[e1]; j < off[e1+1] && own; ++j) if (prop_get(v2f[j]) != p) own = false;
            if (!own) continue;
            verts[3*e0] = vnew[t].x; verts[3*e0+1] = vnew[t].y; verts[3*e0+2] = vnew[t].z;
            vdead[e1] = 1;
            for (int j = off[e0]; j < off[e0+1]; ++j) { int f = v2f[j]; int* ff = &faces[3*f]; if (ff[0]==e1||ff[1]==e1||ff[2]==e1) fdead[f] = 1; }
            for (int j = off[e1]; j < off[e1+1]; ++j) { int f = v2f[j]; int* ff = &faces[3*f]; for (int k = 0; k < 3; ++k) if (ff[k]==e1) ff[k] = e0; }
        }
    });
    phase(5);

    // compact vertices (drop collapsed) + faces (drop deleted/degenerate)
    std::vector<int> vmap((size_t)V, -1); int nV = 0;
    for (int i = 0; i < V; ++i) if (!vdead[i]) vmap[i] = nV++;
    std::vector<float> nv((size_t)nV * 3);
    for (int i = 0; i < V; ++i) if (vmap[i] >= 0) { nv[3*vmap[i]] = verts[3*i]; nv[3*vmap[i]+1] = verts[3*i+1]; nv[3*vmap[i]+2] = verts[3*i+2]; }
    std::vector<int32_t> nf; nf.reserve(faces.size());
    for (int f = 0; f < F; ++f) {
        if (fdead[f]) continue;
        int a = vmap[faces[3*f]], b = vmap[faces[3*f+1]], c = vmap[faces[3*f+2]];
        if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c) continue;
        nf.push_back(a); nf.push_back(b); nf.push_back(c);
    }
    // Set V/F from the NEW buffers BEFORE swapping (after faces.swap(nf), nf aliases the
    // old, larger buffer -- reading its size there set F too high and walked past the mesh).
    V = nV; F = (int)nf.size() / 3;
    verts.swap(nv); faces.swap(nf);
    phase(6);
    if (dbg) {
        double tot = 0; for (int k = 0; k < 7; ++k) tot += tp[k];
        fprintf(stderr, "[decim] V %d->%d F %d->%d E=%d thresh=%.1e | adj %.2f edges %.2f qem %.2f cost %.2f prop %.2f collapse %.2f compact %.2f | %.2f s\n",
                V_in, V, F_in, F, E, (double)thresh, tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], tp[6], tot);
    }
}

} // namespace

static void decimate_qem_cpu_clean(const std::vector<float>& in_verts, int V0, const std::vector<int32_t>& in_faces, int F0,
                                   int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);

// 入口の面検査（decimate_qem / decimate_qem_cpu 共通）: 範囲外 index と同じ頂点を 2 回含む
// 縮退面を落とす（各ラウンド末の圧縮と同じ規則）。これで local_edges の前提「各面は相異なる
// 3 頂点」が成り立つ。変更前の実装は縮退面 (a,a,b) から自己ループ辺 (a,a) を作り、それが勝つと
// a とその全隣接面を消していた。int の添字（3F、E）が溢れる入力は契約外として弾く。
static bool sanitize_faces(int V0, const std::vector<int32_t>& in_faces, int F0, std::vector<int32_t>& faces) {
    if (F0 < 0 || (int64_t)F0 * 3 > (int64_t)INT32_MAX || (size_t)F0 * 3 > in_faces.size()) {
        fprintf(stderr, "[decimate] invalid face count %d (max %d, buffer %zu); mesh left untouched\n",
                F0, INT32_MAX / 3, in_faces.size() / 3);
        return false;
    }
    faces.clear(); faces.reserve((size_t)F0 * 3);
    for (int f = 0; f < F0; ++f) {
        int a = in_faces[3*f], b = in_faces[3*f+1], c = in_faces[3*f+2];
        if (a < 0 || b < 0 || c < 0 || a >= V0 || b >= V0 || c >= V0 || a == b || b == c || a == c) continue;
        faces.push_back(a); faces.push_back(b); faces.push_back(c);
    }
    const int F = (int)(faces.size() / 3);
    if (F != F0) { printf("  decimate_qem: dropped %d invalid/degenerate input faces\n", F0 - F); fflush(stdout); }
    return true;
}

void decimate_qem(const std::vector<float>& in_verts, int V0, const std::vector<int32_t>& in_faces, int F0,
                  int target_faces, std::vector<float>& ov, std::vector<int32_t>& of) {
    std::vector<int32_t> faces;
    if (!sanitize_faces(V0, in_faces, F0, faces)) { ov = in_verts; of = in_faces; return; }
    const int F = (int)(faces.size() / 3);
    if (F <= target_faces) { ov = in_verts; of = faces; return; }   // pass-through (cleaned faces, positions untouched)

#ifdef TRELLIS_HAVE_GPU_DECIMATE
    // Run the whole simplification on the GPU when a CUDA/HIP backend is built in; on any
    // failure (no device, alloc/kernel error) fall through to the validated CPU path.
    if (decimate_qem_gpu(in_verts, V0, faces, F, target_faces, ov, of)) return;
    // Any message above (e.g. "device kernel image is invalid" when the kernel was
    // built for a different GPU arch — issue #14) is non-fatal: the mesh is still
    // decimated correctly on the CPU below, just slower.
    fprintf(stderr, "[decimate] GPU decimation unavailable; falling back to the CPU path (output is unaffected)\n");
#endif

#ifdef TRELLIS_HAVE_VK_DECIMATE
    // Same, on a headless Vulkan compute device (used in Vulkan-only builds with no CUDA/HIP
    // kernel). Falls through to the CPU path on any failure or when the device lacks 64-bit atomics.
    if (decimate_qem_vk(in_verts, V0, faces, F, target_faces, ov, of)) return;
#endif

    decimate_qem_cpu_clean(in_verts, V0, faces, F, target_faces, ov, of);
}

void decimate_qem_cpu(const std::vector<float>& in_verts, int V0, const std::vector<int32_t>& in_faces, int F0,
                      int target_faces, std::vector<float>& ov, std::vector<int32_t>& of) {
    std::vector<int32_t> faces;
    if (!sanitize_faces(V0, in_faces, F0, faces)) { ov = in_verts; of = in_faces; return; }
    const int F = (int)(faces.size() / 3);
    if (F <= target_faces) { ov = in_verts; of = faces; return; }   // pass-through (cleaned faces, positions untouched)
    decimate_qem_cpu_clean(in_verts, V0, faces, F, target_faces, ov, of);
}

// 検査済み（相異なる有効 3 頂点、F > target）の面を受ける CPU ラダー本体。
static void decimate_qem_cpu_clean(const std::vector<float>& in_verts, int V0, const std::vector<int32_t>& in_faces, int F0,
                                   int target_faces, std::vector<float>& ov, std::vector<int32_t>& of) {
    std::vector<float> verts = in_verts;
    std::vector<int32_t> faces = in_faces;
    int V = V0, F = F0;

    float thresh = 1e-8f;
    const float lam_len = 1e-2f, lam_skinny = 1e-3f;
    int prevF = F, stalls = 0;
    for (int round = 0; round < 400 && F > target_faces; ++round) {
        simplify_round(verts, V, faces, F, lam_len, lam_skinny, thresh);
        if (F <= target_faces) break;
        int removed = prevF - F;
        if (removed <= 0) { if (++stalls >= 2) { thresh *= 10.0f; stalls = 0; } }
        else { stalls = 0; if ((float)removed / prevF < 1e-2f) thresh *= 10.0f; }
        prevF = F;
        if (thresh > 1e12f) break;   // fully collapsed within tolerance; stop escalating
    }

    // final compaction to referenced vertices only (drop any isolated leftovers)
    std::vector<int> used((size_t)V, -1); int nV = 0;
    for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) { int v = faces[3*f+k]; if (used[v] < 0) used[v] = nV++; }
    ov.assign((size_t)nV * 3, 0.f);
    for (int i = 0; i < V; ++i) if (used[i] >= 0) { ov[3*used[i]] = verts[3*i]; ov[3*used[i]+1] = verts[3*i+1]; ov[3*used[i]+2] = verts[3*i+2]; }
    of.resize((size_t)F * 3);
    for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) of[3*f+k] = used[faces[3*f+k]];
    printf("  decimate_qem(target=%d): V %d->%d, F %d->%d (thresh=%.1e, %d threads)\n", target_faces, V0, nV, F0, F, (double)thresh, decim_threads());
    fflush(stdout);
}

} // namespace trellis
