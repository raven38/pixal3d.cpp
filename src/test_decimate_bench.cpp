// decimate_qem のベンチ / A-B ハーネス（issue #29）。
//   手続きメッシュ:  trellis-test-decimate-bench [--legacy] torus <R> [target] [out.bin]
//                    （R×R のトーラス = 2R² 面。R=3000 で ~18M 面、MV 1024 の remesh 出力と同規模）
//   dump-post から:  trellis-test-decimate-bench [--legacy] <dump.bin> [target] [out.bin]
//                    （`trellis-cli --dump-post` の layout: i32 V,F,Mv,res; f32 verts; i32 faces; …
//                     PBR 部分は読まない）
//   --legacy: 変更前の CPU 実装（エッジ ID を (a,b) 昇順に固定したもの、test_decimate_qem_legacy.inc）
//             を走らせる。新実装とは bit 一致するはずなので compare で突き合わせる（不一致は exit 3）。
// 出力メッシュ（i32 V,F; f32 verts; i32 faces）を out.bin に書けば、別実装の結果と
// `compare <a.bin> <b.bin>` で V/F/bbox/開放辺/連結成分を突き合わせられる（不一致は exit 3）。
//   dist <a.bin> <b.bin> [N]: 両メッシュから面積重みで N 点（既定 500k）ずつサンプルし、相手メッシュ
//   への正確な最近点距離（TriBvh）を両方向で取る。mean / p50 / p95 / p99 / max（= サンプル Hausdorff）
//   を bbox 対角比でも印字する。
#include "uv_bake.h"
#include "tri_bvh.h"
#include <chrono>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <limits>

#include "test_decimate_qem_legacy.inc"

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void make_torus(int R, std::vector<float>& v, std::vector<int32_t>& f) {
    const float Rmaj = 1.0f, rmin = 0.35f, TAU = 6.2831853f;
    v.resize((size_t)R * R * 3);
    for (int i = 0; i < R; ++i) for (int j = 0; j < R; ++j) {
        float u = TAU * i / R, w = TAU * j / R;
        float* p = &v[3 * ((size_t)i * R + j)];
        p[0] = (Rmaj + rmin * std::cos(w)) * std::cos(u);
        p[1] = (Rmaj + rmin * std::cos(w)) * std::sin(u);
        p[2] = rmin * std::sin(w);
    }
    f.reserve((size_t)R * R * 6);
    for (int i = 0; i < R; ++i) for (int j = 0; j < R; ++j) {
        int i1 = (i + 1) % R, j1 = (j + 1) % R;
        int a = i * R + j, b = i1 * R + j, c = i1 * R + j1, d = i * R + j1;
        f.push_back(a); f.push_back(b); f.push_back(c);
        f.push_back(a); f.push_back(c); f.push_back(d);
    }
}

// union-find で連結成分数（頂点共有ベース）
static int count_components(int V, const std::vector<int32_t>& f) {
    std::vector<int> parent(V);
    for (int i = 0; i < V; ++i) parent[i] = i;
    auto find = [&](int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const size_t F = f.size() / 3;
    for (size_t t = 0; t < F; ++t) {
        int a = find(f[3*t]), b = find(f[3*t+1]), c = find(f[3*t+2]);
        if (a != b) parent[a] = b;
        b = find(b);
        if (b != c) parent[b] = c;
    }
    std::vector<uint8_t> used(V, 0);
    for (size_t t = 0; t < F; ++t) for (int k = 0; k < 3; ++k) used[f[3*t+k]] = 1;
    int n = 0;
    for (int i = 0; i < V; ++i) if (used[i] && find(i) == i) ++n;
    return n;
}

struct Stats { int V, F; float bb[6]; long open, nonmanifold; int comps; bool finite; };
static Stats analyze(const std::vector<float>& v, const std::vector<int32_t>& f) {
    Stats s; s.finite = true;
    s.V = (int)v.size()/3; s.F = (int)f.size()/3;
    s.bb[0]=s.bb[1]=s.bb[2]= 1e30f; s.bb[3]=s.bb[4]=s.bb[5]=-1e30f;
    for (int i = 0; i < s.V; ++i) for (int k = 0; k < 3; ++k) {
        float x = v[3*i+k];
        if (!std::isfinite(x)) s.finite = false;
        if (x < s.bb[k]) s.bb[k] = x;
        if (x > s.bb[3+k]) s.bb[3+k] = x;
    }
    std::unordered_map<uint64_t,int> ec; ec.reserve((size_t)s.F*2);
    auto key=[](int a,int b){ if(a>b){int t=a;a=b;b=t;} return ((uint64_t)(uint32_t)a<<32)|(uint32_t)b; };
    for (int t = 0; t < s.F; ++t) { int a=f[3*t],b=f[3*t+1],c=f[3*t+2];
        ec[key(a,b)]++; ec[key(b,c)]++; ec[key(c,a)]++; }
    s.open = 0; s.nonmanifold = 0;
    for (auto& kv : ec) { if (kv.second == 1) s.open++; else if (kv.second > 2) s.nonmanifold++; }
    s.comps = count_components(s.V, f);
    return s;
}
static void report(const char* tag, const std::vector<float>& v, const std::vector<int32_t>& f) {
    Stats s = analyze(v, f);
    std::printf("  [%s] V=%d F=%d bbox=[%.5f,%.5f,%.5f]..[%.5f,%.5f,%.5f] open_edges=%ld nonmanifold=%ld comps=%d finite=%d\n",
                tag, s.V, s.F, s.bb[0],s.bb[1],s.bb[2], s.bb[3],s.bb[4],s.bb[5], s.open, s.nonmanifold, s.comps, (int)s.finite);
    std::fflush(stdout);
}

static bool read_mesh(const char* path, std::vector<float>& v, std::vector<int32_t>& f) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    int V = 0, F = 0;
    if (std::fread(&V, 4, 1, fp) != 1 || std::fread(&F, 4, 1, fp) != 1) { std::fclose(fp); return false; }
    v.resize((size_t)V * 3); f.resize((size_t)F * 3);
    if (std::fread(v.data(), 4, v.size(), fp) != v.size()) { std::fclose(fp); return false; }
    if (std::fread(f.data(), 4, f.size(), fp) != f.size()) { std::fclose(fp); return false; }
    std::fclose(fp);
    return true;
}
static bool write_mesh(const char* path, const std::vector<float>& v, const std::vector<int32_t>& f) {
    FILE* fp = std::fopen(path, "wb");
    if (!fp) return false;
    int V = (int)v.size()/3, F = (int)f.size()/3;
    std::fwrite(&V,4,1,fp); std::fwrite(&F,4,1,fp);
    std::fwrite(v.data(),4,v.size(),fp); std::fwrite(f.data(),4,f.size(),fp);
    std::fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s torus <R> [target] [out.bin]\n"
                             "       %s <dump-post.bin> [target] [out.bin]\n"
                             "       %s compare <a.bin> <b.bin>\n", argv[0], argv[0], argv[0]);
        return 1;
    }
    int argi = 1;
    bool use_legacy = false;
    if (std::string(argv[argi]) == "--legacy") { use_legacy = true; ++argi; }
    if (argc <= argi) return 1;
    std::string mode = argv[argi];
    if (mode == "compare") {
        if (argc < argi + 3) return 1;
        std::vector<float> va, vb; std::vector<int32_t> fa, fb;
        if (!read_mesh(argv[argi+1], va, fa) || !read_mesh(argv[argi+2], vb, fb)) return 1;
        report("a", va, fa);
        report("b", vb, fb);
        bool same = va.size() == vb.size() && fa.size() == fb.size()
                 && std::memcmp(va.data(), vb.data(), va.size()*4) == 0
                 && std::memcmp(fa.data(), fb.data(), fa.size()*4) == 0;
        std::printf("  bitwise identical: %s\n", same ? "yes" : "no");
        return same ? 0 : 3;
    }

    if (mode == "dist") {
        if (argc < argi + 3) return 1;
        std::vector<float> va, vb; std::vector<int32_t> fa, fb;
        if (!read_mesh(argv[argi+1], va, fa) || !read_mesh(argv[argi+2], vb, fb)) return 1;
        const int N = argc > argi + 3 ? std::atoi(argv[argi+3]) : 500000;
        Stats sa = analyze(va, fa);
        double diag = 0; for (int k = 0; k < 3; ++k) diag += (double)(sa.bb[3+k]-sa.bb[k])*(sa.bb[3+k]-sa.bb[k]);
        diag = std::sqrt(diag);
        auto sample = [&](const std::vector<float>& v, const std::vector<int32_t>& f, std::vector<float>& pts) {
            const int F = (int)f.size()/3;
            std::vector<double> cum((size_t)F + 1, 0.0);
            for (int t = 0; t < F; ++t) {
                const float* a=&v[3*f[3*t]], *b=&v[3*f[3*t+1]], *c=&v[3*f[3*t+2]];
                double u[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, w[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
                double n[3]={u[1]*w[2]-u[2]*w[1], u[2]*w[0]-u[0]*w[2], u[0]*w[1]-u[1]*w[0]};
                cum[t+1] = cum[t] + 0.5*std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            }
            std::mt19937_64 rng(12345);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            pts.resize((size_t)N*3);
            for (int i = 0; i < N; ++i) {
                double r = U(rng) * cum[F];
                int t = (int)(std::upper_bound(cum.begin(), cum.end(), r) - cum.begin()) - 1;
                if (t < 0) t = 0; if (t >= F) t = F - 1;
                double r1 = std::sqrt(U(rng)), r2 = U(rng);
                double wa = 1 - r1, wb = r1 * (1 - r2), wc = r1 * r2;
                const float* a=&v[3*f[3*t]], *b=&v[3*f[3*t+1]], *c=&v[3*f[3*t+2]];
                for (int k = 0; k < 3; ++k) pts[3*i+k] = (float)(wa*a[k] + wb*b[k] + wc*c[k]);
            }
        };
        auto side = [&](const char* tag, const std::vector<float>& vs, const std::vector<int32_t>& fs,
                        const std::vector<float>& vt, const std::vector<int32_t>& ft) {
            std::vector<float> pts; sample(vs, fs, pts);
            trellis::TriBvh bvh = trellis::TriBvh::build(vt.data(), (int64_t)vt.size()/3, ft.data(), (int64_t)ft.size()/3);
            std::vector<double> d((size_t)N);
            double sum = 0;
            for (int i = 0; i < N; ++i) { auto h = bvh.closest(&pts[3*i]); d[i] = std::sqrt((double)h.dist2); sum += d[i]; }
            std::sort(d.begin(), d.end());
            auto q = [&](double p) { return d[std::min((size_t)(p * N), (size_t)N - 1)]; };
            std::printf("  [%s] N=%d mean=%.3e p50=%.3e p95=%.3e p99=%.3e max=%.3e  (max/diag=%.2e, diag=%.4f)\n",
                        tag, N, sum / N, q(0.5), q(0.95), q(0.99), d[N-1], d[N-1] / diag, diag);
        };
        side("a->b", va, fa, vb, fb);
        side("b->a", vb, fb, va, fa);
        return 0;
    }

    std::vector<float> v; std::vector<int32_t> f;
    int ai = argi + 1;
    if (mode == "torus") {
        int R = argc > argi + 1 ? std::atoi(argv[argi+1]) : 500;
        make_torus(R, v, f);
        ai = argi + 2;
    } else {
        // dump-post layout: i32 V,F,Mv,res; f32 verts[V*3]; i32 faces[F*3]; (PBR 以降は無視)
        FILE* fp = std::fopen(argv[argi], "rb");
        if (!fp) { std::fprintf(stderr, "cannot open %s\n", argv[argi]); return 1; }
        int V, F, Mv, res;
        if (std::fread(&V,4,1,fp)+std::fread(&F,4,1,fp)+std::fread(&Mv,4,1,fp)+std::fread(&res,4,1,fp) != 4) return 1;
        v.resize((size_t)V*3); f.resize((size_t)F*3);
        if (std::fread(v.data(),4,v.size(),fp) != v.size()) return 1;
        if (std::fread(f.data(),4,f.size(),fp) != f.size()) return 1;
        std::fclose(fp);
        std::printf("loaded %s: V=%d F=%d voxels=%d res=%d\n", argv[argi], V, F, Mv, res);
    }
    int target = argc > ai ? std::atoi(argv[ai]) : 1000000;
    const char* out = argc > ai + 1 ? argv[ai + 1] : nullptr;

    const int V = (int)v.size()/3, F = (int)f.size()/3;
    std::printf("input V=%d F=%d target=%d\n", V, F, target);
    {   // 頂点次数の分布（局所エッジ抽出の作業量 = 2×次数。p50/p95/p99/max）
        std::vector<int> deg((size_t)V, 0);
        for (int t = 0; t < F; ++t) for (int k = 0; k < 3; ++k) deg[f[3*t+k]]++;
        std::vector<int> d(deg); std::sort(d.begin(), d.end());
        auto q = [&](double p) { return d.empty() ? 0 : d[std::min((size_t)(p * d.size()), d.size() - 1)]; };
        std::printf("  degree: p50=%d p95=%d p99=%d max=%d\n", q(0.5), q(0.95), q(0.99), d.empty() ? 0 : d.back());
    }
    std::fflush(stdout);

    std::vector<float> ov; std::vector<int32_t> of;
    double t0 = now();
    if (use_legacy) legacy::decimate_qem_legacy_sorted(v, V, f, F, target, ov, of);
    else            trellis::decimate_qem_cpu(v, V, f, F, target, ov, of);   // GPU ビルドでも CPU 経路を測る
    double dt = now() - t0;
    std::printf("%s: %.2f s\n", use_legacy ? "decimate_qem_legacy_sorted" : "decimate_qem", dt);
    report("output", ov, of);
    if (out) { if (write_mesh(out, ov, of)) std::printf("wrote %s\n", out); }
    return 0;
}
