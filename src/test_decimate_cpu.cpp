// CPU decimate_qem の回帰テスト（issue #29 の並列化）。
// 新実装は「変更前の実装にエッジ ID の (a,b) 昇順固定だけを足したもの」（test_decimate_qem_legacy.inc）
// と全段で float の計算順が同じになる設計なので、出力は bit 一致するはず。ここではそれを memcmp で
// 判定し、あわせて幾何の健全性（有限・閉メッシュなら開放辺 0・成分数 1・目標面数以下）を見る。
//   ./trellis-test-decimate-cpu            全ケース（数十秒）
//   ./trellis-test-decimate-cpu <R>        torus の解像度を変える（既定 300 = 180k 面）
// decimate_qem_cpu() を直接呼ぶので、CUDA / Vulkan の decimate が有効なビルドでも CPU 経路
// （= GPU 失敗時の fallback）を検証する。スレッド数は TRELLIS_DBG_DECIM_THREADS で固定でき、
// 最後のケースで 1/2/3/8/16 スレッドの出力が bit 一致することを確認する（子プロセスで実行）。
#include "uv_bake.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "test_decimate_qem_legacy.inc"

static void make_torus(int R, std::vector<float>& v, std::vector<int32_t>& f, int skip_col = -1) {
    const float Rmaj = 1.0f, rmin = 0.35f, TAU = 6.2831853f;
    v.resize((size_t)R * R * 3);
    for (int i = 0; i < R; ++i) for (int j = 0; j < R; ++j) {
        float u = TAU * i / R, w = TAU * j / R;
        float* p = &v[3 * ((size_t)i * R + j)];
        p[0] = (Rmaj + rmin * std::cos(w)) * std::cos(u);
        p[1] = (Rmaj + rmin * std::cos(w)) * std::sin(u);
        p[2] = rmin * std::sin(w);
    }
    for (int i = 0; i < R; ++i) for (int j = 0; j < R; ++j) {
        if (i == skip_col) continue;   // 1 列抜くと開いたメッシュ（境界頂点あり）になる
        int i1 = (i + 1) % R, j1 = (j + 1) % R;
        int a = i * R + j, b = i1 * R + j, c = i1 * R + j1, d = i * R + j1;
        f.push_back(a); f.push_back(b); f.push_back(c);
        f.push_back(a); f.push_back(c); f.push_back(d);
    }
}

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

struct Stats { long open, nonmanifold; int comps; bool finite; bool valid_idx; };
static Stats analyze(const std::vector<float>& v, const std::vector<int32_t>& f) {
    Stats s; s.finite = true; s.valid_idx = true;
    const int V = (int)v.size()/3, F = (int)f.size()/3;
    for (float x : v) if (!std::isfinite(x)) s.finite = false;
    std::unordered_map<uint64_t,int> ec; ec.reserve((size_t)F*2);
    auto key=[](int a,int b){ if(a>b){int t=a;a=b;b=t;} return ((uint64_t)(uint32_t)a<<32)|(uint32_t)b; };
    for (int t = 0; t < F; ++t) {
        int a=f[3*t],b=f[3*t+1],c=f[3*t+2];
        if (a < 0 || b < 0 || c < 0 || a >= V || b >= V || c >= V || a == b || b == c || a == c) s.valid_idx = false;
        ec[key(a,b)]++; ec[key(b,c)]++; ec[key(c,a)]++;
    }
    s.open = 0; s.nonmanifold = 0;
    for (auto& kv : ec) { if (kv.second == 1) s.open++; else if (kv.second > 2) s.nonmanifold++; }
    s.comps = count_components(V, f);
    return s;
}

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

// 1 ケース: legacy(sorted) と新実装を同じ入力で走らせ、bit 一致と健全性を判定する。
// compare_legacy=false は「変更前の挙動が未定義な入力」（縮退面・範囲外 index）用で、新実装の
// 健全性だけを見る。
// closed: 閉メッシュとして開放辺 0 を要求。check_topo: 境界・成分数の保存を見る（目標 0 や
// 面数が数枚の入力は全部つぶれてよいので見ない）。
static void run_case(const char* name, const std::vector<float>& v, const std::vector<int32_t>& f, int target,
                     bool closed, bool compare_legacy = true, bool check_topo = true) {
    const int V = (int)v.size()/3, F = (int)f.size()/3;
    std::printf("[%s] V=%d F=%d target=%d\n", name, V, F, target);
    std::vector<float> ov_n; std::vector<int32_t> of_n;
    trellis::decimate_qem_cpu(v, V, f, F, target, ov_n, of_n);
    if (compare_legacy) {
        std::vector<float> ov_l; std::vector<int32_t> of_l;
        legacy::decimate_qem_legacy_sorted(v, V, f, F, target, ov_l, of_l);
        const bool same = ov_l.size() == ov_n.size() && of_l.size() == of_n.size()
                       && std::memcmp(ov_l.data(), ov_n.data(), ov_l.size()*4) == 0
                       && std::memcmp(of_l.data(), of_n.data(), of_l.size()*4) == 0;
        std::printf("  legacy V=%zu F=%zu | new V=%zu F=%zu\n", ov_l.size()/3, of_l.size()/3, ov_n.size()/3, of_n.size()/3);
        check(same, "bitwise identical to legacy(sorted)");
    } else {
        std::printf("  new V=%zu F=%zu\n", ov_n.size()/3, of_n.size()/3);
    }
    Stats s = analyze(ov_n, of_n);
    Stats si = analyze(v, f);
    check(s.finite, "all vertices finite");
    check(s.valid_idx, "face indices valid and non-degenerate");
    if (F > target) check((int)of_n.size()/3 <= target, "F <= target");
    else check(of_n == f && ov_n == v, "input at/below target is passed through unchanged");
    if (closed) check(s.open == 0, "closed input stays closed (open_edges == 0)");
    else if (check_topo && si.open > 0) check(s.open > 0 && s.open <= si.open, "open input keeps a boundary no longer than the input's");
    if (check_topo) check(s.comps == si.comps, "component count preserved");
}

// スレッド数を変えて同じ入力を子プロセスで走らせ、出力ファイルが bit 一致することを見る
static void thread_determinism(const char* self, int R) {
    std::printf("[threads] R=%d, TRELLIS_DBG_DECIM_THREADS in {1,2,3,8,16}\n", R);
    const int nts[] = {1, 2, 3, 8, 16, 1, 2, 3, 8, 16, 1, 2, 3, 8, 16};   // 各スレッド数 3 回（断続的な競合も拾う）
    std::vector<std::vector<char>> outs;
    for (int nt : nts) {
        std::string path = std::string("/tmp/trellis-test-decimate-cpu-") + std::to_string(nt) + ".bin";
        std::string cmd = std::string("TRELLIS_DBG_DECIM_THREADS=") + std::to_string(nt) + " \"" + self +
                          "\" --emit " + std::to_string(R) + " " + path + " > /dev/null";
        int rc = std::system(cmd.c_str());
        FILE* fp = std::fopen(path.c_str(), "rb");
        std::vector<char> buf;
        if (fp) { std::fseek(fp, 0, SEEK_END); long n = std::ftell(fp); std::fseek(fp, 0, SEEK_SET);
                  buf.resize((size_t)n); if (n > 0 && std::fread(buf.data(), 1, (size_t)n, fp) != (size_t)n) buf.clear(); std::fclose(fp); }
        std::remove(path.c_str());
        check(rc == 0 && !buf.empty(), (std::string("run with ") + std::to_string(nt) + " threads").c_str());
        outs.push_back(std::move(buf));
    }
    bool same = !outs.empty() && !outs[0].empty();
    for (size_t i = 1; i < outs.size(); ++i) if (outs[i].empty() || outs[i] != outs[0]) same = false;
    check(same, "outputs bit-identical across thread counts (3 runs each)");
}

int main(int argc, char** argv) {
    // --emit <R> <out.bin>: 子プロセスモード（thread_determinism から呼ばれる）。torus R を減らして
    // 出力メッシュ（i32 V,F; f32 verts; i32 faces）を書くだけ。
    if (argc == 4 && std::string(argv[1]) == "--emit") {
        const int R = std::atoi(argv[2]);
        std::vector<float> v; std::vector<int32_t> f;
        make_torus(R, v, f, /*skip_col=*/R / 3);
        std::vector<float> ov; std::vector<int32_t> of;
        trellis::decimate_qem_cpu(v, (int)v.size()/3, f, (int)f.size()/3, (int)(f.size()/3) / 10, ov, of);
        FILE* fp = std::fopen(argv[3], "wb");
        if (!fp) return 2;
        int V = (int)ov.size()/3, F = (int)of.size()/3;
        std::fwrite(&V,4,1,fp); std::fwrite(&F,4,1,fp);
        std::fwrite(ov.data(),4,ov.size(),fp); std::fwrite(of.data(),4,of.size(),fp);
        std::fclose(fp);
        return 0;
    }
    const int R = argc > 1 ? std::atoi(argv[1]) : 300;
    {   // 閉じた torus（規則格子: コストのタイが多く、エッジ ID の付番順が効くケース）
        std::vector<float> v; std::vector<int32_t> f;
        make_torus(R, v, f);
        run_case("torus-closed", v, f, (int)(f.size()/3) / 10, true);
    }
    {   // 1 列抜いた開いた torus（境界重み w0 の分岐を通す）
        std::vector<float> v; std::vector<int32_t> f;
        make_torus(R, v, f, /*skip_col=*/R / 2);
        run_case("torus-open", v, f, (int)(f.size()/3) / 10, false);
    }
    {   // 目標面数以下の入力: そのまま返す
        std::vector<float> v; std::vector<int32_t> f;
        make_torus(20, v, f);
        run_case("below-target", v, f, (int)(f.size()/3) + 1, true);
    }
    {   // 小さな閉メッシュを極端に減らす（ラダーが 1e12 まで上がる／stall 分岐）
        std::vector<float> v; std::vector<int32_t> f;
        make_torus(40, v, f);
        run_case("tiny-target", v, f, 8, true);
    }
    {   // target == 0: 全部つぶしてもよい（ラダーの打ち切りで止まる）
        std::vector<float> v; std::vector<int32_t> f;
        make_torus(40, v, f);
        run_case("target-zero", v, f, 0, /*closed=*/false, true, /*check_topo=*/false);
    }
    {   // 三角形 1 枚（開いたメッシュ、E=3、collapse すると面が消える）
        std::vector<float> v = {0,0,0, 1,0,0, 0,1,0}; std::vector<int32_t> f = {0,1,2};
        run_case("single-triangle", v, f, 0, false, true, /*check_topo=*/false);
    }
    {   // V>0, F=0
        std::vector<float> v = {0,0,0, 1,0,0, 0,1,0}; std::vector<int32_t> f;
        run_case("no-faces", v, f, 0, false);
    }
    {   // 非多様体辺: 辺 (0,1) を 3 面で共有（boundary ではなく nonmanifold として扱う）
        std::vector<float> v = {0,0,0, 1,0,0, 0,1,0, 0,-1,0, 0,0,1};
        std::vector<int32_t> f = {0,1,2, 0,3,1, 0,1,4};
        Stats si = analyze(v, f);
        check(si.nonmanifold == 1 && si.open == 6, "fixture: 1 nonmanifold edge, 6 open edges");
        run_case("nonmanifold-fan", v, f, 1, false, true, /*check_topo=*/false);
    }
    {   // 高次数 fan（1 頂点に 200 面）: 局所ソートの std::sort フォールバック（n > 32）を通す
        const int n = 200;
        std::vector<float> v = {0,0,0}; std::vector<int32_t> f;
        for (int i = 0; i < n; ++i) { float a = 6.2831853f * i / n; v.push_back(std::cos(a)); v.push_back(std::sin(a)); v.push_back(0.1f * (i % 3)); }
        for (int i = 0; i < n; ++i) { f.push_back(0); f.push_back(1 + i); f.push_back(1 + (i + 1) % n); }
        run_case("high-degree-fan", v, f, n / 4, false);
    }
    {   // 同じ面を 2 回持つ入力 + 孤立頂点（legacy と同じ扱い: 重複面は非多様体辺として残る）
        std::vector<float> v; std::vector<int32_t> f;
        make_torus(40, v, f);
        for (int k = 0; k < 9; ++k) f.push_back(f[k]);           // 最初の 3 面を複製
        v.push_back(5); v.push_back(5); v.push_back(5);            // 孤立頂点
        check(analyze(v, f).nonmanifold >= 3, "fixture: duplicated faces make edges with count >= 3");
        run_case("dup-faces-isolated-vertex", v, f, (int)(f.size()/3) / 4, false);
    }
    {   // 非多様体辺が多く E >= 4096 の入力（並列 collapse が非多様体辺を跨いで走る）: torus R=60 の
        // 8 面に 1 面を複製 → 面数 3 以上の辺が数千本
        std::vector<float> v; std::vector<int32_t> f;
        make_torus(60, v, f);
        const int F0 = (int)f.size()/3;
        for (int t = 0; t < F0; t += 8) for (int k = 0; k < 3; ++k) f.push_back(f[3*t+k]);
        Stats si = analyze(v, f);
        check(si.nonmanifold >= 1000, "fixture: >= 1000 nonmanifold edges");
        run_case("nonmanifold-parallel", v, f, F0 / 4, false, true, /*check_topo=*/false);
    }
    {   // 目標以下でも不正面は入口で落とされる（pass-through は「位置そのまま + 検査済みの面」）
        std::vector<float> v = {0,0,0, 1,0,0, 0,1,0}; std::vector<int32_t> f = {0,0,99, 0,1,2};
        std::vector<float> ov; std::vector<int32_t> of;
        std::printf("[below-target-invalid] F=2 (1 invalid) target=2\n");
        trellis::decimate_qem_cpu(v, 3, f, 2, 2, ov, of);
        check(ov == v && of == std::vector<int32_t>{0,1,2}, "pass-through keeps positions and drops the invalid face");
        trellis::decimate_qem(v, 3, f, 2, 2, ov, of);
        check(ov == v && of == std::vector<int32_t>{0,1,2}, "decimate_qem entry does the same");
    }
    {   // 縮退面 (0,0,1) / (0,0,0) と範囲外 index: 入口で落とされ、残りは通常どおり減る
        std::vector<float> v; std::vector<int32_t> f;
        make_torus(40, v, f);
        const int F0 = (int)f.size()/3;
        f.push_back(0); f.push_back(0); f.push_back(1);
        f.push_back(0); f.push_back(0); f.push_back(0);
        f.push_back(0); f.push_back(1); f.push_back(999999);
        std::vector<float> ov; std::vector<int32_t> of;
        std::printf("[degenerate-input] F=%d (+3 invalid)\n", (int)f.size()/3);
        trellis::decimate_qem_cpu(v, (int)v.size()/3, f, (int)f.size()/3, F0 / 4, ov, of);
        Stats s = analyze(ov, of);
        check(s.finite && s.valid_idx, "output finite with valid non-degenerate faces");
        check((int)of.size()/3 <= F0 / 4, "F <= target");
        // 同じ入力から不正面を除いたものと bit 一致する（入口の除去 = 事前にきれいな入力を渡すのと同じ）
        std::vector<int32_t> fc(f.begin(), f.begin() + 3 * F0);
        std::vector<float> ov2; std::vector<int32_t> of2;
        trellis::decimate_qem_cpu(v, (int)v.size()/3, fc, F0, F0 / 4, ov2, of2);
        check(ov == ov2 && of == of2, "identical to running on the pre-cleaned input");
    }
    thread_determinism(argv[0], std::max(60, R / 3));
    std::printf(g_fail ? "FAIL: %d check(s) failed\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
