// Texture-1024 段（S=1024, R=64, NAF target=1024）の conditioning を実入力で回し、
// メモリと数値を測るテスト。docs/spec/30-pixal3d-cond.md 節2の tex_1024 行が対象。
//
//   trellis-test-pixal3d-cond-tex <dinov3.gguf> <pixal3d_naf.gguf> <views_dir> [gpu]
//        [--naf-t T] [--S S] [--R R] [--views V] [--stride K] [--coords f.npy]
//        [--chunk B] [--host] [--dense] [--save-prefix P]
//        [--mode gpu|host|both] [--profile-cond] [--naf-ops] [--naf-direct-conv] [--naf-generic] [--naf-generic-gn] [--naf-compare]
//
// views_dir は transforms.json + pre-matted RGBA frame（src/pixal3d_input.cpp）。
// 既定は sparse coords 経路（--stride K で R^3 を K おきに間引いた合成 coords、
// --coords で hr_coords.npy を渡せば実 token 集合）。--dense は従来の dense R^3 経路
// （T=1024 では ~4GiB を踏むので、失敗の再現用）。--host は CPU 参照 pixal3d_cond_slat
// と突き合わせる（S=1024/T=1024 では host 側も数 GiB 使う）。
//
// 計測用（docs/design/2026-09-20-conditioning-profiler.md §5）:
//   --mode gpu|host|both  片方だけを別プロセスで回す（A/B の順序効果を避ける。既定 both =
//                         従来どおり GPU 版 → host 版、--host 指定時のみ host 版も走る）
//   --profile-cond        [cond-v] の sub-stage ラップを印字（g_profile_cond。printf のみ）
//   --naf-ops             診断: view 0 で DINOv3 → naf_upsample_ggml を 1 回回し、NAF グラフの
//                         op 別テーブル（node 単位再実行、ISOLATED）を印字して終了。A/B とは
//                         別プロセスで使う。--naf-direct-conv で encoder を im2col でなく
//                         ggml_conv_2d_direct にして同じ表を出す（Metal での A/B 用）。
//                         --naf-generic = generic_lowering（GroupNorm + reflect pad + pool の再表現）、
//                         --naf-generic-gn = GroupNorm だけ再表現。--naf-compare を足すと同じ入力で
//                         native をもう 1 回回して出力差（max|d| / L2rel）を印字する
#include "pixal3d_cond.h"
#include "pixal3d_input.h"
#include "trellis_model.h"
#include "trellis_args.h"
#include "dinov3.h"
#include "naf.h"
#include "npy.h"
#include "ggml-backend.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::string; using std::vector;
using namespace trellis;

static void print_stats(const char* tag, const Pixal3dCondStats& st) {
    printf("  [%s] V=%d weights=%.1f MB cond=%.1f MB view_alloc=%.1f MB peak=%.1f MB"
           "  total=%.0f ms  slowest view=%.0f ms\n",
           tag, st.views, st.weight_bytes / 1048576.0, st.cond_bytes / 1048576.0,
           st.view_alloc_bytes / 1048576.0, st.peak_bytes / 1048576.0, st.total_ms, st.view_ms_max);
}

// 2つの [N,C] を比較して max abs / mean abs / L2 relative / cosine を出す。
static bool compare(const char* name, const vector<float>& a, const vector<float>& b, double tol) {
    if (a.size() != b.size()) { printf("  %-22s SIZE MISMATCH %zu vs %zu\n", name, a.size(), b.size()); return false; }
    double maxabs = 0, sumabs = 0, dn = 0, an = 0, bn = 0, dot = 0, refmax = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = a[i], y = b[i], d = std::fabs(x - y);
        maxabs = std::max(maxabs, d); sumabs += d; dn += d * d; an += x * x; bn += y * y; dot += x * y;
        refmax = std::max(refmax, std::fabs(y));
    }
    const double l2rel = bn > 0 ? std::sqrt(dn / bn) : std::sqrt(dn);
    const double cosv = (an > 0 && bn > 0) ? dot / (std::sqrt(an) * std::sqrt(bn)) : 0.0;
    const bool ok = l2rel < tol;
    printf("  %-22s max|d|=%.4e mean|d|=%.4e L2rel=%.4e cos=%.8f refmax=%.3e  %s\n",
           name, maxabs, sumabs / a.size(), l2rel, cosv, refmax, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    vector<string> pos; int gpu = 0;
    int S = 1024, R = 64, Tn = 1024, nviews = 0, stride = 8, chunk = 0;
    bool do_host = false, dense = false, naf_ops = false, naf_direct = false, naf_generic = false, naf_generic_gn = false, naf_compare = false;
    string coords_npy, save_prefix, mode = "both";
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--mode" && i + 1 < argc) { mode = argv[++i];
            if (mode != "gpu" && mode != "host" && mode != "both") { fprintf(stderr, "--mode: expected gpu|host|both, got '%s'\n", mode.c_str()); return 2; } }
        else if (a == "--profile-cond") g_profile_cond = true;
        else if (a == "--naf-ops") naf_ops = true;
        else if (a == "--naf-direct-conv") naf_direct = true;
        else if (a == "--naf-generic") naf_generic = true;          // generic_lowering（GroupNorm + pad + pool の再表現）
        else if (a == "--naf-generic-gn") naf_generic_gn = true;    // GroupNorm だけ再表現（pad / pool は native）
        else if (a == "--naf-compare") naf_compare = true;          // native の出力と比較（max|d| / L2rel）
        else if (a == "--naf-t" && i + 1 < argc) Tn = atoi(argv[++i]);
        else if (a == "--S" && i + 1 < argc) S = atoi(argv[++i]);
        else if (a == "--R" && i + 1 < argc) R = atoi(argv[++i]);
        else if (a == "--views" && i + 1 < argc) nviews = atoi(argv[++i]);
        else if (a == "--stride" && i + 1 < argc) stride = atoi(argv[++i]);
        else if (a == "--chunk" && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (a == "--coords" && i + 1 < argc) coords_npy = argv[++i];
        else if (a == "--save-prefix" && i + 1 < argc) save_prefix = argv[++i];
        else if (a == "--host") do_host = true;
        else if (a == "--dense") dense = true;
        else pos.push_back(a);
    }
    if (pos.size() < 3) {
        fprintf(stderr, "usage: %s <dinov3.gguf> <pixal3d_naf.gguf> <views_dir> [gpu]"
                        " [--naf-t T] [--S S] [--R R] [--views V] [--stride K] [--coords f.npy]"
                        " [--chunk B] [--host] [--dense] [--save-prefix P]"
                        " [--mode gpu|host|both] [--profile-cond] [--naf-ops] [--naf-direct-conv] [--naf-generic] [--naf-generic-gn] [--naf-compare]\n", argv[0]);
        return 2;
    }
    if (pos.size() >= 4) gpu = atoi(pos[3].c_str());
    if (mode == "host") do_host = true;
    setvbuf(stdout, nullptr, _IONBF, 0);   // 計測行をその場で出す（ログへリダイレクトしても順序が崩れない）

    Pixal3dInputViews in; string err;
    if (!pixal3d_load_input_views(pos[2], in, err, nviews)) { fprintf(stderr, "input: %s\n", err.c_str()); return 2; }
    const vector<Pixal3dView>& views = (S == 1024) ? in.views1024 : in.views512;
    printf("views: %zu  S=%d R=%d naf_T=%d mesh_scale=%.4f  chunk=%d\n",
           views.size(), S, R, Tn, in.mesh_scale, chunk);

    // coords: --coords の npy（[N,3] か [N,4]）か、R^3 を stride で間引いた合成集合。
    vector<std::array<int,3>> coords;
    if (!coords_npy.empty()) {
        npy::Array a = npy::load(coords_npy);
        const int64_t N = a.shape[0], cw = a.shape.size() > 1 ? a.shape[1] : 3;
        coords.resize((size_t)N);
        for (int64_t n = 0; n < N; ++n)
            coords[n] = {(int)std::lround(a.data[n*cw + cw-3]), (int)std::lround(a.data[n*cw + cw-2]),
                         (int)std::lround(a.data[n*cw + cw-1])};
    } else {
        for (int x = 0; x < R; x += stride) for (int y = 0; y < R; y += stride) for (int z = 0; z < R; ++z)
            coords.push_back({x, y, z});
    }
    printf("coords: %s N=%zu (gather 対象 %zu)\n", dense ? "dense R^3" : "sparse",
           dense ? (size_t)R*R*R : coords.size(), coords.size());

    Model dinov3 = Model::load(pos[0], gpu);
    Model naf = Model::load(pos[1], gpu);
    printf("backend: %s\n", dinov3.backend ? ggml_backend_name(dinov3.backend) : "(none)");

    Pixal3dSlatCondParams prm{S, R, Tn, in.mesh_scale};
    prm.naf_block_chunk = chunk;

    if (naf_ops) {
        // 診断: view 0 の DINOv3 patch map を作り、naf_upsample_ggml を profile_ops 付きで 1 回。
        // 本番経路（naf_upsample の自動ディスパッチ）は profile_ops を立てないので、この表は
        // ここでしか出ない。数値は使わず、op 表だけが成果物。
        const int Hp = S / 16, Wp = Hp;
        if (!naf_ggml_available(naf, S, Tn, Hp, Wp)) { fprintf(stderr, "--naf-ops: naf_upsample_ggml is not available on this backend/config\n"); return 2; }
        vector<float> normed = pixal3d_imagenet_normalize(views[0].rgb_premult, S);
        vector<float> tok = dinov3_encode(dinov3, normed, S);              // [Ntok, 1024] token-major
        vector<float> chw((size_t)1024 * Hp * Wp);                          // [1024, Hp, Wp]
        for (int h = 0; h < Hp; ++h) for (int w = 0; w < Wp; ++w) {
            const float* src = tok.data() + (size_t)(5 + h * Wp + w) * 1024;
            for (int ch = 0; ch < 1024; ++ch) chw[(size_t)ch * Hp * Wp + (size_t)h * Wp + w] = src[ch];
        }
        NafGgmlOpts o = naf_ggml_opts_for(naf);
        o.profile_ops = true;
        if (naf_direct) o.direct_conv = true;
        if (naf_generic) o.generic_lowering = true;
        if (naf_generic_gn) o.generic_groupnorm = true;
        NafGgmlStats ns;
        const auto t0 = std::chrono::steady_clock::now();
        vector<float> hr = naf_upsample_ggml(naf, views[0].rgb_premult.data(), S, chw.data(), 1024, Hp, Wp, Tn, nullptr, &ns, &o);
        printf("naf-ops: S=%d T=%d direct_conv=%d generic=%d generic_gn=%d nodes=%d activations=%.1f MB output=%.1f MB compute=%.0f ms total(with op pass)=%.1f s\n",
               S, Tn, (int)o.direct_conv, (int)o.generic_lowering, (int)o.generic_groupnorm, ns.n_nodes, ns.alloc_bytes / 1048576.0,
               ns.output_bytes / 1048576.0, ns.compute_ms, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        size_t bad = 0; for (float v : hr) if (!std::isfinite(v)) bad++;
        printf("naf-ops: output finite check: %zu non-finite of %zu\n", bad, hr.size());
        if (naf_compare && (o.direct_conv || o.generic_lowering || o.generic_groupnorm)) {
            // 同じ入力で native（im2col、GroupNorm / pad / pool は backend の op）を回して出力を比較する。
            // 同一プロセス・同一入力なので、差は lowering / conv 経路の丸め順の違いだけ。
            NafGgmlOpts on = naf_ggml_opts_for(naf);
            on.direct_conv = false; on.generic_lowering = false; on.generic_groupnorm = false;
            NafGgmlStats nn;
            vector<float> ref = naf_upsample_ggml(naf, views[0].rgb_premult.data(), S, chw.data(), 1024, Hp, Wp, Tn, nullptr, &nn, &on);
            double num = 0, den = 0, maxd = 0, sumd = 0; size_t worst = 0;
            for (size_t i = 0; i < hr.size(); ++i) {
                const double d = (double)hr[i] - (double)ref[i];
                num += d * d; den += (double)ref[i] * (double)ref[i]; sumd += std::fabs(d);
                if (std::fabs(d) > maxd) { maxd = std::fabs(d); worst = i; }
            }
            double refmax = 0; for (float v : ref) refmax = std::max(refmax, (double)std::fabs(v));
            printf("naf-ops: compare vs native (im2col, native GroupNorm/pad/pool): max|d|=%.4e (at %zu: %.6f vs %.6f) mean|d|=%.4e L2rel=%.4e refmax=%.4e  native compute=%.0f ms\n",
                   maxd, worst, hr[worst], ref[worst], sumd / hr.size(), std::sqrt(num / std::max(den, 1e-30)), refmax, nn.compute_ms);
        }
        dinov3.free(); naf.free();
        return bad ? 1 : 0;
    }

    Pixal3dCond c; Pixal3dCondStats st;
    vector<float> gpu_proj;
    if (mode != "host") {
        const auto t0 = std::chrono::steady_clock::now();
        c = pixal3d_cond_slat_gpu(dinov3, naf, views, prm, &st, dense ? nullptr : &coords);
        printf("gpu path: global=%zu proj=%zu (%.1f MB) d_proj=%d  %.1f s\n", c.global.size(), c.proj.size(),
               c.proj.size() * 4 / 1048576.0, c.d_proj,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        print_stats("gpu", st);
        gpu_proj = dense ? pixal3d_gather_proj(c.proj, R, c.d_proj, coords) : c.proj;
        if (!save_prefix.empty()) {
            npy::save(save_prefix + "_global.npy", c.global.data(), {5, 1024});
            npy::save(save_prefix + "_proj.npy", gpu_proj.data(),
                      {(int64_t)(gpu_proj.size() / (size_t)c.d_proj), (int64_t)c.d_proj});
            printf("saved %s_{global,proj}.npy\n", save_prefix.c_str());
        }
    }

    bool ok = true;
    if (do_host) {
        const auto t1 = std::chrono::steady_clock::now();
        Pixal3dCond h = pixal3d_cond_slat(dinov3, naf, views, prm);
        printf("host path: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
        vector<float> hg = pixal3d_gather_proj(h.proj, R, h.d_proj, coords);
        if (mode == "host" && !save_prefix.empty()) {
            npy::save(save_prefix + "_host_global.npy", h.global.data(), {5, 1024});
            npy::save(save_prefix + "_host_proj.npy", hg.data(), {(int64_t)(hg.size() / (size_t)h.d_proj), (int64_t)h.d_proj});
            printf("saved %s_host_{global,proj}.npy\n", save_prefix.c_str());
        }
        if (mode != "host") {
            ok &= compare("global", c.global, h.global, 5e-3);
            ok &= compare("proj[lr||hr]", gpu_proj, hg, 5e-3);
        }
    }
    dinov3.free(); naf.free();
    printf("%s\n", ok ? "COND_TEX: PASS" : "COND_TEX: FAIL");
    return ok ? 0 : 1;
}
