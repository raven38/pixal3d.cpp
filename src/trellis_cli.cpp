// trellis-cli: image -> 3D mesh (.glb), the TRELLIS.2 geometry pipeline in GGML.
//   trellis-cli <image.png> <out.glb> [gpu] [models_dir] [seed]
// Models are loaded/freed per stage to keep VRAM modest.
#include "trellis_model.h"
#include "preprocess.h"
#include "dinov3.h"
#include "flow_runner.h"
#include "ss_decoder.h"
#include "shape_decoder.h"
#include "dual_grid.h"
#include "mesh_glb.h"
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "stb_image_write.h"
#include "trellis_run.h"
#include "pixal3d_cond.h"
#include "proj_grid.h"   // mat4_inverse_d: 入力段で transform_matrix の可逆性を確認する
#include "transforms_json.h"
#include <filesystem>
// Declarations only (no *_IMPLEMENTATION define here) -- stbi_load/stbir_resize_uint8 are
// implemented once in preprocess.cpp, part of trellis_core, which this binary links against.
#include "stb_image.h"
#include "stb_image_resize.h"

#include <cstdio>
#include <random>
#include <vector>
#include <string>
#include <chrono>
#include <set>
#include <array>
#include <cmath>
#include <cstring>

using std::vector;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// [dbg] overall stats of a flat tensor — used to compare LR vs HR shape-SLAT in decode space.
static void slat_stats(const char* tag, const vector<float>& v) {
    if (v.empty()) { printf("      [stats] %s EMPTY\n", tag); return; }
    double s = 0, s2 = 0; float mn = 0, mx = 0; bool first = true; size_t bad = 0;
    for (float x : v) {
        if (std::isnan(x) || std::isinf(x)) { bad++; continue; }
        s += x; s2 += (double)x * x;
        if (first) { mn = mx = x; first = false; } else { if (x < mn) mn = x; if (x > mx) mx = x; }
    }
    size_t n = v.size() - bad; double mean = n ? s / n : 0, var = n ? s2 / n - mean * mean : 0;
    printf("      [stats] %s n=%zu mean=%.4f std=%.4f min=%.3f max=%.3f nan/inf=%zu\n",
           tag, v.size(), mean, var > 0 ? std::sqrt(var) : 0.0, mn, mx, bad);
}

static const float SHAPE_MEAN[32]={0.781296f,0.018091f,-0.495192f,-0.558457f,1.060530f,0.093252f,1.518149f,-0.933218f,-0.732996f,2.604095f,-0.118341f,-2.143904f,0.495076f,-2.179512f,-2.130751f,-0.996944f,0.261421f,-2.217463f,1.260067f,-0.150213f,3.790713f,1.481266f,-1.046058f,-1.523667f,-0.059621f,2.220780f,1.621212f,0.877230f,0.567247f,-3.175944f,-3.186688f,1.578665f};
static const float SHAPE_STD[32]={5.972266f,4.706852f,5.445010f,5.209927f,5.320220f,4.547237f,5.020802f,5.444004f,5.226681f,5.683095f,4.831436f,5.286469f,5.652043f,5.367606f,5.525084f,4.730578f,4.805265f,5.124013f,5.530808f,5.619001f,5.103930f,5.417670f,5.269677f,5.547194f,5.634698f,5.235274f,6.110351f,5.511298f,6.237273f,4.879207f,5.347008f,5.405691f};
static const float TEX_MEAN[32]={3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,-1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293000f,-1.925608f,-1.217717f,1.213905f,0.971588f,-0.023631f,0.106750f,2.021786f,0.250524f,-0.662387f,-0.768862f};
static const float TEX_STD[32]={2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,2.295916f,3.234985f,3.233086f,2.260140f,2.874801f,2.810596f,3.292720f,2.674999f,2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.775190f};
// Verified equal (2026-09-06) to pipeline_mv.json's shape_slat_normalization / tex_slat_normalization
// (task spec's quoted mean/std prefixes match SHAPE_MEAN/SHAPE_STD/TEX_MEAN/TEX_STD above byte-for-byte)
// -- Pixal3D reuses the same TRELLIS.2 SLAT normalization tables, so no separate table is needed here.

// ===========================================================================================
// Pixal3D multiview mode (--views DIR): V posed RGBA views -> GLB, via the same TRELLIS.2
// postprocess tail (decode/remesh/decimate/UV/bake/GLB) but with ProjectAttention-conditioned
// flow sampling (docs/spec/30-pixal3d-cond.md) instead of single-image DINOv3 cross-attention.
// A separate function (not a branch inside trellis_run) so the existing TRELLIS.2 code above
// is untouched byte-for-byte; the shared postprocess calls below are the SAME trellis:: API
// calls TRELLIS.2 uses (dual_grid.cpp/remesh_dc.cpp/decimate_qem.cpp/uv_bake.cpp/mesh_glb.cpp
// are not modified for this feature).
// ===========================================================================================

// Decodes one view file to RGBA8 host memory (no resize). has_alpha follows preprocess.cpp's
// image_has_alpha convention: a real (not all-255) alpha channel, since this mode does no
// matting -- a flat/opaque alpha means the caller should error out, not silently proceed.
static std::vector<unsigned char> mv_load_rgba(const std::string& path, int& W, int& H, bool& has_alpha) {
    int ch; unsigned char* img = stbi_load(path.c_str(), &W, &H, &ch, 4);
    if (!img) { has_alpha = false; return {}; }
    has_alpha = false;
    for (size_t i = 0; i < (size_t)W * H; ++i) if (img[4*i+3] < 250) { has_alpha = true; break; }
    std::vector<unsigned char> out(img, img + (size_t)W * H * 4);
    stbi_image_free(img);
    return out;
}

// inference_mv.to_cond_tensor: resize to SxS then premultiply RGB by alpha -> [3,S,S] in [0,1],
// NOT ImageNet-normalized (pixal3d_cond_ss/slat normalize internally per view). The reference
// uses PIL LANCZOS; this repo's resize helper (preprocess.cpp's normalize_cutout) is stb_image_resize's
// stbir_resize_uint8, whose default filter is NOT Lanczos (roughly Catmull-Rom-like upsampling /
// a box-ish downsampling per stb_image_resize.h) -- noted here as a known, unverified source of
// small pixel-level divergence from the PyTorch reference at the resize step.
static std::vector<float> mv_resize_premult(const std::vector<unsigned char>& rgba, int W, int H, int S) {
    std::vector<unsigned char> rs((size_t)S * S * 4);
    stbir_resize_uint8(rgba.data(), W, H, 0, rs.data(), S, S, 0, 4);
    std::vector<float> out((size_t)3 * S * S);
    for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
        const size_t si = ((size_t)y * S + x) * 4;
        const float a = rs[si + 3] / 255.0f;
        for (int c = 0; c < 3; ++c) out[(size_t)c * S * S + (size_t)y * S + x] = (rs[si + c] / 255.0f) * a;
    }
    return out;
}

// Loads transforms.json + every frame's view at both 512 and 1024 (each file decoded once,
// then resized twice -- mirrors inference_mv.load_views without the rembg fallback: every
// view here must already carry a real alpha channel). Returns false (stderr message already
// printed) on any error: missing/malformed transforms.json, missing camera_angle_x, a view
// with no real alpha channel, or an unreadable image file.
static bool mv_load_views(const trellis::TrellisParams& cfg,
                           std::vector<trellis::Pixal3dView>& v512,
                           std::vector<trellis::Pixal3dView>& v1024,
                           float& mesh_scale) {
    trellis::TransformsFile tf;
    std::string err;
    if (!trellis::load_views_metadata(cfg.views, cfg.mesh_scale, cfg.mesh_scale_set, tf, err)) {
        fprintf(stderr, "[trellis] %s\n", err.c_str());
        return false;
    }
    const bool synthesized = tf.frames.size() == (size_t)trellis::CANONICAL_RIG_VIEWS &&
                             !std::filesystem::exists(cfg.views + "/transforms.json");
    if (synthesized) {
        printf("      no transforms.json: canonical turntable rig (front/right/back/left), "
               "mesh_scale=%.4f\n", (double)tf.mesh_scale);
        for (size_t i = 0; i < tf.frames.size(); ++i) printf("        %zu %s\n", i + 1, tf.frames[i].file_path.c_str());
        // 合成モードで視点数を削ると rig の前提（4 視点）が崩れるので、4 以外は拒否する。
        if (cfg.num_views_set && cfg.num_views != trellis::CANONICAL_RIG_VIEWS) {
            fprintf(stderr, "[trellis] --num-views %d is not usable without transforms.json; the canonical rig needs all %d views\n",
                    cfg.num_views, trellis::CANONICAL_RIG_VIEWS);
            return false;
        }
    }

    const int nframes = (int)tf.frames.size();
    const int use_n = (cfg.num_views > 0) ? std::min(cfg.num_views, nframes) : nframes;
    mesh_scale = tf.mesh_scale;
    v512.clear(); v1024.clear();
    v512.reserve(use_n); v1024.reserve(use_n);

    for (int i = 0; i < use_n; ++i) {
        const trellis::TransformsFrame& fr = tf.frames[i];
        float fov = 0.0f;
        if (fr.has_camera_angle_x) fov = fr.camera_angle_x;
        else if (tf.has_camera_angle_x) fov = tf.camera_angle_x;
        else { fprintf(stderr, "[trellis] view %d (%s): no camera_angle_x (per-frame or top-level)\n", i, fr.file_path.c_str()); return false; }
        // FOV と姿勢は、壊れていても下流が黙って進んでしまう（camera_angle_x=0 だと
        // tan(fov/2)=0 で focal が無限大、特異な transform_matrix だと proj_grid.cpp:127/:206 の
        // mat4_inverse_d() の失敗が無視される）。V=1 では他視点による平均緩和も無いので、
        // 壊れた 1 枚がそのまま条件になる。ここで落とす。
        if (!std::isfinite(fov) || fov <= 0.0f) {
            fprintf(stderr, "[trellis] view %d (%s): camera_angle_x must be finite and > 0, got %g\n",
                    i, fr.file_path.c_str(), (double)fov);
            return false;
        }
        {
            double c2w_d[16], inv[16];
            for (int k = 0; k < 16; ++k) {
                if (!std::isfinite(fr.transform_matrix[k])) {
                    fprintf(stderr, "[trellis] view %d (%s): transform_matrix has a non-finite entry\n",
                            i, fr.file_path.c_str());
                    return false;
                }
                c2w_d[k] = (double)fr.transform_matrix[k];
            }
            if (!trellis::mat4_inverse_d(c2w_d, inv)) {
                fprintf(stderr, "[trellis] view %d (%s): transform_matrix is singular (not invertible)\n",
                        i, fr.file_path.c_str());
                return false;
            }
            const double d2 = c2w_d[3]*c2w_d[3] + c2w_d[7]*c2w_d[7] + c2w_d[11]*c2w_d[11];
            if (!(d2 > 0.0)) {
                fprintf(stderr, "[trellis] view %d (%s): camera sits at the origin (distance 0)\n",
                        i, fr.file_path.c_str());
                return false;
            }
        }

        const std::string path = cfg.views + "/" + fr.file_path;
        int W, H; bool has_alpha;
        std::vector<unsigned char> rgba = mv_load_rgba(path, W, H, has_alpha);
        if (rgba.empty()) { fprintf(stderr, "[trellis] view %d: cannot load %s\n", i, path.c_str()); return false; }
        if (!has_alpha) {
            fprintf(stderr, "[trellis] view %d (%s): no real alpha channel -- --views mode does not matte"
                            " (pre-matted RGBA views only)\n", i, path.c_str());
            return false;
        }

        trellis::Pixal3dView a{}, b{};
        a.rgb_premult = mv_resize_premult(rgba, W, H, 512);  a.fov_x = fov;
        b.rgb_premult = mv_resize_premult(rgba, W, H, 1024); b.fov_x = fov;
        std::memcpy(a.c2w, fr.transform_matrix, 16 * sizeof(float));
        std::memcpy(b.c2w, fr.transform_matrix, 16 * sizeof(float));
        v512.push_back(std::move(a));
        v1024.push_back(std::move(b));
    }
    return true;
}

// Applies the same net final-frame rotation inference_mv.py's `rot = [[-1,0,0],[0,0,-1],
// [0,-1,0]]` (glb.apply_transform(rot), applied once to the ALREADY-EXPORTED reference GLB)
// produces, but composed for THIS codebase's export path. mesh_glb.cpp's own internal
// Zup->glTF-Yup conversion ((x,y,z)->(x,z,-y), applied inside write_glb/write_glb_textured,
// unchanged/untouched here) still runs on top of whatever we pass it, so this is the
// pre-transform P on the SAME Zup vertex arrays TRELLIS.2 always passes to those writers such
// that mesh_glb(P(v)) == rot(mesh_glb(v)): P(x,y,z) = (-x, z, y). Solved by composing
// M(x,y,z)=(x,z,-y) with rot and verified 2026-09-06 against the reference GLB's measured
// bbox (tools/glb_metrics.py): before this fix, ours (x,y,z) extents (0.758,0.989,0.848) vs
// reference (0.757,0.845,0.987) -- matching up to a y<->z swap, which P corrects (both
// P and mesh_glb's own conversion have det=+1, so winding/normals need no extra flip).
// Applied only in --views mode, at the very end (after baking, so it can never desync the
// UV-baked texture from a voxel-index-space PBR sample -- see mv_gather_proj's coordinate
// contract, untouched here).
static void mv_apply_reference_frame(std::vector<float>& verts) {
    for (size_t i = 0; i + 2 < verts.size(); i += 3) {
        const float x = verts[i], y = verts[i + 1], z = verts[i + 2];
        verts[i] = -x; verts[i + 1] = z; verts[i + 2] = y;
    }
}

// One sparse Pixal3D ProjectAttention shape-SLAT flow sample (shape 512 or shape HR): loads
// the checkpoint, verifies it is a Pixal3D checkpoint (detected proj_in matches the fused
// [lr||hr] 2048-wide condition), samples, frees the model. Empty return = error (already
// logged) -- never a legitimate zero-length result since coords is always non-empty here.
static std::vector<float> mv_shape_flow(const std::string& path, const trellis::TrellisParams& cfg,
                                        bool F32, int gpu,
                                        const std::vector<std::array<int,3>>& coords,
                                        const float* cnd, const float* ncnd, int lc,
                                        const float* proj, const float* nproj,
                                        std::vector<float>& noise_buf) {
    const int n = (int)coords.size();
    trellis::Model m = trellis::Model::load(path, gpu);
    trellis::DiTParams p; p.in_ch = 32; p.out_ch = 32; p.d_cond = 1024; p.cast_f32 = F32;
    if (!trellis::dit_detect_proj_attn(m, p)) {
        fprintf(stderr, "[trellis] %s: not a Pixal3D ProjectAttention checkpoint (missing"
                        " blocks.0.cross_attn.proj_linear.weight)\n", path.c_str());
        m.free(); return {};
    }
    if (p.d_proj != 2048) {
        fprintf(stderr, "[trellis] %s: proj_in=%d, expected 2048 ([lr||hr] fused condition)\n", path.c_str(), p.d_proj);
        m.free(); return {};
    }
    trellis::DitRunner* run = trellis::make_sparse_runner(m, p, coords, lc);
    trellis::FlowFwdProj fwd = [&](const std::vector<float>& x, float ts, const float* c, const float* pj) {
        return run->forward(x, ts, c, pj);
    };
    trellis::SamplerParams sp; sp.steps = 12; sp.guidance_strength = cfg.gsh; sp.guidance_rescale = 0.5f;
    sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 3.0f;
    std::vector<float> out = trellis::sample_flow(fwd, noise_buf, cnd, ncnd, proj, nproj, sp);
    delete run; m.free();
    return out;
}

// Pixal3D flow 重みの系列（--pixal3d-weights sv|mv）を 4 つのファイル名へ落とす。
// 公式は 4 段それぞれに単視点版（接尾辞なし）と多視点版（_mv）を配布しており、config json と
// テンソル名/shape/dtype は一致する（2026-09-15 実測）。共有の 5 モデル（dinov3 / naf /
// ss_dec / shape_dec / tex_dec）は系列に依存しないので名前を変えない。
struct Pixal3dWeights {
    std::string ss, shape512, shape1024, tex;
};

static Pixal3dWeights mv_weight_paths(const std::string& models, const std::string& variant) {
    const std::string sfx = (variant == "sv") ? "_sv" : "_mv";
    return { models + "/pixal3d_ss_flow"        + sfx + ".gguf",
             models + "/pixal3d_shape_flow_512" + sfx + ".gguf",
             models + "/pixal3d_shape_flow_1024"+ sfx + ".gguf",
             models + "/pixal3d_tex_flow_1024"  + sfx + ".gguf" };
}

// 推論を始める前に、選んだ系列で実際に要るファイルが揃っているか確認する。
// 段階ロードのままだと texture flow だけ欠けていたときに 3 段走ってから落ち、しかも
// Model::load() の例外を main が捕まえないので診断が出ない。
static bool mv_check_weights(const Pixal3dWeights& w, const std::string& variant, bool want_tex) {
    std::vector<const std::string*> need = { &w.ss, &w.shape512, &w.shape1024 };
    if (want_tex) need.push_back(&w.tex);
    std::vector<std::string> missing;
    for (const std::string* p : need) {
        FILE* f = fopen(p->c_str(), "rb");
        if (f) fclose(f); else missing.push_back(*p);
    }
    if (missing.empty()) return true;
    fprintf(stderr, "[trellis] --pixal3d-weights %s: missing %zu of %zu flow model(s):\n",
            variant.c_str(), missing.size(), need.size());
    for (const std::string& m : missing) fprintf(stderr, "            %s\n", m.c_str());
    fprintf(stderr, "          (the other variant is not substituted; pass --models DIR with the "
                    "matching set)\n");
    return false;
}

// 検証済みの組み合わせだけを無警告にする。条件付けは全視点を単純平均するので、系列が想定する
// 視点数から外れると別分布の条件になる。実測（TASK-047）: カメラ・入力を共通にして視点数だけ
// 4->1 にすると多視点重みが 85 勝 0 敗（d30 85 体）/ 20 勝 0 敗（game 設定画 20 体、IoU 中央値
// +0.32）で負け、1 視点同士でも多視点重みは単視点重みに 3 勝 17 敗で負ける。
// 実行は止めない（判断はユーザーに残す）。
static void mv_warn_view_count(const std::string& variant, int V) {
    if (variant == "sv" && V != 1)
        fprintf(stderr, "[trellis] warning: --pixal3d-weights sv with V=%d. The single-view weights "
                        "are validated at V=1; extra views are averaged into the condition.\n", V);
    if (variant == "mv" && V != 4)
        fprintf(stderr, "[trellis] warning: --pixal3d-weights mv with V=%d. The multiview weights "
                        "are validated at V=4; at V=1 they break down badly (85-0 against the "
                        "single-view weights on 85 subjects).\n", V);
}

int trellis_run_mv(const trellis::TrellisParams& cfg) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    uint32_t run_seed = cfg.seed;
    if (run_seed == 0) {
        std::random_device rd;
        run_seed = (uint32_t(rd()) << 16) ^ uint32_t(rd()) ^
                   uint32_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        if (run_seed == 0) run_seed = 1;
        fprintf(stderr, "[trellis] generating model with seed %u (auto)\n", run_seed);
    } else {
        fprintf(stderr, "[trellis] generating model with seed %u\n", run_seed);
    }
    const bool F32 = cfg.f32; trellis::g_sparse_cast_f32 = F32;
    trellis::g_no_fa = cfg.no_fa;
    trellis::g_require_gpu = cfg.require_gpu;
    trellis::g_cpu_threads = cfg.threads;
    const std::string& outglb = cfg.output;
    const std::string& M = cfg.models;
    const int gpu = cfg.gpu;
    std::mt19937 rng(run_seed); std::normal_distribution<float> randn(0.f, 1.f);
    auto noise = [&](size_t n){ vector<float> v(n); for (auto& x : v) x = randn(rng); return v; };
    double t0 = now();

    if (!cfg.cascade) {
        fprintf(stderr, "[trellis] --views mode requires --res 1024 or 1536 (Pixal3D has no res-512"
                        " texture flow; the cascade is mandatory)\n");
        return 1;
    }
    const bool do_tex = cfg.texture;

    printf("[1/6] Pixal3D multiview: load %s\n", cfg.views.c_str());
    std::vector<trellis::Pixal3dView> views512, views1024;
    float mesh_scale = 1.0f;
    if (!mv_load_views(cfg, views512, views1024, mesh_scale)) return 1;
    const int V = (int)views512.size();
    printf("      V=%d views, mesh_scale=%.4f\n", V, mesh_scale);

    const Pixal3dWeights W = mv_weight_paths(M, cfg.pixal3d_weights);
    mv_warn_view_count(cfg.pixal3d_weights, V);
    if (!mv_check_weights(W, cfg.pixal3d_weights, do_tex)) return 1;
    printf("      flow weights: %s\n", cfg.pixal3d_weights.c_str());
    double t_stage = now();

    printf("[2/6] SS proj conditioning + flow\n");
    vector<std::array<int,3>> coords;
    {
        trellis::Pixal3dCond c;
        { trellis::Model dino = trellis::Model::load(M + "/dinov3.gguf", gpu);
          c = trellis::pixal3d_cond_ss(dino, views512, 512, 16, mesh_scale);
          dino.free(); }
        vector<float> neg_g(c.global.size(), 0.0f), neg_p(c.proj.size(), 0.0f);

        trellis::Model m = trellis::Model::load(W.ss, gpu);
        trellis::DiTParams p; p.in_ch = 8; p.out_ch = 8; p.d_cond = 1024; p.cast_f32 = F32;
        if (!trellis::dit_detect_proj_attn(m, p)) { fprintf(stderr, "[trellis] %s: not a Pixal3D checkpoint\n", W.ss.c_str()); return 1; }
        trellis::DitRunner* run = trellis::make_dense_runner(m, p, 16, c.n_global);
        trellis::FlowFwdProj fwd = [&](const vector<float>& x, float ts, const float* cn, const float* pj){ return run->forward(x, ts, cn, pj); };
        trellis::SamplerParams sp; sp.steps=12; sp.guidance_strength=cfg.gss; sp.guidance_rescale=0.7f; sp.gi0=0.6f; sp.gi1=1.0f; sp.rescale_t=5.0f;
        vector<float> z = trellis::sample_flow(fwd, noise(8*4096), c.global.data(), neg_g.data(), c.proj.data(), neg_p.data(), sp);
        delete run; m.free();
        vector<float> zdec(8*4096);
        for (int cc = 0; cc < 8; ++cc) for (int spx = 0; spx < 4096; ++spx) zdec[(size_t)cc*4096 + spx] = z[cc + 8*spx];
        trellis::Model d = trellis::Model::load(M + "/ss_dec.gguf", gpu);
        vector<float> logits = trellis::ss_decode(d, zdec); d.free();
        coords = trellis::ss_coords(logits, 64, 32);
    }
    if (cfg.voxply) { FILE*f=fopen("out/myvox.ply","wb"); fprintf(f,"ply\nformat binary_little_endian 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n",coords.size()); for(auto&c:coords){float pp[3]={(c[0]+0.5f)/32-0.5f,(c[1]+0.5f)/32-0.5f,(c[2]+0.5f)/32-0.5f}; fwrite(pp,4,3,f);} fclose(f); }
    printf("      active voxels @res32 = %d  (%.1fs)\n", (int)coords.size(), now() - t_stage);
    if (coords.empty()) { fprintf(stderr, "no voxels produced\n"); return 1; }
    t_stage = now();

    printf("[3/6] shape SLAT flow (LR 512 -> upsample -> HR %d cascade, max_tok=%d)\n", cfg.hr_res, cfg.max_tokens);
    vector<float> lr_norm, lr_dn;
    {
        trellis::Pixal3dCond c;
        { trellis::Model dino = trellis::Model::load(M + "/dinov3.gguf", gpu);
          trellis::Model naf  = trellis::Model::load(M + "/pixal3d_naf.gguf", gpu);
          trellis::Pixal3dSlatCondParams prm{512, 32, 512, mesh_scale};
          c = trellis::pixal3d_cond_slat(dino, naf, views512, prm);
          dino.free(); naf.free(); }
        vector<float> proj_sp = trellis::pixal3d_gather_proj(c.proj, 32, c.d_proj, coords);
        c.proj.clear(); c.proj.shrink_to_fit();
        vector<float> neg_g(c.global.size(), 0.0f), neg_p(proj_sp.size(), 0.0f);
        vector<float> nz = noise((size_t)32 * coords.size());
        lr_norm = mv_shape_flow(W.shape512, cfg, F32, gpu, coords,
                                c.global.data(), neg_g.data(), c.n_global, proj_sp.data(), neg_p.data(), nz);
        if (lr_norm.empty()) return 1;
    }
    lr_dn.resize(lr_norm.size());
    for (size_t n = 0; n < coords.size(); ++n) for (int c = 0; c < 32; ++c)
        lr_dn[(size_t)c + 32*n] = lr_norm[(size_t)c + 32*n]*SHAPE_STD[c] + SHAPE_MEAN[c];
    slat_stats("LR slat (res32, MV)", lr_dn);
    printf("      LR shape SLAT (%.1fs)\n", now() - t_stage); t_stage = now();

    vector<std::array<int,3>> hr_coords;
    { trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
      hr_coords = trellis::shape_upsample(m, lr_dn, coords); m.free(); }
    int hr_res = cfg.hr_res;
    vector<std::array<int,3>> shc;
    for (;;) {
        // Pixal3DImageTo3DPipeline.run()'s OWN inlined 1024_cascade/1536_cascade
        // quantization (verified against tools/ref_pixal3d_hr_sample.py's fixture,
        // 2026-09-06) -- NOT the TRELLIS.2 sample_shape_slat_cascade() helper's formula
        // (int() truncation, grid_res, no round) that trellis_cli.cpp's single-image
        // cascade path below still uses unchanged. Two differences: round() instead of
        // truncation, and (grid_res - 1) instead of grid_res as the scale. For
        // hr_res==1024 the token-budget check never actually fires -- the reference's
        // own break condition is `... or actual_hr_resolution == 1024`, so only
        // 1536_cascade ever backs off; 1024_cascade always quantizes once at grid 64.
        const int gi = hr_res / 16;
        const float gm1 = (float)(gi - 1);
        std::set<std::array<int,3>> q;
        for (auto& c : hr_coords)
            q.insert({ (int)std::lround((c[0]+0.5f)/512.f*gm1),
                       (int)std::lround((c[1]+0.5f)/512.f*gm1),
                       (int)std::lround((c[2]+0.5f)/512.f*gm1) });
        if ((int)q.size() < cfg.max_tokens || hr_res == 1024) {
            shc.assign(q.begin(), q.end());
            printf("      upsampled coords @res512=%d -> quantized @res%d (grid %d, Pixal3D round/grid-1 formula) = %d tokens\n",
                   (int)hr_coords.size(), hr_res, gi, (int)shc.size());
            break;
        }
        printf("      res%d (grid %d) -> %d tokens >= %d, backing off -128\n", hr_res, gi, (int)q.size(), cfg.max_tokens);
        hr_res -= 128;
    }
    const int grid = hr_res / 16;
    const int RES = hr_res;

    vector<float> slat_norm;
    {
        trellis::Pixal3dCond c;
        { trellis::Model dino = trellis::Model::load(M + "/dinov3.gguf", gpu);
          trellis::Model naf  = trellis::Model::load(M + "/pixal3d_naf.gguf", gpu);
          trellis::Pixal3dSlatCondParams prm{1024, grid, 512, mesh_scale};
          c = trellis::pixal3d_cond_slat(dino, naf, views1024, prm);
          dino.free(); naf.free(); }
        vector<float> proj_sp = trellis::pixal3d_gather_proj(c.proj, grid, c.d_proj, shc);
        c.proj.clear(); c.proj.shrink_to_fit();
        vector<float> neg_g(c.global.size(), 0.0f), neg_p(proj_sp.size(), 0.0f);
        vector<float> nz = noise((size_t)32 * shc.size());
        slat_norm = mv_shape_flow(W.shape1024, cfg, F32, gpu, shc,
                                  c.global.data(), neg_g.data(), c.n_global, proj_sp.data(), neg_p.data(), nz);
        if (slat_norm.empty()) return 1;
    }
    const int N = (int)shc.size();
    vector<float> slat_dn(slat_norm.size());
    for (int n = 0; n < N; ++n) for (int c = 0; c < 32; ++c)
        slat_dn[(size_t)c + 32*n] = slat_norm[(size_t)c + 32*n]*SHAPE_STD[c] + SHAPE_MEAN[c];
    slat_stats("HR slat (MV)", slat_dn);
    if (cfg.dump_slat) {
        FILE* f = fopen("/tmp/hr_slat.bin", "wb");
        if (f) { int n = N, res = RES; fwrite(&n,4,1,f); fwrite(&res,4,1,f);
            for (auto& c : shc) { int xyz[3] = {c[0],c[1],c[2]}; fwrite(xyz,4,3,f); }
            fwrite(slat_dn.data(),4,slat_dn.size(),f); fclose(f);
            printf("      [dump] /tmp/hr_slat.bin: N=%d res=%d feats=%zu\n", n, res, slat_dn.size()); }
    }
    printf("      HR shape SLAT (%.1fs)\n", now() - t_stage); t_stage = now();

    printf("[4/6] FlexiDualGrid shape decode -> mesh @res%d\n", RES);
    trellis::Mesh mesh;
    trellis::ShapeOut so;
    {
        trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
        so = trellis::shape_decode(m, slat_dn, shc, RES); m.free();
        printf("      decoded voxels @res%d = %d\n", so.res, (int)so.coords.size());
        mesh = trellis::dual_grid_to_mesh(so);
    }
    printf("      mesh V=%d F=%d\n", mesh.V(), mesh.F());
    { const int nh = trellis::fill_holes(mesh.verts, mesh.faces, 3e-2f);
      if (nh) printf("      filled %d small holes -> V=%d F=%d\n", nh, mesh.V(), mesh.F()); }
    printf("      shape decode (%.1fs)\n", now() - t_stage); t_stage = now();

    vector<float> colors, pbr6;
    const vector<std::array<int,3>>* pbr_coords = &so.coords;
    const int pbr_res = so.res;
    if (do_tex) {
        printf("[5/6] texture SLAT flow (HR %d, NAF@1024) + PBR decode\n", RES);
        vector<float> texlat;
        {
            trellis::Pixal3dCond c;
            { trellis::Model dino = trellis::Model::load(M + "/dinov3.gguf", gpu);
              trellis::Model naf  = trellis::Model::load(M + "/pixal3d_naf.gguf", gpu);
              trellis::Pixal3dSlatCondParams prm{1024, grid, 1024, mesh_scale};
              c = trellis::pixal3d_cond_slat(dino, naf, views1024, prm);
              dino.free(); naf.free(); }
            vector<float> proj_sp = trellis::pixal3d_gather_proj(c.proj, grid, c.d_proj, shc);
            c.proj.clear(); c.proj.shrink_to_fit();
            vector<float> neg_g(c.global.size(), 0.0f), neg_p(proj_sp.size(), 0.0f);

            trellis::Model m = trellis::Model::load(W.tex, gpu);
            trellis::DiTParams p; p.in_ch = 64; p.out_ch = 32; p.d_cond = 1024; p.cast_f32 = F32;
            if (!trellis::dit_detect_proj_attn(m, p)) { fprintf(stderr, "[trellis] %s: not a Pixal3D checkpoint\n", W.tex.c_str()); return 1; }
            trellis::DitRunner* run = trellis::make_sparse_runner(m, p, shc, c.n_global);
            // Tex flow needs the proj-aware sample_flow overload (ProjectAttention DiT).
            trellis::FlowFwdProj fwdp = [&](const vector<float>& st, float ts, const float* cn, const float* pj) {
                vector<float> x64((size_t)64 * N);
                for (int n = 0; n < N; ++n) {
                    for (int k = 0; k < 32; ++k) x64[(size_t)k + 64*n]      = st[(size_t)k + 32*n];
                    for (int k = 0; k < 32; ++k) x64[(size_t)32 + k + 64*n] = slat_norm[(size_t)k + 32*n];
                }
                return run->forward(x64, ts, cn, pj);
            };
            trellis::SamplerParams sp; sp.steps=12; sp.guidance_strength=1.0f; sp.guidance_rescale=0.0f; sp.gi0=0.6f; sp.gi1=0.9f; sp.rescale_t=3.0f;
            texlat = trellis::sample_flow(fwdp, noise((size_t)32*N), c.global.data(), neg_g.data(), proj_sp.data(), neg_p.data(), sp);
            delete run; m.free();
            for (int n = 0; n < N; ++n) for (int cc = 0; cc < 32; ++cc) texlat[(size_t)cc + 32*n] = texlat[(size_t)cc + 32*n]*TEX_STD[cc] + TEX_MEAN[cc];
        }
        {
            trellis::Model m = trellis::Model::load(M + "/tex_dec.gguf", gpu);
            vector<float> pbr = trellis::tex_decode(m, texlat, shc, so.subs); m.free();
            const int Mv = (int)pbr_coords->size();
            colors.resize((size_t)Mv * 3); pbr6.resize((size_t)Mv * 6);
            auto cl = [](float v){ return v < 0 ? 0.f : (v > 1 ? 1.f : v); };
            for (int i = 0; i < Mv; ++i) {
                for (int k = 0; k < 6; ++k) pbr6[(size_t)i*6 + k] = cl(pbr[(size_t)k + 6*i] * 0.5f + 0.5f);
                for (int k = 0; k < 3; ++k) colors[(size_t)i*3 + k] = pbr6[(size_t)i*6 + k];
            }
            printf("      PBR voxels=%d @res%d\n", Mv, pbr_res);
        }
        if (colors.size() != (size_t)mesh.V() * 3) {
            if (colors.size() < (size_t)mesh.V() * 3) colors.resize((size_t)mesh.V() * 3, 0.5f);
            else colors.clear();
        }
        printf("      texture SLAT + decode (%.1fs)\n", now() - t_stage); t_stage = now();
    }

    // ---- shared TRELLIS.2 postprocess tail (unchanged: same trellis:: calls as the
    // single-image cascade path below, just invoked from this second call site) ----
    if (!cfg.dump_post.empty()) {
        trellis::weld_vertices(mesh.verts, mesh.faces, nullptr, 1.0f / ((float)so.res * 8.0f));
        trellis::fill_small_holes(mesh.faces);
        trellis::TriBvh dbvh = trellis::TriBvh::build(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F());
        // MV mode default: remesh_band=1 (inference_mv.py's o_voxel.postprocess.to_glb(remesh_band=1)
        // default), not TRELLIS.2's res/512 auto-scale -- unless the user passes --band.
        const int dband = cfg.band > 0 ? cfg.band : 1;
        trellis::Mesh rm = trellis::remesh_narrow_band_dc(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), dbvh, so.res, dband);
        if (rm.F() > 0) { trellis::clean_mesh(rm.V(), rm.faces); trellis::drop_small_components(rm.verts, rm.faces, 0.02f); }
        const std::vector<float>& dverts = rm.F() > 0 ? rm.verts : mesh.verts;
        const std::vector<int32_t>& dfaces = rm.F() > 0 ? rm.faces : mesh.faces;
        FILE* dfp = fopen(cfg.dump_post.c_str(), "wb");
        if (!dfp) { fprintf(stderr, "cannot write %s\n", cfg.dump_post.c_str()); return 1; }
        const int dV = (int)dverts.size()/3, dFc = (int)dfaces.size()/3;
        const int Mv = pbr6.empty() ? 0 : (int)pbr_coords->size(), res = pbr_res;
        fwrite(&dV,4,1,dfp); fwrite(&dFc,4,1,dfp); fwrite(&Mv,4,1,dfp); fwrite(&res,4,1,dfp);
        fwrite(dverts.data(),4,(size_t)dV*3,dfp); fwrite(dfaces.data(),4,(size_t)dFc*3,dfp);
        if (Mv) { for (auto& c : *pbr_coords) { int xyz[3]={c[0],c[1],c[2]}; fwrite(xyz,4,3,dfp); } fwrite(pbr6.data(),4,(size_t)Mv*6,dfp); }
        fclose(dfp);
        printf("[6/6] cleaned dump -> %s (V=%d F=%d%s, PBR=%d @res%d)\n", cfg.dump_post.c_str(), dV, dFc,
               rm.F() > 0 ? " remeshed" : " raw (remesh empty)", Mv, res);
        printf("done in %.1fs -> %s\n", now() - t0, cfg.dump_post.c_str());
        return 0;
    }

    printf("[6/6] write %s\n", outglb.c_str());
    bool textured = false;
    if (!pbr6.empty()) {
        const bool boxuv = !cfg.xatlas;
        // MV mode defaults (unless overridden by --atlas/--tex, --decim): atlas 4096 and a
        // 1,000,000-face quadric target, matching inference_mv.py's o_voxel.postprocess.to_glb(
        // texture_size=4096, decimation_target=1_000_000) -- TRELLIS.2's 2048/300K stay as-is below.
        const int T = cfg.tex >= 0 ? cfg.tex : 4096;
        trellis::weld_vertices(mesh.verts, mesh.faces, colors.empty() ? nullptr : &colors, 1.0f / ((float)so.res * 8.0f));
        trellis::fill_small_holes(mesh.faces);
        trellis::TriBvh bvh = trellis::TriBvh::build(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F());
        int remesh_band = cfg.band > 0 ? cfg.band : 1;   // MV default: 1 (see dump_post branch above)
        trellis::Mesh rm = trellis::remesh_narrow_band_dc(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), bvh, so.res, remesh_band);
        if (rm.F() > 0) {
            trellis::clean_mesh(rm.V(), rm.faces);
            int ndrop = trellis::drop_small_components(rm.verts, rm.faces, 0.02f);
            printf("  remesh postproc: dropped %d floater comps -> V=%d F=%d\n", ndrop, rm.V(), rm.F());
        }
        const std::vector<float>& sverts = rm.F() > 0 ? rm.verts : mesh.verts;
        const std::vector<int32_t>& sfaces = rm.F() > 0 ? rm.faces : mesh.faces;
        std::vector<float> dv, dp; std::vector<int32_t> df;
        if (cfg.decim > 0) {
            trellis::decimate_cluster(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, {}, cfg.decim, dv, df, dp);
        } else if (cfg.decim == 0) {
            dv = sverts; df = sfaces;
        } else {
            trellis::decimate_qem(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, 1000000, dv, df);
            trellis::weld_vertices(dv, df, nullptr, 1.0f / ((float)so.res * 8.0f));
            trellis::fill_small_holes(df);
            int ndrop2 = trellis::drop_small_components(dv, df, 0.03f);
            if (ndrop2) printf("  decimated postproc: dropped %d more comps -> F=%d\n", ndrop2, (int)df.size()/3);
        }
        const int dV = (int)dv.size()/3, dF = (int)df.size()/3;
        trellis::VoxelPbr vox{pbr_coords, &pbr6, pbr_res, &bvh};
        const std::vector<float> no_vp;
        trellis::BakedMesh bm = boxuv ? trellis::uv_box_project(dv, dV, df, dF, no_vp, T, &vox)
                                      : trellis::uv_bake(dv, dV, df, dF, no_vp, T, &vox);
        if (!boxuv && !bm.ok()) bm = trellis::uv_chart_project(dv, dV, df, dF, no_vp, T, &vox);
        if (bm.ok()) {
            mv_apply_reference_frame(bm.verts);   // baking (UVs/atlas) is done -- safe to rotate now
            trellis::write_glb_textured(outglb.c_str(), bm.verts.data(), (int64_t)bm.verts.size()/3, bm.uv.data(),
                                        bm.faces.data(), (int64_t)bm.faces.size()/3, bm.base.data(), bm.mr.data(), bm.T,
                                        /*double_sided=*/rm.F() == 0, run_seed,
                                        cfg.copyright.empty() ? nullptr : cfg.copyright.c_str(),
                                        /*use_webp=*/cfg.webp != 0);
            std::string tex = outglb.substr(0, outglb.find_last_of('.')) + "_base.png";
            stbi_write_png(tex.c_str(), bm.T, bm.T, 4, bm.base.data(), bm.T*4);
            textured = true;
            printf("      textured GLB (atlas %d, +%s)\n", bm.T, tex.c_str());
        } else printf("      uv_bake failed; falling back to vertex colors\n");
    }
    mv_apply_reference_frame(mesh.verts);   // matches inference_mv.py's post-hoc glb.apply_transform(rot)
    if (!textured)
        trellis::write_glb(outglb.c_str(), mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(),
                           colors.empty() ? nullptr : colors.data(), run_seed,
                           cfg.copyright.empty() ? nullptr : cfg.copyright.c_str());
    std::string ply = outglb.substr(0, outglb.find_last_of('.')) + ".ply";
    trellis::write_ply(ply.c_str(), mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), colors.empty() ? nullptr : colors.data());
    printf("done in %.1fs -> %s (+ %s)\n", now() - t0, outglb.c_str(), ply.c_str());
    return 0;
}

int trellis_run(const trellis::TrellisParams& cfg) {
    if (!cfg.views.empty()) return trellis_run_mv(cfg);
    // Unbuffered, not line-buffered: MSVCRT treats _IOLBF as full buffering, which
    // swallows stage progress when piped (e.g. under Lemonade) if the process crashes.
    setvbuf(stdout, nullptr, _IONBF, 0);
    uint32_t run_seed = cfg.seed;
    if (run_seed == 0) {
        std::random_device rd;
        run_seed = (uint32_t(rd()) << 16) ^ uint32_t(rd()) ^
                   uint32_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        if (run_seed == 0) run_seed = 1;
        fprintf(stderr, "[trellis] generating model with seed %u (auto)\n", run_seed);
    } else {
        fprintf(stderr, "[trellis] generating model with seed %u\n", run_seed);
    }
    // Publish the cross-module flags this run wants (modules read them with an env fallback).
    const bool F32 = cfg.f32; trellis::g_sparse_cast_f32 = F32;  // f16 default (rope bug was the real issue)
    trellis::g_no_fa = cfg.no_fa;
    trellis::g_require_gpu = cfg.require_gpu;
    trellis::g_cpu_threads = cfg.threads;
    const std::string& img = cfg.image;
    const std::string& outglb = cfg.output;
    const std::string& M = cfg.models;
    const int gpu = cfg.gpu;
    const bool cascade = cfg.cascade;   // 1024 cascade is the TRELLIS default; --res 512 forces the light path
    std::mt19937 rng(run_seed); std::normal_distribution<float> randn(0.f, 1.f);
    auto noise = [&](size_t n){ vector<float> v(n); for (auto& x : v) x = randn(rng); return v; };
    double t0 = now();

    bool birefnet = cfg.birefnet == 1;
    if (cfg.birefnet < 0) {
        // Auto bg-removal. A pre-matted image keeps its own alpha (threshold path uses it
        // as-is). Otherwise prefer the neural matte: the white-threshold rule reads specular
        // highlights (min(RGB)>=232) as background, and conditioning the flow on an object
        // with punched-out highlights makes it generate HOLES right there (issue #1
        // follow-ups: helmet-crest and axe-edge highlights on assets/goblin.png).
        if (trellis::image_has_alpha(img)) {
            birefnet = false;
        } else {
            FILE* bf = fopen((M + "/birefnet.gguf").c_str(), "rb");
            if (bf) { fclose(bf); birefnet = true; }
            else printf("      (birefnet.gguf not found -- falling back to threshold matte;"
                        " bright highlights may punch holes)\n");
        }
    }
    vector<float> chw, chw1024;
    std::vector<unsigned char> cutout; int cut_sz = 0;   // the bg-removal result (for --dump-bg / --bg-only)
    if (birefnet) {
        printf("[1/6] preprocess %s (BiRefNet bg removal, %s)\n", img.c_str(), cascade ? "1024 cascade" : "512");
        // Full BiRefNet (Swin-L backbone + deformable-conv decoder) runs on the GPU. Cutout computed
        // once, normalized for 512 and 1024.
        trellis::Model bm = trellis::Model::load(M + "/birefnet.gguf", gpu);
        cutout = trellis::birefnet_cutout(img, bm, gpu < 0 ? 0 : gpu, cut_sz);
        bm.free();
        if (cutout.empty()) return 1;
        chw = trellis::normalize_cutout(cutout, cut_sz, 512);
        if (cascade) chw1024 = trellis::normalize_cutout(cutout, cut_sz, 1024);
    } else {
        printf("[1/6] preprocess %s (%s)\n", img.c_str(), cascade ? "1024 cascade" : "512");
        cutout = trellis::threshold_cutout(img, cut_sz);
        if (cutout.empty()) return 1;
        chw = trellis::normalize_cutout(cutout, cut_sz, 512);
        if (cascade) chw1024 = trellis::normalize_cutout(cutout, cut_sz, 1024);
    }

    // --dump-bg: write the bg-removal cutout next to the output; --bg-only: stop here.
    if (cfg.dump_bg || cfg.bg_only) {
        const std::string cut_png = outglb.substr(0, outglb.find_last_of('.')) + "_cutout.png";
        const int cut_channels = cut_sz > 0 && cutout.size() == (size_t)cut_sz * cut_sz * 4 ? 4 : 3;
        if (cut_sz > 0 && stbi_write_png(cut_png.c_str(), cut_sz, cut_sz, cut_channels,
                                        cutout.data(), cut_sz*cut_channels))
            printf("      bg-removal cutout -> %s\n", cut_png.c_str());
        else
            fprintf(stderr, "      [warn] could not write bg-removal cutout to %s\n", cut_png.c_str());
        if (cfg.bg_only) { printf("[bg-only] done (%.1fs)\n", now() - t0); return 0; }
    }

    printf("[2/6] DINOv3 conditioning\n");
    vector<float> cond, cond1024;
    { trellis::Model m = trellis::Model::load(M + "/dinov3.gguf", gpu);
      cond = trellis::dinov3_encode(m, chw, 512);
      if (cascade) cond1024 = trellis::dinov3_encode(m, chw1024, 1024);
      m.free(); }
    const int Lc = (int)(cond.size() / 1024);
    vector<float> neg(cond.size(), 0.0f);
    const int Lc1024 = cascade ? (int)(cond1024.size() / 1024) : 0;
    vector<float> neg1024(cond1024.size(), 0.0f);
    printf("      cond tokens=%d%s\n", Lc, cascade ? (" / 1024-cond tokens=" + std::to_string(Lc1024)).c_str() : "");
    slat_stats("cond_512 (DINOv3@512)", cond);
    if (cascade) slat_stats("cond_1024 (DINOv3@1024)", cond1024);

    printf("[3/6] sparse-structure flow + decode\n");
    vector<std::array<int,3>> coords;
    {
        trellis::Model m = trellis::Model::load(M + "/ss_flow.gguf", gpu);
        trellis::DiTParams p; p.in_ch = 8; p.out_ch = 8; p.d_cond = 1024; p.cast_f32 = F32;
        trellis::DitRunner* run = trellis::make_dense_runner(m, p, 16, Lc);
        trellis::FlowFwd fwd = [&](const vector<float>& x, float ts, const float* c){ return run->forward(x, ts, c); };
        trellis::SamplerParams sp; sp.steps=12; sp.guidance_strength=cfg.gss; sp.guidance_rescale=0.7f; sp.gi0=0.6f; sp.gi1=1.0f; sp.rescale_t=5.0f;
        vector<float> z = trellis::sample_flow(fwd, noise(8*4096), cond.data(), neg.data(), sp);  // [8,4096] ne0=8
        delete run; m.free();
        // transpose [8,L] -> torch [8,16,16,16] memory (c*4096 + sp)
        vector<float> zdec(8*4096);
        for (int c = 0; c < 8; ++c) for (int sp2 = 0; sp2 < 4096; ++sp2) zdec[(size_t)c*4096 + sp2] = z[c + 8*sp2];
        trellis::Model d = trellis::Model::load(M + "/ss_dec.gguf", gpu);
        vector<float> logits = trellis::ss_decode(d, zdec); d.free();
        coords = trellis::ss_coords(logits, 64, 32);
    }
    if (cfg.voxply) { FILE*f=fopen("out/myvox.ply","wb"); fprintf(f,"ply\nformat binary_little_endian 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n",coords.size()); for(auto&c:coords){float p[3]={(c[0]+0.5f)/32-0.5f,(c[1]+0.5f)/32-0.5f,(c[2]+0.5f)/32-0.5f}; fwrite(p,4,3,f);} fclose(f); }
    printf("      active voxels @res32 = %d\n", (int)coords.size());
    if (coords.empty()) { fprintf(stderr, "no voxels produced\n"); return 1; }

    const bool do_tex = cfg.texture;

    // one shape SLAT flow run -> normalized [32,n] (sparse, CFG 7.5, gi[0.6,1], rescale_t 3)
    auto shape_flow = [&](const std::string& path, const vector<std::array<int,3>>& cds,
                          const float* cnd, const float* ncnd, int lc) {
        const int n = (int)cds.size();
        trellis::Model m = trellis::Model::load(path, gpu);
        trellis::DiTParams p; p.in_ch = 32; p.out_ch = 32; p.d_cond = 1024; p.cast_f32 = F32;
        trellis::DitRunner* run = trellis::make_sparse_runner(m, p, cds, lc);
        trellis::FlowFwd fwd = [&](const vector<float>& x, float ts, const float* c){ return run->forward(x, ts, c); };
        trellis::SamplerParams sp; sp.steps=12; sp.guidance_strength=cfg.gsh; sp.guidance_rescale=0.5f; sp.gi0=0.6f; sp.gi1=1.0f; sp.rescale_t=3.0f;
        vector<float> sn = trellis::sample_flow(fwd, noise((size_t)32*n), cnd, ncnd, sp);   // [32,n]
        delete run; m.free();
        return sn;
    };

    vector<float> slat_norm, slat_dn;        // normalized (for tex concat) and denormalized (for decode)
    vector<std::array<int,3>> shc;           // coords where the shape SLAT lives + is decoded
    int RES = 512;                           // final grid resolution
    const float* cond_dec = cond.data();     // cond used by HR shape + tex flows
    const float* neg_dec  = neg.data();
    int Lc_dec = Lc;
    vector<float> lr_norm, lr_dn;            // LR (res-512) shape slat @res32 — reused for the res-512 tex path
    if (cascade) {
        // Cascade target resolution: 1024 (default, '1024_cascade') or e.g. 1536 ('1536_cascade').
        // Both reuse the SAME shape_flow_1024 / tex_flow_1024 / cond_1024 — only the HR quantization
        // grid (res//16) and the final decode resolution change. The reference floors the backoff at
        // 1024 and caps the HR token count at max_num_tokens (49152) to stay within VRAM; the backoff
        // steps the grid down by -128 while the unique res//16 token count would exceed it.
        // Verified end-to-end: a clean reconstruction reaches grid 96 (res1536, ~15k tokens) and
        // decodes a coherent ~5.4M-vert mesh — the c26fc76 FA holds at the cascade's token counts.
        // (Caveat: a *bad TRELLIS seed* can yield a degenerate ~4x-bloated SLAT — ~44k tokens, garbage
        //  geometry, and it can NaN the HR FA at grid 80. That's a reconstruction-quality problem, not
        //  an FA limit: re-roll the seed rather than lowering this cap.)
        const int hr_target = cfg.hr_res;
        const int max_tok   = cfg.max_tokens;
        printf("[4/7] shape SLAT flow (LR 512 -> upsample -> HR %d cascade, max_tok=%d)\n", hr_target, max_tok);
        // (1) LR shape flow @res32 with cond_512
        lr_norm = shape_flow(M + "/shape_flow_512.gguf", coords, cond.data(), neg.data(), Lc);
        lr_dn.resize(lr_norm.size());
        for (size_t n = 0; n < coords.size(); ++n) for (int c = 0; c < 32; ++c)
            lr_dn[(size_t)c + 32*n] = lr_norm[(size_t)c + 32*n]*SHAPE_STD[c] + SHAPE_MEAN[c];
        slat_stats("LR slat (res32, decodes OK via upsample)", lr_dn);
        // (2) decoder.upsample(LR slat, 4) -> res512 coords
        vector<std::array<int,3>> hr_coords;
        { trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
          hr_coords = trellis::shape_upsample(m, lr_dn, coords); m.free(); }
        // (3) quantize res512 -> res(hr_res//16) with the reference's adaptive token-budget backoff
        //     (sample_shape_slat_cascade): start at hr_target, step -128 toward the 1024 floor while
        //     the unique token count would exceed max_num_tokens. grid = hr_res//16 is integral since
        //     128/16 = 8 (1536->96, 1408->88, ..., 1024->64).
        int hr_res = hr_target;
        for (;;) {
            const int gi = hr_res / 16;          // integral grid (ref's hr_resolution//16)
            const float g = (float)gi;
            std::set<std::array<int,3>> q;
            for (auto& c : hr_coords) q.insert({ (int)((c[0]+0.5f)/512.f*g), (int)((c[1]+0.5f)/512.f*g), (int)((c[2]+0.5f)/512.f*g) });
            if ((int)q.size() < max_tok || hr_res <= 1024) {
                shc.assign(q.begin(), q.end());
                printf("      upsampled coords @res512=%d -> quantized @res%d (grid %d) = %d tokens\n",
                       (int)hr_coords.size(), hr_res, gi, (int)shc.size());
                break;
            }
            printf("      res%d (grid %d) -> %d tokens >= %d, backing off -128\n",
                   hr_res, gi, (int)q.size(), max_tok);
            hr_res -= 128;
        }
        // (4) HR shape flow @res(hr_res//16) with cond_1024
        slat_norm = shape_flow(M + "/shape_flow_1024.gguf", shc, cond1024.data(), neg1024.data(), Lc1024);
        RES = hr_res; cond_dec = cond1024.data(); neg_dec = neg1024.data(); Lc_dec = Lc1024;
    } else {
        printf("[4/7] shape SLAT flow (512)\n");
        shc = coords;
        slat_norm = shape_flow(M + "/shape_flow_512.gguf", coords, cond.data(), neg.data(), Lc);
    }
    const int N = (int)shc.size();
    slat_dn.resize(slat_norm.size());
    for (int n = 0; n < N; ++n) for (int c = 0; c < 32; ++c)
        slat_dn[(size_t)c + 32*n] = slat_norm[(size_t)c + 32*n]*SHAPE_STD[c] + SHAPE_MEAN[c];
    slat_stats(cascade ? "HR slat" : "slat (res32)", slat_dn);
    if (cascade && cfg.dump_slat) {   // dump HR slat for the reference-decoder diff
        FILE* f = fopen("/tmp/hr_slat.bin", "wb");
        if (f) {
            int n = N, res = RES; fwrite(&n, 4, 1, f); fwrite(&res, 4, 1, f);
            for (auto& c : shc) { int xyz[3] = { c[0], c[1], c[2] }; fwrite(xyz, 4, 3, f); }
            fwrite(slat_dn.data(), 4, slat_dn.size(), f); fclose(f);
            printf("      [dump] /tmp/hr_slat.bin: N=%d res=%d feats=%zu\n", n, res, slat_dn.size());
        }
    }

    printf("[5/7] FlexiDualGrid shape decode -> mesh @res%d\n", RES);
    trellis::Mesh mesh;
    trellis::ShapeOut so;
    {
        trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
        so = trellis::shape_decode(m, slat_dn, shc, RES); m.free();
        printf("      decoded voxels @res%d = %d\n", so.res, (int)so.coords.size());
        mesh = trellis::dual_grid_to_mesh(so);
    }
    printf("      mesh V=%d F=%d\n", mesh.V(), mesh.F());
    {   // reference postprocess fills small holes BEFORE the remesh (max_hole_perimeter=3e-2):
        // open cracks make the narrow-band in/out test ambiguous and leak holes into the
        // remeshed surface (ours had ~30x the reference's boundary edges without this).
        const int nh = trellis::fill_holes(mesh.verts, mesh.faces, 3e-2f);
        if (nh) printf("      filled %d small holes -> V=%d F=%d\n", nh, mesh.V(), mesh.F());
    }
    if (const char* dp = std::getenv("TRELLIS_DUMP_DECMESH")) {   // pre-remesh decoded mesh (int V,F,verts,faces) for remesh A/B
        FILE* f = fopen(dp, "wb"); if (f) { int v=mesh.V(), fc=mesh.F();
            fwrite(&v,4,1,f); fwrite(&fc,4,1,f); fwrite(mesh.verts.data(),4,mesh.verts.size(),f); fwrite(mesh.faces.data(),4,mesh.faces.size(),f); fclose(f);
            printf("      [dump] pre-remesh mesh -> %s\n", dp); fflush(stdout); }
    }

    vector<float> colors, pbr6;   // colors = base RGB (PLY); pbr6 = per-vertex [V*6] for UV bake
    trellis::ShapeOut so_tex;                                    // res-512 tex-guide decode (mixed-res path)
    const vector<std::array<int,3>>* pbr_coords = &so.coords;    // coords/res the bake samples the PBR at
    int pbr_res = so.res;
    if (do_tex) {
        // Texture PBR resolution. A dense res-1024 decode paints its outermost voxel layer with a
        // partial-coverage "skin" (dark/incoherent), and the bake, snapping the decimated mesh through
        // that layer, samples it inconsistently -> salt-and-pepper / colour-patch speckle. (Verified on
        // the reference decoder: it produces the same skin at res-1024.) The res-512 decode never
        // resolves that layer, so its clean, coherent PBR baked onto the res-1024 mesh keeps the
        // geometry detail without the speckle. Auto: drop to 512 above ~9M voxels; --tex-res forces it.
        constexpr int DENSE_TEX = 9000000;
        const int tex_res = cfg.tex_res > 0 ? cfg.tex_res
                          : (cascade && (int)so.coords.size() > DENSE_TEX ? 512 : RES);
        const bool mixed = cascade && tex_res != RES;   // res-1024 geometry + res-512 texture
        printf("[6/7] texture SLAT flow + PBR decode%s\n", mixed ? "  (res-512 texture on res-1024 mesh)" : "");

        if (mixed) {   // decode a res-512 shape (from the LR slat) to guide the res-512 tex decode
            trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
            so_tex = trellis::shape_decode(m, lr_dn, coords, 512); m.free();
            pbr_coords = &so_tex.coords; pbr_res = so_tex.res;
            printf("      res-512 tex-guide decode: %d voxels\n", (int)so_tex.coords.size());
        }
        // tex flow + decode inputs: HR path (shc/slat_norm/cond_dec/so.subs) vs res-512 mixed path
        // (coords/lr_norm/cond_512/so_tex.subs). The tex decoder upsamples via the guide subdivision.
        const std::string tflow = M + (mixed ? "/tex_flow_512.gguf" : (cascade ? "/tex_flow_1024.gguf" : "/tex_flow_512.gguf"));
        const vector<std::array<int,3>>& tcoords = mixed ? coords : shc;
        const vector<float>& tslat = mixed ? lr_norm : slat_norm;
        const float* tcond = mixed ? cond.data() : cond_dec;
        const float* tneg  = mixed ? neg.data()  : neg_dec;
        const int    tlc   = mixed ? Lc : Lc_dec;
        const int    tN    = (int)tcoords.size();
        const std::vector<std::vector<uint8_t>>& tsubs = mixed ? so_tex.subs : so.subs;

        vector<float> texlat;
        {
            trellis::Model m = trellis::Model::load(tflow, gpu);
            trellis::DiTParams p; p.in_ch = 64; p.out_ch = 32; p.d_cond = 1024; p.cast_f32 = F32;
            trellis::DitRunner* run = trellis::make_sparse_runner(m, p, tcoords, tlc);
            // state is the 32-ch noise; each forward concat [noise(32) ; shape_slat_norm(32)] -> 64ch
            trellis::FlowFwd fwd = [&](const vector<float>& st, float ts, const float* c) {
                vector<float> x64((size_t)64 * tN);
                for (int n = 0; n < tN; ++n) {
                    for (int k = 0; k < 32; ++k) x64[(size_t)k + 64*n]      = st[(size_t)k + 32*n];
                    for (int k = 0; k < 32; ++k) x64[(size_t)32 + k + 64*n] = tslat[(size_t)k + 32*n];
                }
                return run->forward(x64, ts, c);
            };
            trellis::SamplerParams sp; sp.steps=12; sp.guidance_strength=1.0f; sp.guidance_rescale=0.0f; sp.gi0=0.6f; sp.gi1=0.9f; sp.rescale_t=3.0f;
            texlat = trellis::sample_flow(fwd, noise((size_t)32*tN), tcond, tneg, sp);  // [32,tN]
            delete run; m.free();
            for (int n = 0; n < tN; ++n) for (int c = 0; c < 32; ++c) texlat[(size_t)c + 32*n] = texlat[(size_t)c + 32*n]*TEX_STD[c] + TEX_MEAN[c];
        }
        {
            trellis::Model m = trellis::Model::load(M + "/tex_dec.gguf", gpu);
            vector<float> pbr = trellis::tex_decode(m, texlat, tcoords, tsubs); m.free();   // [6,Mv] pre-scale
            const int Mv = (int)pbr_coords->size();
            colors.resize((size_t)Mv * 3); pbr6.resize((size_t)Mv * 6);
            auto cl = [](float v){ return v < 0 ? 0.f : (v > 1 ? 1.f : v); };
            for (int i = 0; i < Mv; ++i) {
                for (int k = 0; k < 6; ++k) pbr6[(size_t)i*6 + k] = cl(pbr[(size_t)k + 6*i] * 0.5f + 0.5f);
                for (int k = 0; k < 3; ++k) colors[(size_t)i*3 + k] = pbr6[(size_t)i*6 + k];
            }
            printf("      PBR voxels=%d @res%d\n", Mv, pbr_res);
        }
        // `colors` is per-VOXEL but consumed per-VERTEX (weld, vertex-color GLB, PLY),
        // relying on dual_grid_to_mesh's vertex==voxel correspondence. fill_holes adds
        // cap vertices beyond Mv -- pad them (neutral grey; the caps are sub-voxel and
        // the real texture comes from the voxel-volume bake, not these colors). In
        // mixed-res mode the correspondence never held at all (res-512 voxels vs
        // res-1024 vertices): drop the colors instead of reading out of bounds.
        // (Windows surfaced this as a silent crash at [7/7]; Linux read garbage.)
        if (colors.size() != (size_t)mesh.V() * 3) {
            if (pbr_res == so.res && colors.size() < (size_t)mesh.V() * 3)
                colors.resize((size_t)mesh.V() * 3, 0.5f);
            else
                colors.clear();
        }
    }

    if (!cfg.dump_post.empty()) {
        // Machine-readable output for external post-processing pipelines
        // (simplification budgets / UV / baking done by the caller, e.g.
        // QtMeshEditor): the CLEANED geometry mesh + the sparse PBR volume it
        // would be baked from, then exit before decimate/UV/bake/GLB.
        //
        // The reference cleanup chain runs first (same as the GLB path):
        // weld -> fill_small_holes -> narrow-band DC remesh -> clean ->
        // drop floaters. The raw dual-grid decode is non-manifold at
        // sub-voxel scale with inconsistently-wound patches — downstream
        // consumers hit unwrap/simplify/culling pathologies on it — while
        // the remeshed shell is watertight with consistent winding. Falls
        // back to the raw decoded mesh if the remesh produces nothing.
        //
        // Binary layout matches the TRELLIS_DUMP_POST debug env:
        // i32 V,F,Mv,res; f32 verts[V*3]; i32 faces[F*3]; i32 coords[Mv*3];
        // f32 pbr6[Mv*6] (base_color.rgb, metallic, roughness, alpha in [0,1]).
        trellis::weld_vertices(mesh.verts, mesh.faces, nullptr,
                               1.0f / ((float)so.res * 8.0f));
        trellis::fill_small_holes(mesh.faces);
        trellis::TriBvh dbvh = trellis::TriBvh::build(mesh.verts.data(), mesh.V(),
                                                      mesh.faces.data(), mesh.F());
        const int dband = cfg.band > 0 ? cfg.band : std::max(1, so.res / 512);
        trellis::Mesh rm = trellis::remesh_narrow_band_dc(mesh.verts.data(), mesh.V(),
                                                          mesh.faces.data(), mesh.F(),
                                                          dbvh, so.res, dband);
        if (rm.F() > 0) {
            trellis::clean_mesh(rm.V(), rm.faces);
            trellis::drop_small_components(rm.verts, rm.faces, 0.02f);
        }
        const std::vector<float>&   dverts = rm.F() > 0 ? rm.verts : mesh.verts;
        const std::vector<int32_t>& dfaces = rm.F() > 0 ? rm.faces : mesh.faces;
        FILE* dfp = fopen(cfg.dump_post.c_str(), "wb");
        if (!dfp) { fprintf(stderr, "cannot write %s\n", cfg.dump_post.c_str()); return 1; }
        const int dV = (int)dverts.size() / 3, dFc = (int)dfaces.size() / 3;
        const int Mv = pbr6.empty() ? 0 : (int)pbr_coords->size(), res = pbr_res;
        fwrite(&dV,4,1,dfp); fwrite(&dFc,4,1,dfp); fwrite(&Mv,4,1,dfp); fwrite(&res,4,1,dfp);
        fwrite(dverts.data(),4,(size_t)dV*3,dfp);
        fwrite(dfaces.data(),4,(size_t)dFc*3,dfp);
        if (Mv) {
            for (auto& c : *pbr_coords) { int xyz[3] = {c[0],c[1],c[2]}; fwrite(xyz,4,3,dfp); }
            fwrite(pbr6.data(),4,(size_t)Mv*6,dfp);
        }
        fclose(dfp);
        printf("[7/7] cleaned dump -> %s (V=%d F=%d%s, PBR=%d @res%d)\n",
               cfg.dump_post.c_str(), dV, dFc,
               rm.F() > 0 ? " remeshed" : " raw (remesh empty)", Mv, res);
        printf("done in %.1fs -> %s\n", now() - t0, cfg.dump_post.c_str());
        return 0;
    }

    printf("[7/7] write %s\n", outglb.c_str());
    bool textured = false;
    if (!pbr6.empty()) {   // UV-baked textured GLB (PBR material)
        // UV method: xatlas unwrap by default — unique chart space per face, no projection
        // overlap. --box-uv selects the voxel-native 6-way box projection: O(F) and seconds vs
        // xatlas's ~superlinear chart-compute (grid 384 -> 382K faces took >9min pre-decimation),
        // occlusion-aware bucket assignment + depth-tested raster keep its bleed low.
        const bool boxuv = !cfg.xatlas;
        const int T = cfg.tex >= 0 ? cfg.tex : (cascade ? 2048 : 1024);
        if (const char* dp = std::getenv("TRELLIS_DUMP_POST")) {
            FILE* dfp = fopen(dp, "wb");
            if (dfp) {   // geometry mesh + the PBR volume the bake samples (may be res-512 in mixed mode)
                int dV = mesh.V(), dFc = mesh.F(), Mv = (int)pbr_coords->size(), res = pbr_res;
                fwrite(&dV,4,1,dfp); fwrite(&dFc,4,1,dfp); fwrite(&Mv,4,1,dfp); fwrite(&res,4,1,dfp);
                fwrite(mesh.verts.data(),4,(size_t)dV*3,dfp);
                fwrite(mesh.faces.data(),4,(size_t)dFc*3,dfp);
                for (auto& c : *pbr_coords) { int xyz[3] = {c[0],c[1],c[2]}; fwrite(xyz,4,3,dfp); }
                fwrite(pbr6.data(),4,(size_t)Mv*6,dfp);
                fclose(dfp);
                printf("      [dump] post-stage inputs -> %s\n", dp);
            }
        }
        trellis::weld_vertices(mesh.verts, mesh.faces, colors.empty() ? nullptr : &colors,
                               1.0f / ((float)so.res * 8.0f));
        trellis::fill_small_holes(mesh.faces);
        // Reference production pipeline: rebuild the noisy dual-grid mesh as
        // the narrow-band offset shell (watertight manifold), then quadric
        // simplify to the face target. The BVH over the original hole-filled
        // mesh serves both the remesh UDF and the bake's texel snap.
        trellis::TriBvh bvh = trellis::TriBvh::build(mesh.verts.data(), mesh.V(),
                                                     mesh.faces.data(), mesh.F());
        // The narrow-band remesh offset is eps = band*scale/res, i.e. it shrinks with
        // resolution. At res-512 (band=1) that offset absorbs the decoder's sub-voxel
        // "outer-skin" noise; at res-1024 the SAME band=1 halves the world-space offset,
        // so the noise survives as the issue-#22 speckle. Default band (cfg.band==0)
        // scales with resolution to keep the offset resolution-independent (512->1,
        // 1024->2, 1536->3); an explicit --band / per-request band forces that value.
        int remesh_band = cfg.band > 0 ? cfg.band : std::max(1, so.res / 512);
        trellis::Mesh rm = trellis::remesh_narrow_band_dc(mesh.verts.data(), mesh.V(),
                                                          mesh.faces.data(), mesh.F(),
                                                          bvh, so.res, remesh_band);
        // Clean the narrow-band DC output (drop degenerate faces, unify winding), then drop
        // decode floaters (the reference is a single watertight component; ours shattered into
        // 50+ pieces). The faithful QEM simplifier below handles surface smoothing via its
        // quadric + skinny-triangle cost, so no Taubin pre-pass is needed (and none is applied,
        // which keeps the mesh aligned to the voxel PBR volume for correct texture sampling).
        if (rm.F() > 0) {
            trellis::clean_mesh(rm.V(), rm.faces);
            int ndrop = trellis::drop_small_components(rm.verts, rm.faces, 0.02f);
            printf("  remesh postproc: dropped %d floater comps -> V=%d F=%d\n", ndrop, rm.V(), rm.F());
            fflush(stdout);
        }
        const std::vector<float>& sverts = rm.F() > 0 ? rm.verts : mesh.verts;
        const std::vector<int32_t>& sfaces = rm.F() > 0 ? rm.faces : mesh.faces;
        std::vector<float> dv, dp; std::vector<int32_t> df;
        if (cfg.decim > 0) {
            trellis::decimate_cluster(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, {}, cfg.decim, dv, df, dp);
        } else if (cfg.decim == 0) {
            dv = sverts; df = sfaces;
        } else {
            trellis::decimate_qem(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3,
                                  cascade ? 300000 : 150000, dv, df);
            trellis::weld_vertices(dv, df, nullptr, 1.0f / ((float)so.res * 8.0f));
            trellis::fill_small_holes(df);
            // Second component pass on the decimated mesh: a hallucinated ground plane
            // survives the dense-mesh drop (it decimates to a large flat slab) but is a
            // small fraction here and disconnected from the body. Ref is a single component.
            int ndrop2 = trellis::drop_small_components(dv, df, 0.03f);
            if (ndrop2) { printf("  decimated postproc: dropped %d more comps -> F=%d\n", ndrop2, (int)df.size()/3); fflush(stdout); }
        }
        const int dV = (int)dv.size()/3, dF = (int)df.size()/3;
        // Texels are shaded straight from the per-voxel PBR volume (trilinear sampling, the
        // reference bake behavior) rather than from decimation-averaged vertex colors, so
        // full material detail survives simplification.
        trellis::VoxelPbr vox{pbr_coords, &pbr6, pbr_res, &bvh};
        const std::vector<float> no_vp;
        trellis::BakedMesh bm = boxuv ? trellis::uv_box_project(dv, dV, df, dF, no_vp, T, &vox)
                                      : trellis::uv_bake(dv, dV, df, dF, no_vp, T, &vox);
        if (!boxuv && !bm.ok()) bm = trellis::uv_chart_project(dv, dV, df, dF, no_vp, T, &vox);
        if (bm.ok()) {
            trellis::write_glb_textured(outglb.c_str(), bm.verts.data(), (int64_t)bm.verts.size()/3, bm.uv.data(),
                                        bm.faces.data(), (int64_t)bm.faces.size()/3, bm.base.data(), bm.mr.data(), bm.T,
                                        /*double_sided=*/rm.F() == 0, run_seed,
                                        cfg.copyright.empty() ? nullptr : cfg.copyright.c_str(),
                                        /*use_webp=*/cfg.webp != 0);
            std::string tex = outglb.substr(0, outglb.find_last_of('.')) + "_base.png";
            stbi_write_png(tex.c_str(), bm.T, bm.T, 4, bm.base.data(), bm.T*4);
            textured = true;
            printf("      textured GLB (atlas %d, +%s)\n", bm.T, tex.c_str());
        } else printf("      uv_bake failed; falling back to vertex colors\n");
    }
    if (!textured)
        trellis::write_glb(outglb.c_str(), mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(),
                           colors.empty() ? nullptr : colors.data(), run_seed,
                           cfg.copyright.empty() ? nullptr : cfg.copyright.c_str());
    std::string ply = outglb.substr(0, outglb.find_last_of('.')) + ".ply";
    trellis::write_ply(ply.c_str(), mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), colors.empty() ? nullptr : colors.data());
    printf("done in %.1fs -> %s (+ %s)\n", now() - t0, outglb.c_str(), ply.c_str());
    return 0;
}
