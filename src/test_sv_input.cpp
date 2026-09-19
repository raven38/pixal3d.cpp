// 単一視点（SV）入力経路の回帰テスト。設計書 docs/design/2026-09-19-desktop-0.10.0-sv-parity.md
// の D1（gauge カメラ合成）と D2（pre-matted RGBA クロップ）の契約を native 側で固定する。
//
//   ./build/trellis-test-sv-input [tests/fixtures/sv_gauge]
//
// 引数はブラウザ実装（web/app/single_view.js）から生成した gauge フィクスチャのディレクトリ。
// 既定 "tests/fixtures/sv_gauge" はリポジトリルートからの相対なので、ルートで実行すること。
// D9 の C++<->JS 一致検査はこのフィクスチャを両側が読む形で行う
// （JS 側は tools/test_sv_gauge_parity.mjs）。
#include "image_preprocess.h"
#include "transforms_json.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

using namespace trellis;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
static void check_eq(int got, int want, const char* what) {
    const bool ok = got == want;
    printf("%s %s (got %d, want %d)\n", ok ? "ok  " : "FAIL", what, got, want);
    if (!ok) ++failures;
}

// ---- テスト用の合成 RGBA -------------------------------------------------------------
// 全面 alpha=bg、矩形 [x0,x1]x[y0,y1] を alpha=fg で塗る。RGB は alpha と同じ値にして、
// premultiply 後の比較でも位置がわかるようにする。
static std::vector<unsigned char> make_rgba(int w, int h, int bg,
                                            int x0, int y0, int x1, int y1, int fg) {
    std::vector<unsigned char> img((size_t)w * h * 4, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const bool inside = (x >= x0 && x <= x1 && y >= y0 && y <= y1);
            const unsigned char a = (unsigned char)(inside ? fg : bg);
            const size_t i = ((size_t)y * w + x) * 4;
            img[i + 0] = img[i + 1] = img[i + 2] = a;
            img[i + 3] = a;
        }
    }
    return img;
}

// ---- D1: gauge カメラ ----------------------------------------------------------------
static void test_gauge() {
    printf("-- D1 single-view gauge --\n");
    TransformsFile tf;
    std::string err;

    check(synthesize_single_view_gauge("input.png", SINGLE_VIEW_DEFAULT_FOV, tf, err),
          "gauge with the default 20 deg FOV");
    check(tf.frames.size() == 1, "exactly one frame");
    if (tf.frames.size() == 1) {
        check(tf.frames[0].file_path == "input.png", "file_path is preserved");
        // distance = 1 / (2 * mesh_scale * tan(fov/2))、mesh_scale は 1.0 固定。
        const double d = 1.0 / (2.0 * (double)SINGLE_VIEW_MESH_SCALE *
                                std::tan((double)SINGLE_VIEW_DEFAULT_FOV / 2.0));
        const float want[16] = { 1, 0,  0, 0,
                                 0, 0, -1, (float)-d,
                                 0, 1,  0, 0,
                                 0, 0,  0, 1 };
        bool mat_ok = true;
        for (int k = 0; k < 16; ++k) {
            if (std::fabs(tf.frames[0].transform_matrix[k] - want[k]) > 1e-6f) {
                mat_ok = false;
                printf("     element %d: got %.9g want %.9g\n", k,
                       (double)tf.frames[0].transform_matrix[k], (double)want[k]);
            }
        }
        check(mat_ok, "transform_matrix matches the front gauge pose");
        // JS は FOV を double リテラルで持つのに対し、ここは float（TransformsFrame が float
        // 格納なので上げても意味がない）。差は相対 3e-8 で、D9 の許容 1e-6 に収まる。
        const double js_distance = 2.8356409098088546;   // web/app/single_view.js の 20 deg
        const double rel = std::fabs(d - js_distance) / js_distance;
        printf("     gauge distance: C++ %.16g / JS %.16g (rel %.3g)\n", d, js_distance, rel);
        check(rel < 1e-6, "20 deg gauge distance matches web/app/single_view.js within 1e-6");
    }
    check(tf.has_camera_angle_x && tf.camera_angle_x == SINGLE_VIEW_DEFAULT_FOV,
          "camera_angle_x is the requested FOV");
    check(tf.mesh_scale == SINGLE_VIEW_MESH_SCALE, "mesh_scale is the fixed gauge 1.0");

    // FOV が大きいほどカメラは近づく（distance は単調減少）。
    TransformsFile narrow, wide;
    check(synthesize_single_view_gauge("a.png", 0.2f, narrow, err) &&
          synthesize_single_view_gauge("a.png", 1.0f, wide, err),
          "gauge accepts other valid FOVs");
    if (narrow.frames.size() == 1 && wide.frames.size() == 1) {
        check(std::fabs(wide.frames[0].transform_matrix[7]) <
              std::fabs(narrow.frames[0].transform_matrix[7]),
              "a wider FOV puts the camera closer");
    }

    // 範囲外・非有限は fail closed。既定へ落とさない。
    const float bad[] = { 0.0f, -0.1f, 3.14159274f, 4.0f,
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity() };
    bool all_rejected = true;
    for (float f : bad) {
        TransformsFile t;
        std::string e;
        if (synthesize_single_view_gauge("a.png", f, t, e)) {
            all_rejected = false;
            printf("     FOV %g was accepted\n", (double)f);
        }
    }
    check(all_rejected, "0, negative, >= pi, NaN and inf FOVs are rejected");

    TransformsFile t;
    check(!synthesize_single_view_gauge("", SINGLE_VIEW_DEFAULT_FOV, t, err),
          "an empty file name is rejected");
}

// ---- D2: pre-matted RGBA クロップ ------------------------------------------------------
static void test_crop() {
    printf("-- D2 pre-matted RGBA crop --\n");
    std::string err;

    // 1. 縮小なし（長辺 <= 1024）。bbox [10,29]x[5,24] -> span 19 -> crop 20、左上 (9,4)。
    {
        auto img = make_rgba(100, 60, 0, 10, 5, 29, 24, 255);
        PrematteCrop out;
        check(preprocess_prematted_rgba(img.data(), 100, 60, out, err), "plain RGBA crop");
        check_eq(out.resized_w, 100, "no resize below 1024 (width)");
        check_eq(out.resized_h, 60, "no resize below 1024 (height)");
        check_eq(out.rect.left, 9, "crop left");
        check_eq(out.rect.top, 4, "crop top");
        check_eq(out.rect.size, 20, "crop size = 2*floor(floor(span*1.1)/2)");
        check_eq((int)out.rgba.size(), 20 * 20 * 4, "buffer matches the crop square");
    }

    // 2. foreground が画像端に接する場合、クロップは画像外へ出て透明でパディングされる。
    {
        auto img = make_rgba(40, 40, 0, 0, 0, 9, 9, 255);
        PrematteCrop out;
        check(preprocess_prematted_rgba(img.data(), 40, 40, out, err), "edge-touching crop");
        // bbox [0,9]x[0,9] -> cx=cy=4.5, span=9, crop=max(2,floor(9.9))=9 -> half=4 -> size 8
        check_eq(out.rect.size, 8, "odd crop_size is rounded down to an even square");
        check_eq(out.rect.left, 0, "left = floor(4.5 - 4)");
        check_eq(out.rect.top, 0, "top = floor(4.5 - 4)");
    }
    {
        // 左端に寄せると left が負になり、その列は透明で埋まる。
        auto img = make_rgba(40, 40, 0, 0, 10, 1, 29, 255);
        PrematteCrop out;
        check(preprocess_prematted_rgba(img.data(), 40, 40, out, err), "crop that runs off the left edge");
        check(out.rect.left < 0, "left is negative (crop extends past the image)");
        bool padded = true;
        for (int y = 0; y < out.rect.size && padded; ++y) {
            const int src_x = out.rect.left;   // 最左列は画像外
            if (src_x >= 0) break;
            const size_t i = ((size_t)y * out.rect.size + 0) * 4;
            if (out.rgba[i + 3] != 0 || out.rgba[i + 0] != 0) padded = false;
        }
        check(padded, "out-of-image pixels are transparent (0,0,0,0)");
    }

    // 3. foreground 1 px でも 2x2 の最小クロップが出る。
    {
        auto img = make_rgba(32, 32, 0, 16, 16, 16, 16, 255);
        PrematteCrop out;
        check(preprocess_prematted_rgba(img.data(), 32, 32, out, err), "single-pixel foreground");
        check_eq(out.rect.size, 2, "crop_size floors to the 2 px minimum");
    }

    // 4. alpha 0.8 しきい値。alpha=200 (<= 204) は foreground に数えない。
    {
        auto img = make_rgba(50, 50, 0, 5, 5, 44, 44, 200);
        PrematteCrop out;
        check(!preprocess_prematted_rgba(img.data(), 50, 50, out, err),
              "alpha 200 is below the 0.8 foreground threshold -> rejected");
    }
    {
        auto img = make_rgba(50, 50, 0, 5, 5, 44, 44, 205);
        PrematteCrop out;
        check(preprocess_prematted_rgba(img.data(), 50, 50, out, err),
              "alpha 205 is above the 0.8 foreground threshold");
    }

    // 5. 完全不透明 / 完全透明は拒否する（pre-matted RGBA でない入力）。
    {
        auto img = make_rgba(16, 16, 255, 0, 0, 15, 15, 255);
        PrematteCrop out;
        check(!preprocess_prematted_rgba(img.data(), 16, 16, out, err),
              "a fully opaque image is rejected");
    }
    {
        // 最小 alpha 252: src/pixal3d_input.cpp の load_rgba と web/app/single_view.js に合わせ、
        // 250 未満の画素が 1 つも無ければ「real alpha 無し」として拒否する。
        auto img = make_rgba(16, 16, 252, 4, 4, 11, 11, 255);
        PrematteCrop out;
        check(!preprocess_prematted_rgba(img.data(), 16, 16, out, err),
              "min alpha 252 (no pixel below 250) is rejected like the MV loader does");
    }
    {
        auto img = make_rgba(16, 16, 0, 0, 0, -1, -1, 0);
        PrematteCrop out;
        check(!preprocess_prematted_rgba(img.data(), 16, 16, out, err),
              "a fully transparent image is rejected");
    }

    // 6. 長辺 > 1024 のときだけ画像全体を縮小してから bbox を取る。出力は 1024 を超えうる。
    //    （公式 Belle 入力が 1124^2 になるのがこの経路。縮小は resampler が絡むので、
    //     ここでは縮小寸法（自前計算）と「1024 を超えうる」性質だけを固定し、
    //     クロップ矩形のピクセル完全一致は D9 のフィクスチャ比較に委ねる。）
    {
        const int W = 2048, H = 1024;
        auto img = make_rgba(W, H, 0, 2, 2, W - 3, H - 3, 255);   // 2 px の透明縁
        PrematteCrop out;
        check(preprocess_prematted_rgba(img.data(), W, H, out, err), "oversized input is resized first");
        check_eq(out.resized_w, 1024, "resized width = floor(2048 * 1024/2048)");
        check_eq(out.resized_h, 512, "resized height keeps the aspect ratio");
        check(out.rect.size > 1024 && out.rect.size <= 1124,
              "the crop square may exceed 1024 (1.1x of a near-full frame)");
        printf("     crop size for the near-full 2048x1024 frame: %d\n", out.rect.size);
    }
}

// ---- D9: ブラウザ実装との一致（フィクスチャ経由）----------------------------------------
static void test_js_parity(const std::string& fixture_dir) {
    printf("-- D9 parity with web/app/single_view.js --\n");
    // ファイル名 -> そのフィクスチャを作った FOV。値は JS 側の double 演算で確定している。
    const struct { const char* file; double fov; } cases[] = {
        { "fov_20deg.json", 0.3490658503988659 },
        { "fov_30deg.json", 3.14159265358979323846 / 6.0 },
        { "fov_45deg.json", 3.14159265358979323846 / 4.0 },
        { "fov_min.json",   1e-3 },
        { "fov_max.json",   3.14159265358979323846 - 1e-3 },
    };
    for (const auto& c : cases) {
        const std::string path = fixture_dir + "/" + c.file;
        if (!std::filesystem::exists(path)) {
            check(false, (std::string("fixture is missing: ") + path).c_str());
            continue;
        }
        TransformsFile js;
        if (!load_transforms_json(path, js)) {
            check(false, (std::string("cannot parse fixture ") + c.file).c_str());
            continue;
        }
        TransformsFile cpp;
        std::string err;
        if (!synthesize_single_view_gauge(js.frames.empty() ? "input.png" : js.frames[0].file_path,
                                          (float)c.fov, cpp, err)) {
            check(false, (std::string("gauge failed for ") + c.file).c_str());
            continue;
        }
        bool ok = cpp.frames.size() == js.frames.size() && cpp.frames.size() == 1;
        // mesh_scale と FOV、行列 16 要素を相対誤差 1e-6 で比較する（C++ は float 格納、
        // JS は double なので完全一致は要求しない。D9 の許容）。
        auto close = [&](double a, double b) {
            const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
            return std::fabs(a - b) / scale < 1e-6;
        };
        ok = ok && close(cpp.mesh_scale, js.mesh_scale);
        ok = ok && close(cpp.camera_angle_x, js.camera_angle_x);
        if (cpp.frames.size() == 1 && js.frames.size() == 1) {
            for (int k = 0; k < 16; ++k) {
                if (!close(cpp.frames[0].transform_matrix[k], js.frames[0].transform_matrix[k])) {
                    ok = false;
                    printf("     %s element %d: C++ %.16g / JS %.16g\n", c.file, k,
                           (double)cpp.frames[0].transform_matrix[k],
                           (double)js.frames[0].transform_matrix[k]);
                }
            }
        }
        check(ok, (std::string("matches ") + c.file).c_str());
    }
}

int main(int argc, char** argv) {
    // --crop IN.png OUT.png : D2 のクロップだけを適用して PNG を書き、矩形を 1 行で出す。
    // tools/test_crop_parity.py が Python 実装と突き合わせるために使う（モデル不要）。
    if (argc >= 4 && std::string(argv[1]) == "--crop") {
        std::string err;
        int w = 0, h = 0, ch = 0;
        unsigned char* px = stbi_load(argv[2], &w, &h, &ch, 4);
        if (!px) { fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }
        PrematteCrop crop;
        const bool ok = preprocess_prematted_rgba(px, w, h, crop, err);
        stbi_image_free(px);
        if (!ok) { printf("REJECT %s\n", err.c_str()); return 3; }
        if (!stbi_write_png(argv[3], crop.rect.size, crop.rect.size, 4,
                            crop.rgba.data(), crop.rect.size * 4)) {
            fprintf(stderr, "cannot write %s\n", argv[3]); return 2;
        }
        printf("CROP left=%d top=%d size=%d resized=%dx%d\n",
               crop.rect.left, crop.rect.top, crop.rect.size, crop.resized_w, crop.resized_h);
        return 0;
    }
    const std::string fixture_dir = argc >= 2 ? argv[1] : "tests/fixtures/sv_gauge";
    test_gauge();
    test_crop();
    test_js_parity(fixture_dir);
    printf(failures ? "\nSV_INPUT_TEST_FAIL (%d)\n" : "\nSV_INPUT_TEST_OK\n", failures);
    return failures ? 1 : 0;
}
