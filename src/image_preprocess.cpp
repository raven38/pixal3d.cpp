#include "image_preprocess.h"

#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace trellis {
namespace {

// src/pixal3d_input.cpp の load_rgba と web/app/single_view.js に合わせたしきい値。
constexpr int kRealAlphaBelow = 250;      // これ未満の画素が 1 つも無ければ pre-matted でない
constexpr int kForegroundAbove = 204;     // 0.8 * 255。公式 pipeline.py の foreground しきい値
constexpr int kMaxSide = 1024;            // 手順 3 の縮小先（長辺）

} // namespace

bool preprocess_prematted_rgba(const unsigned char* rgba, int w, int h,
                               PrematteCrop& out, std::string& error) {
    if (!rgba || w <= 0 || h <= 0) {
        error = "empty image";
        return false;
    }

    // 2. pre-matted かどうかを縮小前に判定する（Python 実装と同じ順序）。
    int amin = 255, amax = 0;
    for (size_t i = 0, n = (size_t)w * h; i < n; ++i) {
        const int a = rgba[4 * i + 3];
        amin = std::min(amin, a);
        amax = std::max(amax, a);
    }
    if (amin >= kRealAlphaBelow) {
        error = "input image has no real alpha matte; provide a pre-matted RGBA image";
        return false;
    }
    if (amax == 0) {
        error = "input image is fully transparent";
        return false;
    }

    // 3. 長辺が 1024 を超えるときだけ画像全体を縮小する。
    const int max_side = std::max(w, h);
    std::vector<unsigned char> resized;
    const unsigned char* img = rgba;
    int rw = w, rh = h;
    if (max_side > kMaxSide) {
        const double scale = (double)kMaxSide / (double)max_side;
        rw = std::max(1, (int)std::floor((double)w * scale));
        rh = std::max(1, (int)std::floor((double)h * scale));
        resized.resize((size_t)rw * rh * 4);
        if (!stbir_resize_uint8(rgba, w, h, 0, resized.data(), rw, rh, 0, 4)) {
            error = "failed to resize the input image";
            return false;
        }
        img = resized.data();
    }
    out.resized_w = rw;
    out.resized_h = rh;

    // 4. alpha > 0.8*255 のマスクの bbox（両端含む）。
    int x0 = rw, y0 = rh, x1 = -1, y1 = -1;
    for (int y = 0; y < rh; ++y) {
        for (int x = 0; x < rw; ++x) {
            if (img[((size_t)y * rw + x) * 4 + 3] <= kForegroundAbove) continue;
            x0 = std::min(x0, x); x1 = std::max(x1, x);
            y0 = std::min(y0, y); y1 = std::max(y1, y);
        }
    }
    if (x1 < x0 || y1 < y0) {
        error = "alpha matte has no pixels above the Pixal3D foreground threshold (0.8)";
        return false;
    }

    // 5-6. 中心と 1.1 倍の偶数正方。
    const double cx = (x0 + x1) / 2.0;
    const double cy = (y0 + y1) / 2.0;
    const int span = std::max(x1 - x0, y1 - y0);
    int crop = std::max(2, (int)std::floor((double)span * 1.1));
    const int half = std::max(1, crop / 2);
    crop = 2 * half;

    // 7. 画像外は透明で埋める。
    out.rect.left = (int)std::floor(cx - (double)half);
    out.rect.top  = (int)std::floor(cy - (double)half);
    out.rect.size = crop;
    out.rgba.assign((size_t)crop * crop * 4, 0);
    for (int y = 0; y < crop; ++y) {
        const int sy = out.rect.top + y;
        if (sy < 0 || sy >= rh) continue;
        for (int x = 0; x < crop; ++x) {
            const int sx = out.rect.left + x;
            if (sx < 0 || sx >= rw) continue;
            std::memcpy(&out.rgba[((size_t)y * crop + x) * 4],
                        &img[((size_t)sy * rw + sx) * 4], 4);
        }
    }
    return true;
}

bool preprocess_prematted_rgba_file(const std::string& src, const std::string& dst,
                                    std::string& error) {
    int w = 0, h = 0, ch = 0;
    unsigned char* pixels = stbi_load(src.c_str(), &w, &h, &ch, 4);
    if (!pixels) {
        error = "cannot read input image " + src + ": " + (stbi_failure_reason() ? stbi_failure_reason() : "unknown");
        return false;
    }
    PrematteCrop crop;
    const bool ok = preprocess_prematted_rgba(pixels, w, h, crop, error);
    stbi_image_free(pixels);
    if (!ok) return false;

    if (!stbi_write_png(dst.c_str(), crop.rect.size, crop.rect.size, 4,
                        crop.rgba.data(), crop.rect.size * 4)) {
        error = "cannot write preprocessed image " + dst;
        return false;
    }
    return true;
}

} // namespace trellis
