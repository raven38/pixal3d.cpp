// Pixal3D の pre-matted RGBA 前処理（公式 pipeline.py の preprocess_image 相当）を共有 C++ に
// 置いたもの。これまで tools/run_pixal3d_estimated.py と web/app/single_view.js に二重実装
// されていて native には無く、単一画像から SV 推論するには Python が必要だった。
//
// 手順は docs/design/2026-09-19-desktop-0.10.0-sv-parity.md D2 で凍結した以下のとおり:
//   1. RGBA へデコードする
//   2. 250 未満の alpha が 1 画素も無ければ拒否（pre-matted でない）。全画素 alpha=0 も拒否
//      ※ しきい値 250 は src/pixal3d_input.cpp の load_rgba・web/app/single_view.js と同じ。
//         255 にすると、ここを通った画像を同じバイナリの pixal3d_load_input_views が
//         「has no real alpha channel」で弾く自己矛盾が起きる
//   3. scale = min(1, 1024/max(w,h))。scale < 1 のときだけ画像全体を縮小する
//      （w,h は max(1, floor(w*scale))）
//   4. alpha > 0.8*255 のマスクの bbox（両端含む）。空なら拒否
//   5. cx=(x0+x1)/2, cy=(y0+y1)/2, span=max(x1-x0, y1-y0)
//   6. crop_size = max(2, floor(span*1.1)) -> half = max(1, crop_size/2) -> crop_size = 2*half
//   7. left = floor(cx-half), top = floor(cy-half) の正方。画像外は透明 (0,0,0,0) で埋める
//   8. 出力の 1 辺は 1024 を超えうる（最大 1124。公式 Belle 入力が実際に 1124^2 になる）
//
// 既知の差分: alpha の拒否判定を Python は縮小前・JS は縮小後に行う。ここは Python と同じく
// 縮小前に判定する（縮小で透明度が平均化されて判定が変わるのを避けるため）。bbox は 3 実装とも
// 縮小後に取る。リサンプラは Python=LANCZOS / JS=canvas / ここ=stb_image_resize なので画素値は
// 一致しない。一致検査（D9）はクロップ矩形と出力サイズを完全一致、画素は許容差つきで見る。
#pragma once
#include <string>
#include <vector>

namespace trellis {

// 手順 7 のクロップ矩形。座標系は「手順 3 の縮小後」の画像。
struct CropRect {
    int left = 0;
    int top  = 0;
    int size = 0;   // 正方なので 1 辺
};

struct PrematteCrop {
    std::vector<unsigned char> rgba;   // size*size*4、RGBA8（premultiply はしない）
    CropRect rect{};
    int resized_w = 0;                 // 手順 3 の後の寸法（縮小が起きなければ入力のまま）
    int resized_h = 0;
};

// メモリ上の RGBA8（`w*h*4` バイト、行優先）へ手順 2〜7 を適用する。
// 戻り値 false のとき `error` に理由が入る（fail closed。既定値へ落とさない）。
bool preprocess_prematted_rgba(const unsigned char* rgba, int w, int h,
                               PrematteCrop& out, std::string& error);

// `src` を読み、上と同じ処理をして `dst` へ PNG(RGBA) で書く。
bool preprocess_prematted_rgba_file(const std::string& src, const std::string& dst,
                                    std::string& error);

} // namespace trellis
