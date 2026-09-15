#pragma once
#include "pixal3d_cond.h"
#include <string>
#include <vector>

namespace trellis {
struct Pixal3dInputViews {
    std::vector<Pixal3dView> views512;
    std::vector<Pixal3dView> views1024;
    float mesh_scale = 1.0f;
};
// Load <dir>/transforms.json and pre-matted RGBA frame files, producing the exact host-side
// Pixal3dView representation used by the MV pipeline. Works with native files and WORKERFS.
// Returns false and fills error on malformed JSON, missing FOV/image, or opaque/non-alpha input.
//
// transforms.json が無いディレクトリでは、mesh_scale を明示した場合に限り canonical rig
// （4 視点ターンテーブル、transforms_json.h 参照）で合成する。mesh_scale_set=false の
// 既定では従来どおり transforms.json が必須。
bool pixal3d_load_input_views(const std::string& dir, Pixal3dInputViews& out,
                              std::string& error, int max_views = 0,
                              float mesh_scale = 0.0f, bool mesh_scale_set = false);
}
