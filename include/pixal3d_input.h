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
bool pixal3d_load_input_views(const std::string& dir, Pixal3dInputViews& out,
                              std::string& error, int max_views = 0);
}
