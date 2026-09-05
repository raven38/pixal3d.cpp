// Minimal tolerant JSON parser for Pixal3D multiview `transforms.json` (Blender/NeRF-style:
// top-level camera_angle_x/mesh_scale + frames[] with file_path/transform_matrix/optional
// per-frame camera_angle_x). Not a general-purpose JSON library -- there was none in this repo
// (grepped trellis-server.cpp/trellis_model.cpp: only ad-hoc JSON *writers* exist, e.g.
// mesh_glb.cpp's glTF chunk and trellis-server.cpp's error responses) -- so this is a small,
// self-contained recursive-descent reader good enough for that one file shape.
#pragma once
#include <string>
#include <vector>

namespace trellis {

struct TransformsFrame {
    std::string file_path;
    float transform_matrix[16] = {};   // row-major 4x4 camera-to-world (Blender/NeRF convention)
    float camera_angle_x = 0.0f;       // per-frame horizontal FOV override, radians
    bool  has_camera_angle_x = false;
};

struct TransformsFile {
    std::vector<TransformsFrame> frames;
    float camera_angle_x = 0.0f;   // top-level fallback FOV, radians
    bool  has_camera_angle_x = false;
    float mesh_scale = 1.0f;
};

// Parses `path` into `out`. Returns false (with an stderr message describing what's wrong)
// on a missing file, malformed JSON, or a frame missing file_path/transform_matrix.
bool load_transforms_json(const std::string& path, TransformsFile& out);

} // namespace trellis
