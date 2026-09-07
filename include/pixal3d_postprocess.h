#pragma once
#include "dual_grid.h"
#include <array>
#include <string>
#include <vector>

namespace trellis {
struct Pixal3dPostprocessOptions {
    int texture_size = 4096;
    int target_faces = 1000000;
    int remesh_band = 1;
    bool use_xatlas = false;
    bool use_webp = false; // browser build defaults PNG; native may enable WebP.
};
// Reference-style MV tail: weld/hole fill -> narrow-band DC remesh -> QEM -> UV atlas
// -> voxel PBR bake -> textured GLB. pbr6 is [M,6] row-major in [0,1], coords at `res`.
bool pixal3d_write_production_glb(const std::string& out_glb, Mesh mesh,
                                  const std::vector<std::array<int,3>>& coords,
                                  const std::vector<float>& pbr6, int res,
                                  const Pixal3dPostprocessOptions& opt,
                                  std::string* report = nullptr);
}
