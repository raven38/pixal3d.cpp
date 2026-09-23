#pragma once
#include "dual_grid.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trellis {
struct Pixal3dPostprocessOptions {
    int texture_size = 4096;
    int target_faces = 1000000;
    int remesh_band = 1;
    // narrow-band DC remesh のグリッド解像度。0 = デコード解像度 res をそのまま使う（native の既定）。
    // res=1024 の remesh は実測で 7.8M 頂点 / 15.6M 面を作り、その後の QEM と合わせて
    // wasm32 の 4 GiB ヒープに収まらない（ブラウザで std::bad_alloc）。ブラウザ側は
    // 512 を渡して同じ経路を粗いグリッドで回す。使った値は report に出る。
    int remesh_res = 0;
    // wasm32 host-heap budget for the whole tail in bytes (#83). 0 = unbounded (native default).
    // When set, remesh_res is the first rung of a deterministic ladder (remesh_res, 3/4, 1/2 of it):
    // the tail predicts each rung's host peak from the decoded mesh size before building anything
    // and starts at the first rung that fits; a std::bad_alloc inside remesh still steps down one
    // rung. Every decision is logged; running out of rungs (or memory in any other stage) fails
    // with an explicit HOST_HEAP_EXHAUSTED diagnostic instead of an unhandled std::bad_alloc.
    uint64_t host_budget_bytes = 0;
    bool use_xatlas = false;
    bool use_webp = false; // browser build defaults PNG; native may enable WebP.
};
// Host budget the browser (wasm32, -sMAXIMUM_MEMORY=4 GiB) drivers give the tail: the whole
// linear memory minus a 128 MiB margin for allocator slack / fragmentation (#83).
constexpr uint64_t kPixal3dWasm32TailBudget = (uint64_t)(4096 - 128) << 20;
// Reference-style MV tail: weld/hole fill -> narrow-band DC remesh -> QEM -> UV atlas
// -> voxel PBR bake -> textured GLB. pbr6 is [M,6] row-major in [0,1], coords at `res`.
bool pixal3d_write_production_glb(const std::string& out_glb, Mesh mesh,
                                  const std::vector<std::array<int,3>>& coords,
                                  const std::vector<float>& pbr6, int res,
                                  const Pixal3dPostprocessOptions& opt,
                                  std::string* report = nullptr);
}
