// Pixal3D SS-stage image conditioning: per-view DINOv3 features -> global (CLS+4reg,
// averaged over views) and proj (ProjGridMV-sampled patch tokens, averaged over views).
// Ports the "ss" stage of DinoV3ProjMultiViewFeatureExtractor
// (pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py) -- see
// docs/spec/30-pixal3d-cond.md sections 1-3. Also declares the SLAT-stage
// (shape/texture) variant, which additionally fuses in NAF high-res features
// (docs/spec/30-pixal3d-cond.md section 4) -- proj = [lr || hr], 2048-wide.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trellis {
struct Model;

// One posed input view. rgb_premult: torch [3,S,S] memory (== ggml [S,S,3,1]),
// values in [0,1], RGB already premultiplied by alpha (inference_mv.to_cond_tensor);
// NOT ImageNet-normalized -- pixal3d_cond_ss normalizes internally before DINOv3.
// c2w: row-major 4x4 camera-to-world (Blender/NeRF convention). View 0 must be the
// main/front view (compute_relative_calc_mat uses its distance as distance0).
struct Pixal3dView {
    std::vector<float> rgb_premult;
    float fov_x;
    float c2w[16];
};

// global: [n_global*1024] token-major (CLS, then 4 register tokens), averaged over views.
// proj:   [R^3 * d_proj] token-major (token k = ix*R*R + iy*R + iz), averaged over views.
struct Pixal3dCond {
    std::vector<float> global;
    std::vector<float> proj;
    int n_global = 5;
    int d_proj = 0;
};

// ImageNet-normalizes an alpha-premultiplied [0,1] torch [3,S,S] image into the tensor
// dinov3_encode expects (dinov3_encode itself does not normalize -- see src/dinov3.cpp /
// src/preprocess.cpp::normalize_cutout, which normalizes before calling it).
std::vector<float> pixal3d_imagenet_normalize(const std::vector<float>& rgb_premult, int S);

// Computes the Pixal3D SS-stage condition from V posed views (view 0 = main/front view).
// Per view: ImageNet-normalize -> dinov3_encode(S) -> split into CLS+4reg (global) and the
// [1024,Hp,Wp] patch map (Hp=Wp=S/16) -> proj_grid_sample(patch map, R, S,
// Camera{fov_x_v, mesh_scale, calc_mat_v}); calc_mat via mv_calc_mats(c2w, V, distance0),
// distance0 = norm(views[0].c2w[:3,3]). global and proj are averaged over views.
Pixal3dCond pixal3d_cond_ss(const Model& dinov3, const std::vector<Pixal3dView>& views,
                             int S, int R, float mesh_scale);

// Per-view / per-stage resource accounting for the device-resident conditioning path.
struct Pixal3dCondStats {
    size_t weight_bytes = 0;     // DINOv3 weight buffer (resident for the whole call)
    size_t cond_bytes = 0;       // persistent accumulators (global + proj) on the device
    size_t view_alloc_bytes = 0; // largest per-view graph buffer (DINOv3 activations + projection temporaries)
    size_t peak_bytes = 0;       // max over the graphs run of weight_bytes + persistent buffers resident at
                                 // that point + that graph's buffer (<= weight + cond + view_alloc)
    double total_ms = 0;         // wall time of the whole call
    double view_ms_max = 0;      // slowest single view (graph build + alloc + compute)
    int views = 0;
    // pixal3d_cond_slat_gpu preflight (#82): the largest single device tensor it plans to create
    // (uint64, computed before any allocation) and the backend's per-buffer limit
    // (ggml_backend_get_max_size: maxStorageBufferBindingSize on WebGPU). A plan above the limit
    // throws before allocating instead of surfacing as an allocator/OOM failure.
    uint64_t largest_planned_bytes = 0;
    uint64_t backend_max_buffer = 0;
};

// Device-resident variant of pixal3d_cond_ss with identical semantics: per view, one ggml graph
// runs DINOv3 -> pixel-aligned bilinear projection (proj_grid_bilinear_taps: get_rows x4 +
// weighted sum over the [1024, Hp*Wp] patch map, never leaving the device) -> running average
// into two persistent accumulators on the model's backend; the view's temporaries are released
// before the next view, so peak memory is independent of V. The fused global/proj are read back
// once at the end (same host layout as pixal3d_cond_ss). Numerics differ from the host path
// only by f32 (device) vs f64 (host) accumulation of the four bilinear taps.
Pixal3dCond pixal3d_cond_ss_gpu(const Model& dinov3, const std::vector<Pixal3dView>& views,
                                 int S, int R, float mesh_scale, Pixal3dCondStats* stats = nullptr);

// SLAT-stage (shape or texture) config: S = view image resolution (512 or 1024),
// R = ProjGrid resolution (32 or 64), naf_T = NAF target resolution (512 for the
// shape stages, 1024 for the texture stage; see docs/spec/30-pixal3d-cond.md
// section 2 table).
struct Pixal3dSlatCondParams {
    int S;
    int R;
    int naf_T;
    float mesh_scale;
    // NAF の neighborhood attention を何 block ずつ建てるか（0 = 自動）。NAF 出力
    // [1024, T*T] は T=1024 で 4 GiB になり WebGPU の maxBufferSize を超えるため、
    // 出力を block chunk に割って chunk ごとに projection tap を回収する
    // (docs/PIXAL3D_WEBGPU_MEMORY.md)。chunk は blocks 単位（1 block = 低解像度画素
    // 1つに対応する d x d = (T/h)^2 画素）。
    int naf_block_chunk = 0;
};

// Memory plan of one pixal3d_cond_slat_gpu call (#82), computed in uint64 before anything is
// allocated. tokens = active sparse tokens (sparse) or R^3 (dense). largest_bytes is the largest
// single device tensor among the accumulators and the known large intermediates (DINOv3 attention
// scores, NAF encoder / attention tensors) -- what a WebGPU maxStorageBufferBindingSize bounds;
// what names it. It is the up-front estimate; the exact per-node gate is check_graph_supported
// (ggml-webgpu's supports_op rejects any node or source above the binding size, before allocation).
struct Pixal3dCondSlatPlan {
    bool sparse = true, chunked = false;
    int64_t tokens = 0;
    uint64_t largest_bytes = 0;
    const char* what = "";
    uint64_t host_proj_bytes = 0;   // the [tokens, 2048] f32 host result
};
Pixal3dCondSlatPlan pixal3d_cond_slat_plan(const Pixal3dSlatCondParams& prm, int64_t tokens, bool sparse);

// Computes the Pixal3D SLAT-stage (shape/texture) fused condition from V posed
// views (view 0 = main/front view). Per view: ImageNet-normalize -> dinov3_encode(S)
// -> split into CLS+4reg (global) and the [1024,Hp,Wp] patch map (Hp=Wp=S/16, same
// split as pixal3d_cond_ss) -> lr = proj_grid_sample(patch map, R, S, calc_mat_v);
// hr = proj_grid_sample(naf_upsample(naf, views[v].rgb_premult (un-normalized,
// premultiplied), S, patch map, 1024, Hp, Wp, naf_T), R, S, calc_mat_v); proj_v =
// [lr || hr] (2048-wide per token). global and proj are averaged over views,
// accumulating one view at a time (never holds all V views' buffers at once).
Pixal3dCond pixal3d_cond_slat(const Model& dinov3, const Model& naf,
                               const std::vector<Pixal3dView>& views,
                               const Pixal3dSlatCondParams& prm);

// Device-resident variant of pixal3d_cond_slat with identical semantics (the WebGPU/WASM path):
// per view, ONE ggml graph on the DINOv3 model's backend runs DINOv3 -> NAF (naf_build, the
// all-ggml upsampler, fed the DINOv3 patch map straight from the same graph) -> the two
// pixel-aligned bilinear projections (get_rows x4 over the [1024, Hp*Wp] patch map for lr and
// over the [1024, T*T] NAF map for hr, tap indices remapped to naf_block_order) -> running
// averages into three persistent accumulators (global, lr, hr). The view's temporaries are
// released before the next view; the fused condition is read back once at the end and
// interleaved to the [R^3, 2048] host layout. `naf` must be loaded on the same device as
// `dinov3` (its weights are read by the DINOv3 backend's graph).
// `coords` を渡すと dense な R^3 グリッドを一切作らず、その疎な active voxel
// 集合ぶんだけを蓄積する（返る proj は [coords.size(), 2048]、token 順は coords の順。
// pixal3d_gather_proj を後から呼んではいけない)。texture 段 (S=1024,R=64,naf_T=1024) は
// dense だと accumulator だけで 2 GiB、NAF 出力が 4 GiB になるため、この経路が必須。
// nullptr なら従来どおり dense [R^3, 2048] を返す。
Pixal3dCond pixal3d_cond_slat_gpu(const Model& dinov3, const Model& naf,
                                   const std::vector<Pixal3dView>& views,
                                   const Pixal3dSlatCondParams& prm, Pixal3dCondStats* stats = nullptr,
                                   const std::vector<std::array<int, 3>>* coords = nullptr);

// Gathers the dense SLAT proj condition [R^3, C] (token k = x*R*R + y*R + z, as
// produced by pixal3d_cond_slat) at a sparse set of active voxel coords (x,y,z),
// producing the [N, C] layout the SLAT DiT's SparseTensor feats consume.
std::vector<float> pixal3d_gather_proj(const std::vector<float>& proj_dense, int R, int C,
                                        const std::vector<std::array<int, 3>>& coords);

} // namespace trellis
