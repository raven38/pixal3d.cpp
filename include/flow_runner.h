// Flow-DiT runner (dense grid or sparse voxel) + FlowEuler guidance-interval sampler.
#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <functional>
#include <map>
#include <string>
#include "dit.h"

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
typedef struct ggml_gallocr* ggml_gallocr_t;

namespace trellis {
struct Model;

struct SamplerParams {
    int    steps             = 12;
    float  guidance_strength = 7.5f;
    float  guidance_rescale  = 0.0f;
    // gi0/gi1/rescale_t は double: 参照実装（trellis2/pipelines/samplers/flow_euler.py）は
    // t-schedule を numpy float64 で組み、guidance interval の判定も double で行い、モデルに
    // 渡す直前（1000*t）だけ float32 に narrow する。float だと 0.6f が
    // 0.600000023841857910156250 へ widen され、guidance interval の厳密な小数境界（t==0.6）で
    // 参照実装とずれる（fix(flow) が修正した境界バグ）。呼び出し側は `0.6` のように f サフィックス
    // なしのリテラルで代入すること。
    double gi0               = 0.6;
    double gi1               = 1.0;
    double rescale_t         = 1.0;
    float  sigma_min         = 1e-5f;
};

// numpy `np.linspace(1, 0, steps+1)` + rescale を逐語再現する t-schedule（flow_runner.cpp）。
// sample_flow / sample_flow_multi 共有の単一の真実源。steps<=0 は std::invalid_argument。
std::vector<double> flow_t_schedule(int steps, double rescale_t);
inline bool flow_in_guidance_interval(double t, double gi0, double gi1) { return gi0 <= t && t <= gi1; }

// Sparse-structure (SS) flow stage の本番 SamplerParams（trellis_cli.cpp の trellis_run() /
// trellis_run_mv() 両方の [SS] ブロックが使う値。steps=12, guidance_rescale=0.7, gi=[0.6,1.0],
// rescale_t=5.0）を単一の真実源として公開する（TASK-PORT F2: 以前は trellis_cli.cpp とテストが
// 同じリテラルを別々に書いていて「本番値と一致しているか」が目視確認頼みだった。本番側もこの
// 関数を呼ぶようにし、本番定数がドリフトしたらテスト側の呼び出しも自動的に追随する設計にする）。
// guidance_strength だけは --gss で CLI 上書き可能なので引数化する（既定 7.5、TrellisParams::gss
// の既定値と同じ）。
SamplerParams ss_production_sampler_params(float guidance_strength = 7.5f);

// One DiT graph (built once for a fixed token count N), re-run per sampler step.
// Token axis N = R^3 (dense) or number of active voxels (sparse); RoPE tables are
// supplied by the factory (grid index math vs real voxel coords).
class DitRunner {
public:
    DitRunner(const Model& m, const DiTParams& p, int N, int n_cond,
              const std::vector<float>& rope_cos, const std::vector<float>& rope_sin);
    ~DitRunner();
    // xt: [in_ch*N] channel-major. cond: [d_cond*n_cond]. Returns velocity [out_ch*N].
    // proj: [d_proj*N] Pixal3D proj_cond, required iff p.proj_attn (ignored otherwise).
    std::vector<float> forward(const std::vector<float>& xt, float t_scaled, const float* cond,
                               const float* proj = nullptr);
    int N() const { return N_; }
    // Bytes of the gallocr-owned activation/temporary buffer for one forward (weights excluded).
    size_t alloc_bytes() const { return alloc_bytes_; }
private:
    // 組んだグラフがこのデバイスの予算に収まるかを確保直後に判定し、超えていたら投げる。
    // 超えたまま走らせるとユニファイドメモリ上でスラッシングしてマシンごと固まるため、
    // 「走らせてから固まる」のではなく走らせる前に落とす。TRELLIS_ALLOW_OVER_BUDGET=1 で無効化。
    void check_device_budget() const;
    const Model& m_; DiTParams p_; int N_, Lc_;
    ggml_context* ctx_ = nullptr; ggml_cgraph* g_ = nullptr; ggml_gallocr_t alloc_ = nullptr;
    ggml_tensor *gh0_, *gtf_, *gcond_, *gcos_, *gsin_, *gout_, *gproj_ = nullptr;
    std::vector<float> rcos_, rsin_;   // re-uploaded each forward (gallocr may reuse input buffers)
    size_t alloc_bytes_ = 0;
    std::map<std::string, ggml_tensor*> inter_;   // [dbg] named intermediates for NaN localization
    bool dbg_nan_ = false, dbg_done_ = false;
    // --profile: forward counter (the profiling passes run after forward #1, once the backend's
    // lazy pipeline compiles on #0 are done) and the whole-graph time of every forward so far.
    int fwd_count_ = 0;
    std::vector<double> whole_s_;
    void profile_forward(const std::function<void()>& reupload);   // --profile side passes (flow_runner.cpp)
};

// Dense factory: RoPE from R^3 grid (ij meshgrid, z fastest). N = R^3.
DitRunner* make_dense_runner(const Model& m, const DiTParams& p, int R, int n_cond);
// Sparse factory: RoPE from real voxel coords [N][3]. N = coords.size().
DitRunner* make_sparse_runner(const Model& m, const DiTParams& p,
                              const std::vector<std::array<int,3>>& coords, int n_cond);

// FlowEuler guidance-interval sampler over an arbitrary forward functor.
using FlowFwd = std::function<std::vector<float>(const std::vector<float>&, float, const float*)>;
std::vector<float> sample_flow(const FlowFwd& fwd, std::vector<float> sample,
                               const float* cond, const float* neg_cond,
                               const SamplerParams& sp,
                               std::vector<std::vector<float>>* trace = nullptr);

enum class MultiCondMode {
    Stochastic,
    MultiDiffusion,
};

// TRELLIS / TRELLIS.2 experimental multi-image sampler semantics.
// Conditions are an ordered bank of same-shaped image-conditioning tensors.
// - Stochastic: one positive condition per sampler step, cycling k % V.
// - MultiDiffusion: evaluate every positive condition at the same x_t/t, average the
//   raw velocity predictions, then apply one negative prediction + CFG/rescale.
// V=1 is numerically equivalent to the single-condition sampler (apart from the
// reference PR's intentionally preserved extra negative forward for multidiffusion
// when guidance_strength==1 inside the guidance interval).
std::vector<float> sample_flow_multi(const FlowFwd& fwd, std::vector<float> sample,
                                     const std::vector<const float*>& conds,
                                     const float* neg_cond,
                                     const SamplerParams& sp,
                                     MultiCondMode mode,
                                     std::vector<std::vector<float>>* trace = nullptr,
                                     int* stochastic_counter = nullptr);


// Pixal3D ProjectAttention variant: the forward functor also takes the proj_cond pointer, threaded
// from `proj`/`neg_proj` the same way `cond`/`neg_cond` are. The FlowFwd overload above wraps this
// one with proj = nullptr. neg_proj is typically all-zeros (proj_linear(0) = bias).
using FlowFwdProj = std::function<std::vector<float>(const std::vector<float>&, float,
                                                      const float*, const float*)>;
std::vector<float> sample_flow(const FlowFwdProj& fwd, std::vector<float> sample,
                               const float* cond, const float* neg_cond,
                               const float* proj, const float* neg_proj,
                               const SamplerParams& sp,
                               std::vector<std::vector<float>>* trace = nullptr);

} // namespace trellis
