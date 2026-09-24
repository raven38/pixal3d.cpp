// Sparse submanifold 3D conv + ConvNeXt block (TRELLIS.2 shape/tex VAE).
#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <string>

struct ggml_context;
struct ggml_tensor;

namespace trellis {
struct Model;

// Neighbor table for submanifold 3x3x3 conv, TAP-MAJOR int32 [27*N]:
//   nbr[t*N + i] = row index of the voxel at coords[i] + offset(t), or N if absent.
//   tap t = kd*9 + kh*3 + kw, offset = (kd-1, kh-1, kw-1) on (x,y,z). (N = sentinel zero row.)
std::vector<int32_t> build_neighbor_table(const std::vector<std::array<int,3>>& coords);

// Graph ops (feats are channel-major ggml [C, N]; nbr is i32 ggml [N, 27]).
// Submanifold conv: weight key `prefix.weight` ggml ne [Ci,27,Co], bias `prefix.bias` [Co] -> [Co,N].
ggml_tensor* sparse_submconv(ggml_context* c, const Model& m, const std::string& prefix,
                             ggml_tensor* feats, ggml_tensor* nbr, int N);
// SparseConvNeXtBlock3d: conv -> rowLN(affine,1e-6) -> Linear(C,4C)->SiLU->Linear(4C,C) -> + input.
ggml_tensor* sparse_convnext(ggml_context* c, const Model& m, const std::string& prefix,
                             ggml_tensor* feats, ggml_tensor* nbr, int N);

// SparseResBlockC2S3d up-block (channel->spatial ×2). Host-orchestrated (subdiv readback).
// feats_in: [Cin*N] channel-major; returns new feats [Cout*M] + new coords (res ×2).
// ext_subdiv: if non-null, use this [8*N] binarized mask (tex decoder guide_subs); else predict
// via to_subdiv (shape decoder). The returned `subdiv` is the [8*N] mask actually used.
struct C2SResult { std::vector<float> feats; std::vector<std::array<int,3>> coords; int C = 0; std::vector<uint8_t> subdiv; };

// デコーダ最終段のヘッド（LayerNorm(affine 無し) + Linear）を c2s のグラフに融合するための指定。
// 最終 C2S の出力 [Cout, M] は res-1024 のテクスチャデコーダで M=4.76M / Cout=64、つまり
// 1.22 GB になる。従来はこれを host へ読み戻してから output_layer を掛けていたので、wasm32 の
// 4 GiB ヒープでは読み戻しの時点で std::bad_alloc になっていた。ヘッドを渡すと chunk ごとに
// [Cout, nr] → LN → Linear → [out_ch, nr] まで進めて 64ch の chunk を捨てるので、device 上に
// [Cout, M] を作らず、host が受け取るのも [out_ch, M]（6ch なら 114 MB）だけになる。
// 数値は linear_rows(pre_norm=true) と同じ（eps 1e-5・affine 無しの norm → mul_mat → bias）で、
// chunk 幅も同じ 1M 列上限なので分割の切れ目も一致する。
struct C2SFinalHead {
    std::string prefix;      // 重みは m.get(prefix + ".weight") / ".bias"
    int out_ch = 0;
    bool pre_norm = true;    // LayerNorm(affine 無し, eps 1e-5) を先に掛ける
};
// `tag`: label for graph_dump.h (TRELLIS_DUMP_OPS), identifying the caller's stage; purely diagnostic.
// `head`: 非 null なら上記の融合を行い、返る feats は [out_ch * M]・C は out_ch になる。
// `coords_only`: 分割マスクと新しい coords だけを返し、conv1/conv2（feats [Cout, M] と M 個分の
// 近傍表）を作らない。coords はマスクだけで決まるので結果は同じ。cascade upsample の最終段用
// （ss_res=64 では M ≈ 4.8M、捨てる feats だけで 1.2 GB、wasm32 のヒープを 4 GiB 近くまで押し上げていた）。
C2SResult sparse_c2s(const Model& m, const std::string& prefix,
                     const std::vector<float>& feats_in, int Cin,
                     const std::vector<std::array<int,3>>& coords, int Cout,
                     const std::vector<uint8_t>* ext_subdiv = nullptr,
                     const char* tag = "c2s",
                     const C2SFinalHead* head = nullptr,
                     bool coords_only = false);

} // namespace trellis
