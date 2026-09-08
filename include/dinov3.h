// DINOv3 ViT-L/16 image conditioner -> cross-attention tokens [N, 1024].
#pragma once
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace trellis {
struct Model;

// chw: preprocessed image, torch [3,S,S] memory (== ggml [S,S,3,1]); S = 512 (or 1024).
// Returns conditioning tokens [Ntok * 1024], Ntok = (S/16)^2 + 5 (cls + 4 reg + patches),
// channel-major per token (ggml [1024, Ntok] read out as token-major [Ntok,1024]).
std::vector<float> dinov3_encode(const Model& m, const std::vector<float>& chw, int S);

// Graph-building half of dinov3_encode, for callers that want to keep the token map on the
// device and consume it in the same graph (Pixal3D GPU conditioning, src/pixal3d_cond_gpu.cpp).
// Creates the three input tensors in `c` (flagged ggml_set_input; the caller uploads them after
// allocation: img = the [S,S,3,1] image, cos/sin = the [64,1,Ntok] RoPE tables from
// dinov3_rope_tables) and returns the final-LN output [1024, Ntok] (channel-major).
struct Dinov3Inputs {
    ggml_tensor* img; ggml_tensor* cos; ggml_tensor* sin; int ntok;
    // TRELLIS_DBG_DINOV3_LAYERS=<dir> のときだけ使う。各ブロック出力の先頭 5 トークン
    // （cls + register 4 本）を層順に受け取り、PyTorch との層別突き合わせに使う。
    std::vector<ggml_tensor*>* layers = nullptr;
};
ggml_tensor* dinov3_build(ggml_context* c, const Model& m, int S, Dinov3Inputs& in);

// Host RoPE tables for dinov3_build: [Ntok*64] each (prefix tokens identity, patches 2D
// half-split), Ntok = (S/16)^2 + 5.
void dinov3_rope_tables(int S, std::vector<float>& rcos, std::vector<float>& rsin);

} // namespace trellis
