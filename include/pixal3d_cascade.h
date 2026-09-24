#pragma once
// Pixal3D LR -> HR cascade coordinate handling, parameterized by the sparse-structure
// resolution (ss_res). Pixal3DImageTo3DPipeline.run() hardcodes ss_res=32 and quantizes the
// shape decoder's upsampled coords with `round((c + 0.5) / 512 * (grid - 1))`; 512 there is
// the upsample span ss_res * 16 (4 x2 upsample stages), which only equals the literal 512 at
// ss_res=32. Dividing an ss_res=64 span (1024) by 512 doubles every HR coord (~128-cell grid at
// res 1024) -- visualbruno/ComfyUI-Trellis2#193. See docs/spec/33-pixal3d-ss-res.md.
#include <array>
#include <vector>

namespace trellis {

// Sparse-structure resolutions the Pixal3D cascade supports (32 = reference default).
bool pixal3d_ss_res_supported(int ss_res);

// Span of shape_upsample() coords for LR coords on an ss_res^3 grid (16x upsample).
inline int pixal3d_ss_source_span(int ss_res) { return ss_res * 16; }

// Quantizes decoder-upsampled coords (each axis in [0, ss_res*16)) onto the HR flow grid
// hr_res/16 with Pixal3D's own formula: lround((c + 0.5) / (ss_res*16) * (grid - 1)), then
// sorts + de-duplicates (== torch unique(dim=0)). For ss_res=32 this is bit-identical to the
// former literal-512 expression. Throws if an input coord lies outside the source span (the
// caller passed the wrong ss_res) or an output coord leaves [0, grid).
std::vector<std::array<int,3>> pixal3d_quantize_hr_coords(const std::vector<std::array<int,3>>& upsampled,
                                                         int ss_res, int hr_res);

struct Pixal3dCascadeCoords {
    int hr_res = 0;                               // selected HR resolution (after backoff)
    int grid = 0;                                 // hr_res / 16 -- the model grid, never max(coord)+1
    std::vector<std::array<int,3>> coords;        // quantized, sorted, unique
};

// run()'s token-budget loop: quantize at hr_res; while the token count reaches max_tokens and
// hr_res != 1024, back off by 128 (so only a 1536 cascade ever backs off). Prints one line per
// tried resolution when verbose.
Pixal3dCascadeCoords pixal3d_cascade_select(const std::vector<std::array<int,3>>& upsampled,
                                            int ss_res, int hr_res, int max_tokens, bool verbose = false);

} // namespace trellis
