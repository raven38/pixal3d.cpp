#include "pixal3d_cascade.h"

#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>

namespace trellis {

bool pixal3d_ss_res_supported(int ss_res) { return ss_res == 32 || ss_res == 64; }

std::vector<std::array<int,3>> pixal3d_quantize_hr_coords(const std::vector<std::array<int,3>>& upsampled,
                                                         int ss_res, int hr_res) {
    if (!pixal3d_ss_res_supported(ss_res))
        throw std::runtime_error("pixal3d_quantize_hr_coords: unsupported ss_res " + std::to_string(ss_res));
    const int grid = hr_res / 16;
    if (hr_res % 16 != 0 || grid < 2)
        throw std::runtime_error("pixal3d_quantize_hr_coords: bad hr_res " + std::to_string(hr_res));
    const int span = pixal3d_ss_source_span(ss_res);
    const float fspan = (float)span, gm1 = (float)(grid - 1);
    std::set<std::array<int,3>> q;
    for (const auto& c : upsampled) {
        std::array<int,3> o;
        for (int a = 0; a < 3; ++a) {
            if (c[a] < 0 || c[a] >= span)
                throw std::runtime_error("pixal3d_quantize_hr_coords: upsampled coord " + std::to_string(c[a]) +
                                         " outside the ss_res=" + std::to_string(ss_res) + " span [0," +
                                         std::to_string(span) + ") -- ss_res does not match the LR coords");
            o[a] = (int)std::lround((c[a] + 0.5f) / fspan * gm1);
            if (o[a] < 0 || o[a] >= grid)
                throw std::runtime_error("pixal3d_quantize_hr_coords: quantized coord outside the HR grid");
        }
        q.insert(o);
    }
    return std::vector<std::array<int,3>>(q.begin(), q.end());
}

Pixal3dCascadeCoords pixal3d_cascade_select(const std::vector<std::array<int,3>>& upsampled,
                                            int ss_res, int hr_res, int max_tokens, bool verbose) {
    Pixal3dCascadeCoords r;
    for (;;) {
        r.coords = pixal3d_quantize_hr_coords(upsampled, ss_res, hr_res);
        if ((int)r.coords.size() < max_tokens || hr_res == 1024) break;
        if (verbose)
            printf("      res%d (grid %d) -> %d tokens >= %d, backing off -128\n",
                   hr_res, hr_res / 16, (int)r.coords.size(), max_tokens);
        hr_res -= 128;
    }
    r.hr_res = hr_res;
    r.grid = hr_res / 16;
    return r;
}

} // namespace trellis
