#include "sparse.h"
#include "trellis_model.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    trellis::Model model;
    const std::vector<float> feats(8, 0.0f);
    const std::vector<std::array<int, 3>> coords = {{0, 0, 0}};
    const std::vector<uint8_t> subdiv(8, 0);

    try {
        (void)trellis::sparse_c2s(model, "test", feats, 8, coords, 8, &subdiv);
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        if (message.find("no active voxels") != std::string::npos) {
            std::printf("PASS: %s\n", e.what());
            return 0;
        }
        std::fprintf(stderr, "unexpected error: %s\n", e.what());
        return 1;
    }

    std::fprintf(stderr, "expected an empty-subdivision error\n");
    return 1;
}
