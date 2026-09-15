#include "camera_estimator.h"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace trellis;

static int failures = 0;
static void check(bool ok, const char* label) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", label);
    if (!ok) ++failures;
}

int main() {
    constexpr float fov = 0.3490658503988659f;
    constexpr float expected_distance = 2.8356409f;
    check(std::fabs(single_view_distance_from_fov(fov, 1.0f) - expected_distance) < 1e-5f,
          "official single-view distance equation");

    TransformsFile tf;
    std::string err;
    check(synthesize_single_view_camera("front.png", fov, 1.0f, tf, err),
          "single-view metadata synthesis");
    check(tf.frames.size() == 1, "one frame emitted");
    if (tf.frames.size() == 1)
        check(std::fabs(tf.frames[0].transform_matrix[7] + expected_distance) < 1e-5f,
              "camera translation uses estimated distance");
    check(!synthesize_single_view_camera("front.png", 0.0f, 1.0f, tf, err), "zero FOV rejected");
    check(!synthesize_single_view_camera("front.png", 0.3f, 0.0f, tf, err), "zero scale rejected");

    const auto dir = std::filesystem::temp_directory_path() / "pixal3d_estimator_contract";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    { std::ofstream f((dir / "front.png").string()); f << "fixture"; }
    synthesize_single_view_camera("front.png", fov, 1.0f, tf, err);
    check(validate_estimated_transforms(dir.string(), 1.0f, tf, err), "matching estimate accepted");
    check(!validate_estimated_transforms(dir.string(), 0.5f, tf, err), "scale mismatch rejected");
    tf.frames[0].file_path = "other.png";
    check(!validate_estimated_transforms(dir.string(), 1.0f, tf, err), "frame mismatch rejected");
    std::filesystem::remove_all(dir, ec);

    return failures ? 1 : 0;
}
