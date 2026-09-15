// transforms.json 無し入力（canonical rig）の回帰テスト。
// web/app/test_canonical_rig.mjs の項目 1・2・3 を native 側で行う:
//   1. 同梱 transforms.json を持つ views dir の 4 枚から mesh_scale を合わせて合成すると、
//      parse 結果と全フィールド（fov / mesh_scale / file_path 順 / 16 要素）がビット一致する
//   2. mesh_scale を明示しないと合成は失敗する（1.0 に落ちない）
//   3. 4 枚以外は拒否される。transforms.json が壊れているときは合成へ落ちない
//   4. 自然順が "view2" < "view10"、"view02" < "view2" になる
//
//   ./build/trellis-test-transforms-synth <views_dir>
#include "transforms_json.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace trellis;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <views_dir with transforms.json>\n", argv[0]); return 2; }
    const std::string dir = argv[1];

    TransformsFile ref;
    if (!load_transforms_json(dir + "/transforms.json", ref)) { fprintf(stderr, "cannot parse reference transforms.json\n"); return 2; }

    // 1. 同じ 4 枚から合成した rig が、同梱 transforms.json と完全に一致する
    std::vector<std::string> images = list_view_images(dir);
    TransformsFile syn;
    std::string err;
    const bool synth_ok = synthesize_canonical_rig(images, ref.mesh_scale, syn, err);
    check(synth_ok, "synthesize_canonical_rig on the bundled views");
    if (synth_ok) {
        check(syn.frames.size() == ref.frames.size(), "frame count matches");
        check(syn.mesh_scale == ref.mesh_scale, "mesh_scale matches");
        const float ref_fov = ref.has_camera_angle_x ? ref.camera_angle_x : 0.0f;
        check(syn.has_camera_angle_x && syn.camera_angle_x == ref_fov, "camera_angle_x matches bit-for-bit");
        bool paths_ok = true, mats_ok = true;
        for (size_t i = 0; i < ref.frames.size() && i < syn.frames.size(); ++i) {
            if (syn.frames[i].file_path != ref.frames[i].file_path) paths_ok = false;
            if (memcmp(syn.frames[i].transform_matrix, ref.frames[i].transform_matrix, 16 * sizeof(float)) != 0) {
                mats_ok = false;
                printf("     frame %zu (%s) differs\n", i, ref.frames[i].file_path.c_str());
                for (int k = 0; k < 16; ++k) printf("       [%2d] ref %.17g  syn %.17g\n", k,
                        (double)ref.frames[i].transform_matrix[k], (double)syn.frames[i].transform_matrix[k]);
            }
        }
        check(paths_ok, "file_path order matches the bundled frame order");
        check(mats_ok, "all 4 transform matrices match bit-for-bit");
    }

    // 2. mesh_scale を明示しないと合成しない
    TransformsFile t2;
    check(!synthesize_canonical_rig(images, 0.0f, t2, err), "mesh_scale=0 is rejected");
    check(!synthesize_canonical_rig(images, -1.0f, t2, err), "negative mesh_scale is rejected");

    // 3. 4 枚以外は拒否
    std::vector<std::string> three(images.begin(), images.begin() + (images.size() > 3 ? 3 : images.size()));
    check(!synthesize_canonical_rig(three, 1.0f, t2, err), "3 views are rejected");
    std::vector<std::string> five = images; five.push_back("extra.png");
    check(!synthesize_canonical_rig(five, 1.0f, t2, err), "5 views are rejected");

    // transforms.json があるディレクトリでは、mesh_scale 未指定でも従来どおり parse される
    TransformsFile t3;
    check(load_views_metadata(dir, 0.0f, false, t3, err), "existing transforms.json still loads without --mesh-scale");
    check(load_views_metadata(dir, 0.5f, true, t3, err) && t3.mesh_scale == 0.5f, "--mesh-scale overrides an existing transforms.json");

    // 壊れた transforms.json は合成へ落ちない
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "pixal3d_synth_test";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp, ec);
    for (const std::string& n : images) std::filesystem::copy_file(dir + "/" + n, tmp / n, ec);
    { std::ofstream bad((tmp / "transforms.json").string()); bad << "{ this is not json"; }
    TransformsFile t4;
    check(!load_views_metadata(tmp.string(), 1.0f, true, t4, err), "a malformed transforms.json fails instead of synthesizing");
    std::filesystem::remove(tmp / "transforms.json", ec);
    check(load_views_metadata(tmp.string(), 1.0f, true, t4, err), "the same dir without transforms.json synthesizes");
    std::filesystem::remove_all(tmp, ec);

    // 走査できないディレクトリは「画像 0 枚」ではなくエラーとして扱う（fail closed）
    {
        const std::filesystem::path locked = std::filesystem::temp_directory_path() / "pixal3d_synth_locked";
        std::error_code lec;
        std::filesystem::remove_all(locked, lec);
        std::filesystem::create_directories(locked, lec);
        std::filesystem::permissions(locked, std::filesystem::perms::none, lec);
        std::string list_err;
        const std::vector<std::string> none = list_view_images(locked.string(), &list_err);
        // root で実行すると権限が効かないので、その場合だけ検査を飛ばす
        const bool enforced = !list_err.empty();
        check(!enforced || none.empty(), "an unreadable dir reports an error instead of 0 images");
        TransformsFile t5;
        check(!enforced || !load_views_metadata(locked.string(), 1.0f, true, t5, err),
              "an unreadable dir does not synthesize");
        std::filesystem::permissions(locked, std::filesystem::perms::owner_all, lec);
        std::filesystem::remove_all(locked, lec);
    }

    // transforms.json という名前のディレクトリは「存在する」ので合成へ落とさない
    {
        const std::filesystem::path d2 = std::filesystem::temp_directory_path() / "pixal3d_synth_dirjson";
        std::error_code dec;
        std::filesystem::remove_all(d2, dec);
        std::filesystem::create_directories(d2 / "transforms.json", dec);
        for (const std::string& n : images) std::filesystem::copy_file(dir + "/" + n, d2 / n, dec);
        TransformsFile t6;
        check(!load_views_metadata(d2.string(), 1.0f, true, t6, err),
              "a directory named transforms.json is an error, not an absence");
        std::filesystem::remove_all(d2, dec);
    }

    // 4. 自然順
    check(natural_name_less("view2.png", "view10.png"), "view2 < view10");
    check(!natural_name_less("view10.png", "view2.png"), "view10 > view2");
    check(natural_name_less("view02.png", "view2.png"), "view02 < view2 (byte tie-break)");
    check(natural_name_less("a.png", "b.png"), "plain lexicographic still holds");

    printf("%s (%d failure(s))\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
