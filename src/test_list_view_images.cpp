// TASK-PORT P5: trellis::list_view_images() 単体テスト（自己完結、外部fixture不要）。
//
// 背景: 旧トラック（~/Downloads/pixal3d-mv-60、commit e92311a）は専用関数
// trellis2_mv_list_images() の「1枚ディレクトリを拒否しない」バグを直したが、このリポジトリ
// (PR #71) には trellis2_mv_list_images() 自体が存在しない。下限・上限チェックは
// trellis::list_view_images()（列挙・natural sort・隠しファイル/非画像除外のみを担当、下限上限は
// 持たない）+ trellis_cli.cpp 側の `names.size() < 2 || names.size() > 8` チェックという構成に
// なっており、このバグは既に存在しない（コード確認済み、docs/design/
// 2026-09-21-trellis2-mv-port-hardening.md §P5参照）。
//
// 2..8境界は実CLIを叩く tests/trellis2_mv_cli_contract.sh 側に追加した（list_view_images単体では
// 検証できない、本番の閾値そのものを検証するため）。ここでは list_view_images() 自身の契約
// （natural sort・隠しファイル/非画像除外・決定性）のうち、既存の trellis-test-transforms-synth
// （natural_name_less の文字列比較、権限エラーのfail-closed）がカバーしていない観点だけを足す。
#include "transforms_json.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace trellis;
namespace fs = std::filesystem;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

static void touch(const fs::path& p) {
    std::ofstream f(p.string());
    f << "not a real image, extension-based filtering only";
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "pixal3d_list_view_images_test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    // Natural sort with zero-padded tie, mixed with a non-numeric name, through the real
    // directory-scanning path (not just natural_name_less on hardcoded strings, which
    // trellis-test-transforms-synth already covers).
    touch(dir / "view10.png");
    touch(dir / "view2.png");
    touch(dir / "view02.png");
    touch(dir / "aview.png");
    // Non-image files: must be excluded regardless of extension-less or unknown extension.
    touch(dir / "readme.txt");
    touch(dir / "notes");
    // Hidden / AppleDouble files: must be excluded (dot-prefixed).
    touch(dir / ".hidden.png");
    touch(dir / "._view2.png");
    // Case-insensitive extension match.
    touch(dir / "VIEW99.PNG");
    // A subdirectory with an image-like name must not be listed (regular files only).
    fs::create_directories(dir / "subdir.png", ec);

    std::string err;
    std::vector<std::string> names = list_view_images(dir.string(), &err);
    check(err.empty(), "no error listing a readable directory");
    check(names.size() == 5, "hidden/non-image/directory entries excluded, 5 real images kept (got " +
          std::to_string(names.size()) + ")");

    // Expected natural order: aview.png, view02.png, view2.png, VIEW99.PNG (case-insensitive
    // extension, but filename comparison is still byte-wise so 'V' < 'a'... actually
    // natural_name_less compares raw bytes for the non-numeric parts, so uppercase-first names
    // sort before lowercase ones in ASCII). Assert only the two properties this test cares about
    // instead of a single brittle full-order string, so an unrelated ASCII-case detail doesn't
    // make this test flaky.
    auto pos = [&](const std::string& n) -> int {
        for (size_t i = 0; i < names.size(); ++i) if (names[i] == n) return (int)i;
        return -1;
    };
    const int p_v02 = pos("view02.png"), p_v2 = pos("view2.png"), p_v10 = pos("view10.png");
    check(p_v02 >= 0 && p_v2 >= 0 && p_v10 >= 0, "all three view*.png names are present");
    if (p_v02 >= 0 && p_v2 >= 0 && p_v10 >= 0) {
        check(p_v02 < p_v2, "view02.png sorts before view2.png (zero-padded byte tie-break)");
        check(p_v2 < p_v10, "view2.png sorts before view10.png (natural/numeric order, not lexicographic)");
    }

    // Determinism: repeated calls against the same unchanged directory return the same order.
    std::vector<std::string> names2 = list_view_images(dir.string(), &err);
    std::vector<std::string> names3 = list_view_images(dir.string(), &err);
    check(names == names2 && names2 == names3, "list_view_images() is deterministic across repeated calls");

    fs::remove_all(dir, ec);

    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
