// Minimal tolerant JSON parser for Pixal3D multiview `transforms.json` (Blender/NeRF-style:
// top-level camera_angle_x/mesh_scale + frames[] with file_path/transform_matrix/optional
// per-frame camera_angle_x). Not a general-purpose JSON library -- there was none in this repo
// (grepped trellis-server.cpp/trellis_model.cpp: only ad-hoc JSON *writers* exist, e.g.
// mesh_glb.cpp's glTF chunk and trellis-server.cpp's error responses) -- so this is a small,
// self-contained recursive-descent reader good enough for that one file shape.
#pragma once
#include <string>
#include <vector>

namespace trellis {

struct TransformsFrame {
    std::string file_path;
    float transform_matrix[16] = {};   // row-major 4x4 camera-to-world (Blender/NeRF convention)
    float camera_angle_x = 0.0f;       // per-frame horizontal FOV override, radians
    bool  has_camera_angle_x = false;
};

struct TransformsFile {
    std::vector<TransformsFrame> frames;
    float camera_angle_x = 0.0f;   // top-level fallback FOV, radians
    bool  has_camera_angle_x = false;
    // Pixal3D projection scale is semantically required for multiview inputs. A wrong default
    // can still produce a plausible-looking but geometrically broken reconstruction, so the
    // parser rejects a missing/non-positive mesh_scale instead of silently assuming 1.0.
    float mesh_scale = 0.0f;
};

// Parses `path` into `out`. Returns false (with an stderr message describing what's wrong)
// on a missing file, malformed JSON, missing/invalid mesh_scale, or a frame missing
// file_path/transform_matrix.
bool load_transforms_json(const std::string& path, TransformsFile& out);

// ---- transforms.json を持たない入力（canonical rig）-------------------------------------
// ブラウザ側 (web/real_e2e/calibration.js の CANONICAL_RIG) と同一の規約:
// ちょうど 4 枚のターンテーブル視点を front / right / back / left と解釈し、仰角 0・距離
// 3.1192049980163574・camera_angle_x 20° の固定姿勢を割り当てる。mesh_scale は呼び出し側が
// 明示する（既定 1.0 を仮定しない。誤ったスケールは形状を静かに壊すため）。
constexpr int CANONICAL_RIG_VIEWS = 4;

// ファイル名の順序。locale に依存しない自然順: ASCII・大小文字を区別し、数字列は整数として
// 比較、数値が同値なら元のバイト列で比較する（"view2" < "view10"、"view02" < "view2"）。
bool natural_name_less(const std::string& a, const std::string& b);

// `dir` 直下の画像（png/jpg/jpeg/webp）を自然順で返す。再帰しない。走査できなかった場合は
// 空を返し、`error` が非 null ならその理由を入れる（「画像が 0 枚」と区別するため）。
std::vector<std::string> list_view_images(const std::string& dir, std::string* error = nullptr);

// `image_files`（呼び出し側が確定させた、自然順のファイル名リスト）へ canonical rig を割り当てる。
// ちょうど CANONICAL_RIG_VIEWS 枚でなければ false。mesh_scale は有限かつ正であること。
bool synthesize_canonical_rig(const std::vector<std::string>& image_files, float mesh_scale,
                              TransformsFile& out, std::string& error);

// ---- 単一視点入力（SV gauge）-----------------------------------------------------------
// 画像 1 枚だけの Pixal3D SV 推論に使う、正面固定の「ゲージ」カメラ。web/app/single_view.js の
// makeSingleViewTransforms と同一の規約で、これを共有 C++ 側の正本とする（JS はブラウザ経路の
// 都合で残るが、tests/fixtures/sv_gauge/*.json を共有フィクスチャとして
// trellis-test-sv-input と web/app/test_single_view.mjs の双方が読み、一致を検査する）。
//
//   mesh_scale = 1.0 固定（単一視点では被写体の実寸を決められないので、既定の物差しを置く）
//   distance   = 1 / (2 * mesh_scale * tan(fov/2))
//   姿勢       = canonical rig の front と同じ向きで、距離だけ上式
//
// canonical rig（4 視点）と違い距離が FOV から決まるのは、1 枚では三角測量ができず
// 「FOV と距離の積」しか効かないため。FOV を変えたら距離も連動させないと画角が壊れる。
constexpr float SINGLE_VIEW_DEFAULT_FOV = 0.3490658503988659f;   // 20 deg
constexpr float SINGLE_VIEW_MESH_SCALE  = 1.0f;

// `image_file` 1 枚の TransformsFile を作る。fov_rad は有限かつ 0 < fov < pi。
// 範囲外・非有限・空のファイル名は false（既定 FOV へ落とさない）。
bool synthesize_single_view_gauge(const std::string& image_file, float fov_rad,
                                  TransformsFile& out, std::string& error);

// `tf` を transforms.json として `path` へ書く。SV の staging ディレクトリを
// 「普通の --views 入力」として成立させるために使う（合成したカメラをファイル化するので、
// 読み込み側に SV 専用の分岐が要らず、視点数からモードを推測する余地も生まれない）。
// 生成物は人と tools/silhouette_iou.py がそのまま読める。
bool write_transforms_json(const std::string& path, const TransformsFile& tf, std::string& error);

// multiview 入力ディレクトリのメタデータを 1 本の経路で解決する。
//   - `dir/transforms.json` が存在する  → 従来どおり parse（壊れていれば失敗。合成へ落とさない）。
//     mesh_scale_set のときだけ、parse 成功後に mesh_scale を上書きする。
//   - 存在しない                        → dir 内の画像 4 枚から合成（mesh_scale_set が必須）。
bool load_views_metadata(const std::string& dir, float mesh_scale, bool mesh_scale_set,
                         TransformsFile& out, std::string& error);

} // namespace trellis
