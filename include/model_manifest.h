// pixal3d-models.json（モデルセット manifest）の契約を native 側で判定する。
//
// 同じ契約が既に 3 か所にある:
//   - tools/model_manifest.py   … 生成と検証（SHA256 まで見る正本）
//   - web/app/model_family.js   … ブラウザ側の name<->role / model_family 判定
//   - app/src-tauri/src/model_cache.rs … 管理キャッシュの削除対象決定（name のみ読む）
// ここは trellis-server が「この models ディレクトリで生成してよいか」を答えるためのもので、
// 規則は web/app/model_family.js と 1 対 1 に揃える（設計書 D6 の available 判定）:
//   1. model_family が明示されていれば mv | sv
//   2. flow 4 role の name が全て _mv.gguf なら mv、全て _sv.gguf なら sv、混在・不明は不定
//   3. 明示値と推定値が両方あって不一致なら拒否
//   4. family = 明示 || 推定 || 既定 mv
//   5. 既知 name は固定 role・required:true。未知 name に既知 role を付けるのも拒否
//   6. name 重複・role 重複を拒否
//   7. family の必須 9 name / 9 role が全て揃う
//
// SHA256 は再計算しない（F16 セットは 12.7 GiB あり、リクエストごとに読めない）。
// バイト検証は installer の --model-manifest / --verify-models の責務で、manifest が
// ディレクトリに在ること自体が「検証済みで完全」の印、という既存の契約を踏襲する。
// ここが見るのは「manifest が契約に合う」「9 ファイルが実在しサイズが一致する」まで。
#pragma once
#include <string>
#include <vector>

namespace trellis {

enum class ModelFamily { MV, SV };

const char* model_family_name(ModelFamily f);
// name -> role の必須表（Python model_files(family) と同じ順序・同じ内容）。
std::vector<std::pair<std::string, std::string>> required_files_for_family(ModelFamily f);

struct ModelManifestFile {
    std::string name;
    std::string role;
    std::string sha256;
    long long   size_bytes = -1;
    bool        required = false;
};

struct ModelManifest {
    int schema_version = 0;
    std::string model_set;
    std::string version;
    ModelFamily family = ModelFamily::MV;
    bool family_explicit = false;
    std::vector<ModelManifestFile> files;
};

// `path` を読み、上の 1〜7 を検査する。落ちた理由は `error` に入る。
bool load_model_manifest(const std::string& path, ModelManifest& out, std::string& error);

struct ModelSetStatus {
    bool configured = false;     // ディレクトリが指定されている
    bool available  = false;     // 設計書 D6 の 5 条件すべてを満たす
    std::string reason;          // available=false のときだけ埋まる
    std::string model_set;
    std::string version;
    std::string family;
};

// `dir` が `expected` family のモデルセットとして使えるかを判定する。
// dir が空なら configured=false。
ModelSetStatus inspect_model_set(const std::string& dir, ModelFamily expected);

} // namespace trellis
