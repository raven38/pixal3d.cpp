#include "transforms_json.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <filesystem>
#include <sstream>
#include <utility>

namespace trellis {
namespace {

enum class JsonType { Null, Bool, Number, String, Array, Object };
struct JsonValue;
using JsonArray  = std::vector<JsonValue>;
using JsonObject = std::vector<std::pair<std::string, JsonValue>>;   // order-preserving, small N

struct JsonValue {
    JsonType type = JsonType::Null;
    double num = 0.0;
    std::string str;
    std::shared_ptr<JsonArray> arr;
    std::shared_ptr<JsonObject> obj;
};

// Small recursive-descent JSON parser (values only -- no streaming) sufficient for the
// object/array/string/number shapes transforms.json actually uses.
struct Parser {
    const std::string& s;
    size_t i = 0;
    bool ok = true;
    std::string err;
    explicit Parser(const std::string& s_) : s(s_) {}

    void skip_ws() { while (i < s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r')) ++i; }
    char peek() const { return i < s.size() ? s[i] : '\0'; }
    void fail(const std::string& m) { if (ok) { ok = false; err = m; } }

    JsonValue parse_value() {
        skip_ws();
        if (i >= s.size()) { fail("unexpected end of input"); return {}; }
        switch (peek()) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': { JsonValue v; v.type = JsonType::String; v.str = parse_raw_string(); return v; }
            case 't': case 'f': return parse_bool();
            case 'n': return parse_null();
            default:  return parse_number();
        }
    }

    JsonValue parse_object() {
        JsonValue v; v.type = JsonType::Object; v.obj = std::make_shared<JsonObject>();
        ++i; skip_ws();
        if (peek() == '}') { ++i; return v; }
        for (;;) {
            skip_ws();
            if (peek() != '"') { fail("expected string key in object"); return v; }
            std::string key = parse_raw_string();
            skip_ws();
            if (peek() != ':') { fail("expected ':' after object key"); return v; }
            ++i;
            v.obj->emplace_back(std::move(key), parse_value());
            if (!ok) return v;
            skip_ws();
            if (peek() == ',') { ++i; continue; }
            if (peek() == '}') { ++i; break; }
            fail("expected ',' or '}' in object"); break;
        }
        return v;
    }

    JsonValue parse_array() {
        JsonValue v; v.type = JsonType::Array; v.arr = std::make_shared<JsonArray>();
        ++i; skip_ws();
        if (peek() == ']') { ++i; return v; }
        for (;;) {
            v.arr->push_back(parse_value());
            if (!ok) return v;
            skip_ws();
            if (peek() == ',') { ++i; continue; }
            if (peek() == ']') { ++i; break; }
            fail("expected ',' or ']' in array"); break;
        }
        return v;
    }

    std::string parse_raw_string() {
        ++i;   // opening quote
        std::string out;
        while (i < s.size() && s[i] != '"') {
            char c = s[i];
            if (c == '\\' && i + 1 < s.size()) {
                char n = s[i + 1];
                switch (n) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    default:  out.push_back(n); break;   // \uXXXX etc: not needed for this file, pass through
                }
                i += 2;
            } else { out.push_back(c); ++i; }
        }
        if (i < s.size()) ++i; else fail("unterminated string");
        return out;
    }

    JsonValue parse_bool() {
        JsonValue v; v.type = JsonType::Bool;
        if (s.compare(i, 4, "true") == 0) { v.num = 1; i += 4; }
        else if (s.compare(i, 5, "false") == 0) { v.num = 0; i += 5; }
        else fail("bad literal (expected true/false)");
        return v;
    }
    JsonValue parse_null() {
        JsonValue v; v.type = JsonType::Null;
        if (s.compare(i, 4, "null") == 0) i += 4; else fail("bad literal (expected null)");
        return v;
    }
    JsonValue parse_number() {
        size_t start = i;
        if (peek() == '-' || peek() == '+') ++i;
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i]=='.' || s[i]=='e' || s[i]=='E' || s[i]=='+' || s[i]=='-')) ++i;
        JsonValue v; v.type = JsonType::Number;
        if (i == start) { fail("expected a number"); return v; }
        v.num = std::strtod(s.substr(start, i - start).c_str(), nullptr);
        return v;
    }
};

const JsonValue* obj_get(const JsonValue& v, const char* key) {
    if (v.type != JsonType::Object || !v.obj) return nullptr;
    for (auto& kv : *v.obj) if (kv.first == key) return &kv.second;
    return nullptr;
}

constexpr double kPi = 3.14159265358979323846;
bool valid_camera_fov(double fov) {
    return std::isfinite(fov) && fov > 0.0 && fov < kPi;
}

} // namespace

bool load_transforms_json(const std::string& path, TransformsFile& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "transforms_json: cannot open %s\n", path.c_str()); return false; }
    std::ostringstream ss; ss << f.rdbuf();
    const std::string content = ss.str();

    Parser p(content);
    JsonValue root = p.parse_value();
    if (!p.ok) { fprintf(stderr, "transforms_json: parse error in %s: %s\n", path.c_str(), p.err.c_str()); return false; }
    if (root.type != JsonType::Object) { fprintf(stderr, "transforms_json: %s root is not a JSON object\n", path.c_str()); return false; }

    if (const JsonValue* v = obj_get(root, "camera_angle_x"); v && v->type == JsonType::Number) {
        if (!valid_camera_fov(v->num)) {
            fprintf(stderr,
                    "transforms_json: %s has invalid camera_angle_x=%g; expected a finite value with 0 < fov < pi radians\n",
                    path.c_str(), v->num);
            return false;
        }
        out.camera_angle_x = (float)v->num; out.has_camera_angle_x = true;
    }

    const JsonValue* mesh_scale = obj_get(root, "mesh_scale");
    if (!mesh_scale || mesh_scale->type != JsonType::Number) {
        fprintf(stderr,
                "transforms_json: %s is missing required top-level 'mesh_scale'; refusing to assume 1.0 because an incorrect scale can silently corrupt Pixal3D multiview geometry\n",
                path.c_str());
        return false;
    }
    out.mesh_scale = (float)mesh_scale->num;
    if (!std::isfinite(out.mesh_scale) || out.mesh_scale <= 0.0f) {
        fprintf(stderr,
                "transforms_json: %s has invalid mesh_scale=%g; expected a finite value > 0\n",
                path.c_str(), (double)out.mesh_scale);
        return false;
    }

    const JsonValue* frames = obj_get(root, "frames");
    if (!frames || frames->type != JsonType::Array) {
        fprintf(stderr, "transforms_json: %s has no 'frames' array\n", path.c_str());
        return false;
    }

    out.frames.clear();
    out.frames.reserve(frames->arr->size());
    for (const JsonValue& fr : *frames->arr) {
        if (fr.type != JsonType::Object) { fprintf(stderr, "transforms_json: a frame entry is not an object\n"); return false; }
        TransformsFrame tf{};
        if (const JsonValue* fp = obj_get(fr, "file_path"); fp && fp->type == JsonType::String) tf.file_path = fp->str;
        if (tf.file_path.empty()) { fprintf(stderr, "transforms_json: frame missing 'file_path'\n"); return false; }
        if (const JsonValue* ca = obj_get(fr, "camera_angle_x"); ca && ca->type == JsonType::Number) {
            if (!valid_camera_fov(ca->num)) {
                fprintf(stderr,
                        "transforms_json: frame '%s' has invalid camera_angle_x=%g; expected a finite value with 0 < fov < pi radians\n",
                        tf.file_path.c_str(), ca->num);
                return false;
            }
            tf.camera_angle_x = (float)ca->num; tf.has_camera_angle_x = true;
        }
        const JsonValue* tm = obj_get(fr, "transform_matrix");
        if (!tm || tm->type != JsonType::Array || tm->arr->size() != 4) {
            fprintf(stderr, "transforms_json: frame '%s' missing/malformed 4x4 transform_matrix\n", tf.file_path.c_str());
            return false;
        }
        bool bad = false;
        for (int r = 0; r < 4 && !bad; ++r) {
            const JsonValue& row = (*tm->arr)[r];
            if (row.type != JsonType::Array || row.arr->size() != 4) { bad = true; break; }
            for (int c = 0; c < 4; ++c) {
                const JsonValue& cell = (*row.arr)[c];
                if (cell.type != JsonType::Number) { bad = true; break; }
                tf.transform_matrix[r * 4 + c] = (float)cell.num;
            }
        }
        if (bad) { fprintf(stderr, "transforms_json: frame '%s' has a malformed transform_matrix row\n", tf.file_path.c_str()); return false; }
        out.frames.push_back(std::move(tf));
    }
    if (out.frames.empty()) { fprintf(stderr, "transforms_json: %s has zero frames\n", path.c_str()); return false; }
    return true;
}

// ---- canonical rig（transforms.json を持たない入力）---------------------------------------

// web/real_e2e/calibration.js の CANONICAL_RIG と同一の値。front / right / back / left、
// 仰角 0、距離 3.1192049980163574、camera_angle_x 20°(0.3490658503988659 rad)。
namespace {
constexpr float kCanonicalFov = 0.3490658503988659f;
constexpr float kCanonicalDist = 3.1192049980163574f;
const float kCanonicalPoses[CANONICAL_RIG_VIEWS][16] = {
    { 1, 0,  0, 0,   0, 0, -1, -kCanonicalDist,   0, 1, 0, 0,   0, 0, 0, 1 },  // front (azim 0)
    { 0, 0,  1, kCanonicalDist,   1, 0, 0, 0,     0, 1, 0, 0,   0, 0, 0, 1 },  // right (azim 90)
    {-1, 0,  0, 0,   0, 0,  1,  kCanonicalDist,   0, 1, 0, 0,   0, 0, 0, 1 },  // back  (azim 180)
    { 0, 0, -1, -kCanonicalDist, -1, 0, 0, 0,     0, 1, 0, 0,   0, 0, 0, 1 },  // left  (azim 270)
};
const char* kCanonicalNames[CANONICAL_RIG_VIEWS] = {
    "front (azim 0)", "right (azim 90)", "back (azim 180)", "left (azim 270)"
};

bool has_image_extension(const std::string& name) {
    const size_t dot = name.rfind('.');
    if (dot == std::string::npos) return false;
    std::string ext = name.substr(dot + 1);
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    return ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "webp";
}
} // namespace

bool natural_name_less(const std::string& a, const std::string& b) {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        const bool da = std::isdigit((unsigned char)a[i]) != 0;
        const bool db = std::isdigit((unsigned char)b[j]) != 0;
        if (da && db) {
            // 数字列は整数として比較する（先頭の 0 を無視した桁数 → 辞書順）。
            size_t ia = i, ib = j;
            while (ia < a.size() && std::isdigit((unsigned char)a[ia])) ++ia;
            while (ib < b.size() && std::isdigit((unsigned char)b[ib])) ++ib;
            size_t sa = i, sb = j;
            while (sa + 1 < ia && a[sa] == '0') ++sa;
            while (sb + 1 < ib && b[sb] == '0') ++sb;
            const size_t la = ia - sa, lb = ib - sb;
            if (la != lb) return la < lb;
            const int cmp = a.compare(sa, la, b, sb, lb);
            if (cmp != 0) return cmp < 0;
            i = ia; j = ib;              // 数値は同値。次の区間へ
            continue;
        }
        if (a[i] != b[j]) return (unsigned char)a[i] < (unsigned char)b[j];
        ++i; ++j;
    }
    if (i < a.size() || j < b.size()) return a.size() < b.size();
    return a < b;                        // 数値同値（"view02" vs "view2"）はバイト列で決める
}

std::vector<std::string> list_view_images(const std::string& dir, std::string* error) {
    std::vector<std::string> names;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    if (ec) {
        // 走査できなかったのを「画像が足りない」と誤って報告しない（fail closed）。
        if (error) *error = "cannot list " + dir + ": " + ec.message();
        return {};
    }
    for (const auto& e : it) {
        if (!e.is_regular_file(ec) || ec) { ec.clear(); continue; }
        const std::string name = e.path().filename().string();
        if (name.empty() || name[0] == '.') continue;   // ._* 等の付随ファイルを拾わない
        if (has_image_extension(name)) names.push_back(name);
    }
    std::sort(names.begin(), names.end(), natural_name_less);
    return names;
}

bool synthesize_canonical_rig(const std::vector<std::string>& image_files, float mesh_scale,
                              TransformsFile& out, std::string& error) {
    if ((int)image_files.size() != CANONICAL_RIG_VIEWS) {
        error = "without transforms.json exactly " + std::to_string(CANONICAL_RIG_VIEWS) +
                " views are required (front, right, back, left on a turntable); found " +
                std::to_string(image_files.size());
        return false;
    }
    if (!std::isfinite(mesh_scale) || mesh_scale <= 0.0f) {
        error = "a synthesized rig needs an explicit positive mesh_scale; refusing to assume 1.0 "
                "because an incorrect scale can silently corrupt Pixal3D multiview geometry";
        return false;
    }
    out = TransformsFile{};
    out.camera_angle_x = kCanonicalFov;
    out.has_camera_angle_x = true;
    out.mesh_scale = mesh_scale;
    out.frames.reserve(CANONICAL_RIG_VIEWS);
    for (int i = 0; i < CANONICAL_RIG_VIEWS; ++i) {
        TransformsFrame tf{};
        tf.file_path = image_files[i];
        for (int k = 0; k < 16; ++k) tf.transform_matrix[k] = kCanonicalPoses[i][k];
        out.frames.push_back(std::move(tf));
    }
    return true;
}

bool load_views_metadata(const std::string& dir, float mesh_scale, bool mesh_scale_set,
                         TransformsFile& out, std::string& error) {
    const std::string json_path = dir + "/transforms.json";
    std::error_code ec;
    // fallback は「transforms.json が存在しない」ときだけ。空ファイル・壊れた JSON・ディレクトリ・
    // 読めない場合は従来どおり失敗させる（fail closed）。存在判定そのものが失敗したときも、
    // 「無い」と解釈せずエラーにする（permission 等で合成へ落ちるのを防ぐ）。
    const bool has_json = std::filesystem::exists(json_path, ec);
    if (ec) {
        error = "failed to inspect " + json_path + ": " + ec.message();
        return false;
    }
    if (has_json) {
        if (!load_transforms_json(json_path, out)) { error = "failed to load " + json_path; return false; }
        if (mesh_scale_set) {
            if (!std::isfinite(mesh_scale) || mesh_scale <= 0.0f) {
                error = "invalid --mesh-scale; expected a finite value > 0";
                return false;
            }
            out.mesh_scale = mesh_scale;   // parse 成功後の上書きだけを許す
        }
        return true;
    }
    if (!std::filesystem::is_directory(dir, ec) || ec) { error = dir + " is not a directory"; return false; }
    std::string list_err;
    const std::vector<std::string> images = list_view_images(dir, &list_err);
    if (!list_err.empty()) { error = list_err; return false; }
    if (!synthesize_canonical_rig(images, mesh_scale_set ? mesh_scale : 0.0f, out, error)) {
        error = dir + "/transforms.json not found and the canonical rig could not be used: " + error;
        return false;
    }
    return true;
}

} // namespace trellis
