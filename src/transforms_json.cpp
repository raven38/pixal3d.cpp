#include "transforms_json.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
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

} // namespace trellis
