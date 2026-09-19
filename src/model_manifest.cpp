#include "model_manifest.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace trellis {
namespace {

// ---- pixal3d-models.json だけを読む最小パーサ ------------------------------------------
// このリポジトリには汎用 JSON ライブラリが無い（transforms_json.cpp も同じ理由で自前）。
// 対象は「トップレベルが object、files が flat object の array」という既知の 1 形状だけ。
struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object } kind = Kind::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json* find(const std::string& key) const {
        for (const auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

struct Parser {
    const std::string& s;
    size_t i = 0;
    std::string err;

    explicit Parser(const std::string& text) : s(text) {}

    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i; }
    bool lit(const char* text) {
        const size_t n = std::char_traits<char>::length(text);
        if (s.compare(i, n, text) != 0) return false;
        i += n;
        return true;
    }

    bool parse_string(std::string& out) {
        if (i >= s.size() || s[i] != '"') { err = "expected a string"; return false; }
        ++i;
        out.clear();
        while (i < s.size()) {
            const char c = s[i++];
            if (c == '"') return true;
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) break;
            const char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    // manifest に非 ASCII は出ないが、来ても壊れないよう読み飛ばす。
                    if (i + 4 > s.size()) { err = "truncated \\u escape"; return false; }
                    i += 4;
                    out += '?';
                    break;
                }
                default: err = "invalid escape"; return false;
            }
        }
        err = "unterminated string";
        return false;
    }

    bool parse(Json& out) {
        ws();
        if (i >= s.size()) { err = "unexpected end of input"; return false; }
        const char c = s[i];
        if (c == '{') {
            ++i;
            out.kind = Json::Kind::Object;
            ws();
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            while (true) {
                ws();
                std::string key;
                if (!parse_string(key)) return false;
                ws();
                if (i >= s.size() || s[i] != ':') { err = "expected ':'"; return false; }
                ++i;
                Json value;
                if (!parse(value)) return false;
                out.obj.emplace_back(std::move(key), std::move(value));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return true; }
                err = "expected ',' or '}'";
                return false;
            }
        }
        if (c == '[') {
            ++i;
            out.kind = Json::Kind::Array;
            ws();
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            while (true) {
                Json value;
                if (!parse(value)) return false;
                out.arr.push_back(std::move(value));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return true; }
                err = "expected ',' or ']'";
                return false;
            }
        }
        if (c == '"') { out.kind = Json::Kind::String; return parse_string(out.str); }
        if (lit("true"))  { out.kind = Json::Kind::Bool; out.b = true;  return true; }
        if (lit("false")) { out.kind = Json::Kind::Bool; out.b = false; return true; }
        if (lit("null"))  { out.kind = Json::Kind::Null; return true; }
        {
            char* end = nullptr;
            const double d = std::strtod(s.c_str() + i, &end);
            if (end == s.c_str() + i) { err = "invalid value"; return false; }
            i = (size_t)(end - s.c_str());
            out.kind = Json::Kind::Number;
            out.num = d;
            return true;
        }
    }
};

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const char* const kFlowRoles[] = { "ss_flow", "shape_flow_512", "shape_flow_1024", "texture_flow_1024" };

bool is_flow_role(const std::string& role) {
    for (const char* r : kFlowRoles) if (role == r) return true;
    return false;
}

} // namespace

const char* model_family_name(ModelFamily f) { return f == ModelFamily::SV ? "sv" : "mv"; }

std::vector<std::pair<std::string, std::string>> required_files_for_family(ModelFamily f) {
    const std::string fam = model_family_name(f);
    return {
        { "dinov3.gguf",                              "image_encoder"     },
        { "pixal3d_naf.gguf",                         "naf"               },
        { "pixal3d_ss_flow_" + fam + ".gguf",         "ss_flow"           },
        { "ss_dec.gguf",                              "ss_decoder"        },
        { "pixal3d_shape_flow_512_" + fam + ".gguf",  "shape_flow_512"    },
        { "shape_dec.gguf",                           "shape_decoder"     },
        { "pixal3d_shape_flow_1024_" + fam + ".gguf", "shape_flow_1024"   },
        { "pixal3d_tex_flow_1024_" + fam + ".gguf",   "texture_flow_1024" },
        { "tex_dec.gguf",                             "texture_decoder"   },
    };
}

bool load_model_manifest(const std::string& path, ModelManifest& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { error = "cannot read " + path; return false; }
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    Parser parser(text);
    Json root;
    if (!parser.parse(root) || root.kind != Json::Kind::Object) {
        error = "invalid pixal3d-models.json: " + (parser.err.empty() ? std::string("not an object") : parser.err);
        return false;
    }

    out = ModelManifest{};
    if (const Json* v = root.find("schema_version")) {
        if (v->kind != Json::Kind::Number) { error = "schema_version must be a number"; return false; }
        out.schema_version = (int)v->num;
    }
    if (out.schema_version != 0 && out.schema_version != 1) {
        error = "unsupported model manifest schema_version: " + std::to_string(out.schema_version);
        return false;
    }
    if (const Json* v = root.find("model_set")) { if (v->kind == Json::Kind::String) out.model_set = v->str; }
    if (const Json* v = root.find("version"))   { if (v->kind == Json::Kind::String) out.version = v->str; }

    // 1. 明示 family
    bool explicit_family = false;
    ModelFamily explicit_value = ModelFamily::MV;
    if (const Json* v = root.find("model_family")) {
        if (v->kind != Json::Kind::String || (v->str != "mv" && v->str != "sv")) {
            error = "model_family must be one of: mv, sv";
            return false;
        }
        explicit_family = true;
        explicit_value = (v->str == "sv") ? ModelFamily::SV : ModelFamily::MV;
    }

    const Json* files = root.find("files");
    if (!files || files->kind != Json::Kind::Array || files->arr.empty()) {
        error = "files must be a non-empty array";
        return false;
    }
    for (size_t k = 0; k < files->arr.size(); ++k) {
        const Json& ent = files->arr[k];
        if (ent.kind != Json::Kind::Object) {
            error = "files[" + std::to_string(k) + "] must be an object";
            return false;
        }
        ModelManifestFile mf;
        const Json* n = ent.find("name");
        const Json* r = ent.find("role");
        if (!n || n->kind != Json::Kind::String || n->str.empty()) {
            error = "files[" + std::to_string(k) + "].name invalid";
            return false;
        }
        if (!r || r->kind != Json::Kind::String || r->str.empty()) {
            error = "files[" + std::to_string(k) + "].role invalid";
            return false;
        }
        mf.name = n->str;
        mf.role = r->str;
        if (const Json* v = ent.find("sha256")) { if (v->kind == Json::Kind::String) mf.sha256 = v->str; }
        if (const Json* v = ent.find("size_bytes")) {
            if (v->kind != Json::Kind::Number) { error = mf.name + ": size_bytes must be a number"; return false; }
            mf.size_bytes = (long long)v->num;
        }
        if (const Json* v = ent.find("required")) { mf.required = (v->kind == Json::Kind::Bool) && v->b; }
        out.files.push_back(std::move(mf));
    }

    // 2. flow 4 role のファイル名から family を推定
    bool inferred_any = false;
    ModelFamily inferred = ModelFamily::MV;
    {
        std::vector<std::string> flow_names;
        for (const auto& mf : out.files) if (is_flow_role(mf.role)) flow_names.push_back(mf.name);
        if (!flow_names.empty()) {
            for (ModelFamily fam : { ModelFamily::MV, ModelFamily::SV }) {
                const std::string suffix = std::string("_") + model_family_name(fam) + ".gguf";
                if (std::all_of(flow_names.begin(), flow_names.end(),
                                [&](const std::string& s) { return ends_with(s, suffix); })) {
                    inferred_any = true;
                    inferred = fam;
                    break;
                }
            }
        }
    }

    // 3. 明示と推定の不一致は拒否
    if (explicit_family && inferred_any && explicit_value != inferred) {
        error = std::string("model_family ") + model_family_name(explicit_value) +
                " does not match the flow file names (" + model_family_name(inferred) + ")";
        return false;
    }
    // 4. family の確定
    out.family_explicit = explicit_family;
    out.family = explicit_family ? explicit_value : (inferred_any ? inferred : ModelFamily::MV);

    // 5-6. name<->role の固定、重複の拒否
    const auto expected = required_files_for_family(out.family);
    std::map<std::string, std::string> role_of_name;
    std::map<std::string, std::string> name_of_role;
    for (const auto& kv : expected) {
        role_of_name[kv.first] = kv.second;
        name_of_role[kv.second] = kv.first;
    }
    std::set<std::string> seen_names, seen_roles;
    for (size_t k = 0; k < out.files.size(); ++k) {
        const ModelManifestFile& mf = out.files[k];
        const std::string at = "files[" + std::to_string(k) + "]";
        if (!seen_names.insert(mf.name).second) { error = "duplicate file name: " + mf.name; return false; }
        if (!seen_roles.insert(mf.role).second) { error = "duplicate role: " + mf.role; return false; }
        const auto rn = role_of_name.find(mf.name);
        if (rn != role_of_name.end()) {
            if (rn->second != mf.role) {
                error = at + ": " + mf.name + " must have role " + rn->second + ", not " + mf.role;
                return false;
            }
            if (!mf.required) { error = at + ": " + mf.name + " must be required"; return false; }
        } else {
            const auto nr = name_of_role.find(mf.role);
            if (nr != name_of_role.end()) {
                error = at + ": role " + mf.role + " must be " + nr->second + ", not " + mf.name;
                return false;
            }
        }
    }

    // 7. 必須 9 name / 9 role が揃う
    std::vector<std::string> missing_names, missing_roles;
    for (const auto& kv : expected) {
        if (!seen_names.count(kv.first)) missing_names.push_back(kv.first);
        if (!seen_roles.count(kv.second)) missing_roles.push_back(kv.second);
    }
    if (!missing_names.empty()) {
        std::ostringstream os;
        os << "manifest missing required model files:";
        for (const auto& n : missing_names) os << " " << n;
        error = os.str();
        return false;
    }
    if (!missing_roles.empty()) {
        std::ostringstream os;
        os << "manifest missing required roles:";
        for (const auto& r : missing_roles) os << " " << r;
        error = os.str();
        return false;
    }
    return true;
}

ModelSetStatus inspect_model_set(const std::string& dir, ModelFamily expected) {
    ModelSetStatus st;
    st.family = model_family_name(expected);
    if (dir.empty()) {
        st.reason = std::string("no ") + model_family_name(expected) + " model directory is configured";
        return st;
    }
    st.configured = true;

    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec) || ec) {
        st.reason = "not a directory: " + dir;
        return st;
    }
    const std::string manifest_path = (std::filesystem::path(dir) / "pixal3d-models.json").string();
    if (!std::filesystem::exists(manifest_path, ec) || ec) {
        st.reason = "pixal3d-models.json is missing (an incomplete or unverified model directory)";
        return st;
    }

    ModelManifest manifest;
    std::string err;
    if (!load_model_manifest(manifest_path, manifest, err)) {
        st.reason = err;
        return st;
    }
    st.model_set = manifest.model_set;
    st.version = manifest.version;
    st.family = model_family_name(manifest.family);

    if (manifest.family != expected) {
        st.reason = std::string("model_family mismatch: this endpoint needs ") +
                    model_family_name(expected) + ", the manifest says " + st.family;
        return st;
    }

    for (const auto& mf : manifest.files) {
        if (!mf.required) continue;
        const std::filesystem::path p = std::filesystem::path(dir) / mf.name;
        if (!std::filesystem::exists(p, ec) || ec) {
            st.reason = "missing model file: " + mf.name;
            return st;
        }
        if (mf.size_bytes >= 0) {
            const auto actual = std::filesystem::file_size(p, ec);
            if (ec) { st.reason = "cannot stat model file: " + mf.name; return st; }
            if ((long long)actual != mf.size_bytes) {
                st.reason = "size mismatch " + mf.name + ": " + std::to_string((long long)actual) +
                            " != " + std::to_string(mf.size_bytes);
                return st;
            }
        }
    }

    st.available = true;
    return st;
}

} // namespace trellis
