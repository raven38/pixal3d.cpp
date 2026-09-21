// trellis-server — resident HTTP wrapper around the TRELLIS.2 / Pixal3D pipelines.
//
//   GET  /health       -> "ok"  （本文は変えない。Studio が文字列比較している）
//   GET  /capabilities -> JSON。生成中か（busy / completed）と、MV / SV の各モデルセットが
//                        使えるか（configured / available / reason）を申告する。
//                        Studio はこれで SV モードの可否と、Stop waiting 後に Generate を
//                        戻してよい時点を決める（設計書 D6 / D11）。
//   POST /generate      multipart/form-data with an "image" file part; optional text
//                        fields "seed", "resolution" (512/1024/1536), "bg_removal"
//                        (threshold|birefnet), "uv" (xatlas = default, unique
//                        chart space; box = faster projection), "band" (narrow-band
//                        DC remesh band width, default 1 — see --band). Returns
//                        model/gltf-binary.
//   POST /generate-trellis2-mv  TRELLIS.2 pose-free multi-image endpoint. Takes
//                        image0..imageN-1, num_images, fusion=stochastic|multidiffusion.
//                        Uses the same model set as /generate; no transforms/mesh_scale.
//   POST /generate-mv   Pixal3D multiview endpoint. Multipart may contain a
//                        "transforms" file part (transforms.json); without it, exactly 4
//                        turntable views plus a "mesh_scale" field select the canonical
//                        rig (front/right/back/left). It also takes one or more
//                        view image parts named "view0", "view1", ... . Each view's
//                        multipart filename is staged relative to transforms.json,
//                        so it must match the frame file_path used by transforms.json.
//                        Optional fields: seed, resolution (1024/1536), uv, band, webp,
//                        num_views. Views must already be pre-matted RGBA, matching
//                        trellis-cli --views semantics. Returns model/gltf-binary.
//
// Launch-time defaults come from CLI flags (see trellis::parse_args);
// each request copies those defaults and applies its own overrides. The model
// directory is resolved once; each request runs the full pipeline via trellis_run()
// (per-stage load/free, like trellis-cli), serialized by a mutex. Keeping the
// process resident avoids re-initializing the Vulkan backend on every request.
#include "trellis_args.h"
#include "transforms_json.h"
#include "image_preprocess.h"
#include "model_manifest.h"
#include "trellis_run.h"
#include "httplib.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <cmath>
#include <random>
#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif
#include <string>

namespace {

std::string read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool write_file_bytes(const std::string& path, const std::string& data) {
    std::filesystem::path pp(path);
    std::error_code ec;
    if (!pp.parent_path().empty()) std::filesystem::create_directories(pp.parent_path(), ec);
    if (ec) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), (std::streamsize) data.size());
    return f.good();
}

// std::tmpnam on MSVC yields drive-root paths ("\\sXXX.N") that a non-elevated
// process cannot write; stage scratch files in the real temp directory instead.
std::string temp_stem() {
    static std::atomic<unsigned> counter{0};
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = ".";
    auto n = counter.fetch_add(1);
    // プロセス毎に 0 から振り直すと、クラッシュ後の残骸ディレクトリを次のリクエストが再利用
    // しうる。multiview はディレクトリ内の画像を数えるので、残骸 1 枚で「4 枚」を誤認する。
    // pid + 乱数で衝突しない名前にし、作成も排他で行う（呼び出し側で存在チェック）。
    static const std::string salt = [] {
        std::random_device rd;
        char buf[32];
        snprintf(buf, sizeof(buf), "%llx-%llx",
                 (unsigned long long)
#ifdef _WIN32
                 _getpid(),
#else
                 getpid(),
#endif
                 ((unsigned long long)rd() << 32) ^ (unsigned long long)rd());
        return std::string(buf);
    }();
    return (dir / ("trellis-req-" + salt + "-" + std::to_string(n))).string();
}

void set_error(httplib::Response& res, int status, const std::string& message) {
    std::string escaped;
    for (char c : message) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n";  break;
            case '\r': escaped += "\\r";  break;
            case '\t': escaped += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) break;
                escaped += c;
        }
    }
    res.status = status;
    res.set_content("{\"error\":\"" + escaped + "\"}", "application/json");
}

// 生成は gen_mu で直列化されている。その外側で「今 1 本走っているか」と「何本終わったか」を
// 公開する。Studio の Stop waiting（応答待ちの中止）はサーバ処理を止められないので、
// 再度 Generate を押してよい時点をこの 2 つで判定する（設計書 D11）。
std::atomic<bool>     g_busy{false};
std::atomic<uint64_t> g_completed{0};

struct BusyScope {
    BusyScope()  { g_busy.store(true); }
    ~BusyScope() { g_busy.store(false); g_completed.fetch_add(1); }
};

// 旧実装は atoi/atof で、"abc" が黙って 0 になっていた。入力の誤りを静かに飲まない
// （設計書 D4。seed=abc が 0 として通るのと 400 で落ちるのとでは再現性が別物になる）。
bool parse_u32_strict(const std::string& text, uint32_t& out) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
    if (!end || *end != '\0' || errno == ERANGE || v > 0xFFFFFFFFull) return false;
    if (text[0] == '-' || text[0] == '+') return false;
    out = (uint32_t)v;
    return true;
}

bool parse_int_strict(const std::string& text, long& out) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(text.c_str(), &end, 10);
    if (!end || *end != '\0' || errno == ERANGE) return false;
    out = v;
    return true;
}

bool parse_double_strict(const std::string& text, double& out) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (!end || *end != '\0' || !std::isfinite(v)) return false;
    out = v;
    return true;
}

// 共通の生成オーバーライド。不正な値は既定へ落とさず、理由を返して 400 にする。
bool apply_common_overrides(const httplib::Request& req, trellis::TrellisParams& p, std::string& error) {
    if (req.has_file("seed")) {
        uint32_t seed = 0;
        if (!parse_u32_strict(req.get_file_value("seed").content, seed)) {
            error = "seed must be an integer in [0, 4294967295]";
            return false;
        }
        p.seed = seed;
    }
    if (req.has_file("resolution")) {
        long res = 0;
        if (!parse_int_strict(req.get_file_value("resolution").content, res) ||
            (res != 512 && res != 1024 && res != 1536)) {
            error = "resolution must be one of 512, 1024, 1536";
            return false;
        }
        p.set_res((int)res);
    }
    if (req.has_file("uv")) {
        const std::string& uv = req.get_file_value("uv").content;
        if (uv != "xatlas" && uv != "box") { error = "uv must be 'xatlas' or 'box'"; return false; }
        p.xatlas = (uv == "xatlas");
    }
    if (req.has_file("band")) {
        long band = 0;
        if (!parse_int_strict(req.get_file_value("band").content, band) || band < 0 || band > 16) {
            error = "band must be an integer in [0, 16]";
            return false;
        }
        p.band = (int)band;
    }
    if (req.has_file("webp")) {
        const std::string& w = req.get_file_value("webp").content;
        p.webp = (w == "off" || w == "0" || w == "false") ? 0
               : (w == "on"  || w == "1" || w == "true")  ? 1 : -1;
    }
    return true;
}

// PNG の magic と IHDR から寸法を読む。デコード前に画素数を上限で弾くため
// （巨大画像でメモリを踏ませない。設計書 D4）。
bool png_dimensions(const std::string& bytes, long long& w, long long& h) {
    static const unsigned char kMagic[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    if (bytes.size() < 24) return false;
    if (std::memcmp(bytes.data(), kMagic, 8) != 0) return false;
    auto be32 = [&](size_t off) {
        return ((long long)(unsigned char)bytes[off] << 24) | ((long long)(unsigned char)bytes[off + 1] << 16) |
               ((long long)(unsigned char)bytes[off + 2] << 8) | (long long)(unsigned char)bytes[off + 3];
    };
    w = be32(16);
    h = be32(20);
    return w > 0 && h > 0;
}

std::string json_escape(const std::string& in) {
    std::string out;
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c >= 0x20) out += c;
        }
    }
    return out;
}

// Accept a relative upload filename, but never let a request escape its staging directory.
bool safe_stage_path(const std::filesystem::path& root, const std::string& filename,
                     std::filesystem::path& out) {
    if (filename.empty()) return false;
    std::filesystem::path rel(filename);
    if (rel.is_absolute()) return false;
    rel = rel.lexically_normal();
    if (rel.empty() || rel == ".") return false;
    for (const auto& part : rel) {
        if (part == "..") return false;
    }
    out = root / rel;
    return true;
}

struct Trellis2Status {
    bool available = false;
    std::string reason;
};

Trellis2Status inspect_trellis2_models(const std::string& dir) {
    Trellis2Status st;
    if (dir.empty()) { st.reason = "TRELLIS.2 models directory is not configured"; return st; }
    static const char* const required[] = {
        "dinov3.gguf", "ss_flow.gguf", "ss_dec.gguf",
        "shape_flow_512.gguf", "shape_flow_1024.gguf", "shape_dec.gguf",
        "tex_flow_512.gguf", "tex_flow_1024.gguf", "tex_dec.gguf",
    };
    std::error_code ec;
    for (const char* name : required) {
        const std::filesystem::path p = std::filesystem::path(dir) / name;
        if (!std::filesystem::is_regular_file(p, ec) || ec) {
            st.reason = std::string("missing TRELLIS.2 model file: ") + name;
            return st;
        }
        const auto n = std::filesystem::file_size(p, ec);
        if (ec || n == 0) {
            st.reason = std::string("empty/unreadable TRELLIS.2 model file: ") + name;
            return st;
        }
    }
    st.available = true;
    return st;
}

void cleanup_outputs(const std::string& stem) {
    std::remove((stem + ".glb").c_str());
    std::remove((stem + ".ply").c_str());
    std::remove((stem + "_base.png").c_str());
}

}  // namespace

int main(int argc, char** argv) {
    // Stage progress goes to stdout, which is fully buffered when piped (e.g.
    // under Lemonade's output capture) — keep it line-visible for diagnostics.
    setvbuf(stdout, nullptr, _IONBF, 0);

    trellis::TrellisParams base;
    if (!trellis::parse_args(argc, argv, base)) {
        trellis::print_usage(argv[0], /*server=*/true);
        return base.help ? 0 : 1;
    }

    std::mutex gen_mu;
    httplib::Server svr;

    // Trellis Studio (and any browser client) calls this server from a different
    // origin — a Tauri webview is tauri://localhost / http://tauri.localhost, and a
    // browser-served UI is another port — so every response needs permissive CORS
    // headers, and a multipart POST with non-simple headers may be preflighted with
    // OPTIONS. Applied to every route via the post-routing hook + a catch-all OPTIONS.
    svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Max-Age", "86400");
    });
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;  // headers added by the post-routing handler above
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("image")) {
            set_error(res, 400, "missing 'image' file part");
            return;
        }
        const auto& image = req.get_file_value("image");

        // Per-request params start from the launch defaults, then apply overrides.
        trellis::TrellisParams p = base;
        {
            std::string perr;
            if (!apply_common_overrides(req, p, perr)) { set_error(res, 400, perr); return; }
        }
        if (req.has_file("bg_removal")) p.birefnet = (req.get_file_value("bg_removal").content == "birefnet") ? 1 : 0;

        const std::string stem = temp_stem();
        p.image  = stem + ".png";
        p.views.clear();
        p.output = stem + ".glb";

        std::string glb;
        std::string error_message = "3D reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            BusyScope busy;
            if (!write_file_bytes(p.image, image.content)) {
                set_error(res, 500, "failed to stage input image");
                return;
            }
            fprintf(stderr, "[trellis-server] generate: %zu-byte image, seed %u, res %s, bg %s, uv %s\n",
                    image.content.size(), p.seed, p.cascade ? std::to_string(p.hr_res).c_str() : "512",
                    p.birefnet < 0 ? "auto" : (p.birefnet ? "birefnet" : "threshold"), p.xatlas ? "xatlas" : "box");
            try {
                int rc = trellis_run(p);
                if (rc == 0) glb = read_file_bytes(p.output);
            } catch (const std::exception& e) {
                fprintf(stderr, "[trellis-server] generate failed: %s\n", e.what());
                error_message = e.what();
            }
            std::remove(p.image.c_str());
            cleanup_outputs(stem);
        }

        if (glb.empty()) {
            set_error(res, 500, error_message);
            return;
        }
        res.set_content(glb.data(), glb.size(), "model/gltf-binary");
    });

    svr.Post("/generate-mv", [&](const httplib::Request& req, httplib::Response& res) {
        // transforms.json は任意。無い場合は canonical rig（4 視点）で合成するので、
        // 代わりに mesh_scale フォームフィールドが必須になる。
        const bool has_transforms = req.has_file("transforms");

        trellis::TrellisParams p = base;
        {
            std::string perr;
            if (!apply_common_overrides(req, p, perr)) { set_error(res, 400, perr); return; }
        }
        // endpoint ごとに family を強制する（起動時の既定から継承しない。設計書 D3）。
        p.models = base.models;
        p.pixal3d_weights = "mv";
        p.pixal3d_weights_set = true;
        if (req.has_file("num_views")) {
            const std::string v = req.get_file_value("num_views").content;
            char* end = nullptr;
            const long n = std::strtol(v.c_str(), &end, 10);
            if (!end || *end != '\0' || n <= 0 || n > 1024) {
                set_error(res, 400, "num_views must be a positive integer");
                return;
            }
            p.num_views = (int)n;
            p.num_views_set = true;
        }
        if (req.has_file("mesh_scale")) {
            const std::string v = req.get_file_value("mesh_scale").content;
            char* end = nullptr;
            const double d = std::strtod(v.c_str(), &end);
            if (!end || *end != '\0' || !std::isfinite(d) || d <= 0.0) {
                set_error(res, 400, "mesh_scale must be a finite value > 0");
                return;
            }
            p.mesh_scale = (float)d;
            p.mesh_scale_set = true;
        }
        if (!has_transforms && !p.mesh_scale_set) {
            set_error(res, 400, "without a 'transforms' part, an explicit 'mesh_scale' field is required "
                                "(4 turntable views: front, right, back, left)");
            return;
        }
        if (!p.cascade || p.hr_res < 1024) {
            set_error(res, 400, "multiview generation requires resolution 1024 or 1536");
            return;
        }

        const std::string stem = temp_stem();
        const std::filesystem::path view_dir = stem + "-mv";
        p.image.clear();
        p.views = view_dir.string();
        p.output = stem + ".glb";

        std::string glb;
        std::string error_message = "multiview 3D reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            BusyScope busy;
            std::error_code ec;
            // 既存ディレクトリを使い回さない（残骸の画像が視点数の判定に混ざるため）。
            if (!std::filesystem::create_directory(view_dir, ec) || ec) {
                set_error(res, 500, "failed to create a fresh multiview staging directory");
                return;
            }

            bool staged_ok = true;
            if (has_transforms) {
                staged_ok = write_file_bytes((view_dir / "transforms.json").string(),
                                             req.get_file_value("transforms").content);
            }
            int staged_views = 0;
            if (staged_ok) {
                for (const auto& kv : req.files) {
                    if (kv.first.rfind("view", 0) != 0) continue;
                    const auto& upload = kv.second;
                    std::filesystem::path dst;
                    if (!safe_stage_path(view_dir, upload.filename, dst)) {
                        staged_ok = false;
                        error_message = "invalid multiview upload filename: " + upload.filename;
                        break;
                    }
                    if (!write_file_bytes(dst.string(), upload.content)) {
                        staged_ok = false;
                        error_message = "failed to stage multiview image: " + upload.filename;
                        break;
                    }
                    ++staged_views;
                }
            }

            if (!staged_ok || staged_views == 0) {
                std::filesystem::remove_all(view_dir, ec);
                cleanup_outputs(stem);
                set_error(res, staged_views == 0 && staged_ok ? 400 : 500,
                          staged_views == 0 && staged_ok ? "missing view image parts (view0, view1, ...)" : error_message);
                return;
            }

            // モデルを読む前に入力契約を検証し、具体的な理由を 400 で返す
            // （従来は generic 500 になり、視点数や mesh_scale の誤りが利用者に伝わらなかった）。
            {
                trellis::TransformsFile probe;
                std::string verr;
                if (!trellis::load_views_metadata(p.views, p.mesh_scale, p.mesh_scale_set, probe, verr)) {
                    // サーバ内部の staging パスを応答に出さない（利用者には意味がなく、実体も漏らす）。
                    const std::string prefix = p.views;
                    size_t at = verr.find(prefix);
                    while (at != std::string::npos) {
                        verr.replace(at, prefix.size(), "the uploaded views");
                        at = verr.find(prefix);
                    }
                    std::filesystem::remove_all(view_dir, ec);
                    cleanup_outputs(stem);
                    set_error(res, 400, verr);
                    return;
                }
                if (!has_transforms && p.num_views_set && p.num_views != trellis::CANONICAL_RIG_VIEWS) {
                    std::filesystem::remove_all(view_dir, ec);
                    cleanup_outputs(stem);
                    set_error(res, 400, "num_views must be " + std::to_string(trellis::CANONICAL_RIG_VIEWS) +
                                        " when no transforms part is sent");
                    return;
                }
            }

            fprintf(stderr,
                    "[trellis-server] generate-mv: %d staged views (%s), seed %u, res %d, uv %s, num_views=%d\n",
                    staged_views, has_transforms ? "transforms.json" : "canonical rig", p.seed, p.hr_res,
                    p.xatlas ? "xatlas" : "box", p.num_views);
            try {
                int rc = trellis_run(p);
                if (rc == 0) glb = read_file_bytes(p.output);
            } catch (const std::exception& e) {
                fprintf(stderr, "[trellis-server] generate-mv failed: %s\n", e.what());
                error_message = e.what();
            }
            std::filesystem::remove_all(view_dir, ec);
            cleanup_outputs(stem);
        }

        if (glb.empty()) {
            set_error(res, 500, error_message);
            return;
        }
        res.set_content(glb.data(), glb.size(), "model/gltf-binary");
    });

    svr.Post("/generate-trellis2-mv", [&](const httplib::Request& req, httplib::Response& res) {
        // Explicit pose-free TRELLIS.2 multi-image endpoint (#66). Keep this contract
        // disjoint from Pixal3D /generate-mv: no transforms, mesh_scale, FOV or view cameras.
        int num_images = 0;
        if (!req.has_file("num_images")) {
            set_error(res, 400, "missing num_images");
            return;
        }
        {
            long n = 0;
            if (!parse_int_strict(req.get_file_value("num_images").content, n) || n < 2 || n > 8) {
                set_error(res, 400, "num_images must be an integer in [2, 8]");
                return;
            }
            num_images = (int)n;
        }

        std::string fusion = "stochastic";
        if (req.has_file("fusion")) fusion = req.get_file_value("fusion").content;
        if (fusion != "stochastic" && fusion != "multidiffusion") {
            set_error(res, 400, "fusion must be 'stochastic' or 'multidiffusion'");
            return;
        }

        // Whitelist all multipart fields. imageN must be contiguous 0..num_images-1.
        for (const auto& kv : req.files) {
            const std::string& k = kv.first;
            bool ok = k == "num_images" || k == "fusion" || k == "seed" ||
                      k == "resolution" || k == "uv" || k == "band" ||
                      k == "webp" || k == "bg_removal";
            if (!ok && k.rfind("image", 0) == 0) {
                const std::string tail = k.substr(5);
                if (!tail.empty() && tail.find_first_not_of("0123456789") == std::string::npos) {
                    char* end = nullptr;
                    const long idx = std::strtol(tail.c_str(), &end, 10);
                    ok = end && *end == '\0' && idx >= 0 && idx < num_images;
                }
            }
            if (!ok) {
                set_error(res, 400, "unexpected field for generate-trellis2-mv: " + k);
                return;
            }
        }

        constexpr size_t kMaxOne = 64ull * 1024 * 1024;
        constexpr size_t kMaxTotal = 128ull * 1024 * 1024;
        size_t total = 0;
        for (int i = 0; i < num_images; ++i) {
            const std::string key = "image" + std::to_string(i);
            if (!req.has_file(key)) {
                set_error(res, 400, "missing " + key + " (image fields must be contiguous)");
                return;
            }
            const auto& im = req.get_file_value(key);
            if (im.content.empty() || im.content.size() > kMaxOne) {
                set_error(res, 400, key + " is empty or exceeds 64 MiB");
                return;
            }
            total += im.content.size();
            if (total > kMaxTotal) {
                set_error(res, 400, "combined images exceed 128 MiB");
                return;
            }
        }
        trellis::TrellisParams p = base;
        {
            std::string perr;
            if (!apply_common_overrides(req, p, perr)) { set_error(res, 400, perr); return; }
        }
        if (req.has_file("bg_removal")) {
            const std::string bg = req.get_file_value("bg_removal").content;
            if (bg == "birefnet") p.birefnet = 1;
            else if (bg == "threshold") p.birefnet = 0;
            else if (bg == "auto") p.birefnet = -1;
            else { set_error(res, 400, "bg_removal must be auto, threshold, or birefnet"); return; }
        }
        {
            const Trellis2Status st = inspect_trellis2_models(base.models);
            if (!st.available) {
                set_error(res, 503, "TRELLIS.2 multiview model set is not available: " + st.reason);
                return;
            }
        }
        p.image.clear();
        p.views.clear();
        p.sv_image.clear();
        p.pixal3d_weights_set = false;
        p.mesh_scale_set = false;
        p.num_views_set = false;
        p.trellis2_mv_mode = fusion;
        p.trellis2_mv_mode_set = true;

        const std::string stem = temp_stem();
        const std::filesystem::path view_dir = stem + "-trellis2-mv";
        p.trellis2_mv = view_dir.string();
        p.output = stem + ".glb";

        std::string glb;
        std::string error_message = "TRELLIS.2 multiview reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            BusyScope busy;
            std::error_code ec;
            if (!std::filesystem::create_directory(view_dir, ec) || ec) {
                set_error(res, 500, "failed to create a fresh TRELLIS.2 multiview staging directory");
                return;
            }

            bool staged = true;
            for (int i = 0; i < num_images; ++i) {
                char name[32];
                snprintf(name, sizeof name, "view%02d.png", i);
                if (!write_file_bytes((view_dir / name).string(),
                                      req.get_file_value("image" + std::to_string(i)).content)) {
                    staged = false;
                    error_message = "failed to stage image" + std::to_string(i);
                    break;
                }
            }
            if (!staged) {
                std::filesystem::remove_all(view_dir, ec);
                cleanup_outputs(stem);
                set_error(res, 500, error_message);
                return;
            }

            fprintf(stderr,
                    "[trellis-server] generate-trellis2-mv: V=%d fusion=%s seed=%u res=%s uv=%s\n",
                    num_images, fusion.c_str(), p.seed,
                    p.cascade ? std::to_string(p.hr_res).c_str() : "512",
                    p.xatlas ? "xatlas" : "box");
            try {
                const int rc = trellis_run(p);
                if (rc == 0) glb = read_file_bytes(p.output);
            } catch (const std::exception& e) {
                fprintf(stderr, "[trellis-server] generate-trellis2-mv failed: %s\n", e.what());
                error_message = e.what();
            }
            std::filesystem::remove_all(view_dir, ec);
            cleanup_outputs(stem);
        }

        if (glb.empty()) { set_error(res, 500, error_message); return; }
        res.set_content(glb.data(), glb.size(), "model/gltf-binary");
    });

    svr.Get("/capabilities", [&](const httplib::Request&, httplib::Response& res) {
        // 毎回ディレクトリを見る（Studio の設定変更や installer の追加導入を再起動なしで反映する）。
        // SHA256 は再計算しない — manifest の存在 + 契約 + 実在 + サイズ一致まで（設計書 D6）。
        const trellis::ModelSetStatus mv = trellis::inspect_model_set(base.models, trellis::ModelFamily::MV);
        const trellis::ModelSetStatus sv = trellis::inspect_model_set(base.models_sv, trellis::ModelFamily::SV);
        auto one = [](const trellis::ModelSetStatus& st) {
            std::string j = "{\"configured\":" + std::string(st.configured ? "true" : "false") +
                            ",\"available\":" + std::string(st.available ? "true" : "false");
            if (!st.model_set.empty()) j += ",\"model_set\":\"" + json_escape(st.model_set) + "\"";
            if (!st.version.empty())   j += ",\"version\":\"" + json_escape(st.version) + "\"";
            if (!st.family.empty())    j += ",\"model_family\":\"" + json_escape(st.family) + "\"";
            if (!st.available)         j += ",\"reason\":\"" + json_escape(st.reason) + "\"";
            return j + "}";
        };
        const Trellis2Status t2 = inspect_trellis2_models(base.models);
        std::string t2j = "{\"available\":" + std::string(t2.available ? "true" : "false") +
                          ",\"max_images\":8,\"modes\":[\"stochastic\",\"multidiffusion\"]";
        if (!t2.available) t2j += ",\"reason\":\"" + json_escape(t2.reason) + "\"";
        t2j += "}";
        const std::string body = "{\"busy\":" + std::string(g_busy.load() ? "true" : "false") +
                                 ",\"completed\":" + std::to_string(g_completed.load()) +
                                 ",\"mv\":" + one(mv) + ",\"sv\":" + one(sv) +
                                 ",\"trellis2_mv\":" + t2j + "}";
        res.set_content(body, "application/json");
    });

    svr.Post("/generate-sv", [&](const httplib::Request& req, httplib::Response& res) {
        // 許可フィールドは whitelist（設計書 D4）。mesh_scale / transforms / view* /
        // num_views / band / webp は SV では意味を持たないので、来たら 400 にする。
        static const char* const kAllowed[] = { "image", "fov", "seed", "resolution", "uv" };
        for (const auto& kv : req.files) {
            bool ok = false;
            for (const char* a : kAllowed) if (kv.first == a) { ok = true; break; }
            if (!ok) {
                if (kv.first == "mesh_scale") {
                    set_error(res, 400, "generate-sv does not take mesh_scale (the single-view gauge fixes it at 1.0)");
                } else {
                    set_error(res, 400, "unexpected field for generate-sv: " + kv.first);
                }
                return;
            }
        }
        if (!req.has_file("image")) { set_error(res, 400, "missing 'image' file part"); return; }
        const auto& image = req.get_file_value("image");

        // 画像は PNG のみ（pre-matted RGBA が要るため）。デコード前に寸法とバイト数を上限で切る。
        constexpr size_t kMaxImageBytes = 64ull * 1024 * 1024;
        constexpr long long kMaxPixels = 64ll * 1024 * 1024;
        if (image.content.size() > kMaxImageBytes) { set_error(res, 400, "image exceeds 64 MiB"); return; }
        long long iw = 0, ih = 0;
        if (!png_dimensions(image.content, iw, ih)) { set_error(res, 400, "image must be a PNG with an alpha matte"); return; }
        if (iw * ih > kMaxPixels) { set_error(res, 400, "image exceeds 64 megapixels"); return; }

        float fov = trellis::SINGLE_VIEW_DEFAULT_FOV;
        if (req.has_file("fov")) {
            double d = 0.0;
            if (!parse_double_strict(req.get_file_value("fov").content, d) || !(d > 0.0) || !(d < 3.14159265358979323846)) {
                set_error(res, 400, "fov must be a finite number of radians with 0 < fov < pi");
                return;
            }
            fov = (float)d;
        }

        trellis::TrellisParams p = base;
        {
            std::string perr;
            if (!apply_common_overrides(req, p, perr)) { set_error(res, 400, perr); return; }
        }
        // D10: SV のリリース対応解像度は 1024 のみ。
        if (!p.cascade || p.hr_res != 1024) {
            set_error(res, 400, "single-view generation supports resolution 1024 only");
            return;
        }

        // モデルを読み込む前に SV セットの可否を答える（無ければ 503）。
        const trellis::ModelSetStatus sv = trellis::inspect_model_set(base.models_sv, trellis::ModelFamily::SV);
        if (!sv.available) {
            set_error(res, 503, "single-view model set is not available: " + sv.reason);
            return;
        }
        // endpoint ごとに family とディレクトリを強制する（設計書 D3）。
        p.models = base.models_sv;
        p.pixal3d_weights = "sv";
        p.pixal3d_weights_set = true;
        p.mesh_scale_set = false;
        p.num_views = 1;
        p.num_views_set = true;

        const std::string stem = temp_stem();
        const std::filesystem::path view_dir = stem + "-sv";
        p.image.clear();
        p.views = view_dir.string();
        p.output = stem + ".glb";

        std::string glb;
        std::string error_message = "single-view 3D reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            BusyScope busy;
            std::error_code ec;
            if (!std::filesystem::create_directory(view_dir, ec) || ec) {
                set_error(res, 500, "failed to create a fresh single-view staging directory");
                return;
            }
            // 生画像 → 共有 C++ のクロップ → gauge カメラを transforms.json として実体化。
            // 以後は普通の --views 入力なので、読み込み側に SV 専用の分岐は無い。
            const std::string raw = (view_dir / "upload.png").string();
            const std::string cropped = (view_dir / "input.png").string();
            std::string err;
            bool staged = write_file_bytes(raw, image.content);
            if (staged) staged = trellis::preprocess_prematted_rgba_file(raw, cropped, err);
            std::remove(raw.c_str());
            trellis::TransformsFile tf;
            if (staged) staged = trellis::synthesize_single_view_gauge("input.png", fov, tf, err);
            if (staged) staged = trellis::write_transforms_json((view_dir / "transforms.json").string(), tf, err);
            if (!staged) {
                std::filesystem::remove_all(view_dir, ec);
                cleanup_outputs(stem);
                // クロップの拒否（alpha 無し等）は入力の問題なので 400。
                set_error(res, err.empty() ? 500 : 400, err.empty() ? "failed to stage the single-view input" : err);
                return;
            }

            fprintf(stderr, "[trellis-server] generate-sv: %lldx%lld PNG, fov %.6f rad, seed %u, res %d, uv %s\n",
                    iw, ih, (double)fov, p.seed, p.hr_res, p.xatlas ? "xatlas" : "box");
            try {
                int rc = trellis_run(p);
                if (rc == 0) glb = read_file_bytes(p.output);
            } catch (const std::exception& e) {
                fprintf(stderr, "[trellis-server] generate-sv failed: %s\n", e.what());
                error_message = e.what();
            }
            std::filesystem::remove_all(view_dir, ec);
            cleanup_outputs(stem);
        }

        if (glb.empty()) { set_error(res, 500, error_message); return; }
        res.set_content(glb.data(), glb.size(), "model/gltf-binary");
    });

    // gpu=auto when --gpu was omitted: the device is chosen at load time (Vulkan: discrete over
    // UMA, #23) and logged by make_backend as "[trellis] using ... [auto]".
    fprintf(stderr, "[trellis-server] models=%s models-sv=%s gpu=%s listening on http://%s:%d\n",
            base.models.c_str(), base.models_sv.empty() ? "(none)" : base.models_sv.c_str(),
            base.gpu_set ? std::to_string(base.gpu).c_str() : "auto", base.host.c_str(), base.port);
    if (!svr.listen(base.host, base.port)) {
        fprintf(stderr, "[trellis-server] failed to bind %s:%d\n", base.host.c_str(), base.port);
        return 1;
    }
    return 0;
}
