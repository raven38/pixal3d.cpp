// trellis-server — resident HTTP wrapper around the TRELLIS.2 / Pixal3D pipelines.
//
//   GET  /health       -> "ok"
//   POST /generate      multipart/form-data with an "image" file part; optional text
//                        fields "seed", "resolution" (512/1024/1536), "bg_removal"
//                        (threshold|birefnet), "uv" (xatlas = default, unique
//                        chart space; box = faster projection), "band" (narrow-band
//                        DC remesh band width, default 1 — see --band). Returns
//                        model/gltf-binary.
//   POST /generate-mv   Pixal3D multiview endpoint. Multipart must contain a
//                        "transforms" file part (transforms.json) and one or more
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
#include "trellis_run.h"
#include "httplib.h"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
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
    return (dir / ("trellis-req-" + std::to_string(n))).string();
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

void apply_common_overrides(const httplib::Request& req, trellis::TrellisParams& p) {
    if (req.has_file("seed")) p.seed = (uint32_t) atoi(req.get_file_value("seed").content.c_str());
    if (req.has_file("resolution")) p.set_res(atoi(req.get_file_value("resolution").content.c_str()));
    if (req.has_file("uv")) p.xatlas = (req.get_file_value("uv").content == "xatlas");
    if (req.has_file("band")) p.band = atoi(req.get_file_value("band").content.c_str());
    if (req.has_file("webp")) {
        const std::string& w = req.get_file_value("webp").content;
        p.webp = (w == "off" || w == "0" || w == "false") ? 0
               : (w == "on"  || w == "1" || w == "true")  ? 1 : -1;
    }
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
        apply_common_overrides(req, p);
        if (req.has_file("bg_removal")) p.birefnet = (req.get_file_value("bg_removal").content == "birefnet") ? 1 : 0;

        const std::string stem = temp_stem();
        p.image  = stem + ".png";
        p.views.clear();
        p.output = stem + ".glb";

        std::string glb;
        std::string error_message = "3D reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
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
        if (!req.has_file("transforms")) {
            set_error(res, 400, "missing 'transforms' file part");
            return;
        }

        trellis::TrellisParams p = base;
        apply_common_overrides(req, p);
        if (req.has_file("num_views")) p.num_views = atoi(req.get_file_value("num_views").content.c_str());
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
            std::error_code ec;
            std::filesystem::create_directories(view_dir, ec);
            if (ec) {
                set_error(res, 500, "failed to create multiview staging directory");
                return;
            }

            bool staged_ok = write_file_bytes((view_dir / "transforms.json").string(),
                                               req.get_file_value("transforms").content);
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

            fprintf(stderr,
                    "[trellis-server] generate-mv: %d staged views, seed %u, res %d, uv %s, num_views=%d\n",
                    staged_views, p.seed, p.hr_res, p.xatlas ? "xatlas" : "box", p.num_views);
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

    fprintf(stderr, "[trellis-server] models=%s gpu=%d listening on http://%s:%d\n",
            base.models.c_str(), base.gpu, base.host.c_str(), base.port);
    if (!svr.listen(base.host, base.port)) {
        fprintf(stderr, "[trellis-server] failed to bind %s:%d\n", base.host.c_str(), base.port);
        return 1;
    }
    return 0;
}
