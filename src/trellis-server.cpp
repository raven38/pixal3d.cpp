// trellis-server — resident HTTP wrapper around the TRELLIS.2 image->3D pipeline.
//
//   GET  /health     -> "ok"
//   POST /generate_mv multipart/form-data for the Pixal3D multiview cascade:
//                      one "transforms" part (the transforms.json bytes) plus one
//                      "views" file part per frame, named exactly as the frames'
//                      file_path. Optional text fields "seed", "num_views",
//                      "uv", "band", "webp". Returns model/gltf-binary.
//                      transforms.json MUST carry mesh_scale: without it the
//                      pipeline silently defaults to 1.0 and, when that does not
//                      match how the views were rendered, ProjGrid samples the
//                      object at the wrong scale and the reconstruction collapses
//                      with no error (measured 2026-09-08: silhouette IoU 0.107
//                      instead of 0.909). The endpoint rejects the request instead.
//   POST /generate    multipart/form-data with an "image" file part; optional text
//                      fields "seed", "resolution" (512/1024/1536), "bg_removal"
//                      (threshold|birefnet), "uv" (xatlas = default, unique
//                      chart space; box = faster projection), "band" (narrow-band
//                      DC remesh band width, default 1 — see --band). Returns
//                      model/gltf-binary.
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
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), (std::streamsize) data.size());
    return f.good();
}

// std::tmpnam on MSVC yields drive-root paths ("\sXXX.N") that a non-elevated
// process cannot write; stage scratch files in the real temp directory instead.
std::string temp_stem() {
    static std::atomic<unsigned> counter{0};
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = ".";
    auto n = counter.fetch_add(1);
    return (dir / ("trellis-req-" + std::to_string(n))).string();
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

    // Pixal3D multiview: N 枚の posed RGBA + transforms.json -> textured GLB。
    // trellis_run() は p.views が非空なら Pixal3D cascade に入る（trellis_cli.cpp）。
    svr.Post("/generate_mv", [&](const httplib::Request& req, httplib::Response& res) {
        auto fail = [&](int code, const std::string& msg) {
            res.status = code;
            std::string esc;
            for (char c : msg) { if (c == '"' || c == '\\') esc += '\\'; esc += c; }
            res.set_content("{\"error\":\"" + esc + "\"}", "application/json");
        };
        if (!req.has_file("transforms")) return fail(400, "missing 'transforms' part (transforms.json bytes)");
        const std::string tj = req.get_file_value("transforms").content;
        const auto views = req.get_file_values("views");
        if (views.empty()) return fail(400, "missing 'views' file parts");

        // mesh_scale は必須。無いと既定 1.0 で静かに壊れる（上のコメント参照）。
        if (tj.find("\"mesh_scale\"") == std::string::npos)
            return fail(400, "transforms.json has no mesh_scale; the pipeline would silently "
                             "default to 1.0 and produce a collapsed reconstruction");

        const std::string stem = temp_stem();
        const std::string dir = stem + "_views";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) return fail(500, "failed to stage views directory");
        auto cleanup = [&] { std::error_code e; std::filesystem::remove_all(dir, e); };

        if (!write_file_bytes(dir + "/transforms.json", tj)) { cleanup(); return fail(500, "failed to stage transforms.json"); }
        for (const auto& v : views) {
            // filename は transforms.json の file_path と一致している必要がある。
            // ディレクトリ脱出を防ぐため区切り文字を含む名前は拒否する。
            if (v.filename.empty() || v.filename.find('/') != std::string::npos
                || v.filename.find('\\') != std::string::npos || v.filename.find("..") != std::string::npos) {
                cleanup(); return fail(400, "view part has an unsafe or empty filename: " + v.filename);
            }
            if (!write_file_bytes(dir + "/" + v.filename, v.content)) { cleanup(); return fail(500, "failed to stage " + v.filename); }
        }

        trellis::TrellisParams p = base;
        p.views = dir;
        p.image.clear();
        if (req.has_file("seed"))      p.seed = (uint32_t) atoi(req.get_file_value("seed").content.c_str());
        if (req.has_file("num_views")) p.num_views = atoi(req.get_file_value("num_views").content.c_str());
        if (req.has_file("uv"))        p.xatlas = (req.get_file_value("uv").content == "xatlas");
        if (req.has_file("band"))      p.band = atoi(req.get_file_value("band").content.c_str());
        if (req.has_file("webp")) {
            const std::string& w = req.get_file_value("webp").content;
            p.webp = (w == "off" || w == "0" || w == "false") ? 0
                   : (w == "on"  || w == "1" || w == "true")  ? 1 : -1;
        }
        p.output = stem + ".glb";

        std::string glb, error_message = "3D reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            fprintf(stderr, "[trellis-server] generate_mv: %zu view(s), seed %u, uv %s\n",
                    views.size(), p.seed, p.xatlas ? "xatlas" : "box");
            try {
                if (trellis_run(p) == 0) glb = read_file_bytes(p.output);
            } catch (const std::exception& e) {
                fprintf(stderr, "[trellis-server] generate_mv failed: %s\n", e.what());
                error_message = e.what();
            }
            std::remove(p.output.c_str());
        }
        cleanup();
        if (glb.empty()) return fail(500, error_message);
        res.set_content(glb, "model/gltf-binary");
    });

    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("image")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'image' file part\"}", "application/json");
            return;
        }
        const auto& image = req.get_file_value("image");

        // Per-request params start from the launch defaults, then apply overrides.
        trellis::TrellisParams p = base;
        if (req.has_file("seed")) p.seed = (uint32_t) atoi(req.get_file_value("seed").content.c_str());
        if (req.has_file("resolution")) p.set_res(atoi(req.get_file_value("resolution").content.c_str()));
        if (req.has_file("bg_removal")) p.birefnet = (req.get_file_value("bg_removal").content == "birefnet") ? 1 : 0;
        if (req.has_file("uv")) p.xatlas = (req.get_file_value("uv").content == "xatlas");
        if (req.has_file("band")) p.band = atoi(req.get_file_value("band").content.c_str());
        if (req.has_file("webp")) {
            const std::string& w = req.get_file_value("webp").content;
            p.webp = (w == "off" || w == "0" || w == "false") ? 0
                   : (w == "on"  || w == "1" || w == "true")  ? 1 : -1;
        }

        const std::string stem = temp_stem();
        p.image  = stem + ".png";
        p.output = stem + ".glb";

        std::string glb;
        std::string error_message = "3D reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            if (!write_file_bytes(p.image, image.content)) {
                res.status = 500;
                res.set_content("{\"error\":\"failed to stage input image\"}", "application/json");
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
            std::remove(p.output.c_str());
            // trellis_run also writes sibling debug artifacts; clean them up too.
            std::remove((stem + ".ply").c_str());
            std::remove((stem + "_base.png").c_str());
        }

        if (glb.empty()) {
            res.status = 500;
            std::string escaped;
            for (char c : error_message) {
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
            res.set_content("{\"error\":\"" + escaped + "\"}", "application/json");
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
