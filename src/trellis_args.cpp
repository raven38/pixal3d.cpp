#include "trellis_args.h"

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <string>

namespace trellis {
namespace {
// atoi/atof は "abc" を 0、"4x" を 4 として黙って受けるので、全体消費を要求する厳密版を使う。
bool parse_int_strict(const char* s, int& out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno == ERANGE || !end || *end != '\0' || v < INT_MIN || v > INT_MAX) return false;
    out = (int)v;
    return true;
}
bool parse_float_strict(const char* s, float& out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s, &end);
    if (!end || *end != '\0') return false;
    out = (float)v;
    return true;
}
} // namespace


void print_usage(const char* argv0, bool server) {
    if (server) {
        fprintf(stderr,
            "usage: %s [--host H] [--port P] [--models DIR] [--models-sv DIR] [--gpu N]\n"
            "           [generation defaults...]\n"
            "\n"
            "      --models-sv DIR  single-view (SV) model set for POST /generate-sv. The MV and\n"
            "                       SV sets cannot share one directory (five files have the same\n"
            "                       names with different contents), so this is a second directory.\n"
            "                       Without it /generate-sv returns 503 and GET /capabilities\n"
            "                       reports sv.configured=false.\n",
            argv0);
    } else {
        fprintf(stderr,
            "usage: %s <image.png> <out.glb> [options]\n"
            "   or: %s --image <image.png> --output <out.glb> [options]\n",
            argv0, argv0);
    }
    fprintf(stderr,
        "\n"
        "  -i, --image PATH        input image                  (image->3D)\n"
        "  -o, --output PATH       output .glb                  (default model.glb)\n"
        "      --copyright TEXT    glTF asset.copyright metadata\n"
        "  -m, --models DIR        GGUF model directory\n"
        "      --gpu N             explicit GPU index, <0 = CPU (default: auto)\n"
        "  -s, --seed N            RNG seed                     (default 42)\n"
        "      --res 512|1024|1536 geometry resolution\n"
        "      --max-tokens N      HR token budget              (default 49152)\n"
        "      --views DIR         Pixal3D multiview mode: DIR has transforms.json + RGBA\n"
        "      --sv-image PATH     Pixal3D single-view mode: one pre-matted RGBA image.\n"
        "                          Crops it like the reference preprocess, synthesizes the\n"
        "                          front gauge camera (mesh_scale 1.0) and runs the SV flow\n"
        "                          weights. Mutually exclusive with --views and an input image.\n"
        "                          The staged crop + transforms.json are kept next to the GLB.\n"
        "      --fov RAD           --sv-image gauge FOV in radians (default 0.349066 = 20 deg).\n"
        "      --pixal3d-weights V sv|mv flow weights (default mv). sv expects 1 view;\n"
        "                          mv expects 4. Requires --views.\n"
        "                          views (frame 0 = main/front view). Mutually exclusive\n"
        "                          with the positional/--image input; mandatory cascade\n"
        "                          (--res 512 is not supported -- no res-512 texture flow)\n"
        "      --num-views N       use only the first N transforms.json frames (default: all)\n"
        "      --mesh-scale F      Pixal3D projection scale (> 0). Required when --views DIR\n"
        "                          has no transforms.json (exactly 4 turntable views, natural\n"
        "                          filename order = front, right, back, left, elevation 0,\n"
        "                          FOV 20 deg); with transforms.json it overrides its value.\n"
        "      --bg-removal MODE   threshold | birefnet   (default: auto -- a pre-matted\n"
        "                          image keeps its alpha; otherwise BiRefNet when its model\n"
        "                          is present. The plain threshold matte cuts out specular\n"
        "                          highlights, which the flow then turns into holes.)\n"
        "      --birefnet          alias for --bg-removal birefnet\n"
        "      --no-texture        geometry only\n"
        "      --xatlas            xatlas UV unwrap (default)\n"
        "      --box-uv            voxel-native box projection (faster)\n"
        "      --band N            narrow-band DC remesh band width (default: auto —\n"
        "                          res/512, i.e. 1 @512 / 2 @1024, which suppresses the\n"
        "                          res-1024 outer-skin speckle; N forces that width)\n"
        "      --decim GRID        legacy cluster-grid decimation (default: quadric\n"
        "                          simplify to 300K faces @1024 / 150K @512; 0 = none)\n"
        "      --atlas PX          UV atlas size (default 2048 @1024 / 1024 @512)\n"
        "      --tex-res N         texture PBR resolution 512/1024 (default: auto — drops\n"
        "                          a dense res-1024 decode to a clean res-512 PBR volume)\n"
        "      --webp on|off       encode GLB textures as WebP (default: on when built with\n"
        "                          WebP support; off = PNG)\n"
        "      --dump-bg           also write the background-removal cutout as <out>_cutout.png\n"
        "      --dump-post PATH    write the raw decoded mesh + sparse PBR volume to PATH and\n"
        "                          exit (skips remesh/decimate/UV/bake/GLB) -- for external\n"
        "                          post-processing pipelines\n"
        "      --bg-only           background removal only: write the cutout and skip the rest\n"
        "      --f32               f32 sparse-conv compute\n"
        "      --no-fa             disable FlashAttention\n"
        "      --profile           time one forward of every flow DiT node by node and print\n"
        "                          the per-role / per-op / per-block breakdown\n"
        "      --require-gpu       refuse CPU fallback\n"
        "      --threads N         CPU backend threads      (default all cores)\n"
        "      --gss F  --gsh F    guidance strengths\n"
        "      --host H  --port P  trellis-server bind address\n"
        "      --voxply            also dump the voxel point cloud as .ply\n"
        "      --dump-slat         dump the structured latent to disk\n"
        "  -h, --help              show this help\n");
}

bool parse_args(int argc, char** argv, TrellisParams& p) {
    // Positionals are collected and only assigned to image/output after the whole argv has
    // been scanned, since --views (which can appear anywhere) changes what the first bare
    // positional means (output, not image) -- see the assignment below.
    std::string pos[2]; int npos = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "[trellis] %s needs a value\n", name); return nullptr; }
            return argv[++i];
        };
        auto need = [&](const char* name) -> const char* {
            const char* v = next(name);
            return v;
        };

        if      (a == "-h" || a == "--help")    { p.help = true; return false; }
        else if (a == "-i" || a == "--image")   { const char* v = need(a.c_str()); if (!v) return false; p.image = v; }
        else if (a == "-o" || a == "--output")  { const char* v = need(a.c_str()); if (!v) return false; p.output = v; }
        else if (a == "--copyright")            { const char* v = need(a.c_str()); if (!v) return false; p.copyright = v; }
        else if (a == "-m" || a == "--models")  { const char* v = need(a.c_str()); if (!v) return false; p.models = v; }
        else if (a == "--gpu")                  { const char* v = need(a.c_str()); if (!v) return false;
                                                  if (!parse_int_strict(v, p.gpu)) {
                                                      fprintf(stderr, "[trellis] --gpu expects an integer, got '%s'\n", v); return false;
                                                  }
                                                  p.gpu_set = true; }
        else if (a == "-s" || a == "--seed")    { const char* v = need(a.c_str()); if (!v) return false; p.seed = (uint32_t)atoi(v); }
        else if (a == "--res")                  { const char* v = need(a.c_str()); if (!v) return false; p.set_res(atoi(v)); }
        else if (a == "--max-tokens")           { const char* v = need(a.c_str()); if (!v) return false; p.max_tokens = atoi(v); }
        else if (a == "--views")                { const char* v = need(a.c_str()); if (!v) return false; p.views = v; }
        else if (a == "--models-sv")            { const char* v = need(a.c_str()); if (!v) return false; p.models_sv = v; }
        else if (a == "--sv-image")             { const char* v = need(a.c_str()); if (!v) return false; p.sv_image = v; }
        else if (a == "--fov")                  { const char* v = need(a.c_str()); if (!v) return false;
                                                  char* end = nullptr; const double d = strtod(v, &end);
                                                  if (!end || *end != '\0' || !std::isfinite(d)) {
                                                      fprintf(stderr, "[trellis] --fov expects a finite number of radians, got '%s'\n", v); return false; }
                                                  p.sv_fov = (float)d; p.sv_fov_set = true; }
        else if (a == "--pixal3d-weights")      { const char* v = need(a.c_str()); if (!v) return false;
                                                  const std::string w = v;
                                                  if (w != "sv" && w != "mv") {
                                                      fprintf(stderr, "[trellis] --pixal3d-weights expects 'sv' or 'mv', got '%s'\n", v); return false; }
                                                  p.pixal3d_weights = w; p.pixal3d_weights_set = true; }
        else if (a == "--num-views")            { const char* v = need(a.c_str()); if (!v) return false;
                                                  if (!parse_int_strict(v, p.num_views) || p.num_views <= 0) {
                                                      fprintf(stderr, "[trellis] --num-views expects a positive integer, got '%s'\n", v); return false; }
                                                  p.num_views_set = true; }
        else if (a == "--mesh-scale")           { const char* v = need(a.c_str()); if (!v) return false;
                                                  if (!parse_float_strict(v, p.mesh_scale) || !std::isfinite(p.mesh_scale) || p.mesh_scale <= 0.0f) {
                                                      fprintf(stderr, "[trellis] --mesh-scale expects a finite value > 0, got '%s'\n", v); return false; }
                                                  p.mesh_scale_set = true; }
        else if (a == "--bg-removal")           { const char* v = need(a.c_str()); if (!v) return false; p.birefnet = (std::strcmp(v, "birefnet") == 0) ? 1 : 0; }
        else if (a == "--birefnet")             { p.birefnet = 1; }
        else if (a == "--no-texture")           { p.texture = false; }
        else if (a == "--xatlas")               { p.xatlas = true; }
        else if (a == "--box-uv")               { p.xatlas = false; }
        else if (a == "--dump-post")            { const char* v = need(a.c_str()); if (!v) return false; p.dump_post = v; }
        else if (a == "--band")                 { const char* v = need(a.c_str()); if (!v) return false; p.band = atoi(v); }
        else if (a == "--decim")                { const char* v = need(a.c_str()); if (!v) return false; p.decim = atoi(v); }
        else if (a == "--atlas" || a == "--tex"){ const char* v = need(a.c_str()); if (!v) return false; p.tex = atoi(v); }
        else if (a == "--tex-res")              { const char* v = need(a.c_str()); if (!v) return false; p.tex_res = atoi(v); }
        else if (a == "--webp")                 { const char* v = need(a.c_str()); if (!v) return false;
                                                  p.webp = (std::strcmp(v,"off")==0 || std::strcmp(v,"0")==0 || std::strcmp(v,"false")==0) ? 0
                                                         : (std::strcmp(v,"on")==0 || std::strcmp(v,"1")==0 || std::strcmp(v,"true")==0) ? 1 : -1; }
        else if (a == "--dump-bg")              { p.dump_bg = true; }
        else if (a == "--bg-only")              { p.bg_only = true; p.dump_bg = true; }
        else if (a == "--f32")                  { p.f32 = true; }
        else if (a == "--no-fa")                { p.no_fa = true; }
        else if (a == "--profile")              { p.profile = true; }
        else if (a == "--require-gpu")          { p.require_gpu = true; }
        else if (a == "--threads")              { const char* v = need(a.c_str()); if (!v) return false; p.threads = atoi(v); }
        else if (a == "--gss")                  { const char* v = need(a.c_str()); if (!v) return false; p.gss = (float)atof(v); }
        else if (a == "--gsh")                  { const char* v = need(a.c_str()); if (!v) return false; p.gsh = (float)atof(v); }
        else if (a == "--host")                 { const char* v = need(a.c_str()); if (!v) return false; p.host = v; }
        else if (a == "--port")                 { const char* v = need(a.c_str()); if (!v) return false; p.port = atoi(v); }
        else if (a == "--voxply")               { p.voxply = true; }
        else if (a == "--dump-slat")            { p.dump_slat = true; }
        else if (!a.empty() && a[0] == '-')     { fprintf(stderr, "[trellis] unknown option: %s\n", a.c_str()); return false; }
        else if (npos < 2)                      { pos[npos++] = a; }
        else                                    { fprintf(stderr, "[trellis] unexpected argument: %s\n", a.c_str()); return false; }
    }

    // Assign positionals now that --views (if any) is known: normally <image> <out.glb>;
    // in --views mode there is no positional image, so the lone positional is the output.
    // --pixal3d-weights は Pixal3D (--views) 経路専用。指定だけして TRELLIS.2 経路で走ると
    // 「SV を実行した」と誤認されるので、ここで落とす。
    if (p.pixal3d_weights_set && p.views.empty() && p.sv_image.empty()) {
        fprintf(stderr, "[trellis] --pixal3d-weights requires --views DIR or --sv-image PATH (it selects Pixal3D flow weights)\n");
        return false;
    }
    if (!p.sv_image.empty()) {
        if (!p.views.empty()) { fprintf(stderr, "[trellis] --sv-image and --views are mutually exclusive\n"); return false; }
        if (!p.image.empty()) { fprintf(stderr, "[trellis] --sv-image and --image/positional image are mutually exclusive\n"); return false; }
        // --pixal3d-weights mv を SV 入力に付けても通さない（V=1 で MV 重みを回すのは
        // 既知の破綻モードで、デスクトップからは選べないようにする）。
        if (p.pixal3d_weights_set && p.pixal3d_weights != "sv") {
            fprintf(stderr, "[trellis] --sv-image always uses the single-view flow weights; drop --pixal3d-weights mv\n");
            return false;
        }
        p.pixal3d_weights = "sv";
        p.pixal3d_weights_set = true;
        if (!(p.sv_fov > 0.0f) || !(p.sv_fov < 3.14159274f) || !std::isfinite(p.sv_fov)) {
            fprintf(stderr, "[trellis] --fov must satisfy 0 < fov < pi radians\n");
            return false;
        }
        // D10: SV のリリース対応解像度は 1024 のみ。1536 は API/UI/CLI とも受けない。
        if (!p.cascade || p.hr_res != 1024) {
            fprintf(stderr, "[trellis] --sv-image supports --res 1024 only in this release\n");
            return false;
        }
        // --views と同じく、位置引数は出力 GLB 1 つだけ。2 つあるなら 1 つ目は入力画像の
        // つもりなので、汎用の "unexpected argument" ではなく排他として落とす。
        if (npos > 1) {
            fprintf(stderr, "[trellis] --sv-image and a positional input image are mutually exclusive\n");
            return false;
        }
        if (npos == 1) p.output = pos[0];
        return true;
    }
    if (p.sv_fov_set && p.sv_image.empty()) {
        fprintf(stderr, "[trellis] --fov requires --sv-image PATH (it sets the single-view gauge camera)\n");
        return false;
    }
    if (!p.views.empty()) {
        if (!p.image.empty()) { fprintf(stderr, "[trellis] --views and --image/positional image are mutually exclusive\n"); return false; }
        if (npos > 1)         { fprintf(stderr, "[trellis] unexpected argument: %s\n", pos[1].c_str()); return false; }
        if (npos == 1) p.output = pos[0];
    } else {
        if (npos >= 1) p.image  = pos[0];
        if (npos >= 2) p.output = pos[1];
    }
    return true;
}

}  // namespace trellis
