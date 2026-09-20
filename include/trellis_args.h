#pragma once
#include <cstdint>
#include <string>

namespace trellis {

// Cross-module runtime flags. Set from parsed params at the start of trellis_run;
// the modules that own them read them with an environment fallback so test
// binaries (which don't parse args) keep their historical TRELLIS_* behavior.
extern bool g_sparse_cast_f32;  // defined in sparse.cpp        (TRELLIS_F32)
extern bool g_no_fa;            // defined in dit.cpp           (TRELLIS_NOFA)
extern bool g_profile;          // defined in dit.cpp; --profile (per-op flow DiT timing)
extern bool g_require_gpu;      // defined in trellis_model.cpp (TRELLIS_REQUIRE_GPU)
extern bool g_gpu_auto;         // defined in trellis_model.cpp; true when --gpu was not specified
extern int  g_cpu_threads;      // defined in trellis_model.cpp (TRELLIS_THREADS)

// Every knob for one TRELLIS.2 image->3D run. Resolved as default -> environment
// (the historical TRELLIS_* / GSS / GSH names) -> CLI flag, with the CLI winning.
// trellis-cli and trellis-server share the parser: the server runs it once for its
// launch defaults, then per request to apply overrides (resolution, bg removal, ...).
struct TrellisParams {
    std::string image;                                          // input image (image->3D)
    std::string views;          // Pixal3D multiview mode: directory with transforms.json +
                                 // RGBA views (--views DIR). Mutually exclusive with `image`;
                                 // non-empty selects the Pixal3D cascade instead of TRELLIS.2.
    int num_views = 0;           // --num-views N: use only the first N transforms.json frames
                                 // (0/unset = all frames).
    bool num_views_set = false;  // --num-views が明示されたか（0 や負値と未指定を区別する）
    float mesh_scale = 0.0f;     // --mesh-scale F: transforms.json が無いときの必須スケール。
                                 // ある場合は parse 成功後の上書きとして働く。
    bool mesh_scale_set = false; // --mesh-scale が明示されたか（0.0 は無効値であって未指定ではない）
    // --pixal3d-weights sv|mv: どちらの flow 重み系列を読むか。公式は 4 段それぞれに単視点版
    // （接尾辞なし）と多視点版（_mv）を配布しており、config とテンソル構造は同一で重みだけが
    // 違う。既定は mv（従来の挙動）。共有の 5 モデルは系列に依存しないので名前を変えない。
    // 注意: 本フラグは「SV 重みを --views 規約（transforms.json、frames 1 件）で動かす」ための
    // もので、公式 SV パイプライン（inference.py の camera_params / MoGe 推定）の再現ではない。
    std::string pixal3d_weights = "mv";
    bool pixal3d_weights_set = false;
    // --sv-image PATH: 画像 1 枚から Pixal3D SV 推論を回す入口。`image`（TRELLIS.2）とも
    // `views`（MV）とも排他で、指定されると SV 重みを選ぶ。視点数からモードを推測する経路は
    // 作らない（docs/design/2026-09-19-desktop-0.10.0-sv-parity.md D1）。実行時は
    // <出力 GLB の stem>.svviews/ にクロップ済み input.png と合成 transforms.json を書き、
    // その普通の --views 入力として走る。staging は消さない（IoU ゲートが読むため）。
    std::string sv_image;
    // --fov RAD: SV gauge カメラの水平 FOV。0 < fov < pi。--sv-image 専用。
    float sv_fov = 0.3490658503988659f;   // 20 deg
    bool  sv_fov_set = false;
    std::string output = "model.glb";                           // output .glb
    std::string copyright;                                      // glTF asset.copyright metadata
    std::string models = "models";              // GGUF dir; override with --models DIR
    // --models-sv DIR: trellis-server 専用。単一視点(SV)モデルセットのディレクトリ。
    // MV(f16) と SV(q8_0) は共有 5 ファイルが同名で中身が違うため 1 つのディレクトリに
    // 同居できない（設計書 D3）。未指定なら /generate-sv は 503 を返す。
    std::string models_sv;
    std::string host   = "127.0.0.1";                           // trellis-server only
    int      port = 8080;                                       // trellis-server only
    int      gpu  = 0;                                          // >=0 GPU index, <0 CPU
    bool     gpu_set = false;                                    // --gpu was explicitly specified
    uint32_t seed = 0;

    bool cascade    = true;     // 1024 cascade (default); --res 512 selects the light path
    int  hr_res     = 1024;     // HR cascade target resolution (1024 / 1536)
    int  max_tokens = 49152;    // HR token budget (backoff floors at 1024)

    int birefnet = -1;          // bg removal: 1 BiRefNet, 0 white-threshold, -1 auto
                                // (auto: keep a pre-matted image's alpha; else BiRefNet when
                                // birefnet.gguf is present; threshold as last resort. The
                                // threshold matte reads specular highlights [min(RGB)>=232]
                                // as background and the model then generates holes there.)
    bool texture  = true;       // texture flow + UV bake (else geometry-only)
    bool xatlas   = true;       // xatlas UV unwrap (else voxel-native box projection)
    int  band     = 0;          // narrow-band DC remesh band width (remesh_dc.h).
                                //   0 = auto: scale with resolution (res/512) so the
                                //   smoothing offset is resolution-independent — 1 @512,
                                //   2 @1024 — which suppresses the res-1024 "outer-skin"
                                //   speckle (issue #22). >0 forces that width (e.g. 1 for
                                //   the thin-wall reference look, 2 for a thicker shell).
    int  decim    = -1;         // decimation cluster grid   (-1 => per-cascade default)
    int  tex      = -1;         // UV atlas size in px        (-1 => per-cascade default)
    int  tex_res  = -1;         // texture PBR resolution: -1 => auto (drop dense res-1024 tex to
                                //   512, whose clean coarse PBR bakes onto the res-1024 mesh
                                //   without the partial-coverage "skin" speckle); else force 512/1024
    int  webp     = -1;         // GLB texture encoding: -1 auto (WebP if built with it), 1 on, 0 off (PNG)
    bool f32      = false;      // f32 sparse-conv compute
    bool no_fa    = false;      // disable FlashAttention (manual softmax)
    bool profile  = false;      // per-op / per-block timing of one forward per flow DiT
    bool require_gpu = false;   // refuse CPU fallback if no GPU is usable
    int  threads  = 0;          // CPU backend thread count; 0 = all cores
    float gss = 7.5f;           // sparse-structure guidance strength
    float gsh = 7.5f;           // shape-SLAT guidance strength
    bool voxply = false;        // dump out/myvox.ply              (debug)
    bool dump_slat = false;     // dump /tmp/hr_slat.bin           (debug)
    bool dump_bg = false;       // also write the bg-removal cutout as <out>_cutout.png
    std::string dump_post;      // write the raw decoded mesh + sparse PBR volume to this
                                // path and exit (skips remesh/decimate/UV/bake/GLB) --
                                // machine-readable output for external post-processing
                                // pipelines (binary: i32 V,F,Mv,res; f32 verts[V*3];
                                // i32 faces[F*3]; i32 coords[Mv*3]; f32 pbr6[Mv*6];
                                // same layout as the TRELLIS_DUMP_POST debug env)
    bool bg_only = false;       // background removal only: write the cutout and skip the rest

    bool help = false;          // --help requested

    // 512 -> light single-res path; 1024/1536 -> cascade with that HR target.
    void set_res(int res) {
        if (res <= 512) { cascade = false; hr_res = 512; }
        else            { cascade = true;  hr_res = res; }
    }
};

void print_usage(const char* argv0, bool server);

// Apply environment fallbacks, then parse argv (CLI wins). The first two bare
// (non-flag) positionals fill `image` then `output`. Returns false on a parse
// error OR when --help was requested; check p.help to tell them apart.
bool parse_args(int argc, char** argv, TrellisParams& p);

}  // namespace trellis
