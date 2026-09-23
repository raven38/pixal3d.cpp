// Pixal3D postprocess-tail replay (#83): runs pixal3d_write_production_glb -- the exact tail the
// browser real/partial/full E2E drivers use -- on a decoded mesh + PBR voxel volume dumped by
// `PIXAL3D_DUMP_FIXTURE=<dir> trellis-test-pixal3d-real-e2e ...` (tail_{verts,faces,coords,pbr,res}.npy).
// Built natively (trellis-pixal3d-tail-replay) and as a wasm32 Node program
// (web/ss: pixal3d-tail-replay-node, -sMAXIMUM_MEMORY=4GiB, same allocator as the browser build) so
// the 4 GiB host-heap behaviour of the browser tail can be measured and regression-tested in seconds
// instead of after a ~50-minute browser flow run. Every tail stage logs grown/live heap (plog).
//
//   pixal3d-tail-replay <dump_dir> <out.glb> [--remesh-res N] [--target-faces N] [--atlas N]
//                       [--budget-mib N] [--prefill-mib N]
//   --budget-mib   host budget for the tail's bounded policy (0 = unbounded, the native default;
//                  the browser drivers pass their remaining wasm32 heap).
//   --prefill-mib  allocate and keep N MiB before the tail (emulates the heap the flow stages leave
//                  behind in a full browser run).
// Defaults are the browser drivers' options (remesh 512, 500k faces, atlas 4096, box UV).
#include "pixal3d_postprocess.h"
#include "npy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

using namespace trellis;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <dump_dir> <out.glb> [--remesh-res N] [--target-faces N] [--atlas N]"
                        " [--budget-mib N] [--prefill-mib N]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1], out = argv[2];
    Pixal3dPostprocessOptions opt;
    opt.remesh_band = 1; opt.use_xatlas = false; opt.use_webp = false; opt.texture_size = 4096;
    opt.remesh_res = 512; opt.target_faces = 500000;
    long prefill_mib = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i]; const long v = atol(argv[i + 1]);
        if (a == "--remesh-res") opt.remesh_res = (int)v;
        else if (a == "--target-faces") opt.target_faces = (int)v;
        else if (a == "--atlas") opt.texture_size = (int)v;
        else if (a == "--budget-mib") opt.host_budget_bytes = (uint64_t)v << 20;
        else if (a == "--prefill-mib") prefill_mib = v;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    try {
        std::vector<std::vector<char>> prefill;
        for (long m = 0; m < prefill_mib; m += 64) prefill.emplace_back((size_t)64 << 20, 1);
        Mesh mesh;
        { auto v = npy::load(dir + "/tail_verts.npy"); mesh.verts = std::move(v.data); }
        { auto f = npy::load_i32(dir + "/tail_faces.npy"); mesh.faces = std::move(f.data); }
        auto c = npy::load_i32(dir + "/tail_coords.npy");
        std::vector<std::array<int,3>> coords((size_t)c.shape[0]);
        for (size_t i = 0; i < coords.size(); ++i) coords[i] = {c.data[i*3], c.data[i*3+1], c.data[i*3+2]};
        std::vector<int32_t>().swap(c.data);
        auto pbr = npy::load(dir + "/tail_pbr.npy");
        const int res = npy::load_i32(dir + "/tail_res.npy").data[0];
        printf("tail input: V=%d F=%d voxels=%zu res=%d prefill=%ld MiB\n", mesh.V(), mesh.F(), coords.size(), res, prefill_mib);
        std::string rep;
        const bool ok = pixal3d_write_production_glb(out, std::move(mesh), coords, pbr.data, res, opt, &rep);
        printf("TAIL_REPLAY_RESULT: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::bad_alloc&) {
        printf("TAIL_REPLAY_RESULT: UNHANDLED std::bad_alloc\n");
        return 3;
    } catch (const std::exception& e) {
        printf("TAIL_REPLAY_RESULT: EXCEPTION %s\n", e.what());
        return 4;
    }
}
