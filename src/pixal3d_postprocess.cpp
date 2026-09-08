#include "pixal3d_postprocess.h"
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "mesh_glb.h"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <sstream>
#ifdef __EMSCRIPTEN__
#include <emscripten/heap.h>
#endif

namespace trellis {
namespace {
static void apply_reference_frame(std::vector<float>& verts) {
    for (size_t i=0;i+2<verts.size();i+=3) { float x=verts[i], y=verts[i+1], z=verts[i+2]; verts[i]=-x; verts[i+1]=z; verts[i+2]=y; }
}
// wasm32 の 4 GiB ヒープでは tail の途中で std::bad_alloc になることがあり、そのとき
// report 文字列は呼び出し側に返らない。どの段で落ちたかを残すため、各段の begin/end は
// report に積むと同時に stdout へ即時 flush する。ブラウザでは heap の実サイズも出す。
static void plog(std::ostringstream& os, const char* fmt, ...) {
    char b[512]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    std::string line = b;
#ifdef __EMSCRIPTEN__
    char h[64]; snprintf(h, sizeof h, " [heap %.0f MB]", emscripten_get_heap_size() / 1048576.0);
    line += h;
#endif
    line += "\n";
    os << line; fputs(line.c_str(), stdout); fflush(stdout);
}
}
bool pixal3d_write_production_glb(const std::string& out_glb, Mesh mesh,
                                  const std::vector<std::array<int,3>>& coords,
                                  const std::vector<float>& pbr6, int res,
                                  const Pixal3dPostprocessOptions& opt,
                                  std::string* report) {
    std::ostringstream os;
    if (mesh.F() <= 0 || coords.empty() || pbr6.size() != coords.size()*6) {
        plog(os, "postprocess: invalid input mesh/PBR"); if(report)*report=os.str(); return false;
    }
    const int rres = opt.remesh_res > 0 ? opt.remesh_res : res;
    plog(os, "postprocess: input V=%d F=%d voxels=%zu remesh_res=%d target_faces=%d atlas=%d",
         mesh.V(), mesh.F(), coords.size(), rres, opt.target_faces, opt.texture_size);
    plog(os, "postprocess: weld begin");
    weld_vertices(mesh.verts, mesh.faces, nullptr, 1.0f / ((float)res * 8.0f));
    fill_small_holes(mesh.faces);
    plog(os, "postprocess: weld done V=%d F=%d", mesh.V(), mesh.F());
    plog(os, "postprocess: tri_bvh begin F=%d", mesh.F());
    TriBvh bvh = TriBvh::build(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F());
    plog(os, "postprocess: tri_bvh done");
    plog(os, "postprocess: remesh begin res=%d band=%d", rres, opt.remesh_band);
    Mesh rm = remesh_narrow_band_dc(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), bvh, rres, opt.remesh_band);
    if (rm.F() > 0) {
        clean_mesh(rm.V(), rm.faces);
        int ndrop = drop_small_components(rm.verts, rm.faces, 0.02f);
        plog(os, "postprocess: remesh done V=%d F=%d dropped=%d", rm.V(), rm.F(), ndrop);
    } else plog(os, "postprocess: remesh empty, using decoded mesh");

    const std::vector<float>& src_v = rm.F() > 0 ? rm.verts : mesh.verts;
    const std::vector<int32_t>& src_f = rm.F() > 0 ? rm.faces : mesh.faces;
    std::vector<float> dv; std::vector<int32_t> df;
    if (opt.target_faces > 0 && (int)src_f.size()/3 > opt.target_faces) {
        plog(os, "postprocess: qem begin F=%d -> %d", (int)src_f.size()/3, opt.target_faces);
        decimate_qem(src_v, (int)src_v.size()/3, src_f, (int)src_f.size()/3, opt.target_faces, dv, df);
        weld_vertices(dv, df, nullptr, 1.0f / ((float)rres * 8.0f));
        fill_small_holes(df);
        drop_small_components(dv, df, 0.03f);
    } else { plog(os, "postprocess: qem skipped (F=%d <= target)", (int)src_f.size()/3); dv = src_v; df = src_f; }
    plog(os, "postprocess: qem done V=%zu F=%zu", dv.size()/3, df.size()/3);

    VoxelPbr vox{&coords, &pbr6, res, &bvh};
    const std::vector<float> none;
    plog(os, "postprocess: uv begin atlas=%d xatlas=%d", opt.texture_size, opt.use_xatlas?1:0);
    BakedMesh bm = opt.use_xatlas ? uv_bake(dv, (int)dv.size()/3, df, (int)df.size()/3, none, opt.texture_size, &vox)
                                  : uv_box_project(dv, (int)dv.size()/3, df, (int)df.size()/3, none, opt.texture_size, &vox);
    if (!bm.ok() && opt.use_xatlas) bm = uv_chart_project(dv, (int)dv.size()/3, df, (int)df.size()/3, none, opt.texture_size, &vox);
    if (!bm.ok()) { plog(os, "postprocess: UV/PBR bake failed"); if(report)*report=os.str(); return false; }
    plog(os, "postprocess: uv done V=%zu F=%zu", bm.verts.size()/3, bm.faces.size()/3);
    apply_reference_frame(bm.verts);
    const bool ok = write_glb_textured(out_glb.c_str(), bm.verts.data(), (int64_t)bm.verts.size()/3,
                                       bm.uv.data(), bm.faces.data(), (int64_t)bm.faces.size()/3,
                                       bm.base.data(), bm.mr.data(), bm.T,
                                       /*double_sided=*/rm.F()==0, -1, nullptr, opt.use_webp);
    plog(os, "postprocess: atlas=%d textured_glb=%s", bm.T, ok?"OK":"FAIL");
    if(report)*report=os.str(); return ok;
}
}
