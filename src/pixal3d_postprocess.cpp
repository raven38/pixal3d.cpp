#include "pixal3d_postprocess.h"
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "mesh_glb.h"
#include <algorithm>
#include <sstream>

namespace trellis {
namespace {
static void apply_reference_frame(std::vector<float>& verts) {
    for (size_t i=0;i+2<verts.size();i+=3) { float x=verts[i], y=verts[i+1], z=verts[i+2]; verts[i]=-x; verts[i+1]=z; verts[i+2]=y; }
}
}
bool pixal3d_write_production_glb(const std::string& out_glb, Mesh mesh,
                                  const std::vector<std::array<int,3>>& coords,
                                  const std::vector<float>& pbr6, int res,
                                  const Pixal3dPostprocessOptions& opt,
                                  std::string* report) {
    std::ostringstream os;
    if (mesh.F() <= 0 || coords.empty() || pbr6.size() != coords.size()*6) {
        os << "postprocess: invalid input mesh/PBR\n"; if(report)*report=os.str(); return false;
    }
    weld_vertices(mesh.verts, mesh.faces, nullptr, 1.0f / ((float)res * 8.0f));
    fill_small_holes(mesh.faces);
    TriBvh bvh = TriBvh::build(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F());
    const int rres = opt.remesh_res > 0 ? opt.remesh_res : res;
    os << "postprocess: input V=" << mesh.V() << " F=" << mesh.F() << " voxels=" << coords.size()
       << " remesh_res=" << rres << " target_faces=" << opt.target_faces << " atlas=" << opt.texture_size << "\n";
    Mesh rm = remesh_narrow_band_dc(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), bvh, rres, opt.remesh_band);
    if (rm.F() > 0) {
        clean_mesh(rm.V(), rm.faces);
        int ndrop = drop_small_components(rm.verts, rm.faces, 0.02f);
        os << "postprocess: remesh V=" << rm.V() << " F=" << rm.F() << " dropped=" << ndrop << "\n";
    } else os << "postprocess: remesh empty, using decoded mesh\n";

    const std::vector<float>& src_v = rm.F() > 0 ? rm.verts : mesh.verts;
    const std::vector<int32_t>& src_f = rm.F() > 0 ? rm.faces : mesh.faces;
    std::vector<float> dv; std::vector<int32_t> df;
    if (opt.target_faces > 0 && (int)src_f.size()/3 > opt.target_faces) {
        decimate_qem(src_v, (int)src_v.size()/3, src_f, (int)src_f.size()/3, opt.target_faces, dv, df);
        weld_vertices(dv, df, nullptr, 1.0f / ((float)rres * 8.0f));
        fill_small_holes(df);
        drop_small_components(dv, df, 0.03f);
    } else { dv = src_v; df = src_f; }
    os << "postprocess: qem V=" << dv.size()/3 << " F=" << df.size()/3 << "\n";

    VoxelPbr vox{&coords, &pbr6, res, &bvh};
    const std::vector<float> none;
    BakedMesh bm = opt.use_xatlas ? uv_bake(dv, (int)dv.size()/3, df, (int)df.size()/3, none, opt.texture_size, &vox)
                                  : uv_box_project(dv, (int)dv.size()/3, df, (int)df.size()/3, none, opt.texture_size, &vox);
    if (!bm.ok() && opt.use_xatlas) bm = uv_chart_project(dv, (int)dv.size()/3, df, (int)df.size()/3, none, opt.texture_size, &vox);
    if (!bm.ok()) { os << "postprocess: UV/PBR bake failed\n"; if(report)*report=os.str(); return false; }
    apply_reference_frame(bm.verts);
    const bool ok = write_glb_textured(out_glb.c_str(), bm.verts.data(), (int64_t)bm.verts.size()/3,
                                       bm.uv.data(), bm.faces.data(), (int64_t)bm.faces.size()/3,
                                       bm.base.data(), bm.mr.data(), bm.T,
                                       /*double_sided=*/rm.F()==0, -1, nullptr, opt.use_webp);
    os << "postprocess: atlas=" << bm.T << " textured_glb=" << (ok?"OK":"FAIL") << "\n";
    if(report)*report=os.str(); return ok;
}
}
