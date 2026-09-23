#include "pixal3d_postprocess.h"
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "mesh_glb.h"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <new>
#include <sstream>
#ifdef __EMSCRIPTEN__
#include <emscripten/heap.h>
#include <malloc.h>
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
    struct mallinfo mi = mallinfo();
    char h[96]; snprintf(h, sizeof h, " [grown %.0f live %.0f free %.0f MB]",
                         emscripten_get_heap_size() / 1048576.0,
                         (double)(unsigned)mi.uordblks / 1048576.0, (double)(unsigned)mi.fordblks / 1048576.0);
    line += h;
#endif
    line += "\n";
    os << line; fputs(line.c_str(), stdout); fflush(stdout);
}

// Live host heap (wasm32: dlmalloc in-use bytes); 0 where it is not tracked (native).
static uint64_t host_live_bytes() {
#ifdef __EMSCRIPTEN__
    return (uint64_t)(unsigned)mallinfo().uordblks;
#else
    return 0;
#endif
}
// #83 tail host-memory model, calibrated with the wasm32 tail replay (pixal3d-tail-replay-node,
// 2026-09-24) on the cyclops res-1024 decode (F = 9.44M): measured grown-heap peaks from a clean
// heap were 1982 / 2089 / 2641 / 3185 / 3809 MiB at remesh_res 512 / 640 / 768 / 896 / 1024; see
// docs/spec/33-pixal3d-ss-res.md section 6. All bytes are on top of the heap live at the tail's start.
constexpr double kWeldBvhPeakPerFace   = 180.0; // weld + hole fill + TriBvh build transient peak, per decoded face (measured 176)
constexpr double kPersistPerFace       = 80.0;  // welded mesh + BVH kept for remesh/bake, per decoded face (measured 76)
constexpr double kRemeshFacesPerFace   = 1.95;  // remesh faces per decoded face at remesh_res == res, x (rres/res)^2 (measured 1.90-1.95)
constexpr double kRemeshQemPerOutFace  = 160.0; // remesh + clean + QEM peak, per remesh face (measured 149-157)
static uint64_t predict_bvh_peak(int F) { return (uint64_t)(kWeldBvhPeakPerFace * F); }
static uint64_t predict_remesh_peak(int F, int rres, int res) {
    const double s = (double)rres / (double)res;
    return (uint64_t)(kPersistPerFace * F + kRemeshQemPerOutFace * kRemeshFacesPerFace * F * s * s);
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
    const int rres0 = opt.remesh_res > 0 ? opt.remesh_res : res;
    plog(os, "postprocess: input V=%d F=%d voxels=%zu remesh_res=%d target_faces=%d atlas=%d",
         mesh.V(), mesh.F(), coords.size(), rres0, opt.target_faces, opt.texture_size);
    const char* stage = "weld";
    try {
    // #83 bounded policy: decide the remesh grid from a prediction made before anything big is
    // built. Rungs are deterministic (rres0 x 1, 3/4, 1/2, 3/8, rounded to 16, >= 128); the first
    // rung whose predicted peak (heap live now + max(weld/BVH, remesh+QEM)) fits the budget is
    // taken, and a std::bad_alloc inside remesh/QEM still steps down one rung.
    std::vector<int> rungs{ rres0 };
    size_t rung = 0;
    if (opt.host_budget_bytes > 0) {
        for (int r : { rres0 * 3 / 4 / 16 * 16, rres0 / 2 / 16 * 16, rres0 * 3 / 8 / 16 * 16 })
            if (r >= 128 && r < rungs.back()) rungs.push_back(r);
        const uint64_t live = host_live_bytes(), bvh = predict_bvh_peak(mesh.F());
        auto peak = [&](size_t k) { return live + std::max(bvh, predict_remesh_peak(mesh.F(), rungs[k], res)); };
        if (live + bvh > opt.host_budget_bytes) {
            plog(os, "postprocess: HOST_HEAP_EXHAUSTED (preflight): live %.0f MiB + weld/BVH %.0f MiB for F=%d exceeds the %.0f MiB budget",
                 live / 1048576.0, bvh / 1048576.0, mesh.F(), opt.host_budget_bytes / 1048576.0);
            if (report) *report = os.str();
            return false;
        }
        while (rung + 1 < rungs.size() && peak(rung) > opt.host_budget_bytes) ++rung;
        const bool fits = peak(rung) <= opt.host_budget_bytes;
        for (size_t k = 0; k < rungs.size(); ++k)
            plog(os, "postprocess: host budget %.0f MiB, live %.0f MiB: remesh@%d predicted peak %.0f MiB%s",
                 opt.host_budget_bytes / 1048576.0, live / 1048576.0, rungs[k], peak(k) / 1048576.0,
                 k == rung && fits ? "  <- selected" : (peak(k) > opt.host_budget_bytes ? "  (over budget)" : ""));
        if (!fits) {
            plog(os, "postprocess: HOST_HEAP_EXHAUSTED (preflight): no remesh rung fits the %.0f MiB budget (smallest remesh@%d needs %.0f MiB)",
                 opt.host_budget_bytes / 1048576.0, rungs.back(), peak(rungs.size() - 1) / 1048576.0);
            if (report) *report = os.str();
            return false;
        }
    }
    plog(os, "postprocess: weld begin");
    weld_vertices(mesh.verts, mesh.faces, nullptr, 1.0f / ((float)res * 8.0f));
    fill_small_holes(mesh.faces);
    plog(os, "postprocess: weld done V=%d F=%d", mesh.V(), mesh.F());
    stage = "tri_bvh";
    plog(os, "postprocess: tri_bvh begin F=%d", mesh.F());
    TriBvh bvh = TriBvh::build(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F());
    plog(os, "postprocess: tri_bvh done");
    Mesh rm;
    std::vector<float> dv; std::vector<int32_t> df;
    int rres = rungs[rung];
    for (;;) {
        rres = rungs[rung];
        try {
            stage = "remesh";
            plog(os, "postprocess: remesh begin res=%d band=%d", rres, opt.remesh_band);
            rm = remesh_narrow_band_dc(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), bvh, rres, opt.remesh_band);
            if (rm.F() > 0) {
                clean_mesh(rm.V(), rm.faces);
                int ndrop = drop_small_components(rm.verts, rm.faces, 0.02f);
                plog(os, "postprocess: remesh done V=%d F=%d dropped=%d", rm.V(), rm.F(), ndrop);
            } else plog(os, "postprocess: remesh empty, using decoded mesh");
            const std::vector<float>& src_v = rm.F() > 0 ? rm.verts : mesh.verts;
            const std::vector<int32_t>& src_f = rm.F() > 0 ? rm.faces : mesh.faces;
            stage = "qem";
            if (opt.target_faces > 0 && (int)src_f.size()/3 > opt.target_faces) {
                plog(os, "postprocess: qem begin F=%d -> %d", (int)src_f.size()/3, opt.target_faces);
                decimate_qem(src_v, (int)src_v.size()/3, src_f, (int)src_f.size()/3, opt.target_faces, dv, df);
                weld_vertices(dv, df, nullptr, 1.0f / ((float)rres * 8.0f));
                fill_small_holes(df);
                drop_small_components(dv, df, 0.03f);
            } else { plog(os, "postprocess: qem skipped (F=%d <= target)", (int)src_f.size()/3); dv = src_v; df = src_f; }
            plog(os, "postprocess: qem done V=%zu F=%zu", dv.size()/3, df.size()/3);
            break;
        } catch (const std::bad_alloc&) {
            rm = Mesh{}; std::vector<float>().swap(dv); std::vector<int32_t>().swap(df);
            if (opt.host_budget_bytes == 0 || rung + 1 >= rungs.size()) throw;
            plog(os, "postprocess: %s@%d exhausted the host heap (std::bad_alloc); stepping down to remesh_res %d",
                 stage, rres, rungs[rung + 1]);
            ++rung;
        }
    }
    if (rres != rres0) plog(os, "postprocess: BOUNDED_FALLBACK remesh_res %d -> %d", rres0, rres);

    stage = "uv/bake";
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
    } catch (const std::bad_alloc&) {
        plog(os, "postprocess: HOST_HEAP_EXHAUSTED: std::bad_alloc during %s", stage);
        if (report) *report = os.str();
        return false;
    }
}
}
