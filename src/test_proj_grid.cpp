// Validate include/proj_grid.h (ProjGrid / ProjGridMV camera-aware pixel-
// aligned projection + multiview calc_mat/average fusion) against either a
// hand-computed self-test or PyTorch golden tensors dumped by
// tools/ref_pixal3d_proj_grid.py.
//
//   trellis-test-proj-grid --selftest
//   trellis-test-proj-grid <fixture_dir>
//
// Reference: pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
#include "proj_grid.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using std::string; using std::vector;
using namespace trellis;

// ---------------------------------------------------------------------------
// shared compare helper (tolerance tuned for f32 CPU vs torch f32 CPU parity)
// ---------------------------------------------------------------------------
static bool compare(const char* name, const vector<float>& mine, const npy::Array& ref, double tol = 1e-4) {
    if ((int64_t)mine.size() != ref.numel()) {
        printf("  %-24s SIZE MISMATCH mine=%zu ref=%lld\n", name, mine.size(), (long long)ref.numel());
        return false;
    }
    double maxabs = 0, sumabs = 0, refmax = 0;
    for (size_t i = 0; i < mine.size(); ++i) {
        double d = std::fabs((double)mine[i] - ref.data[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)ref.data[i]));
    }
    double rel = refmax > 0 ? maxabs / refmax : maxabs;
    bool ok = rel < tol;
    printf("  %-24s max|d|=%.4e mean|d|=%.4e rel=%.4e  %s\n",
           name, maxabs, sumabs / mine.size(), rel, ok ? "PASS" : "FAIL");
    return ok;
}

// proj_grid_bilinear_taps must reproduce proj_grid_sample (host, f64 accumulation) up to the
// f32 rounding of the four products it hands the device: max|d| ~1e-6 on O(1) features.
static bool check_taps(const char* name, const float* fmap, int C, int H, int W, int R, int S, const Camera& cam,
                       const vector<float>& z_host) {
    vector<int32_t> idx[4]; vector<float> w[4];
    proj_grid_bilinear_taps(H, W, R, S, cam, idx, w);
    const size_t N = idx[0].size();
    vector<float> z(N * (size_t)C);
    for (size_t k = 0; k < N; ++k)
        for (int c = 0; c < C; ++c) {
            const float* plane = fmap + (size_t)c * H * W;
            float acc = 0.f;
            for (int t = 0; t < 4; ++t) acc += plane[idx[t][k]] * w[t][k];
            z[k * (size_t)C + c] = acc;
        }
    bool ok = true;
    for (size_t k = 0; k < N; ++k) for (int t = 0; t < 4; ++t) if (idx[t][k] < 0 || idx[t][k] >= H * W) ok = false;
    printf("  %-24s idx in [0,%d): %s\n", name, H * W, ok ? "PASS" : "FAIL");
    npy::Array ref; ref.shape = {(int64_t)z_host.size()}; ref.data = z_host;
    return compare(name, z, ref, 1e-5) && ok;
}

static bool try_load(const string& path, npy::Array& out) {
    try { out = npy::load(path); return true; } catch (...) { return false; }
}

// try a list of candidate basenames (handles fixture-naming ambiguity), return
// the first that loads.
static bool try_load_any(const string& dir, std::initializer_list<const char*> names, npy::Array& out, string* matched = nullptr) {
    for (const char* n : names) {
        if (try_load(dir + "/" + n + ".npy", out)) { if (matched) *matched = n; return true; }
    }
    return false;
}

// BHWC [1,H,W,C] -> channel-major [C,H,W]
static vector<float> bhwc_to_chw(const npy::Array& a, int& H, int& W, int& C) {
    H = (int)a.shape[1]; W = (int)a.shape[2]; C = (int)a.shape[3];
    vector<float> out((size_t)C * H * W);
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            for (int c = 0; c < C; ++c)
                out[(size_t)c * H * W + h * W + w] = a.data[((size_t)h * W + w) * C + c];
    return out;
}

// -----------------------------------------------------------------------
// --selftest: hand-computable case (no PyTorch needed)
// -----------------------------------------------------------------------
static int run_selftest() {
    printf("=== proj_grid --selftest ===\n");
    bool all_ok = true;
    const int R = 2, S = 512;
    Camera cam{}; cam.fov_x = 0.6911f; cam.distance = 2.f; cam.mesh_scale = 1.f; cam.has_c2w = false;

    vector<float> pix, norm; vector<uint8_t> valid;
    proj_grid_project(R, S, cam, pix, norm, valid);

    // token k = ix*4 + iy*2 + iz for R=2
    auto tok = [](int ix, int iy, int iz) { return (size_t)(ix * 4 + iy * 2 + iz); };
    for (int iy = 0; iy < 2; ++iy) {
        for (int iz = 0; iz < 2; ++iz) {
            size_t k0 = tok(0, iy, iz), k1 = tok(1, iy, iz);
            double sx = pix[k0 * 2 + 0] + pix[k1 * 2 + 0];
            double sy = pix[k0 * 2 + 1] - pix[k1 * 2 + 1]; // y independent of ix -> should match
            bool ok = std::fabs(sx - 2.0 * S / 2.0) < 1e-3 && std::fabs(sy) < 1e-3;
            printf("  mirror-x iy=%d iz=%d: x_pix[0]+x_pix[1]=%.4f (want %.1f), y diff=%.4e  %s\n",
                   iy, iz, sx, (double)S, sy, ok ? "PASS" : "FAIL");
            all_ok &= ok;
        }
    }
    for (int ix = 0; ix < 2; ++ix) {
        for (int iz = 0; iz < 2; ++iz) {
            size_t k0 = tok(ix, 0, iz), k1 = tok(ix, 1, iz);
            double sy = pix[k0 * 2 + 1] + pix[k1 * 2 + 1];
            double sx = pix[k0 * 2 + 0] - pix[k1 * 2 + 0];
            bool ok = std::fabs(sy - 2.0 * S / 2.0) < 1e-3 && std::fabs(sx) < 1e-3;
            printf("  mirror-y ix=%d iz=%d: y_pix[0]+y_pix[1]=%.4f (want %.1f), x diff=%.4e  %s\n",
                   ix, iz, sy, (double)S, sx, ok ? "PASS" : "FAIL");
            all_ok &= ok;
        }
    }
    for (size_t k = 0; k < 8; ++k)
        printf("  token %zu: pix=(%.3f,%.3f) norm=(%.4f,%.4f) valid=%d\n",
               k, pix[k*2+0], pix[k*2+1], norm[k*2+0], norm[k*2+1], (int)valid[k]);

    // constant feature map -> constant sample everywhere (border padding, bilinear of constant plane)
    const int H = 32, W = 32, C = 1;
    vector<float> fmap((size_t)H * W, 7.0f);
    vector<float> out = proj_grid_sample(fmap.data(), C, H, W, R, S, cam);
    bool const_ok = true;
    for (float v : out) if (std::fabs(v - 7.0f) > 1e-5f) const_ok = false;
    printf("  constant-fmap sample: all==7.0 %s\n", const_ok ? "PASS" : "FAIL");
    all_ok &= const_ok;
    {   // taps decomposition on a non-constant map
        vector<float> ramp((size_t)H * W);
        for (int h = 0; h < H; ++h) for (int w = 0; w < W; ++w) ramp[(size_t)h * W + w] = 0.1f * h - 0.07f * w + 0.5f;
        vector<float> zh = proj_grid_sample(ramp.data(), 1, H, W, R, S, cam);
        all_ok &= check_taps("taps(ramp) vs sample", ramp.data(), 1, H, W, R, S, cam, zh);
    }

    printf("=== selftest %s ===\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// infer image resolution S from a reference (pix, norm) pair:
//   norm = (pix + 0.5) / S * 2 - 1  =>  S = (pix + 0.5) * 2 / (norm + 1)
// -----------------------------------------------------------------------
static int infer_S(const npy::Array& points_2d, const npy::Array& norm) {
    // points_2d: [1,N,2], norm: [1,N,2]
    double px = points_2d.data[0], nx = norm.data[0];
    double s = (px + 0.5) * 2.0 / (nx + 1.0);
    return (int)std::lround(s);
}

static int cube_root(int64_t n) {
    for (int r = 1; r <= 4096; ++r) if ((int64_t)r * r * r == n) return r;
    return -1;
}

// -----------------------------------------------------------------------
// fixture comparison
// -----------------------------------------------------------------------
static int run_fixture(const string& dir) {
    printf("=== proj_grid fixture: %s ===\n", dir.c_str());
    npy::Array fmap_bhwc, camera_angle_x, distance, mesh_scale, grid_points_ref, points_2d, image_points_norm, z_proj;
    bool have_base =
        try_load(dir + "/fmap_bhwc.npy", fmap_bhwc) &&
        try_load(dir + "/camera_angle_x.npy", camera_angle_x) &&
        try_load(dir + "/distance.npy", distance) &&
        try_load(dir + "/mesh_scale.npy", mesh_scale) &&
        try_load(dir + "/grid_points.npy", grid_points_ref) &&
        try_load(dir + "/points_2d.npy", points_2d) &&
        try_load(dir + "/image_points_norm.npy", image_points_norm) &&
        try_load(dir + "/z_proj.npy", z_proj);

    if (!have_base) {
        printf("base fixture files not found under %s -- parity UNVERIFIED\n", dir.c_str());
        return -1;
    }

    bool all_ok = true;
    const int R = cube_root(grid_points_ref.shape[0]);
    const int S = infer_S(points_2d, image_points_norm);
    printf("inferred R=%d S=%d\n", R, S);

    Camera cam{}; cam.fov_x = camera_angle_x.data[0]; cam.distance = distance.data[0];
    cam.mesh_scale = mesh_scale.data[0]; cam.has_c2w = false;

    // grid_points
    vector<float> gp; proj_grid_points(R, gp);
    all_ok &= compare("grid_points", gp, grid_points_ref);

    // points_2d / image_points_norm
    vector<float> pix, norm; vector<uint8_t> valid;
    proj_grid_project(R, S, cam, pix, norm, valid);
    all_ok &= compare("points_2d", pix, points_2d);
    all_ok &= compare("image_points_norm", norm, image_points_norm);

    // z_proj
    int H, W, C; vector<float> chw = bhwc_to_chw(fmap_bhwc, H, W, C);
    vector<float> z = proj_grid_sample(chw.data(), C, H, W, R, S, cam);
    all_ok &= compare("z_proj", z, z_proj);
    all_ok &= check_taps("taps vs z_proj(host)", chw.data(), C, H, W, R, S, cam, z);

    // --- R=32 variant (same camera; its own fmap if the fixture provides one) ---
    npy::Array r32_z;
    if (try_load(dir + "/r32_z_proj.npy", r32_z)) {
        int R2 = cube_root(r32_z.shape[1]);
        if (R2 < 0) R2 = 32;
        npy::Array r32_fmap;
        vector<float> chw2 = chw; int H2 = H, W2 = W, C2 = C;
        if (try_load(dir + "/r32_fmap_bhwc.npy", r32_fmap)) chw2 = bhwc_to_chw(r32_fmap, H2, W2, C2);
        vector<float> z2 = proj_grid_sample(chw2.data(), C2, H2, W2, R2, S, cam);
        all_ok &= compare("r32_z_proj", z2, r32_z);
        npy::Array r32_gp;
        if (try_load(dir + "/r32_grid_points.npy", r32_gp)) {
            vector<float> gp2; proj_grid_points(R2, gp2);
            all_ok &= compare("r32_grid_points", gp2, r32_gp);
        }
        npy::Array r32_pix;
        if (try_load(dir + "/r32_points_2d.npy", r32_pix)) {
            vector<float> pix2, norm2; vector<uint8_t> valid2;
            proj_grid_project(R2, S, cam, pix2, norm2, valid2);
            all_ok &= compare("r32_points_2d", pix2, r32_pix);
            npy::Array r32_norm;
            if (try_load(dir + "/r32_image_points_norm.npy", r32_norm))
                all_ok &= compare("r32_image_points_norm", norm2, r32_norm);
        }
    } else {
        printf("  r32_z_proj.npy not found -- R=32 case SKIPPED\n");
    }

    // --- high-res feature map variant (same R=16 grid/camera, bigger fmap) ---
    npy::Array hr_fmap, hr_z;
    if (try_load(dir + "/hr_fmap_bchw.npy", hr_fmap) && try_load(dir + "/hr_z_proj.npy", hr_z)) {
        int hC = (int)hr_fmap.shape[1], hH = (int)hr_fmap.shape[2], hW = (int)hr_fmap.shape[3];
        // already [1,C,H,W] -> squeeze batch, contiguous C,H,W layout matches directly
        const float* hr_data = hr_fmap.data.data();
        vector<float> hz = proj_grid_sample(hr_data, hC, hH, hW, R, S, cam);
        all_ok &= compare("hr_z_proj", hz, hr_z);
    } else {
        printf("  hr_fmap_bchw.npy/hr_z_proj.npy not found -- HR case SKIPPED\n");
    }

    // --- multiview: calc_mat + per-view projection + average fusion ---
    npy::Array transform_matrix;
    if (try_load(dir + "/mv_transform_matrix.npy", transform_matrix) ||
        try_load(dir + "/transform_matrix.npy", transform_matrix)) {
        const int V = (int)transform_matrix.shape[1];
        printf("multiview: V=%d\n", V);

        npy::Array mv_cax, mv_dist, mv_fmap, calc_mat_ref;
        string matched;
        bool have_cax = try_load_any(dir, {"mv_camera_angle_x", "camera_angle_x_mv"}, mv_cax, &matched);
        bool have_dist = try_load_any(dir, {"mv_distance", "distance_mv"}, mv_dist, &matched);
        bool have_fmap = try_load(dir + "/mv_fmap_bhwc.npy", mv_fmap);
        bool have_calc = try_load(dir + "/mv_calc_mat.npy", calc_mat_ref) ||
                         try_load(dir + "/calc_mat.npy", calc_mat_ref);

        float distance0 = have_dist ? mv_dist.data[0] : distance.data[0];
        vector<float> calc;
        // transform_matrix is [1,V,4,4] row-major; drop leading batch dim.
        mv_calc_mats(transform_matrix.data.data(), V, distance0, calc);
        if (have_calc) all_ok &= compare("calc_mat", calc, calc_mat_ref);
        else printf("  calc_mat.npy not found -- calc_mat comparison SKIPPED\n");

        if (have_fmap) {
            const int mH = (int)mv_fmap.shape[1], mW = (int)mv_fmap.shape[2], mC = (int)mv_fmap.shape[3];
            vector<vector<float>> zviews(V);
            bool have_any_view_ref = false, view_ok = true;
            vector<float> sum((size_t)R * R * R * (size_t)mC, 0.f);
            for (int vi = 0; vi < V; ++vi) {
                // slice view vi out of mv_fmap [V,H,W,C] and convert HWC->CHW
                vector<float> chwv((size_t)mC * mH * mW);
                const float* base = mv_fmap.data.data() + (size_t)vi * mH * mW * mC;
                for (int h = 0; h < mH; ++h)
                    for (int w = 0; w < mW; ++w)
                        for (int c = 0; c < mC; ++c)
                            chwv[(size_t)c * mH * mW + h * mW + w] = base[((size_t)h * mW + w) * mC + c];

                Camera cmv{}; cmv.has_c2w = true; cmv.mesh_scale = cam.mesh_scale;
                for (int i = 0; i < 16; ++i) cmv.c2w[i] = calc[(size_t)vi * 16 + i];
                cmv.fov_x = have_cax ? mv_cax.data[vi] : cam.fov_x;

                zviews[vi] = proj_grid_sample(chwv.data(), mC, mH, mW, R, S, cmv);
                for (size_t i = 0; i < zviews[vi].size(); ++i) sum[i] += zviews[vi][i];
                {
                    string tn = "mv taps v" + std::to_string(vi);
                    all_ok &= check_taps(tn.c_str(), chwv.data(), mC, mH, mW, R, S, cmv, zviews[vi]);
                }

                npy::Array vref;
                string vname = "mv_z_proj_v" + std::to_string(vi);
                if (try_load(dir + "/" + vname + ".npy", vref)) {
                    have_any_view_ref = true;
                    view_ok &= compare(vname.c_str(), zviews[vi], vref);
                }
            }
            if (have_any_view_ref) all_ok &= view_ok;
            else printf("  mv_z_proj_v*.npy not found -- per-view comparison SKIPPED\n");

            for (float& v : sum) v /= (float)V;
            npy::Array avg_ref;
            if (try_load(dir + "/mv_z_proj_avg.npy", avg_ref))
                all_ok &= compare("mv_z_proj_avg", sum, avg_ref);
            else
                printf("  mv_z_proj_avg.npy not found -- average-fusion comparison SKIPPED\n");
        } else {
            printf("  mv_fmap_bhwc.npy not found -- per-view/avg comparison SKIPPED\n");
        }
    } else {
        printf("  transform_matrix.npy not found -- multiview case SKIPPED\n");
    }

    printf("=== fixture comparison %s ===\n", all_ok ? "PASS" : "FAIL (see above)");
    return all_ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s --selftest | <fixture_dir>\n", argv[0]);
        return 1;
    }
    string arg1 = argv[1];
    if (arg1 == "--selftest") return run_selftest();
    int rc = run_fixture(arg1);
    if (rc < 0) return 0; // fixture absent: not a build/test failure, just unverified
    return rc;
}
