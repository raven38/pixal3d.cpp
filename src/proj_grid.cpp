// Camera-aware pixel-aligned projection grid (Pixal3D ProjGrid / ProjGridMV port).
// See include/proj_grid.h and
// pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py for the
// PyTorch reference this ports (ProjGrid.forward, ProjGridMV.forward,
// project_points_to_image_batch, sample_features, compute_relative_calc_mat).
#include "proj_grid.h"

#include <algorithm>
#include <cmath>

namespace trellis {

namespace {

// Row-major 4x4 double matrices throughout (index i*4+j = row i, col j).

void front_view_matrix_d(double distance, double out[16]) {
    out[0] = 1;  out[1] = 0;  out[2] = 0;  out[3] = 0;
    out[4] = 0;  out[5] = 0;  out[6] = -1; out[7] = -distance;
    out[8] = 0;  out[9] = 1;  out[10] = 0; out[11] = 0;
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
}

// Gauss-Jordan 4x4 inverse in double. Returns false (and zeroes out) if
// singular, which should not happen for well-formed camera matrices.
bool mat4_inverse_d(const double m[16], double out[16]) {
    double a[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) a[i][j] = m[i * 4 + j];
        for (int j = 0; j < 4; ++j) a[i][4 + j] = (i == j) ? 1.0 : 0.0;
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        double best = std::fabs(a[col][col]);
        for (int r = col + 1; r < 4; ++r) {
            double v = std::fabs(a[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) for (int k = 0; k < 8; ++k) std::swap(a[col][k], a[piv][k]);
        double d = a[col][col];
        if (d == 0.0) { std::fill(out, out + 16, 0.0); return false; }
        for (int k = 0; k < 8; ++k) a[col][k] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            double f = a[r][col];
            if (f == 0.0) continue;
            for (int k = 0; k < 8; ++k) a[r][k] -= f * a[col][k];
        }
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out[i * 4 + j] = a[i][4 + j];
    return true;
}

void mat4_mul_d(const double A[16], const double B[16], double out[16]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += A[i * 4 + k] * B[k * 4 + j];
            out[i * 4 + j] = s;
        }
    }
}

// torch grid_sample(mode='bilinear', align_corners=False, padding_mode='border')
// for a single [H,W] plane at normalized coord (nx, ny) in [-1,1].
float grid_sample_bilinear(const float* plane, int H, int W, float nx, float ny) {
    double u = (((double)nx + 1.0) * (double)W - 1.0) / 2.0;
    double v = (((double)ny + 1.0) * (double)H - 1.0) / 2.0;
    long x0 = (long)std::floor(u), y0 = (long)std::floor(v);
    long x1 = x0 + 1, y1 = y0 + 1;
    double wx = u - (double)x0, wy = v - (double)y0;
    auto clampi = [](long x, long lo, long hi) { return x < lo ? lo : (x > hi ? hi : x); };
    long cx0 = clampi(x0, 0, W - 1), cx1 = clampi(x1, 0, W - 1);
    long cy0 = clampi(y0, 0, H - 1), cy1 = clampi(y1, 0, H - 1);
    double f00 = plane[cy0 * (size_t)W + cx0];
    double f01 = plane[cy0 * (size_t)W + cx1];
    double f10 = plane[cy1 * (size_t)W + cx0];
    double f11 = plane[cy1 * (size_t)W + cx1];
    double top = f00 * (1.0 - wx) + f01 * wx;
    double bot = f10 * (1.0 - wx) + f11 * wx;
    return (float)(top * (1.0 - wy) + bot * wy);
}

} // namespace

void proj_grid_points(int R, std::vector<float>& out) {
    out.assign((size_t)R * R * R * 3, 0.f);
    std::vector<float> lin(R);
    if (R == 1) {
        lin[0] = -1.f; // torch.linspace(-1,1,1) == [-1]
    } else {
        for (int i = 0; i < R; ++i) lin[i] = -1.f + 2.f * (float)i / (float)(R - 1);
    }
    // meshgrid(indexing='ij'): x=lin[ix], y=lin[iy], z=lin[iz]; token k = ix*R*R+iy*R+iz.
    // Rotation Rot=[[1,0,0],[0,0,-1],[0,1,0]] applied as p @ Rot^T == Rot @ p:
    // (new_x, new_y, new_z) = (x, -z, y).
    for (int ix = 0; ix < R; ++ix) {
        for (int iy = 0; iy < R; ++iy) {
            for (int iz = 0; iz < R; ++iz) {
                float x0 = lin[ix], y0 = lin[iy], z0 = lin[iz];
                size_t k = ((size_t)ix * R + iy) * (size_t)R + iz;
                out[k * 3 + 0] = x0;
                out[k * 3 + 1] = -z0;
                out[k * 3 + 2] = y0;
            }
        }
    }
}

void proj_grid_project(int R, int S, const Camera& cam,
                        std::vector<float>& pix, std::vector<float>& norm,
                        std::vector<uint8_t>& valid) {
    std::vector<float> grid;
    proj_grid_points(R, grid);
    size_t N = grid.size() / 3;
    pix.assign(N * 2, 0.f);
    norm.assign(N * 2, 0.f);
    valid.assign(N, 0);

    double c2w[16];
    if (cam.has_c2w) {
        for (int i = 0; i < 16; ++i) c2w[i] = (double)cam.c2w[i];
    } else {
        front_view_matrix_d((double)cam.distance, c2w);
    }
    double w2c[16];
    mat4_inverse_d(c2w, w2c);

    const double focal_mm = 16.0 / std::tan((double)cam.fov_x / 2.0);
    const double focal_px = focal_mm * (double)S / 32.0;
    const double ms = (double)cam.mesh_scale;
    const double Sf = (double)S;

    for (size_t k = 0; k < N; ++k) {
        double x = (double)grid[k * 3 + 0] / ms / 2.0;
        double y = (double)grid[k * 3 + 1] / ms / 2.0;
        double z = (double)grid[k * 3 + 2] / ms / 2.0;
        // points_camera = w2c @ [x,y,z,1] (drop homogeneous row).
        double xc = w2c[0] * x + w2c[1] * y + w2c[2] * z + w2c[3];
        double yc = w2c[4] * x + w2c[5] * y + w2c[6] * z + w2c[7];
        double zc = w2c[8] * x + w2c[9] * y + w2c[10] * z + w2c[11];
        double depth = -zc;
        double x_ndc = focal_px * xc / (-zc + 1e-8);
        double y_ndc = focal_px * yc / (-zc + 1e-8);
        double x_pix = x_ndc + Sf / 2.0;
        double y_pix = -y_ndc + Sf / 2.0;
        bool v = (x_pix >= 0.0) && (x_pix < Sf) && (y_pix >= 0.0) && (y_pix < Sf) && (depth > 0.0);
        pix[k * 2 + 0] = (float)x_pix;
        pix[k * 2 + 1] = (float)y_pix;
        norm[k * 2 + 0] = (float)((x_pix + 0.5) / Sf * 2.0 - 1.0);
        norm[k * 2 + 1] = (float)((y_pix + 0.5) / Sf * 2.0 - 1.0);
        valid[k] = v ? 1 : 0;
    }
}

std::vector<float> proj_grid_sample(const float* fmap, int C, int H, int W,
                                     int R, int S, const Camera& cam) {
    std::vector<float> pix, norm;
    std::vector<uint8_t> valid;
    proj_grid_project(R, S, cam, pix, norm, valid);
    const size_t N = norm.size() / 2;
    std::vector<float> out(N * (size_t)C, 0.f);
    for (size_t k = 0; k < N; ++k) {
        const float nx = norm[k * 2 + 0], ny = norm[k * 2 + 1];
        for (int c = 0; c < C; ++c) {
            const float* plane = fmap + (size_t)c * H * W;
            out[k * (size_t)C + c] = grid_sample_bilinear(plane, H, W, nx, ny);
        }
    }
    return out;
}

void proj_grid_bilinear_taps(int H, int W, int R, int S, const Camera& cam,
                              std::vector<int32_t> idx[4], std::vector<float> w[4]) {
    std::vector<float> pix, norm;
    std::vector<uint8_t> valid;
    proj_grid_project(R, S, cam, pix, norm, valid);
    const size_t N = norm.size() / 2;
    for (int t = 0; t < 4; ++t) { idx[t].resize(N); w[t].resize(N); }
    auto clampi = [](long x, long lo, long hi) { return x < lo ? lo : (x > hi ? hi : x); };
    for (size_t k = 0; k < N; ++k) {
        // Same arithmetic as grid_sample_bilinear (torch grid_sample, align_corners=False,
        // padding_mode='border'), with the four corner fetches split out.
        const float nx = norm[k * 2 + 0], ny = norm[k * 2 + 1];
        double u = (((double)nx + 1.0) * (double)W - 1.0) / 2.0;
        double v = (((double)ny + 1.0) * (double)H - 1.0) / 2.0;
        long x0 = (long)std::floor(u), y0 = (long)std::floor(v);
        long x1 = x0 + 1, y1 = y0 + 1;
        double wx = u - (double)x0, wy = v - (double)y0;
        long cx0 = clampi(x0, 0, W - 1), cx1 = clampi(x1, 0, W - 1);
        long cy0 = clampi(y0, 0, H - 1), cy1 = clampi(y1, 0, H - 1);
        idx[0][k] = (int32_t)(cy0 * W + cx0); w[0][k] = (float)((1.0 - wx) * (1.0 - wy));
        idx[1][k] = (int32_t)(cy0 * W + cx1); w[1][k] = (float)(wx * (1.0 - wy));
        idx[2][k] = (int32_t)(cy1 * W + cx0); w[2][k] = (float)((1.0 - wx) * wy);
        idx[3][k] = (int32_t)(cy1 * W + cx1); w[3][k] = (float)(wx * wy);
    }
}

void mv_calc_mats(const float* c2w, int V, float distance0, std::vector<float>& calc) {
    calc.assign((size_t)V * 16, 0.f);
    double Fp[16];
    front_view_matrix_d((double)distance0, Fp);
    double C0[16];
    for (int i = 0; i < 16; ++i) C0[i] = (double)c2w[i]; // view 0 = main view
    double C0inv[16];
    mat4_inverse_d(C0, C0inv);
    for (int vi = 0; vi < V; ++vi) {
        double Ci[16];
        for (int i = 0; i < 16; ++i) Ci[i] = (double)c2w[(size_t)vi * 16 + i];
        double rel[16], out[16];
        mat4_mul_d(C0inv, Ci, rel); // inv(C0) @ C_i
        mat4_mul_d(Fp, rel, out);   // F' @ rel
        for (int i = 0; i < 16; ++i) calc[(size_t)vi * 16 + i] = (float)out[i];
    }
}

} // namespace trellis
