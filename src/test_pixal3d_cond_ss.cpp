// Validate include/pixal3d_cond.h (Pixal3D SS-stage image conditioning: DINOv3 +
// ProjGridMV average fusion) against PyTorch golden tensors dumped by
// tools/ref_pixal3d_cond_ss.py.
//
//   trellis-test-pixal3d-cond-ss <dinov3.gguf> <fixture_dir> [gpu] [stage]
//
// gpu: backend device index (-1 = CPU; in a -DGGML_WEBGPU=ON -DGGML_METAL=OFF build, 0 = the
// ggml WebGPU device). stage (default `all`): `dino` (step 1 only), `cond` (steps 1-3, host
// projection), `gpu` (step 4 only: device-resident projection + MV fusion, V=1/2/4, memory and
// timing), `all`.
//
// Reference: pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
// (DinoV3ProjMultiViewFeatureExtractor, "ss" IMAGE_COND_CONFIGS entry).
#include "pixal3d_cond.h"
#include "dinov3.h"
#include "trellis_model.h"
#include "npy.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using std::string; using std::vector;
using namespace trellis;

static bool compare(const char* name, const vector<float>& mine, const vector<float>& ref, double tol = 2e-2) {
    if (mine.size() != ref.size()) {
        printf("  %-24s SIZE MISMATCH mine=%zu ref=%zu\n", name, mine.size(), ref.size());
        return false;
    }
    double maxabs = 0, sumabs = 0, refmax = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < mine.size(); ++i) {
        double d = std::fabs((double)mine[i] - ref[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)ref[i]));
        dot += (double)mine[i] * ref[i]; na += (double)mine[i] * mine[i]; nb += (double)ref[i] * ref[i];
    }
    double rel = refmax > 0 ? maxabs / refmax : maxabs;
    double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    bool ok = rel < tol;
    printf("  %-24s max|d|=%.4e mean|d|=%.4e rel=%.4e cos=%.7f  %s\n",
           name, maxabs, sumabs / mine.size(), rel, cos, ok ? "PASS" : "FAIL");
    return ok;
}
static bool compare(const char* name, const vector<float>& mine, const npy::Array& ref, double tol = 2e-2) {
    return compare(name, mine, ref.data, tol);
}

static void print_stats(const char* tag, const Pixal3dCondStats& st) {
    printf("  [%s] V=%d weights=%.1f MB cond=%.1f MB view_alloc=%.1f MB peak=%.1f MB  total=%.0f ms  slowest view=%.0f ms\n",
           tag, st.views, st.weight_bytes / 1048576.0, st.cond_bytes / 1048576.0, st.view_alloc_bytes / 1048576.0,
           st.peak_bytes / 1048576.0, st.total_ms, st.view_ms_max);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <dinov3.gguf> <fixture_dir> [gpu] [stage: dino|cond|gpu|all]\n", argv[0]);
        return 1;
    }
    const string gguf = argv[1], dir = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;
    const string stage = argc > 4 ? argv[4] : "all";
    const bool do_dino = stage == "all" || stage == "dino" || stage == "cond";
    const bool do_cond = stage == "all" || stage == "cond";
    const bool do_gpu  = stage == "all" || stage == "gpu";

    npy::Array images_512      = npy::load(dir + "/images_512.npy");        // [1,V,3,512,512]
    npy::Array camera_angle_x  = npy::load(dir + "/camera_angle_x.npy");    // [1,V]
    npy::Array transform_matrix = npy::load(dir + "/transform_matrix.npy"); // [1,V,4,4]
    npy::Array mesh_scale      = npy::load(dir + "/mesh_scale.npy");        // [1]
    npy::Array dino_tokens     = npy::load(dir + "/dino_tokens.npy");       // [V,1029,1024]
    npy::Array z_global        = npy::load(dir + "/z_global.npy");          // [1,5,1024]
    npy::Array z_proj          = npy::load(dir + "/z_proj.npy");            // [1,4096,1024]
    npy::Array v1_z_global     = npy::load(dir + "/v1_z_global.npy");       // [1,5,1024]
    npy::Array v1_z_proj       = npy::load(dir + "/v1_z_proj.npy");         // [1,4096,1024]

    const int V = (int)images_512.shape[1];
    const int S = (int)images_512.shape[3];
    const int64_t Ntok = dino_tokens.shape[1], Dc = dino_tokens.shape[2];
    const int R = (int)std::lround(std::cbrt((double)(z_proj.shape[1])));
    const float ms = mesh_scale.data[0];
    printf("fixture: V=%d S=%d Ntok=%lld Dc=%lld R=%d mesh_scale=%.4f\n",
           V, S, (long long)Ntok, (long long)Dc, R, ms);

    trellis::Model m = trellis::Model::load(gguf, gpu);
    printf("loaded %s (%zu tensors) on backend %s\n", m.arch.c_str(), m.tensors.size(), ggml_backend_name(m.backend));

    // ---- build per-view inputs ----
    const size_t plane3 = (size_t)3 * S * S;
    vector<Pixal3dView> views(V);
    for (int v = 0; v < V; ++v) {
        views[v].rgb_premult.assign(images_512.data.data() + (size_t)v * plane3,
                                     images_512.data.data() + (size_t)(v + 1) * plane3);
        views[v].fov_x = camera_angle_x.data[v];
        std::memcpy(views[v].c2w, transform_matrix.data.data() + (size_t)v * 16, 16 * sizeof(float));
    }

    bool all_ok = true;
    // ---- step 1: per-view DINOv3 encode vs dino_tokens[v] ----
    if (do_dino) {
    printf("\n=== step 1: per-view DINOv3 tokens ===\n");
    for (int v = 0; v < V; ++v) {
        vector<float> normed = pixal3d_imagenet_normalize(views[v].rgb_premult, S);
        vector<float> tok = dinov3_encode(m, normed, S);
        npy::Array ref_v;
        ref_v.shape = {Ntok, Dc};
        ref_v.data.assign(dino_tokens.data.data() + (size_t)v * Ntok * Dc,
                           dino_tokens.data.data() + (size_t)(v + 1) * Ntok * Dc);
        char nm[64]; snprintf(nm, sizeof(nm), "dino_tokens[v=%d]", v);
        all_ok &= compare(nm, tok, ref_v);
    }
    }

    vector<Pixal3dView> views1{views[0]};
    if (do_cond) {
    // ---- step 2: pixal3d_cond_ss for V views vs z_global/z_proj ----
    printf("\n=== step 2: pixal3d_cond_ss V=%d (host projection) ===\n", V);
    Pixal3dCond condV = pixal3d_cond_ss(m, views, S, R, ms);
    all_ok &= compare("z_global", condV.global, z_global);
    all_ok &= compare("z_proj", condV.proj, z_proj);

    // ---- step 3: V=1 (view 0 only) vs v1_z_global/v1_z_proj ----
    printf("\n=== step 3: pixal3d_cond_ss V=1 (host projection) ===\n");
    Pixal3dCond cond1 = pixal3d_cond_ss(m, views1, S, R, ms);
    all_ok &= compare("v1_z_global", cond1.global, v1_z_global);
    all_ok &= compare("v1_z_proj", cond1.proj, v1_z_proj);
    }

    if (do_gpu) {
    // ---- step 4: device-resident projection + MV fusion (pixal3d_cond_ss_gpu) ----
    // V=1 and V=V against the PyTorch fixtures; every V also against the host-projection path
    // run on the same backend (same DINOv3 numerics, so this isolates the get_rows/weighted-sum
    // projection + accumulation from the DINOv3 drift). V=2 has no fixture: host-path only.
    printf("\n=== step 4: pixal3d_cond_ss_gpu (device projection + MV average) ===\n");
    vector<int> vcounts;
    for (int n : {1, 2, 4}) if (n <= V && (n == V || n < V)) vcounts.push_back(n);
    if (V != 1 && V != 2 && V != 4) vcounts.push_back(V);
    for (int n : vcounts) {
        vector<Pixal3dView> vs(views.begin(), views.begin() + n);
        Pixal3dCondStats st;
        Pixal3dCond g = pixal3d_cond_ss_gpu(m, vs, S, R, ms, &st);
        print_stats("gpu", st);
        char nm[64];
        if (n == V) {
            snprintf(nm, sizeof nm, "gpu V=%d z_global vs ref", n); all_ok &= compare(nm, g.global, z_global);
            snprintf(nm, sizeof nm, "gpu V=%d z_proj vs ref", n);   all_ok &= compare(nm, g.proj, z_proj);
        }
        if (n == 1) {
            all_ok &= compare("gpu V=1 z_global vs ref", g.global, v1_z_global);
            all_ok &= compare("gpu V=1 z_proj vs ref", g.proj, v1_z_proj);
        }
        Pixal3dCond h = pixal3d_cond_ss(m, vs, S, R, ms);
        snprintf(nm, sizeof nm, "gpu V=%d z_global vs host", n); all_ok &= compare(nm, g.global, h.global, 1e-3);
        snprintf(nm, sizeof nm, "gpu V=%d z_proj vs host", n);   all_ok &= compare(nm, g.proj, h.proj, 1e-3);
    }
    }

    m.free();
    printf("\n=== overall %s ===\n", all_ok ? "PASS" : "FAIL (see above)");
    return all_ok ? 0 : 1;
}
