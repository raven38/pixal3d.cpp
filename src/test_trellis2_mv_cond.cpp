// trellis-test-trellis2-mv-cond: TASK-PORT P6. Ports the old track's cond-bank structural /
// self-consistency test (~/Downloads/pixal3d-mv-62/src/test_trellis2_mv_cond.cpp, 47 checks)
// onto #71's actual cond-bank shape. See docs/design/2026-09-21-trellis2-mv-port-hardening.md
// section "P6" for the full design.
//
// #71 has no Trellis2MvCondBank class / trellis2_mv_build_cond_bank() function -- the cond bank
// is a plain `vector<vector<float>>` built inline in trellis_cli.cpp's trellis_run()
// (~:862-872: threshold_cutout/birefnet_cutout -> normalize_cutout -> dinov3_encode, cascade
// builds a parallel 1024 bank, pointers materialized in the sample_bank lambda ~:898-899). This
// test's build_cond_bank() below is a file-local static helper that mirrors that exact sequence
// from the same shared primitives -- not a new production type, not a reimplementation of
// decision logic that lives elsewhere. trellis_cli.cpp is not touched.
//
// Dropped vs. the old track (see design doc for the full rationale):
//   - test_list_images() (12 checks): tested the old trellis2_mv_list_images() API. #71's
//     trellis::list_view_images() already has exhaustive coverage in test_list_view_images.cpp
//     (TASK-PORT P5) plus the 2/8/9-image boundary cases in tests/trellis2_mv_cli_contract.sh.
//   - Test 5's "view 1: " error-prefix match and "out unchanged after failure" check: those were
//     Trellis2MvCondBank's own API contract. Testing them against build_cond_bank() below would
//     only be testing this file's own helper, not #71.
//   - Test 6's bank.cascade()/cond_ptrs()/std::out_of_range: replaced with direct checks on the
//     plain vectors this port returns (see build_cond_bank()'s CondBank struct).
//
// Usage: trellis-test-trellis2-mv-cond [models_dir] [gpu]
// models_dir defaults to this machine's local Pixal3D q8_0 GGUF set; gpu defaults to -1 (CPU) to
// avoid shared-Metal contention (see this project's lessons on parallel sessions on one Mac).
#include "preprocess.h"
#include "dinov3.h"
#include "trellis_model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace trellis;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}
static void check_eq_size(size_t got, size_t want, const std::string& what) {
    const bool ok = got == want;
    printf("%s %s (got %zu, want %zu)\n", ok ? "ok  " : "FAIL", what.c_str(), got, want);
    if (!ok) ++g_fail;
}

static bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// Mirrors trellis_cli.cpp's trellis_run() inline per-view preprocess+encode loop (~:804-872),
// threshold-matte only (birefnet is a separate, already-tested path -- not this test's concern).
// Commits to out512/out1024 only on full success, so a failed call leaves prior contents of
// out512/out1024 untouched by construction (they are never assigned to before every view has
// preprocessed and encoded cleanly) -- mirroring the production fail-fast at trellis_cli.cpp:834-838.
struct CondBank { std::vector<std::vector<float>> cond512, cond1024; };

static bool build_cond_bank(const std::vector<std::string>& images, const std::string& models_dir,
                             int gpu, bool cascade, CondBank& out, std::string& err,
                             std::vector<size_t>* attempted = nullptr) {
    std::vector<std::vector<float>> chw_bank(images.size()), chw1024_bank;
    if (cascade) chw1024_bank.resize(images.size());
    for (size_t v = 0; v < images.size(); ++v) {
        if (attempted) attempted->push_back(v);
        int sz = 0;
        std::vector<unsigned char> cutout = threshold_cutout(images[v], sz);
        if (cutout.empty()) { err = "view " + std::to_string(v) + ": failed to read/preprocess image"; return false; }
        chw_bank[v] = normalize_cutout(cutout, sz, 512);
        if (cascade) chw1024_bank[v] = normalize_cutout(cutout, sz, 1024);
    }
    CondBank bank;
    bank.cond512.resize(images.size());
    if (cascade) bank.cond1024.resize(images.size());
    Model m = Model::load(models_dir + "/dinov3.gguf", gpu);
    for (size_t v = 0; v < images.size(); ++v) {
        bank.cond512[v] = dinov3_encode(m, chw_bank[v], 512);
        if (cascade) bank.cond1024[v] = dinov3_encode(m, chw1024_bank[v], 1024);
    }
    m.free();
    out.cond512 = std::move(bank.cond512);
    out.cond1024 = std::move(bank.cond1024);
    return true;
}

int main(int argc, char** argv) {
    const std::string models_dir = argc > 1 ? argv[1] : "/Users/<redacted-user>/nfs/weights/pixal3d/gguf-q8_0";
    const int gpu = argc > 2 ? std::atoi(argv[2]) : -1;

    { FILE* f = fopen((models_dir + "/dinov3.gguf").c_str(), "rb");
      if (!f) { fprintf(stderr, "cannot find %s/dinov3.gguf -- pass a models dir as argv[1]\n", models_dir.c_str()); return 2; }
      fclose(f); }

    const std::string goblin = "assets/goblin.png";
    const std::string axe = "assets/showcase/axe/axe.png";
    const std::string golem = "assets/showcase/golem/golem.png";
    for (const std::string& p : {goblin, axe, golem}) {
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) { fprintf(stderr, "missing test asset %s -- run from the repo root\n", p.c_str()); return 2; }
        fclose(f);
    }

    // ---- 1. determinism: two independent V=1 builds of the same image are bit-identical -------
    // (build_cond_bank() IS trellis_cli.cpp's own preprocess+encode chain, so this can't check
    // "matches a separate reference implementation" like the old track did -- it checks the
    // shared primitives leave no non-deterministic state across independent Model::load calls.)
    printf("-- 1-image determinism --\n");
    {
        CondBank a, b;
        std::string err;
        check(build_cond_bank({goblin}, models_dir, gpu, false, a, err), "cond bank builds for a single view (" + err + ")");
        check(build_cond_bank({goblin}, models_dir, gpu, false, b, err), "cond bank builds again for the same single view (" + err + ")");
        check_eq_size(a.cond512.size(), 1, "view_count == 1");
        if (a.cond512.size() == 1 && b.cond512.size() == 1)
            check(same(a.cond512[0], b.cond512[0]), "two independent V=1 builds of the same image are bit-identical");
    }

    // ---- 2/3/4. 2-view independence, reordering, duplicates --------------------------------
    printf("-- 2-view independence / reordering / duplicates --\n");
    std::vector<float> axe_standalone, golem_standalone;
    {
        CondBank a, g;
        std::string err;
        check(build_cond_bank({axe}, models_dir, gpu, false, a, err), "axe standalone (V=1) builds");
        check(build_cond_bank({golem}, models_dir, gpu, false, g, err), "golem standalone (V=1) builds");
        if (a.cond512.size() == 1) axe_standalone = a.cond512[0];
        if (g.cond512.size() == 1) golem_standalone = g.cond512[0];
    }
    {
        CondBank bank;
        std::string err;
        check(build_cond_bank({axe, golem}, models_dir, gpu, false, bank, err), "cond bank builds for 2 views (" + err + ")");
        check_eq_size(bank.cond512.size(), 2, "view_count == 2");
        if (bank.cond512.size() == 2) {
            check_eq_size(bank.cond512[0].size(), (size_t)1029 * 1024, "cond512[0] token count == 1029*1024");
            check_eq_size(bank.cond512[1].size(), (size_t)1029 * 1024, "cond512[1] token count == 1029*1024");

            // No averaging: each bank entry bit-matches its own V=1 standalone run.
            check(same(bank.cond512[0], axe_standalone), "bank.cond512[0] (axe) matches axe standalone V=1");
            check(same(bank.cond512[1], golem_standalone), "bank.cond512[1] (golem) matches golem standalone V=1");

            // Mutation test: a real averaging implementation would produce this vector instead --
            // prove our equality check actually distinguishes it from both standalone results.
            std::vector<float> averaged(bank.cond512[0].size());
            for (size_t i = 0; i < averaged.size(); ++i)
                averaged[i] = 0.5f * (bank.cond512[0][i] + bank.cond512[1][i]);
            check(!same(averaged, axe_standalone) && !same(averaged, golem_standalone),
                  "mutation test: the elementwise average of the two views is neither standalone result "
                  "(proves the standalone-match checks above can actually detect averaging)");
        }

        CondBank swapped;
        check(build_cond_bank({golem, axe}, models_dir, gpu, false, swapped, err), "cond bank builds for the reordered view list");
        if (bank.cond512.size() == 2 && swapped.cond512.size() == 2) {
            check(same(swapped.cond512[0], bank.cond512[1]) && same(swapped.cond512[1], bank.cond512[0]),
                  "reordering [axe,golem] -> [golem,axe] only permutes the bank, values unchanged");
        }

        CondBank dup;
        check(build_cond_bank({axe, axe}, models_dir, gpu, false, dup, err), "cond bank builds for a duplicated view list");
        if (dup.cond512.size() == 2)
            check(same(dup.cond512[0], dup.cond512[1]), "duplicate paths produce bit-identical duplicate entries");
    }

    // ---- 5. malformed image: fails whole request, only prior+failing views attempted ----------
    printf("-- malformed image --\n");
    {
        std::vector<size_t> attempted;
        std::string err;
        CondBank out;
        const bool ok = build_cond_bank({axe, "assets/showcase/does_not_exist.png", golem}, models_dir, gpu, false, out, err, &attempted);
        check(!ok, "a malformed path (index 1 of 3) fails the whole request");
        check_eq_size(attempted.size(), 2, "only views 0 and 1 were attempted (not 2)");
        if (attempted.size() == 2) check(attempted[0] == 0 && attempted[1] == 1, "attempted views are exactly {0,1} in order");
        check(out.cond512.empty(), "out is never assigned to after a failed build (mirrors trellis_cli.cpp's fail-fast)");
    }

    // ---- 6. 512/1024 shapes under cascade ---------------------------------------------------
    printf("-- cascade 512/1024 shapes --\n");
    {
        CondBank bank;
        std::string err;
        check(build_cond_bank({axe, golem}, models_dir, gpu, /*cascade=*/true, bank, err), "cascade cond bank builds (" + err + ")");
        if (bank.cond512.size() == 2) {
            check_eq_size(bank.cond512[0].size(), (size_t)1029 * 1024, "cascade cond512[0] token count");
            check_eq_size(bank.cond1024.size(), 2, "cond1024 bank has one entry per view under cascade");
            if (bank.cond1024.size() == 2) {
                check_eq_size(bank.cond1024[0].size(), (size_t)4101 * 1024, "cascade cond1024[0] token count == 4101*1024");
                check_eq_size(bank.cond1024[1].size(), (size_t)4101 * 1024, "cascade cond1024[1] token count == 4101*1024");
            }
            // Same pointer-materialization idiom as trellis_cli.cpp's sample_bank lambda (~:898-899).
            std::vector<const float*> ptrs512, ptrs1024;
            for (const auto& c : bank.cond512) ptrs512.push_back(c.data());
            for (const auto& c : bank.cond1024) ptrs1024.push_back(c.data());
            check_eq_size(ptrs512.size(), 2, "cond512 ptrs vector has one entry per view");
            check(ptrs512[0] == bank.cond512[0].data() && ptrs1024[1] == bank.cond1024[1].data(),
                  "ptrs vectors point at the bank's own storage");
        }

        CondBank non_cascade;
        check(build_cond_bank({axe}, models_dir, gpu, /*cascade=*/false, non_cascade, err), "non-cascade cond bank builds");
        check(non_cascade.cond1024.empty(), "non-cascade build leaves cond1024 empty");
    }

    // ---- 7. #59 numeric parity against the pinned Python fixture (DEV-ONLY, optional) ------
    printf("-- #59 fixture parity (DEV ONLY, not part of issue #60 acceptance) --\n");
    {
        // Same as the old track: tools/ref_trellis2_mv.py::real_run() is still not implemented in
        // this tree, so there is no fixture to compare against. This SKIP is expected and is NOT
        // evidence that any #59/#60 fixture-parity acceptance item is satisfied.
        const char* fixture_dir = std::getenv("TRELLIS2_MV_FIXTURE_DIR");
        if (!fixture_dir) {
            printf("SKIP: TRELLIS2_MV_FIXTURE_DIR not set and the #59 fixture generator is not "
                   "implemented yet -- this SKIP does not satisfy any fixture-parity acceptance item, "
                   "it only means this dev-only check has nothing to compare.\n");
        } else {
            fprintf(stderr, "TRELLIS2_MV_FIXTURE_DIR=%s given but this port does not implement "
                    "fixture loading (no fixture exists yet to load) -- fix this once the fixture lands.\n", fixture_dir);
            ++g_fail;
        }
    }

    printf(g_fail ? "\nTRELLIS2_MV_COND_TEST_FAIL (%d)\n" : "\nTRELLIS2_MV_COND_TEST_OK\n", g_fail);
    return g_fail ? 1 : 0;
}
