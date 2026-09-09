// Validate SS conv decoder vs PyTorch golden (deterministic — feed same latent).
//   trellis-test-ss-dec <ss_dec.gguf> <samples.npy> <ref_logits.npy> [gpu] [dump_npy]
// dump_npy: 指定すると occupancy logits をそのまま書き出す。同じ機械の別バックエンド同士
// （CPU backend gpu=-1 と CUDA gpu=0 など）を突き合わせるために使う。golden を持ち込めない
// 環境で backend 間の一致を測る用途。
#include "trellis_model.h"
#include "ss_decoder.h"
#include "npy.h"
#include <cmath>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <ss_dec.gguf> <samples.npy> <ref_logits.npy> [gpu] [dump_npy]\n", argv[0]); return 1; }
    const int gpu = argc > 4 ? atoi(argv[4]) : 0;
    const char* dump_npy = argc > 5 && argv[5][0] ? argv[5] : nullptr;
    trellis::Model m = trellis::Model::load(argv[1], gpu);
    npy::Array z = npy::load(argv[2]);            // [1,8,16,16,16]
    npy::Array gold = npy::load(argv[3]);         // [1,1,64,64,64]

    std::vector<float> logits = trellis::ss_decode(m, z.data);
    if (dump_npy) npy::save(dump_npy, logits.data(), {1, 1, 64, 64, 64});

    int nonfinite = 0;
    for (float v : logits) if (!std::isfinite(v)) ++nonfinite;
    printf("ss_logits n=%zu nonfinite=%d\n", logits.size(), nonfinite);

    if ((int64_t)logits.size() != gold.numel()) { printf("SIZE MISMATCH %zu vs %lld\n", logits.size(), (long long)gold.numel()); return 1; }

    double maxd = 0, gmax = 0; int act_mine = 0, act_ref = 0, agree = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        maxd = std::max(maxd, (double)std::fabs(logits[i] - gold.data[i]));
        gmax = std::max(gmax, (double)std::fabs(gold.data[i]));
        bool am = logits[i] > 0, ar = gold.data[i] > 0;
        act_mine += am; act_ref += ar; agree += (am == ar);
    }
    double iou_num = 0, iou_den = 0;
    for (size_t i = 0; i < logits.size(); ++i) { bool am=logits[i]>0, ar=gold.data[i]>0; iou_num += (am&&ar); iou_den += (am||ar); }
    printf("logits max|d|=%.4e rel=%.4e\n", maxd, maxd / gmax);
    printf("active: mine=%d ref=%d  voxel-agree=%.4f%%  IoU=%.4f\n",
           act_mine, act_ref, 100.0 * agree / logits.size(), iou_den > 0 ? iou_num / iou_den : 1.0);
    printf("%s\n", (maxd / gmax < 5e-2 && (iou_den == 0 || iou_num / iou_den > 0.95)) ? "OK" : "**");
    m.free();
    return 0;
}
