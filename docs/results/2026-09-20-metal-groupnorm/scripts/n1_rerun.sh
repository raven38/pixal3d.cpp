#!/bin/zsh
# N1 の再実行（PR #56 レビュー: 計時区間を kernel 完了までにした版）。ロック + ゲートを通してから Metal → CPU の順に 1 回ずつ。
REPO=/Users/<redacted-user>/Downloads/pixal3d-metal-groupnorm
D=$REPO/docs/results/2026-09-20-metal-groupnorm
SNAP=$D/numerics/n1_rerun_ps.log; SELFKEY=pixal3d-metal-groupnorm
source $D/scripts/gate_lib.sh
gate "gn-oracle ~30s"; snapshot "before n1 rerun"
$REPO/build-metal/trellis-test-naf --gn-oracle $HOME/nfs/weights/pixal3d/gguf-q8_0/pixal3d_naf.gguf 0 > $D/numerics/n1_metal_sync.log 2>&1; echo "metal exit=$?"
$REPO/build-metal/trellis-test-naf --gn-oracle $HOME/nfs/weights/pixal3d/gguf-q8_0/pixal3d_naf.gguf -1 > $D/numerics/n1_cpu_sync.log 2>&1; echo "cpu exit=$?"
snapshot "after n1 rerun"; lock_release
