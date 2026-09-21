#!/bin/zsh
# U1 補足: パッチ入り Metal backend で ggml の test-backend-ops -o GROUP_NORM（Metal vs CPU、ggml 標準の合否）。
set -u
REPO=/Users/<redacted-user>/Downloads/pixal3d-metal-groupnorm
D=$REPO/docs/results/2026-09-20-metal-groupnorm; OUT=$D/nth; mkdir -p $OUT
SNAP=$OUT/ps_snapshots.log; SELFKEY=pixal3d-metal-groupnorm
GPU_QUIET=${GPU_QUIET:-40}; QUIET_SECS=${QUIET_SECS:-30}
source $D/scripts/gate_lib.sh
gate "test-backend-ops GROUP_NORM ~1min"; snapshot "before tbo"
$REPO/build-metal-nth/test-backend-ops test -b MTL0 -o GROUP_NORM > $OUT/test_backend_ops_group_norm_nth.log 2>&1; echo "tbo GROUP_NORM exit=$?"
$REPO/build-metal-nth/test-backend-ops test -b MTL0 -o GROUP_NORM_MUL_ADD > $OUT/test_backend_ops_group_norm_mul_add_nth.log 2>&1; echo "tbo GROUP_NORM_MUL_ADD exit=$?"
snapshot "after tbo"; lock_release
echo "TBO DONE $(date)"
