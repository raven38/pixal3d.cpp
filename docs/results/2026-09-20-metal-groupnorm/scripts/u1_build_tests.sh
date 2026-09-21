#!/bin/zsh
# U1 補足: パッチ適用中に build-metal-nth へ ggml の test-backend-ops を足す（GGML_BUILD_TESTS=ON）。終わったら必ず戻す。
set -u
REPO=/Users/s25705/Downloads/pixal3d-metal-groupnorm
S=$REPO/docs/results/2026-09-20-metal-groupnorm/scripts
cd $REPO
python3 $S/nth_patch.py apply || exit 1
cmake -B build-metal-nth -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_TESTS=ON > build-metal-nth-configure2.log 2>&1 || { echo "configure failed"; python3 $S/nth_patch.py revert; exit 2; }
cmake --build build-metal-nth -j6 --target test-backend-ops > build-metal-nth-build2.log 2>&1; rc=$?
echo "build rc=$rc"; tail -2 build-metal-nth-build2.log
python3 $S/nth_patch.py revert
( cd thirdparty/ggml && git status --short && echo "submodule status printed above (empty = clean)" )
find build-metal-nth -name "test-backend-ops" -type f
exit $rc
