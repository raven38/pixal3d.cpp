#!/bin/zsh
# U1: nth パッチを submodule の作業ツリーへ当て、別 build dir（build-metal-nth）で 2 つのテストだけをビルドし、必ず戻す。
# GGML_METAL_EMBED_LIBRARY=ON なので shader はビルド時に埋め込まれ、走っている build-metal のバイナリには影響しない。
# build-metal はパッチ適用中に絶対に再ビルドしない。
set -u
REPO=/Users/s25705/Downloads/pixal3d-metal-groupnorm
S=$REPO/docs/results/2026-09-20-metal-groupnorm/scripts
cd $REPO
python3 $S/nth_patch.py apply || exit 1
python3 $S/nth_patch.py check
( cd thirdparty/ggml && git diff --stat )
cmake -B build-metal-nth -G Ninja -DCMAKE_BUILD_TYPE=Release > build-metal-nth-configure.log 2>&1 || { echo "configure failed"; python3 $S/nth_patch.py revert; exit 2; }
cmake --build build-metal-nth -j6 --target trellis-test-naf trellis-test-pixal3d-cond-tex > build-metal-nth-build.log 2>&1; rc=$?
echo "build rc=$rc"; tail -2 build-metal-nth-build.log
python3 $S/nth_patch.py revert
( cd thirdparty/ggml && git status --short && echo "submodule status printed above (empty = clean)" )
exit $rc
