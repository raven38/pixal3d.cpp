#!/usr/bin/env bash
# Weightless TRELLIS.2 multi-image CLI contract (#65).
set -euo pipefail

CLI="${1:?usage: trellis2_mv_cli_contract.sh <trellis-cli>}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/views" "$WORK/one" "$WORK/eight" "$WORK/nine" "$WORK/models"

python3 - "$WORK/views/view10.png" "$WORK/views/view2.png" "$WORK/one/only.png" "$WORK/eight" "$WORK/nine" <<'PY'
import struct, sys, zlib
def write(path, rgb):
    w=h=16
    rows=[]
    for y in range(h):
        row=bytearray([0])
        for x in range(w):
            inside=3 <= x < 13 and 3 <= y < 13
            row += bytes([*rgb,255]) if inside else bytes([0,0,0,0])
        rows.append(bytes(row))
    raw=b"".join(rows)
    def chunk(t,d): return struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
    png=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,6,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b"")
    open(path,"wb").write(png)
write(sys.argv[1], (200,80,40))
write(sys.argv[2], (40,180,90))
write(sys.argv[3], (80,90,220))
import os
for i in range(8):
    write(os.path.join(sys.argv[4], f"view{i}.png"), (10*i, 20*i % 256, 30*i % 256))
for i in range(9):
    write(os.path.join(sys.argv[5], f"view{i}.png"), (10*i, 20*i % 256, 30*i % 256))
PY

fail=0
ok(){ echo "ok   $*"; }
bad(){ echo "FAIL $*" >&2; fail=$((fail+1)); }

# 2 images + --bg-only stops before DINO/model loading. This verifies the real
# directory enumerator and printed deterministic natural order.
if "$CLI" --trellis2-mv "$WORK/views" --trellis2-mv-mode stochastic     --models "$WORK/models" --bg-only "$WORK/out.glb" >"$WORK/run.log" 2>&1; then
  ok "2-image TRELLIS.2 MV reaches preprocessing without model weights"
else
  bad "2-image bg-only run failed: $(tail -20 "$WORK/run.log")"
fi
a="$(grep -n 'view0 view2.png' "$WORK/run.log" | cut -d: -f1 || true)"
b="$(grep -n 'view1 view10.png' "$WORK/run.log" | cut -d: -f1 || true)"
if [ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ]; then
  ok "natural order is view2 before view10"
else
  bad "natural-order log missing/wrong"; cat "$WORK/run.log"
fi
grep -q 'TRELLIS.2 multiview: V=2 mode=stochastic' "$WORK/run.log"   && ok "mode/view count logged" || bad "missing mode/view-count log"

# One image is rejected before model loading.
if "$CLI" --trellis2-mv "$WORK/one" --models "$WORK/models" --bg-only "$WORK/o.glb" >"$WORK/one.log" 2>&1; then
  bad "one-image directory was accepted"
elif grep -q 'requires 2..8 images' "$WORK/one.log"; then
  ok "one image fails closed"
else
  bad "one-image error was not actionable: $(cat "$WORK/one.log")"
fi

# Parser-level separation from camera-aware Pixal3D.
if "$CLI" --trellis2-mv "$WORK/views" --trellis2-mv-mode bad "$WORK/o.glb" >"$WORK/badmode.log" 2>&1; then
  bad "invalid fusion mode was accepted"
elif grep -q "expects 'stochastic' or 'multidiffusion'" "$WORK/badmode.log"; then
  ok "invalid fusion mode rejected"
else bad "bad-mode diagnostic changed"; fi

if "$CLI" --trellis2-mv-mode stochastic "$WORK/o.glb" >"$WORK/nodir.log" 2>&1; then
  bad "fusion mode without directory was accepted"
elif grep -q 'requires --trellis2-mv DIR' "$WORK/nodir.log"; then
  ok "fusion mode requires TRELLIS.2 MV directory"
else bad "no-dir diagnostic changed"; fi

if "$CLI" --trellis2-mv "$WORK/views" --views "$WORK/views" "$WORK/o.glb" >"$WORK/conflict.log" 2>&1; then
  bad "TRELLIS.2 MV + Pixal3D --views was accepted"
elif grep -q 'mutually exclusive' "$WORK/conflict.log"; then
  ok "TRELLIS.2 MV and Pixal3D --views are separate modes"
else bad "mode-conflict diagnostic changed"; fi

# 2..8 upper-bound boundary (TASK-PORT P5): 8 images is the accepted maximum, 9 is rejected.
# Exercises the real CLI (trellis_cli.cpp's `names.size() < 2 || names.size() > 8` gate), not a
# reimplementation, so a regression in the production bound is caught here.
if "$CLI" --trellis2-mv "$WORK/eight" --models "$WORK/models" --bg-only "$WORK/e.glb" >"$WORK/eight.log" 2>&1; then
  ok "eight images (upper bound) is accepted"
else
  bad "eight-image directory was rejected: $(tail -20 "$WORK/eight.log")"
fi
if "$CLI" --trellis2-mv "$WORK/nine" --models "$WORK/models" --bg-only "$WORK/n.glb" >"$WORK/nine.log" 2>&1; then
  bad "nine images (over the upper bound) was accepted"
elif grep -q 'requires 2..8 images' "$WORK/nine.log"; then
  ok "nine images (over the upper bound) fails closed"
else
  bad "nine-image error was not actionable: $(cat "$WORK/nine.log")"
fi

if [ "$fail" -eq 0 ]; then echo "TRELLIS2_MV_CLI_CONTRACT_OK"; exit 0; fi
echo "TRELLIS2_MV_CLI_CONTRACT_FAIL ($fail)" >&2
exit 1
