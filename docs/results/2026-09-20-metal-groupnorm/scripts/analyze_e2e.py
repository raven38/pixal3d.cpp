#!/usr/bin/env python3
"""E1 の集計: 旧 / 新 の E2E ログから (1) active voxels / decoded voxels、(2) [cond] の段別ラップ（cond_slat 3 段の合計）、
(3) 全体時間、を表にする。GLB の指標は tools/glb_metrics.py、レンダは render5.py を別途回す。
使い方: analyze_e2e.py <e2e_old.log> <e2e_new.log>"""
import re, sys
def parse(path):
    s = open(path, errors="replace").read()
    r = {}
    m = re.search(r"active voxels @res32 = (\d+)", s); r["ss_voxels"] = int(m.group(1)) if m else None
    m = re.search(r"decoded voxels @res1024 = (\d+)", s); r["decoded_voxels"] = int(m.group(1)) if m else None
    m = re.search(r"done in ([\d.]+)s", s); r["total_s"] = float(m.group(1)) if m else None
    r["cond_slat"] = [(x.group(1), float(x.group(2))) for x in re.finditer(r"\[cond\] (cond_slat S=\d+ R=\S+ T=\d+ \([^)]*\)) \(([\d.]+)s\)", s)]
    r["cond_ss"] = [float(x.group(1)) for x in re.finditer(r"\[cond\] cond_ss [^(]*\([^)]*\) \(([\d.]+)s\)", s)]
    r["flow"] = [float(x.group(1)) for x in re.finditer(r"\[cond\] flow \(load \+ runner \+ sampler\) \(([\d.]+)s\)", s)]
    r["accounted"] = [(x.group(1), float(x.group(2))) for x in re.finditer(r"\[cond\] (.+?) accounted \(([\d.]+)s\)", s)]
    r["hr_voxels"] = [int(x.group(1)) for x in re.finditer(r"(?:HR|hr)[^\n]*?(\d{4,6}) (?:voxels|tokens)", s)][:2]
    m = re.search(r"\[post\][^\n]*decimate[^\n]*\(([\d.]+)s\)", s); r["decimate_s"] = float(m.group(1)) if m else None
    return r
a, b = parse(sys.argv[1]), parse(sys.argv[2])
print(f"{'':28s} {'old':>12s} {'new':>12s}")
for k in ("ss_voxels", "decoded_voxels", "total_s"):
    print(f"{k:28s} {str(a[k]):>12s} {str(b[k]):>12s}")
print("cond_slat stages:")
ta = tb = 0.0
for (na, va), (nb, vb) in zip(a["cond_slat"], b["cond_slat"]):
    ta += va; tb += vb; print(f"  {na[:60]:60s} {va:8.1f} {vb:8.1f}  d={vb - va:+.1f}")
print(f"  {'cond_slat total':60s} {ta:8.1f} {tb:8.1f}  d={tb - ta:+.1f}")
print("flow laps:", a["flow"], b["flow"])
print("accounted:", a["accounted"]); print("           ", b["accounted"])
