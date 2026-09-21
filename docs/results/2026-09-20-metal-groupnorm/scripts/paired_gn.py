#!/usr/bin/env python3
"""issue #55 の paired block 集計。ab2/ の <arm>_<name>_<HHMMSS>.log を時刻順に並べ、k 番目の new と k 番目の old を block k の pair とみなす
（ab_gn2.sh は block ごとに ABBA で順序を入れ替えるので、時刻順の k 番目同士が同じ block）。
P1（nafops_*）: whole graph 秒、GROUP_NORM / NORM 合計、ms/op。P2（tex / hr / lr）: [gpu] total 秒、naf_enc compute/view（4 view の中央値）。
汚染 run（--contam file: 1 行 1 ファイル名、classify_runs.py の CONTAMINATED）を含む pair は差の統計から除外し、* を付けて表示する。
使い方: paired_gn.py <ab_dir> [--contam contam.txt]"""
import re, glob, os, sys, statistics
D = sys.argv[1]; contam = set()
if "--contam" in sys.argv:
    contam = set(l.strip() for l in open(sys.argv[sys.argv.index("--contam") + 1]) if l.strip())
def parse(f):
    s = open(f, errors="replace").read(); r = {}
    w = re.search(r"whole graph ([\d.]+)s", s)
    if w:
        r["main"] = float(w.group(1)); r["gn"] = 0.0
        for op in ("GROUP_NORM", "NORM"):
            m = re.search(r"\[prof\]\s+%s\s+([\d.]+)s\s+([\d.]+)%%\s+\(n=(\d+), ([\d.]+) ms/op\)" % op, s)
            if m: r["gn"] += float(m.group(1)); r["op"] = f"{op} {float(m.group(4)):.1f} ms/op ({m.group(2)}%)"
    else:
        t = re.search(r"\[gpu\].*total=(\d+) ms\s+slowest view=(\d+) ms", s)
        if not t: return None
        r["main"] = int(t.group(1)) / 1000.0; r["slowest"] = int(t.group(2)) / 1000.0
        enc = [float(m.group(1)) for m in re.finditer(r"naf_enc x\d+: build [\d.]+ up [\d.]+ compute ([\d.]+)", s)]
        single = [float(m.group(1)) for m in re.finditer(r"cond_slat_gpu_single[^\n]*\| compute ([\d.]+) \|", s)]
        r["enc"] = statistics.median(enc) if enc else (statistics.median(single) if single else None)
        r["enc_kind"] = "naf_enc" if enc else ("single-graph" if single else "-")
    return r
names = sorted(set(re.match(r"(new|old)_(\S+)_\d{6}\.log", os.path.basename(f)).group(2) for f in glob.glob(f"{D}/*_*.log") if re.match(r"(new|old)_(\S+)_\d{6}\.log", os.path.basename(f))))
for name in names:
    rows = {}
    for arm in ("new", "old"):
        rows[arm] = [(os.path.basename(f), parse(f), os.path.basename(f) in contam) for f in sorted(glob.glob(f"{D}/{arm}_{name}_*.log"))]
        rows[arm] = [r for r in rows[arm] if r[1] is not None]
    print(f"== {name}  (new n={len(rows['new'])}, old n={len(rows['old'])}, contaminated: {sum(r[2] for a in rows for r in rows[a])})")
    for arm in ("new", "old"):
        for b, r, ct in rows[arm]:
            extra = f"gn {r['gn']:.2f} s {r.get('op','')}" if "gn" in r else f"slowest view {r['slowest']:.2f} s  {r['enc_kind']} compute/view {r['enc']:.2f} s" if r.get("enc") is not None else ""
            print(f"  {arm} {b[-10:-4]}  {r['main']:.2f} s  {extra} {'CONTAM' if ct else ''}")
    pairs = list(zip(rows["new"], rows["old"]))
    diffs = [(o[1]["main"] - n[1]["main"], n[1]["main"], o[1]["main"], n[2] or o[2]) for n, o in pairs]
    clean = [d for d in diffs if not d[3]]
    print("  paired old-new:", " / ".join(f"{d[0]:.2f}{'*' if d[3] else ''}" for d in diffs), "(s; * = pair with a contaminated run, excluded below)")
    if clean:
        print(f"  clean pairs n={len(clean)}: diff median {statistics.median([d[0] for d in clean]):.2f} / min {min(d[0] for d in clean):.2f} / max {max(d[0] for d in clean):.2f} s; "
              f"old/new median {statistics.median([d[2]/d[1] for d in clean]):.2f}x")
    for arm in ("new", "old"):
        v = [r[1]["main"] for r in rows[arm] if not r[2]]
        if v: print(f"  {arm} clean n={len(v)}: median {statistics.median(v):.2f} / min {min(v):.2f} / max {max(v):.2f} s")
    if all(r[1].get("enc") is not None for a in rows for r in rows[a]) and rows["new"]:
        for arm in ("new", "old"):
            v = [r[1]["enc"] for r in rows[arm] if not r[2]]
            if v: print(f"  {arm} {rows[arm][0][1]['enc_kind']} compute/view: median {statistics.median(v):.2f} / min {min(v):.2f} / max {max(v):.2f} s")
