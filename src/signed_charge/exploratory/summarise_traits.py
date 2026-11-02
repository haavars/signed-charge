"""Summary of master_traits.py's lines, by finger: summarise_traits.py TRAITS.jsonl"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

rows = [json.loads(line) for line in open(sys.argv[1])]
FINGERS = ["thumb", "index", "middle", "ring", "little"]
LOOPS = ("left_loop", "right_loop")


def ms(xs):
    xs = np.array([x for x in xs if x is not None], float)
    if not len(xs):
        return "-"
    d = 3 if abs(xs.mean()) < 2 else 1
    return f"{xs.mean():.{d}f} ({xs.std():.{d}f}) n={len(xs)}"


print(f"{len(rows)} masters, {len({r['seed'] for r in rows})} people\n")
print("Pattern frequencies by finger")
for f in FINGERS:
    c = Counter(r["pattern"] for r in rows if r["finger"] == f)
    n = sum(c.values())
    print(f"  {f:7s}", ", ".join(f"{k} {v / n:.2f}" for k, v in c.most_common()))
print("\nRidge count, mean (SD): loops | whorls (larger count) | all patterns, arches 0")
for f in FINGERS:
    mine = [r for r in rows if r["finger"] == f]
    loops = [r["ridge_count"] for r in mine if r["pattern"] in LOOPS]
    whorls = [r.get("ridge_count") for r in mine if r["pattern"] not in LOOPS and "arch" not in r["pattern"]]
    every = [r.get("ridge_count", 0) if "arch" not in r["pattern"] else 0 for r in mine]
    print(f"  {f:7s} {ms(loops)} | {ms(whorls)} | {ms(every)}")
tfrc = defaultdict(int)
for r in rows:
    tfrc[r["seed"]] += r.get("ridge_count", 0) if "arch" not in r["pattern"] else 0
print(f"  TFRC per person: {ms(list(tfrc.values()))}")
print("\nLoops: core-delta distance in the print (mm), delta direction (deg, 0 right, 90 down)")
for f in FINGERS:
    mine = [r for r in rows if r["finger"] == f and r["pattern"] in LOOPS]
    ang = [abs(90 - abs(r["delta_direction"])) for r in mine]  # off vertical, either side
    print(f"  {f:7s} distance {ms([r['core_delta_mm'] for r in mine])}  off vertical {ms(ang)}")
print("\nRidge spacing in the rolled print (mm) and its spread between people")
for f in FINGERS:
    print(f"  {f:7s} {ms([r['spacing_mm'] for r in rows if r['finger'] == f])}")
print(f"  all     {ms([r['spacing_mm'] for r in rows])}")
print("\nMinutiae per mm² of print, by distance from the core (mm)")
for ring in ("0-3", "3-6", "6-9"):
    print(f"  {ring}: {ms([r['minutiae_per_mm2'][ring] for r in rows if 'minutiae_per_mm2' in r])}")
print("\nCore position on the master (fraction of width, height): SD")
xy = np.array([r["core"] for r in rows if "core" in r])
print(f"  x {xy[:, 0].mean():.3f} ± {xy[:, 0].std():.3f}, y {xy[:, 1].mean():.3f} ± {xy[:, 1].std():.3f}")
