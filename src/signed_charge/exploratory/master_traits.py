"""Traits of Phantom's master fingers, per finger: master_traits.py FIRST PEOPLE

One JSON line per master: pattern; core-delta ridge counts (ridges crossing the straight line between a core and
a delta; a whorl's larger, as Holt counts); core-delta distance (mm, in the print) and direction (degrees, 0 to the
right, 90 down, as printed); the core's place on the master; ridge spacing in the rolled print (mm); minutiae per
mm² of print in rings around the core. CPU only (`synthesis.use_device("cpu")`), so it can run beside GPU work.

Run: python -m signed_charge.exploratory.master_traits 2000 50 > traits.jsonl
"""

import json
import sys

import numpy as np

from ridgegen import finger as fingers
from ridgegen import fingerprints as fp
from ridgegen import synthesis

synthesis.use_device("cpu")
MM = 500 / 25.4


def ridge_count(ridges, a, b):
    n = max(int(np.hypot(b[0] - a[0], b[1] - a[1])), 2)
    t = np.linspace(0.06, 0.94, n)
    xs = np.clip((a[0] + t * (b[0] - a[0])).round().astype(int), 0, ridges.shape[1] - 1)
    ys = np.clip((a[1] + t * (b[1] - a[1])).round().astype(int), 0, ridges.shape[0] - 1)
    v = ridges[ys, xs] > 0
    return int(np.sum(v[1:] & ~v[:-1]))


def traits(seed, f):
    m = fingers.master(seed, f)
    # The print's scale, as a rolled print draws it (fingerprints.rolled_capture, without the capture's jitter).
    s = fp.finger_scale(seed, f) * min(1.0, 1.02 / m.scale)
    out = {"seed": seed, "fgp": f, "finger": m.name, "hand": m.hand, "pattern": m.pattern, "spacing_mm": m.period * s / MM, "print_scale": s}
    if m.cores:
        cx, cy = np.mean(m.cores, 0)
        out["core"] = [cx / fingers.MASTER_W, cy / fingers.MASTER_H]
        xy = np.array([(p["x"], p["y"]) for p in m.minutiae], float)
        r = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) * s / MM  # mm in the print
        rings = {}
        for lo, hi in ((0, 3), (3, 6), (6, 9)):
            rings[f"{lo}-{hi}"] = float(np.sum((r >= lo) & (r < hi)) / (np.pi * (hi**2 - lo**2)))
        out["minutiae_per_mm2"] = rings
    if m.cores and m.deltas:
        pairs = []
        for d in m.deltas:
            c = min(m.cores, key=lambda c: np.hypot(c[0] - d[0], c[1] - d[1]))
            pairs.append((ridge_count(m.ridges, c, d), np.hypot(d[0] - c[0], d[1] - c[1]) * s / MM, float(np.degrees(np.arctan2(d[1] - c[1], d[0] - c[0])))))
        rc, dist, angle = max(pairs)
        out.update(ridge_count=rc, ridge_counts=[p[0] for p in pairs], core_delta_mm=dist, delta_direction=angle)
    return out


first, people = int(sys.argv[1]), int(sys.argv[2])
for seed in range(first, first + people):
    for f in range(1, 11):
        print(json.dumps(traits(seed, f)), flush=True)
    print(seed, file=sys.stderr, flush=True)
