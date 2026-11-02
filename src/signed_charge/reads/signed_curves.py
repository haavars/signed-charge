"""Signed charge's full curves and what they survive (D-046, E-077): signed_curves.py OUT.json

For the paper. The sets and filters of `signed_false`, with h(r) and C(R) at every range kept, by finger, and three changes to the measurement itself:

- `no_singular`: leaving out minutiae and pair segments within `GUARD` px of a core or delta, where lifting a sign along a segment is ambiguous, as the flow's turn round a +½ point depends on which side the segment passes.
- `scale6`, `scale14`: the flow smoothed over 6 and 14 px, against 9, on all minutiae.
- `phantom`: Phantom's rendered slap fingertips, ridgegen 0.25, a diffusion render of model 2's masters: `all` only, as they lack FVC's eight impressions.

CIs by finger. FVC B only, and the fresh people of `signed_false`.

Run: python -m signed_charge.reads.signed_curves OUT.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_check as fc
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("SIGNED_JOBS", 8))
GUARD, RESAMPLES = 30, 2000
KEYS = ("all", "confirmed", "no_singular", "scale6", "scale14")
BLOCK = fs.B


def singular(theta, fg):
    """Cores and deltas, (x, y) in px: as `field_similarity.cores`, of both signs."""
    th = theta[BLOCK // 2 :: BLOCK, BLOCK // 2 :: BLOCK]
    f = cv2.erode(fg.astype(np.uint8), np.ones((5 * BLOCK, 5 * BLOCK), np.uint8))[BLOCK // 2 :: BLOCK, BLOCK // 2 :: BLOCK].astype(bool)
    d = lambda a, b: (b - a + np.pi / 2) % np.pi - np.pi / 2  # noqa: E731
    a, b, c, e = th[:-1, :-1], th[:-1, 1:], th[1:, 1:], th[1:, :-1]
    index = (d(a, b) + d(b, c) + d(c, e) + d(e, a)) / (2 * np.pi)
    ok = f[:-1, :-1] & f[1:, 1:] & f[:-1, 1:] & f[1:, :-1]
    ys, xs = np.nonzero((np.abs(index) > 0.4) & ok)
    return (np.stack([xs, ys], 1) + 1) * BLOCK


def smoothed(im, scale):
    keep = sm.ORIENTATION_SCALE
    sm.ORIENTATION_SCALE = scale
    field = sm.direction(im)
    sm.ORIENTATION_SCALE = keep
    return field


def finger(task):
    name, label, images = task
    ready = []
    for im in images:
        fg = fs.foreground(im)
        if fg is None:
            continue
        inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
        guarded = inner.copy()
        yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
        for x, y in singular(fs.field(im), fg):
            guarded &= np.hypot(xx - x, yy - y) >= GUARD
        ready.append((im, inner, guarded))
    out = {"set": name, "finger": label, "prints": len(ready)}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        points = [sf.inside(extract(im), m) for im, m, _ in ready]
        counts = sf.support(points, y_sign) if len(ready) > 2 else [np.zeros(len(p), int) for p in points]
        sums = {k: [np.zeros(len(sm.EDGES) - 1), np.zeros(len(sm.EDGES) - 1), 0] for k in KEYS}
        for p, c, (im, m, guarded) in zip(points, counts, ready):
            field = sm.direction(im)
            cases = {"all": (p, field, m), "confirmed": (p[c >= sf.SUPPORT] if len(p) else p, field, m), "no_singular": (sf.inside(p, guarded), field, guarded), "scale6": (p, smoothed(im, 6.0), m), "scale14": (p, smoothed(im, 14.0), m)}
            for key, (use, f, mask) in cases.items():
                total, count, n = sf.binned(use, y_sign, f, mask)
                sums[key][0] += total
                sums[key][1] += count
                sums[key][2] += n
        out[reader] = {k: {"sum": v[0], "count": v[1], "minutiae": v[2]} for k, v in sums.items()}
    return out


def fvc_task(task):
    name, label, paths = task
    return finger((name, label, [fs.at_500(cv2.imread(p, 0), cf.PPI.get(name, 500)) for p in paths]))


def variant_task(task):
    variant, seed, fgp = task
    return finger((variant, f"PH{seed}/{fgp}", [fc.image((variant, seed, fgp, c))[1] for c in sf.CAPTURES]))


def curves(rows, reader, key, rng):
    total = np.array([r[reader][key]["sum"] for r in rows])
    count = np.array([r[reader][key]["count"] for r in rows])
    minutiae = np.array([r[reader][key]["minutiae"] for r in rows], float)
    h = lambda i: total[i].sum(0) / np.maximum(count[i].sum(0), 1)  # noqa: E731
    screen = lambda i, R: 2 * total[i][:, sm.EDGES[1:] <= R].sum() / max(minutiae[i].sum(), 1)  # noqa: E731
    every = np.arange(len(rows))
    draws = [rng.integers(len(rows), size=len(rows)) for _ in range(RESAMPLES)]
    hs = np.array([h(d) for d in draws])
    return {"fingers": len(rows), "minutiae_a_print": float(minutiae.sum() / max(sum(r["prints"] for r in rows), 1)), "r": ((sm.EDGES[:-1] + sm.EDGES[1:]) / 2).tolist(), "h": h(every).tolist(), "h_low": np.percentile(hs, 2.5, 0).tolist(), "h_high": np.percentile(hs, 97.5, 0).tolist(), "pairs": count.sum(0).tolist(),
            "c": {str(R): {"mean": float(screen(every, R)), "ci": [float(np.percentile([screen(d, R) for d in draws], q)) for q in (2.5, 97.5)]} for R in sm.RADII}}


def main(out):
    started = provenance.start()
    fingers = [(v, s, f) for v in sf.VARIANTS for s in sf.FRESH for f in sf.FINGERS]
    groups = {}
    for (_, key, _), im in cf.phantom(60):
        groups.setdefault(key, []).append(im)
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(fvc_task, sf.fvc_fingers(), chunksize=1))
        print("fvc", len(rows), flush=True)
        rows += list(pool.map(variant_task, fingers, chunksize=1))
        print("variants", len(rows), flush=True)
        rows += list(pool.map(finger, [("phantom", k, v) for k, v in groups.items()], chunksize=1))
    sets = {"real": [r for r in rows if r["set"].startswith("FVC") and "DB4" not in r["set"]], "sfinge": [r for r in rows if "DB4" in r["set"]], "phantom": [r for r in rows if r["set"] == "phantom"], **{v: [r for r in rows if r["set"] == v] for v in sf.VARIANTS}}
    rng = np.random.default_rng(0)
    report = {reader: {name: {key: curves(group, reader, key, rng) for key in KEYS if group and (key != "confirmed" or name != "phantom")} for name, group in sets.items()} for reader in sm.EXTRACTORS}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    for reader in sm.EXTRACTORS:
        for name, entry in report[reader].items():
            print(reader, name.ljust(12), {k: (v["fingers"], round(v["c"]["60"]["mean"], 2), [round(c, 2) for c in v["c"]["60"]["ci"]], round(v["c"]["150"]["mean"], 2)) for k, v in entry.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
