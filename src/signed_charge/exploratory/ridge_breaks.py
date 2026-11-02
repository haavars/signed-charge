"""Are real prints' opposite pairs along a ridge ink, or the skin? (D-073): ridge_breaks.py OUT.json

`ridge_frame`'s g and h on FVC DB1-3 B and SFinGe, for all minutiae, for `confirmed` ones, found again in at least 3 of a finger's other impressions, as `signed_false`, so a break in the ink doesn't count, and for those in quality 4 cells. For pairs at 15-30 px on the same ridge, the share that are ending and ending, fork and fork, or one of each, by sign.

Run: python -m signed_charge.exploratory.ridge_breaks OUT.json
LIMIT=n takes the first n fingers of each set.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import ridge_frame as rf
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 8))
LIMIT = int(os.environ.get("LIMIT", 0))
KEYS = ("all", "confirmed", "clean")


def kinds(p, heads, field, mask):
    """At 15-30 px on the same ridge, counts by the pair's types and sign: ending-ending, fork-fork, mixed, each opposite then same."""
    out = np.zeros(6)
    h, w = field.shape
    for i in range(len(p) - 1):
        for j in range(i + 1, len(p)):
            dx, dy = p[j, :2] - p[i, :2]
            d = np.hypot(dx, dy)
            if d >= 34:
                continue
            n = max(int(d / 3), 4)
            t = np.linspace(0, 1, n)
            x = np.clip(np.round(p[i, 0] + t * dx).astype(int), 0, w - 1)
            y = np.clip(np.round(p[i, 1] + t * dy).astype(int), 0, h - 1)
            if not mask[y, x].all():
                continue
            ridge = np.unwrap(np.angle(field[y, x])) / 2 + np.pi / 2
            m = ridge[n // 2]
            u, v = abs(dx * np.cos(m) + dy * np.sin(m)), abs(-dx * np.sin(m) + dy * np.cos(m))
            if not (15 <= u < 30 and v < 4.5):
                continue
            ends = np.stack([np.cos(ridge[[0, -1]]), np.sin(ridge[[0, -1]])], 1)
            same = np.sign(heads[i] @ ends[0]) * np.sign(heads[j] @ ends[1]) > 0
            kind = int(p[i, 3] > 0) + int(p[j, 3] > 0)
            out[kind * 2 + int(same)] += 1
    return out


def finger(task):
    name, label, paths = task
    ready = []
    for path in paths:
        im = fs.at_500(cv2.imread(path, 0), cf.PPI.get(name, 500))
        fg = fs.foreground(im)
        if fg is None:
            continue
        inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
        yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
        for x, y in sc.singular(fs.field(im), fg):
            inner &= np.hypot(xx - x, yy - y) >= sc.GUARD
        ready.append((inner, sm.direction(im), sf.quality(im), im))
    out = {"set": "sfinge" if "DB4" in name else "real", "finger": label}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        points = [sf.inside(extract(r[3]), r[0]) for r in ready]
        counts = sf.support(points, y_sign)
        sums = {k: np.zeros((3, len(rf.ALONG) - 1, len(rf.ACROSS) - 1)) for k in KEYS}
        types = {k: np.zeros(6) for k in KEYS}
        for p, c, (inner, field, q, _) in zip(points, counts, ready):
            if not len(p):
                continue
            keep = p[c >= sf.SUPPORT]
            clean = inner & (q == 4)
            for key, use, mask in (("all", p, inner), ("confirmed", keep, inner), ("clean", sf.inside(p, clean), clean)):
                if len(use) < 2:
                    continue
                heads = np.stack([np.cos(use[:, 2]), y_sign * np.sin(use[:, 2])], 1)
                sums[key][:2] += np.stack(rf.pairs(use[:, :2], heads, field, mask))
                ys, xs = np.nonzero(mask)
                pick = np.random.default_rng(len(use)).integers(len(xs), size=len(use))
                sums[key][2] += rf.pairs(np.c_[xs[pick], ys[pick]].astype(float), None, field, mask)[0]
                types[key] += kinds(use, heads, field, mask)
        out[reader] = {"sums": sums, "types": types}
    return out


def main(out):
    started = provenance.start()
    tasks = sf.fvc_fingers()
    real = [t for t in tasks if "DB4" not in t[0]][: LIMIT or None]
    sfinge = [t for t in tasks if "DB4" in t[0]][: LIMIT or None]
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(finger, real + sfinge, chunksize=1))
    rng = np.random.default_rng(0)
    report = {}
    for name in ("real", "sfinge"):
        group = [r for r in rows if r["set"] == name]
        report[name] = {"fingers": len(group)}
        for reader in sm.EXTRACTORS:
            report[name][reader] = {}
            for key in KEYS:
                table = np.stack([r[reader]["sums"][key] for r in group])
                stat = lambda idx: (lambda c, p, n: (c / np.maximum(n, 1), p / np.maximum(c, 1)))(*table[idx].sum(0))  # noqa: E731
                g, h = stat(np.arange(len(group)))
                draws = [stat(rng.integers(len(group), size=len(group))) for _ in range(rf.RESAMPLES)]
                report[name][reader][key] = {
                    "g": g.tolist(),
                    "h": h.tolist(),
                    "pairs": table[:, 0].sum(0).tolist(),
                    "h_ci": np.percentile([d[1] for d in draws], (2.5, 97.5), axis=0).tolist(),
                    "types": sum(r[reader]["types"][key] for r in group).tolist(),
                }
        print(name, report[name]["fingers"], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
