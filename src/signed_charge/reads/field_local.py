"""Does E-078's law of mismatch carry to a print? (D-050): field_local.py OUT.json RUN.npz...

Each planar print of `field_print`, read over the sheet inside `fs.foreground`'s inner part, with cells within `GUARD` px of a core or delta left out. θ is the angle between the flow the cue follows and the ridges the front would lay, from (d · ∇T / |∇T|)² = sin²θ. Off the seam, density by θ in `BINS`, by mindtct and LEADER. On the seam, k at least `SEAM` of `minutiae_seams`, density against off and h(15-45 px). CI by print.

Run: python -m signed_charge.reads.field_local OUT.json RUN.npz...
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as nd
from scipy import stats

from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_seams as seams
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 16))
RESAMPLES, SMOOTH = 2000, 2.0
BINS = {f"{a}-{a + 15}": (a, a + 15) for a in range(0, 90, 15)}


def kink(arrival):
    """`minutiae_seams.kink` on a flat sheet, which needs no flattening."""
    smooth = cv2.GaussianBlur(arrival.astype(np.float32), (0, 0), seams.BLUR)
    gx, gy = cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    size = np.hypot(gx, gy) + 1e-6
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), seams.SCALE)  # noqa: E731
    return 1 - np.hypot(blur(gx / size), blur(gy / size))


def mismatch(theta, arrival):
    gy, gx = np.gradient(nd.gaussian_filter(arrival, SMOOTH, mode="nearest"))
    norm = np.maximum(np.hypot(gx, gy), 1e-9)
    return np.rad2deg(np.arcsin(np.clip(np.abs(np.cos(theta) * gx + np.sin(theta) * gy) / norm, 0, 1)))


def one(path):
    d = np.load(path)
    u, arrival = d["u"], d["arrival"]
    im = np.where(u > np.median(u), 40, 235).astype(np.uint8)
    fg = fs.foreground(im)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    guarded = inner.copy()
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    for x, y in sc.singular(fs.field(im), fg):
        guarded &= np.hypot(xx - x, yy - y) >= sc.GUARD
    k = kink(arrival)
    on, off = guarded & (k >= seams.SEAM), guarded & (nd.maximum_filter(k, 25) < seams.SEAM)
    angle = mismatch(d["theta"], arrival)
    field = sm.direction(im)
    out = {"name": Path(path).stem, "on_cells": int(on.sum()), "off_cells": int(off.sum()), "inner": int(inner.sum())}
    edges = (sm.EDGES[:-1] >= sf.SHORT[0]) & (sm.EDGES[1:] <= sf.SHORT[1])
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        points = extract(im)
        seam_points = sf.inside(points, on)
        total, count, _ = sf.binned(seam_points, y_sign, field, inner)
        out[f"{reader}/seam"] = {"on": len(seam_points), "off": len(sf.inside(points, off)), "h": [float(total[edges].sum()), float(count[edges].sum())]}
        for name, (a, b) in BINS.items():
            mask = off & (angle >= a) & (angle <= b)
            out[f"{reader}/{name}"] = {"cells": int(mask.sum()), "minutiae": len(sf.inside(points, mask))}
    return out


def main(out, paths):
    started = provenance.start()
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(one, sorted(paths), chunksize=1))
    rng = np.random.default_rng(0)
    draws = [rng.integers(len(rows), size=len(rows)) for _ in range(RESAMPLES)]
    every = np.arange(len(rows))
    per = lambda n, c: (lambda i: n[i].sum() / max(c[i].sum() * ms.MM2, 1e-6) * 100)  # noqa: E731
    with_ci = lambda f: {"mean": float(f(every)), "ci": [float(np.percentile([f(i) for i in draws], q)) for q in (2.5, 97.5)]}  # noqa: E731
    report = {"prints": len(rows), "seam_share": float(sum(r["on_cells"] for r in rows) / sum(r["inner"] for r in rows))}
    for reader in sm.EXTRACTORS:
        entry = {}
        for name in BINS:
            cells = np.array([r[f"{reader}/{name}"]["cells"] for r in rows], float)
            minutiae = np.array([r[f"{reader}/{name}"]["minutiae"] for r in rows], float)
            entry[name] = {**with_ci(per(minutiae, cells)), "share_of_off": float(cells.sum() / max(sum(r["off_cells"] for r in rows), 1))}
        entry["spearman"] = float(stats.spearmanr(range(len(BINS)), [entry[n]["mean"] for n in BINS])[0])
        on_cells, off_cells = [np.array([r[k] for r in rows], float) for k in ("on_cells", "off_cells")]
        on, off = [np.array([r[f"{reader}/seam"][k] for r in rows], float) for k in ("on", "off")]
        h = np.array([r[f"{reader}/seam"]["h"] for r in rows])
        entry["seam_on"], entry["seam_off"] = with_ci(per(on, on_cells)), with_ci(per(off, off_cells))
        entry["on_over_off"] = with_ci(lambda i: per(on, on_cells)(i) / max(per(off, off_cells)(i), 1e-6))
        entry["h_on"] = {**with_ci(lambda i: h[i, 0].sum() / max(h[i, 1].sum(), 1)), "pairs": int(h[:, 1].sum())}
        report[reader] = entry
        print(reader, json.dumps(entry)[:1400], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
