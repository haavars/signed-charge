"""The population's seams by crossing angle (D-047): seams_by_angle.py OUT.json RUN.npz...

E-072's prints and seam, k at least `SEAM`. Each seam cell takes the angle at which two equal fronts give the greatest k within `REACH` px, α = 2 arccos(1 - k). As k tends to 1 at cores and deltas, cells within `GUARD` px of one are left out (D-048). Density on the seam, and h(15-45 px) among minutiae on it, in `BINS`, by mindtct and LEADER. CI by print.

Run: python -m signed_charge.exploratory.seams_by_angle OUT.json RUN.npz...
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as nd

from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_seams as seams
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 16))
REACH, RESAMPLES = 12, 2000
BINS = {"52-90": (52, 90), "90-120": (90, 120), "120-150": (120, 150), "over 150": (150, 181)}


def one(path):
    d = np.load(path)
    im = screen.print_of(d["u"])
    inner = cv2.erode(fs.foreground(im).astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    k = seams.kink(d["arrival"])
    peak = np.rad2deg(2 * np.arccos(np.clip(1 - nd.maximum_filter(k, 2 * REACH + 1), -1, 1)))
    guarded = inner.copy()
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    for x, y in sc.singular(fs.field(im), fs.foreground(im)):
        guarded &= np.hypot(xx - x, yy - y) >= sc.GUARD
    seam = (k >= seams.SEAM) & guarded
    field = sm.direction(im)
    out = {"name": Path(path).stem, "seam": int(seam.sum()), "inner": int(inner.sum())}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        points = extract(im)
        for name, (a, b) in BINS.items():
            mask = seam & (peak >= a) & (peak < b)
            p = sf.inside(points, mask)
            total, count, _ = sf.binned(p, y_sign, field, inner)
            edges = (sm.EDGES[:-1] >= sf.SHORT[0]) & (sm.EDGES[1:] <= sf.SHORT[1])
            out[f"{reader}/{name}"] = {"cells": int(mask.sum()), "minutiae": len(p), "h": [float(total[edges].sum()), float(count[edges].sum())]}
    return out


def main(out, paths):
    started = provenance.start()
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(one, sorted(paths), chunksize=1))
    rng = np.random.default_rng(0)
    report = {"prints": len(rows), "seam_share": float(sum(r["seam"] for r in rows) / sum(r["inner"] for r in rows))}
    for reader in sm.EXTRACTORS:
        for name in BINS:
            cells = np.array([r[f"{reader}/{name}"]["cells"] for r in rows], float)
            minutiae = np.array([r[f"{reader}/{name}"]["minutiae"] for r in rows], float)
            h = np.array([r[f"{reader}/{name}"]["h"] for r in rows])
            density = lambda i: minutiae[i].sum() / max(cells[i].sum() * ms.MM2, 1e-6) * 100  # noqa: E731
            charge = lambda i: h[i, 0].sum() / max(h[i, 1].sum(), 1)  # noqa: E731
            draws = [rng.integers(len(rows), size=len(rows)) for _ in range(RESAMPLES)]
            every = np.arange(len(rows))
            report[f"{reader}/{name}"] = {"share_of_seam": float(cells.sum() / max(sum(r["seam"] for r in rows), 1)), "density": density(every), "density_ci": [float(np.percentile([density(i) for i in draws], q)) for q in (2.5, 97.5)], "h": charge(every), "h_ci": [float(np.percentile([charge(i) for i in draws], q)) for q in (2.5, 97.5)], "pairs": int(h[:, 1].sum())}
            print(reader, name.ljust(10), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report[f"{reader}/{name}"].items() if "ci" not in k}, flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
