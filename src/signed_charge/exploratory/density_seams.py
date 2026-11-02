"""Does density against speed survive off the seams? (D-044, E-075): density_seams.py OUT.json RUN.npz...

The saved sweeps, D-010's and D-015's, with `minutiae_seams`' seams: for each run the minutiae count on and off them, a skeleton's as `defects_direct.count`, and mindtct's and LEADER's, per 100 mm² of print. Then density against front speed, within each noise level, slope by least squares with a CI by resampling runs, for arches, and for loops and whorls, which carry E-051's quarter-turn pad and are given only for comparison.

Run: python -m signed_charge.exploratory.density_seams OUT.json RUN.npz...
"""
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as nd
from skimage.morphology import skeletonize

from signed_charge.measure import defects_direct as dd
from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_seams as seams
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 16))
RESAMPLES = 2000


def skeleton_points(im):
    """Endings and forks by the skeleton, leaving out islands, as (x, y) rows."""
    skeleton = dd.prune(skeletonize(im == 40))
    number = dd.crossing(skeleton)
    labels, n = nd.label(skeleton, np.ones((3, 3)))
    sizes = nd.sum(skeleton, labels, range(1, n + 1))
    forks = nd.sum(number >= 3, labels, range(1, n + 1))
    small = np.isin(labels, np.array([i + 1 for i in range(n) if forks[i] == 0 and sizes[i] < 30]))
    ys, xs = np.nonzero(((number == 1) & ~small) | (number >= 3))
    return np.stack([xs, ys, np.zeros(len(xs)), np.zeros(len(xs))], 1).astype(float)


def one(path):
    name = Path(path).stem
    match = re.match(r"(\w+?)_a([\d.]+)_s", name)
    d = np.load(path)
    im = screen.print_of(d["u"])
    inner = cv2.erode(fs.foreground(im).astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    on = (seams.kink(d["arrival"]) >= seams.SEAM) & inner
    off = inner & ~on
    out = {"pattern": match.group(1), "speed": float(match.group(2)), "noise": 0.01 if "sweep_d010" in str(path) or "noise0.01" in str(path) else 0.003, "area_on": float(on.sum() / max(inner.sum(), 1))}
    readers = {"direct": skeleton_points, **{k: v[0] for k, v in sm.EXTRACTORS.items()}}
    for reader, extract in readers.items():
        points = extract(im)
        out[reader] = {key: len(sf.inside(points, mask)) / max(inner.sum() * ms.MM2, 1e-6) * 100 for key, mask in (("on", on), ("off", off))}
        out[reader]["off_density"] = out[reader]["off"] / max(1 - out["area_on"], 1e-6)  # per 100 mm² of the print off seams, not of the whole print
    return out


def slope(rows, reader, key, rng):
    """Density against speed within noise levels: each level's mean taken off."""
    x, y = np.array([r["speed"] for r in rows]), np.array([r[reader][key] for r in rows])
    noise = np.array([r["noise"] for r in rows])
    for level in set(noise):
        x[noise == level] -= x[noise == level].mean()
        y[noise == level] -= y[noise == level].mean()
    fit = lambda i: float((x[i] * y[i]).sum() / max((x[i] ** 2).sum(), 1e-9))  # noqa: E731
    draws = [fit(rng.integers(len(x), size=len(x))) for _ in range(RESAMPLES)]
    return {"per_unit_speed": fit(np.arange(len(x))), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}


def main(out, paths):
    started = provenance.start()
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(one, sorted(paths), chunksize=1))
    rng = np.random.default_rng(0)
    report = {}
    for pattern in ("arch", "loop", "whorl"):
        group = [r for r in rows if r["pattern"] == pattern]
        report[pattern] = {"runs": len(group)}
        for reader in ("direct", *sm.EXTRACTORS):
            entry = {key: slope(group, reader, key, rng) for key in ("on", "off", "off_density")}
            entry["by_speed"] = {}
            for noise in sorted({r["noise"] for r in group}):
                for speed in sorted({r["speed"] for r in group}):
                    picked = [r for r in group if r["noise"] == noise and r["speed"] == speed]
                    if picked:
                        entry["by_speed"][f"{noise}/{speed}"] = {k: float(np.mean([r[reader][k] for r in picked])) for k in ("on", "off", "off_density")} | {"runs": len(picked), "area_on": float(np.mean([r["area_on"] for r in picked]))}
            report[pattern][reader] = entry
            print(pattern, reader.ljust(8), {k: (round(entry[k]["per_unit_speed"], 1), [round(c, 1) for c in entry[k]["ci"]]) for k in ("on", "off", "off_density")}, flush=True)
            print("   ", {k: (round(v["on"], 1), round(v["off"], 1), round(v["off_density"], 1)) for k, v in entry["by_speed"].items()}, flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
