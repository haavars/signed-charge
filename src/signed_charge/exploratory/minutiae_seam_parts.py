"""What's left without the seams, and do real prints clump so? (D-042, E-073): minutiae_seam_parts.py CACHE.pkl OUT.json

The classifier of `minutiae_cored`, through windows on the core, real FVC B against the population in three forms: `population`; `offseam`, without the minutiae on `minutiae_seams`' seams; and `thinned`, the control, without as many of its minutiae drawn at random, redrawn each seed.
Clumping, which needs no seams and so can be read on real: the share of a print's minutiae in the densest 19 % of its area, the seams' share, density being its minutiae under a Gaussian of `KERNEL` px over its inner foreground. Real as is, the rest as cut; and the population through windows on the core, the frame to compare with real. CIs by finger.

Run: python -m signed_charge.exploratory.minutiae_seam_parts CACHE.pkl OUT.json
"""
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.reads import classifier_floor as cl
from signed_charge.reads import minutiae_cored as mo
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.reads import minutiae_seams as seams
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_false as sf

JOBS = int(os.environ.get("READ_JOBS", 8))
EQUAL = float(os.environ.get("REAL_KEEP", 1.0))  # D-043: the share of each real print's minutiae kept at random, so that a count alone tells nothing
KERNEL, SHARE, RESAMPLES = 20.0, 0.19, 2000
CASES = [("cored", "full", "none"), ("cored", "near", "none"), ("cored", "far", "none"), ("cored", "no_near", "none")]
NAMES = ("population", "offseam", "thinned")


def on_seam(name):
    return seams.kink(np.load(place.RUNS / f"{name}.npz")["arrival"]) >= seams.SEAM


def clumped(points, mask):
    """The share of minutiae in the densest `SHARE` of the mask."""
    if len(points) < 5:
        return None
    dots = np.zeros(mask.shape, np.float32)
    np.add.at(dots, (points[:, 1].astype(int), points[:, 0].astype(int)), 1)
    density = cv2.GaussianBlur(dots, (0, 0), KERNEL)
    top = density >= np.quantile(density[mask], 1 - SHARE)
    return float(top[points[:, 1].astype(int), points[:, 0].astype(int)].mean())


def histograms(task):
    reader, seed = task
    rng = np.random.default_rng([seed, parts.READERS.index(reader), 3])
    out = {}
    for name, rows in parts.ROWS.items():
        got = []
        for row in rows:
            points = row["points"][reader]
            if name == "real" and EQUAL < 1:
                points = points[rng.uniform(size=len(points)) < EQUAL]
            if name != "real":
                points, _ = mo.through(row, reader, rng)
                if name != "population" and len(points):
                    x, y = points[:, 0].astype(int), points[:, 1].astype(int)
                    seam = row["seam"][y, x]
                    keep = ~seam if name == "offseam" else rng.permutation(len(points)) >= seam.sum()
                    points = points[keep]
            got.append(((row["set"], row["finger"], 1), ms.histogram(points), len(points)))
        out[name] = got
    return (reader, "cored", "none", seed), out


def clump_row(task):
    reader, name, index = task
    row = CLUMP[name][index]
    rng = np.random.default_rng(index)
    value = clumped(row["points"][reader], row["mask"])
    if name == "population":
        window, (dy, dx) = mo.window_for(row, rng)
        mask = place.placed(window["mask"], (dy, dx), row["mask"].shape) & row["mask"]
        return reader, name, row["finger"], value, clumped(sf.inside(row["points"][reader], mask), mask)
    return reader, name, row["finger"], value, None


CLUMP = {}


def interval(by, rng):
    means = np.array([np.mean(v) for v in by.values() if v])
    draws = [means[rng.integers(len(means), size=len(means))].mean() for _ in range(RESAMPLES)]
    return {"mean": float(means.mean()), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)], "fingers": len(means)}


def main(cache, out):
    started = provenance.start()
    rows = pickle.load(open(cache, "rb"))["rows"]
    where = place.sources(rows)
    tasks = []
    for row in rows:
        if id(row) in where and (parts.member(row, "real") or parts.member(row, "population")):
            place.ROWS.append(row)
            tasks.append((len(place.ROWS) - 1, where[id(row)]))
    with ProcessPoolExecutor(JOBS) as pool:
        cores = list(pool.map(mo.topmost, tasks, chunksize=8))
    for (i, _), core in zip(tasks, cores):
        place.ROWS[i]["core"] = core
    real = [dict(r, mask=parts.mask_of(r)) for r in place.ROWS if parts.member(r, "real")]
    population = [dict(r, mask=parts.mask_of(r), seam=on_seam(r["finger"])) for r in place.ROWS if parts.member(r, "population")]
    parts.ROWS.clear()
    parts.ROWS.update({"real": real, **{n: population for n in NAMES}})
    for r in real:
        (mo.FOR_CORE if r["core"] is not None else mo.NONE).append({"mask": r["mask"], "centroid": parts.centroid(r["mask"]), "core": r["core"]})
    with ProcessPoolExecutor(JOBS) as pool:
        for key, hists in pool.map(histograms, [(reader, seed) for reader in parts.READERS for seed in range(cl.SEEDS)], chunksize=1):
            parts.HIST[key] = hists
    jobs = [("fold", reader, name, case, seed) for reader in parts.READERS for case in CASES for name in NAMES for seed in range(cl.SEEDS)]
    jobs += [("draw", reader, name, case, seed) for reader in parts.READERS for case in CASES[:1] for name in NAMES for seed in range(parts.DRAWS)]
    results = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for task, value in pool.map(parts.score, jobs, chunksize=4):
            results.setdefault(task[:4], []).append(value)
    report = {}
    for reader in parts.READERS:
        report[reader] = {case[1]: {name: cl.spread(results[("fold", reader, name, case)]) for name in NAMES} for case in CASES}
        report[reader]["full_ci"] = {name: [float(np.percentile(results[("draw", reader, name, CASES[0])], q)) for q in (2.5, 97.5)] for name in NAMES}
        report[reader]["counts"] = {name: float(np.mean([np.median([g[2] for g in parts.HIST[(reader, "cored", "none", s)][name]]) for s in range(cl.SEEDS)])) for name in ("real", *NAMES)}
        for key, entry in report[reader].items():
            print(reader, key.ljust(8), {k: round(v["mean"], 3) if isinstance(v, dict) else v for k, v in entry.items()}, flush=True)
    if EQUAL < 1:
        report["_provenance"] = provenance.record(started)
        json.dump(report, open(out, "w"), indent=1)
        return
    CLUMP.update({"real": real, "population": population})
    for name, label in (("sfinge", "FVC"), ("model2", "model2"), ("e027", "d_s1_sp2_f3")):
        CLUMP[name] = [dict(r, mask=parts.mask_of(r)) for r in rows if parts.member(r, name)]
    jobs = [(reader, name, i) for reader in parts.READERS for name, v in CLUMP.items() for i in range(len(v))]
    by = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for reader, name, finger, value, windowed in pool.map(clump_row, jobs, chunksize=32):
            if value is not None:
                by.setdefault((reader, name, "cut"), {}).setdefault(finger, []).append(value)
            if windowed is not None:
                by.setdefault((reader, name, "cored"), {}).setdefault(finger, []).append(windowed)
    rng = np.random.default_rng(0)
    report["clumping"] = {f"{reader}/{name}/{frame}": interval(v, rng) for (reader, name, frame), v in by.items()}
    for key, value in report["clumping"].items():
        print(key.ljust(30), round(value["mean"], 3), [round(c, 3) for c in value["ci"]], value["fingers"], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
