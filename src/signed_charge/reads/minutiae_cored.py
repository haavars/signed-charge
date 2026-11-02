"""The classifier through real windows laid on the core (D-039, E-070): minutiae_cored.py CACHE.pkl OUT.json

As `minutiae_parts`, base `cored`. A print with a core takes a real window whose print has one too, moved so the two topmost cores coincide; a print with none takes a window from a real print with none, on the contact's centroid with `minutiae_parts`' jitter. Real prints stay as they are. Windows and transforms are redrawn each seed.
Also where each set's topmost core sits against its inner foreground's centroid, px.

Run: python -m signed_charge.reads.minutiae_cored CACHE.pkl OUT.json
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from signed_charge.reads import classifier_floor as cl
from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance

JOBS = int(os.environ.get("READ_JOBS", 8))
CASES = [("cored", p, "none") for p in parts.PARTS] + [("cored", "full", t) for t in parts.TRANSFORMS]
FOR_CORE, NONE = [], []


def topmost(task):
    """The topmost core (y, x), or None."""
    index, source = task
    cores = fs.cores(fs.field(place.image(source)), parts.mask_of(place.ROWS[index]))
    return min(((float(c[1]), float(c[0])) for c in cores), default=None)


def window_for(row, rng):
    if row["core"] is not None:
        window = FOR_CORE[rng.integers(len(FOR_CORE))]
        return window, (int(round(row["core"][0] - window["core"][0])), int(round(row["core"][1] - window["core"][1])))
    window = NONE[rng.integers(len(NONE))]
    py, px = parts.centroid(row["mask"])
    wy, wx = window["centroid"]
    return window, (int(round(py - wy + rng.uniform(-parts.JITTER, parts.JITTER))), int(round(px - wx + rng.uniform(-parts.JITTER, parts.JITTER))))


def through(row, reader, rng):
    window, (dy, dx) = window_for(row, rng)
    points = row["points"][reader]
    x, y = points[:, 0].astype(int) - dx, points[:, 1].astype(int) - dy
    h, w = window["mask"].shape
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    ok[ok] &= window["mask"][y[ok], x[ok]]
    return points[ok], parts.overlap(window["mask"], (dy, dx), row["mask"]) / window["mask"].sum()


def histograms(task):
    reader, transform, seed = task
    rng = np.random.default_rng([seed, parts.READERS.index(reader), 2, ("none", *parts.TRANSFORMS).index(transform)])
    out, coverage = {}, {}
    for name, rows in parts.ROWS.items():
        got, cover = [], []
        for row in rows:
            points = row["points"][reader]
            if name != "real":
                points, share = through(row, reader, rng)
                cover.append(share)
            got.append(((row["set"], row["finger"], 1), ms.histogram(parts.transformed(points, transform, rng)), len(points)))
        out[name] = got
        coverage[name] = {"on_print": float(np.mean(cover)) if cover else 1.0, "count": float(np.median([g[2] for g in got]))}
    return (reader, "cored", transform, seed), out, coverage


def main(cache, out):
    started = provenance.start()
    rows = pickle.load(open(cache, "rb"))["rows"]
    where = place.sources(rows)
    tasks = []
    for row in rows:
        if id(row) in where:
            place.ROWS.append(row)
            tasks.append((len(place.ROWS) - 1, where[id(row)]))
    with ProcessPoolExecutor(JOBS) as pool:
        cores = list(pool.map(topmost, tasks, chunksize=8))
    for (i, _), core in zip(tasks, cores):
        place.ROWS[i]["core"] = core
    report = {"core_offset": {}}
    for name in ("real", *parts.SETS):
        parts.ROWS[name] = [dict(r, mask=parts.mask_of(r)) for r in place.ROWS if parts.member(r, name)]
        offsets = np.array([np.subtract(r["core"], parts.centroid(r["mask"])) for r in parts.ROWS[name] if r["core"] is not None])
        report["core_offset"][name] = {
            "prints": len(parts.ROWS[name]),
            "with_core": len(offsets),
            "median_down": float(np.median(offsets[:, 0])),
            "median_abs_down": float(np.median(np.abs(offsets[:, 0]))),
            "median_abs_across": float(np.median(np.abs(offsets[:, 1]))),
        }
    for r in parts.ROWS["real"]:
        window = {"mask": r["mask"], "centroid": parts.centroid(r["mask"]), "core": r["core"]}
        (FOR_CORE if r["core"] is not None else NONE).append(window)
    print(report["core_offset"], len(FOR_CORE), len(NONE), flush=True)
    tasks = [(reader, transform, seed) for reader in parts.READERS for transform in ("none", *parts.TRANSFORMS) for seed in range(cl.SEEDS)]
    coverage = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for key, hists, cover in pool.map(histograms, tasks, chunksize=1):
            parts.HIST[key] = hists
            coverage[key] = cover
    jobs = [("fold", reader, name, case, seed) for reader in parts.READERS for case in CASES for name in parts.SETS for seed in range(cl.SEEDS)]
    jobs += [("draw", reader, "population", case, seed) for reader in parts.READERS for case in CASES for seed in range(parts.DRAWS)]
    results = {}
    with ProcessPoolExecutor(JOBS) as pool:  # after HIST, so the workers fork with it
        for task, value in pool.map(parts.score, jobs, chunksize=4):
            results.setdefault(task[:4], []).append(value)
    for reader in parts.READERS:
        report[reader] = {
            "cases": {},
            "coverage": {
                name: float(np.mean([coverage[(reader, "cored", "none", s)][name]["on_print"] for s in range(cl.SEEDS)])) for name in ("real", *parts.SETS)
            },
            "counts": {
                name: float(np.mean([coverage[(reader, "cored", "none", s)][name]["count"] for s in range(cl.SEEDS)])) for name in ("real", *parts.SETS)
            },
            "effects": {name: parts.effects(reader, "cored", name) for name in parts.SETS},
        }
        for case in CASES:
            label = "/".join(x for x in case if x != "none")
            entry = {name: cl.spread(results[("fold", reader, name, case)]) for name in parts.SETS}
            entry["population_ci"] = [float(np.percentile(results[("draw", reader, "population", case)], q)) for q in (2.5, 97.5)]
            report[reader]["cases"][label] = entry
            print(
                reader,
                label.ljust(20),
                {k: round(v["mean"], 3) for k, v in entry.items() if isinstance(v, dict)},
                [round(x, 3) for x in entry["population_ci"]],
                flush=True,
            )
        print(reader, report[reader]["coverage"], report[reader]["counts"], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
