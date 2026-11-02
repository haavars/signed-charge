"""What does the minutiae classifier read? (D-035, E-067): minutiae_parts.py CACHE.pkl OUT.json

E-065's classifier, `conformality.accuracy` on the minutiae histogram, real FVC DB1-3 B against the population, SFinGe, model 2 and E-027's best, from `minutiae_cache.py`, on parts of what it sees. Its 10 × 10 bins are 20 px by 18°.

- Bases. `cut`: each print as it is. `windows`: each synthetic print seen through a real print's window: a real inner foreground drawn at random, its centroid laid on the print's with up to `JITTER` px of shift, keeping the first
  placing with at least `INSIDE` of it on the print, or the best of `TRIES`, and only the minutiae inside. Real prints stay as they are.
- Parts of the histogram. `distance` and `angle`, its margins; `near`, `middle`, `far`, the bands of `BANDS` alone, each renormalised, so what's left is how directions are spread at that range and not how much of the print is at it; `no_near` and so on, the rest without a band, renormalised.
- Transforms, on real and synthetic alike, the full histogram only. `flip`: each minutia's direction turned 180° with chance ½, which keeps the flow and takes away which way a minutia points, and so its charge. `shuffle`: directions dealt out at random among a print's minutiae, which keeps positions and the print's directions and takes away which direction goes where.

Each over 10 fold seeds, redrawing windows and transforms each seed. For the population, 30 draws of its prints with replacement as `population_read.scored`. Real against real for each, as its floor: 30 draws, 200 prints of four sensor sets against five.

Run: python -m signed_charge.reads.minutiae_parts CACHE.pkl OUT.json
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from signed_charge.reads import classifier_floor as cl
from signed_charge.measure import conformality as cf
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance

JOBS = int(os.environ.get("READ_JOBS", 8))
READERS = ("mindtct", "leader")
SETS = {"population": "population/", "sfinge": "FVC", "model2": "model2", "e027": "d_s1_sp2_f3"}
BANDS = {"near": (0, 2), "middle": (2, 6), "far": (6, 10)}  # rows: under 40, 40-120 and 120-200 px
PARTS = ("full", "distance", "angle", *BANDS, *(f"no_{b}" for b in BANDS))
TRANSFORMS = ("flip", "shuffle")
JITTER, INSIDE, TRIES, DRAWS, SIZE = 40, 0.9, 20, 30, 200
CASES = [(base, part, "none") for base in ("cut", "windows") for part in PARTS] + [(base, "full", t) for base in ("cut", "windows") for t in TRANSFORMS]
ROWS, MASKS, HIST = {}, [], {}


def member(row, name):
    if name == "sfinge":
        return "DB4" in row["set"]
    if name == "real":
        return row["set"].startswith("FVC") and "DB4" not in row["set"]
    return row["set"].startswith(SETS[name])


def mask_of(row):
    h, w = row["shape"]
    return np.unpackbits(row["inner"], count=h * w).reshape(h, w).astype(bool)


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return ys.mean(), xs.mean()


def overlap(window, place, print_mask):
    """How much of `window`, moved by `place`, is on `print_mask`."""
    dy, dx = place
    h, w = window.shape
    H, W = print_mask.shape
    y0, x0, y1, x1 = max(dy, 0), max(dx, 0), min(dy + h, H), min(dx + w, W)
    if y1 <= y0 or x1 <= x0:
        return 0
    return int((window[y0 - dy : y1 - dy, x0 - dx : x1 - dx] & print_mask[y0:y1, x0:x1]).sum())


def seen_through(points, print_mask, window, rng):
    """The minutiae inside a real window laid on a print, and how much of the window is on the print."""
    wy, wx = window["centroid"]
    py, px = window["print_centroid"](print_mask)
    area = window["mask"].sum()
    best, top = None, -1
    for _ in range(TRIES):
        place = (int(round(py - wy + rng.uniform(-JITTER, JITTER))), int(round(px - wx + rng.uniform(-JITTER, JITTER))))
        got = overlap(window["mask"], place, print_mask)
        if got > top:
            best, top = place, got
        if got >= INSIDE * area:
            break
    dy, dx = best
    x, y = points[:, 0].astype(int) - dx, points[:, 1].astype(int) - dy
    h, w = window["mask"].shape
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    ok[ok] &= window["mask"][y[ok], x[ok]]
    return points[ok], top / area


def transformed(points, transform, rng):
    if transform == "none" or not len(points):
        return points
    out = points.copy()
    if transform == "flip":
        out[:, 2] += np.pi * rng.integers(2, size=len(out))
    else:
        out[:, 2] = rng.permutation(out[:, 2])
    return out


def histograms(task):
    """One seed's histograms for one reader and case: {set: [(key, hist)]}, with the windows' coverage."""
    reader, base, transform, seed = task
    rng = np.random.default_rng([seed, READERS.index(reader), ("cut", "windows").index(base), ("none", *TRANSFORMS).index(transform)])
    out, coverage = {}, {}
    for name, rows in ROWS.items():
        got, cover = [], []
        for row in rows:
            points = row["points"][reader]
            if base == "windows" and name != "real":
                points, share = seen_through(points, row["mask"], MASKS[rng.integers(len(MASKS))], rng)
                cover.append(share)
            got.append(((row["set"], row["finger"], 1), ms.histogram(transformed(points, transform, rng)), len(points)))
        out[name] = got
        coverage[name] = {"mean_on_print": float(np.mean(cover)) if cover else 1.0, "median_count": float(np.median([g[2] for g in got]))}
    return (reader, base, transform, seed), out, coverage


def part_of(h, part):
    m = h.reshape(ms.MH_BINS, ms.MH_BINS)
    if part == "full":
        return h
    if part == "distance":
        return m.sum(1)
    if part == "angle":
        return m.sum(0)
    if part.startswith("no_"):
        a, b = BANDS[part[3:]]
        kept = np.vstack([m[:a], m[b:]]).ravel()
    else:
        a, b = BANDS[part]
        kept = m[a:b].ravel()
    return kept / max(kept.sum(), 1e-12)


def labelled(reader, base, transform, seed, name, part):
    return [(k, {"x": part_of(h, part)}) for k, h, _ in HIST[(reader, base, transform, seed)][name]]


def score(task):
    kind, reader, name, (base, part, transform), seed = task
    real = labelled(reader, base, transform, seed % cl.SEEDS, "real", part)
    rng = np.random.default_rng([seed, 7])
    if kind == "floor":
        sensors = sorted({k[0] for k, _ in real})
        a = {sensors[i] for i in rng.permutation(len(sensors))[:4]}
        left = [x for x in real if x[0][0] in a]
        right = [x for x in real if x[0][0] not in a]
        return task, cf.accuracy([left[i] for i in rng.permutation(len(left))[:SIZE]], [right[i] for i in rng.permutation(len(right))[:SIZE]], "x", seed=seed)
    mine = labelled(reader, base, transform, seed % cl.SEEDS, name, part)
    if kind == "draw":
        mine = [mine[i] for i in rng.integers(len(mine), size=len(mine))]
    return task, cf.accuracy(real, mine, "x", seed=seed)


def effects(reader, base, name):
    """Bin by bin, the mean histograms and the standardised difference, real less the set, at seed 0."""
    a = np.array([h for _, h, _ in HIST[(reader, base, "none", 0)]["real"]])
    b = np.array([h for _, h, _ in HIST[(reader, base, "none", 0)][name]])
    pooled = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-9
    return {"real": a.mean(0).tolist(), "set": b.mean(0).tolist(), "effect": ((a.mean(0) - b.mean(0)) / pooled).tolist()}


def main(cache, out):
    started = provenance.start()
    rows = pickle.load(open(cache, "rb"))["rows"]
    for name in ("real", *SETS):
        ROWS[name] = [dict(r, mask=mask_of(r)) for r in rows if member(r, name)]
    for r in ROWS["real"]:
        cy, cx = centroid(r["mask"])
        MASKS.append({"mask": r["mask"], "centroid": (cy, cx), "print_centroid": centroid})
    report = {
        "sizes": {
            name: {
                "prints": len(v),
                "median_area_mm2": float(np.median([r["mask"].sum() * ms.MM2 for r in v])),
                "median_counts": {reader: float(np.median([len(r["points"][reader]) for r in v])) for reader in READERS},
            }
            for name, v in ROWS.items()
        }
    }
    print(json.dumps(report["sizes"]), flush=True)
    tasks = [
        (reader, base, transform, seed) for reader in READERS for base in ("cut", "windows") for transform in ("none", *TRANSFORMS) for seed in range(cl.SEEDS)
    ]
    coverage = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for key, hists, cover in pool.map(histograms, tasks, chunksize=1):
            HIST[key] = hists
            coverage[key] = cover
    print("histograms", flush=True)
    jobs = []
    for reader in READERS:
        for case in CASES:
            for name in SETS:
                jobs += [("fold", reader, name, case, seed) for seed in range(cl.SEEDS)]
            jobs += [("draw", reader, "population", case, seed) for seed in range(DRAWS)]
            if case[0] == "cut":
                jobs += [("floor", reader, "real", case, seed) for seed in range(DRAWS)]
    print(len(jobs), "scores", flush=True)
    results = {}
    with ProcessPoolExecutor(JOBS) as pool:  # after HIST, so the workers fork with it
        for task, value in pool.map(score, jobs, chunksize=4):
            results.setdefault(task[:4], []).append(value)
    for reader in READERS:
        out_reader = report.setdefault(reader, {"cases": {}, "effects": {}, "coverage": {}})
        for case in CASES:
            label = "/".join(x for x in case if x != "none")
            entry = {name: cl.spread(results[("fold", reader, name, case)]) for name in SETS}
            entry["population_ci"] = [float(np.percentile(results[("draw", reader, "population", case)], q)) for q in (2.5, 97.5)]
            if case[0] == "cut":
                entry["floor"] = cl.spread(results[("floor", reader, "real", case)])
            out_reader["cases"][label] = entry
        for base in ("cut", "windows"):
            out_reader["coverage"][base] = {
                name: float(np.mean([coverage[(reader, base, "none", s)][name]["mean_on_print"] for s in range(cl.SEEDS)])) for name in ("real", *SETS)
            }
            out_reader["coverage"][f"{base}_counts"] = {
                name: float(np.mean([coverage[(reader, base, "none", s)][name]["median_count"] for s in range(cl.SEEDS)])) for name in ("real", *SETS)
            }
            out_reader["effects"][base] = {name: effects(reader, base, name) for name in SETS}
        for label, entry in out_reader["cases"].items():
            print(
                reader,
                label.ljust(22),
                {k: round(v["mean"], 3) for k, v in entry.items() if isinstance(v, dict)},
                [round(x, 3) for x in entry["population_ci"]],
                flush=True,
            )
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
