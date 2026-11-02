"""Is the flow too parallel, or do minutiae avoid where it turns? (D-038, E-069): minutiae_place.py CACHE.pkl OUT.json

Through each print's window, as `minutiae_parts` lays it, with `signed_minutiae.direction`:

- `field`: |mean exp(2iθ)| over the window: 1 where the flow is parallel throughout, near 0 where it turns through every direction.
- `minutiae`: the same over the window's minutiae, from their directions, doubled so that a direction and its opposite count alike.
- `turning`: t = 1 - c, c the flow's coherence over a Gaussian of `SCALE` px, how much the flow turns locally. `seeking` is the mean t at the window's minutiae over the mean t across the window: above 1 minutiae lie where the flow turns more than most of the print, below 1 they avoid it.

`centred` lays the window on the contact's centroid with `minutiae_parts`' jitter. `cored` lays it so a real print's core falls on the print's, where each has exactly one core by `field_similarity.cores`: the check that `field` doesn't only say the window missed the core. Real prints stay in their own windows. Means by print, and CIs by finger.

Run: python -m signed_charge.reads.minutiae_place CACHE.pkl OUT.json
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
from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_check as fc
from signed_charge.measure import minutiae_cache as mc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge import paths

JOBS = int(os.environ.get("READ_JOBS", 8))
SCALE, RESAMPLES = 24.0, 2000
SETS = ("real", "population", "sfinge", "model2")
RUNS = paths.GENERATED / "population_d020"
ROWS, WINDOWS, CORED = [], [], []


def image(source):
    if source[0] == "fvc":
        return fs.at_500(cv2.imread(source[2], 0), cf.PPI.get(source[1], 500))
    if source[0] == "run":
        return screen.print_of(np.load(RUNS / f"{source[1]}.npz")["u"])
    return fc.image(source[1])[1]


def measures(points, unit, turning, mask):
    """`field`, `minutiae`, `seeking` over a mask on the print."""
    if mask.sum() < 400 or len(points) < 5:
        return None
    x, y = points[:, 0].astype(int), points[:, 1].astype(int)
    return {"field": float(np.abs(unit[mask].mean())), "minutiae": float(np.abs(np.exp(2j * points[:, 2]).mean())), "seeking": float(turning[y, x].mean() / max(turning[mask].mean(), 1e-6))}


def placed(window, place, shape):
    """A real window, moved by (dy, dx), as a mask on a print of this shape."""
    dy, dx = place
    h, w = window.shape
    H, W = shape
    out = np.zeros(shape, bool)
    y0, x0, y1, x1 = max(dy, 0), max(dx, 0), min(dy + h, H), min(dx + w, W)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = window[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


def core_of(task):
    """The core (y, x) of a real print with exactly one, or None."""
    index, source = task
    cores = fs.cores(fs.field(image(source)), parts.mask_of(ROWS[index]))
    return (float(cores[0][1]), float(cores[0][0])) if len(cores) == 1 else None


def centred(inner, rng):
    window = WINDOWS[rng.integers(len(WINDOWS))]
    wy, wx = window["centroid"]
    py, px = parts.centroid(inner)
    best, top = None, -1
    for _ in range(parts.TRIES):
        place = (int(round(py - wy + rng.uniform(-parts.JITTER, parts.JITTER))), int(round(px - wx + rng.uniform(-parts.JITTER, parts.JITTER))))
        got = parts.overlap(window["mask"], place, inner)
        if got > top:
            best, top = place, got
        if got >= parts.INSIDE * window["mask"].sum():
            break
    return placed(window["mask"], best, inner.shape) & inner


def read(task):
    index, kind, source = task
    row = ROWS[index]
    rng = np.random.default_rng(index)
    im = image(source)
    inner = parts.mask_of(row)
    z = sm.direction(im)
    unit = z / np.maximum(np.abs(z), 1e-6)
    blur = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), SCALE)  # noqa: E731
    turning = 1 - np.hypot(blur(unit.real), blur(unit.imag))
    cores = fs.cores(fs.field(im), inner)
    core = (float(cores[0][1]), float(cores[0][0])) if len(cores) == 1 else None
    if kind == "real":
        masks = {"centred": inner, "cored": inner if core else None}
    else:
        masks = {"centred": centred(inner, rng), "cored": None}
        if core:
            window = CORED[rng.integers(len(CORED))]
            masks["cored"] = placed(window["mask"], (int(round(core[0] - window["core"][0])), int(round(core[1] - window["core"][1]))), inner.shape) & inner
    out = {"set": row["set"], "finger": row["finger"], "kind": kind}
    for reader in parts.READERS:
        out[reader] = {base: measures(sf.inside(row["points"][reader], mask), unit, turning, mask) if mask is not None else None for base, mask in masks.items()}
    return out


def summary(rows, reader, base, key, rng):
    by = {}
    for r in rows:
        if r[reader][base]:
            by.setdefault(r["finger"], []).append(r[reader][base][key])
    means = np.array([np.mean(v) for v in by.values()])
    if not len(means):
        return None
    draws = [means[rng.integers(len(means), size=len(means))].mean() for _ in range(RESAMPLES)]
    return {"mean": float(means.mean()), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)], "fingers": len(means)}


def sources(rows):
    """Each cached row's image, as `minutiae_cache` read them: it left out prints with too little foreground."""
    fvc = iter(r for r in rows if r["set"].startswith("FVC"))
    captures = {}
    out = {}
    for key, im in cf.fvc(("DB1", "DB2", "DB3", "DB4")):
        if mc.inner_of(im) is None:
            continue
        row = next(fvc)
        assert (row["set"], row["finger"]) == (key[0], key[1])
        out[id(row)] = ("fvc", key[0], f"{cf.FVC}/{key[1]}_{key[2]}.tif")
    for row in rows:
        if row["set"].startswith("population/"):
            out[id(row)] = ("run", row["finger"])
        elif row["set"] in cl.VARIANTS:
            seed, finger = row["finger"][2:].split("/")
            label = (row["set"], row["finger"])
            captures[label] = captures.get(label, 0) + 1
            out[id(row)] = ("variant", (row["set"], int(seed), int(finger), captures[label]))
    return out


def main(cache, out):
    started = provenance.start()
    rows = pickle.load(open(cache, "rb"))["rows"]
    where = sources(rows)
    tasks = []
    for name in SETS:
        for row in rows:
            if parts.member(row, name):
                ROWS.append(row)
                tasks.append((len(ROWS) - 1, name, where[id(row)]))
    real = [(i, source) for i, name, source in tasks if name == "real"]
    with ProcessPoolExecutor(JOBS) as pool:
        cores = list(pool.map(core_of, real, chunksize=8))
    for (i, _), core in zip(real, cores):
        mask = parts.mask_of(ROWS[i])
        window = {"mask": mask, "centroid": parts.centroid(mask)}
        WINDOWS.append(window)
        if core:
            CORED.append({**window, "core": core})
    print("windows", len(WINDOWS), "with one core", len(CORED), flush=True)
    with ProcessPoolExecutor(JOBS) as pool:  # after WINDOWS, so workers fork with them
        results = list(pool.map(read, tasks, chunksize=8))
    rng = np.random.default_rng(0)
    report = {}
    keys = ("field", "minutiae", "seeking")
    for reader in parts.READERS:
        report[reader] = {name: {base: {k: summary([r for r in results if r["kind"] == name], reader, base, k, rng) for k in keys} for base in ("centred", "cored")} for name in SETS}
        report[reader]["by_pattern"] = {p: {base: {k: summary([r for r in results if r["set"] == f"population/{p}"], reader, base, k, rng) for k in keys} for base in ("centred", "cored")} for p in ("arch", "loop", "whorl")}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    for reader in parts.READERS:
        for name in SETS:
            print(reader, name.ljust(11), {b: {k: (round(v[k]["mean"], 3), [round(c, 3) for c in v[k]["ci"]]) if v[k] else None for k in v} for b, v in report[reader][name].items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
