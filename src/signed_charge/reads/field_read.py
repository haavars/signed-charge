"""Fronts on a given flow against the ridges drawn on it (D-049): field_read.py CACHE.pkl OUT.json SHEET.jpg RUN.npz...

Each planar print of `field_print`, and model 2's own print of the same finger, capture 1, seen through real windows laid on the core, as `minutiae_cored`. The classifier against FVC DB1-3 B, on the full histogram and its parts, over 10 fold seeds and 30 draws. Signed charge through the same windows, `DRAW` a finger, the same windows for both sets of a finger, CI by finger, and the paired difference. Density, pair shares and forks over the sheet as `population_read`.

Run: python -m signed_charge.reads.field_read CACHE.pkl OUT.json SHEET.jpg RUN.npz...
"""

import json
import os
import pickle
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.reads import classifier_floor as cl
from signed_charge.measure import flow_check as fc
from signed_charge.measure import minutiae_cache as mc
from signed_charge.reads import minutiae_cored as cored
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.reads import population_read as pr
from signed_charge.measure import provenance
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 16))
DRAW, RESAMPLES = 5, 2000
CASES = [("cored", p, "none") for p in ("full", "distance", "angle", "near", "middle", "far", "no_near")] + [("cored", "full", t) for t in parts.TRANSFORMS]
SHOWN = ("field", "same2", "population", "sfinge", "model2", "e027")


def sheet(path):
    u = np.load(path)["u"]
    return np.where(u > np.median(u), 40, 235).astype(np.uint8)


def image(source):
    if source[0] == "field":
        return sheet(source[1])
    return ORIGINAL_IMAGE(source)


def with_new(rows):
    out = ORIGINAL_SOURCES(rows)
    for r in rows:
        if r["set"] == "field":
            out[id(r)] = ("field", r["path"])
        elif r["set"] == "same2":
            seed, finger = r["finger"][2:].split("/")
            out[id(r)] = ("variant", ("model2", int(seed), int(finger), 1))
    return out


ORIGINAL_IMAGE, ORIGINAL_SOURCES = place.image, place.sources
place.image, place.sources = image, with_new


def new_row(path):
    seed, finger = re.search(r"field_(\d+)_(\d+)", Path(path).stem).groups()
    label = f"PH{seed}/{finger}"
    left = mc.row("field", label, sheet(path))
    right = mc.row("same2", label, fc.image(("model2", int(seed), int(finger), 1))[1])
    return dict(left, path=str(path)) if left else None, right


def charge(task):
    """One finger of one set: charge within `DRAW` real windows, the same windows drawn for both sets."""
    index, source = task
    row = dict(place.ROWS[index], mask=parts.mask_of(place.ROWS[index]))
    im = place.image(source)
    field = sm.direction(im)
    rng = np.random.default_rng(zlib.crc32(row["finger"].encode()))
    out = {"set": row["set"], "finger": row["finger"]}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        points = extract(im)
        total, count, n = np.zeros(len(sm.EDGES) - 1), np.zeros(len(sm.EDGES) - 1), 0
        for _ in range(DRAW):
            window, shift = cored.window_for(row, rng)
            mask = place.placed(window["mask"], shift, im.shape) & row["mask"]
            got = sf.binned(sf.inside(points, mask), y_sign, field, mask)
            total, count, n = total + got[0], count + got[1], n + got[2]
        out[reader] = (total, count, n)
    return out


def summary(rows, reader, rng):
    total = np.array([r[reader][0] for r in rows])
    count = np.array([r[reader][1] for r in rows])
    n = np.array([r[reader][2] for r in rows], float)
    band = lambda a, b: (sm.EDGES[:-1] >= a) & (sm.EDGES[1:] <= b)  # noqa: E731
    measures = {
        "h_short": lambda i: total[i][:, band(*sf.SHORT)].sum() / max(count[i][:, band(*sf.SHORT)].sum(), 1),
        "h_long": lambda i: total[i][:, band(*sf.LONG)].sum() / max(count[i][:, band(*sf.LONG)].sum(), 1),
        "c60": lambda i: 2 * total[i][:, sm.EDGES[1:] <= 60].sum() / max(n[i].sum(), 1),
        "c150": lambda i: 2 * total[i].sum() / max(n[i].sum(), 1),
    }
    draws = [rng.integers(len(rows), size=len(rows)) for _ in range(RESAMPLES)]
    every = np.arange(len(rows))
    return (
        {
            "fingers": len(rows),
            "minutiae_a_window": float(n.mean() / DRAW),
            **{k: {"mean": float(f(every)), "ci": [float(np.percentile([f(d) for d in draws], q)) for q in (2.5, 97.5)]} for k, f in measures.items()},
        },
        n,
        total,
    )


def main(cache, out, figure, paths):
    started = provenance.start()
    base = pickle.load(open(cache, "rb"))["rows"]
    with ProcessPoolExecutor(JOBS) as pool:
        added = list(pool.map(new_row, sorted(paths), chunksize=1))
        fresh = [r for pair in added for r in pair if r]
        rows = base + fresh
        sheets = list(pool.map(pr.read_image, [(("field", Path(p).stem, 1), sheet(p)) for p in sorted(paths)], chunksize=1))
    parts.SETS.update({"field": "field", "same2": "same2"})
    where = place.sources(rows)
    tasks = []
    for row in rows:
        if id(row) in where:
            place.ROWS.append(row)
            tasks.append((len(place.ROWS) - 1, where[id(row)]))
    store = Path(cache).with_name("cores.pkl")  # the slow step; a print's core doesn't change between reads
    known = pickle.load(open(store, "rb")) if store.exists() else {}
    todo = [t for t in tasks if repr(t[1]) not in known]
    with ProcessPoolExecutor(JOBS) as pool:
        known.update(zip((repr(t[1]) for t in todo), pool.map(cored.topmost, todo, chunksize=8)))
    pickle.dump(known, open(store, "wb"))
    for i, source in tasks:
        place.ROWS[i]["core"] = known[repr(source)]
    report = {"core_offset": {}}
    for name in ("real", *parts.SETS):
        parts.ROWS[name] = [dict(r, mask=parts.mask_of(r)) for r in place.ROWS if parts.member(r, name)]
        report["core_offset"][name] = {"prints": len(parts.ROWS[name]), "with_core": sum(r["core"] is not None for r in parts.ROWS[name])}
    for r in parts.ROWS["real"]:
        (cored.FOR_CORE if r["core"] is not None else cored.NONE).append({"mask": r["mask"], "centroid": parts.centroid(r["mask"]), "core": r["core"]})
    print(report["core_offset"], flush=True)
    with ProcessPoolExecutor(JOBS) as pool:
        charged = list(pool.map(charge, [t for t in tasks if place.ROWS[t[0]]["set"] in ("field", "same2")], chunksize=1))
    rng = np.random.default_rng(0)
    for reader in sm.EXTRACTORS:
        report.setdefault(reader, {})
        field = sorted((r for r in charged if r["set"] == "field"), key=lambda r: r["finger"])
        same = {r["finger"]: r for r in charged if r["set"] == "same2"}
        field = [r for r in field if r["finger"] in same]
        twin = [same[r["finger"]] for r in field]
        report[reader]["charge"] = {}
        report[reader]["charge"]["field"], n_a, t_a = summary(field, reader, rng)
        report[reader]["charge"]["same2"], n_b, t_b = summary(twin, reader, rng)
        reach = sm.EDGES[1:] <= 60
        difference = lambda i: 2 * t_a[i][:, reach].sum() / max(n_a[i].sum(), 1) - 2 * t_b[i][:, reach].sum() / max(n_b[i].sum(), 1)  # noqa: E731
        draws = [difference(rng.integers(len(field), size=len(field))) for _ in range(RESAMPLES)]
        report[reader]["charge"]["c60_field_less_same2"] = {
            "mean": float(difference(np.arange(len(field)))),
            "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)],
        }
        print(reader, json.dumps(report[reader]["charge"])[:600], flush=True)
        report[reader]["sheet"] = pr.described(sheets, reader, rng)
        report[reader]["sheet"].pop("h", None)
        report[reader]["sheet"].pop("pairs_binned", None)
    tasks = [(reader, transform, seed) for reader in parts.READERS for transform in ("none", *parts.TRANSFORMS) for seed in range(cl.SEEDS)]
    with ProcessPoolExecutor(JOBS) as pool:
        for key, hists, _ in pool.map(cored.histograms, tasks, chunksize=1):
            parts.HIST[key] = hists
    jobs = [("fold", reader, name, case, seed) for reader in parts.READERS for case in CASES for name in parts.SETS for seed in range(cl.SEEDS)]
    jobs += [("draw", reader, name, case, seed) for reader in parts.READERS for case in CASES for name in ("field", "same2") for seed in range(parts.DRAWS)]
    results = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for task, value in pool.map(parts.score, jobs, chunksize=4):
            results.setdefault(task[:4], []).append(value)
    for reader in parts.READERS:
        report[reader]["cases"] = {}
        for case in CASES:
            label = "/".join(x for x in case if x not in ("none", "cored"))
            entry = {name: cl.spread(results[("fold", reader, name, case)]) for name in parts.SETS}
            entry["ci"] = {name: [float(np.percentile(results[("draw", reader, name, case)], q)) for q in (2.5, 97.5)] for name in ("field", "same2")}
            report[reader]["cases"][label] = entry
            print(reader, label.ljust(14), {k: round(entry[k]["mean"], 3) for k in SHOWN}, entry["ci"], flush=True)
    tiles = [cv2.resize(sheet(p), (160, 160), interpolation=cv2.INTER_AREA) for p in sorted(paths)[:48]]
    if len(tiles) >= 8:
        cv2.imwrite(figure, np.vstack([np.hstack(tiles[i : i + 12]) for i in range(0, len(tiles) // 12 * 12, 12)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
