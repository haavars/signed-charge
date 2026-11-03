"""Exploratory, unregistered (E-094): minutiae both extractors find, per 100 mm², on the author's reading by eye, more forks and endings than real prints (O-011): density_consensus.py CACHE.pkl OUT.json RUN.npz...

A minutia counts if mindtct's and LEADER's are mutual nearest within `NEAR` px. Two regions, as a planar sheet's far parts are padded flow with few minutiae: `sheet` the inner foreground, and `window`, `WINDOWS` real windows laid on the core as E-070, each real print's through other real prints' windows too. RUN.npz are planar prints, named by the start of their stem. CI by finger.

Run: python -m signed_charge.exploratory.density_consensus CACHE.pkl OUT.json RUN.npz...
"""

import json
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from signed_charge.reads import agreement_scales as ass
from signed_charge.reads import minutiae_cored as cored
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import signed_false as sf

JOBS = int(os.environ.get("READ_JOBS", 3))
NEAR, RESAMPLES, MM2, WINDOWS = 10.0, 2000, (25.4 / 500) ** 2, 10
SETS = ("real", "sfinge", "model2", "population")


def counted(row, mask):
    a, b = sf.inside(row["points"]["mindtct"], mask), sf.inside(row["points"]["leader"], mask)
    both = 0
    if len(a) and len(b):
        ta, tb = cKDTree(a[:, :2]), cKDTree(b[:, :2])
        da, ia = tb.query(a[:, :2])
        _, ib = ta.query(b[:, :2])
        both = int(((da <= NEAR) & (ib[ia] == np.arange(len(a)))).sum())
    return float(mask.sum() * MM2), len(a), len(b), both


def one(row):
    row = dict(row, mask=parts.mask_of(row))
    sheet = counted(row, row["mask"])
    rng = np.random.default_rng(zlib.crc32(row["finger"].encode()))
    window = np.zeros(4)
    for _ in range(WINDOWS):
        chosen, shift = cored.window_for(row, rng)
        window += counted(row, place.placed(chosen["mask"], shift, row["mask"].shape) & row["mask"])
    return row["finger"], sheet, window


def summary(rows, rng):
    fingers = sorted({r[0] for r in rows})
    where = {f: i for i, f in enumerate(fingers)}
    draws = [rng.integers(len(fingers), size=len(fingers)) for _ in range(RESAMPLES)]
    every = np.arange(len(fingers))
    report = {"prints": len(rows), "fingers": len(fingers)}
    for region, part in (("sheet", 1), ("window", 2)):
        table = np.zeros((len(fingers), 4))
        for r in rows:
            table[where[r[0]]] += r[part]
        report[region] = {}
        for name, column in (("mindtct", 1), ("leader", 2), ("both", 3)):
            value = lambda i: 100 * table[i, column].sum() / max(table[i, 0].sum(), 1e-9)  # noqa: E731
            report[region][name] = {"mean": float(value(every)), "ci": [float(np.percentile([value(i) for i in draws], q)) for q in (2.5, 97.5)]}
    return report


def main(cache, out, paths):
    started = provenance.start()
    base = pickle.load(open(cache, "rb"))["rows"]
    with ProcessPoolExecutor(JOBS) as pool:
        grown = [r for r in pool.map(ass.planar, sorted(paths), chunksize=1) if r]
    rows = base + grown
    where = place.sources(rows)
    tasks = []
    for row in rows:
        if id(row) in where:
            place.ROWS.append(row)
            tasks.append((len(place.ROWS) - 1, where[id(row)]))
    store = Path(cache).with_name("cores.pkl")
    known = pickle.load(open(store, "rb")) if store.exists() else {}
    todo = [t for t in tasks if repr(t[1]) not in known]
    with ProcessPoolExecutor(JOBS) as pool:
        fresh = list(pool.map(cored.topmost, todo, chunksize=8))
    known.update(zip((repr(t[1]) for t in todo), fresh))
    named = {}
    for i, source in tasks:
        row = place.ROWS[i]
        row["core"] = known[repr(source)]
        name = row.get("arm") or next((n for n in SETS if parts.member(row, n)), None)
        if name:
            named.setdefault(name, []).append(row)
    for r in named["real"]:
        mask = parts.mask_of(r)
        (cored.FOR_CORE if r["core"] is not None else cored.NONE).append({"mask": mask, "centroid": parts.centroid(mask), "core": r["core"]})
    rng = np.random.default_rng(0)
    with ProcessPoolExecutor(JOBS) as pool:
        report = {name: summary(list(pool.map(one, rows, chunksize=16)), rng) for name, rows in named.items()}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
