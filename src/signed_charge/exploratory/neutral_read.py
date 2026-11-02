"""Signed charge of D-065's pilot cells (D-065): neutral_read.py CACHE.pkl OUT.json RUN.npz...

`flow_read` without its classifier, for any cells, each named by the start of its prints' stems: charge through real windows laid on the core, CI by finger.

Run: python -m signed_charge.exploratory.neutral_read CACHE.pkl OUT.json RUN.npz...
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from signed_charge.reads import field_read as fr
from signed_charge.measure import minutiae_cache as mc
from signed_charge.reads import minutiae_cored as cored
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 16))


def new_row(path):
    stem = Path(path).stem
    row = mc.row(stem.split("_")[0], stem, fr.sheet(path))
    return dict(row, path=str(path)) if row else None


def main(cache, out, paths):
    started = provenance.start()
    base = pickle.load(open(cache, "rb"))["rows"]
    with ProcessPoolExecutor(JOBS) as pool:
        fresh = [r for r in pool.map(new_row, sorted(paths), chunksize=1) if r]
    rows = base + fresh
    where = place.sources(rows)
    for r in fresh:
        where[id(r)] = ("field", r["path"])
    ARMS = sorted({r["set"] for r in fresh})
    parts.SETS.update({arm: arm for arm in ARMS})
    tasks = []
    for row in rows:
        if id(row) in where:
            place.ROWS.append(row)
            tasks.append((len(place.ROWS) - 1, where[id(row)]))
    store = Path(cache).with_name("cores.pkl")
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
        charged = list(pool.map(fr.charge, [t for t in tasks if place.ROWS[t[0]]["set"] in ARMS], chunksize=1))
    rng = np.random.default_rng(0)
    for reader in sm.EXTRACTORS:
        report[reader] = {"charge": {}}
        for arm in ARMS:
            report[reader]["charge"][arm] = fr.summary(sorted((r for r in charged if r["set"] == arm), key=lambda r: r["finger"]), reader, rng)[0]
        print(reader, json.dumps(report[reader]["charge"])[:900], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
