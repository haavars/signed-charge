"""Look 1 of docs/held-out-protocol.md: real's own signed band (E-060): held_out_signed.py OUT.json [IMAGES]

C(60) on both extractors, per sensor and pooled, CI by finger, on FVC DB1-3 B and, with IMAGES, on the checking fingers of DB1-3 A and SFinGe's DB4 A. IMAGES holds <year>/<db>_A/<finger>_<impression>.tif, and any finger outside
`held_out_checking.json` stops the run. The verdict applies D-031's three criteria.

Run: python -m signed_charge.reads.held_out_signed OUT.json [IMAGES]
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import provenance
from signed_charge.measure import signed_minutiae as sm
from signed_charge import paths

JOBS = int(os.environ.get("HELD_JOBS", 4))
RESAMPLES = 2000
BANDS = {"mindtct": (0.03, 1.40), "leader": (0.25, 1.51)}  # E-040, by sensor, on B
SENSORS_IN_BAND = 8
ALLOWED = json.load(open(paths.RESULTS / "held_out_checking.json"))


def load(item):
    path, name, finger, impression = item
    return (name, f"{name}/{finger}", impression), fs.at_500(cv2.imread(path, 0), cf.PPI.get(name.replace("_A", "_B"), 500))


def measured(task):
    return sm.measure(load(task))


def tasks_a(root, dbs):
    out = []
    for name, fingers in ALLOWED.items():
        if name.split("/")[1][:3] not in dbs:
            continue
        for path in sorted(glob.glob(f"{root}/{name}/*.tif")):
            finger, impression = Path(path).stem.split("_")
            if int(finger) not in fingers:
                raise SystemExit(f"{path} is not a checking finger")
            out.append((path, name, finger, int(impression)))
    return out


def tasks_b(dbs):
    out = []
    for year in ("FVC2000", "FVC2002", "FVC2004"):
        for db in dbs:
            name = f"{year}/{db}_B"
            for path in sorted(glob.glob(f"{cf.FVC}/{name}/*.tif")):
                finger, impression = Path(path).stem.split("_")
                out.append((path, name, finger, int(impression)))
    return out


def charge(rows, extractor, rng):
    """C(60) and its CI by finger."""
    near = sm.EDGES[1:] <= 60
    by = {}
    for (key, value) in rows:
        if value and value[extractor]:
            s = by.setdefault(key[1], [0.0, 0.0])
            s[0] += 2 * np.asarray(value[extractor]["sum"])[near].sum()
            s[1] += value[extractor]["minutiae"]
    a = np.array(list(by.values()))
    draws = [a[d, 0].sum() / a[d, 1].sum() for d in (rng.integers(len(a), size=len(a)) for _ in range(RESAMPLES))]
    return {"mean": float(a[:, 0].sum() / a[:, 1].sum()), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)], "fingers": len(a), "prints": sum(1 for _, v in rows if v and v[extractor])}


def summary(rows, rng):
    sensors = sorted({k[0] for k, _ in rows})
    return {name: {"pooled": charge(rows, name, rng), "sensors": {s: charge([r for r in rows if r[0][0] == s], name, rng) for s in sensors}} for name in sm.EXTRACTORS}


def verdict(a, b, sfinge):
    out = {}
    for name, (low, high) in BANDS.items():
        pa, pb, ps = a[name]["pooled"], b[name]["pooled"], sfinge[name]["pooled"]
        inside = sum(low <= v["mean"] <= high for v in a[name]["sensors"].values())
        out[name] = {"overlaps_b": bool(pa["ci"][0] <= pb["ci"][1] and pb["ci"][0] <= pa["ci"][1]), "sensors_in_band": inside, "sfinge_above": bool(ps["ci"][0] > pa["ci"][1])}
        out[name]["holds"] = bool(out[name]["overlaps_b"] and inside >= SENSORS_IN_BAND and out[name]["sfinge_above"])
    out["holds"] = all(out[n]["holds"] for n in BANDS)
    return out


def main(out, root=None):
    started = provenance.start()
    rng = np.random.default_rng(0)
    real = ("DB1", "DB2", "DB3")
    with ProcessPoolExecutor(JOBS) as pool:
        report = {"b": summary(list(pool.map(measured, tasks_b(real), chunksize=4)), rng)}
        report["b_sfinge"] = summary(list(pool.map(measured, tasks_b(("DB4",)), chunksize=4)), rng)
        if root:
            report["a"] = summary(list(pool.map(measured, tasks_a(root, real), chunksize=4)), rng)
            report["a_sfinge"] = summary(list(pool.map(measured, tasks_a(root, ("DB4",)), chunksize=4)), rng)
            report["verdict"] = verdict(report["a"], report["b"], report["a_sfinge"])
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    for key in report:
        if key not in ("_provenance", "verdict"):
            print(key, {n: (round(v["pooled"]["mean"], 2), [round(x, 2) for x in v["pooled"]["ci"]], v["pooled"]["fingers"], v["pooled"]["prints"]) for n, v in report[key].items()}, flush=True)
    if "verdict" in report:
        print(json.dumps(report["verdict"]), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
