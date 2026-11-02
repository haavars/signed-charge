"""The low-noise sweep read against D-015's criteria (E-043): sweep_low_read.py OUT.json SHEET.jpg RUN.npz...

Every print on the D-009 cut, as `sweep_read.py`, with `defects_direct.count` and `signed_minutiae`. The sensor bands are E-039's, from `sweep_d010.json`, and C(60)'s are E-040's, so
the test can't move. A criterion holds at a pair of adjacent speeds if it holds at both; D-015 asks for one pair at which criteria 2-5 all hold.

Run: python -m signed_charge.exploratory.sweep_low_read OUT.json SHEET.jpg RUN.npz...
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from signed_charge.measure import defects_direct as dd
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.measure import signed_minutiae as sm
from signed_charge.exploratory import sweep_read as sr
from signed_charge import paths

JOBS = int(os.environ.get("READ_JOBS", 8))
SIGNED_BANDS = {"mindtct": [0.03, 1.40], "leader": [0.25, 1.51]}  # E-040, C(60) by sensor
ISLANDS = 2


def one(path):
    row = sr.one(path)
    row["direct"] = dd.count(screen.print_of(np.load(path)["u"]))
    row["signed"] = sm.developmental(path)[1]
    return row


def read(rows, bands, rng):
    report = {"excluded": [r["name"] for r in rows if r["unreached"] > 0]}
    for pattern in sorted({r["pattern"] for r in rows}):
        everything = [r for r in rows if r["pattern"] == pattern]
        mine = [r for r in everything if r["unreached"] == 0]
        speeds = sorted({r["speed"] for r in mine})
        cells = {}
        for s in speeds:
            cell = [r for r in mine if r["speed"] == s]
            c = {"prints": len(cell), "order": sr.agreement(cell), "period_px": float(np.median([r["period_px"] for r in cell])),
                 "islands": float(np.mean([r["direct"]["islands"] for r in cell])), "direct": float(np.median([r["direct"]["density"] for r in cell])),
                 "direct_no_islands": float(np.median([r["direct"]["density_no_islands"] for r in cell]))}
            for name in sr.EXTRACTORS:
                xs = [r["extractors"][name] for r in cell if r["extractors"][name]]
                signed = sm.curve([r["signed"] for r in cell if r["signed"]], name, rng)
                c[name] = {**sr.summary(xs), "ci": sr.interval(xs, rng), "forks": float(np.mean([x["forks"] for x in xs if x["forks"] is not None])),
                           "c60": signed["screening"]["60"], "h": signed["h"], "pairs_binned": signed["pairs"]}
            cells[s] = c
        ok = {s: {"islands": cells[s]["islands"] <= ISLANDS,
                  "density": all(sr.overlaps(cells[s][n]["ci"]["density"], bands[n]["density"]) for n in sr.EXTRACTORS),
                  "pairs": all(sr.overlaps(cells[s][n]["ci"][k], bands[n][k]) for n in sr.EXTRACTORS for k in sr.SHARES),
                  "signed": all(sr.overlaps(cells[s][n]["c60"]["ci"], SIGNED_BANDS[n]) for n in sr.EXTRACTORS)} for s in speeds}
        pairs = {f"{a}-{b}": {k: bool(ok[a][k] and ok[b][k]) for k in ok[a]} for a, b in zip(speeds, speeds[1:])}
        report[pattern] = {"valid": len(mine) == len(everything), "prints": len(mine), "of": len(everything), "cells": {str(s): c for s, c in cells.items()},
                           "by_speed": {str(s): v for s, v in ok.items()}, "by_pair": pairs, "all_hold": [p for p, v in pairs.items() if all(v.values())]}
    return report


def main(out, figure, paths):
    started = provenance.start()
    bands = json.load(open(paths.RESULTS / "sweep_d010.json"))["bands"]
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(one, sorted(paths), chunksize=1))
    report = read(rows, bands, np.random.default_rng(0))
    for pattern, r in report.items():
        if isinstance(r, dict):
            print(pattern, f"{r['prints']}/{r['of']} valid", "ALL HOLD at", r["all_hold"], {p: "".join(k[0] if v else "-" for k, v in x.items()) for p, x in r["by_pair"].items()}, flush=True)
    sr.sheet(rows, figure)
    report.update(bands=bands, signed_bands=SIGNED_BANDS, _provenance=provenance.record(started))
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
