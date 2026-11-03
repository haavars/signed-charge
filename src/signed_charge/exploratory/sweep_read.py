"""D-068 step 3: the sweep against real prints' patches (D-075): sweep_read.py pilot OUT.json | sweep_read.py test OUT.json b s k m

    pilot   `GRID` on `PILOT` sheets a cell, seeds from 1000. Each cell's N, X, h(30-60 px) on one ridge and period spread, and D, the distance D-075 freezes on
    test    one setting on `TEST` new sheets, seeds from 100000, read as `flux_scale` and `ridge_frame`, CI by sheet

Run: python -m signed_charge.exploratory.sweep_read ...
"""

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from signed_charge.reads import flux_scale as fx
from signed_charge.measure import ridge_frame as rf
from signed_charge.exploratory import sweep

w, sc, sf, sm, fs, fo = fx.w, fx.sc, fx.sf, fx.sm, fx.fs, fx.fo
JOBS = int(os.environ.get("READ_JOBS", 8))
GRID = list(itertools.product((0.0012, 0.0018, 0.0025), (1.0, 3.0, 6.0), (0.1, 0.3), (0.0, 0.3, 0.6)))
if os.environ.get("D080"):
    GRID = list(itertools.product((0.0025, 0.0035), (1.0, 3.0), (0.1, 0.3, 0.45), (0.3,), (9.0, 9.6)))
PILOT, TEST = int(os.environ.get("PILOT", 12)), 200
SEED_PILOT, SEED_TEST = int(os.environ.get("SEED_PILOT", 1000)), int(os.environ.get("SEED_TEST", 100000))
REAL = {"N": (4.14, 4.07), "X": (2.8, 2.73), "h": (0.01, 0.05), "cv": 0.037}
NEAR, NEXT = 2, 0  # `ridge_frame` bins: 30-60 px along, same ridge


def one(task):
    seed, params = task
    im = sweep.sheet(seed, *params)
    inner = w.inner_of(im)
    points = w.fresh_points(im, inner)
    fg = fs.foreground(im)
    _, _, period = fo.fields(im, fg)
    cvs = []
    for i0 in range(32, im.shape[0] - 128, 48):
        for j0 in range(32, im.shape[1] - 128, 48):
            p = period[i0 : i0 + 96, j0 : j0 + 96]
            cvs.append(float(p.std() / p.mean()))
    finger = f"sweep/{seed}"
    return {
        "patches": fx.read(im, points, inner, finger),
        "frame": rf.read(im, points, inner, finger),
        "cv": float(np.mean(cvs)),
        "period": float(period[inner].mean()),
        "finger": finger,
    }


def excess(patches, q):
    return float(np.mean([r[q] ** 2 - r["n_" + q] for r in patches])) if patches else float("nan")


def score(rows):
    patches = [p for r in rows for p in r["patches"]]
    out = {"cv": float(np.mean([r["cv"] for r in rows])), "period": float(np.mean([r["period"] for r in rows])), "patches": len(patches)}
    d = ((out["cv"] - REAL["cv"]) / 0.01) ** 2
    for i, q in enumerate(("mindtct", "leader")):
        n = float(np.mean([p["n_" + q] for p in patches])) if patches else float("nan")
        x = excess(patches, q)
        c, m, _ = sum(np.stack(r["frame"][q]) for r in rows)
        h = float(m[NEAR, NEXT] / max(c[NEAR, NEXT], 1))
        out[q] = {"N": n, "X": x, "h": h}
        d += ((n - REAL["N"][i]) / 0.5) ** 2 + ((x - REAL["X"][i]) / 1.25) ** 2 + ((h - REAL["h"][i]) / 0.1) ** 2
    out["D"] = float(d)
    return out


def main(mode, out, rest):
    with ProcessPoolExecutor(JOBS) as pool:
        if mode == "pilot":
            cells = list(enumerate(GRID))
            tasks = [(SEED_PILOT + c * PILOT + i, params) for c, params in cells for i in range(PILOT)]
            rows = list(pool.map(one, tasks, chunksize=2))
            report = [dict(params=list(params), **score(rows[c * PILOT : (c + 1) * PILOT])) for c, params in cells]
            report.sort(key=lambda r: r["D"] if r["D"] == r["D"] else 1e9)
            for r in report[:6]:
                print(r, flush=True)
            json.dump(report, open(out, "w"), indent=1)
            return
        params = tuple(float(v) for v in rest)
        rows = list(pool.map(one, [(SEED_TEST + i, params) for i in range(TEST)], chunksize=2))
    rng = np.random.default_rng(0)
    patches = [p for r in rows for p in r["patches"]]
    report = {
        "params": list(params),
        "sheets": len(rows),
        "flux": fx.summary(patches, rng),
        "frame": rf.summary([r["frame"] for r in rows], rng),
        "cv": fx.boot([{"finger": r["finger"], "v": r["cv"]} for r in rows], lambda p: float(np.mean([x["v"] for x in p])), rng),
        "period": float(np.mean([r["period"] for r in rows])),
    }
    report["_provenance"] = fx.provenance.record(fx.provenance.start())
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
