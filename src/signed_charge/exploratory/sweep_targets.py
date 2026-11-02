"""Real prints' numbers the sweep is built to and tested on (D-075): sweep_targets.py OUT.json

In `flux_scale`'s patches of FVC DB1-3 B, the spread of the ridge direction smoothed at 50 px, which sets the sweep's blueprint, and of the local ridge period, its fourth target. By finger.

Run: python -m signed_charge.exploratory.sweep_targets OUT.json
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from signed_charge.reads import flux_scale as fx

w, sc, sf, sm, fs, fo, parts, place = fx.w, fx.sc, fx.sf, fx.sm, fx.fs, fx.fo, fx.parts, fx.place


def spread(z2, fg, inner, singular, quality, period):
    out = []
    smooth = fx.smooth(z2, fg, 50)
    for i0, j0 in w.patches(inner, singular, fx.SIDE):
        if quality[i0 : i0 + fx.SIDE, j0 : j0 + fx.SIDE].min() < 3:
            continue
        beta, mismatch = w.unwrap_patch(np.angle(smooth), i0, i0 + fx.SIDE, j0, j0 + fx.SIDE)
        if abs(mismatch) > 1:
            continue
        p = period[i0 : i0 + fx.SIDE, j0 : j0 + fx.SIDE]
        out.append((float(beta.std()), float(p.std() / p.mean()), float(p.mean())))
    return out


def task(t):
    row, source = t
    im = place.image(source)
    fg = fs.foreground(im)
    z2 = sm.direction(im)
    z2 = z2 / (np.abs(z2) + 1e-9)
    _, _, period = fo.fields(im, fg)
    return row["finger"], spread(z2, fg, parts.mask_of(row), sc.singular(fs.field(im), fg), sf.quality(im), period)


def main(out):
    rows = pickle.load(open(w.CACHE, "rb"))["rows"]
    where = place.sources(rows)
    tasks = [(dict(finger=r["finger"], mask=r.get("mask"), **{k: r[k] for k in r if k not in ("finger", "mask")}), where[id(r)]) for r in rows if parts.member(r, "real") and id(r) in where]
    with ProcessPoolExecutor(fx.JOBS) as pool:
        got = list(pool.map(task, tasks, chunksize=4))
    values = np.array([v for _, g in got for v in g])
    report = {"patches": len(values), "direction_sd": float(values[:, 0].mean()), "period_cv": float(values[:, 1].mean()), "period_cv_q": np.percentile(values[:, 1], (10, 90)).tolist(), "period": float(values[:, 2].mean())}
    print(report)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
