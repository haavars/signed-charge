"""Is a real print's smoothed splay d the finger's, or the capture's? Two impressions of one finger, rigidly aligned on
their mindtct minutiae (signed_false.align), the frame-free splay vector s = d k compared cell by cell in the common
guarded disc, at each scale: correlation and the slope of impression 2 on impression 1. The orientation's own
agreement (unit double angle, turned) checks the alignment. Sets: real (FVC DB1-3 B), SFinGe, model 2 (2 captures).
Run from python_biometrics: RIDGEGEN_TOOLS=$PWD/tools PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python repeat_d.py OUT.json
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_charge as fc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge import paths

JOBS = int(os.environ.get("JOBS", 4))
LIMIT = int(os.environ.get("LIMIT", 0))
SCALES = (30, 50, 70, 100, 150)
CACHE = str(paths.CACHE)
ROWS = []


def fields(row, source):
    im = place.image(source)
    inner, fg = parts.mask_of(row), fs.foreground(im)
    ys, xs = np.nonzero(inner)
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    guarded = inner & (np.hypot(xx - xs.mean(), yy - ys.mean()) <= fc.DISC)
    near = np.full(im.shape, np.inf)
    for x, y in sc.singular(fs.field(im), fg):
        r = np.hypot(xx - x, yy - y)
        guarded &= r >= sc.GUARD
        near = np.minimum(near, r)
    z2 = sm.direction(im)
    z2 = z2 / (np.abs(z2) + 1e-9)
    splay = {}
    for s in SCALES:
        ang, d = fc.spreading(z2, float(s))
        splay[s] = np.stack([d * np.cos(ang), d * np.sin(ang)], -1)
    return guarded, near, z2, splay


def pair(task):
    (ia, sa), (ib, sb) = task
    ra, rb = ROWS[ia], ROWS[ib]
    pa, pb = ra["points"]["mindtct"], rb["points"]["mindtct"]
    if len(pa) < sf.AT_LEAST or len(pb) < sf.AT_LEAST:
        return None
    y_sign = sm.EXTRACTORS["mindtct"][1]
    move = sf.align(pa[:, :2], sf.heading(pa, y_sign), pb[:, :2], sf.heading(pb, y_sign))
    if move is None:
        return None
    turn, shift, count = move
    co, si = np.cos(turn), np.sin(turn)
    rotate = np.array([[co, si], [-si, co]])
    ga, na, za, spa = fields(ra, sa)
    gb, nb, zb, spb = fields(rb, sb)
    h, w = gb.shape
    yb, xb = np.nonzero(gb)
    back = (np.c_[xb, yb] - shift) @ rotate.T
    xa, ya = np.rint(back[:, 0]).astype(int), np.rint(back[:, 1]).astype(int)
    ok = (xa >= 0) & (xa < ga.shape[1]) & (ya >= 0) & (ya < ga.shape[0])
    ok[ok] &= ga[ya[ok], xa[ok]]
    if ok.sum() < 500:
        return None
    xb, yb, xa, ya = xb[ok], yb[ok], xa[ok], ya[ok]
    out = {
        "finger": ra["finger"],
        "cells": int(ok.sum()),
        "matches": int(count),
        "orientation": float(np.abs(np.mean(zb[yb, xb] * np.conj(za[ya, xa] * np.exp(2j * turn))))),
    }
    near_b = np.minimum(nb[yb, xb], na[ya, xa])
    for s in SCALES:
        vb = spb[s][yb, xb]
        va = spa[s][ya, xa] @ rotate
        for part, pick in (("far", near_b >= s), ("all", np.ones(len(xb), bool))):
            if pick.sum() < 50:
                continue
            a, b = va[pick], vb[pick]
            out[f"{s}/{part}"] = (
                float((a * b).sum()),
                float((a * a).sum()),
                float((b * b).sum()),
                float(np.linalg.norm(a, axis=1).sum()),
                float(np.linalg.norm(b, axis=1).sum()),
                int(pick.sum()),
            )
    return out


def main(out):
    base = pickle.load(open(CACHE, "rb"))["rows"]
    where = place.sources(base)
    groups = {}
    for row in base:
        name = next((n for n in ("real", "sfinge", "model2") if parts.member(row, n)), None)
        if name is None or id(row) not in where:
            continue
        ROWS.append(row)
        groups.setdefault((name, row["finger"]), []).append((len(ROWS) - 1, where[id(row)]))
    tasks = []
    for (name, finger), items in groups.items():
        for k in range(0, len(items) - 1, 2):
            tasks.append((name, (items[k], items[k + 1])))
    if LIMIT:
        keep = {}
        tasks = [t for t in tasks if keep.setdefault(t[0], []).append(t) or len(keep[t[0]]) <= LIMIT]
    print(len(tasks), "pairs", flush=True)
    with ProcessPoolExecutor(JOBS) as pool:
        done = list(pool.map(pair, [t[1] for t in tasks], chunksize=1))
    rng = np.random.default_rng(0)
    report = {}
    for name in sorted({t[0] for t in tasks}):
        rows = [r for t, r in zip(tasks, done) if t[0] == name and r]
        fingers = sorted({r["finger"] for r in rows})
        rep = {
            "pairs": len(rows),
            "fingers": len(fingers),
            "aligned_of": sum(1 for t in tasks if t[0] == name),
            "orientation": float(np.mean([r["orientation"] for r in rows])),
            "cells": float(np.mean([r["cells"] for r in rows])),
        }
        for s in SCALES:
            for part in ("far", "all"):
                key = f"{s}/{part}"
                have = [r for r in rows if key in r]
                if len(have) < 3:
                    continue
                t = np.array([r[key] for r in have])
                f = np.array([fingers.index(r["finger"]) for r in have])
                stat = lambda i: (t[i, 0].sum() / np.sqrt(t[i, 1].sum() * t[i, 2].sum()), t[i, 0].sum() / t[i, 1].sum(), t[i, 4].sum() / t[i, 3].sum())  # noqa: E731
                full = stat(np.arange(len(t)))
                draws = np.array([stat(np.concatenate([np.flatnonzero(f == j) for j in rng.integers(len(fingers), size=len(fingers))])) for _ in range(500)])
                rep[key] = {
                    "pairs": len(have),
                    "cells": int(t[:, 5].sum()),
                    "corr": float(full[0]),
                    "corr_ci": [float(np.percentile(draws[:, 0], q)) for q in (2.5, 97.5)],
                    "slope": float(full[1]),
                    "abs_ratio_b_over_a": float(full[2]),
                    "mean_abs_a": float(t[:, 3].sum() / t[:, 5].sum()),
                }
        report[name] = rep
        print("==", name, rep["pairs"], "pairs of", rep["aligned_of"], "orientation agreement", round(rep["orientation"], 3))
        for s in SCALES:
            for part in ("far", "all"):
                e = rep.get(f"{s}/{part}")
                if e:
                    print(
                        f"  {s} {part}: corr {e['corr']:.2f} ({e['corr_ci'][0]:.2f}-{e['corr_ci'][1]:.2f}) slope {e['slope']:.2f} |d| {e['mean_abs_a']:.2e} cells {e['cells']}"
                    )
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
