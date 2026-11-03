"""Where is Phi d_sigma unpaid: near singular points, near the print's edge, or everywhere? The cell test of winding.py with
mindtct's minutiae as the only charges, cells split by distance to the nearest core/delta and to the foreground's edge.
Run from python_biometrics: RIDGEGEN_TOOLS=$PWD/tools PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python edge_cells.py OUT.json SETS...
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.measure import winding as wd
from signed_charge.measure.winding import fc, fo, fs, parts, place, sc, sf, sm

JOBS = int(os.environ.get("JOBS", 4))
LIMIT = int(os.environ.get("LIMIT", 0))
SCALES = (50, 100, 150)
ROWS = []


def one(task):
    index, source = task
    row = ROWS[index]
    im = place.image(source)
    inner = parts.mask_of(row)
    fg = fs.foreground(im)
    h, w = im.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(inner)
    disc = inner & (np.hypot(xx - xs.mean(), yy - ys.mean()) <= fc.DISC)
    singular = sc.singular(fs.field(im), fg)
    guarded = disc.copy()
    near = np.full(im.shape, np.inf)
    for x, y in singular:
        r = np.hypot(xx - x, yy - y)
        guarded &= r >= sc.GUARD
        near = np.minimum(near, r)
    edge = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
    z2 = sm.direction(im)
    z2 = z2 / (np.abs(z2) + 1e-9)
    th2 = np.angle(z2)
    _, _, period = fo.fields(im, fg)
    phi = 1.0 / np.maximum(period, 1.0)
    cy, cx = np.nonzero(guarded[:: wd.STRIDE, :: wd.STRIDE])
    cells = np.c_[cx * wd.STRIDE, cy * wd.STRIDE].astype(float)
    out = {"finger": row["finger"], "singular": len(singular)}
    if len(cells) < 10:
        return out
    ci_, cj_ = cells[:, 1].astype(int), cells[:, 0].astype(int)
    beta_c = th2[ci_, cj_] / 2
    kcx, kcy = wd.rot(np.cos(beta_c), np.sin(beta_c))
    p = sf.inside(row["points"]["mindtct"], inner)
    if len(p) < 3:
        return out
    hd = sf.heading(p, sm.EXTRACTORS["mindtct"][1])
    beta_m = th2[p[:, 1].astype(int), p[:, 0].astype(int)] / 2
    kmx, kmy = wd.rot(np.cos(beta_m), np.sin(beta_m))
    q = np.sign(np.cos(hd) * kmx + np.sin(hd) * kmy + 1e-12)
    lift = wd.lifted(th2, cells, p[:, :2], q)
    dist = np.hypot(cells[:, 0][:, None] - p[:, 0][None], cells[:, 1][:, None] - p[:, 1][None])
    near_c, edge_c, phi_c = near[ci_, cj_], edge[ci_, cj_], phi[ci_, cj_]
    for s in SCALES:
        ang, d = fc.spreading(z2, float(s))
        dc = d[ci_, cj_] * np.sign(np.cos(ang[ci_, cj_]) * kcx + np.sin(ang[ci_, cj_]) * kcy + 1e-12) * phi_c
        rho = (lift * np.exp(-(dist**2) / (2 * s * s)) / (2 * np.pi * s * s)).sum(1)
        out[s] = {}
        for part, pick in (
            ("near_singular", near_c < s),
            ("far_shallow", (near_c >= s) & (edge_c < s)),
            ("far_deep", (near_c >= s) & (edge_c >= s)),
            ("far_deep2", (near_c >= 2 * s) & (edge_c >= 2 * s)),
            ("all", np.ones(len(cells), bool)),
        ):
            if pick.sum() < 3:
                continue
            out[s][part] = (
                int(pick.sum()),
                float(np.abs(dc[pick]).sum()),
                float(dc[pick] @ dc[pick]),
                float(np.abs(rho[pick]).sum()),
                float(rho[pick] @ rho[pick]),
                float(rho[pick] @ dc[pick]),
            )
    return out


def main(out, sets):
    base = pickle.load(open(wd.CACHE, "rb"))["rows"]
    where = place.sources(base)
    tasks, seen = [], {}
    for row in base:
        name = next((n for n in ("real", "sfinge", "model2") if parts.member(row, n)), None)
        if name is None or name not in sets or id(row) not in where or (LIMIT and seen.get(name, 0) >= LIMIT):
            continue
        seen[name] = seen.get(name, 0) + 1
        ROWS.append(row)
        tasks.append((name, (len(ROWS) - 1, where[id(row)])))
    print(len(tasks), "tasks", flush=True)
    with ProcessPoolExecutor(JOBS) as pool:
        done = list(pool.map(one, [t[1] for t in tasks], chunksize=2))
    report = {}
    for name in sorted(seen):
        rows = [r for t, r in zip(tasks, done) if t[0] == name]
        rep = {"prints": len(rows), "singular_mean": float(np.mean([r["singular"] for r in rows]))}
        print("==", name, len(rows), "prints, singular per print", round(rep["singular_mean"], 2))
        for s in SCALES:
            for part in ("near_singular", "far_shallow", "far_deep", "far_deep2", "all"):
                t = np.array([r[s][part] for r in rows if s in r and part in r[s]])
                if len(t) < 3:
                    continue
                n = t[:, 0].sum()
                e = {
                    "prints": len(t),
                    "cells": int(n),
                    "abs_phi_d": float(t[:, 1].sum() / n),
                    "abs_rho": float(t[:, 3].sum() / n),
                    "d_on_rho": float(t[:, 5].sum() / max(t[:, 4].sum(), 1e-24)),
                    "corr": float(t[:, 5].sum() / np.sqrt(max(t[:, 2].sum() * t[:, 4].sum(), 1e-30))),
                }
                rep[f"{s}/{part}"] = e
                print(
                    f"  s={s} {part:13s} prints={len(t):4d} cells={int(n):7d} |Phi d|={e['abs_phi_d']:.2e} |rho|={e['abs_rho']:.2e} ratio={e['abs_rho'] / e['abs_phi_d']:.2f} d/rho={e['d_on_rho']:.2f} corr={e['corr']:.2f}"
                )
        report[name] = rep
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["real", "model2"])
