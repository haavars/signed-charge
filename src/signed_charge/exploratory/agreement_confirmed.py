"""Is real minutiae's low agreement with the flow false minutiae? (D-056): agreement_confirmed.py OUT.json

`flow_charge`'s agreement at 50 px, for FVC DB1-3 B and SFinGe, 8 impressions a finger, and model 2, E-068's 96 fingers cut to 30, with every minutia in the inner foreground, 30 px from cores and deltas, and:
    confirmed   found again in at least 3 of the finger's 7 other impressions, `signed_false.support`
    strong      in at least 5
    clean       in mindtct's quality-4 blocks
Also by the flow's own strength, |d| terciles over all the set's minutiae, and endings and forks apart. CI by finger.

Run: python -m signed_charge.exploratory.agreement_confirmed OUT.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import flow_charge as fc
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 4))
RESAMPLES = 2000
STRONG = 5
KEYS = ("all", "confirmed", "strong", "clean")


def finger(task):
    name, label, images = task
    ready = []
    for im in images:
        fg = fs.foreground(im)
        if fg is None:
            continue
        inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
        guarded = inner.copy()
        yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
        for x, y in sc.singular(fs.field(im), fg):
            guarded &= np.hypot(xx - x, yy - y) >= sc.GUARD
        ready.append((inner, guarded, *fc.spreading(sm.direction(im), fc.SCALE), sf.quality(im), im))
    out = {"set": name, "finger": label}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        points = [sf.inside(extract(r[5]), r[0]) for r in ready]
        counts = sf.support(points, y_sign)
        rows = []
        for p, c, (_, guarded, angle, d, q, _) in zip(points, counts, ready):
            if not len(p):
                continue
            xi, yi = p[:, 0].astype(int), p[:, 1].astype(int)
            ok = guarded[yi, xi]
            heading = sf.heading(p, y_sign)
            sign = np.sign(d[yi, xi] * np.sign(np.cos(heading) * np.cos(angle[yi, xi]) + np.sin(heading) * np.sin(angle[yi, xi])))
            rows.append(np.stack([sign, np.abs(d[yi, xi]), c, q[yi, xi] == 4, p[:, 3]], 1)[ok])
        out[reader] = np.concatenate(rows) if rows else np.zeros((0, 5))
    return out


def pick(a, key):
    return {"all": a, "confirmed": a[a[:, 2] >= sf.SUPPORT], "strong": a[a[:, 2] >= STRONG], "clean": a[a[:, 3] > 0]}[key]


def report_for(rows, reader, rng):
    tables = [r[reader] for r in rows]
    draws = [rng.integers(len(tables), size=len(tables)) for _ in range(RESAMPLES)]
    every = np.arange(len(tables))
    with_ci = lambda s, n: {"mean": float(s.sum() / max(n.sum(), 1)), "ci": [float(np.percentile([s[i].sum() / max(n[i].sum(), 1) for i in draws], q)) for q in (2.5, 97.5)], "minutiae": int(n.sum())}  # noqa: E731
    out = {}
    for key in KEYS:
        got = [pick(t, key) for t in tables]
        out[key] = with_ci(np.array([g[:, 0].sum() for g in got]), np.array([len(g) for g in got], float))
    edges = np.percentile(np.concatenate(tables)[:, 1], [100 / 3, 200 / 3])
    for i, name in enumerate(("weak", "middle", "steep")):
        got = [t[np.digitize(t[:, 1], edges) == i] for t in tables]
        out[name] = with_ci(np.array([g[:, 0].sum() for g in got]), np.array([len(g) for g in got], float))
    for value, name in ((0, "endings"), (1, "forks")):
        got = [t[t[:, 4] == value] for t in tables]
        out[name] = with_ci(np.array([g[:, 0].sum() for g in got]), np.array([len(g) for g in got], float))
    return out


def main(out):
    started = provenance.start()
    with ProcessPoolExecutor(JOBS) as pool:
        fvc = list(pool.map(fvc_task, sf.fvc_fingers(), chunksize=1))
        model = list(pool.map(variant_task, [("model2", seed, f) for seed in sf.FRESH[:4] for f in sf.FINGERS[:8]][:30], chunksize=1))
    rng = np.random.default_rng(0)
    groups = {"real": [r for r in fvc if "DB4" not in r["set"]], "sfinge": [r for r in fvc if "DB4" in r["set"]], "model2": model}
    report = {name: {reader: report_for(rows, reader, rng) for reader in sm.EXTRACTORS} | {"fingers": len(rows)} for name, rows in groups.items()}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


def fvc_task(task):
    name, label, paths = task
    return finger((name, label, [fs.at_500(cv2.imread(p, 0), sf.cf.PPI.get(name, 500)) for p in paths]))


def variant_task(task):
    variant, seed, fgp = task
    return finger((variant, f"PH{seed}/{fgp}", [sf.fc.image((variant, seed, fgp, c))[1] for c in sf.CAPTURES]))


if __name__ == "__main__":
    main(sys.argv[1])
