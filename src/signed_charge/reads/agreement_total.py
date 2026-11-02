"""Does agreement rise when ridge frequency counts too? (D-057): agreement_total.py CACHE.pkl OUT.json RUN.npz...

Wieditz et al. 2022: ridges must be added where ridges spread or their frequency rises. Along k, at frequency Φ, the demand is Φ(div k + k · ∇ ln Φ), so `flow_charge`'s agreement, at 50 px, on d alone and on the whole of it, and on the second term alone. ln Φ from `fanout.fields`' period, smoothed at the same scale within the foreground. Every set of `flow_charge`, with the sizes of the two terms.

Run: python -m signed_charge.reads.agreement_total CACHE.pkl OUT.json RUN.npz...
LIMIT=n takes the first n prints of each set, to try the code.
"""
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import fanout as fo
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_charge as fc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import real_mismatch as rm
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 4))
LIMIT = int(os.environ.get("LIMIT", 0))
RESAMPLES = 2000
ROWS = []


def frequency_term(im, fg, angle, scale):
    """k · ∇ ln Φ, k at `angle`, ln Φ smoothed at `scale` within the foreground."""
    _, _, period = fo.fields(im, fg)
    weight = fg.astype(np.float32)
    log_f = -np.log(np.maximum(period, 1.0)).astype(np.float32) * weight
    smooth = cv2.GaussianBlur(log_f, (0, 0), scale, borderType=cv2.BORDER_CONSTANT) / np.maximum(cv2.GaussianBlur(weight, (0, 0), scale, borderType=cv2.BORDER_CONSTANT), 1e-3)
    gy, gx = np.gradient(smooth)
    return np.cos(angle) * gx + np.sin(angle) * gy


def one(task):
    index, source = task
    row = ROWS[index]
    im = place.image(source)
    inner, fg = parts.mask_of(row), fs.foreground(im)
    ys, xs = np.nonzero(inner)
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    guarded = inner & (np.hypot(xx - xs.mean(), yy - ys.mean()) <= fc.DISC)
    for x, y in sc.singular(fs.field(im), fg):
        guarded &= np.hypot(xx - x, yy - y) >= sc.GUARD
    angle, d = fc.spreading(sm.direction(im), fc.SCALE)
    term = frequency_term(im, fg, angle, fc.SCALE)
    total = d + term
    out = {"finger": row["finger"], "sizes": (float(np.abs(d[guarded]).mean()), float(np.abs(term[guarded]).mean()))}
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(row["points"][reader], guarded)
        heading = sf.heading(p, y_sign)
        cy, cx = p[:, 1].astype(int), p[:, 0].astype(int)
        along = np.sign(np.cos(heading) * np.cos(angle[cy, cx]) + np.sin(heading) * np.sin(angle[cy, cx]))
        out[reader] = [float(np.sign(a[cy, cx] * along).sum()) for a in (d, term, total)] + [len(p)]
    return out


def summary(rows, rng):
    fingers = sorted({r["finger"] for r in rows})
    where = {f: i for i, f in enumerate(fingers)}
    draws = [rng.integers(len(fingers), size=len(fingers)) for _ in range(RESAMPLES)]
    report = {"prints": len(rows), "fingers": len(fingers), "size_flow": float(np.mean([r["sizes"][0] for r in rows])), "size_frequency": float(np.mean([r["sizes"][1] for r in rows]))}
    for reader in sm.EXTRACTORS:
        table = np.zeros((len(fingers), 4))
        for r in rows:
            table[where[r["finger"]]] += r[reader]
        report[reader] = {}
        for column, name in enumerate(("flow", "frequency", "total")):
            value = lambda i: table[i, column].sum() / max(table[i, 3].sum(), 1)  # noqa: E731
            report[reader][name] = {"mean": float(value(np.arange(len(fingers)))), "ci": [float(np.percentile([value(i) for i in draws], q)) for q in (2.5, 97.5)]}
    return report


def main(cache, out, paths):
    started = provenance.start()
    base = pickle.load(open(cache, "rb"))["rows"]
    with ProcessPoolExecutor(JOBS) as pool:
        planar = list(pool.map(rm.field_row, sorted(paths)[: LIMIT or None], chunksize=1))
    rows = base + [r for r in planar if r]
    where = place.sources(rows)
    tasks, seen = [], {}
    for row in rows:
        name = next((n for n in fc.SETS if parts.member(row, n)), "field" if row["set"] == "field" else None)
        if name is None or id(row) not in where or (LIMIT and seen.get(name, 0) >= LIMIT):
            continue
        seen[name] = seen.get(name, 0) + 1
        ROWS.append(row)
        tasks.append((name, len(ROWS) - 1, where[id(row)]))
    with ProcessPoolExecutor(JOBS) as pool:
        read = list(pool.map(one, [t[1:] for t in tasks], chunksize=2))
    rng = np.random.default_rng(0)
    report = {name: summary([r for t, r in zip(tasks, read) if t[0] == name], rng) for name in seen}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
