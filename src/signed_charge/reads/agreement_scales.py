"""At what scale do a print's minutiae pay for its flow's spreading? (D-058): agreement_scales.py CACHE.pkl OUT.json RUN.npz...

`flow_charge`'s agreement at each of `SCALES` px, for FVC DB1-3 B, SFinGe, model 2, the planar prints RUN.npz, named by the start of their stem, and `SEEDS` flat sheets of `SPIRALS` independent spirals, real's density, where there's nothing to follow. CI by finger, or by sheet.

Run: python -m signed_charge.reads.agreement_scales CACHE.pkl OUT.json RUN.npz...
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

from signed_charge.reads import agreement_null as an
from signed_charge.reads import field_read as fr
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_charge as fc
from signed_charge.measure import minutiae_cache as mc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 6))
LIMIT = int(os.environ.get("LIMIT", 0))
SCALES, SETS = (30, 50, 70, 100, 150), ("real", "sfinge", "model2")
SPIRALS, SEEDS, RESAMPLES = 36, 30, 2000
ROWS = []


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
    field = sm.direction(im)
    spread = {scale: fc.spreading(field, float(scale), fg) for scale in SCALES}
    out = {"finger": row["finger"]}
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(row["points"][reader], guarded)
        out[reader] = {scale: an.agreement(p, y_sign, *spread[scale]) for scale in SCALES}
    return out


def sheet(seed):
    rng = np.random.default_rng(seed)
    h, w = sm.SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rot = rng.uniform(0, np.pi)
    psi = 2 * np.pi * (-(xx - w / 2) * np.sin(rot) + (yy - h / 2) * np.cos(rot)) / sm.PERIOD + rng.uniform(0, 2 * np.pi)
    for _ in range(SPIRALS):
        cx, cy = rng.uniform(40, w - 40), rng.uniform(40, h - 40)
        psi += rng.choice([-1, 1]) * np.arctan2(yy - cy, xx - cx)
    im = np.where(np.cos(psi) > 0, 40, 235).astype(np.uint8)
    fg = fs.foreground(im)
    field = sm.direction(im)
    spread = {scale: fc.spreading(field, float(scale), fg) for scale in SCALES}
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for x, y in sc.singular(fs.field(im), fg):
        inner &= np.hypot(xx - x, yy - y) >= sc.GUARD
    out = {"finger": str(seed)}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(extract(im), inner)
        out[reader] = {scale: an.agreement(p, y_sign, *spread[scale]) for scale in SCALES}
    return out


def summary(rows, rng):
    fingers = sorted({r["finger"] for r in rows})
    where = {f: i for i, f in enumerate(fingers)}
    draws = [rng.integers(len(fingers), size=len(fingers)) for _ in range(RESAMPLES)]
    report = {"prints": len(rows), "fingers": len(fingers)}
    for reader in sm.EXTRACTORS:
        report[reader] = {}
        for scale in SCALES:
            table = np.zeros((len(fingers), 2))
            for r in rows:
                table[where[r["finger"]]] += r[reader][scale]
            value = lambda i: table[i, 0].sum() / max(table[i, 1].sum(), 1)  # noqa: E731
            report[reader][scale] = {
                "mean": float(value(np.arange(len(fingers)))),
                "ci": [float(np.percentile([value(i) for i in draws], q)) for q in (2.5, 97.5)],
            }
    return report


def planar(path):
    stem = Path(path).stem
    row = mc.row("field", "PH" + stem, fr.sheet(path))
    return dict(row, path=str(path), arm=stem.split("_")[0]) if row else None


def main(cache, out, paths):
    started = provenance.start()
    base = pickle.load(open(cache, "rb"))["rows"]
    with ProcessPoolExecutor(JOBS) as pool:
        grown = [r for r in pool.map(planar, sorted(paths), chunksize=1) if r]
    rows = base + grown
    where = place.sources(rows)
    tasks, seen = [], {}
    for row in rows:
        name = row.get("arm") or next((n for n in SETS if parts.member(row, n)), None)
        if name is None or id(row) not in where or (LIMIT and seen.get(name, 0) >= LIMIT):
            continue
        seen[name] = seen.get(name, 0) + 1
        ROWS.append(row)
        tasks.append((name, len(ROWS) - 1, where[id(row)]))
    with ProcessPoolExecutor(JOBS) as pool:
        read = list(pool.map(one, [t[1:] for t in tasks], chunksize=2))
        null = list(pool.map(sheet, range(LIMIT or SEEDS), chunksize=1))
    rng = np.random.default_rng(0)
    report = {name: summary([r for t, r in zip(tasks, read) if t[0] == name], rng) for name in seen}
    report["null"] = summary(null, rng)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
