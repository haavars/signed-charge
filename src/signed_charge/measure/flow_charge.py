"""What a print's flow says about its charge, and how patchy its minutiae are, with no front in it (D-054): flow_charge.py CACHE.pkl OUT.json RUN.npz...

For FVC DB1-3 B, SFinGe, model 2, E-027's best, the population, and the planar prints RUN.npz, inside the inner foreground within `DISC` px of its centroid, cells within `sc.GUARD` px of a core or delta out. The flow is `sm.direction` smoothed as a unit double angle at `SCALE` px, 50, as at 12 px a minutia's own bend in the ridges is read as the flow's (E-084), and also at 30 for comparison; k along the ridges, d = div k.

    agreement   mean of sign(d) with k taken along the minutia's direction: does a minutia point the way the ridges spread?
    flow h      `FLOW_POINTS` cells drawn in proportion to |d|, each given the direction in which the ridges spread: h(15-45) and h(60-100) of these, by `sf.binned`, so a charge with no minutia in it
    K(L)        boxes of L px, over 95 % inside: the within-box pairs against the across-box pairs, less 1, so 0 for a uniform scatter, unchanged if minutiae are thinned at random, and print to print differences in count taken out

CI by finger.

Run: python -m signed_charge.measure.flow_charge CACHE.pkl OUT.json RUN.npz...
LIMIT=n takes the first n prints of each set, to try the code.
"""
import json
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import real_mismatch as rm
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 8))
LIMIT = int(os.environ.get("LIMIT", 0))
RESAMPLES = 2000
DISC, SCALE, ALSO, FLOW_POINTS = 200, 50.0, 30.0, 150
BOXES = (60, 100)
SETS = ("real", "sfinge", "model2", "e027", "population")
ROWS = []


def spreading(field, scale, fg=None):
    """(k's angle, div k), k along the ridges, from the unit double angle at `scale`.

    With `fg`, only cells on the print count and nothing lies past the image's edge (D-062). Without it, as every read to E-097, cv2 reflects the image at its edge and the background is blurred in, which on a print of 300 px replaces much of d from 30 px up (E-099)."""
    z = field / (np.abs(field) + 1e-9)
    if fg is None:
        z = cv2.GaussianBlur(z.real.astype(np.float32), (0, 0), scale) + 1j * cv2.GaussianBlur(z.imag.astype(np.float32), (0, 0), scale)
    else:
        z = z * fg
        z = cv2.GaussianBlur(z.real.astype(np.float32), (0, 0), scale, borderType=cv2.BORDER_CONSTANT) + 1j * cv2.GaussianBlur(z.imag.astype(np.float32), (0, 0), scale, borderType=cv2.BORDER_CONSTANT)
    z = z / (np.abs(z) + 1e-9)
    zx, zy = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) / 2, (np.roll(z, -1, 0) - np.roll(z, 1, 0)) / 2
    tx, ty = np.imag(np.conj(z) * zx) / 2, np.imag(np.conj(z) * zy) / 2  # z is the gradient's: its angle less 90° is twice k's, and the turning is the same
    angle = np.angle(z) / 2 + np.pi / 2
    return angle, -np.sin(angle) * tx + np.cos(angle) * ty


def box_pairs(points, mask, size):
    """(within-box pairs per box, across-box pairs per pair of boxes), from the print's whole boxes."""
    h, w = mask.shape
    counts = []
    labels = np.full(mask.shape, -1, int)
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            if mask[y : y + size, x : x + size].mean() > 0.95:
                labels[y : y + size, x : x + size] = len(counts)
                counts.append(0)
    if len(counts) < 2 or not len(points):
        return None
    xi, yi = points[:, 0].astype(int), points[:, 1].astype(int)
    hit = labels[yi, xi]
    n = np.bincount(hit[hit >= 0], minlength=len(counts)).astype(float)
    return float((n * (n - 1)).sum() / len(n)), float((n.sum() ** 2 - (n**2).sum()) / (len(n) * (len(n) - 1)))


def one(task):
    index, source = task
    row = ROWS[index]
    im = place.image(source)
    inner, fg = parts.mask_of(row), fs.foreground(im)
    ys, xs = np.nonzero(inner)
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    disc = inner & (np.hypot(xx - xs.mean(), yy - ys.mean()) <= DISC)
    guarded = disc.copy()
    for x, y in sc.singular(fs.field(im), fg):
        guarded &= np.hypot(xx - x, yy - y) >= sc.GUARD
    field = sm.direction(im)
    angle, d = spreading(field, SCALE)
    angle_also, d_also = spreading(field, ALSO)
    out = {"finger": row["finger"]}
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(row["points"][reader], guarded)
        heading = sf.heading(p, y_sign)
        cy, cx = p[:, 1].astype(int), p[:, 0].astype(int)
        agree = lambda a, b: float(np.sign(b[cy, cx] * np.sign(np.cos(heading) * np.cos(a[cy, cx]) + np.sin(heading) * np.sin(a[cy, cx]))).sum())  # noqa: E731
        out[reader] = {"agree": (agree(angle, d), len(p)), "agree_also": (agree(angle_also, d_also), len(p)), "boxes": {size: box_pairs(sf.inside(row["points"][reader], disc), disc, size) for size in BOXES}}
    rng = np.random.default_rng(zlib.crc32(row["finger"].encode()))
    cells = np.flatnonzero(guarded)
    weight = np.abs(d).ravel()[cells]
    if len(cells) > FLOW_POINTS and weight.sum() > 0:
        chosen = rng.choice(cells, FLOW_POINTS, replace=False, p=weight / weight.sum())
        py, px = np.unravel_index(chosen, im.shape)
        pointing = angle[py, px] + np.where(d[py, px] < 0, np.pi, 0)
        total, count, _ = sf.binned(np.stack([px, py, pointing], 1).astype(float), 1.0, field, guarded)
        band = lambda a, b: (sm.EDGES[:-1] >= a) & (sm.EDGES[1:] <= b)  # noqa: E731
        out["flow"] = [float(total[band(*sf.SHORT)].sum()), float(count[band(*sf.SHORT)].sum()), float(total[band(*sf.LONG)].sum()), float(count[band(*sf.LONG)].sum())]
    return out


def summary(rows, rng):
    fingers = sorted({r["finger"] for r in rows})
    where = {f: i for i, f in enumerate(fingers)}
    draws = [rng.integers(len(fingers), size=len(fingers)) for _ in range(RESAMPLES)]
    every = np.arange(len(fingers))
    with_ci = lambda f: {"mean": float(f(every)), "ci": [float(np.percentile([f(i) for i in draws], q)) for q in (2.5, 97.5)]}  # noqa: E731
    report = {"prints": len(rows), "fingers": len(fingers)}
    flow = np.zeros((len(fingers), 4))
    for r in rows:
        if "flow" in r:
            flow[where[r["finger"]]] += r["flow"]
    report["flow_h_short"] = with_ci(lambda i: flow[i, 0].sum() / max(flow[i, 1].sum(), 1))
    report["flow_h_long"] = with_ci(lambda i: flow[i, 2].sum() / max(flow[i, 3].sum(), 1))
    for reader in sm.EXTRACTORS:
        entry = {}
        for key, name in (("agree", "agreement"), ("agree_also", "agreement_also")):
            agree = np.zeros((len(fingers), 2))
            for r in rows:
                agree[where[r["finger"]]] += r[reader][key]
            entry[name] = with_ci(lambda i: agree[i, 0].sum() / max(agree[i, 1].sum(), 1))
        for size in BOXES:
            values = [[] for _ in fingers]
            for r in rows:
                got = r[reader]["boxes"][size]
                if got and got[1] > 0:
                    values[where[r["finger"]]].append(got[0] / got[1] - 1)
            means = np.array([np.mean(v) if v else np.nan for v in values])
            keep = np.isfinite(means)
            entry[f"K{size}"] = with_ci(lambda i: np.nanmean(means[i])) | {"fingers": int(keep.sum())}
        report[reader] = entry
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
        name = next((n for n in SETS if parts.member(row, n)), "field" if row["set"] == "field" else None)
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
