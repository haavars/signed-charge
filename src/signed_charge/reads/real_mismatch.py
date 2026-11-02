"""Do real minutiae sit where the flow crosses what a wave would lay? (D-053): real_mismatch.py CACHE.pkl OUT.json RUN.npz...

One reader for every set: FVC DB1-3 B, SFinGe (DB4 B) and model 2 from the cache, and the planar prints RUN.npz. A print's topmost core, its read flow, and T from `field_print.arrival`'s sites formula centred on that core, a ring and two lines 216 px either side, the lines turned to the inner foreground's long axis. θ from the flow and ∇T. Cells within `sc.GUARD` px of a core or delta out, and within 12 px of where T kinks. Prints with no core out.

Three measures, on mindtct, LEADER and `both`, minutiae of mindtct with one of LEADER's within `AGREE` px:
    rate ratio, 60-90 against 0-30: raw, and Mantel-Haenszel over position (`R_EDGES` by 30° of azimuth from the tip, folded left to right), then with splay terciles and a coherence half, the set's own, CI by finger
    mirror pairs: cells and their reflection across the long axis through the core, which have the same T, both kept, differing in θ by over `APART`°: density on the high side against the low
    density by θ in six bins, for the shelf of E-080

Run: python -m signed_charge.reads.real_mismatch CACHE.pkl OUT.json RUN.npz...
LIMIT=n takes the first n prints of each set, to try the code.
"""

import json
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as nd

from signed_charge.measure import fanout as fo
from signed_charge.reads import field_local as fl
from signed_charge.reads import field_read as fr
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import minutiae_cache as mc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.reads import minutiae_seams as seams
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc

JOBS = int(os.environ.get("READ_JOBS", 8))
LIMIT = int(os.environ.get("LIMIT", 0))
RESAMPLES = 2000
SIDE, RING = 216.0, 15.0  # px, half the sheet and the ring's start, as `field_print` on a 9 px period
LONGEST, ROUND = 30.0, 1.15  # the most a finger is taken to lean, and how elongated the foreground must be to say
R_EDGES, AZIMUTH = (80, 130), 30
APART, AGREE, STEP = 30.0, 10.0, 3
READERS = ("mindtct", "leader", "both")
LOW, HIGH = (0, 30), (60, 90.01)
ROWS = []


def field_row(path):
    seed, finger = re.search(r"field_(\d+)_(\d+)", Path(path).stem).groups()
    row = mc.row("field", f"PH{seed}/{finger}", fr.sheet(path))
    return dict(row, path=str(path)) if row else None


def long_axis(inner):
    """Unit vector (x, y) down the finger, from the foreground's second moments; straight down if it's too round or leans too far."""
    ys, xs = np.nonzero(inner)
    values, vectors = np.linalg.eigh(np.cov(np.stack([xs, ys]).astype(float)))
    a = vectors[:, 1] * (1 if vectors[1, 1] >= 0 else -1)
    if np.sqrt(values[1] / max(values[0], 1e-9)) < ROUND or np.rad2deg(np.arccos(np.clip(a[1], -1, 1))) > LONGEST:
        return np.array([0.0, 1.0])
    return a


def arrival(shape, core, axis):
    """The sites formula: the ring, and lines `SIDE` px down and up the long axis. Returns T and (along, across) from the core."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    dx, dy = xx - core[1], yy - core[0]
    along, across = dx * axis[0] + dy * axis[1], dx * axis[1] - dy * axis[0]
    return np.minimum.reduce([np.maximum(np.hypot(dx, dy) - RING, 0), SIDE - along, SIDE + along]), along, across


def splay_of(im, fg, inner):
    """How fast the flow turns across itself, per ridge period: `fanout`'s double angle, as the change of direction along the normal."""
    z, coherence, period = fo.fields(im, fg)
    zx, zy = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) / 2, (np.roll(z, -1, 0) - np.roll(z, 1, 0)) / 2
    tx, ty = np.imag(np.conj(z) * zx) / 2, np.imag(np.conj(z) * zy) / 2
    th = np.angle(z) / 2
    return np.abs(-np.sin(th) * tx + np.cos(th) * ty) * float(np.median(period[inner])), coherence


def agreed(a, b):
    if not len(a) or not len(b):
        return a[:0]
    d = np.hypot(a[:, 0, None] - b[:, 0], a[:, 1, None] - b[:, 1])
    return a[d.min(1) <= AGREE]


def cells_of(points):
    return points[:, 1].astype(int), points[:, 0].astype(int)


def one(task):
    index, source = task
    row = ROWS[index]
    im = place.image(source)
    inner, fg = parts.mask_of(row), fs.foreground(im)
    theta = fs.field(im)
    cores = fs.cores(theta, inner)
    out = {"set": row["set"], "finger": row["finger"], "cores": len(cores)}
    if not cores or fg is None:
        return out
    core = min((float(c[1]), float(c[0])) for c in cores)
    axis = long_axis(inner)
    T, along, across = arrival(im.shape, core, axis)
    guarded = inner.copy()
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    for x, y in sc.singular(theta, fg):
        guarded &= np.hypot(xx - x, yy - y) >= sc.GUARD
    off = guarded & (nd.maximum_filter(fl.kink(T), 25) < seams.SEAM)
    angle = fl.mismatch(theta, T)
    splay, coherence = splay_of(im, fg, inner)
    ring = np.minimum(np.digitize(np.hypot(along, across), R_EDGES), len(R_EDGES))
    azimuth = np.minimum((np.rad2deg(np.arctan2(np.abs(across), -along)) // AZIMUTH).astype(int), 180 // AZIMUTH - 1)
    position = (ring * (180 // AZIMUTH) + azimuth).astype(np.uint8)
    kind = np.full(im.shape, -1, np.int8)
    kind[off & (angle >= LOW[0]) & (angle < LOW[1])] = 0
    kind[off & (angle >= HIGH[0]) & (angle < HIGH[1])] = 1
    sample = np.zeros(im.shape, bool)
    sample[::STEP, ::STEP] = True
    keep = sample & (kind >= 0)
    pack = lambda ys, xs: np.stack([position[ys, xs], np.float32(splay[ys, xs]), np.float32(coherence[ys, xs]), kind[ys, xs]], 1).astype(np.float32)  # noqa: E731
    out["cells"] = pack(*np.nonzero(keep))
    points = {"mindtct": row["points"]["mindtct"], "leader": row["points"]["leader"]}
    points["both"] = agreed(points["mindtct"], points["leader"])
    mx = np.rint(core[1] + along * axis[0] - across * axis[1]).astype(int)
    my = np.rint(core[0] + along * axis[1] + across * axis[0]).astype(int)
    fits = (mx >= 0) & (mx < im.shape[1]) & (my >= 0) & (my < im.shape[0])
    paired = fits.copy()
    paired[fits] &= off[fits] & off[my[fits], mx[fits]]
    other = np.zeros(im.shape, np.float32)
    other[paired] = angle[my[paired], mx[paired]]
    high, low = paired & (angle - other > APART), paired & (other - angle > APART)
    out["mirror_cells"] = (int(high.sum()), int(low.sum()))
    out["bins"] = [int((off & (angle >= a) & (angle <= b)).sum()) for a, b in fl.BINS.values()]
    for reader, p in points.items():
        ys, xs = cells_of(p)
        hit = kind[ys, xs] >= 0
        out[reader] = {
            "minutiae": pack(ys[hit], xs[hit]),
            "mirror": (int(high[ys, xs].sum()), int(low[ys, xs].sum())),
            "bins": [int((off[ys, xs] & (angle[ys, xs] >= a) & (angle[ys, xs] <= b)).sum()) for a, b in fl.BINS.values()],
        }
    return out


def stratum(a, splay_edges, coherence_edge, level):
    """A stratum number for rows of `pack`: position alone, with splay, or with coherence too."""
    base = a[:, 0].astype(int) if level >= 1 else np.zeros(len(a), int)
    if level >= 2:
        base = base * 3 + np.digitize(a[:, 1], splay_edges)
    if level >= 3:
        base = base * 2 + (a[:, 2] >= coherence_edge)
    return base


def counts(rows, reader, splay_edges, coherence_edge, level):
    """(finger, stratum, kind) tables of cells and minutiae."""
    fingers = sorted({r["finger"] for r in rows})
    where = {f: i for i, f in enumerate(fingers)}
    size = (len(R_EDGES) + 1) * (180 // AZIMUTH) * 3 * 2
    cells, minutiae = np.zeros((len(fingers), size, 2)), np.zeros((len(fingers), size, 2))
    for r in rows:
        i = where[r["finger"]]
        for table, a in ((cells, r["cells"]), (minutiae, r[reader]["minutiae"])):
            if len(a):
                np.add.at(table[i], (stratum(a, splay_edges, coherence_edge, level), a[:, 3].astype(int)), 1)
    return cells, minutiae


def mantel_haenszel(c, m):
    t = c.sum(1)
    use = t > 0
    return float((m[use, 1] * c[use, 0] / t[use]).sum() / max((m[use, 0] * c[use, 1] / t[use]).sum(), 1e-9))


def with_ci(c, m, rng):
    draws = [rng.integers(len(c), size=len(c)) for _ in range(RESAMPLES)]
    values = [mantel_haenszel(c[i].sum(0), m[i].sum(0)) for i in draws]
    return {
        "ratio": mantel_haenszel(c.sum(0), m.sum(0)),
        "ci": [float(np.percentile(values, q)) for q in (2.5, 97.5)],
        "minutiae": [float(x) for x in m.sum((0, 1))],
    }


def group(rows, rng):
    every = np.concatenate([r["cells"] for r in rows])
    splay_edges, coherence_edge = np.percentile(every[:, 1], [100 / 3, 200 / 3]), float(np.median(every[:, 2]))
    report = {"prints": len(rows), "fingers": len({r["finger"] for r in rows})}
    for reader in READERS:
        entry = {}
        for level, name in enumerate(("raw", "position", "position_splay", "full")):
            entry[name] = with_ci(*counts(rows, reader, splay_edges, coherence_edge, level), rng)
        fingers = sorted({r["finger"] for r in rows})
        where = {f: i for i, f in enumerate(fingers)}
        pairs, bins, areas = np.zeros((len(fingers), 4)), np.zeros((len(fingers), 6)), np.zeros((len(fingers), 6))
        for r in rows:
            i = where[r["finger"]]
            pairs[i] += [*r[reader]["mirror"], *r["mirror_cells"]]
            bins[i] += r[reader]["bins"]
            areas[i] += r["bins"]
        ratio = lambda i: (pairs[i, 0].sum() / max(pairs[i, 2].sum(), 1)) / max(pairs[i, 1].sum() / max(pairs[i, 3].sum(), 1), 1e-9)  # noqa: E731
        draws = [rng.integers(len(fingers), size=len(fingers)) for _ in range(RESAMPLES)]
        entry["mirror"] = {
            "ratio": float(ratio(np.arange(len(fingers)))),
            "ci": [float(np.percentile([ratio(i) for i in draws], q)) for q in (2.5, 97.5)],
            "minutiae": [float(x) for x in pairs[:, :2].sum(0)],
            "cells": [float(x) for x in pairs[:, 2:].sum(0)],
        }
        entry["density"] = [float(a / max(b * ms.MM2, 1e-6) * 100) for a, b in zip(bins.sum(0), areas.sum(0))]
        entry["shelf"] = float(entry["density"][3] / max(entry["density"][5], 1e-6))
        report[reader] = entry
    return report


def main(cache, out, paths):
    started = provenance.start()
    base = pickle.load(open(cache, "rb"))["rows"]
    with ProcessPoolExecutor(JOBS) as pool:
        planar = list(pool.map(field_row, sorted(paths)[: LIMIT or None], chunksize=1))
    rows = base + [r for r in planar if r]
    where = place.sources(rows)
    tasks, seen = [], {}
    for row in rows:
        name = next((n for n in ("real", "sfinge", "model2") if parts.member(row, n)), "field" if row["set"] == "field" else None)
        if name is None or id(row) not in where or (LIMIT and seen.get(name, 0) >= LIMIT):
            continue
        seen[name] = seen.get(name, 0) + 1
        ROWS.append(row)
        tasks.append((name, len(ROWS) - 1, where[id(row)]))
    with ProcessPoolExecutor(JOBS) as pool:
        read = list(pool.map(one, [t[1:] for t in tasks], chunksize=2))
    rng = np.random.default_rng(0)
    report = {"seen": seen}
    for name in seen:
        got = [r for t, r in zip(tasks, read) if t[0] == name]
        kept = [r for r in got if "cells" in r]
        report[name] = {"read": len(got), "with_core": len(kept), "all": group(kept, rng)}
        for label, wanted in (("one_core", 1), ("two_cores", 2)):
            subset = [r for r in kept if r["cores"] == wanted]
            if len(subset) >= 8:
                report[name][label] = group(subset, rng)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
