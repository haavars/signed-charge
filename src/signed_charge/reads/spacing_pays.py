"""Does ridge spacing pay for spreading at large scale? (D-059): spacing_pays.py CACHE.pkl OUT.json RUN.npz...

The demand is Φ(d + t), d = div k and t = k · ∇ ln Φ (Wieditz et al. 2022), so if spacing pays, t = -d. Payment is -⟨dt⟩/⟨d²⟩ over a finger's cells, at each of `SCALES` px, with the sizes and what's left of d's variance in d + t. CI by finger.

Run: python -m signed_charge.reads.spacing_pays CACHE.pkl OUT.json RUN.npz...
LIMIT=n takes the first n prints of each set, to try the code.
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from signed_charge.reads import agreement_scales as ass
from signed_charge.reads import agreement_total as at
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_charge as fc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 4))
LIMIT = int(os.environ.get("LIMIT", 0))
SCALES, SETS, RESAMPLES = (50, 100, 150), ("real", "sfinge", "model2"), 2000
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
    out = {"finger": row["finger"]}
    for scale in SCALES:
        angle, d = fc.spreading(field, float(scale), fg)
        t = at.frequency_term(im, fg, angle, float(scale))
        a, b = d[guarded], t[guarded]
        out[scale] = (float(a @ b), float(a @ a), float(((a + b) ** 2).sum()), float(np.abs(a).sum()), float(np.abs(b).sum()), float(a.size))
    return out


def summary(rows, rng):
    fingers = sorted({r["finger"] for r in rows})
    where = {f: i for i, f in enumerate(fingers)}
    draws = [rng.integers(len(fingers), size=len(fingers)) for _ in range(RESAMPLES)]
    report = {"prints": len(rows), "fingers": len(fingers)}
    every = np.arange(len(fingers))
    for scale in SCALES:
        table = np.zeros((len(fingers), 6))
        for r in rows:
            table[where[r["finger"]]] += r[scale]
        with_ci = lambda f: {"mean": float(f(every)), "ci": [float(np.percentile([f(i) for i in draws], q)) for q in (2.5, 97.5)]}  # noqa: E731
        report[scale] = {
            "payment": with_ci(lambda i: -table[i, 0].sum() / max(table[i, 1].sum(), 1e-12)),
            "left": with_ci(lambda i: table[i, 2].sum() / max(table[i, 1].sum(), 1e-12)),
            "size_d": float(table[:, 3].sum() / table[:, 5].sum()),
            "size_t": float(table[:, 4].sum() / table[:, 5].sum()),
        }
    return report


def main(cache, out, paths):
    started = provenance.start()
    base = pickle.load(open(cache, "rb"))["rows"]
    with ProcessPoolExecutor(JOBS) as pool:
        grown = [r for r in pool.map(ass.planar, sorted(paths), chunksize=1) if r]
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
    rng = np.random.default_rng(0)
    report = {name: summary([r for t, r in zip(tasks, read) if t[0] == name], rng) for name in seen}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
