"""What mindtct reads at the pairs a phase-drawn master carries (E-019): pair_readout.py OUT [PEOPLE]

Round 3 (E-017) added ± spiral pairs laid out across the ridges (an ending facing a fork), on a diagonal or along a
ridge, yet fingertips' close pairs face each other 14-20 % of the time against real's 41 %. Our extractor reads an
across pair as facing (E-013); this asks what mindtct reads, on fingertips inked flat as the screens use them:

For each added pair whose two spirals both land at least 16 px inside a tip's foreground: mindtct's minutiae
(reliability ≥ 0.2) within 0.6 of the print's ridge period of each spiral; whether both are found, their types,
their angle difference (> 144° facing, as `minutiae_pairs.py` counts it) and their distance. The same for the
flow's own spirals (singles): found or not. By layout and by the pair's separation in periods.

Masters: model 2's flow, the flow's spirals spaced 2 periods (E-017's best), 3 pairs per 100 mm² of master in equal
shares of the three layouts, so each is well sampled.

Run: python -m signed_charge.exploratory.pair_readout OUT.json [PEOPLE]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.spatial import cKDTree

from signed_charge.generate import minutiae_growth as growth
from signed_charge.measure import minutiae_stats as ms
from signed_charge.generate import phase_global as pg
from signed_charge.measure import provenance
from signed_charge.measure.conformality import fs

RATE, MIX, SPACING, SMOOTH = 3.0, (1 / 3, 1 / 3, 1 / 3), 2.0, 2.0
JOBS = 8


def master_with_pairs(seed, fgp):
    """The master's ridges from one phase, its flow's spirals and its pairs (with their layouts)."""
    from ridgegen import finger as fingers
    from ridgegen.synthesis import rng_for

    m = fingers.master(seed, fgp)
    shape, period = m.ridges.shape, m.period
    singular = m.cores + m.deltas
    barrier = pg.cuts(shape, singular)
    beta_u = pg.unwrap_direction(m.orientation, barrier)
    flow, _, _ = pg.flow_spirals(beta_u, period, barrier, rng_for(seed, fgp, pg.FLOW_SALT), smooth=SMOOTH, spacing=SPACING)
    labels = []
    added = pg.pairs(beta_u, period, flow, singular, RATE, MIX, rng_for(seed, fgp, pg.PAIR_SALT), labels=labels)
    psi_c, psi_s, _ = pg.solve_continuous(beta_u, period, flow + added)
    return m, pg.render(psi_c + psi_s), flow, added, labels


def one(task):
    seed, fgp = task
    from ridgegen import finger as fingers
    from ridgegen import fingerprints
    from ridgegen.synthesis import rng_for

    growth.use("base")
    original = fingers.master
    m, ridges, flow, added, labels = master_with_pairs(seed, fgp)
    drawn = type(m)(**{**m.__dict__, "ridges": ridges, "minutiae": []})
    fingers.master = lambda *a, **k: drawn
    try:
        rng = rng_for(seed, fgp, fingerprints.PLAIN, 0, 13 if fgp <= 5 else 14)
        _, tip, contact, map_x, map_y, pasted = fingerprints.plain_finger(seed, fgp, 0, rng)
    finally:
        fingers.master = original
    im = np.where((contact > 0.5) & ~pasted & (tip > 0), 40, 235).astype(np.uint8)
    fg = fs.foreground(im)
    if fg is None:
        return []
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1,) * 2, np.uint8)).astype(bool)
    found = ms.mindtct(im)

    # Master points to the tip: the nearest canvas pixel whose map lands there (within 1.5 px).
    ys, xs = np.nonzero(contact > 0.5)
    tree = cKDTree(np.stack([map_x[ys, xs], map_y[ys, xs]], 1))
    scale = float(np.median(np.hypot(np.diff(map_x, axis=1), np.diff(map_y, axis=1))[contact[:, 1:] > 0.5]))
    period_tip = m.period / scale  # canvas px

    def to_tip(x, y):
        d, i = tree.query((x, y))
        if d > 1.5:
            return None
        cx, cy = int(xs[i]), int(ys[i])
        return (cx, cy) if inner[cy, cx] else None

    def nearest(pt):
        if not len(found):
            return None
        d = np.hypot(found[:, 0] - pt[0], found[:, 1] - pt[1])
        i = int(np.argmin(d))
        return i if d[i] <= 0.6 * period_tip else None

    rows = []
    for k, layout in enumerate(labels):
        (x1, y1, _), (x2, y2, _) = added[2 * k], added[2 * k + 1]
        a, b = to_tip(x1, y1), to_tip(x2, y2)
        if a is None or b is None:
            continue
        ia, ib = nearest(a), nearest(b)
        row = {
            "kind": "pair",
            "layout": layout,
            "separation": float(np.hypot(x2 - x1, y2 - y1) / m.period),
            "found": int(ia is not None) + int(ib is not None and ib != ia),
        }
        if ia is not None and ib is not None and ia != ib:
            p, q = found[ia], found[ib]
            da = np.degrees(abs((p[2] - q[2] + np.pi) % (2 * np.pi) - np.pi))
            row.update(angle=float(da), distance=float(np.hypot(p[0] - q[0], p[1] - q[1])), types="".join(sorted("B" if t else "E" for t in (p[3], q[3]))))
        rows.append(row)
    for x, y, _ in flow:
        a = to_tip(x, y)
        if a is not None:
            rows.append({"kind": "flow", "found": int(nearest(a) is not None)})
    for row in rows:
        row.update(seed=seed, fgp=fgp)
    return rows


def summary(rows):
    out = {}
    flow = [r for r in rows if r["kind"] == "flow"]
    out["flow_found"] = float(np.mean([r["found"] for r in flow])) if flow else None
    out["flow_n"] = len(flow)
    for layout in pg.LAYOUTS:
        xs = [r for r in rows if r["kind"] == "pair" and r["layout"] == layout]
        both = [r for r in xs if r["found"] == 2]
        out[layout] = {
            "pairs": len(xs),
            "found_0_1_2": np.bincount([r["found"] for r in xs], minlength=3).tolist(),
            "facing_over_144": float(np.mean([r["angle"] > 144 for r in both])) if both else None,
            "angle_median": float(np.median([r["angle"] for r in both])) if both else None,
            "distance_median_px": float(np.median([r["distance"] for r in both])) if both else None,
            "types": {t: sum(r["types"] == t for r in both) for t in ("EE", "BE", "BB")},
        }
    return out


def main(out, people=10):
    started = provenance.start()
    tasks = [(s, f) for s in range(3000, 3000 + people) for f in (2, 3, 4, 5, 7, 8, 9, 10)]
    with ProcessPoolExecutor(JOBS) as pool:
        rows = [r for rs in pool.map(one, tasks) for r in rs]
    report = {"summary": summary(rows), "rows": rows}
    print(json.dumps(report["summary"], indent=1))
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
