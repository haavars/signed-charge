"""D-068 step 2: minutiae pairs in the ridges' own frame (D-072): ridge_frame.py OUT.json [SETS...]

For pairs under `REACH` px whose segment stays on the print, away from singular points, the separation split along and across the ridge direction at the pair's middle, lifted as `signed_minutiae.charge_pairs`:
    g(u, v)   pairs in a bin against as many uniform random points in the same mask, `DRAWS` draws: 1 for no order
    h(u, v)   mean product of the two lifted signs: 0 for independent signs
`ACROSS` bins are the same ridge, the next and the next but one, at 9 px a ridge; `ALONG` runs out to 5 mm. Each pair once. CI by finger.

Run: python -m signed_charge.measure.ridge_frame OUT.json [SETS...]
LIMIT=n takes the first n prints of each set.
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

from signed_charge.reads import field_read as fr
from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge.measure import winding as w

JOBS = int(os.environ.get("READ_JOBS", 8))
LIMIT = int(os.environ.get("LIMIT", 0))
ALONG, ACROSS, REACH, DRAWS, RESAMPLES = (0, 15, 30, 60, 100), (0, 4.5, 13.5, 22.5), 100.0, 5, 1000
FLOW_DIR = {"realflow": ("realflow_*.npz", w.FLOW), "readflow": ("readflow_*.npz", w.FLOW), "lead": ("lead_*.npz", Path(w.FLOW).parent / "lead_d067")}
OUTSIDE = Path(w.FLOW).parent / "outside_d066"
ROWS = []


def pairs(xy, heads, field, mask):
    """Counts and summed sign products in (along, across) bins."""
    count, product = np.zeros((len(ALONG) - 1, len(ACROSS) - 1)), np.zeros((len(ALONG) - 1, len(ACROSS) - 1))
    h, w_ = field.shape
    for i in range(len(xy) - 1):
        d = np.hypot(*(xy[i + 1 :] - xy[i]).T)
        for j in np.nonzero(d < REACH)[0] + i + 1:
            n = max(int(d[j - i - 1] / 3), 4)
            t = np.linspace(0, 1, n)
            x = np.clip(np.round(xy[i, 0] + t * (xy[j, 0] - xy[i, 0])).astype(int), 0, w_ - 1)
            y = np.clip(np.round(xy[i, 1] + t * (xy[j, 1] - xy[i, 1])).astype(int), 0, h - 1)
            if not mask[y, x].all():
                continue
            ridge = np.unwrap(np.angle(field[y, x])) / 2 + np.pi / 2
            middle = ridge[n // 2]
            dx, dy = xy[j] - xy[i]
            u, v = abs(dx * np.cos(middle) + dy * np.sin(middle)), abs(-dx * np.sin(middle) + dy * np.cos(middle))
            a, b = np.searchsorted(ALONG, u, side="right") - 1, np.searchsorted(ACROSS, v, side="right") - 1
            if a >= len(ALONG) - 1 or b >= len(ACROSS) - 1:
                continue
            count[a, b] += 1
            if heads is not None:
                ends = np.stack([np.cos(ridge[[0, -1]]), np.sin(ridge[[0, -1]])], 1)
                product[a, b] += np.sign(heads[i] @ ends[0]) * np.sign(heads[j] @ ends[1])
    return count, product


def read(im, points, inner, finger):
    fg = fs.foreground(im)
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    mask = inner.copy()
    for x, y in sc.singular(fs.field(im), fg):
        mask &= np.hypot(xx - x, yy - y) >= sc.GUARD
    field = sm.direction(im)
    rng = np.random.default_rng(zlib.crc32(finger.encode()))
    ys, xs = np.nonzero(mask)
    out = {"finger": finger}
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(points[reader], mask)
        heads = np.stack([np.cos(p[:, 2]), y_sign * np.sin(p[:, 2])], 1) if len(p) else np.zeros((0, 2))
        count, product = pairs(p[:, :2], heads, field, mask) if len(p) > 1 else (np.zeros((4, 3)), np.zeros((4, 3)))
        null = np.zeros_like(count)
        for _ in range(DRAWS):
            pick = rng.integers(len(xs), size=len(p))
            if len(p) > 1:
                null += pairs(np.c_[xs[pick], ys[pick]].astype(float), None, field, mask)[0] / DRAWS
        out[reader] = (count, product, null)
    return out


def task(t):
    kind, payload = t
    if kind == "cache":
        index, source = payload
        row = ROWS[index]
        return read(place.image(source), row["points"], parts.mask_of(row), row["finger"])
    im = fr.sheet(payload) if kind == "flow" else cv2.imread(payload, 0)
    inner = w.inner_of(im)
    return read(im, w.fresh_points(im, inner), inner, Path(payload).stem)


def summary(rows, rng):
    fingers = sorted({r["finger"] for r in rows})
    at = {f: i for i, f in enumerate(fingers)}
    out = {"prints": len(rows), "fingers": len(fingers)}
    for reader in sm.EXTRACTORS:
        tables = np.zeros((len(fingers), 3, len(ALONG) - 1, len(ACROSS) - 1))
        for r in rows:
            tables[at[r["finger"]]] += np.stack(r[reader])
        def stats(idx):
            c, p, n = tables[idx].sum(0)
            return c / np.maximum(n, 1), p / np.maximum(c, 1)
        g, h = stats(np.arange(len(fingers)))
        draws = [stats(rng.integers(len(fingers), size=len(fingers))) for _ in range(RESAMPLES)]
        out[reader] = {"g": g.tolist(), "h": h.tolist(), "pairs": tables[:, 0].sum(0).tolist(), "g_ci": np.percentile([d[0] for d in draws], (2.5, 97.5), axis=0).tolist(), "h_ci": np.percentile([d[1] for d in draws], (2.5, 97.5), axis=0).tolist()}
    return out


def main(out, names):
    started = provenance.start()
    names = names or ["real", "sfinge", "model2", "realflow", "readflow", "lead"]
    tasks = []
    if set(names) & {"real", "sfinge", "model2"}:
        rows = pickle.load(open(w.CACHE, "rb"))["rows"]
        where = place.sources(rows)
        seen = {}
        for row in rows:
            name = next((n for n in names if n in ("real", "sfinge", "model2") and parts.member(row, n)), None)
            if name and id(row) in where and (not LIMIT or seen.get(name, 0) < LIMIT):
                seen[name] = seen.get(name, 0) + 1
                ROWS.append(row)
                tasks.append((name, ("cache", (len(ROWS) - 1, where[id(row)]))))
    for name in names:
        if name in FLOW_DIR:
            pattern, folder = FLOW_DIR[name]
            tasks += [(name, ("flow", str(p))) for p in sorted(folder.glob(pattern))[: LIMIT or None]]
    for name in names:
        if name in ("synfing", "anguli"):
            tasks += [(name, ("image", str(p))) for p in sorted(OUTSIDE.glob(name + "_*"))[: LIMIT or None]]
    with ProcessPoolExecutor(JOBS) as pool:
        got = list(pool.map(task, [t[1] for t in tasks], chunksize=2))
    rng = np.random.default_rng(0)
    report = {"along": ALONG, "across": ACROSS}
    for name in names:
        report[name] = summary([r for t, r in zip(tasks, got) if t[0] == name], rng)
        print(name, report[name]["prints"], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
