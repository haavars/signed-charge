"""Does the sign along a ridge hold sensor by sensor, and with no extractor? (D-077): ridge_robust.py OUT.json

`ridge_frame`'s h on FVC DB1-3 B by sensor, by mindtct and LEADER, and on real, SFinGe and model 2 by phase winding with opposite pairs under 15 px cancelled, and by `shared` residues, those within 8 px of a mindtct minutia, an exploratory addition. CI by finger.

Run: python -m signed_charge.reads.ridge_robust OUT.json
"""

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.spatial import cKDTree

from signed_charge.measure import ridge_frame as rf

w, fs, sc, sf, sm, parts, place = rf.w, rf.fs, rf.sc, rf.sf, rf.sm, rf.parts, rf.place


SHARED_ONLY = len(sys.argv) > 2


def one(task):
    index, source, name = task
    row = rf.ROWS[index]
    im = place.image(source)
    inner = parts.mask_of(row)
    fg = fs.foreground(im)
    out = rf.read(im, row["points"], inner, row["finger"])
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    mask = inner.copy()
    for x, y in sc.singular(fs.field(im), fg):
        mask &= np.hypot(xx - x, yy - y) >= sc.GUARD
    z2 = sm.direction(im)
    z2 = z2 / (np.abs(z2) + 1e-9)
    b, V = w.demodulate(im, fg, z2)
    amp = cv2.GaussianBlur(np.abs(b + 1j * np.imag(np.exp(-0.5j * np.angle(z2)) * V)).astype(np.float32), (0, 0), 4)
    strong = mask & (amp > w.AMPLITUDE * np.median(amp[mask]))
    rx, ry, rq, rb = w.residues(b, V, z2, strong)
    heading = w.heading_of(rq, rb)
    keep, _ = w.cancel(rx, ry, heading, w.DIPOLE_FAR)
    xy, hd = np.c_[rx, ry][keep], heading[keep]
    heads = np.stack([np.cos(hd), np.sin(hd)], 1)
    count, product = rf.pairs(xy, heads, z2, mask) if len(xy) > 1 else (np.zeros((4, 3)), np.zeros((4, 3)))
    out["winding"] = (count, product, np.ones_like(count))
    near = sf.inside(row["points"]["mindtct"], mask)
    shared = cKDTree(near[:, :2]).query(xy, distance_upper_bound=w.MATCH)[0] < w.MATCH if len(near) and len(xy) else np.zeros(len(xy), bool)
    count, product = rf.pairs(xy[shared], heads[shared], z2, mask) if shared.sum() > 1 else (np.zeros((4, 3)), np.zeros((4, 3)))
    out["shared"] = (count, product, np.ones_like(count))
    out["set"], out["name"] = row["set"], name
    return out


def summary(rows, readers, rng):
    fingers = sorted({r["finger"] for r in rows})
    at = {f: i for i, f in enumerate(fingers)}
    out = {"fingers": len(fingers)}
    for reader in readers:
        table = np.zeros((len(fingers), 2, 4, 3))
        for r in rows:
            table[at[r["finger"]]] += np.stack(r[reader][:2])
        stat = lambda i: table[i, 1].sum(0) / np.maximum(table[i, 0].sum(0), 1)  # noqa: E731
        draws = np.array([stat(rng.integers(len(fingers), size=len(fingers))) for _ in range(rf.RESAMPLES)])
        h = stat(np.arange(len(fingers)))
        low, high = np.percentile(draws, (2.5, 97.5), axis=0)
        out[reader] = {"pairs": table[:, 0].sum(0)[:, 0].tolist(), "h": h[:, 0].tolist(), "low": low[:, 0].tolist(), "high": high[:, 0].tolist()}
    return out


def main(out):
    rows = pickle.load(open(w.CACHE, "rb"))["rows"]
    where = place.sources(rows)
    tasks = []
    for row in rows:
        name = next((n for n in ("real", "sfinge", "model2") if parts.member(row, n)), None)
        if name and id(row) in where:
            rf.ROWS.append(row)
            tasks.append((len(rf.ROWS) - 1, where[id(row)], name))
    with ProcessPoolExecutor(rf.JOBS) as pool:
        got = list(pool.map(one, tasks, chunksize=2))
    rng = np.random.default_rng(0)
    report = {"along": rf.ALONG, "sets": {}, "sensors": {}}
    for name in ("real", "sfinge", "model2"):
        report["sets"][name] = summary([r for r in got if r["name"] == name], ("winding", "shared", "mindtct", "leader"), rng)
    for sensor in sorted({r["set"] for r in got if r["name"] == "real"}) if not SHARED_ONLY else []:
        report["sensors"][sensor] = summary([r for r in got if r["set"] == sensor and r["name"] == "real"], ("mindtct", "leader"), rng)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
