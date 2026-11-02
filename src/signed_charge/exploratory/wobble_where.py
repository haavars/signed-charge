"""Where ridges wobble: O-003's reading, tested (E-048): wobble_where.py OUT.json RUN.npz...

Real prints, by the author's eye, have smooth lines with wobble only where lines meet; developmental prints wobble everywhere; Phantom's are uniform. As `wobble.py`, but the direction gap at each ridge pixel is kept, not averaged: its median, the smooth stretches; its 90th percentile; the same by
distance from the nearest minutia; and how concentrated it is, the share of the print's squared gap in its worst 10 % of 32 px tiles, 10 % if wobble is evenly spread. Medians over prints.

Run: python -m signed_charge.exploratory.wobble_where OUT.json RUN.npz...
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_check as fc
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.exploratory import wobble as wb

JOBS = int(os.environ.get("WOBBLE_JOBS", 2))
RINGS = ((0, 14), (14, 28), (28, 56), (56, 999))
TILE, SHARE = 32, 4


def measure(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, None
    points, ridges = mp.mindtct(im)
    if len(points) < 5:
        return key, None
    b = cv2.GaussianBlur(ridges.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * wb.BORDER + 1,) * 2, np.uint8)).astype(bool)
    edge = inner & (np.hypot(gx, gy) > 0.05)
    if edge.sum() < 3000:
        return key, None
    fine, coarse = wb.direction(gx, gy, wb.FINE), wb.direction(gx, gy, wb.COARSE)
    gap = np.degrees(np.abs(0.5 * np.angle(np.exp(2j * (fine - coarse)))))
    mask = np.ones(im.shape, np.uint8)
    for x, y in points[:, :2].astype(int):
        if 0 <= y < im.shape[0] and 0 <= x < im.shape[1]:
            mask[y, x] = 0
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    out = {"median": float(np.median(gap[edge])), "p90": float(np.percentile(gap[edge], 90))}
    for low, high in RINGS:
        ring = edge & (distance >= low) & (distance < high)
        out[f"ring{low}"] = float(np.median(gap[ring])) if ring.sum() > 300 else None
    h, w = im.shape
    tiles = []
    for y in range(0, h - TILE, TILE):
        for x in range(0, w - TILE, TILE):
            m = edge[y : y + TILE, x : x + TILE]
            if m.sum() > 60:
                tiles.append(float(np.mean(gap[y : y + TILE, x : x + TILE][m] ** 2)))
    tiles = np.sort(tiles)[::-1]
    out["concentration"] = float(tiles[: max(len(tiles) // 10, 1)].sum() / tiles.sum()) if len(tiles) > 20 else None
    return key, out


def rendered(task):
    return measure(fc.image(task))


def developmental(path):
    name = Path(path).parent.name + "_" + Path(path).stem.split("_s")[0]
    return measure(((name, Path(path).stem, 1), screen.print_of(np.load(path)["u"])))


def main(out, runs):
    started = provenance.start()
    real = [x for i, x in enumerate(cf.fvc(("DB1", "DB2", "DB3"))) if i % SHARE == 0]
    tasks = [("model2", s, f, 1) for s in range(7000, 7006) for f in (2, 3, 4, 5, 7, 8, 9, 10)]
    sets = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for k, v in pool.map(measure, real, chunksize=4):
            if v:
                sets.setdefault("real", []).append(v)
                sets.setdefault(k[0], []).append(v)
        for k, v in pool.map(rendered, tasks, chunksize=2):
            if v:
                sets.setdefault("model2", []).append(v)
        for k, v in pool.map(developmental, runs, chunksize=2):
            if v:
                sets.setdefault(k[0], []).append(v)
    report = {}
    for name, rows in sets.items():
        report[name] = {"prints": len(rows)}
        for key in rows[0]:
            xs = [r[key] for r in rows if r[key] is not None]
            report[name][key] = float(np.median(xs)) if xs else None
        print(name.ljust(30), {k: round(v, 2) if isinstance(v, float) else v for k, v in report[name].items()}, flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
