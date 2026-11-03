"""Seams: lines along which ridge direction jumps (E-051): seams.py OUT.json RUN.npz...

What a person sees and an average doesn't. Direction from a structure tensor at one ridge period; singular points found by Poincaré index on a coarse grid and masked 2 periods round; then, over the rest of the print, the share of area where direction turns faster than `LIMIT` degrees a
period, and the 99th percentile of that turning. A print with smooth flow has almost none; a seam is a line of it. A coherent line is what's wanted, so `longest` is the largest connected piece of such pixels, in px, along its longest side: a seam is long, a minutia's kink isn't.

Run: python -m signed_charge.exploratory.seams OUT.json RUN.npz...
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as nd

from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_check as fc
from signed_charge.measure import provenance
from signed_charge.measure import screen

PERIOD, LIMIT, BORDER, CELL = 9.5, 10.0, 24, 12
JOBS = int(os.environ.get("SEAM_JOBS", 1))
SHARE = 6


def double_angle(im, fg):
    a = im.astype(np.float32)
    a -= a[fg].mean()
    b = cv2.GaussianBlur(a, (0, 0), 1.2) - cv2.GaussianBlur(a, (0, 0), 6.0)
    gx, gy = cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)
    smooth = lambda x: cv2.GaussianBlur(x, (0, 0), PERIOD)
    z = smooth(gx * gx - gy * gy) + 2j * smooth(gx * gy)
    return z / (np.abs(z) + 1e-9)


def singular(z, inner):
    """Cells round which the double angle winds, as a mask."""
    coarse = z[CELL // 2 :: CELL, CELL // 2 :: CELL]
    step = lambda a, b: np.angle(b * np.conj(a))
    wind = (
        step(coarse[:-1, :-1], coarse[:-1, 1:])
        + step(coarse[:-1, 1:], coarse[1:, 1:])
        + step(coarse[1:, 1:], coarse[1:, :-1])
        + step(coarse[1:, :-1], coarse[:-1, :-1])
    )
    mask = np.zeros(z.shape, np.uint8)
    for i, j in zip(*np.nonzero(np.abs(wind) > np.pi)):
        cv2.circle(mask, (int((j + 1) * CELL), int((i + 1) * CELL)), int(2 * PERIOD), 1, -1)
    return mask > 0


def measure(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, None
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * BORDER + 1,) * 2, np.uint8)).astype(bool)
    if inner.sum() < 8000:
        return key, None
    z = double_angle(im, fg)
    keep = inner & ~singular(z, inner)
    zx, zy = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) / 2, (np.roll(z, -1, 0) - np.roll(z, 1, 0)) / 2
    turn = np.degrees(np.hypot(np.imag(np.conj(z) * zx), np.imag(np.conj(z) * zy)) / 2) * PERIOD
    fast = keep & (turn > LIMIT)
    labels, n = nd.label(fast, np.ones((3, 3)))
    longest = max([max(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in nd.find_objects(labels)], default=0)
    return key, {
        "share": float(fast.sum() / keep.sum()),
        "p99": float(np.percentile(turn[keep], 99)),
        "longest": int(longest),
        "turn": float(np.median(turn[keep])),
    }


def rendered(task):
    return measure(fc.image(task))


def developmental(path):
    name = Path(path).parent.name + "_" + Path(path).stem.split("_")[0]
    return measure(((name, Path(path).stem, 1), screen.print_of(np.load(path)["u"])))


def main(out, runs):
    started = provenance.start()
    real = [x for i, x in enumerate(cf.fvc(("DB1", "DB2", "DB3"))) if i % SHARE == 0]
    tasks = [("model2", s, f, 1) for s in range(7000, 7004) for f in (2, 3, 4, 5, 7, 8, 9, 10)]
    sets = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for k, v in pool.map(measure, real, chunksize=4):
            if v:
                sets.setdefault("real", []).append(v)
        for k, v in pool.map(rendered, tasks, chunksize=2):
            if v:
                sets.setdefault("model2", []).append(v)
        for k, v in pool.map(developmental, runs, chunksize=2):
            if v:
                sets.setdefault(k[0], []).append(v)
    report = {k: {"prints": len(v), **{m: float(np.median([x[m] for x in v])) for m in v[0]}} for k, v in sets.items()}
    for k, v in report.items():
        print(k.ljust(24), {m: round(x, 3) for m, x in v.items()}, flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
