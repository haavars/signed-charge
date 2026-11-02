"""How much does each set's flow fan out? (E-042): fanout.py OUT.json [RUN.npz...]

Splay |div F|, ridges spreading or closing up, read the same way from every image: ridge direction from a structure tensor at `SCALE`, two periods, and the
splay of the director from its double angle, so no lift. Bend, curvature along the ridges, beside it: splay needs minutiae or spacing to pay for it, bend doesn't. Inside the
print's foreground, `EDGE` px in, away from singular points, where both diverge. Multiplied by the print's own ridge period, as ridges of fan-out per ridge length: at fixed
spacing a generator must place splay / period minutiae per px². Also the print's spread of ln period at the same scale, on a band-passed image (E-042).

Run: python -m signed_charge.measure.fanout OUT.json [RUN.npz...]
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
from signed_charge.measure import provenance
from signed_charge.measure import signed_minutiae as sm

SCALE, EDGE, CLEAR, MM2 = float(os.environ.get("FANOUT_SCALE", 18)), 24, 45, (25.4 / 500) ** 2
JOBS = int(os.environ.get("FANOUT_JOBS", 8))
VARIANTS = ("model2", "d_s1_sp2_f3")
SHARE = 3  # every third real print, enough for a median, and light


def bandpassed(im, fg):
    a = im.astype(np.float32)
    a -= a[fg].mean()
    return cv2.GaussianBlur(a, (0, 0), 1.2) - cv2.GaussianBlur(a, (0, 0), 6.0)


def fields(im, fg):
    """Unit double angle of the ridges and the local period."""
    b = bandpassed(im, fg)
    gx, gy = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3) / 8
    smooth = lambda a: cv2.GaussianBlur(a, (0, 0), SCALE)
    z = -(smooth(gx * gx - gy * gy) + 2j * smooth(gx * gy))  # the gradient's double angle turned 90°: along the ridges
    coherence = np.abs(z) / (smooth(gx * gx + gy * gy) + 1e-6)
    freq = np.sqrt(smooth(gx * gx + gy * gy) / (smooth(b * b) + 1e-6)) / (2 * np.pi)
    return z / (np.abs(z) + 1e-9), coherence, 1 / np.maximum(freq, 1e-3)


def measure(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, None
    z, coherence, period = fields(im, fg)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * EDGE + 1,) * 2, np.uint8)).astype(bool)
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    for x, y in fs.cores(fs.field(im), fg):
        inner &= np.hypot(xx - x, yy - y) > CLEAR
    if inner.sum() < 8000:
        return key, None
    zx = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) / 2
    zy = (np.roll(z, -1, 0) - np.roll(z, 1, 0)) / 2
    tx, ty = np.imag(np.conj(z) * zx) / 2, np.imag(np.conj(z) * zy) / 2  # ∇θ
    theta = np.angle(z) / 2
    cos, sin = np.cos(theta), np.sin(theta)
    splay = np.abs(-sin * tx + cos * ty)[inner]
    bend = np.abs(cos * tx + sin * ty)[inner]
    p = float(np.median(period[inner]))
    ln = np.log(period[inner])
    return key, {"splay_per_ridge": float(np.mean(splay) * p), "splay_p90": float(np.percentile(splay, 90) * p), "bend_per_ridge": float(np.mean(bend) * p), "period": p,
                 "demand_per_100mm2": float(np.mean(splay) / p / MM2 * 100), "ln_period_sd": float(ln.std()), "coherence": float(np.mean(coherence[inner]))}


def rendered(task):
    return measure(fc.image(task))


def developmental(path):
    import re

    from signed_charge.measure import screen

    name = Path(path).stem
    pattern, speed = re.fullmatch(r"(\w+?)_a([\d.]+)_s[\d.]+_\d+", name).groups()
    return measure(((f"{pattern}_a{speed}", name, 1), screen.print_of(np.load(path)["u"])))


def main(out, runs):
    started = provenance.start()
    real = [x for i, x in enumerate(cf.fvc(("DB1", "DB2", "DB3"))) if i % SHARE == 0]
    sfinge = [x for i, x in enumerate(cf.fvc(("DB4",))) if i % SHARE == 0]
    tasks = [(v, s, f, 1) for v in VARIANTS for s in range(7000, 7020) for f in (2, 3, 4, 5, 7, 8, 9, 10)]
    sets = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for k, v in pool.map(measure, real, chunksize=4):
            if v:
                sets.setdefault("real", []).append(v)
                sets.setdefault(k[0], []).append(v)
        for k, v in pool.map(measure, sfinge, chunksize=4):
            if v:
                sets.setdefault("sfinge", []).append(v)
        for k, v in pool.map(rendered, tasks, chunksize=2):
            if v:
                sets.setdefault(k[0], []).append(v)
        for k, v in pool.map(developmental, runs, chunksize=2):
            if v:
                sets.setdefault(k[0], []).append(v)
    report = {name: {"prints": len(rows), **{k: float(np.median([r[k] for r in rows])) for k in rows[0]}} for name, rows in sets.items()}
    for name, r in report.items():
        print(name.ljust(16), r["prints"], {k: round(v, 3) for k, v in r.items() if k != "prints"}, flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
