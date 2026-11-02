"""Does ridge direction turn round a delta evenly, or in three steps? (D-024): delta_knees.py OUT.json RUN.npz...

Round a delta, on a circle, direction turns through 180°. A zero-pole formula flow turns it evenly with angle; a three-knee delta (Ercolani et al. 2000) in three steps. S is the share of the turning in the three strongest 30° windows, kept 60° apart: 0.25 if even. Deltas are found by Poincaré index, their circles must lie inside the print,
and a circle must wind 180° within `TOLERANCE`, so a nearby core spoils nothing. For each step, its turning and its width in px at half height.

Run: python -m signed_charge.exploratory.delta_knees OUT.json RUN.npz...
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
from signed_charge.measure import screen
from signed_charge.exploratory import seams as sm

RADII = (30, 45, 60, 80)
SAMPLES, WINDOW, GAP, TOLERANCE = 180, 30, 60, 0.6
JOBS = int(os.environ.get("KNEE_JOBS", 1))
SHARE, BORDER = 3, 16


def turning(z, x, y, radius):
    """The double-angle's steps round a circle, in radians of ridge direction, or None if it leaves the array."""
    phi = np.linspace(0, 2 * np.pi, SAMPLES, endpoint=False)
    px, py = np.round(x + radius * np.cos(phi)).astype(int), np.round(y + radius * np.sin(phi)).astype(int)
    h, w = z.shape
    if px.min() < 0 or py.min() < 0 or px.max() >= w or py.max() >= h:
        return None, px, py
    values = z[py, px]
    return np.angle(np.roll(values, -1) * np.conj(values)) / 2, px, py


def stairs(step):
    """S, and each of the three steps' turning (degrees) and width (samples at half height). Signed, so noise cancels."""
    total = abs(step.sum())
    window = int(WINDOW / 360 * SAMPLES)
    sign = np.sign(step.sum())
    kernel = np.zeros(SAMPLES)
    kernel[:window] = 1
    share = np.real(np.fft.ifft(np.fft.fft(sign * step) * np.conj(np.fft.fft(kernel))))
    centres, used, s = [], np.ones(SAMPLES, bool), 0.0
    for _ in range(3):
        i = int(np.argmax(np.where(used, share, -np.inf)))
        s += share[i] / total
        centres.append((i + window // 2) % SAMPLES)
        gap = int(GAP / 360 * SAMPLES)
        for k in range(-gap, gap + 1):
            used[(i + k) % SAMPLES] = False
    smooth = np.real(np.fft.ifft(np.fft.fft(sign * step) * np.fft.fft(np.roll(np.r_[np.ones(7) / 7, np.zeros(SAMPLES - 7)], -3))))
    steps = []
    for c in centres:
        peak = max(smooth[(c + k) % SAMPLES] for k in range(-window // 2, window // 2 + 1))
        left = right = 0
        while left < SAMPLES // 6 and smooth[(c - left) % SAMPLES] > peak / 2:
            left += 1
        while right < SAMPLES // 6 and smooth[(c + right) % SAMPLES] > peak / 2:
            right += 1
        steps.append((float(np.degrees(sum(sign * step[(c + k) % SAMPLES] for k in range(-SAMPLES // 12, SAMPLES // 12)))), left + right))
    return float(s), steps


def deltas(z):
    coarse = z[sm.CELL // 2 :: sm.CELL, sm.CELL // 2 :: sm.CELL]
    step = lambda a, b: np.angle(b * np.conj(a))
    wind = step(coarse[:-1, :-1], coarse[:-1, 1:]) + step(coarse[:-1, 1:], coarse[1:, 1:]) + step(coarse[1:, 1:], coarse[1:, :-1]) + step(coarse[1:, :-1], coarse[:-1, :-1])
    return wind


def measure(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, []
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * BORDER + 1,) * 2, np.uint8)).astype(bool)
    z = sm.double_angle(im, fg)
    wind = deltas(z)
    found, rows = [], []
    for i, j in zip(*np.nonzero(np.abs(wind) > np.pi)):
        x, y = (j + 1) * sm.CELL, (i + 1) * sm.CELL
        if any(np.hypot(x - a, y - b) < 20 for a, b in found):
            continue
        found.append((x, y))
        for radius in RADII:
            step, px, py = turning(z, x, y, radius)
            if step is None or not inner[py, px].all():
                continue
            total = step.sum()
            if abs(abs(total) - np.pi) > TOLERANCE:
                continue
            s, steps = stairs(step)
            rows.append({"sign": int(np.sign(total)), "radius": radius, "s": s, "steps": [(a, w * 2 * np.pi * radius / SAMPLES) for a, w in steps]})
    return key, rows


def rendered(task):
    return measure(fc.image(task))


def developmental(path):
    name = Path(path).parent.name + "_" + Path(path).stem.split("_")[0]
    return measure(((name, Path(path).stem, 1), screen.print_of(np.load(path)["u"])))


def control(kind, smooth, rng):
    """The measure on a field made to order: even turning, or three tanh steps, with noise."""
    size = 301
    yy, xx = np.mgrid[0:size, 0:size].astype(float) - size // 2
    phi = np.arctan2(yy, xx)
    if kind == "even":
        theta = -phi / 2
    else:
        theta = -np.sum([(1 + np.tanh((phi - c) * smooth)) / 2 for c in (-2.094, 0, 2.094)], 0) * np.pi / 3  # three steps of 60°, winding 180°
    z = np.exp(2j * theta) + 0.05 * (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
    return z / np.abs(z)


def main(out, runs):
    started = provenance.start()
    rng = np.random.default_rng(0)
    report = {"controls": {}}
    for kind, smooth in (("even", 0), ("steps", 3), ("steps", 8)):
        z = control(kind, smooth, rng)
        step, _, _ = turning(z, 150, 150, 60)
        report["controls"][f"{kind}{smooth}"] = stairs(step)[0]
    print(report["controls"], flush=True)
    real = [x for i, x in enumerate(cf.fvc(("DB1", "DB2", "DB3"))) if i % SHARE == 0]
    sfinge = [x for i, x in enumerate(cf.fvc(("DB4",))) if i % SHARE == 0]
    tasks = [("model2", s, f, 1) for s in range(7000, 7010) for f in (2, 3, 4, 5, 7, 8, 9, 10)]
    sets = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for k, rows in pool.map(measure, real, chunksize=4):
            sets.setdefault("real", []).extend(rows)
        for k, rows in pool.map(measure, sfinge, chunksize=4):
            sets.setdefault("sfinge", []).extend(rows)
        for k, rows in pool.map(rendered, tasks, chunksize=2):
            sets.setdefault("model2", []).extend(rows)
        for k, rows in pool.map(developmental, runs, chunksize=2):
            sets.setdefault(k[0], []).extend(rows)
    for name, rows in sets.items():
        # A delta winds opposite to a core; which sign that is here is read from the control's sign.
        for sign in (-1, 1):
            mine = [r for r in rows if r["sign"] == sign]
            by = {str(R): [r["s"] for r in mine if r["radius"] == R] for R in RADII}
            steps = [st for r in mine for st in r["steps"]]
            report[f"{name}_{sign}"] = {"circles": len(mine), "s": {R: (round(float(np.median(v)), 3), len(v)) for R, v in by.items() if v}}
            if steps:
                a = np.abs(np.array(steps))
                big = a[:, 0] > 20
                report[f"{name}_{sign}"].update(step_turn=float(np.median(a[:, 0])), step_width=float(np.median(a[:, 1])), width_sin=float(np.median((a[:, 1] * np.sin(np.radians(a[:, 0]) / 2))[big])) if big.any() else None)
            print(f"{name}_{sign}".ljust(26), json.dumps(report[f"{name}_{sign}"]), flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
