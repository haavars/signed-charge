"""Do minutiae pay for the flow's fan-out? (E-042): who_pays.py OUT.json [RUN.npz...]

At each minutia, its charge s against the sign of the splay div F at `SCALE`, six periods. Both flip with F, so their product needs no lift. P = mean s sign(div F): 1 if every minutia sits where the flow
spreads and adds a ridge, 0 if charge ignores the flow. P_strong is the same where |div F| is in the print's top third, where a demand is plainest. A minutia bends the field a little around itself, which would make P
positive with no flow at all: planted independent spirals on straight stripes measure that, and `aligned`, a print with + spirals on one side and - on the other, measures the sign.

Run: python -m signed_charge.reads.who_pays OUT.json [RUN.npz...]
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("FANOUT_SCALE", "54")
from signed_charge.measure import conformality as cf
from signed_charge.measure import fanout as fo
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_check as fc
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("PAYS_JOBS", 1))
RESAMPLES, SHARE = 1000, 3


def measure(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, None
    z, _, _ = fo.fields(im, fg)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * fo.EDGE + 1,) * 2, np.uint8)).astype(bool)
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    for x, y in fs.cores(fs.field(im), fg):
        inner &= np.hypot(xx - x, yy - y) > fo.CLEAR
    if inner.sum() < 8000:
        return key, None
    zx, zy = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) / 2, (np.roll(z, -1, 0) - np.roll(z, 1, 0)) / 2
    tx, ty = np.imag(np.conj(z) * zx) / 2, np.imag(np.conj(z) * zy) / 2
    theta = np.angle(z) / 2
    cos, sin = np.cos(theta), np.sin(theta)
    splay = -sin * tx + cos * ty  # with F = (cos θ, sin θ)
    strong = np.percentile(np.abs(splay[inner]), 67)
    out = {}
    for name, (extract, y_sign) in sm.EXTRACTORS.items():
        points = extract(im)
        if len(points):
            xi, yi = points[:, 0].astype(int), points[:, 1].astype(int)
            ok = (xi >= 0) & (xi < im.shape[1]) & (yi >= 0) & (yi < im.shape[0])
            ok[ok] &= inner[yi[ok], xi[ok]]
            points = points[ok]
        if len(points) < 3:
            out[name] = None
            continue
        xi, yi = points[:, 0].astype(int), points[:, 1].astype(int)
        s = np.sign(np.cos(points[:, 2]) * cos[yi, xi] + y_sign * np.sin(points[:, 2]) * sin[yi, xi])
        d = splay[yi, xi]
        hard = np.abs(d) > strong
        out[name] = {"sum": float(np.sum(s * np.sign(d))), "n": len(s), "strong_sum": float(np.sum((s * np.sign(d))[hard])), "strong_n": int(hard.sum())}
    return key, out


def planted(task):
    kind, seed = task
    rng = np.random.default_rng(seed)
    h, w = sm.SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rot = rng.uniform(0, np.pi)
    v = -(xx - w / 2) * np.sin(rot) + (yy - h / 2) * np.cos(rot)
    u = (xx - w / 2) * np.cos(rot) + (yy - h / 2) * np.sin(rot)
    psi = 2 * np.pi * v / sm.PERIOD + rng.uniform(0, 2 * np.pi)
    for _ in range(26):
        cx, cy = rng.uniform(40, w - 40), rng.uniform(40, h - 40)
        along = (cx - w / 2) * np.cos(rot) + (cy - h / 2) * np.sin(rot)
        p = rng.choice([-1, 1]) if kind == "independent" else (1 if along > 0 else -1)
        psi += p * np.arctan2(yy - cy, xx - cx)
    return measure(((kind, seed, 1), np.where(np.cos(psi) > 0, 40, 235).astype(np.uint8)))


def rendered(task):
    return measure(fc.image(task))


def developmental(path):
    import re

    from signed_charge.measure import screen

    name = Path(path).stem
    pattern, speed = re.fullmatch(r"(\w+?)_a([\d.]+)_s[\d.]+_\d+", name).groups()
    return measure(((f"{pattern}_a{speed}", name, 1), screen.print_of(np.load(path)["u"])))


def score(rows, rng):
    report = {}
    for name in sm.EXTRACTORS:
        xs = [r[name] for r in rows if r[name]]
        total, n = np.array([x["sum"] for x in xs]), np.array([x["n"] for x in xs], float)
        strong, sn = np.array([x["strong_sum"] for x in xs]), np.array([x["strong_n"] for x in xs], float)
        draws = [rng.integers(len(xs), size=len(xs)) for _ in range(RESAMPLES)]
        p = [total[d].sum() / n[d].sum() for d in draws]
        report[name] = {"prints": len(xs), "P": float(total.sum() / n.sum()), "ci": [float(np.percentile(p, q)) for q in (2.5, 97.5)],
                        "P_strong": float(strong.sum() / max(sn.sum(), 1))}
    return report


def main(out, runs):
    started = provenance.start()
    rng = np.random.default_rng(0)
    real = [x for i, x in enumerate(cf.fvc(("DB1", "DB2", "DB3"))) if i % SHARE == 0]
    sfinge = [x for i, x in enumerate(cf.fvc(("DB4",))) if i % SHARE == 0]
    tasks = [(v, s, f, 1) for v in sm.VARIANTS for s in range(7000, 7020) for f in (2, 3, 4, 5, 7, 8, 9, 10)]
    sets = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for k, v in pool.map(planted, [(kind, s) for kind in ("independent", "aligned") for s in range(20)], chunksize=2):
            if v:
                sets.setdefault(k[0], []).append(v)
        print({k: score(v, rng) for k, v in sets.items()}, flush=True)
        for k, v in pool.map(measure, real, chunksize=4):
            if v:
                sets.setdefault("real", []).append(v)
        for k, v in pool.map(measure, sfinge, chunksize=4):
            if v:
                sets.setdefault("sfinge", []).append(v)
        for k, v in pool.map(rendered, tasks, chunksize=2):
            if v:
                sets.setdefault(k[0], []).append(v)
        for k, v in pool.map(developmental, runs, chunksize=2):
            if v:
                sets.setdefault(k[0], []).append(v)
    report = {k: score(v, rng) for k, v in sets.items()}
    for name, r in report.items():
        print(name.ljust(14), {x: (round(y["P"], 3), [round(q, 3) for q in y["ci"]], round(y["P_strong"], 3)) for x, y in r.items()}, flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
