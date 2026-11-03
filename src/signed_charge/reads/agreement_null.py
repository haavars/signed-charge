"""What agreement reads when there is no flow to follow, and on a flow known in advance (D-057): agreement_null.py OUT.json

By the divergence theorem a read flow's spreading is the net charge of the minutiae in it, so `flow_charge`'s agreement isn't zero for independent signs.
    planted     `signed_minutiae.planted`'s straight stripes, `PLANTED` spirals in `sm.SHAPE`, independent or all one sign, read as prints are
    master      model 2's masters, whole, with the master's own orientation beside the read one, for `FINGERS` of the first people

Run: python -m signed_charge.reads.agreement_null OUT.json
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_charge as fc
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 4))
PLANTED, SEEDS, RESAMPLES = (26, 36, 60, 90), 30, 2000
FINGERS = [(seed, finger) for seed in range(9000, 9004) for finger in sf.FINGERS]


def agreement(points, y_sign, angle, d):
    heading = sf.heading(points, y_sign)
    cy, cx = points[:, 1].astype(int), points[:, 0].astype(int)
    return float(np.sign(d[cy, cx] * np.sign(np.cos(heading) * np.cos(angle[cy, cx]) + np.sin(heading) * np.sin(angle[cy, cx]))).sum()), len(points)


def read(im, flows):
    fg = fs.foreground(im)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    for x, y in sc.singular(fs.field(im), fg):
        inner &= np.hypot(xx - x, yy - y) >= sc.GUARD
    out = {}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        points = sf.inside(extract(im), inner)
        out[reader] = {name: agreement(points, y_sign, *fc.spreading(flow, fc.SCALE)) for name, flow in flows.items()}
    return out


def planted(task):
    kind, n, seed = task
    rng = np.random.default_rng(seed)
    h, w = sm.SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rot = rng.uniform(0, np.pi)
    psi = 2 * np.pi * (-(xx - w / 2) * np.sin(rot) + (yy - h / 2) * np.cos(rot)) / sm.PERIOD + rng.uniform(0, 2 * np.pi)
    for _ in range(n):
        cx, cy = rng.uniform(40, w - 40), rng.uniform(40, h - 40)
        psi += (1 if kind == "one_sign" else rng.choice([-1, 1])) * np.arctan2(yy - cy, xx - cx)
    im = np.where(np.cos(psi) > 0, 40, 235).astype(np.uint8)
    return (f"{kind}/{n}", read(im, {"read": sm.direction(im)}))


def master(task):
    from ridgegen import finger as fingers

    m = fingers.master(*task)
    im = np.where(m.ridges > 0, 40, 235).astype(np.uint8)
    return ("master", read(im, {"read": sm.direction(im), "true": -np.exp(2j * m.orientation)}))


def main(out):
    started = provenance.start()
    tasks = [(kind, n, seed) for kind in ("independent", "one_sign") for n in PLANTED for seed in range(SEEDS)]
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(planted, tasks, chunksize=1)) + list(pool.map(master, FINGERS, chunksize=1))
    rng = np.random.default_rng(0)
    report = {}
    for name in sorted({r[0] for r in rows}):
        group = [r[1] for r in rows if r[0] == name]
        report[name] = {}
        for reader in sm.EXTRACTORS:
            for flow in group[0][reader]:
                table = np.array([g[reader][flow] for g in group], float)
                value = lambda i: table[i, 0].sum() / max(table[i, 1].sum(), 1)  # noqa: E731
                draws = [value(rng.integers(len(table), size=len(table))) for _ in range(RESAMPLES)]
                report[name][f"{reader}/{flow}"] = {
                    "mean": float(value(np.arange(len(table)))),
                    "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)],
                    "minutiae": float(table[:, 1].sum() / len(table)),
                }
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
