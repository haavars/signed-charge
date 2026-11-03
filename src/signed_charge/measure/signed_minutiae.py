"""Signed minutiae: are neighbours of opposite charge? (E-040): signed_minutiae.py OUT.json [--developmental RUN.npz...]

A minutia is a dislocation: going along the ridges, it adds a ridge (charge +1) or takes one away (-1). `charge_pairs` is the sign of
s_i s_j for each pair of minutiae, with the ridge direction lifted along the straight segment between them, so no global lift is needed, no
windows, and prints keep their singular regions. h(r) = mean s_i s_j at separation r: 0 for independent signs, below 0 for opposite pairs, above 0 for
one-sign domains. C(R) = sum over a minutia's neighbours within R of s_i s_j: -1 means its charge is screened within R.

The sign rule is calibrated on planted spirals (96 of 96 right, no misses): the minutia's direction against the ridges' gives the charge, for forks and
endings alike; mindtct's angle is read y up, LEADER's y down. `controls` checks the measurement on planted patterns, and a sign shuffle.

Run: python -m signed_charge.measure.signed_minutiae OUT.json
"""

import json
import os
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_check as fc
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import validate_masters as vm

EDGES = np.arange(0, 165, 5.0)  # px
RADII = (30, 60, 90, 120, 150)
ORIENTATION_SCALE = 9.0  # px, a ridge period
EXTRACTORS = {"mindtct": (ms.mindtct, -1.0), "leader": (vm.leader, 1.0)}  # the angle's y, in image coordinates
PERIOD, SHAPE = 9.0, (400, 300)
VARIANTS = ("model2", "d_s1_sp2_f3")
FIRST, PEOPLE, JOBS, RESAMPLES = 7000, 20, int(os.environ.get("SIGNED_JOBS", 8)), 1000


def direction(im):
    """Ridge orientation as a unit double angle of the gradient, image coordinates."""
    u = cv2.GaussianBlur(im.astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(u, cv2.CV_32F, 1, 0), cv2.Sobel(u, cv2.CV_32F, 0, 1)
    smooth = lambda a: cv2.GaussianBlur(a, (0, 0), ORIENTATION_SCALE)
    return smooth(gx * gx - gy * gy) + 2j * smooth(gx * gy)


def charge_pairs(points, y_sign, field, mask):
    """(r, s_i s_j) for pairs within the last radius whose segment stays on the print."""
    h, w = field.shape
    out_r, out_q = [], []
    xy = points[:, :2]
    heads = np.stack([np.cos(points[:, 2]), y_sign * np.sin(points[:, 2])], 1)
    for i in range(len(points) - 1):
        d = np.hypot(*(xy[i + 1 :] - xy[i]).T)
        for j in np.nonzero(d < EDGES[-1])[0] + i + 1:
            n = max(int(d[j - i - 1] / 3), 4)
            t = np.linspace(0, 1, n)
            x = np.clip(np.round(xy[i, 0] + t * (xy[j, 0] - xy[i, 0])).astype(int), 0, w - 1)
            y = np.clip(np.round(xy[i, 1] + t * (xy[j, 1] - xy[i, 1])).astype(int), 0, h - 1)
            if not mask[y, x].all():
                continue
            ridge = np.unwrap(np.angle(field[y, x])) / 2 + np.pi / 2  # along the ridges, continuous on the segment
            ends = np.stack([np.cos(ridge[[0, -1]]), np.sin(ridge[[0, -1]])], 1)
            si = np.sign(heads[i] @ ends[0])
            sj = np.sign(heads[j] @ ends[1])
            out_r.append(d[j - i - 1])
            out_q.append(si * sj)
    return np.array(out_r), np.array(out_q)


def binned(r, q):
    bins = np.digitize(r, EDGES) - 1
    return np.bincount(bins, q, len(EDGES) - 1), np.bincount(bins, minlength=len(EDGES) - 1)[: len(EDGES) - 1]


def measure(item):
    """One print: for each extractor, pair sums by distance, and screening counts."""
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, None
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    field = direction(im)
    out = {}
    for name, (extract, y_sign) in EXTRACTORS.items():
        points = extract(im)
        if len(points):
            xi, yi = points[:, 0].astype(int), points[:, 1].astype(int)
            ok = (xi >= 0) & (xi < im.shape[1]) & (yi >= 0) & (yi < im.shape[0])
            ok[ok] &= inner[yi[ok], xi[ok]]
            points = points[ok]
        if len(points) < 2:
            out[name] = None
            continue
        r, q = charge_pairs(points, y_sign, field, inner)
        total, count = binned(r, q)
        # A weak null, each pair's sign flipped at random; `planted("independent")` is the one to compare with.
        rng = np.random.default_rng(zlib.crc32(str(key).encode()))
        flips = rng.choice([-1.0, 1.0], len(q))
        out[name] = {"sum": total, "count": count, "shuffled": binned(r, q * flips)[0], "minutiae": len(points)}
    return key, out


def curve(rows, name, rng):
    """h(r) with a CI by print, and C(R), the charge a minutia sees within R."""
    xs = [r[name] for r in rows if r[name]]
    total = np.array([x["sum"] for x in xs])
    count = np.array([x["count"] for x in xs])
    shuffled = np.array([x["shuffled"] for x in xs])
    minutiae = np.array([x["minutiae"] for x in xs], float)

    def h(index):
        return total[index].sum(0) / np.maximum(count[index].sum(0), 1)

    def screen(index):
        return {R: float(2 * total[index][:, EDGES[1:] <= R].sum() / minutiae[index].sum()) for R in RADII}

    every = np.arange(len(xs))
    draws = [rng.integers(len(xs), size=len(xs)) for _ in range(RESAMPLES)]
    hs = np.array([h(d) for d in draws])
    cs = [screen(d) for d in draws]
    return {
        "prints": len(xs),
        "minutiae": float(minutiae.mean()),
        "r": ((EDGES[:-1] + EDGES[1:]) / 2).tolist(),
        "h": h(every).tolist(),
        "h_low": np.percentile(hs, 2.5, 0).tolist(),
        "h_high": np.percentile(hs, 97.5, 0).tolist(),
        "pairs": count.sum(0).tolist(),
        "h_shuffled": (shuffled.sum(0) / np.maximum(count.sum(0), 1)).tolist(),
        "screening": {str(R): {"mean": screen(every)[R], "ci": [float(np.percentile([c[R] for c in cs], q)) for q in (2.5, 97.5)]} for R in RADII},
    }


def planted(task):
    """Stripes carrying known spirals: independent signs, opposite pairs, or one sign."""
    kind, seed = task
    rng = np.random.default_rng(seed)
    h, w = SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rot = rng.uniform(0, np.pi)
    v = -(xx - w / 2) * np.sin(rot) + (yy - h / 2) * np.cos(rot)
    psi = 2 * np.pi * v / PERIOD + rng.uniform(0, 2 * np.pi)
    n = 26
    if kind == "pairs":
        for _ in range(n // 2):
            cx, cy, a, sep = rng.uniform(50, w - 50), rng.uniform(50, h - 50), rng.uniform(0, 2 * np.pi), rng.uniform(18, 36)
            for p, sign in ((1, 1), (-1, -1)):
                psi += p * np.arctan2(yy - cy - sign * sep / 2 * np.sin(a), xx - cx - sign * sep / 2 * np.cos(a))
    else:
        for _ in range(n):
            cx, cy = rng.uniform(40, w - 40), rng.uniform(40, h - 40)
            p = 1 if kind == "one_sign" else rng.choice([-1, 1])
            psi += p * np.arctan2(yy - cy, xx - cx)
    return measure(((kind, seed, 1), np.where(np.cos(psi) > 0, 40, 235).astype(np.uint8)))


def rendered(task):
    return measure(fc.image(task))


def reinked(item):
    """A real print as a binary one, through mindtct's binarisation: does the image type alone move the result?"""
    key, im = item
    fg = fs.foreground(im)
    return measure((key, np.where((cf.binarised(im) == 0) & fg, 40, 235).astype(np.uint8)))


def developmental(path):
    from signed_charge.measure import screen

    name = Path(path).stem
    pattern, speed = re.fullmatch(r"(\w+?)_a([\d.]+)_s[\d.]+_\d+", name).groups()
    return measure(((f"{pattern}_a{speed}", name, 1), screen.print_of(np.load(path)["u"])))


def controls(pool, rng):
    report = {}
    tasks = [(kind, s) for kind in ("independent", "pairs", "one_sign") for s in range(30)]
    done = list(pool.map(planted, tasks, chunksize=2))
    for kind in ("independent", "pairs", "one_sign"):
        rows = [v for k, v in done if k[0] == kind and v]
        report[kind] = {name: curve(rows, name, rng) for name in EXTRACTORS}
        print(
            "control",
            kind,
            {n: (round(report[kind][n]["screening"]["60"]["mean"], 2), np.round(report[kind][n]["h"][2:9], 2).tolist()) for n in EXTRACTORS},
            flush=True,
        )
    return report


def main(out, args):
    started = provenance.start()
    rng = np.random.default_rng(0)
    with ProcessPoolExecutor(JOBS) as pool:
        report = {"controls": controls(pool, rng)}
        if "--reinked" in args:
            rows = [v for _, v in pool.map(reinked, list(cf.fvc(("DB1", "DB2", "DB3"))), chunksize=4) if v]
            report["real_reinked"] = {x: curve(rows, x, rng) for x in EXTRACTORS}
        elif "--controls" not in args:
            real = list(cf.fvc(("DB1", "DB2", "DB3")))
            sfinge = list(cf.fvc(("DB4",)))
            tasks = [(v, s, f, c) for v in VARIANTS for s in range(FIRST, FIRST + PEOPLE) for f in (2, 3, 4, 5, 7, 8, 9, 10) for c in (1, 2)]
            runs = [a for a in args if a.endswith(".npz")]
            sets = {}
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
            for name, rows in sets.items():
                report[name] = {x: curve(rows, x, rng) for x in EXTRACTORS}
                print(name, {x: {R: round(c["mean"], 2) for R, c in report[name][x]["screening"].items()} for x in EXTRACTORS}, flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
