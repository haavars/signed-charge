"""The first classifier score for a developmental population (E-045): population_read.py OUT.json SHEET.jpg RUN.npz...

Each print is its own finger, with its own geometry, on the D-009 cut. With mindtct and LEADER: the fair classifier against FVC DB1-3 B over 10 fold seeds, as E-038, and bootstrapped by finger; density, close pairs, forks; signed charge; the direct count.
SFinGe beside it, and real against real at the population's own size, each of 30 draws of as many prints from four sensor sets against five. Model 2's and E-027's best are E-038's at 160 and 320 prints, which bracket this size.

Run: python -m signed_charge.reads.population_read OUT.json SHEET.jpg RUN.npz...
"""

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.reads import classifier_floor as cl
from signed_charge.measure import conformality as cf
from signed_charge.measure import defects_direct as dd
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.measure import signed_minutiae as sm
from signed_charge.measure import validate_masters as vm

JOBS = int(os.environ.get("READ_JOBS", 8))
RESAMPLES = 30
MINIMUM = 10  # prints a pattern needs for its own classifier score
READERS = {"mindtct": (ms.mindtct, mp.mindtct), "leader": (vm.leader, lambda im: (vm.leader(im), cf.binarised(im) == 0))}


def read_image(item):
    key, im = item
    stats, pairs = {}, {}
    for name, (extract, reader) in READERS.items():
        ms.mindtct, mp.mindtct = extract, reader
        stats[name] = ms.describe(item)[1]
        pairs[name] = mp.describe(item)[1]
    return key, stats, pairs, sm.measure(item)[1]


def one(path):
    name = Path(path).stem
    pattern = re.match(r"(\w+?)_a", name).group(1)
    im = screen.print_of(np.load(path)["u"])
    key = (pattern, name, 1)
    return (*read_image((key, im)), dd.count(im), cv2.resize(im, (150, 200), interpolation=cv2.INTER_AREA))


def scored(real, mine, name, rng):
    """The fair classifier: over fold seeds, and over draws of fingers with replacement."""
    r = [(k, s[name]) for k, s, _, _ in real if s[name]]
    m = [(k, s[name]) for k, s, *_ in mine if s[name]]
    folds = cl.spread([cf.accuracy(r, m, "mh", seed=s) for s in range(cl.SEEDS)])
    draws = []
    for seed in range(RESAMPLES):
        sample = [m[i] for i in rng.integers(len(m), size=len(m))]
        draws.append(cf.accuracy(r, sample, "mh", seed=seed))
    return {"prints": len(m), "folds": folds, "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}


def described(rows, name, rng):
    pairs = mp.summary([p[name] for _, _, p, *_ in rows if p[name]])
    signed = sm.curve([q for _, _, _, q, *_ in rows if q], name, rng)
    stats = [s[name] for _, s, *_ in rows if s[name]]
    return {
        "density": pairs["density_per_100mm2"],
        "same": pairs["close_same"],
        "offset": pairs["close_offset"],
        "facing": pairs["close_facing"],
        "forks": float(np.mean([s["bifurcations"] for s in stats if s["bifurcations"] is not None])),
        "c60": signed["screening"]["60"],
        "c90": signed["screening"]["90"],
        "h": signed["h"],
        "pairs_binned": signed["pairs"],
    }


def floor(real, size, name, rng):
    """Real against real at about the population's size: four sets against five."""
    r = [(k, s[name]) for k, s, _, _ in real if s[name]]
    sensors = sorted({k[0] for k, _ in r})
    out = []
    for seed in range(RESAMPLES):
        a = {sensors[i] for i in rng.permutation(len(sensors))[:4]}
        left = [x for x in r if x[0][0] in a]
        right = [x for x in r if x[0][0] not in a]
        left = [left[i] for i in rng.permutation(len(left))[:size]]
        right = [right[i] for i in rng.permutation(len(right))[:size]]
        out.append(cf.accuracy(left, right, "mh", seed=seed))
    return cl.spread(out)


def main(out, figure, paths):
    started = provenance.start()
    with ProcessPoolExecutor(JOBS) as pool:
        real = list(pool.map(read_image, list(cf.fvc(("DB1", "DB2", "DB3"))), chunksize=4))
        sfinge = list(pool.map(read_image, list(cf.fvc(("DB4",))), chunksize=4))
        rows = list(pool.map(one, sorted(paths), chunksize=1))
    rng = np.random.default_rng(0)
    report = {
        "direct": {
            "density": float(np.median([r[4]["density"] for r in rows])),
            "no_islands": float(np.median([r[4]["density_no_islands"] for r in rows])),
            "islands": float(np.mean([r[4]["islands"] for r in rows])),
        }
    }
    for name in READERS:
        report[name] = {
            "population": {**scored(real, rows, name, rng), **described(rows, name, rng)},
            "sfinge": {**scored(real, sfinge, name, rng), **described(sfinge, name, rng)},
            "floor": floor(real, len(rows), name, rng),
        }
        for pattern in sorted({r[0][0] for r in rows}):
            mine = [r for r in rows if r[0][0] == pattern]
            report[name][pattern] = {**(scored(real, mine, name, rng) if len(mine) >= MINIMUM else {}), **described(mine, name, rng)}
        print(
            name,
            {k: round(v["folds"]["mean"], 3) if "folds" in v else round(v.get("mean", v.get("density", 0)), 3) for k, v in report[name].items()},
            flush=True,
        )
    tiles = [r[5] for r in rows][:48]
    if len(tiles) >= 8:
        cols = 12 if len(tiles) >= 12 else len(tiles)
        count = len(tiles) // cols * cols
        cv2.imwrite(figure, np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, count, cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
