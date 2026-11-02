"""Every print's minutiae and window, once, for what the classifier reads (E-067): minutiae_cache.py OUT.pkl RUN.npz...

FVC DB1-3 B (real), DB4 B (SFinGe), the population, and E-038's model 2 and E-027's best. For each print its set, finger, shape, `minutiae_stats.describe`'s inner foreground, and with mindtct and LEADER the minutiae
inside it, (x, y, angle in radians, bifurcation). E-038's prints are rendered again on the CPU, no developmental prints are made, and the checking and proving fingers aren't read.

Run: python -m signed_charge.measure.minutiae_cache OUT.pkl RUN.npz...
"""

import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.reads import classifier_floor as cl
from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_check as fc
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import screen

JOBS = int(os.environ.get("READ_JOBS", 8))
EXTRACTORS = {"mindtct": ms.mindtct, "leader": cl.vm.leader}


def inner_of(im):
    fg = fs.foreground(im)
    if fg is None:
        return None
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    return inner if inner.sum() * ms.MM2 >= 40 else None


def row(group, finger, im):
    inner = inner_of(im)
    if inner is None:
        return None
    points = {}
    h, w = inner.shape
    for name, extract in EXTRACTORS.items():
        p = extract(im)
        xi, yi = p[:, 0].astype(int), p[:, 1].astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        ok[ok] &= inner[yi[ok], xi[ok]]
        points[name] = p[ok]
    return {"set": group, "finger": finger, "shape": (h, w), "inner": np.packbits(inner), "points": points}


def from_fvc(item):
    key, im = item
    return row(key[0], key[1], im)


def from_run(path):
    name = Path(path).stem
    return row(f"population/{re.match(r'(\w+?)_a', name).group(1)}", name, screen.print_of(np.load(path)["u"]))


def from_variant(task):
    key, im = fc.image(task)
    return row(key[0], key[1], im)


def main(out, paths):
    started = provenance.start()
    tasks = [(v, s, f, c) for v in cl.VARIANTS for s in range(cl.FIRST, cl.FIRST + 20) for f in (2, 3, 4, 5, 7, 8, 9, 10) for c in (1, 2)]
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(from_fvc, list(cf.fvc(("DB1", "DB2", "DB3", "DB4"))), chunksize=4))
        print("fvc", len(rows), flush=True)
        rows += list(pool.map(from_run, sorted(paths), chunksize=1))
        print("population", len(rows), flush=True)
        rows += list(pool.map(from_variant, tasks, chunksize=2))
    rows = [r for r in rows if r]
    pickle.dump({"rows": rows, "_provenance": provenance.record(started)}, open(out, "wb"))
    print(len(rows), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
