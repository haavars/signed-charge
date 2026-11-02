"""Training prints for D-079: tstr_data.py OUT_DIR

ZP and ZP+P on fresh people, seeds from `FIRST`, and SynFing's pairs, all as the same ridge map, so the matcher sees ridges and not an image type. Each written once, as PNG; a finished file is skipped.

Run: python -m signed_charge.reads.tstr_data OUT_DIR
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import flow_check as fc
from signed_charge.reads import tstr
from signed_charge import paths

JOBS = int(os.environ.get("READ_JOBS", 24))
FIRST, PEOPLE, FINGERS, CAPTURES = 20000, int(os.environ.get("PEOPLE", 125)), (2, 3, 4, 5, 7, 8, 9, 10), (1, 2, 3, 4, 5, 6)
SYNFING = paths.SYNFING


def made(task):
    variant, name, seed, finger, capture = task
    out = Path(sys.argv[1]) / name / f"{seed}_{finger}_{capture}.png"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), tstr.ridge_map(fc.image((variant, seed, finger, capture))[1]))


def pair(task):
    path, identity, impression = task
    out = Path(sys.argv[1]) / "synfing" / f"{identity}_0_{impression}.png"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), tstr.ridge_map(cv2.imread(str(path), 0)))


if __name__ == "__main__":
    tasks = [(variant, name, seed, finger, capture) for variant, name in (("model2", "zp"), ("d_s1_sp2_f3", "zpp")) for seed in range(FIRST, FIRST + PEOPLE) for finger in FINGERS for capture in CAPTURES]
    images = PEOPLE * len(FINGERS) * len(CAPTURES)
    pairs = [(SYNFING / f"impression_{k}" / f"F_{i * 16:05d}.jpg", i, k) for i in range(images // 2) for k in (1, 2)]
    with ProcessPoolExecutor(JOBS) as pool:
        list(pool.map(pair, pairs, chunksize=32))
        print("synfing", flush=True)
        for i, _ in enumerate(pool.map(made, tasks, chunksize=6)):
            if i % 2000 == 0:
                print(i, len(tasks), flush=True)
