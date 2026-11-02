"""Ridge wobble: how far ridges stray from their local course (E-047): wobble.py OUT.json RUN.npz...

Ridge direction at `FINE` px against `COARSE` px, on ridge pixels, 16 px in, more than `CLEAR` px from any minutia, so that a minutia's own kink doesn't count: the RMS difference in degrees. Every image goes through mindtct's binarisation first, so grey
and drawn images are read alike. Also the same at `FINE` against `MID`, the middle scale.

Run: python -m signed_charge.exploratory.wobble OUT.json RUN.npz...
"""

import json
import os
import re
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

FINE, MID, COARSE, CLEAR, BORDER = 2.5, 9.0, 18.0, 14, 16
JOBS = int(os.environ.get("WOBBLE_JOBS", 2))
SHARE = 4


def direction(gx, gy, scale):
    smooth = lambda a: cv2.GaussianBlur(a, (0, 0), scale)
    return 0.5 * np.angle(smooth(gx * gx - gy * gy) + 2j * smooth(gx * gy))


def measure(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, None
    points, ridges = mp.mindtct(im)
    b = cv2.GaussianBlur(ridges.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * BORDER + 1,) * 2, np.uint8)).astype(bool)
    keep = inner & (np.hypot(gx, gy) > 0.05)
    if len(points):
        mask = np.ones(im.shape, np.uint8)
        for x, y in points[:, :2].astype(int):
            cv2.circle(mask, (int(x), int(y)), CLEAR, 0, -1)
        keep &= mask > 0
    if keep.sum() < 3000:
        return key, None
    fine, mid, coarse = (direction(gx, gy, s) for s in (FINE, MID, COARSE))
    gap = lambda a, b: np.degrees(np.abs(0.5 * np.angle(np.exp(2j * (a - b)))))
    return key, {
        "fine_coarse": float(np.sqrt(np.mean(gap(fine, coarse)[keep] ** 2))),
        "fine_mid": float(np.sqrt(np.mean(gap(fine, mid)[keep] ** 2))),
        "pixels": int(keep.sum()),
    }


def rendered(task):
    return measure(fc.image(task))


def developmental(path):
    from signed_charge.measure import screen

    name = Path(path).parent.name + "_" + re.match(r"(\w+?_a[\d.]+)", Path(path).stem).group(1)
    return measure(((name, Path(path).stem, 1), screen.print_of(np.load(path)["u"])))


def main(out, runs):
    started = provenance.start()
    real = [x for i, x in enumerate(cf.fvc(("DB1", "DB2", "DB3"))) if i % SHARE == 0]
    sfinge = [x for i, x in enumerate(cf.fvc(("DB4",))) if i % SHARE == 0]
    tasks = [(v, s, f, 1) for v in ("model2", "d_s1_sp2_f3") for s in range(7000, 7006) for f in (2, 3, 4, 5, 7, 8, 9, 10)]
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
    report = {
        k: {"prints": len(v), "fine_coarse": float(np.median([x["fine_coarse"] for x in v])), "fine_mid": float(np.median([x["fine_mid"] for x in v]))}
        for k, v in sets.items()
    }
    for k, v in report.items():
        print(k.ljust(30), v["prints"], round(v["fine_coarse"], 2), round(v["fine_mid"], 2), flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
