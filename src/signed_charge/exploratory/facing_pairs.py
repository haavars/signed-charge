"""How far apart real close facing pairs lie (E-019): facing_pairs.py OUT

mindtct (reliability ≥ 0.2, 16 px inside the foreground) on FVC 2000/2002/2004 DB1-3 B; every pair of minutiae
under 40 px apart: distance, angle difference (> 144° facing), types, ridges between (`minutiae_pairs`).

Run: python -m signed_charge.exploratory.facing_pairs OUT.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import provenance
from signed_charge.measure.conformality import fs


def one(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return []
    inner = cv2.erode(fg.astype(np.uint8), np.ones((33, 33), np.uint8)).astype(bool)
    pts, ridges = mp.mindtct(im)
    h, w = inner.shape
    pts = [p for p in pts if 0 <= int(p[1]) < h and 0 <= int(p[0]) < w and inner[int(p[1]), int(p[0])]]
    out = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            p, q = pts[i], pts[j]
            d = np.hypot(p[0] - q[0], p[1] - q[1])
            if d < 40:
                da = np.degrees(abs((p[2] - q[2] + np.pi) % (2 * np.pi) - np.pi))
                out.append((float(d), float(da), int(p[3]) + int(q[3]), mp.ridges_between(ridges, np.array(p[:2]), np.array(q[:2]))))
    return out


def main(out):
    started = provenance.start()
    with ProcessPoolExecutor(8) as pool:
        rows = [r for rs in pool.map(one, list(cf.fvc(("DB1", "DB2", "DB3"))), chunksize=4) for r in rs]
    a = np.array(rows)
    facing = a[a[:, 1] > 144]
    report = {
        "close_pairs": len(a),
        "facing": len(facing),
        "facing_median_px": float(np.median(facing[:, 0])),
        "facing_distance_hist_5px": np.histogram(facing[:, 0], np.arange(0, 45, 5))[0].tolist(),
        "close_distance_hist_5px": np.histogram(a[:, 0], np.arange(0, 45, 5))[0].tolist(),
        "by_distance": {},
    }
    for lo, hi in ((0, 15), (15, 25), (25, 40)):
        f = facing[(facing[:, 0] >= lo) & (facing[:, 0] < hi)]
        report["by_distance"][f"{lo}-{hi}"] = {
            "n": len(f),
            "EE_EB_BB": np.bincount(f[:, 2].astype(int), minlength=3).tolist(),
            "ridges_between_0_1_2plus": np.bincount(np.minimum(f[:, 3], 2).astype(int), minlength=3).tolist(),
        }
    report["_provenance"] = provenance.record(started)
    print(json.dumps(report, indent=1))
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
