"""(E-105) Exploratory, unregistered: what D-065's cells added, read from the prints already made. neutral_what.py OUT.json RUN.npz...

Per cell, each named by the start of its prints' stems, by finger, inside the print less `MARGIN` px:
    misfit      degrees between the flow read from the ridges and the flow asked for, mean and the share of area over 20° and 45°
    seam        share of area on `minutiae_seams`' seams, and mindtct's minutiae a 100 mm² on and off them
    small       islands and lakes under `SMALL` px a 100 mm²

Run: python -m signed_charge.reads.neutral_what OUT.json RUN.npz...
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as nd

from signed_charge.reads import field_read as fr
from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_seams as seams
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 8))
MARGIN, SMALL, RESAMPLES = 24, 150, 2000


def kink(arrival):
    """`minutiae_seams.kink` on a planar sheet, which has nothing to flatten."""
    smooth = cv2.GaussianBlur(arrival.astype(np.float32), (0, 0), seams.BLUR)
    gx, gy = cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    size = np.hypot(gx, gy) + 1e-6
    return 1 - np.hypot(*(cv2.GaussianBlur(a / size, (0, 0), seams.SCALE) for a in (gx, gy)))


def one(path):
    d = np.load(path)
    im = fr.sheet(path)
    inner = np.zeros(im.shape, bool)
    inner[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    off = np.degrees(np.abs(np.angle(np.exp(2j * (fs.field(im) - d["theta"]))))) / 2
    area = inner.sum() * ms.MM2 / 100
    on = (kink(d["arrival"]) >= seams.SEAM) & inner
    points = sm.EXTRACTORS["mindtct"][0](im)
    small = 0
    for mask in (im == 40, im != 40):
        labels, n = nd.label(mask)
        sizes = nd.sum(mask, labels, range(1, n + 1))
        boxes = nd.find_objects(labels)
        small += sum(
            s < SMALL and b[0].start >= MARGIN and b[1].start >= MARGIN and b[0].stop <= im.shape[0] - MARGIN and b[1].stop <= im.shape[1] - MARGIN
            for s, b in zip(sizes, boxes)
        )
    return {
        "cell": Path(path).stem.split("_")[0],
        "misfit": float(off[inner].mean()),
        "over20": float((off[inner] > 20).mean()),
        "over45": float((off[inner] > 45).mean()),
        "seam_area": float(on.sum() / inner.sum()),
        "on_seam": len(sf.inside(points, on)) / max(on.sum() * ms.MM2 / 100, 1e-6),
        "off_seam": len(sf.inside(points, inner & ~on)) / max((inner & ~on).sum() * ms.MM2 / 100, 1e-6),
        "small": float(small / area),
    }


def main(out, paths):
    started = provenance.start()
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(one, sorted(paths), chunksize=1))
    rng = np.random.default_rng(0)
    report = {}
    for cell in sorted({r["cell"] for r in rows}):
        group = [r for r in rows if r["cell"] == cell]
        report[cell] = {}
        for key in group[0]:
            if key != "cell":
                v = np.array([g[key] for g in group])
                draws = [v[rng.integers(len(v), size=len(v))].mean() for _ in range(RESAMPLES)]
                report[cell][key] = {"mean": float(v.mean()), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
