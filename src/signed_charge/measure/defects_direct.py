"""Defects counted with no extractor (E-039): defects_direct.py OUT.json RUN.npz...

The prints are clean binary images from a smooth field, so topology can be read straight off the ridge skeleton: crossing number 1 is an
ending, 3 a fork; an island of ink, whose skeleton has no branch, is two endings of the same ridge. Spurs under `SPUR` px are pruned once.
Counted inside the contact, 16 px in, as the extractors are. A check on mindtct and LEADER, which disagree on these prints (E-039).

Run: python -m signed_charge.measure.defects_direct OUT.json RUN.npz...
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as nd
from skimage.morphology import skeletonize

from signed_charge.measure import provenance
from signed_charge.measure import screen

BORDER, SPUR, MM2 = 16, 5, (25.4 / 500) ** 2
NEIGHBOURS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def crossing(skeleton):
    """The crossing number at each skeleton pixel."""
    s = np.pad(skeleton.astype(np.int8), 1)
    around = np.stack([np.roll(s, (-dy, -dx), (0, 1)) for dy, dx in NEIGHBOURS])
    return (np.abs(around - np.roll(around, -1, 0)).sum(0) // 2 * s)[1:-1, 1:-1]


def prune(skeleton):
    """Remove short spurs: walk in from each ending, and drop the walk if it reaches a fork within `SPUR` px."""
    skeleton = skeleton.copy()
    number = crossing(skeleton)
    h, w = skeleton.shape
    for y0, x0 in zip(*np.nonzero(number == 1)):
        path, y, x = [(y0, x0)], y0, x0
        for _ in range(SPUR):
            step = [(y + dy, x + dx) for dy, dx in NEIGHBOURS if 0 <= y + dy < h and 0 <= x + dx < w and skeleton[y + dy, x + dx] and (y + dy, x + dx) not in path]
            if len(step) != 1:
                break
            y, x = step[0]
            if number[y, x] >= 3:
                for py, px in path:
                    skeleton[py, px] = False
                break
            path.append((y, x))
    return skeleton


def count(im):
    ridges = im == 40
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]].astype(np.float32)
    h, w = im.shape
    contact = ((xx - w / 2) / (w / 2 - 6)) ** 2 + ((yy - h / 2) / (h / 2 - 6)) ** 2 < 1
    inner = cv2.erode(contact.astype(np.uint8), np.ones((2 * BORDER + 1, 2 * BORDER + 1), np.uint8)).astype(bool)
    skeleton = prune(skeletonize(ridges))
    number = crossing(skeleton)
    labels, n = nd.label(skeleton, np.ones((3, 3)))
    sizes = nd.sum(skeleton, labels, range(1, n + 1))
    forks_in = nd.sum(number >= 3, labels, range(1, n + 1))
    islands = np.array([(i + 1) for i in range(n) if forks_in[i] == 0 and sizes[i] < 30])
    small = np.isin(labels, islands)
    area = inner.sum() * MM2
    endings = int(np.sum((number == 1) & inner & ~small))
    forks = int(np.sum((number >= 3) & inner))
    dots = int(np.sum([np.any(inner[labels == i]) for i in islands]))
    return {"endings": endings, "forks": forks, "islands": dots, "area_mm2": float(area),
            "density": (endings + forks + 2 * dots) / area * 100, "density_no_islands": (endings + forks) / area * 100}


def main(out, paths):
    started = provenance.start()
    report = {}
    for path in sorted(paths):
        name = Path(path).stem
        pattern, speed = re.fullmatch(r"(\w+?)_a([\d.]+)_s[\d.]+_\d+", name).groups()
        report[name] = {"pattern": pattern, "speed": float(speed), **count(screen.print_of(np.load(path)["u"]))}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
