"""A simulated fingertip as a print, through the minutiae screens (E-029): screen.py OUT.json NPZ...

A run's activator u on its (s, φ) grid is close to a rolled print at 500 ppi (a ridge period of about 19 model units is
about 9 px, the grid's 2 units about 1 px, so scale 1). The pad is cut from it: `HEIGHT` rows down from just below the pole
and `WIDTH` wide, flattened at the shoulder by `print_of`, in an elliptical contact as a plain print's, ridges where u is
above its median. Then `minutiae_stats.describe` and `minutiae_pairs.describe`, with mindtct.

Run: python -m signed_charge.measure.screen
OUT.json RUN.npz...
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance

WIDTH, HEIGHT = 300, 400
TOP = 20  # rows below the pole
GRID, R = 2.0, 240.0  # the runs' dx and Glover's radius, model units


def flattened(u):
    """Tip up, in an elliptical contact. The grid's columns close up towards the pole, so the shoulder is flattened by a
    sinusoidal (equal-area) projection: each row is cut at its true width (D-009). Returns the field
    and the contact."""
    h, w = u.shape
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    arc = (TOP + yy) * GRID  # surface distance from the pole
    width = np.where(arc < np.pi * R / 2, np.sin(arc / R), 1).astype(np.float32)
    columns = (w - 1) / 2 + (xx + 0.5 - WIDTH / 2) / width
    u = cv2.remap(np.flipud(u).astype(np.float32), columns, TOP + yy, cv2.INTER_LINEAR, borderValue=float("nan"))
    contact = ((xx - WIDTH / 2) / (WIDTH / 2 - 6)) ** 2 + ((yy - HEIGHT / 2) / (HEIGHT / 2 - 6)) ** 2 < 1
    return u, contact & ~np.isnan(u)


def print_of(u):
    """Ink on ridges."""
    u, contact = flattened(u)
    # At the median, ridges and valleys take equal area, as inked prints' roughly do: the activator's stripes are
    # narrower than the gaps between them, and at its mean thin ridges turn forks into endings.
    return np.where(contact & (u > np.median(u[contact])), 40, 235).astype(np.uint8)


def main(out, paths):
    started = provenance.start()
    report = {}
    for path in paths:
        name = Path(path).stem
        d = np.load(path)
        im = print_of(d["u"])
        cv2.imwrite(str(Path(path).with_suffix(".png")), im)
        _, stats = ms.describe((name, im))
        _, pairs = mp.describe((name, im))
        report[name] = {"stats": {k: v for k, v in (stats or {}).items() if k != "mh"}, "pairs": pairs}
        s, p = report[name]["stats"], pairs or {}
        close = p.get("close", {})
        n = max(sum(close.get(k, 0) for k in ("same", "offset", "facing", "other")), 1)
        print(
            name.ljust(16),
            "count",
            s.get("count"),
            "density",
            round(s.get("density", 0), 1),
            "bif",
            round(s.get("bifurcations") or 0, 2),
            "close",
            n,
            [round(close.get(k, 0) / n, 2) for k in ("same", "offset", "facing")],
        )
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
