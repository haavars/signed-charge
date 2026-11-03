"""What lies under a print's flaws, a look (E-062): stress_overlay.py OUT.jpg RUN.npz...

Each print with, in the print's frame: magenta where the stress cue's weight a is under `WEAK`, where the wave alone sets direction; cyan along the meetings of the waves from pad, nail and crease; and beside it the angle at which the stress field crosses the arrival front, 0-90°.
A file name holds pattern and seed, and the finger's geometry follows from the seed at vary 1.

Run: python -m signed_charge.exploratory.stress_overlay OUT.jpg RUN.npz...
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np

from signed_charge.generate import competency_model as cm
from signed_charge.generate import glover_model as g
from signed_charge.exploratory import growth_stress as gs
from signed_charge.measure import screen
from signed_charge.exploratory import stress_read as sr

WEAK = 0.2


def main(out, paths):
    points, tris, s, phi, index = g.mesh(2.0)
    _, es, ep, _, ring = sr.surface(s, phi)
    tiles = []
    for path in paths:
        pattern, seed = re.fullmatch(r"(\w+?)_a[\d.]+_s[\d.]+_(\d+)", Path(path).stem).groups()
        person = cm.person_of(int(seed), 1.0)
        d = np.load(path)
        tensor = gs.carry(gs.solve(pattern, person, **gs.settings(pattern)), s, phi)
        angle, a, _, _ = sr.angles(tensor, es, ep)
        arrival = d["arrival"].astype(float)
        ts, tp = np.gradient(arrival, s, axis=0), np.gradient(arrival, phi, axis=1) / np.maximum(ring, 1.0)[:, None]
        crossing = np.abs((np.arctan2(tp, ts) + np.pi / 2 - angle + np.pi / 2) % np.pi - np.pi / 2)  # the ridge against the front's own line
        domains = cm.site_arrivals(points, index, s, pattern, 0.7, person).argmin(0)[index]
        im = screen.print_of(d["u"])
        over = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        weak, contact = screen.flattened(a.astype(np.float32))
        over[contact & (weak < WEAK)] = (0.5 * over[contact & (weak < WEAK)] + np.array([120, 0, 120])).astype(np.uint8)
        dom, _ = screen.flattened(domains.astype(np.float32))
        edge = cv2.morphologyEx(np.round(dom).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        over[contact & edge] = (255, 255, 0)
        cross, _ = screen.flattened(crossing.astype(np.float32))
        heat = cv2.applyColorMap(np.clip(np.nan_to_num(cross) / (np.pi / 2) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        heat[~contact] = 255
        tiles.append(np.vstack([over, heat]))
        print(
            Path(path).stem,
            "weak share",
            round(float((weak[contact] < WEAK).mean()), 3),
            "median crossing",
            round(float(np.degrees(np.nanmedian(cross[contact]))), 1),
            "share over 45",
            round(float(np.nanmean(cross[contact] > np.pi / 4)), 3),
            flush=True,
        )
    cv2.imwrite(out, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
