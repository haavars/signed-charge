"""Which site's wave reached each point: a print tinted by domain (E-049): domains.py SHEET.jpg RUN.npz...

Each site's wave alone (`competency_model.site_arrivals`), with the run's own geometry; the print tinted by the earliest. Blue nail bed, red crease, green pad.

Run: python -m signed_charge.exploratory.domains SHEET.jpg RUN.npz...
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from signed_charge.generate import competency_model as c
from signed_charge.generate import glover_model as g
from signed_charge.measure import screen

TINT = np.array([(255, 120, 60), (70, 70, 255), (80, 210, 80)], np.float32)  # BGR: nail, crease, pad


def tinted(path):
    run = np.load(path)
    name = Path(path).stem
    pattern, speed, seed = re.fullmatch(r"(\w+?)_a([\d.]+)_s[\d.]+_(\d+)", name).groups()
    command = json.loads(str(run["provenance"]))["command"]
    vary = float(command[command.index("--vary") + 1]) if "--vary" in command else 0.0
    points, _, s, _, index = g.mesh(2.0)
    times = c.site_arrivals(points, index, s, pattern, float(speed), c.person_of(int(seed), vary))
    first = np.argmin(times, 0)[index].astype(np.float32)
    h, w = screen.HEIGHT, screen.WIDTH
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    arc = (screen.TOP + yy) * screen.GRID
    width = np.where(arc < np.pi * screen.R / 2, np.sin(arc / screen.R), 1).astype(np.float32)
    columns = (first.shape[1] - 1) / 2 + (xx + 0.5 - w / 2) / width
    domain = cv2.remap(np.flipud(first), columns, screen.TOP + yy, cv2.INTER_NEAREST, borderValue=0).astype(int)
    im = screen.print_of(run["u"])
    grey = (im[..., None].astype(np.float32) / 255) * np.ones(3, np.float32)
    colour = np.clip(grey * 0.55 + TINT[domain] / 255 * 0.45, 0, 1)
    return (np.where((im == 235)[..., None], 0.8 * colour + 0.2, colour * 0.75) * 255).astype(np.uint8)


if __name__ == "__main__":
    cv2.imwrite(sys.argv[1], np.hstack([tinted(p) for p in sys.argv[2:]]), [cv2.IMWRITE_JPEG_QUALITY, 90])
