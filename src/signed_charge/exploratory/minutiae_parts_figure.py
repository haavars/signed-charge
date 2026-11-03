"""E-067's figure: where each set's minutiae histogram differs from real's, through real windows: minutiae_parts_figure.py PARTS.json FIGURE.png

Standardised difference, real less the set, bin by bin. Blue, the set has too few pairs there; red, too many. Run with python_biometrics' environment.
"""

import json
import sys

import cv2
import numpy as np

CELL, GAP, MARGIN, TOP, LIMIT = 30, 2, 64, 58, 1.0
LOW, MIDDLE, HIGH = np.array((0x1F, 0x5F, 0xB4)), np.array((0xF0, 0xEF, 0xEC)), np.array((0xC4, 0x3A, 0x31))  # RGB: too few, none, too many
SURFACE, INK, MUTED = (0xFB, 0xFC, 0xFC), (0x22, 0x22, 0x22), (0x66, 0x66, 0x66)
SETS = (("population", "Population"), ("sfinge", "SFinGe"), ("model2", "Model 2"))
READERS = (("mindtct", "mindtct"), ("leader", "LEADER"))
FONT = cv2.FONT_HERSHEY_SIMPLEX


def colour(value):
    """BGR on the diverging scale: positive effect means real has more, so the set has too few."""
    x = np.clip(value / LIMIT, -1, 1)
    rgb = MIDDLE + (LOW - MIDDLE) * x if x > 0 else MIDDLE + (HIGH - MIDDLE) * -x
    return tuple(int(v) for v in rgb[::-1])


def text(image, words, at, scale=0.42, ink=INK):
    cv2.putText(image, words, at, FONT, scale, ink, 1, cv2.LINE_AA)


def panel(image, effect, x0, y0):
    for i in range(10):
        for j in range(10):
            cv2.rectangle(image, (x0 + j * CELL, y0 + i * CELL), (x0 + (j + 1) * CELL - GAP, y0 + (i + 1) * CELL - GAP), colour(effect[i, j]), -1)
    for i in (0, 5, 10):
        text(image, f"{i * 20}", (x0 - 32, y0 + i * CELL + (4 if i < 10 else 0)), ink=MUTED)
    for j in (0, 5, 10):
        text(image, f"{j * 18}", (x0 + j * CELL - 9, y0 + 10 * CELL + 16), ink=MUTED)


def main(parts, out):
    data = json.load(open(parts))
    width = MARGIN + len(SETS) * (10 * CELL + MARGIN)
    height = TOP + len(READERS) * (10 * CELL + 58) + 34
    image = np.full((height, width, 3), SURFACE, np.uint8)
    text(image, "Minutiae histogram against real, through real windows", (MARGIN, 22), 0.55)
    text(image, "rows: distance, px; columns: direction difference, degrees", (MARGIN, 42), ink=MUTED)
    for r, (reader, reader_label) in enumerate(READERS):
        y0 = TOP + 20 + r * (10 * CELL + 58)
        for c, (name, label) in enumerate(SETS):
            x0 = MARGIN + c * (10 * CELL + MARGIN)
            text(image, f"{label}, {reader_label}", (x0, y0 - 8), 0.45)
            panel(image, np.array(data[reader]["effects"]["windows"][name]["effect"]).reshape(10, 10), x0, y0)
    y = height - 26
    for k, v in enumerate(np.linspace(1, -1, 41)):
        cv2.rectangle(image, (MARGIN + k * 6, y), (MARGIN + (k + 1) * 6, y + 12), colour(v * LIMIT), -1)
    text(image, "too few", (MARGIN, y - 4), ink=MUTED)
    text(image, "too many, effect size 1", (MARGIN + 41 * 6 - 150, y - 4), ink=MUTED)
    cv2.imwrite(out, image)


if __name__ == "__main__":
    main(*sys.argv[1:])
