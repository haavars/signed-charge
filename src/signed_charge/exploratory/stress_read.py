"""D-027's first two tests, and a picture, for a growth stress field (D-027, D-028): stress_read.py OUT.jpg [--spring L] [--normal L] [--load W,L,A] [--width W] [--crease C] [--height W,L,A]

Test 1: the median angle between ridges and edge, along crease, nail furrow and margin. Test 2: singular points by Poincaré index over the surface, on the (s, φ) grid.
The picture is the whole volar surface, flattened as `screen.flattened` does and tip up, with the print's contact outlined, a tick at each sample and ridge direction in brightness of weight.

Run: python -m signed_charge.exploratory.stress_read OUT.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from signed_charge.generate import glover_model as g
from signed_charge.exploratory import growth_stress as gs
from signed_charge.measure import screen

STEP = 6.0  # the grid these are read on, model units
PAIR = 30.0  # model units, about a ridge and a half


def surface(s, phi):
    """Points of the grid with their tangents along s and φ, and normal."""
    alpha = gs.meridian(s, gs.RAMP)[2] if gs.RAMP else np.clip((s - g.L) / g.R, 0, np.pi / 2)
    ring = g.R * np.cos(alpha)
    ss, pp = np.meshgrid(s, phi, indexing="ij")
    aa = np.repeat(alpha[:, None], len(phi), 1)
    rr = g.R * np.cos(aa)
    point = np.stack([rr * np.cos(pp), rr * np.sin(pp), np.where(ss <= g.L, ss, g.L + g.R * np.sin(aa))], -1)
    es = np.stack([-np.sin(aa) * np.cos(pp), -np.sin(aa) * np.sin(pp), np.cos(aa)], -1)
    ep = np.stack([-np.sin(pp), np.cos(pp), np.zeros_like(pp)], -1)
    return point, es, ep, rr, ring


def angles(tensor, es, ep):
    """Ridge angle in the grid's (s, φ) basis, weight, and the stresses."""
    s1, s2, angle = gs.principal(tensor, es, ep)
    return angle, (s1 - s2) / np.maximum(np.abs(s1) + np.abs(s2), 1e-12), s1, s2


def singular(angle):
    """(row, column, index) of cells whose loop of ridge angles winds. Cores +½, deltas -½."""
    a, b, c, d = angle[:-1, :-1], angle[1:, :-1], angle[1:, 1:], angle[:-1, 1:]  # s towards φ, the way the angle turns
    wrap = lambda x: (x + np.pi / 2) % np.pi - np.pi / 2
    total = wrap(b - a) + wrap(c - b) + wrap(d - c) + wrap(a - d)
    rows, cols = np.nonzero(np.abs(total) > 3)
    return [(int(r), int(k), float(np.sign(total[r, k]) / 2)) for r, k in zip(rows, cols)]


def clean(points, s, phi):
    """Without pairs of opposite sign closer than `PAIR`: a field near equal stress throws those up from the mesh's grain, and they mean nothing. Each kept point also says whether it's inside the print's contact."""
    place = lambda p: np.array([s[p[0]], phi[p[1]] * g.R])
    points = list(points)
    while True:
        close = [(np.linalg.norm(place(a) - place(b)), i, j) for i, a in enumerate(points) for j, b in enumerate(points) if i < j and a[2] * b[2] < 0]
        if not close or min(close)[0] > PAIR:
            break
        _, i, j = min(close)
        points = [p for k, p in enumerate(points) if k not in (i, j)]
    return [(*p, contact(s[p[0]], phi[p[1]], s.max())) for p in points]


def contact(s, phi, top):
    """Inside the print's ellipse, on `screen.flattened`'s cut."""
    a = top - s
    x = np.where(a < np.pi * g.R / 2, np.sin(a / g.R), 1) * (phi - np.pi / 2) * g.R / screen.GRID
    y = a / screen.GRID - screen.TOP
    return bool((x / (screen.WIDTH / 2 - 6)) ** 2 + ((y - screen.HEIGHT / 2) / (screen.HEIGHT / 2 - 6)) ** 2 < 1)


def edges(angle, s):
    """Median ridge-edge angle in degrees along the crease, nail and margin. Edges run along s at the sides, along φ at the crease."""
    fold = lambda x: np.degrees(np.abs((x + np.pi / 2) % np.pi - np.pi / 2))
    side = np.concatenate([angle[:, 1], angle[:, -2]])
    both = np.concatenate([s, s])
    nail = both >= g.NAIL_FROM * g.L
    return {"crease": float(np.median(90 - fold(angle[1, 1:-1]))), "nail": float(np.median(fold(side[nail & (both < both.max() - 12)]))),
            "margin": float(np.median(fold(side[~nail & (both > 30)])))}




def picture(path, angle, weight, s, phi, spots):
    top = s.max()
    h, w = 500, 400
    im = np.zeros((h, w, 3), np.uint8)
    map_xy = lambda a, c: (w / 2 + np.where(a < np.pi * g.R / 2, np.sin(a / g.R), 1) * c / screen.GRID, 10 + a / screen.GRID)
    for i in range(0, len(s), 2):
        for j in range(0, len(phi), 2):
            a = top - s[i]
            ring = g.R * np.cos(np.clip((s[i] - g.L) / g.R, 0, np.pi / 2))
            if ring < 2:
                continue
            c = (phi[j] - np.pi / 2) * g.R
            ds, dt = np.cos(angle[i, j]), np.sin(angle[i, j])
            step = 7.0
            ends = []
            for sign in (-1, 1):
                aa = a - sign * step * ds
                cc = c + sign * step * dt * g.R / ring
                ends.append(tuple(int(round(float(v))) for v in map_xy(np.array(aa), np.array(cc))))
            v = int(70 + 185 * np.clip(weight[i, j], 0, 1))
            cv2.line(im, ends[0], ends[1], (v, v, v), 1, cv2.LINE_AA)
    for r, k, p, _ in spots:
        a = top - s[r]
        c = (phi[k] - np.pi / 2) * g.R
        x, y = map_xy(np.array(a), np.array(c))
        cv2.circle(im, (int(x), int(y)), 5, (0, 0, 255) if p > 0 else (0, 200, 0), 2)
    cv2.ellipse(im, (w // 2, int(10 + screen.TOP + screen.HEIGHT / 2)), (screen.WIDTH // 2 - 6, screen.HEIGHT // 2 - 6), 0, 0, 360, (200, 120, 0), 1)
    cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, 90])


def read(pattern, person, **kwargs):
    result = gs.solve(pattern, person, **kwargs)
    s = np.arange(0, g.L + g.R * np.pi / 2 + 1e-6, STEP)
    phi = np.linspace(0, np.pi, int(round(np.pi * g.R / STEP)) + 1)
    tensor = gs.carry(result, s, phi)
    _, es, ep, _, _ = surface(s, phi)
    angle, weight, s1, s2 = angles(tensor, es, ep)
    inner = singular(angle[:-6, :])  # the rows at the pole are too short to hold an angle
    inner = clean([x for x in inner if 1 <= x[1] < len(phi) - 2 and x[0] >= 1], s, phi)
    seen = [x for x in inner if x[3]]
    return {"edges": edges(angle, s), "points": inner, "sum": sum(x[2] for x in inner), "cores": sum(x[2] > 0 for x in seen), "deltas": sum(x[2] < 0 for x in seen),
            "s1": [float(np.percentile(s1, q)) for q in (5, 50, 95)], "s2": [float(np.percentile(s2, q)) for q in (5, 50, 95)], "weight": float(np.median(weight))}, (angle, weight, s, phi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--spring", type=float, default=480.0)
    parser.add_argument("--normal", type=float, default=60.0)
    parser.add_argument("--load", default="0,0,0", help="whorl, loop, arch")
    parser.add_argument("--width", type=float, default=120.0)
    parser.add_argument("--crease", type=float, default=0.75)
    parser.add_argument("--height", default="0,0,0", help="whorl, loop, arch")
    args = parser.parse_args()
    load = dict(zip(("whorl", "loop", "arch"), (float(v) for v in args.load.split(","))))
    height = dict(zip(("whorl", "loop", "arch"), (float(v) for v in args.height.split(","))))
    tiles = []
    for pattern in ("arch", "loop", "whorl"):
        report, (angle, weight, s, phi) = read(pattern, (0.0, 0.0, 0.0, 0.0), spring=args.spring, normal_spring=args.normal, load=load[pattern], width=args.width, crease_force=args.crease, height=height[pattern])
        print(pattern, json.dumps({k: v for k, v in report.items() if k != "points"}), [(round(float(s[r])), round(float(np.degrees(phi[k]))), p) for r, k, p, _ in report["points"]], flush=True)
        out = Path(args.out).with_suffix(f".{pattern}.jpg")
        picture(out, angle, weight, s, phi, report["points"])
        tiles.append(cv2.imread(str(out)))
        out.unlink()
    cv2.imwrite(args.out, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
