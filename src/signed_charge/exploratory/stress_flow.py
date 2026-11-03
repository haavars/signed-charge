"""D-027's test 3: does the growth stress field predict real flow better than the harmonic guide? (E-057): stress_flow.py OUT.json FIGURE.jpg

One template a pattern, on the canonical finger, flattened as `screen.flattened` does, on blocks of 8 px. Each real print from FVC DB1-3 B is scored against the arch, whorl and both mirrors of the loop, at each of a set of shifts and rotations, and keeps the
best: the mean angle between its flow and the template's over the overlap, with the overlap at least `OVERLAP` of the print's foreground. Candidates are the stress field and the harmonic guide, and as a control on how much the search itself buys, a template of parallel lines.
Scale is held at 1. CIs are by finger, resampling the fingers; the difference between stress and guide is paired.

Run: python -m signed_charge.exploratory.stress_flow OUT.json FIGURE.jpg
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.generate import competency_model as cm
from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.generate import glover_model as g
from signed_charge.exploratory import growth_stress as gs
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.exploratory import stress_read as sr

JOBS = int(os.environ.get("FLOW_JOBS", 2))
B = 8
CANVAS = (560, 460)  # print px, the flattened surface
TURNS = np.deg2rad(np.arange(-30, 31, 10))
SHIFTS = np.arange(-14, 15, 2)  # blocks, ±112 px
OVERLAP, RESAMPLES = 0.7, 2000


def flatten(angle, s, phi):
    """Ridge angle on the (s, φ) grid, from e_s towards e_φ, to z = exp(2iθ) on print blocks, θ in image axes. 0 outside the surface."""
    h, w = CANVAS
    yy, xx = np.mgrid[B // 2 : h : B, B // 2 : w : B].astype(float)
    a = yy * screen.GRID  # surface distance from the pole
    stretch = np.where(a < np.pi * g.R / 2, np.sin(a / g.R), 1.0)
    bend = np.where(a < np.pi * g.R / 2, np.cos(a / g.R) / g.R, 0.0)
    c = (xx - w / 2) * screen.GRID / np.maximum(stretch, 1e-3)
    p = np.pi / 2 + c / g.R
    t = s.max() - a
    inside = (p >= 0) & (p <= np.pi) & (t >= 0) & (stretch > 0.05)
    i = np.clip(np.round(t / (s[1] - s[0])).astype(int), 0, len(s) - 1)
    j = np.clip(np.round(p / (phi[1] - phi[0])).astype(int), 0, len(phi) - 1)
    ds, dt = np.cos(angle[i, j]), np.sin(angle[i, j])  # along s and across, unit
    dy = -ds  # the image's y is distance from the pole
    dx = dt + bend * c * dy  # the sinusoidal map's shear
    return np.where(inside, np.exp(2j * np.arctan2(dy, dx)), 0)


def stress_templates():
    out = {}
    for pattern in gs.PADS:
        _, (angle, _, s, phi) = sr.read(pattern, (0.0, 0.0, 0.0, 0.0), **gs.settings(pattern))
        out[pattern] = flatten(angle, s, phi)
    return out


def guide_templates(dx=gs.COARSE):
    """The harmonic guide's level sets, on the same grid as the stress."""
    points, tris, s, phi, index = g.mesh(dx)
    stiff, _ = g.laplacian(points, tris)
    out = {}
    for pattern in gs.PADS:
        sites = cm.site_vertices(points, index, s, pattern, (0.0, 0.0, 0.0, 0.0))
        psi = cm.guide(stiff, sites, pattern)[index]
        ring = g.R * np.cos(np.clip((s - g.L) / g.R, 0, np.pi / 2))
        ps = np.gradient(psi, s, axis=0)
        pt = np.gradient(psi, phi, axis=1) / np.maximum(ring, 1.0)[:, None]
        out[pattern] = flatten(np.arctan2(ps, -pt), s, phi)
    return out


def parallel_templates():
    h, w = CANVAS
    return {"lines": np.ones((h // B, w // B), complex)}


def posed(template):
    """Each template at each turn, mirrored too where it isn't its own mirror image: arrays (poses, h, w), in print blocks."""
    h, w = template.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = 220 / B, w / 2  # the middle of `screen.flattened`'s contact
    out = []
    for z in (template, np.conj(template[:, ::-1])):
        for turn in TURNS:
            co, si = np.cos(-turn), np.sin(-turn)
            mx = (co * (xx - cx) - si * (yy - cy) + cx).astype(np.float32)
            my = (si * (xx - cx) + co * (yy - cy) + cy).astype(np.float32)
            re = cv2.remap(z.real.astype(np.float32), mx, my, cv2.INTER_NEAREST, borderValue=0)
            im = cv2.remap(z.imag.astype(np.float32), mx, my, cv2.INTER_NEAREST, borderValue=0)
            out.append((re + 1j * im) * np.exp(2j * turn))
    return np.array(out)


def blocks(im):
    """A print's z and foreground on blocks, or None."""
    fg = fs.foreground(im)
    if fg is None:
        return None
    theta = fs.field(im)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((3 * B, 3 * B), np.uint8)).astype(bool)
    z, mask = np.exp(2j * theta)[B // 2 :: B, B // 2 :: B], inner[B // 2 :: B, B // 2 :: B]
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    rows = slice(rows.min(), min(rows.max() + 1, rows.min() + CANVAS[0] // B))  # the print's own extent: sensors' frames vary
    cols = slice(cols.min(), min(cols.max() + 1, cols.min() + CANVAS[1] // B))
    return z[rows, cols], mask[rows, cols]


def best(z, mask, poses):
    """The smallest mean angle over shifts and poses, degrees. Poses are on a canvas larger than any print."""
    h, w = z.shape
    total = mask.sum()
    ph, pw = poses.shape[1:]
    top = None
    for dy in SHIFTS:
        for dx in SHIFTS:
            y0, x0 = 220 // B - h // 2 + dy, (pw - w) // 2 + dx
            if y0 + h > ph or x0 + w > pw:
                continue
            cut = np.zeros((len(poses), h, w), complex)  # above the tip there's no template
            a, c = max(y0, 0), max(x0, 0)
            cut[:, a - y0 :, c - x0 :] = poses[:, a : y0 + h, c : x0 + w]
            valid = (cut != 0) & mask
            count = valid.sum((1, 2))
            good = count >= OVERLAP * total
            if not good.any():
                continue
            gap = np.abs(np.angle(z * np.conj(cut))) / 2
            score = (gap * valid).sum((1, 2)) / np.maximum(count, 1)
            score[~good] = np.inf
            lowest = score.min()
            if top is None or lowest < top:
                top = lowest
    return None if top is None else float(np.degrees(top))


SETS = {}


def score(item):
    key, im = item
    b = blocks(im)
    if b is None:
        return key, None
    z, mask = b
    out = {}
    for name, templates in SETS.items():
        values = {pattern: best(z, mask, poses) for pattern, poses in templates.items()}
        values = {k: v for k, v in values.items() if v is not None}
        out[name] = (min(values.values()), min(values, key=values.get)) if values else None
    return key, out


def interval(rows, name, rng):
    fingers = sorted({k[1] for k, v in rows if v and v.get(name)})
    by = {f: [v[name][0] for k, v in rows if k[1] == f and v and v.get(name)] for f in fingers}
    means = np.array([np.mean(by[f]) for f in fingers])
    draws = [np.mean(means[rng.integers(len(means), size=len(means))]) for _ in range(RESAMPLES)]
    return {
        "mean": float(means.mean()),
        "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)],
        "fingers": len(fingers),
        "by_finger": dict(zip(fingers, means.tolist())),
    }


def main(out, figure):
    started = provenance.start()
    SETS["stress"] = {k: posed(v) for k, v in stress_templates().items()}
    SETS["guide"] = {k: posed(v) for k, v in guide_templates().items()}
    SETS["lines"] = {k: posed(v) for k, v in parallel_templates().items()}
    tiles = []
    for pattern in gs.PADS:
        for name in ("stress", "guide"):
            z = SETS[name][pattern][len(TURNS) // 2]
            tile = np.zeros((*z.shape, 3), np.uint8)
            tile[..., 0] = tile[..., 2] = (127 * (1 + z.real) * (z != 0)).astype(np.uint8)
            tile[..., 1] = (127 * (1 + z.imag) * (z != 0)).astype(np.uint8)
            tiles.append(cv2.resize(tile, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(figure, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(score, list(cf.fvc(("DB1", "DB2", "DB3"))), chunksize=4))
    rng = np.random.default_rng(0)
    report = {name: interval(rows, name, rng) for name in SETS}
    fingers = sorted(set(report["stress"]["by_finger"]) & set(report["guide"]["by_finger"]))
    diff = np.array([report["stress"]["by_finger"][f] - report["guide"]["by_finger"][f] for f in fingers])
    draws = [np.mean(diff[rng.integers(len(diff), size=len(diff))]) for _ in range(RESAMPLES)]
    report["difference"] = {
        "mean": float(diff.mean()),
        "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)],
        "fingers_stress_lower": int((diff < 0).sum()),
        "of": len(diff),
    }
    report["chosen"] = {name: {p: sum(1 for _, v in rows if v and v.get(name) and v[name][1] == p) for p in templates} for name, templates in SETS.items()}
    report["_provenance"] = provenance.record(started)
    for name in SETS:
        print(name, round(report[name]["mean"], 2), [round(x, 2) for x in report[name]["ci"]], report["chosen"][name], flush=True)
    print("difference", report["difference"], flush=True)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
