"""Two controls the review of E-108 asked for (D-071): review_controls.py OUT.json

    waves     vortices of a Gaussian random wave field, whose charges obey the screening sum rule exactly, read through real prints' windows at real's density: C(60), C(100), C(150), by the same sum of signs within R. Does a window hide screening?
    outside   Anguli and SynFing, C(60) with pairs dropped when their segment has coherence under 0.5, rescaled by pairs kept, as `quality_lift`, on each image's own inner print. Does SynFing stay above real on clear ridges?

Run: python -m signed_charge.reads.review_controls OUT.json
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_parts as parts
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import quality_lift as ql
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge.measure import winding as w
from signed_charge import paths

JOBS = int(os.environ.get("READ_JOBS", 8))
SIZE, WINDOWS, RADII, RESAMPLES = 1024, 120, (60, 100, 150), 2000
OUTSIDE = paths.GENERATED / "outside_d066"
MASKS = []


def field_of(seed, scale):
    """A complex Gaussian field with a Gaussian spectrum of width 1 / scale, periodic."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((SIZE, SIZE)) + 1j * rng.standard_normal((SIZE, SIZE))
    fy, fx = np.fft.fftfreq(SIZE)[:, None], np.fft.fftfreq(SIZE)[None, :]
    return np.fft.ifft2(np.fft.fft2(white) * np.exp(-((2 * np.pi * scale) ** 2) * (fx**2 + fy**2) / 2))


def vortices(z):
    a = np.angle(z)
    ring = [a, np.roll(a, -1, 1), np.roll(np.roll(a, -1, 0), -1, 1), np.roll(a, -1, 0), a]
    q = np.rint(sum(w.wrap(ring[i + 1] - ring[i]) for i in range(4)) / (2 * np.pi))
    ys, xs = np.nonzero(q)
    return xs + 0.5, ys + 0.5, q[ys, xs]


def wave(task):
    seed, scale = task
    rng = np.random.default_rng(seed + 10_000)
    xs, ys, qs = vortices(field_of(seed, scale))
    mask = MASKS[seed % len(MASKS)]
    h, w_ = mask.shape
    top, left = rng.integers(0, SIZE - h), rng.integers(0, SIZE - w_)
    inside = (ys >= top) & (ys < top + h) & (xs >= left) & (xs < left + w_)
    x, y, q = xs[inside] - left, ys[inside] - top, qs[inside]
    keep = mask[y.astype(int), x.astype(int)]
    x, y, q = x[keep], y[keep], q[keep]
    d = np.hypot(x[None] - x[:, None], y[None] - y[:, None])
    np.fill_diagonal(d, np.inf)
    product = q[None] * q[:, None]
    return len(x), [float(product[d < r].sum()) for r in RADII]


def outside(path):
    im = cv2.imread(str(path), 0)
    fg = fs.foreground(im)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    field, coherence = sm.direction(im), ql.coherence_of(im)
    out = {"set": path.stem.split("_")[0], "finger": path.stem}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(extract(im), inner)
        out[reader] = {c: ql.by_pairs(p, y_sign, field, inner, coherence, c) for c in (0.0, 0.5)}
    return out


def charge(rows, reader, least):
    total = sum(r[reader][least][0][: sm.EDGES.searchsorted(60)].sum() for r in rows)
    kept = sum(r[reader][least][1][: sm.EDGES.searchsorted(60)].sum() for r in rows)
    every = sum(r[reader][0.0][1][: sm.EDGES.searchsorted(60)].sum() for r in rows)
    n = sum(r[reader][least][3] for r in rows)
    return float(2 * total / max(n, 1) * every / max(kept, 1))


def main(out):
    started = provenance.start()
    rows = pickle.load(open(w.CACHE, "rb"))["rows"]
    MASKS.extend(parts.mask_of(r) for r in [r for r in rows if parts.member(r, "real")][:200])
    report = {}
    rng = np.random.default_rng(0)
    with ProcessPoolExecutor(JOBS) as pool:
        for scale in (8, 12, 18):  # px; the read reports density, so the nearest to real's is chosen after
            read = list(pool.map(wave, [(seed, scale) for seed in range(WINDOWS)], chunksize=4))
            n = np.array([r[0] for r in read], float)
            s = np.array([r[1] for r in read])
            entry = {"windows": WINDOWS, "vortices_a_window": float(n.mean())}
            for i, r in enumerate(RADII):
                draws = [s[j, i].sum() / n[j].sum() for j in (rng.integers(WINDOWS, size=WINDOWS) for _ in range(RESAMPLES))]
                entry[f"C{r}"] = {"mean": float(s[:, i].sum() / n.sum()), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}
            report[f"waves_{scale}"] = entry
            print(scale, json.dumps(entry)[:300], flush=True)
        paths = sorted(OUTSIDE.glob("*"))
        read = list(pool.map(outside, paths, chunksize=2))
    for name in ("synfing", "anguli"):
        group = [r for r in read if r["set"] == name]
        report[name] = {"prints": len(group)}
        for reader in sm.EXTRACTORS:
            report[name][reader] = {}
            for least in (0.0, 0.5):
                draws = [charge([group[i] for i in rng.integers(len(group), size=len(group))], reader, least) for _ in range(RESAMPLES)]
                report[name][reader][str(least)] = {"mean": charge(group, reader, least), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}
        print(name, json.dumps(report[name])[:400], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
