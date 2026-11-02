"""Is real prints' near balance image quality? (D-063): quality_lift.py OUT.json

    lift      triangles of minutiae under `SIDE` px, segments on the guarded print, no singular point inside: the product of the three lifted signs should be +1. Its failures give the chance p that a lifted pair is wrong, by 3p(1-p)² + p³, so a read h is (1 - 2p) of the true. By set, CI by finger.
    filter    h(15-45), h(60-100), C(60) with pairs dropped when any point of the segment has coherence under each of `COHERENCE`; C rescaled by the pairs kept.
    quality   each finger's h(15-45) against its share of quality 4 cells, Spearman, within real and within SFinGe.
    degrade   `field` and `realflow` prints given a real print's window and its ridge amplitude: contrast scaled by a, a in 0-1, noise of s.d. `NOISE`(1 - a), then both extractors. realflow takes the print its flow came from, field a real print in turn. C(60) clean and degraded in the same window, beside the real print through it, CI by finger.

Run: python -m signed_charge.reads.quality_lift OUT.json
LIMIT=n takes the first n fingers of each set, to try the code.
"""
import glob
import json
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

from signed_charge.reads import field_read as fr
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import minutiae_cache as mc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge import paths

JOBS = int(os.environ.get("READ_JOBS", 6))
LIMIT = int(os.environ.get("LIMIT", 0))
CACHE = str(paths.CACHE)
SETS, SIDE, TRIANGLES, COHERENCE, NOISE, N, RESAMPLES = ("real", "sfinge", "model2"), 100.0, 300, (0.0, 0.3, 0.5), 40.0, 432, 2000
ROWS = []
BAND = (sm.EDGES[:-1] >= sf.SHORT[0]) & (sm.EDGES[1:] <= sf.SHORT[1])
LONG = (sm.EDGES[:-1] >= sf.LONG[0]) & (sm.EDGES[1:] <= sf.LONG[1])
NEAR = sm.EDGES[1:] <= 60


def coherence_of(im):
    u = cv2.GaussianBlur(im.astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(u, cv2.CV_32F, 1, 0), cv2.Sobel(u, cv2.CV_32F, 0, 1)
    smooth = lambda a: cv2.GaussianBlur(a, (0, 0), sm.ORIENTATION_SCALE)  # noqa: E731
    return np.abs(smooth(gx * gx - gy * gy) + 2j * smooth(gx * gy)) / (smooth(gx * gx + gy * gy) + 1e-6)


def segment(a, b, r):
    n = max(int(r / 3), 4)
    t = np.linspace(0, 1, n)
    return np.round(a[0] + t * (b[0] - a[0])).astype(int), np.round(a[1] + t * (b[1] - a[1])).astype(int)


def lift(p, i, j, heads, field, mask, coherence=None, least=0.0):
    """s_i s_j along the segment, or None if it leaves the mask or, with `coherence`, falls under `least`."""
    r = np.hypot(*(p[i, :2] - p[j, :2]))
    x, y = segment(p[i], p[j], r)
    h, w = field.shape
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    if not mask[y, x].all() or (coherence is not None and coherence[y, x].min() < least):
        return None
    ridge = np.unwrap(np.angle(field[y, x])) / 2 + np.pi / 2
    ends = np.stack([np.cos(ridge[[0, -1]]), np.sin(ridge[[0, -1]])], 1)
    return float(np.sign(heads[i] @ ends[0]) * np.sign(heads[j] @ ends[1]))


def inside_triangle(q, a, b, c):
    side = lambda p1, p2: (q[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (q[1] - p2[1])  # noqa: E731
    d1, d2, d3 = side(a, b), side(b, c), side(c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def triangles(p, y_sign, field, mask, singular, rng):
    heads = np.stack([np.cos(p[:, 2]), y_sign * np.sin(p[:, 2])], 1)
    good = bad = 0
    if len(p) < 3:
        return good, bad
    for _ in range(TRIANGLES * 4):
        i = int(rng.integers(len(p)))
        d = np.hypot(*(p[:, :2] - p[i, :2]).T)
        near = np.flatnonzero((d < SIDE) & (d > 0))
        if len(near) < 2:
            continue
        j, k = rng.choice(near, 2, replace=False)
        if np.hypot(*(p[j, :2] - p[k, :2])) >= SIDE or any(inside_triangle(s, p[i], p[j], p[k]) for s in singular):
            continue
        q = [lift(p, a, b, heads, field, mask) for a, b in ((i, j), (j, k), (k, i))]
        if None in q:
            continue
        good, bad = (good + 1, bad) if np.prod(q) > 0 else (good, bad + 1)
        if good + bad >= TRIANGLES:
            break
    return good, bad


def by_pairs(p, y_sign, field, mask, coherence, least):
    heads = np.stack([np.cos(p[:, 2]), y_sign * np.sin(p[:, 2])], 1)
    total, count = np.zeros(len(sm.EDGES) - 1), np.zeros(len(sm.EDGES) - 1)
    found = 0
    for i in range(len(p) - 1):
        d = np.hypot(*(p[i + 1 :, :2] - p[i, :2]).T)
        for j in np.nonzero(d < sm.EDGES[-1])[0] + i + 1:
            found += 1
            q = lift(p, i, j, heads, field, mask, coherence, least)
            if q is None:
                continue
            b = min(np.searchsorted(sm.EDGES, np.hypot(*(p[i, :2] - p[j, :2])), side="right") - 1, len(total) - 1)
            total[b] += q
            count[b] += 1
    return total, count, found, len(p)


def one(task):
    index, source = task
    row = ROWS[index]
    im = place.image(source)
    inner, fg = parts.mask_of(row), fs.foreground(im)
    singular = sc.singular(fs.field(im), fg)
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    guarded = inner.copy()
    for x, y in singular:
        guarded &= np.hypot(xx - x, yy - y) >= sc.GUARD
    field, coherence = sm.direction(im), coherence_of(im)
    rng = np.random.default_rng(zlib.crc32((row["finger"] + str(index)).encode()))
    out = {"finger": row["finger"], "q4": float((sf.quality(im)[inner] >= 4).mean())}
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(row["points"][reader], inner)
        out[reader] = {"triangles": triangles(sf.inside(p, guarded), y_sign, field, guarded, singular, rng), "filter": {c: by_pairs(p, y_sign, field, inner, coherence, c) for c in COHERENCE}}
    return out


def placed(im, mask, centre):
    dy, dx = N / 2 - centre[0], N / 2 - centre[1]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(im, M, (N, N), flags=cv2.INTER_NEAREST, borderValue=255), cv2.warpAffine(mask.astype(np.uint8), M, (N, N), flags=cv2.INTER_NEAREST).astype(bool)


def amplitude(im, fg, inner):
    g = im.astype(np.float32)
    b = cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 6.0)
    a = np.sqrt(cv2.GaussianBlur(b * b, (0, 0), 8.0))
    return np.clip(a / np.percentile(a[inner], 75), 0, 1) * fg


def degrade(task):
    path, index, source = task
    row = ROWS[index]
    real = place.image(source)
    fg, inner = fs.foreground(real), parts.mask_of(row)
    ys, xs = np.nonzero(fg)
    centre = min(((c[1], c[0]) for c in fs.cores(fs.field(real), fg)), default=(ys.mean(), xs.mean()))
    a, window = placed(amplitude(real, fg, inner), inner, centre)
    realim, _ = placed(real, inner, centre)
    u = np.load(path)["u"]
    clean = np.where(u > np.median(u), 40, 235).astype(np.float32)
    rng = np.random.default_rng(zlib.crc32(path.encode()))
    bad = 128 + (clean - 128) * a + rng.normal(0, NOISE, clean.shape) * (1 - a)
    bad = np.clip(np.where(a > 0, bad, 255), 0, 255).astype(np.uint8)
    clean = clean.astype(np.uint8)
    out = {"finger": row["finger"], "q4": {}}
    for name, im in (("clean", clean), ("degraded", bad), ("real", realim)):
        got = mc.row("x", "x", im)
        field = sm.direction(im)
        out[name] = {}
        if not got:
            continue
        for reader, (_, y_sign) in sm.EXTRACTORS.items():
            out[name][reader] = sf.binned(sf.inside(got["points"][reader], window), y_sign, field, window)
        out["q4"][name] = float((sf.quality(im)[window] >= 4).mean())
    return out


def boot(fingers, stat, rng):
    every = np.arange(len(fingers))
    return {"mean": float(stat(every)), "ci": [float(np.percentile([stat(rng.integers(len(fingers), size=len(fingers))) for _ in range(RESAMPLES)], q)) for q in (2.5, 97.5)]}


def main(out):
    started = provenance.start()
    base = pickle.load(open(CACHE, "rb"))["rows"]
    where = place.sources(base)
    groups = {}
    for row in base:
        name = next((n for n in SETS if parts.member(row, n)), None)
        if name is None or id(row) not in where:
            continue
        ROWS.append(row)
        groups.setdefault(name, {}).setdefault(row["finger"], []).append((len(ROWS) - 1, where[id(row)]))
    rng = np.random.default_rng(0)
    report = {}
    for name, fingers in groups.items():
        chosen = sorted(fingers)[: LIMIT or None]
        with ProcessPoolExecutor(JOBS) as pool:
            read = list(pool.map(one, [t for f in chosen for t in fingers[f]], chunksize=2))
        at = {f: i for i, f in enumerate(chosen)}
        entry = {"prints": len(read), "fingers": len(chosen)}
        q4 = np.zeros(len(chosen)), np.zeros(len(chosen))
        for r in read:
            q4[0][at[r["finger"]]] += r["q4"]
            q4[1][at[r["finger"]]] += 1
        for reader in sm.EXTRACTORS:
            tri = np.zeros((len(chosen), 2))
            for r in read:
                tri[at[r["finger"]]] += r[reader]["triangles"]
            def pair_error(i):
                e = tri[i, 1].sum() / max(tri[i].sum(), 1)
                grid = np.linspace(0, 0.5, 501)
                return float(grid[np.argmin(np.abs(3 * grid * (1 - grid) ** 2 + grid**3 - e))])
            e = {"triangles": int(tri.sum()), "fail": boot(chosen, lambda i: tri[i, 1].sum() / max(tri[i].sum(), 1), rng), "p": boot(chosen, pair_error, rng), "filter": {}}
            for c in COHERENCE:
                t = np.zeros((len(chosen), len(sm.EDGES) - 1)); k = t.copy(); f = np.zeros(len(chosen)); n = f.copy()
                for r in read:
                    a, b, x, y = r[reader]["filter"][c]
                    t[at[r["finger"]]] += a; k[at[r["finger"]]] += b; f[at[r["finger"]]] += x; n[at[r["finger"]]] += y
                e["filter"][c] = {"kept": float(k.sum() / max(f.sum(), 1)), "h_short": boot(chosen, lambda i: t[i][:, BAND].sum() / max(k[i][:, BAND].sum(), 1), rng), "h_long": boot(chosen, lambda i: t[i][:, LONG].sum() / max(k[i][:, LONG].sum(), 1), rng), "c60_rescaled": boot(chosen, lambda i: 2 * t[i][:, NEAR].sum() / max(n[i].sum(), 1) * f[i].sum() / max(k[i].sum(), 1), rng)}
            t = np.zeros((len(chosen), len(sm.EDGES) - 1)); k = t.copy()
            for r in read:
                a, b, *_ = r[reader]["filter"][0.0]
                t[at[r["finger"]]] += a; k[at[r["finger"]]] += b
            h = t[:, BAND].sum(1) / np.maximum(k[:, BAND].sum(1), 1)
            rho, pv = stats.spearmanr(q4[0] / q4[1], h)
            e["spearman_q4_h"] = [float(rho), float(pv)]
            entry[reader] = e
        entry["q4"] = float((q4[0] / q4[1]).mean())
        report[name] = entry
    real = [t[0] for f, t in sorted(groups["real"].items())]
    arms = {"realflow": [], "field": []}
    for path in sorted(glob.glob(str(paths.GENERATED / "flow_d055/realflow_*.npz"))):
        _, year, db, finger = Path(path).stem.split("_")
        items = groups["real"].get(f"{year}/{db}_B/{finger}")
        if items:
            arms["realflow"].append((path, *items[0]))
    for k, path in enumerate(sorted(glob.glob(str(paths.GENERATED / "field_d049/field_*.npz")))[:90]):
        arms["field"].append((path, *real[k % len(real)]))
    for arm, tasks in arms.items():
        with ProcessPoolExecutor(JOBS) as pool:
            read = list(pool.map(degrade, tasks[: LIMIT or None], chunksize=1))
        read = [r for r in read if all(n in r and sm.EXTRACTORS.keys() <= r[n].keys() for n in ("clean", "degraded", "real"))]
        entry = {"prints": len(read), "q4": {n: float(np.mean([r["q4"][n] for r in read])) for n in ("clean", "degraded", "real")}}
        every = list(range(len(read)))
        for reader in sm.EXTRACTORS:
            e = {}
            for n in ("clean", "degraded", "real"):
                t = np.array([r[n][reader][0] for r in read]); c = np.array([float(r[n][reader][2]) for r in read])
                e[n] = {"per_print": float(c.mean()), "c60": boot(every, lambda i: 2 * t[i][:, NEAR].sum() / max(c[i].sum(), 1), rng), "h_short": boot(every, lambda i: t[i][:, BAND].sum() / max(np.array([r[n][reader][1] for r in read])[i][:, BAND].sum(), 1), rng)}
            entry[reader] = e
        report[arm] = entry
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])
