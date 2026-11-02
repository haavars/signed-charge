"""Does signed charge survive without false minutiae? (D-036, E-068): signed_false.py OUT.json

A ridge broken by noise leaves two facing endings of opposite charge, and false minutiae of mixed sign dilute a one-sign domain, so noise alone lowers C(R) and h(r). E-040's re-inked control keeps both. Three filters, each applied to every set alike:

- `confirmed`: minutiae found again in at least `SUPPORT` of a finger's seven other impressions. A false minutia made by noise doesn't recur. Impressions are aligned pairwise by a rigid move, on votes from `align`.
- `clean`: minutiae and pair segments on mindtct's top-quality blocks only, quality 4 of its map, for both extractors.
- `apart`: all minutiae, leaving out pairs under `DIPOLE` px, which is where a broken ridge's endings lie.

Sets: FVC DB1-3 B, SFinGe's DB4 B, and model 2 and E-027's best on fresh people, `FRESH`, seeds used nowhere before, eight captures of each of 8 fingers of each, as FVC's 8 impressions. Filters change how many minutiae remain, and C(R) counts neighbours, so h(r), a mean, is read beside C(60). All CIs by finger.
The formula generators' `all` row is also D-034's item: it's measured against real's A band from `held_out_look1.json`, which needs no A image, so no look is spent.

Run: python -m signed_charge.reads.signed_false OUT.json
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_check as fc
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("SIGNED_JOBS", 8))
SUPPORT, DIPOLE = 3, 15  # impressions, px
TOLERANCE, TURN = 15.0, np.deg2rad(30)  # how near a minutia must fall, and how far its direction may differ
ROTATIONS = np.deg2rad(np.arange(-30, 31, 5))
BIN, REACH, AT_LEAST, SHARE = (
    8,
    240,
    8,
    0.55,
)  # the vote's grid, its reach in px; aligned takes at least this many matches and this share of the smaller print's minutiae, as different fingers reach 0.54 at most, same fingers 0.75 at the median
FINGERS = (2, 3, 4, 5, 7, 8, 9, 10)
FRESH = range(9000, 9012)
CAPTURES = range(1, 9)
VARIANTS = sm.VARIANTS
RESAMPLES = 2000
SHORT, LONG = (15, 45), (60, 100)


def quality(im):
    """mindtct's quality for each pixel, from its map of 8 px blocks."""
    with tempfile.TemporaryDirectory() as tmp:
        h, w = im.shape
        np.ascontiguousarray(im, dtype=np.uint8).tofile(f"{tmp}/i.raw")
        subprocess.run([cf.NBIS / "cjpegl", "jpl", f"{tmp}/i.raw", "-raw_in", f"{w},{h},8,500"], check=True, capture_output=True)
        subprocess.run([cf.NBIS / "mindtct", "-m1", f"{tmp}/i.jpl", f"{tmp}/o"], check=True, capture_output=True)
        blocks = np.loadtxt(f"{tmp}/o.qm").astype(np.uint8)
    return cv2.resize(blocks, (blocks.shape[1] * 8, blocks.shape[0] * 8), interpolation=cv2.INTER_NEAREST)[:h, :w]


def heading(points, y_sign):
    """Minutia direction in image coordinates, the same for both extractors."""
    return np.arctan2(y_sign * np.sin(points[:, 2]), np.cos(points[:, 2]))


def wrapped(a):
    return np.abs((a + np.pi) % (2 * np.pi) - np.pi)


def align(a, ha, b, hb):
    """The rigid move taking minutiae a onto b, or None: for each turn, translations are voted on by pairs of like direction, and the best is refined. Returns (turn, shift, count)."""
    top = None
    size = 2 * REACH // BIN
    for turn in ROTATIONS:
        co, si = np.cos(turn), np.sin(turn)
        moved = a @ np.array([[co, si], [-si, co]])
        d = b[None] - moved[:, None]
        like = wrapped(hb[None] - (ha + turn)[:, None]) < TURN
        vote = d[like]
        cell = np.floor((vote + REACH) / BIN).astype(int)
        keep = ((cell >= 0) & (cell < size)).all(1)
        if not keep.any():
            continue
        grid = np.zeros((size, size), np.float32)
        np.add.at(grid, (cell[keep, 1], cell[keep, 0]), 1)
        grid = cv2.blur(grid, (3, 3))
        y, x = np.unravel_index(grid.argmax(), grid.shape)
        centre = (np.array([x, y]) + 0.5) * BIN - REACH
        near = np.linalg.norm(vote - centre, axis=1) < 2 * BIN
        if not near.any():
            continue
        shift = vote[near].mean(0)
        count = hits(moved + shift, ha + turn, b, hb).sum()
        if top is None or count > top[2]:
            top = (turn, shift, count)
    return top if top is not None and top[2] >= max(AT_LEAST, SHARE * min(len(a), len(b))) else None


def hits(moved, heads, b, hb):
    """Which of the moved minutiae have one of b's near them, of like direction."""
    d = np.linalg.norm(moved[:, None] - b[None], axis=2)
    return ((d < TOLERANCE) & (wrapped(hb[None] - heads[:, None]) < TURN)).any(1)


def support(points, y_sign):
    """For each impression's minutiae, how many of the finger's other impressions hold them too."""
    heads = [heading(p, y_sign) if len(p) else np.zeros(0) for p in points]
    counts = [np.zeros(len(p), int) for p in points]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if len(points[i]) < AT_LEAST or len(points[j]) < AT_LEAST:
                continue
            move = align(points[i][:, :2], heads[i], points[j][:, :2], heads[j])
            if move is None:
                continue
            turn, shift, _ = move
            co, si = np.cos(turn), np.sin(turn)
            rotate = np.array([[co, si], [-si, co]])
            moved = points[i][:, :2] @ rotate + shift
            counts[i] += hits(moved, heads[i] + turn, points[j][:, :2], heads[j])
            back = (points[j][:, :2] - shift) @ rotate.T
            counts[j] += hits(back, heads[j] - turn, points[i][:, :2], heads[i])
    return counts


def binned(points, y_sign, field, mask):
    if len(points) < 2:
        return np.zeros(len(sm.EDGES) - 1), np.zeros(len(sm.EDGES) - 1), len(points)
    r, q = sm.charge_pairs(points, y_sign, field, mask)
    total, count = sm.binned(r, q)
    return total, count, len(points)


def inside(points, mask):
    if not len(points):
        return points
    xi, yi = points[:, 0].astype(int), points[:, 1].astype(int)
    h, w = mask.shape
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    ok[ok] &= mask[yi[ok], xi[ok]]
    return points[ok]


def finger(task):
    """One finger's impressions: for each extractor and filter, pair sums by distance and minutiae counted."""
    name, label, images = task
    ready = []
    for im in images:
        fg = fs.foreground(im)
        if fg is None:
            continue
        inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
        ready.append((im, inner, sm.direction(im), quality(im)))
    out = {
        "set": name,
        "finger": label,
        "prints": len(ready),
        "top_quality_share": float(np.mean([(q[m] == 4).mean() for _, m, _, q in ready])) if ready else 0.0,
    }
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        points = [inside(extract(im), m) for im, m, _, _ in ready]
        counts = support(points, y_sign)
        sums = {k: [np.zeros(len(sm.EDGES) - 1), np.zeros(len(sm.EDGES) - 1), 0] for k in ("all", "confirmed", "clean")}
        confirmed = 0
        for p, c, (_, m, field, q) in zip(points, counts, ready):
            keep = p[c >= SUPPORT] if len(p) else p
            confirmed += len(keep)
            clean = m & (q == 4)
            for key, (use, mask) in {"all": (p, m), "confirmed": (keep, m), "clean": (inside(p, clean), clean)}.items():
                total, count, n = binned(use, y_sign, field, mask)
                sums[key][0] += total
                sums[key][1] += count
                sums[key][2] += n
        out[reader] = {k: {"sum": v[0], "count": v[1], "minutiae": v[2]} for k, v in sums.items()}
        out[reader]["confirmed_share"] = confirmed / max(sum(len(p) for p in points), 1)
    return out


def fvc_fingers():
    out = {}
    for year in ("FVC2000", "FVC2002", "FVC2004"):
        for db in ("DB1", "DB2", "DB3", "DB4"):
            name = f"{year}/{db}_B"
            for path in sorted(glob.glob(f"{cf.FVC}/{name}/*.tif")):
                out.setdefault((name, f"{name}/{Path(path).stem.split('_')[0]}"), []).append(path)
    return [(name, label, paths) for (name, label), paths in out.items()]


def fvc_task(task):
    name, label, paths = task
    return finger((name, label, [fs.at_500(cv2.imread(p, 0), cf.PPI.get(name, 500)) for p in paths]))


def variant_task(task):
    variant, seed, fgp = task
    return finger((variant, f"PH{seed}/{fgp}", [fc.image((variant, seed, fgp, c))[1] for c in CAPTURES]))


def band(rows, key, reader, edges):
    keep = (sm.EDGES[:-1] >= edges[0]) & (sm.EDGES[1:] <= edges[1])
    return np.array([[r[reader][key]["sum"][keep].sum(), r[reader][key]["count"][keep].sum()] for r in rows])


def interval(values, measure, rng):
    draws = [measure(values[d]) for d in (rng.integers(len(values), size=len(values)) for _ in range(RESAMPLES))]
    return {"mean": float(measure(values)), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}


def read(rows, reader, rng):
    report = {
        "fingers": len(rows),
        "prints": sum(r["prints"] for r in rows),
        "top_quality_share": float(np.mean([r["top_quality_share"] for r in rows])),
        "confirmed_share": float(np.mean([r[reader]["confirmed_share"] for r in rows])),
    }
    ratio = lambda a: a[:, 0].sum() / max(a[:, 1].sum(), 1)  # noqa: E731
    for key in ("all", "confirmed", "clean"):
        near = sm.EDGES[1:] <= 60
        apart = near & (sm.EDGES[:-1] >= DIPOLE)
        c60 = np.array([[2 * r[reader][key]["sum"][near].sum(), r[reader][key]["minutiae"]] for r in rows])
        far = np.array([[2 * r[reader][key]["sum"][apart].sum(), r[reader][key]["minutiae"]] for r in rows])
        report[key] = {
            "minutiae": float(np.mean([r[reader][key]["minutiae"] / max(r["prints"], 1) for r in rows])),
            "c60": interval(c60, ratio, rng),
            "c60_apart": interval(far, ratio, rng),
            "h_short": interval(band(rows, key, reader, SHORT), ratio, rng),
            "h_long": interval(band(rows, key, reader, LONG), ratio, rng),
        }
    return report


def main(out):
    started = provenance.start()
    fingers = [(v, s, f) for v in VARIANTS for s in FRESH for f in FINGERS]
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(fvc_task, fvc_fingers(), chunksize=1))
        print("fvc", len(rows), flush=True)
        rows += list(pool.map(variant_task, fingers, chunksize=1))
    groups = {
        "real": [r for r in rows if r["set"].startswith("FVC") and "DB4" not in r["set"]],
        "sfinge": [r for r in rows if "DB4" in r["set"]],
        **{v: [r for r in rows if r["set"] == v] for v in VARIANTS},
    }
    for name in sorted({r["set"] for r in rows if r["set"].startswith("FVC") and "DB4" not in r["set"]}):
        groups[name] = [r for r in rows if r["set"] == name]
    rng = np.random.default_rng(0)
    report = {reader: {name: read(group, reader, rng) for name, group in groups.items()} for reader in sm.EXTRACTORS}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    for reader in sm.EXTRACTORS:
        for name, r in report[reader].items():
            print(
                reader,
                name.ljust(16),
                r["fingers"],
                "q4",
                round(r["top_quality_share"], 2),
                "conf",
                round(r["confirmed_share"], 2),
                {
                    k: (
                        round(r[k]["minutiae"], 1),
                        round(r[k]["c60"]["mean"], 2),
                        round(r[k]["c60_apart"]["mean"], 2),
                        round(r[k]["h_short"]["mean"], 3),
                        round(r[k]["h_long"]["mean"], 3),
                    )
                    for k in ("all", "confirmed", "clean")
                },
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1])
