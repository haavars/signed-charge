"""Minutiae statistics, real against synthetic (step 2): minutiae_stats.py OUT [PEOPLE]

Plain prints, one extractor for all (NBIS mindtct, reliability at least 0.2, at least 16 px inside the foreground):

- Density, minutiae per 100 mm² of foreground; the share of bifurcations (mindtct's type: endings against
  bifurcations, as FVC's 36-56 % against SFinGe's 29-32 % were measured with one extractor);
  density within 5 mm of the core below it against above it (real rolled prints have about twice as many below:
  57.8 against 29.2 per print, Oregon, inked); prints with one core.
- Each print's minutiae histogram (Gottschlich & Huckemann 2014: all pairs' distance and angle difference, 10 × 10
  bins up to 200 px and 180°), and a linear classifier real against each synthetic set on it, from one impression,
  cross-validated by finger (as `conformality.py`). Real minutiae configurations aren't uniform; SFinGe's are
  separable by them.

Sets: FVC 2000, 2002, 2004 DB1-3 B (real) and DB4 B (SFinGe); Phantom's rendered slap fingertips
(`impressions-slaps-value-03`, ridgegen 0.25). And Phantom's rolled prints' ground truth (every minutia drawn, and
mindtct's count on the rendered print), against the published 85-115 a rolled print (inked 85, SD27 mates 106,
SD302 112-113).

Run: python -m signed_charge.measure.minutiae_stats OUT.json
[PEOPLE | --procedural PEOPLE VARIANT...] (OMP_NUM_THREADS=1 with --procedural)
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
from signed_charge.measure.conformality import fs

from ridgegen import verify
from signed_charge import paths

MM2 = (25.4 / 500) ** 2
RELIABILITY = 0.2
BORDER = 16
NEAR_CORE_MM = 5
MH_DISTANCE, MH_BINS = 200, 10
ROLLED = str(paths.RENDERED)
JOBS = 8


def mindtct(im):
    """(x, y, angle in radians, bifurcation) of mindtct's minutiae at reliability ≥ `RELIABILITY`."""
    with tempfile.TemporaryDirectory() as tmp:
        h, w = im.shape
        raw, jpl, root = f"{tmp}/i.raw", f"{tmp}/i.jpl", f"{tmp}/i"
        np.ascontiguousarray(im, dtype=np.uint8).tofile(raw)
        # As `verify.mindtct`: lossless JPEG, which mindtct reads.
        verify.run([verify.nbis_tool("cjpegl"), "jpl", raw, "-raw_in", f"{w},{h},8,500"])
        verify.run([verify.nbis_tool("mindtct"), "-m1", jpl, root])
        rows = []
        for line in Path(f"{root}.min").read_text().splitlines()[4:]:
            parts = [p.strip() for p in line.split(":")]
            if len(parts) < 5:
                continue
            x, y = (int(v) for v in parts[1].split(","))
            direction, reliability, kind = int(parts[2]), float(parts[3]), parts[4]
            if reliability >= RELIABILITY:
                # mindtct's direction: 0-31 in 11.25° steps, 0 pointing up, clockwise.
                rows.append((x, y, np.deg2rad(90 - direction * 11.25), kind == "BIF"))
    return np.array(rows, dtype=float).reshape(-1, 4)


def histogram(points):
    """The minutiae histogram: all pairs' distance (px) and angle difference (0-180°), normalised."""
    h = np.zeros((MH_BINS, MH_BINS))
    if len(points) < 2:
        return h.ravel()
    xy, a = points[:, :2], points[:, 2]
    i, j = np.triu_indices(len(points), 1)
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    da = np.abs((a[i] - a[j] + np.pi) % (2 * np.pi) - np.pi)
    keep = d < MH_DISTANCE
    di = np.clip((d[keep] / MH_DISTANCE * MH_BINS).astype(int), 0, MH_BINS - 1)
    ai = np.clip((da[keep] / np.pi * MH_BINS).astype(int), 0, MH_BINS - 1)
    np.add.at(h, (di, ai), 1)
    return (h / max(1, h.sum())).ravel()


def describe(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, None
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * BORDER + 1, 2 * BORDER + 1), np.uint8)).astype(bool)
    if inner.sum() * MM2 < 40:
        return key, None
    points = mindtct(im)
    if len(points):
        xi, yi = points[:, 0].astype(int), points[:, 1].astype(int)
        h, w = inner.shape
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        ok[ok] &= inner[yi[ok], xi[ok]]
        points = points[ok]
    out = {"count": int(len(points)), "area_mm2": float(inner.sum() * MM2),
           "density": float(len(points) / (inner.sum() * MM2) * 100),
           "bifurcations": float(points[:, 3].mean()) if len(points) else None, "mh": histogram(points).tolist()}
    cores = fs.cores(fs.field(im), fg)
    if len(cores) == 1:
        cx, cy = cores[0]
        yy, xx = np.mgrid[0:inner.shape[0], 0:inner.shape[1]]
        near = inner & (np.hypot(xx - cx, yy - cy) < NEAR_CORE_MM * 500 / 25.4)
        above, below = near & (yy < cy), near & (yy >= cy)
        if above.sum() * MM2 > 10 and below.sum() * MM2 > 10:
            r = np.hypot(points[:, 0] - cx, points[:, 1] - cy) < NEAR_CORE_MM * 500 / 25.4
            out["above"] = float(np.sum(r & (points[:, 1] < cy)) / (above.sum() * MM2) * 100)
            out["below"] = float(np.sum(r & (points[:, 1] >= cy)) / (below.sum() * MM2) * 100)
    return key, out


def procedural_tip(task):
    """A plain fingertip inked flat from master model 2 (as `field_similarity.procedural`), its masters grown as
    `minutiae_growth`'s `variant`, described."""
    from signed_charge.generate import minutiae_growth as growth

    from ridgegen import fingerprints
    from ridgegen.synthesis import rng_for

    variant, seed, fgp, capture = task
    growth.use(variant)
    rng = rng_for(seed, fgp, fingerprints.PLAIN, capture - 1, 13 if fgp <= 5 else 14)
    _, ridges, contact, _, _, pasted = fingerprints.plain_finger(seed, fgp, capture - 1, rng)
    im = np.where((contact > 0.5) & ~pasted & (ridges > 0), 40, 235).astype(np.uint8)
    return describe(((f"procedural-{variant}", f"PH{seed}/{fgp}", capture), im))


def rolled_truth(people=60):
    """Phantom's rolled prints' ground truth: minutiae drawn, how many mindtct found of them, and how many lie
    below and above the core (the topmost core of a whorl)."""
    rows = []
    for person in sorted(glob.glob(f"{ROLLED}/PH-*"))[:people]:
        for path in sorted(glob.glob(f"{person}/ground_truth/fgp??_*.json")):
            if path.endswith(("_tip.json", "_c2.json")):
                continue
            t = json.load(open(path))
            if t.get("impression") != "rolled":
                continue
            ms = t["minutiae"]
            row = {"count": len(ms), "detected": sum(1 for m in ms if m.get("detected")),
                   "bifurcations": float(np.mean([m["type"] == "bifurcation" for m in ms])) if ms else None}
            if t.get("cores"):
                cy = min(c[1] for c in t["cores"])
                row["above"] = sum(1 for m in ms if m["y"] < cy)
                row["below"] = sum(1 for m in ms if m["y"] >= cy)
            rows.append(row)
    return rows


def summary(xs):
    def med(k):
        v = [x[k] for x in xs if x.get(k) is not None]
        return float(np.median(v)) if v else None

    ratio = [x["below"] / x["above"] for x in xs if x.get("above")]
    # Pairs under 40 px: what share point the same way (< 18°), are offset (18-36°: a bifurcation beside an
    # ending, as growth leaves them) or face each other (> 144°: breaks, short ridges, dots, enclosures).
    mh = np.mean([np.array(x["mh"]).reshape(MH_BINS, MH_BINS) for x in xs], 0)
    close = mh[:2].sum(0) / max(mh[:2].sum(), 1e-12)
    pairs = {"close_same": float(close[0]), "close_offset": float(close[1]), "close_opposite": float(close[-2:].sum())}
    return {**pairs, "prints": len(xs), "count": med("count"), "density_per_100mm2": med("density"),
            "bifurcations": med("bifurcations"), "near_core_below_over_above": float(np.median(ratio)) if ratio else None,
            "with_core": len(ratio)}


def main(out, args):
    """`args`: [PEOPLE], or --procedural PEOPLE VARIANT... (`minutiae_growth.VARIANTS`) for fingertips drawn
    without the GPU from masters grown each way, instead of the rendered ones."""
    procedural = "--procedural" in args
    if procedural:
        people, *variants = args[args.index("--procedural") + 1:]
        people = int(people)
    else:
        people = int(args[0]) if args else 60
    sources = list(cf.fvc(("DB1", "DB2", "DB3"))) + list(cf.fvc(("DB4",)))
    if not procedural:
        sources += list(cf.phantom(people))
    with ProcessPoolExecutor(JOBS) as pool:
        described = [(k, v) for k, v in pool.map(describe, sources, chunksize=4) if v]
        if procedural:
            tasks = [(v, seed, fgp, c) for v in variants for seed in range(3000, 3000 + people)
                     for fgp in (2, 3, 4, 5, 7, 8, 9, 10) for c in (1, 2)]
            described += [(k, v) for k, v in pool.map(procedural_tip, tasks, chunksize=2) if v]
    sets = {}
    for k, v in described:
        name = "real" if k[0].startswith("FVC") and "DB4" not in k[0] else ("sfinge" if "DB4" in k[0] else k[0])
        sets.setdefault(name, []).append((k, v))
    report = {name: summary([v for _, v in xs]) for name, xs in sets.items()}
    for name, xs in sets.items():
        if name != "real":
            report[name]["accuracy_mh"] = cf.accuracy(sets["real"], xs, "mh")
    rolled = [] if procedural else rolled_truth(people)
    ratio = [r["below"] / r["above"] for r in rolled if r.get("above")]
    report["phantom_rolled_truth"] = None if procedural else {
        "prints": len(rolled), "drawn_median": float(np.median([r["count"] for r in rolled])),
        "detected_median": float(np.median([r["detected"] for r in rolled])),
        "bifurcations_median": float(np.median([r["bifurcations"] for r in rolled if r["bifurcations"] is not None])),
        "below_over_above_median": float(np.median(ratio)) if ratio else None,
        "published": "85-115 a rolled print; about twice as many below the core as above (57.8 / 29.2)"}
    print(json.dumps(report, indent=1))
    json.dump({"report": report, "prints": [[list(k), {kk: vv for kk, vv in v.items() if kk != "mh"}]
                                            for k, v in described]}, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
