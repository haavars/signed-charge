"""Minutiae spacing and pairing, real against synthetic: the targets for step 3's point process (route b,
): minutiae_pairs.py OUT [PEOPLE]

Step 2 found Phantom's close pairs pointing the same way where real ones face each other across neighbouring
ridges. A point process that places minutiae needs those statistics fitted, so for each print (NBIS mindtct,
reliability at least 0.2, at least 16 px inside the foreground, as `minutiae_stats.py`):

- **Nearest neighbours**: each minutia's distance to its nearest other, in 5 px bins to 100 px.
- **Clustering**: pairs within r (10-80 px) against as many points placed uniformly in the same foreground (one
  draw a print), so edge effects cancel; above 1 is clustered, below 1 inhibited.
- **Close pairs** (under 40 px): their relative direction (same < 18°, offset 18-36°, facing > 144°), their types
  (ending-ending, ending-fork, fork-fork) and how many ridges lie between them along the segment joining them, on
  mindtct's binarised image (0: on one ridge, 1: neighbouring ridges, 2 or more).

Sets: FVC 2000, 2002, 2004 DB1-3 B (real), DB4 B (SFinGe), Phantom's rendered fingertips (0.25).

Run: python -m signed_charge.reads.minutiae_pairs OUT.json
[PEOPLE]
"""

import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import provenance
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure.conformality import fs

NN_BINS = np.arange(0, 105, 5)
RADII = (10, 20, 30, 40, 60, 80)
CLOSE = 40
JOBS = 8


def mindtct(im):
    """mindtct's minutiae (x, y, angle, bifurcation) at reliability ≥ `ms.RELIABILITY`, and its binarised image
    (ridges True), from one run."""
    with tempfile.TemporaryDirectory() as tmp:
        h, w = im.shape
        np.ascontiguousarray(im, dtype=np.uint8).tofile(f"{tmp}/i.raw")
        subprocess.run([cf.NBIS / "cjpegl", "jpl", f"{tmp}/i.raw", "-raw_in", f"{w},{h},8,500"], check=True, capture_output=True)
        subprocess.run([cf.NBIS / "mindtct", "-b", "-m1", f"{tmp}/i.jpl", f"{tmp}/b"], check=True, capture_output=True)
        b = np.fromfile(f"{tmp}/b.brw", np.uint8).reshape(h, w)
        rows = []
        for line in Path(f"{tmp}/b.min").read_text().splitlines()[4:]:
            parts = [p.strip() for p in line.split(":")]
            if len(parts) < 5:
                continue
            x, y = (int(v) for v in parts[1].split(","))
            if float(parts[3]) >= ms.RELIABILITY:
                rows.append((x, y, np.deg2rad(90 - int(parts[2]) * 11.25), parts[4] == "BIF"))
    # The .brw: ridges 0 (black) when it's 0/255, ridges 1 when it's 0/1.
    ridges = b > 0 if b.max() <= 1 else b == 0
    return np.array(rows, dtype=float).reshape(-1, 4), ridges


def ridges_between(ridges, a, b):
    """Ridges crossed going from a to b (x, y), not counting the ones a and b are on."""
    n = int(np.ceil(np.hypot(*(b - a)))) * 2 + 1
    xs = np.clip(np.round(np.linspace(a[0], b[0], n)).astype(int), 0, ridges.shape[1] - 1)
    ys = np.clip(np.round(np.linspace(a[1], b[1], n)).astype(int), 0, ridges.shape[0] - 1)
    v = ridges[ys, xs].astype(int)
    # Drop the runs at each end (the minutiae's own ridges, or the valley an ending faces into), then count the
    # ridge runs left between.
    inner = v[np.argmax(v != v[0]) :][::-1]
    inner = inner[np.argmax(inner != inner[0]) :][::-1] if len(inner) else inner
    if len(inner) == 0 or (v == v[0]).all():
        return 0
    return int(np.sum(np.diff(np.r_[0, inner]) == 1))


def pairs_within(xy, r):
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    return int((d[np.triu_indices(len(xy), 1)] < r).sum())


def describe(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, None
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1,) * 2, np.uint8)).astype(bool)
    if inner.sum() * ms.MM2 < 40:
        return key, None
    points, ridges = mindtct(im)
    if len(points):
        xi, yi = points[:, 0].astype(int), points[:, 1].astype(int)
        ok = (xi >= 0) & (xi < inner.shape[1]) & (yi >= 0) & (yi < inner.shape[0])
        ok[ok] &= inner[yi[ok], xi[ok]]
        points = points[ok]
    if len(points) < 5:
        return key, None
    xy = points[:, :2]
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    nn = np.histogram(d.min(1), NN_BINS)[0]
    # As many points uniformly in the same foreground: the pair counts' baseline.
    rng = np.random.default_rng(len(points))
    cells = np.argwhere(inner)
    uniform = cells[rng.choice(len(cells), len(points), replace=False)][:, ::-1] + rng.random((len(points), 2))
    clustering = {r: [pairs_within(xy, r), pairs_within(uniform, r)] for r in RADII}
    close = {
        "same": 0,
        "offset": 0,
        "facing": 0,
        "other": 0,
        "EE": 0,
        "EB": 0,
        "BB": 0,
        "across0": 0,
        "across1": 0,
        "across2": 0,
        "facing_across1": 0,
        "facing_EB": 0,
    }
    for i, j in zip(*np.triu_indices(len(points), 1)):
        if d[i, j] >= CLOSE:
            continue
        da = np.degrees(abs((points[i, 2] - points[j, 2] + np.pi) % (2 * np.pi) - np.pi))
        kind = "same" if da < 18 else "offset" if da < 36 else "facing" if da > 144 else "other"
        close[kind] += 1
        types = ("EE", "EB", "BB")[int(points[i, 3] + points[j, 3])]
        close[types] += 1
        across = min(ridges_between(ridges, xy[i], xy[j]), 2)
        close[f"across{across}"] += 1
        if kind == "facing":
            close["facing_across1"] += across == 1
            close["facing_EB"] += types == "EB"
    return key, {"count": len(points), "area_mm2": float(inner.sum() * ms.MM2), "nn": nn.tolist(), "clustering": clustering, "close": close}


def summary(xs):
    nn = np.sum([x["nn"] for x in xs], 0)
    close = {k: sum(x["close"][k] for x in xs) for k in xs[0]["close"]}
    n = max(close["same"] + close["offset"] + close["facing"] + close["other"], 1)
    facing = max(close["facing"], 1)
    clustering = {r: float(sum(x["clustering"][r][0] for x in xs) / max(sum(x["clustering"][r][1] for x in xs), 1)) for r in xs[0]["clustering"]}
    return {
        "prints": len(xs),
        "density_per_100mm2": float(np.median([x["count"] / x["area_mm2"] * 100 for x in xs])),
        "nn_median_px": float(np.interp(0.5, np.cumsum(nn) / nn.sum(), NN_BINS[1:])),
        "nn_share": (nn / nn.sum()).round(4).tolist(),
        "clustering": clustering,
        "close_pairs_per_print": n / len(xs),
        **{f"close_{k}": close[k] / n for k in ("same", "offset", "facing", "EE", "EB", "BB", "across0", "across1", "across2")},
        "facing_across1": close["facing_across1"] / facing,
        "facing_EB": close["facing_EB"] / facing,
    }


def main(out, people=60):
    started = provenance.start()
    sources = list(cf.fvc(("DB1", "DB2", "DB3"))) + list(cf.fvc(("DB4",))) + list(cf.phantom(people))
    with ProcessPoolExecutor(JOBS) as pool:
        described = [(k, v) for k, v in pool.map(describe, sources, chunksize=4) if v]
    sets = {}
    for k, v in described:
        name = "real" if k[0].startswith("FVC") and "DB4" not in k[0] else ("sfinge" if "DB4" in k[0] else k[0])
        sets.setdefault(name, []).append(v)
        if name == "real":
            sets.setdefault(k[0], []).append(v)
    report = {name: summary(xs) for name, xs in sets.items()}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "nn_share"} for k, v in report.items()}, indent=1))
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
