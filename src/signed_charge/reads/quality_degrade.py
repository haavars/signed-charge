"""Generated prints at real prints' quality (D-064): quality_degrade.py OUT.json | quality_degrade.py --calibrate

`quality_lift.degrade`'s pairing and windows, with a recipe that lowers quality everywhere and not only where ink is missing: blur of `BLUR` px, contrast a, Gaussian noise of s.d. `FLOOR` + `NOISE`(1 - a) smoothed by 0.7 px. `--calibrate` prints only the quality 4 share and minutiae count over the settings, on prints not used after, so the recipe is set with C(60) unseen.

Run: python -m signed_charge.reads.quality_degrade OUT.json
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
from scipy.spatial import cKDTree

from signed_charge.reads import field_read as fr
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import minutiae_cache as mc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import quality_lift as ql
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge import paths

JOBS = int(os.environ.get("READ_JOBS", 8))
GRID = [(b, f, n) for b in (1.0, 1.6) for f in (20, 40, 60, 80) for n in (40, 80)]
RECIPE = (1.0, 40.0, 40.0)  # chosen by --calibrate: quality 4 share 0.60, real 0.63, C(60) unseen


def made(u, a, recipe, seed):
    blur, floor, noise = recipe
    clean = cv2.GaussianBlur(np.where(u > np.median(u), 40, 235).astype(np.float32), (0, 0), blur)
    rng = np.random.default_rng(seed)
    grain = cv2.GaussianBlur(rng.normal(0, 1, clean.shape).astype(np.float32), (0, 0), 0.7) / 0.28
    bad = 128 + (clean - 128) * a + grain * (floor + noise * (1 - a))
    return np.clip(np.where(a > 0, bad, 255), 0, 255).astype(np.uint8)


def setup(task):
    path, index, source = task
    row = ql.ROWS[index]
    real = place.image(source)
    fg, inner = fs.foreground(real), parts.mask_of(row)
    ys, xs = np.nonzero(fg)
    centre = min(((c[1], c[0]) for c in fs.cores(fs.field(real), fg)), default=(ys.mean(), xs.mean()))
    a, window = ql.placed(ql.amplitude(real, fg, inner), inner, centre)
    realim, _ = ql.placed(real, inner, centre)
    return row, np.load(path)["u"], a, window, realim, zlib.crc32(path.encode())


def calibrate(task):
    row, u, a, window, realim, seed = setup(task)
    out = {"real": float((sf.quality(realim)[window] >= 4).mean())}
    for recipe in GRID:
        im = made(u, a, recipe, seed)
        got = mc.row("x", "x", im)
        out[recipe] = (float((sf.quality(im)[window] >= 4).mean()), len(sf.inside(got["points"]["mindtct"], window)) if got else 0)
    return out


def degrade(task):
    row, u, a, window, realim, seed = setup(task)
    im = made(u, a, RECIPE, seed)
    out = {"finger": row["finger"], "q4": float((sf.quality(im)[window] >= 4).mean())}
    got = mc.row("x", "x", im)
    if not got:
        return None
    field = sm.direction(im)
    coherence = ql.coherence_of(im)
    a_, b_ = sf.inside(got["points"]["mindtct"], window), sf.inside(got["points"]["leader"], window)
    both = 0
    if len(a_) and len(b_):
        da, ia = cKDTree(b_[:, :2]).query(a_[:, :2])
        _, ib = cKDTree(a_[:, :2]).query(b_[:, :2])
        both = int(((da <= 10) & (ib[ia] == np.arange(len(a_)))).sum())
    out["agreed"] = (both, len(a_), len(b_))
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(got["points"][reader], window)
        out[reader] = {c: ql.by_pairs(p, y_sign, field, window, coherence, c) for c in ql.COHERENCE}
    return out


def tasks():
    base = pickle.load(open(ql.CACHE, "rb"))["rows"]
    where = place.sources(base)
    real = {}
    for row in base:
        if parts.member(row, "real") and id(row) in where:
            ql.ROWS.append(row)
            real.setdefault(row["finger"], []).append((len(ql.ROWS) - 1, where[id(row)]))
    flows = []
    for path in sorted(glob.glob(str(paths.GENERATED / "flow_d055/realflow_*.npz"))):
        _, year, db, finger = Path(path).stem.split("_")
        items = real.get(f"{year}/{db}_B/{finger}")
        if items:
            flows.append((path, *items[0]))
    order = [t[0] for f, t in sorted(real.items())]
    field = [(path, *order[k % len(order)]) for k, path in enumerate(sorted(glob.glob(str(paths.GENERATED / "field_d049/field_*.npz")))[:90])]
    spare = [(path, *order[k % len(order)]) for k, path in enumerate(sorted(glob.glob(str(paths.GENERATED / "field_d049/field_*.npz")))[90:102])]
    return {"realflow": flows, "field": field}, spare


def main(out):
    arms, spare = tasks()
    if out == "--calibrate":
        with ProcessPoolExecutor(JOBS) as pool:
            got = list(pool.map(calibrate, spare, chunksize=1))
        print("real q4", round(float(np.mean([g["real"] for g in got])), 3))
        for recipe in GRID:
            print(recipe, "q4", round(float(np.mean([g[recipe][0] for g in got])), 3), "minutiae", round(float(np.mean([g[recipe][1] for g in got])), 1))
        return
    started = provenance.start()
    rng = np.random.default_rng(0)
    report = {"recipe": RECIPE}
    for arm, items in arms.items():
        with ProcessPoolExecutor(JOBS) as pool:
            read = [r for r in pool.map(degrade, items, chunksize=1) if r]
        agreed = np.array([r["agreed"] for r in read], float).sum(0)
        entry = {"prints": len(read), "q4": float(np.mean([r["q4"] for r in read])), "agreed": [float(agreed[0] / agreed[1]), float(agreed[0] / agreed[2])]}
        every = list(range(len(read)))
        for reader in sm.EXTRACTORS:
            entry[reader] = {}
            for c in ql.COHERENCE:
                t = np.array([r[reader][c][0] for r in read]); k = np.array([r[reader][c][1] for r in read]); f = np.array([float(r[reader][c][2]) for r in read]); n = np.array([float(r[reader][c][3]) for r in read])
                entry[reader][c] = {"per_print": float(n.mean()), "kept": float(k.sum() / max(f.sum(), 1)), "h_short": ql.boot(every, lambda i: t[i][:, ql.BAND].sum() / max(k[i][:, ql.BAND].sum(), 1), rng), "c60": ql.boot(every, lambda i: 2 * t[i][:, ql.NEAR].sum() / max(n[i].sum(), 1) * f[i].sum() / max(k[i].sum(), 1), rng)}
        report[arm] = entry
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])
