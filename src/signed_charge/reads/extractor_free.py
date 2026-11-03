"""Real prints' charge with no extractor in it, and minutiae against a finger's mean flow (D-061): extractor_free.py OUT.json

    charge   C(60), h(15-45), h(60-100) over each print's inner foreground, from `winding`'s phase residues with opposite pairs under 15 px or 6 px cancelled, beside mindtct and LEADER with and without the same cancelling. FVC DB1-3 B, SFinGe, model 2.
    mean     for `REFERENCES` impressions of each finger, the other seven aligned rigidly on minutiae (`signed_false.align`), their unit double angles turned, moved and averaged where at least `COVER` overlap. Agreement of the reference's minutiae, and |d|, on that flow and on its own, in the same cells, at each of `SCALES`. FVC DB1-3 B and SFinGe, the sets with eight impressions.

CI by finger. Run: python -m signed_charge.reads.extractor_free OUT.json
LIMIT=n takes the first n fingers of each set, to try the code.
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.reads import agreement_null as an
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_charge as fc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge.measure import winding as wd
from signed_charge import paths

JOBS = int(os.environ.get("READ_JOBS", 5))
MEAN_ONLY = bool(os.environ.get("MEAN_ONLY"))  # D-062 rereads only the mean flow: the charge never used `spreading`
LIMIT = int(os.environ.get("LIMIT", 0))
CACHE = str(paths.CACHE)
SETS, SCALES, REFERENCES, COVER, RESAMPLES = ("real", "sfinge", "model2"), (50, 70, 100, 150), (0, 4), 3, 2000
READS = ("res15", "res6", "mindtct", "leader", "mindtct15", "leader15")
ROWS = []


def charge(task):
    index, source = task
    row = ROWS[index]
    im = place.image(source)
    inner, fg = parts.mask_of(row), fs.foreground(im)
    z2 = sm.direction(im)
    field = z2
    z2 = z2 / (np.abs(z2) + 1e-9)
    b, V = wd.demodulate(im, fg, z2)
    amp = cv2.GaussianBlur(np.abs(b + 1j * np.imag(np.exp(-0.5j * np.angle(z2)) * V)).astype(np.float32), (0, 0), 4)
    strong = inner & (amp > wd.AMPLITUDE * np.median(amp[inner]))
    rx, ry, rq, rb = wd.residues(b, V, z2, strong)
    rh = wd.heading_of(rq, rb)
    points = {}
    for name, dist in (("res15", wd.DIPOLE_FAR), ("res6", wd.DIPOLE)):
        keep, _ = wd.cancel(rx, ry, rh, dist)
        points[name] = (np.c_[rx, ry, rh][keep], 1.0)
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(row["points"][reader], inner)
        points[reader] = (p, y_sign)
        keep, _ = wd.cancel(p[:, 0], p[:, 1], sf.heading(p, y_sign), wd.DIPOLE_FAR)
        points[reader + "15"] = (p[keep], y_sign)
    return row["finger"], {name: sf.binned(p, y_sign, field, inner) for name, (p, y_sign) in points.items()}


def state(row, source):
    im = place.image(source)
    inner, fg = parts.mask_of(row), fs.foreground(im)
    ys, xs = np.nonzero(inner)
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    guarded = inner & (np.hypot(xx - xs.mean(), yy - ys.mean()) <= fc.DISC)
    for x, y in sc.singular(fs.field(im), fg):
        guarded &= np.hypot(xx - x, yy - y) >= sc.GUARD
    z2 = sm.direction(im)
    return guarded, z2 / (np.abs(z2) + 1e-9), fg


def mean_flow(reference, others):
    """The others' double angles in the reference's frame: (their mean, how many cover each cell)."""
    rows, (_, zb, fb) = reference
    pb = rows["points"]["mindtct"]
    y_sign = sm.EXTRACTORS["mindtct"][1]
    h, w = zb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    total, cover = np.zeros((h, w), complex), np.zeros((h, w), int)
    for row, (_, za, fa) in others:
        pa = row["points"]["mindtct"]
        if len(pa) < sf.AT_LEAST or len(pb) < sf.AT_LEAST:
            continue
        move = sf.align(pa[:, :2], sf.heading(pa, y_sign), pb[:, :2], sf.heading(pb, y_sign))
        if move is None:
            continue
        turn, shift, _ = move
        co, si = np.cos(turn), np.sin(turn)
        back = (np.c_[xx.ravel(), yy.ravel()] - shift) @ np.array([[co, si], [-si, co]]).T
        xa, ya = np.rint(back[:, 0]).astype(int), np.rint(back[:, 1]).astype(int)
        ok = (xa >= 0) & (xa < za.shape[1]) & (ya >= 0) & (ya < za.shape[0])
        ok[ok] &= fa[ya[ok], xa[ok]]
        cells = np.flatnonzero(ok)
        total.ravel()[cells] += za[ya[ok], xa[ok]] * np.exp(2j * turn)
        cover.ravel()[cells] += 1
    return total, cover


def finger(task):
    items = [(ROWS[i], state(ROWS[i], source)) for i, source in task]
    out = []
    for r in REFERENCES:
        if r >= len(items):
            continue
        reference = items[r]
        row, (guarded, zb, _) = reference
        total, cover = mean_flow(reference, items[:r] + items[r + 1 :])
        cells = guarded & (cover >= COVER)
        fb = reference[1][2]
        if cells.sum() < 2000:
            continue
        got = {"finger": row["finger"]}
        for reader, (_, y_sign) in sm.EXTRACTORS.items():
            p = sf.inside(row["points"][reader], cells)
            got[reader] = {}
            for scale in SCALES:
                own, mean = fc.spreading(zb, float(scale), fb), fc.spreading(total, float(scale), cover >= COVER)
                got[reader][scale] = (an.agreement(p, y_sign, *own), an.agreement(p, y_sign, *mean))
                got[f"size{scale}"] = (float(np.abs(own[1][cells]).sum()), float(np.abs(mean[1][cells]).sum()), int(cells.sum()))
        out.append(got)
    return out


def boot(values, fingers, stat, rng):
    every = np.arange(len(fingers))
    draws = [stat(rng.integers(len(fingers), size=len(fingers))) for _ in range(RESAMPLES)]
    return {"mean": float(stat(every)), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}


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
            read = list(pool.map(charge, [t for f in chosen for t in fingers[f]], chunksize=2)) if not MEAN_ONLY else []
        where_f = {f: i for i, f in enumerate(chosen)}
        band = lambda a, b: (sm.EDGES[:-1] >= a) & (sm.EDGES[1:] <= b)  # noqa: E731
        entry = {"prints": len(read), "fingers": len(chosen)}
        for key in READS if not MEAN_ONLY else ():
            total, count, n = np.zeros((len(chosen), len(sm.EDGES) - 1)), np.zeros((len(chosen), len(sm.EDGES) - 1)), np.zeros(len(chosen))
            for f, got in read:
                total[where_f[f]] += got[key][0]
                count[where_f[f]] += got[key][1]
                n[where_f[f]] += got[key][2]
            entry[key] = {
                "per_print": float(n.sum() / len(read)),
                "c60": boot(total, chosen, lambda i: 2 * total[i][:, sm.EDGES[1:] <= 60].sum() / max(n[i].sum(), 1), rng),
                "h_short": boot(total, chosen, lambda i: total[i][:, band(*sf.SHORT)].sum() / max(count[i][:, band(*sf.SHORT)].sum(), 1), rng),
                "h_long": boot(total, chosen, lambda i: total[i][:, band(*sf.LONG)].sum() / max(count[i][:, band(*sf.LONG)].sum(), 1), rng),
            }
        report[name] = {"charge": entry}
        if name != "model2":
            with ProcessPoolExecutor(JOBS) as pool:
                rows = [r for group in pool.map(finger, [fingers[f] for f in chosen], chunksize=1) for r in group]
            where_r = sorted({r["finger"] for r in rows})
            at = {f: i for i, f in enumerate(where_r)}
            mean = {"references": len(rows), "fingers": len(where_r)}
            for scale in SCALES:
                sizes = np.zeros((len(where_r), 3))
                for r in rows:
                    sizes[at[r["finger"]]] += r[f"size{scale}"]
                mean[f"size{scale}"] = {
                    "own": boot(sizes, where_r, lambda i: sizes[i, 0].sum() / sizes[i, 2].sum(), rng),
                    "mean": boot(sizes, where_r, lambda i: sizes[i, 1].sum() / sizes[i, 2].sum(), rng),
                }
            for reader in sm.EXTRACTORS:
                mean[reader] = {}
                for scale in SCALES:
                    table = np.zeros((len(where_r), 3))
                    for r in rows:
                        (a, n), (b, _) = r[reader][scale]
                        table[at[r["finger"]]] += (a, b, n)
                    mean[reader][scale] = {
                        "own": boot(table, where_r, lambda i: table[i, 0].sum() / max(table[i, 2].sum(), 1), rng),
                        "mean": boot(table, where_r, lambda i: table[i, 1].sum() / max(table[i, 2].sum(), 1), rng),
                        "rise": boot(table, where_r, lambda i: (table[i, 1].sum() - table[i, 0].sum()) / max(table[i, 2].sum(), 1), rng),
                    }
            report[name]["mean"] = mean
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
