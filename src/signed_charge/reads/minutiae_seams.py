"""Are the close same-sign pairs lined up on the seams? (D-041, E-072): minutiae_seams.py OUT.json RUN.npz...

A low-angle grain boundary in stripes is a chain of same-sign dislocations. Where a population print's waves meet, ridges may join that way. `arrival` is flattened as the print is, and k = 1 - |mean unit ∇T| over a Gaussian of `SCALE` px: 0 where a front passes evenly, rising where fronts cross. Seam is k ≥ `SEAM`, fixed from k maps alone; the rest of the inner foreground is off it.
For each print and extractor: minutiae density on and off; and h(15-45 px) and h(60-100 px) for pairs both on, and pairs both off. Density is a mean over prints of (on) / (off), and h pools pairs; CIs by print. All, and by pattern.

Run: python -m signed_charge.reads.minutiae_seams OUT.json RUN.npz...
"""
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import field_similarity as fs
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 8))
SCALE, BLUR, SEAM, RESAMPLES = 6.0, 2.0, 0.1, 2000


def kink(arrival):
    t, contact = screen.flattened(arrival)
    t = np.nan_to_num(t, nan=float(np.nanmax(t)))
    smooth = cv2.GaussianBlur(t.astype(np.float32), (0, 0), BLUR)
    gx, gy = cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    size = np.hypot(gx, gy) + 1e-6
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), SCALE)  # noqa: E731
    return 1 - np.hypot(blur(gx / size), blur(gy / size))


def one(path):
    name = Path(path).stem
    pattern = re.match(r"(\w+?)_a", name).group(1)
    d = np.load(path)
    im = screen.print_of(d["u"])
    fg = fs.foreground(im)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    on = (kink(d["arrival"]) >= SEAM) & inner
    off = inner & ~on
    field = sm.direction(im)
    out = {"pattern": pattern, "area_on": float(on.sum() / max(inner.sum(), 1))}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        points = sf.inside(extract(im), inner)
        row = {}
        for key, mask in (("on", on), ("off", off)):
            p = sf.inside(points, mask)
            total, count, n = sf.binned(p, y_sign, field, inner)
            row[key] = {"sum": total, "count": count, "minutiae": n, "density": n / max(mask.sum() * ms.MM2, 1e-6) * 100}
        out[reader] = row
    return out


def ratio(values, rng):
    measure = lambda a: a[:, 0].sum() / max(a[:, 1].sum(), 1)  # noqa: E731
    draws = [measure(values[d]) for d in (rng.integers(len(values), size=len(values)) for _ in range(RESAMPLES))]
    return {"mean": float(measure(values)), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}


def read(rows, reader, rng):
    report = {"prints": len(rows), "area_on": float(np.mean([r["area_on"] for r in rows]))}
    both = [r for r in rows if r[reader]["off"]["density"] > 0]
    enrich = np.array([[r[reader]["on"]["density"], r[reader]["off"]["density"]] for r in both])
    report["enrichment"] = ratio(enrich, rng)
    report["density"] = {"on": float(enrich[:, 0].mean()), "off": float(enrich[:, 1].mean())}
    for key in ("on", "off"):
        report[key] = {"minutiae": float(np.mean([r[reader][key]["minutiae"] for r in rows]))}
        for band, edges in (("h_short", sf.SHORT), ("h_long", sf.LONG)):
            keep = (sm.EDGES[:-1] >= edges[0]) & (sm.EDGES[1:] <= edges[1])
            report[key][band] = ratio(np.array([[r[reader][key]["sum"][keep].sum(), r[reader][key]["count"][keep].sum()] for r in rows]), rng)
            report[key][f"{band}_pairs"] = float(sum(r[reader][key]["count"][keep].sum() for r in rows))
    return report


def main(out, paths):
    started = provenance.start()
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(one, sorted(paths), chunksize=1))
    rng = np.random.default_rng(0)
    groups = {"all": rows, **{p: [r for r in rows if r["pattern"] == p] for p in ("arch", "loop", "whorl")}}
    report = {reader: {name: read(group, reader, rng) for name, group in groups.items()} for reader in sm.EXTRACTORS}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    for reader in sm.EXTRACTORS:
        for name, r in report[reader].items():
            print(reader, name.ljust(6), "area", round(r["area_on"], 2), "density", {k: round(v, 1) for k, v in r["density"].items()}, "enrich", round(r["enrichment"]["mean"], 2), [round(c, 2) for c in r["enrichment"]["ci"]],
                  {k: (round(r[k]["h_short"]["mean"], 3), [round(c, 3) for c in r[k]["h_short"]["ci"]], int(r[k]["h_short_pairs"]), round(r[k]["h_long"]["mean"], 3)) for k in ("on", "off")}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
