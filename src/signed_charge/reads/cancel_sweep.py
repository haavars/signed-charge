"""C(60) against the cancelling distance, for phase residues and both extractors (E-101): cancel_sweep.py OUT.json

Exploratory, on the validation review's request: `extractor_free.charge`'s reads with opposite pairs under each of `DISTANCES` px cancelled, so section 5.7's 15 px isn't a lone choice. FVC DB1-3 B, SFinGe, model 2. CI by finger.

Run: python -m signed_charge.reads.cancel_sweep OUT.json
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.reads import extractor_free as ef
from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge.measure import winding as wd

JOBS = int(os.environ.get("READ_JOBS", 8))
DISTANCES = (0, 4, 6, 8, 10, 12, 15, 20, 24)
READS = ("res", "mindtct", "leader")


def one(task):
    index, source = task
    row = ef.ROWS[index]
    im = place.image(source)
    inner, fg = parts.mask_of(row), fs.foreground(im)
    field = sm.direction(im)
    z2 = field / (np.abs(field) + 1e-9)
    b, V = wd.demodulate(im, fg, z2)
    amp = cv2.GaussianBlur(np.abs(b + 1j * np.imag(np.exp(-0.5j * np.angle(z2)) * V)).astype(np.float32), (0, 0), 4)
    strong = inner & (amp > wd.AMPLITUDE * np.median(amp[inner]))
    rx, ry, rq, rb = wd.residues(b, V, z2, strong)
    rh = wd.heading_of(rq, rb)
    base = {"res": (np.c_[rx, ry, rh], 1.0, rh)}
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(row["points"][reader], inner)
        base[reader] = (p, y_sign, sf.heading(p, y_sign))
    out = {"finger": row["finger"]}
    for name, (p, y_sign, heading) in base.items():
        for dist in DISTANCES:
            keep = wd.cancel(p[:, 0], p[:, 1], heading, dist)[0] if dist else np.ones(len(p), bool)
            total, _, n = sf.binned(p[keep], y_sign, field, inner)
            out[name, dist] = (float(total[sm.EDGES[1:] <= 60].sum()), n)
    return out


def main(out):
    started = provenance.start()
    base = pickle.load(open(ef.CACHE, "rb"))["rows"]
    where = place.sources(base)
    groups = {}
    for row in base:
        name = next((n for n in ef.SETS if parts.member(row, n)), None)
        if name and id(row) in where:
            ef.ROWS.append(row)
            groups.setdefault(name, []).append((row["finger"], (len(ef.ROWS) - 1, where[id(row)])))
    rng = np.random.default_rng(0)
    report = {}
    for name, items in groups.items():
        with ProcessPoolExecutor(JOBS) as pool:
            read = list(pool.map(one, [t for _, t in items], chunksize=2))
        fingers = sorted({r["finger"] for r in read})
        at = {f: i for i, f in enumerate(fingers)}
        report[name] = {}
        for key in READS:
            report[name][key] = {}
            for dist in DISTANCES:
                t = np.zeros((len(fingers), 2))
                for r in read:
                    t[at[r["finger"]]] += r[key, dist]
                stat = lambda i: 2 * t[i, 0].sum() / max(t[i, 1].sum(), 1)  # noqa: E731
                report[name][key][dist] = {
                    "per_print": float(t[:, 1].sum() / len(read)),
                    "c60": float(stat(np.arange(len(fingers)))),
                    "ci": [float(np.percentile([stat(rng.integers(len(fingers), size=len(fingers))) for _ in range(ef.RESAMPLES)], q)) for q in (2.5, 97.5)],
                }
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
