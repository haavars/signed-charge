"""What two fronts leave where they meet (D-045, E-076): flat_read.py OUT.json RUN.npz...

Each run's ridges, as `screen.print_of`, inside its margin of `MARGIN` cells. On the seam is within `BAND` cells of where the fronts meet, off it at least 2 × `BAND` away. Density by the skeleton and by mindtct and LEADER, per 100 mm²; h(15-45 px) for pairs both on, and both off, by mindtct and LEADER. Runs are named SETTING_seed.npz, and rows are their means with CIs over seeds, from a t distribution.

Run: python -m signed_charge.exploratory.flat_read OUT.json RUN.npz...
"""

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from signed_charge.exploratory import density_seams as ds
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 16))
MARGIN, BAND = 60, 24


def one(path):
    name = re.sub(r"_\d+$", "", Path(path).stem)
    d = np.load(path)
    u = d["u"]
    im = np.where(u > np.median(u), 40, 235).astype(np.uint8)
    keep = np.zeros(u.shape, bool)
    keep[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    on, off = keep & (d["seam"] <= BAND), keep & (d["seam"] >= 2 * BAND)
    field = sm.direction(im)
    out = {"name": name}
    readers = {"skeleton": (ds.skeleton_points, None), **sm.EXTRACTORS}
    for reader, (extract, y_sign) in readers.items():
        points = extract(im)
        for key, mask in (("on", on), ("off", off)):
            p = sf.inside(points, mask)
            out[f"{reader}_{key}"] = len(p) / max(mask.sum() * ms.MM2, 1e-6) * 100
            if y_sign is not None:
                total, count, _ = sf.binned(p, y_sign, field, keep)
                edges = (sm.EDGES[:-1] >= sf.SHORT[0]) & (sm.EDGES[1:] <= sf.SHORT[1])
                out[f"{reader}_h_{key}"] = [float(total[edges].sum()), float(count[edges].sum())]
    return out


def main(out, paths):
    started = provenance.start()
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(one, sorted(paths), chunksize=1))
    report = {}
    for name in sorted({r["name"] for r in rows}):
        group = [r for r in rows if r["name"] == name]
        entry = {"runs": len(group)}
        for key in group[0]:
            if key == "name":
                continue
            if key.split("_")[1] == "h":
                pooled = np.sum([r[key] for r in group], 0)
                each = [r[key][0] / r[key][1] for r in group if r[key][1]]
                entry[key] = {
                    "mean": float(pooled[0] / max(pooled[1], 1)),
                    "pairs": int(pooled[1]),
                    "sd": float(np.std(each, ddof=1)) if len(each) > 1 else None,
                }
            else:
                values = np.array([r[key] for r in group])
                half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
                entry[key] = {"mean": float(values.mean()), "half": half}
        report[name] = entry
        print(
            name.ljust(24),
            entry["runs"],
            "skeleton",
            round(entry["skeleton_on"]["mean"], 1),
            round(entry["skeleton_off"]["mean"], 1),
            "mindtct",
            round(entry["mindtct_on"]["mean"], 1),
            round(entry["mindtct_off"]["mean"], 1),
            "leader",
            round(entry["leader_on"]["mean"], 1),
            round(entry["leader_off"]["mean"], 1),
            "h on",
            round(entry["mindtct_h_on"]["mean"], 2),
            round(entry["leader_h_on"]["mean"], 2),
            "off",
            round(entry["mindtct_h_off"]["mean"], 2),
            round(entry["leader_h_off"]["mean"], 2),
            flush=True,
        )
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
