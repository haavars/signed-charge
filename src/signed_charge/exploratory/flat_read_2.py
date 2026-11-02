"""The second flat round (D-047): flat_read_2.py OUT.json RUN.npz...

As `flat_read`, adding over the sheet inside its margin density, h(15-45 px), h(60-100 px), C(60) and islands; and along the seam, for straight fronts at the sheet's middle, whether the sides are in step. For each row the ridges at `NEAR` cells either side of the seam are correlated, mirrored, over `PERIOD` rows; a stretch is in step if that's positive. Density by each reader in in-step and out-of-step stretches, within `BAND` cells of the seam. D-048 adds `aligned`, the mean cos 2Δ off the seam between the ridges and a uniform cue, 1 along it, -1 across, and each reader's share of forks on the seam.

Run: python -m signed_charge.exploratory.flat_read_2 OUT.json RUN.npz...
"""
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as nd
from scipy import stats
from skimage.morphology import skeletonize

from signed_charge.exploratory import density_seams as ds
from signed_charge.exploratory import flat_read as fr
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm

JOBS = int(os.environ.get("READ_JOBS", 16))
MARGIN, BAND, PERIOD = fr.MARGIN, fr.BAND, 10
NEAR = range(4, 11)


def in_step(u):
    """For each row, the correlation of the ridges mirrored across the middle column, averaged over `PERIOD` rows."""
    middle = u.shape[1] // 2
    ridges = np.where(u > np.median(u), 1.0, -1.0)
    left = np.stack([ridges[:, middle - 1 - d] for d in NEAR], 1)
    right = np.stack([ridges[:, middle + d] for d in NEAR], 1)
    return nd.uniform_filter1d((left * right).mean(1), PERIOD, mode="wrap")


def islands(im):
    """Whole ridges too small to be anything but a dot, per print: as `density_seams`, a skeleton component with no fork, under 30 px."""
    skeleton = skeletonize(im == 40)
    labels, n = nd.label(skeleton, np.ones((3, 3)))
    if not n:
        return 0
    return int((nd.sum(skeleton, labels, range(1, n + 1)) < 30).sum())


def one(path):
    name = re.sub(r"_\d+$", "", Path(path).stem)
    d = np.load(path)
    u = d["u"]
    im = np.where(u > np.median(u), 40, 235).astype(np.uint8)
    keep = np.zeros(u.shape, bool)
    keep[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    on, off = keep & (d["seam"] <= BAND), keep & (d["seam"] >= 2 * BAND)
    field = sm.direction(im)
    step = in_step(u)[:, None] > 0
    out = {"name": name, "area_mm2": float(keep.sum() * ms.MM2), "islands": islands(im), "in_step_share": float(step[MARGIN:-MARGIN].mean())}
    cue = np.pi / 2 - np.deg2rad(json.loads(str(d["settings"]))["cue"])  # from x, as the cue's degrees are from y
    ridge = np.angle(field) / 2 + np.pi / 2
    out["aligned"] = float(np.cos(2 * (ridge - cue))[off].mean())
    for reader, (extract, y_sign) in {"skeleton": (ds.skeleton_points, None), **sm.EXTRACTORS}.items():
        points = extract(im)
        for key, mask in (("on", on), ("off", off), ("all", keep), ("step", on & step), ("slip", on & ~step)):
            out[f"{reader}_{key}"] = len(sf.inside(points, mask)) / max(mask.sum() * ms.MM2, 1e-6) * 100
        if y_sign is None:
            continue
        seam_points = sf.inside(points, on)
        out[f"{reader}_forks_on"] = float(seam_points[:, 3].mean()) if len(seam_points) else 0.0
        for key, mask in (("on", on), ("all", keep)):
            total, count, n = sf.binned(sf.inside(points, mask), y_sign, field, keep)
            for band, (a, b) in {"short": sf.SHORT, "long": sf.LONG}.items():
                edges = (sm.EDGES[:-1] >= a) & (sm.EDGES[1:] <= b)
                out[f"{reader}_h{band}_{key}"] = [float(total[edges].sum()), float(count[edges].sum())]
            if key == "all":
                out[f"{reader}_c60"] = 2 * float(total[sm.EDGES[1:] <= 60].sum()) / max(n, 1)
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
            if isinstance(group[0][key], list):
                pooled = np.sum([r[key] for r in group], 0)
                entry[key] = {"mean": float(pooled[0] / max(pooled[1], 1)), "pairs": int(pooled[1])}
            else:
                values = np.array([r[key] for r in group], float)
                half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
                entry[key] = {"mean": float(values.mean()), "half": half}
        report[name] = entry
        short = lambda k: round(entry[k]["mean"], 1)  # noqa: E731
        print(name.ljust(22), "on/off", short("skeleton_on"), short("skeleton_off"), short("mindtct_on"), short("mindtct_off"), short("leader_on"), short("leader_off"), "all", short("mindtct_all"), short("leader_all"), "h", round(entry["mindtct_hshort_on"]["mean"], 2), round(entry["leader_hshort_on"]["mean"], 2), "whole", round(entry["mindtct_hshort_all"]["mean"], 2), round(entry["leader_hshort_all"]["mean"], 2), round(entry["mindtct_hlong_all"]["mean"], 2), round(entry["leader_hlong_all"]["mean"], 2), "c60", short("mindtct_c60"), short("leader_c60"), "isl", short("islands"), "step", round(entry["in_step_share"]["mean"], 2), "slip/step", short("mindtct_slip"), short("mindtct_step"), "aligned", round(entry["aligned"]["mean"], 2), flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
