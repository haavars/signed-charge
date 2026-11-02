"""At what angle do the waves meet? (E-053): seam_angles.py OUT.json

Along each line where two sites' waves meet, the angle between their fronts, from `competency_model.site_arrivals`, inside the part of the finger a print shows. Nothing is simulated, so settings can be scanned in seconds. Fronts meeting head on and parallel, 0°, can join quietly; at an angle stripes
laid on each side can't. Weighted by length of seam.

Run: python -m signed_charge.exploratory.seam_angles OUT.json
"""

import json
import sys

import numpy as np

from signed_charge.generate import competency_model as c
from signed_charge.generate import glover_model as g
from signed_charge.measure import provenance
from signed_charge.measure import screen

SEEDS = (0, 4501, 4502, 4505, 4506, 4507, 4508, 4509)


def angles(times, index):
    """Crossing angle of the fronts at each grid cell where the first and second sites change, inside the print."""
    t = times[:, index]  # site, s, φ
    first = np.argmin(t, 0)
    du = np.gradient(t, axis=1) / 2.0
    dv = np.gradient(t, axis=2) / 2.0
    edge = (first[1:, 1:] != first[:-1, 1:]) | (first[1:, 1:] != first[1:, :-1])
    ns, nphi = first.shape
    top = ns - (screen.TOP + screen.HEIGHT)
    keep = np.zeros_like(edge)
    keep[top : ns - screen.TOP, (nphi - screen.WIDTH) // 2 : (nphi + screen.WIDTH) // 2] = True
    out = []
    for i, j in zip(*np.nonzero(edge & keep)):
        a, b = first[i + 1, j + 1], first[i, j + 1] if first[i, j + 1] != first[i + 1, j + 1] else first[i + 1, j]
        u = np.array([du[a, i, j], dv[a, i, j]])
        v = np.array([du[b, i, j], dv[b, i, j]])
        cosine = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
        out.append(180 - np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return np.array(out)


def main(out):
    started = provenance.start()
    points, _, s, _, index = g.mesh(2.0)
    report = {}
    for pattern in ("loop", "whorl", "arch"):
        collected = []
        for seed in SEEDS:
            times = c.site_arrivals(points, index, s, pattern, 0.7, c.person_of(seed, 1))
            collected.append(angles(times, index))
        all_ = np.concatenate(collected)
        report[pattern] = {
            "median": float(np.median(all_)),
            "p90": float(np.percentile(all_, 90)),
            "over_30": float(np.mean(all_ > 30)),
            "seam_px": float(len(all_) / len(SEEDS)),
        }
        print(pattern, {k: round(v, 2) for k, v in report[pattern].items()}, flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
