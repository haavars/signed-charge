"""Does the model put minutiae where real fingers have them? (D-081): region_read.py OUT.json

D-020's 200 prints, speed 0.7, by pattern, and FVC DB1-3 B as the control for the regions. Density per 100 mm² inside the eroded foreground; for a print with one core, density below and above the core; for a loop with one core and one delta, density in the lower half on the delta's side and on the other. Mean by finger, with a CI by resampling fingers.

Run: python -m signed_charge.exploratory.region_read OUT.json
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge import paths

JOBS, RESAMPLES = int(os.environ.get("READ_JOBS", 8)), 2000
POPULATION = paths.GENERATED / "population_d020"


def density(points, mask):
    return len(sf.inside(points, mask)) / max(mask.sum() * ms.MM2, 1e-6) * 100 if mask.sum() else None


def delta_right(theta, lower):
    """A loop's ridges below the core lean to its opening, and its delta lies on the other side. Ridges running down to the left, sin 2θ below 0 with y down, open left."""
    return float(np.sin(2 * theta[lower]).mean()) < 0


def regions(im, name, finger, pattern):
    fg = fs.foreground(im)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    theta = fs.field(im)
    cores = fs.cores(theta, inner)
    deltas = fs.cores(np.mod(-theta, np.pi), inner)  # the same read with the index's sign turned, so merged as cores are
    out = {"set": name, "finger": finger, "pattern": pattern, "cores": len(cores), "deltas": len(deltas)}
    ys, xs = np.mgrid[: im.shape[0], : im.shape[1]]
    for reader, (extract, *_) in sm.EXTRACTORS.items():
        points = extract(im)
        entry = {"all": density(points, inner)}
        if len(cores) == 1:
            cx, cy = cores[0]
            entry["below"], entry["above"] = density(points, inner & (ys > cy)), density(points, inner & (ys <= cy))
            entry["n_below"], entry["n_above"] = len(sf.inside(points, inner & (ys > cy))), len(sf.inside(points, inner & (ys <= cy)))
            right = delta_right(theta, inner & (ys > cy))  # most deltas lie too near the edge for Poincaré, so the side comes from the flow
            side = xs > cx if right else xs <= cx
            entry["delta_side"], entry["open_side"] = density(points, inner & (ys > cy) & side), density(points, inner & (ys > cy) & ~side)
            if len(deltas) == 1:
                out["rule_agrees"] = bool((deltas[0][0] > cx) == right)
        out[reader] = entry
    return out


def model(path):
    return regions(screen.print_of(np.load(path)["u"]), "model", path.stem, path.stem.split("_")[0])


def real(task):
    name, label, path = task
    return regions(fs.at_500(cv2.imread(path, 0), cf.PPI.get(name, 500)), name, label, "real")


def summary(rows, pick):
    """`pick` of each finger's mean, resampled by finger."""
    by = {}
    for r in rows:
        v = pick(r)
        if v is not None:
            by.setdefault(r["finger"], []).append(v)
    return {k: np.mean(v) for k, v in by.items()}


def boot(values, rng, stat=np.mean):
    values = np.array(list(values))
    draws = [stat(values[rng.integers(len(values), size=len(values))]) for _ in range(RESAMPLES)]
    return {"mean": float(stat(values)), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)], "n": len(values)}


def difference(a, b, rng):
    a, b = np.array(list(a)), np.array(list(b))
    draws = [a[rng.integers(len(a), size=len(a))].mean() - b[rng.integers(len(b), size=len(b))].mean() for _ in range(RESAMPLES)]
    return {"mean": float(a.mean() - b.mean()), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}


def ratio(rows, top, bottom, rng):
    """Mean of `top` over mean of `bottom`, paired by finger."""
    a, b = summary(rows, lambda r: r.get(top)), summary(rows, lambda r: r.get(bottom))
    keys = sorted(set(a) & set(b))
    pairs = np.array([(a[k], b[k]) for k in keys])
    if not len(pairs):
        return None
    draws = []
    for _ in range(RESAMPLES):
        s = pairs[rng.integers(len(pairs), size=len(pairs))]
        draws.append(s[:, 0].mean() / max(s[:, 1].mean(), 1e-6))
    return {
        "ratio": float(pairs[:, 0].mean() / max(pairs[:, 1].mean(), 1e-6)),
        "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)],
        "fingers": len(pairs),
    }


def main(out):
    started = provenance.start()
    tasks = [(name, label, p) for name, label, paths in sf.fvc_fingers() if "DB4" not in name for p in sorted(paths)]
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(model, sorted(POPULATION.glob("*.npz")), chunksize=2)) + list(pool.map(real, tasks, chunksize=4))
    rng = np.random.default_rng(0)
    report = {}
    groups = {p: [r for r in rows if r["pattern"] == p] for p in ("loop", "whorl", "arch", "real")}
    for reader in sm.EXTRACTORS:
        entry = {}
        means = {p: summary(g, lambda r: r[reader]["all"]) for p, g in groups.items()}
        entry["density"] = {p: boot(v.values(), rng) for p, v in means.items()}
        entry["whorl_minus_loop"] = difference(means["whorl"].values(), means["loop"].values(), rng)
        entry["loop_minus_arch"] = difference(means["loop"].values(), means["arch"].values(), rng)
        for p, g in groups.items():
            flat = [{**r[reader], "finger": r["finger"]} for r in g]
            entry[p] = {
                "below_over_above": ratio(flat, "below", "above", rng),
                "counts_below_over_above": ratio(flat, "n_below", "n_above", rng),
                "delta_over_open": ratio(flat, "delta_side", "open_side", rng),
            }
        report[reader] = entry
    report["prints"] = {
        p: {
            "prints": len(g),
            "one_core": sum(r["cores"] == 1 for r in g),
            "one_core_one_delta": sum(r["cores"] == 1 and r["deltas"] == 1 for r in g),
            "rule_agrees": sum(r.get("rule_agrees", False) for r in g),
        }
        for p, g in groups.items()
    }
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "_provenance"}, indent=0)[:6000])


if __name__ == "__main__":
    main(sys.argv[1])
