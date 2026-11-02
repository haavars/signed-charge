"""Are the model's islands more than real's dots and short ridges? (E-122): island_share.py OUT.json

D-020's 200 prints by `defects_direct.count`, inside the contact: islands are forkless skeleton pieces under 30 px, about five ridge widths. Islands as a share of all the features, which needs no area, beside Gutiérrez-Redomero et al. 2011's Table 1: points and small fragments, 4.60 % of 145,794 hand-marked minutiae on 2,000 rolled prints.

Run: python -m signed_charge.exploratory.island_share OUT.json
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from signed_charge.measure import defects_direct as dd
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge import paths

JOBS, RESAMPLES = int(os.environ.get("READ_JOBS", 8)), 2000
POPULATION = paths.GENERATED / "population_d020"


def one(path):
    im = screen.print_of(np.load(path)["u"])
    got = dd.count(im)
    return {"pattern": path.stem.split("_")[0], "islands": got["islands"], "features": got["endings"] + got["forks"]}


def main(out):
    started = provenance.start()
    with ProcessPoolExecutor(JOBS) as pool:
        rows = list(pool.map(one, sorted(POPULATION.glob("*.npz")), chunksize=2))
    rng = np.random.default_rng(0)
    report = {}
    for pattern in ("loop", "whorl", "arch", "all"):
        picked = [r for r in rows if pattern in (r["pattern"], "all")]
        a, b = np.array([r["islands"] for r in picked], float), np.array([r["features"] for r in picked], float)
        share = lambda i: float(a[i].sum() / (a[i].sum() + b[i].sum()) * 100)  # noqa: E731
        draws = [share(rng.integers(len(a), size=len(a))) for _ in range(RESAMPLES)]
        report[pattern] = {
            "prints": len(a),
            "islands_a_print": float(a.mean()),
            "features_a_print": float(b.mean()),
            "share_percent": share(np.arange(len(a))),
            "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)],
        }
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    print({k: v for k, v in report.items() if k != "_provenance"})


if __name__ == "__main__":
    main(sys.argv[1])
