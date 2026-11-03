"""A batch of simulated fingertips against FVC, with model 2 at the same size as a control (E-029):
screen_batch.py OUT.json RUN.npz...

Runs are grouped by the `_s<sigma>_` in their names. Minutiae screens and the minutiae-histogram classifier
(`minutiae_stats`, `minutiae_pairs`), mindtct. The classifier depends on sample size (E-024), so model 2 is drawn at the
same number of prints (one capture each, first of each finger, seeds from 7000) and only those two are comparable.

Run: python -m signed_charge.exploratory.screen_batch OUT.json RUN.npz...
"""

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.generate import phase_masters as pm
from signed_charge.measure import provenance
from signed_charge.measure import screen


def one(path):
    name = Path(path).stem
    im = screen.print_of(np.load(path)["u"])
    key = (re.search(r"_s([\d.]+)_", name).group(1), name, 1)
    return key, ms.describe((key, im))[1], mp.describe((key, im))[1]


def line(name, s, p, accuracy=None):
    c = p["clustering"]
    print(
        name.ljust(16),
        "prints",
        s["prints"],
        "dens %.1f bif %.2f b/a %s"
        % (s["density_per_100mm2"], s["bifurcations"], s["near_core_below_over_above"] and round(s["near_core_below_over_above"], 2)),
        "acc",
        accuracy and round(accuracy, 3),
        "same/off/face %.2f %.2f %.2f" % (p["close_same"], p["close_offset"], p["close_facing"]),
        "EE/EB/BB %.2f %.2f %.2f" % (p["close_EE"], p["close_EB"], p["close_BB"]),
        "c10/20/40 %.2f %.2f %.2f" % (c[10], c[20], c[40]),
    )


def main(out, paths):
    started = provenance.start()
    real = list(cf.fvc(("DB1", "DB2", "DB3")))
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(one, paths))
        real_stats = [(k, v) for k, v in pool.map(ms.describe, real, chunksize=4) if v]
        real_pairs = [v for _, v in pool.map(mp.describe, real, chunksize=4) if v]
        size = len(rows) // len({k[0] for k, _, _ in rows})
        tasks = [("model2", s, f, 1) for s in range(7000, 7000 + size // 8 + 1) for f in (2, 3, 4, 5, 7, 8, 9, 10)][:size]
        control = list(pool.map(pm.tip, tasks, chunksize=2))
    report = {}
    s, p = ms.summary([v for _, v in real_stats]), mp.summary(real_pairs)
    line("real", s, p)
    report["real"] = {"stats": s, "pairs": p}
    s = [(k, v) for k, v, _, _ in control if v]
    a = cf.accuracy(real_stats, s, "mh")
    report["model2"] = {"stats": ms.summary([v for _, v in s]), "pairs": mp.summary([q for _, _, q, _ in control if q]), "accuracy_mh": a}
    line("model2", report["model2"]["stats"], report["model2"]["pairs"], a)
    for group in sorted({k[0] for k, _, _ in rows}):
        s = [(k, v) for k, v, _ in rows if k[0] == group and v]
        q = [x for k, _, x in rows if k[0] == group and x]
        a = cf.accuracy(real_stats, s, "mh")
        report[group] = {"stats": ms.summary([v for _, v in s]), "pairs": mp.summary(q), "accuracy_mh": a}
        line("sigma " + group, report[group]["stats"], report[group]["pairs"], a)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
