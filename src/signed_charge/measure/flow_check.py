"""Do phase-drawn masters keep the flow, and does step 4's wander combine with them? (E-025):
flow_check.py OUT [PEOPLE] [FIRST_SEED]

Conformality (`conformality.py`, on mindtct's binarised images) and the flow tail (`field_similarity`, as
`flow_variants.py`), with the minutiae screens, on fingertips inked flat, against FVC DB1-3 B: master model 2; the
frozen phase-drawn settings (D-005) with the half-size solver; and those on a flow with step 4's wander (0.08 rad at
60 px, `flow_variants.py`).

Run: python -m signed_charge.measure.flow_check OUT.json 20 7000 [VARIANT...]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.generate import flow_variants as fv
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.generate import phase_masters as pm
from signed_charge.measure import provenance

FROZEN = "m3_clump0.5_below1.6"
pm.GLOBAL["m3_wander"] = pm.GLOBAL[FROZEN]
VARIANTS = ["model2", FROZEN, "m3_wander"]
JOBS = 8


def image(task):
    from ridgegen import finger as fingers
    from ridgegen import fingerprints
    from ridgegen.synthesis import rng_for

    variant, seed, fgp, capture = task
    if hasattr(pm.use, "master"):
        fingers.master = pm.use.master  # back to the cached original, which `flow_variants.use` clears
    fv.use("fine0.08" if variant.endswith("wander") else "model2", seed, fgp)
    pm.use(variant)
    rng = rng_for(seed, fgp, fingerprints.PLAIN, capture - 1, 13 if fgp <= 5 else 14)
    _, ridges, contact, _, _, pasted = fingerprints.plain_finger(seed, fgp, capture - 1, rng)
    return (variant, f"PH{seed}/{fgp}", capture), np.where((contact > 0.5) & ~pasted & (ridges > 0), 40, 235).astype(np.uint8)


def tip(task):
    key, im = image(task)
    _, stats = ms.describe((key, im))
    _, pairs = mp.describe((key, im))
    return key, stats, pairs, fv.describe((key, im))[1:]


def main(out, people=20, first=7000, *variants):
    started = provenance.start()
    if variants:
        VARIANTS[:] = variants
    real_items = [((k[0], k[1], k[2]), im) for k, im in cf.fvc(("DB1", "DB2", "DB3"))]
    tasks = [(v, s, f, c) for v in VARIANTS for s in range(first, first + people) for f in (2, 3, 4, 5, 7, 8, 9, 10)
             for c in (1, 2)]
    with ProcessPoolExecutor(JOBS) as pool:
        real_flow = list(pool.map(fv.describe, real_items, chunksize=4))
        real_stats = [(k, v) for k, v in pool.map(ms.describe, real_items, chunksize=4) if v]
        mine = list(pool.map(tip, tasks, chunksize=2))
    real_tail = fv.tail([(("FVC", k[1], k[2]), c, f) for k, c, f in real_flow])
    p1, p5 = np.percentile(real_tail, 1), np.percentile(real_tail, 5)
    real_c = [(k, c) for k, c, _ in real_flow if c]
    # Each variant's workers need the list: tasks carry the variant, so only this process's copy changed.
    report = {"real": {**fv.shares([c for _, c in real_c]), "between_median": float(np.median(real_tail))}}
    for v in VARIANTS:
        xs = [x for x in mine if x[0][0] == v]
        flows = [(k, c, f) for k, _, _, (c, f) in xs]
        t = fv.tail(flows)
        cs = [(k, c) for k, c, _ in flows if c]
        stats = [(k, s) for k, s, _, _ in xs if s]
        report[v] = {**fv.shares([c for _, c in cs]), "accuracy_h10": cf.accuracy(real_c, cs, "h10"),
                     "between_median": float(np.median(t)), "below_fvc_p1": float(np.mean(t < p1)),
                     "below_fvc_p5": float(np.mean(t < p5)), "minutiae": ms.summary([s for _, s in stats]),
                     "accuracy_mh": cf.accuracy(real_stats, stats, "mh"),
                     "pairs": {k: val for k, val in mp.summary([p for _, _, p, _ in xs if p]).items() if k != "nn_share"}}
        print(json.dumps({v: {k: val for k, val in report[v].items() if k not in ("minutiae", "pairs")}}), flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[2:]
    main(sys.argv[1], *[int(a) if a.isdigit() else a for a in args])
