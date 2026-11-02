"""Engineered variants on the signed measure and the fair classifier, against E-027's best, same people (E-044): variant_check.py OUT.json VARIANT...

Run: python -m signed_charge.reads.variant_check OUT.json e_vary e_far
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from signed_charge.reads import classifier_floor as cl
from signed_charge.measure import conformality as cf
from signed_charge.measure import flow_check as fc
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import signed_minutiae as sm
from signed_charge.measure import validate_masters as vm

BEST = "d_s1_sp2_f3"
PAIR_READERS = {"mindtct": mp.mindtct, "leader": lambda im: (vm.leader(im), cf.binarised(im) == 0)}


def both(item):
    key, im = item
    stats = cl.described(item)[1]
    pairs = {}
    for name, reader in PAIR_READERS.items():
        ms.mindtct, mp.mindtct = cl.EXTRACTORS[name], reader
        pairs[name] = mp.describe(item)[1]
    return key, stats, pairs, sm.measure(item)[1]


def rendered(task):
    return both(fc.image(task))


def main(out, variants):
    started = provenance.start()
    items = list(cf.fvc(("DB1", "DB2", "DB3")))
    tasks = [(v, s, f, c) for v in (BEST, *variants) for s in range(7000, 7020) for f in (2, 3, 4, 5, 7, 8, 9, 10) for c in (1, 2)]
    with ProcessPoolExecutor(8) as pool:
        real = list(pool.map(cl.described, items, chunksize=4))
        mine = list(pool.map(rendered, tasks, chunksize=2))
    rng = np.random.default_rng(0)
    report = {}
    for name in cl.EXTRACTORS:
        r = [(k, v[name]) for k, v in real if v[name]]
        report[name] = {}
        for variant in (BEST, *variants):
            rows = [x for x in mine if x[0][0] == variant]
            stats = [(k, s[name]) for k, s, _, _ in rows if s[name]]
            pairs = mp.summary([p[name] for _, _, p, _ in rows if p[name]])
            signed = sm.curve([q for _, _, _, q in rows if q], name, rng)
            report[name][variant] = {"accuracy": cl.spread([cf.accuracy(r, stats, "mh", seed=s) for s in range(cl.SEEDS)]),
                                     "density": pairs["density_per_100mm2"], "same": pairs["close_same"], "offset": pairs["close_offset"], "facing": pairs["close_facing"],
                                     "forks": float(np.mean([v["bifurcations"] for _, v in stats if v["bifurcations"] is not None])),
                                     "c60": signed["screening"]["60"], "c90": signed["screening"]["90"], "h": signed["h"], "pairs": signed["pairs"]}
            x = report[name][variant]
            print(name, variant, round(x["accuracy"]["mean"], 3), round(x["density"], 1), round(x["same"], 2), round(x["offset"], 2), round(x["facing"], 2), round(x["c60"]["mean"], 2), flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
