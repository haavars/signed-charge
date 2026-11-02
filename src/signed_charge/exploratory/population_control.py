"""How much of the population's classifier score is image type? (E-065): population_control.py OUT.json

Real prints re-inked, as `signed_minutiae.reinked` does: through mindtct's binarisation, then taken as prints. The classifier of `population_read`, real against re-inked real, the same prints on both sides, by leaving out sensors: four sets against the re-inked five, 30 draws at 200 prints, as its floor.
If that's near the floor, binary images alone don't fool it, and the population's 93 % is about its minutiae. If it's near 93 %, the score says little.

Run: python -m signed_charge.exploratory.population_control OUT.json
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from signed_charge.reads import classifier_floor as cl
from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.reads import population_read as pr
from signed_charge.measure import provenance

JOBS = int(os.environ.get("READ_JOBS", 8))
SIZE = 200


def reinked(item):
    key, im = item
    fg = fs.foreground(im)
    return pr.read_image((key, np.where((cf.binarised(im) == 0) & (fg if fg is not None else True), 40, 235).astype(np.uint8)))


def main(out):
    started = provenance.start()
    items = list(cf.fvc(("DB1", "DB2", "DB3")))
    with ProcessPoolExecutor(JOBS) as pool:
        real = list(pool.map(pr.read_image, items, chunksize=4))
        inked = list(pool.map(reinked, items, chunksize=4))
    report = {}
    rng = np.random.default_rng(0)
    for name in pr.READERS:
        r = [(k, s[name]) for k, s, _, _ in real if s[name]]
        i = [(k, s[name]) for k, s, _, _ in inked if s[name]]
        sensors = sorted({k[0] for k, _ in r})
        scores = []
        for seed in range(pr.RESAMPLES):
            a = {sensors[j] for j in rng.permutation(len(sensors))[:4]}
            left = [x for x in r if x[0][0] in a]
            right = [x for x in i if x[0][0] not in a]
            scores.append(
                cf.accuracy([left[j] for j in rng.permutation(len(left))[:SIZE]], [right[j] for j in rng.permutation(len(right))[:SIZE]], "mh", seed=seed)
            )
        report[name] = {
            "real_against_reinked": cl.spread(scores),
            "plain_against_inked_all": cl.spread([cf.accuracy(r, i, "mh", seed=s) for s in range(cl.SEEDS)]),
        }
        print(name, report[name], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
