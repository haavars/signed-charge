"""What the minutiae classifier's accuracy is worth (E-038): classifier_floor.py OUT [PEOPLE]

D-009 asked for a floor: nothing says how far real prints can be told from real prints. And `conformality.accuracy` used to
train on the first n real prints, file order, so FVC2000's sensors alone when the synthetic side was smaller, then test on
all nine (fixed). So, with mindtct and LEADER, on the people of E-024 and E-027:

- `stale`: the old sampling, at seed 0, which should reproduce the published figures; `fair`: the fixed one, over 10 seeds;
- `fair_by_people`: the fixed one on the first 5, 10, 20 people, for the sample-size effect of E-024;
- `random_labels`: fingers labelled at random, across all sensors, which should be 50 %;
- `sensor_splits`: 30 random splits of the nine sets, four against five: real against real at about 320 against 390 prints;
- `sensor_pairs` and `against_sensor`: each pair of sets, and each set against 80 prints of a variant, like for like.

Run: python -m signed_charge.reads.classifier_floor OUT.json 20
"""

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import flow_check as fc
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import validate_masters as vm

VARIANTS = ("model2", "d_s1_sp2_f3")  # E-027's best on the flow
EXTRACTORS = {"mindtct": ms.mindtct, "leader": vm.leader}
FIRST, JOBS, SEEDS = 7000, 8, 10


def described(item):
    key, im = item
    out = {}
    for name, extract in EXTRACTORS.items():
        ms.mindtct = extract
        out[name] = ms.describe((key, im))[1]
    return key, out


def rendered(task):
    return described(fc.image(task))


def stale(real, synthetic, feature, folds=5, seed=0):
    """`conformality.accuracy` as it was: the classes balanced by the first n real prints."""
    rows = [(k[1], 1, v[feature]) for k, v in real] + [(k[1], 0, v[feature]) for k, v in synthetic]
    fingers = sorted({r[0] for r in rows})
    rng = np.random.default_rng(seed)
    fold = dict(zip(fingers, rng.permutation(len(fingers)) % folds))
    correct = []
    for f in range(folds):
        train = [r for r in rows if fold[r[0]] != f]
        test = [r for r in rows if fold[r[0]] == f]
        n = min(sum(r[1] for r in train), sum(1 - r[1] for r in train))
        pos = [r for r in train if r[1]][:n]
        neg = [r for r in train if not r[1]]
        neg = [neg[i] for i in rng.permutation(len(neg))[:n]]
        pos = [pos[i] for i in rng.permutation(len(pos))[:n]]
        tr = pos + neg
        predict = cf.logistic(np.array([r[2] for r in tr]), np.array([r[1] for r in tr], float))
        for label in (0, 1):
            group = [r for r in test if r[1] == label]
            if group:
                correct.append(np.mean(predict(np.array([r[2] for r in group])) == label))
    return float(np.mean(correct))


def spread(xs):
    return {
        "mean": float(np.mean(xs)),
        "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
        "min": float(np.min(xs)),
        "max": float(np.max(xs)),
        "n": len(xs),
    }


def floor(real, mine, rng):
    sensors = sorted({k[0] for k, _ in real})
    by_sensor = {s: [x for x in real if x[0][0] == s] for s in sensors}
    report = {}

    labels = []
    for seed in range(SEEDS):
        fingers = sorted({k[1] for k, _ in real})
        heads = dict(zip(fingers, rng.permutation(len(fingers)) % 2))
        labels.append(cf.accuracy([x for x in real if heads[x[0][1]]], [x for x in real if not heads[x[0][1]]], "mh", seed=seed))
    report["random_labels"] = spread(labels)

    splits = []
    for seed in range(30):
        a = {sensors[i] for i in rng.permutation(len(sensors))[:4]}
        splits.append(cf.accuracy([x for x in real if x[0][0] in a], [x for x in real if x[0][0] not in a], "mh", seed=seed))
    report["sensor_splits"] = spread(splits)
    report["sensor_pairs"] = spread([cf.accuracy(by_sensor[a], by_sensor[b], "mh") for a, b in itertools.combinations(sensors, 2)])

    for variant in VARIANTS:
        stats = [(k, v) for k, v in mine if k[0] == variant]
        people = lambda p: [(k, v) for k, v in stats if int(k[1][2:].split("/")[0]) < FIRST + p]
        report[variant] = {
            "prints": len(stats),
            "stale": stale(real, stats, "mh"),
            "fair": spread([cf.accuracy(real, stats, "mh", seed=s) for s in range(SEEDS)]),
            "fair_by_people": {str(p): spread([cf.accuracy(real, people(p), "mh", seed=s) for s in range(SEEDS)]) for p in (5, 10, 20)},
            "against_sensor": spread([cf.accuracy(by_sensor[s], [stats[i] for i in rng.permutation(len(stats))[:80]], "mh") for s in sensors]),
        }
        print(variant, json.dumps(report[variant]), flush=True)
    return report


def main(out, people=20):
    started = provenance.start()
    items = list(cf.fvc(("DB1", "DB2", "DB3")))
    tasks = [(v, s, f, c) for v in VARIANTS for s in range(FIRST, FIRST + people) for f in (2, 3, 4, 5, 7, 8, 9, 10) for c in (1, 2)]
    with ProcessPoolExecutor(JOBS) as pool:
        real = list(pool.map(described, items, chunksize=4))
        print("real", len(real), flush=True)
        mine = list(pool.map(rendered, tasks, chunksize=2))
    report = {}
    for name in EXTRACTORS:
        print(name, flush=True)
        rng = np.random.default_rng(0)
        report[name] = floor([(k, v[name]) for k, v in real if v[name]], [(k, v[name]) for k, v in mine if v[name]], rng)
        print(json.dumps({k: v for k, v in report[name].items() if k not in VARIANTS}), flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
