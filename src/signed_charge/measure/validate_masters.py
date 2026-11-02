"""Frozen master settings, checked on what they weren't tuned on (lab notebook): validate_masters.py OUT PEOPLE
FIRST_SEED VARIANT...

E-016 to E-022 tuned the phase-drawn masters' settings against FVC with mindtct, on people 3000-3019. Here the
settings are fixed in advance and measured on new people and with a second extractor nothing was tuned on:

- **mindtct**, as the screens (`minutiae_stats.py`, `minutiae_pairs.py`), on people from FIRST_SEED.
- **LEADER** (Cappelli & Ferrara, arXiv 2602.15493; pyfing, MIT; its export
  see DATA.md): the image padded to a multiple of 32 with its
  top-left pixel's value, a minutia where the suppressed position map reaches 0.6,
  CPU. Ridges between pairs from mindtct's binarised image.

Real data are the same FVC 2000/2002/2004 DB1-3 B: no untouched real prints are at hand (SOCOFing's are 96 × 103 px;
SD 302 waits on NIST), so with mindtct this checks new people only. Fingertips are inked flat (near-binary), as in
tuning, so LEADER sees them less like FVC's grey images than it would rendered ones.

Run: python -m signed_charge.measure.validate_masters OUT.json 30 7000 model2 VARIANT...
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.generate import phase_masters as pm
from signed_charge.measure import provenance
from signed_charge import paths

LEADER = str(paths.LEADER)
THRESHOLD = 0.6
MULTIPLE = 32
JOBS = 8
session = None


def leader(im):
    """LEADER's minutiae as `ms.mindtct`'s rows: (x, y, angle in radians, bifurcation)."""
    global session
    if session is None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(LEADER, options, providers=["CPUExecutionProvider"])
    h, w = im.shape
    ph, pw = -(-h // MULTIPLE) * MULTIPLE, -(-w // MULTIPLE) * MULTIPLE
    top, left = (ph - h) // 2, (pw - w) // 2
    padded = np.full((ph, pw), float(im[0, 0]), np.float32)
    padded[top : top + h, left : left + w] = im
    out = session.run(None, {"image": padded[None, :, :, None]})[0][0, top : top + h, left : left + w]
    ys, xs = np.nonzero(out[..., 3] >= THRESHOLD)
    rows = [(x, y, -out[y, x, 1], out[y, x, 2] < 0.5) for x, y in zip(xs, ys)]
    return np.array(rows, dtype=float).reshape(-1, 4)


def set_extractor(name):
    if name == "leader":
        ms.mindtct = leader
        mp.mindtct = lambda im: (leader(im), cf.binarised(im) == 0)


def run(extractor, tasks, real):
    with ProcessPoolExecutor(JOBS, initializer=set_extractor, initargs=(extractor,)) as pool:
        stats = [(k, v) for k, v in pool.map(ms.describe, real, chunksize=4) if v]
        pairs = [v for _, v in pool.map(mp.describe, real, chunksize=4) if v]
        mine = list(pool.map(pm.tip, tasks, chunksize=2))
    report = {"real": {**ms.summary([v for _, v in stats]), "pairs": short(mp.summary(pairs))}}
    for variant in sorted({t[0] for t in tasks}):
        xs = [x for x in mine if x[0][0] == variant]
        s = [(k, v) for k, v, _, _ in xs if v]
        report[variant] = {
            **ms.summary([v for _, v in s]),
            "accuracy_mh": cf.accuracy(stats, s, "mh"),
            "pairs": short(mp.summary([p for _, _, p, _ in xs if p])),
        }
    return report


def short(summary):
    return {k: v for k, v in summary.items() if k != "nn_share"}


def main(out, people, first, variants):
    started = provenance.start()
    real = list(cf.fvc(("DB1", "DB2", "DB3")))
    tasks = [(v, s, f, c) for v in variants for s in range(first, first + people) for f in (2, 3, 4, 5, 7, 8, 9, 10) for c in (1, 2)]
    report = {name: run(name, tasks, real) for name in ("mindtct", "leader")}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    for name in ("mindtct", "leader"):
        for k, v in report[name].items():
            p = v["pairs"]
            print(
                name,
                k,
                round(v["density_per_100mm2"], 1),
                round(v["bifurcations"], 2),
                round(v.get("accuracy_mh", 0), 3),
                [round(p[f"close_{x}"], 2) for x in ("same", "offset", "facing")],
                {r: round(c, 2) for r, c in p["clustering"].items()},
            )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])
