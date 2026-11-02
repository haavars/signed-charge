"""SFinGe's long range on more fingers (D-040, E-071): signed_false_a.py OUT.json IMAGES

`signed_false`'s filters on the checking fingers of DB4 A, the 90 SFinGe fingers of `held_out_checking.json`, beside its 30 of DB4 B. No real A image is read. IMAGES as `held_out_signed`.

Run: python -m signed_charge.reads.signed_false_a OUT.json IMAGES
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from signed_charge.reads import held_out_signed as hs
from signed_charge.measure import provenance
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm


def task(item):
    name, label, paths = item
    return sf.finger((name, label, [hs.load((p, name, Path(p).stem.split("_")[0], 1))[1] for p in paths]))


def main(out, root):
    started = provenance.start()
    by = {}
    for path, name, finger, _ in hs.tasks_a(root, ("DB4",)):
        by.setdefault((name, f"{name}/{finger}"), []).append(path)
    with ProcessPoolExecutor(sf.JOBS) as pool:
        rows = list(pool.map(task, [(n, label, paths) for (n, label), paths in by.items()], chunksize=1))
    rng = np.random.default_rng(0)
    report = {reader: sf.read(rows, reader, rng) for reader in sm.EXTRACTORS}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    for reader in sm.EXTRACTORS:
        r = report[reader]
        print(reader, r["fingers"], r["prints"], {k: (round(r[k]["c60"]["mean"], 2), [round(c, 2) for c in r[k]["c60"]["ci"]], round(r[k]["h_long"]["mean"], 3), [round(c, 3) for c in r[k]["h_long"]["ci"]]) for k in ("all", "confirmed", "clean")}, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
