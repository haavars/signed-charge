"""Fork share and close pairs by extractor, on images, not inked masters (notebook E-035): extractor_forks.py OUT [PEOPLE]

mindtct and LEADER (`validate_masters.leader`) on FVC 2000/2002/2004 DB1-3 B (real), their DB4 B (SFinGe) and Phantom's rendered
slap fingertips (`impressions-slaps-value-03`, ridgegen 0.25, with the diffusion pass's texture). The fork target can only be
judged like against like: experts find 26-33 % on inked prints, mindtct 49 % and LEADER 36 % on FVC (D-008).

Run: python -m signed_charge.exploratory.extractor_forks OUT.json 30
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from signed_charge.measure import conformality as cf
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import validate_masters as vm

JOBS = 4


def main(out, people=30):
    started = provenance.start()
    sets = {"real": list(cf.fvc(("DB1", "DB2", "DB3"))), "sfinge": list(cf.fvc(("DB4",))), "phantom_rendered": list(cf.phantom(people))}
    report = {}
    for extractor in ("mindtct", "leader"):
        report[extractor] = {}
        with ProcessPoolExecutor(JOBS, initializer=vm.set_extractor, initargs=(extractor,)) as pool:
            for name, items in sets.items():
                stats = [v for _, v in pool.map(ms.describe, items, chunksize=4) if v]
                pairs = [v for _, v in pool.map(mp.describe, items, chunksize=4) if v]
                s, p = ms.summary(stats), mp.summary(pairs)
                report[extractor][name] = {"stats": s, "pairs": {k: v for k, v in p.items() if k != "nn_share"}}
                c = p["clustering"]
                print(extractor, name.ljust(17), len(stats), "dens %.1f forks %.2f" % (s["density_per_100mm2"], s["bifurcations"]),
                      "same/off/face %.2f %.2f %.2f" % (p["close_same"], p["close_offset"], p["close_facing"]),
                      "EE/EB/BB %.2f %.2f %.2f" % (p["close_EE"], p["close_EB"], p["close_BB"]),
                      "c10/20/40 %.2f %.2f %.2f" % (c[10], c[20], c[40]), flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
