"""D-027's test 2 over fingers, each with its own geometry (E-056): stress_topology.py OUT.json [FINGERS]

D-029's numbers, the `person` of D-016's population seeds, the singular points inside the print's contact.

Run: python -m signed_charge.exploratory.stress_topology OUT.json
"""

import collections
import json
import sys

from signed_charge.generate import competency_model as cm
from signed_charge.exploratory import growth_stress as gs
from signed_charge.measure import provenance
from signed_charge.exploratory import stress_read as sr


def main(out, fingers=20):
    started = provenance.start()
    report = {}
    for pattern in gs.PADS:
        counts = collections.Counter()
        for seed in range(4101, 4101 + fingers):
            r, _ = sr.read(pattern, cm.person_of(seed, 1.0), **gs.settings(pattern))
            counts[f"{r['cores']} cores {r['deltas']} deltas"] += 1
        report[pattern] = dict(counts.most_common())
        print(pattern, report[pattern], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
