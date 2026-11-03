"""E-024: `validate_masters.run` with mindtct on 20 of the new people, the tuning set's size:
validate_same_size.py OUT. Run as validate_masters.py.
"""

import sys
import json
from signed_charge.measure import validate_masters as vm
from signed_charge.measure import conformality as cf
from signed_charge.measure import provenance

if __name__ == "__main__":
    started = provenance.start()
    real = list(cf.fvc(("DB1", "DB2", "DB3")))
    report = {}
    for first in (7000,):
        tasks = [(v, s, f, c) for v in ("model2", "m3_clump0.5_below1.6") for s in range(first, first + 20) for f in (2, 3, 4, 5, 7, 8, 9, 10) for c in (1, 2)]
        report[str(first)] = vm.run("mindtct", tasks, real)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(sys.argv[1], "w"), indent=1)
    for k, v in report["7000"].items():
        print(k, round(v.get("accuracy_mh", 0), 3), round(v["density_per_100mm2"], 1), round(v["pairs"]["close_facing"], 2))
