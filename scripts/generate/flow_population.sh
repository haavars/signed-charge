#!/bin/bash
# D-055's prints: [JOBS=n] flow_population.sh OUT_DIR. Run from the repo root. Fronts on flows read from prints and smoothed at 50 px: FVC DB1-3 B, each finger's first impression, and model 2's prints of 90 of D-049's fingers. Finished or taken runs are skipped, so two of these can share the list.
OUT=$1; mkdir -p "$OUT"; export OUT
job() { mkdir "$OUT/.$1" 2>/dev/null || exit 0; OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.generate.field_print "$OUT/$1.npz" --flow "$2" --smooth 50 --eps -0.3 --speed 0.7 --noise 0.003 --sigma 0.03 --seed $3 > "$OUT/$1.log" 2>&1; }
export -f job
.venv/bin/python - <<'P' | shuf --random-source=<(yes) | xargs -P ${JOBS:-4} -L 1 bash -c 'job $0 $1 $2'
import glob, os
root = __import__("os").environ["FVC_ROOT"]
seed = 5000
for year in ("FVC2000", "FVC2002", "FVC2004"):
    for db in ("DB1", "DB2", "DB3"):
        for path in sorted(glob.glob(f"{root}/{year}/{db}_B/*_1.tif")):
            seed += 1
            print(f"realflow_{year}_{db}_{os.path.basename(path)[:-6]} real:{path} {seed}")
for person in range(9100, 9112):
    for finger in (2, 3, 4, 5, 7, 8, 9, 10):
        seed += 1
        if seed <= 5180:
            print(f"readflow_{person}_{finger} read:{person}:{finger} {seed}")
P
