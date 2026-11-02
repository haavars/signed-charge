#!/bin/bash
# D-067's prints: [JOBS=n] lead_run.sh OUT_DIR pilot | [JOBS=n] lead_run.sh OUT_DIR population SPEED. Run from the repo root. Fronts on model 2's read flows, smoothed at 50 px, cue 0.2, noise 0.03. The pilot is D-065's 20 fingers at three speeds. The population is 200 fingers of 25 people nothing here has read. Finished or taken runs are skipped.
OUT=$1; MODE=$2; SPEED=$3; mkdir -p "$OUT"; export OUT MODE SPEED
job() { mkdir "$OUT/.$1" 2>/dev/null || exit 0; OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.generate.field_print "$OUT/$1.npz" --flow "read:$2:$3" --smooth 50 --eps -0.2 --speed "$4" --noise 0.03 --sigma 0.03 --seed "$5" > "$OUT/$1.log" 2>&1; }
export -f job
python3 - <<'P' | shuf --random-source=<(yes) | xargs -P ${JOBS:-8} -L 1 bash -c 'job "$0" "$1" "$2" "$3" "$4"'
import os
FINGERS = (2, 3, 4, 5, 7, 8, 9, 10)
if os.environ["MODE"] == "pilot":
    seed = 6300
    for speed in (0.6, 0.5, 0.4):
        for person, finger in [(p, f) for p in (9100, 9101, 9102) for f in FINGERS][:20]:
            seed += 1
            print(f"s{int(speed * 100)}_{person}_{finger} {person} {finger} {speed} {seed}")
else:
    seed = 6500
    for person in range(9120, 9145):
        for finger in FINGERS:
            seed += 1
            print(f"lead_{person}_{finger} {person} {finger} {os.environ['SPEED']} {seed}")
P
