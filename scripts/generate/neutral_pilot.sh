#!/bin/bash
# D-065's pilot: [JOBS=n] neutral_pilot.sh OUT_DIR. Run from the repo root. Fronts on model 2's read flows, smoothed at 50 px, at three cues and three noises, 20 fingers a cell, each cell a new draw of noise. Finished or taken runs are skipped.
OUT=$1; mkdir -p "$OUT"; export OUT
job() { mkdir "$OUT/.$1" 2>/dev/null || exit 0; OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.generate.field_print "$OUT/$1.npz" --flow "read:$2:$3" --smooth 50 --eps "$4" --speed 0.7 --noise "$5" --sigma 0.03 --seed "$6" > "$OUT/$1.log" 2>&1; }
export -f job
python3 - <<'P' | shuf --random-source=<(yes) | xargs -P ${JOBS:-8} -L 1 bash -c 'job "$0" "$1" "$2" "$3" "$4" "$5"'
fingers = [(p, f) for p in (9100, 9101, 9102) for f in (2, 3, 4, 5, 7, 8, 9, 10)][:20]
seed = 6000
for eps in (0.3, 0.2, 0.1):
    for noise in (0.003, 0.01, 0.03):
        if (eps, noise) == (0.3, 0.003):
            continue  # D-055's readflow is this cell
        for person, finger in fingers:
            seed += 1
            print(f"e{int(eps * 100)}n{int(noise * 1000):02d}_{person}_{finger} {person} {finger} -{eps} {noise} {seed}")
P
