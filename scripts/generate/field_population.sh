#!/bin/bash
# D-049's prints: field_population.sh OUT_DIR. Run from the repo root. 25 fresh people × 8 fingers, model 2's flows. Finished or taken runs are skipped, so two of these can share the list.
OUT=$1; mkdir -p "$OUT"; export OUT
job() { mkdir "$OUT/.$1_$2" 2>/dev/null || exit 0; OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.generate.field_print "$OUT/field_$1_$2.npz" --flow model2:$1:$2 --eps -0.3 --speed 0.7 --noise 0.003 --sigma 0.03 --front sites --seed $(( $1 * 100 + $2 )); }
export -f job
for person in $(seq 9100 9124); do for finger in 2 3 4 5 7 8 9 10; do echo "$person $finger"; done; done | shuf --random-source=<(yes) | xargs -P ${JOBS:-4} -L 1 bash -c 'job $0 $1'
