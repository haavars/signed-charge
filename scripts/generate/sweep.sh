#!/bin/bash
# The speed sweeps of D-010 and D-015: [NOISE=n] [FIRST=seed] [SPEEDS="..."] sweep.sh OUT_DIR [ROUND...]. Run from the repo root.
# Rounds are one seed of each speed and pattern, so a partial sweep stays balanced; finished runs are skipped.
OUT=$1; shift; mkdir -p "$OUT"; export OUT
export NOISE=${NOISE:-0.01}
SPEEDS=${SPEEDS:-0.5 0.6 0.7 0.8 0.9 1.0}
job() { [ -f "$OUT/$1_a$2_s0.1_$4.npz" ] || OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.generate.competency_model $1 "$OUT/$1_a$2_s0.1_$4.npz" --speed $2 --sigma 0.1 --noise $NOISE --t $3 --seed $4 > "$OUT/$1_a$2_s0.1_$4.log" 2>&1; }
export -f job
seed=${FIRST:-1000}
for round in "${@:-1 2 3 4 5 6 7 8 9 10 11 12}"; do for r in $round; do
  for a in $SPEEDS; do
    for p in loop whorl arch; do
      seed=$((seed+1))
      [ "$p" = arch ] && [ "$r" -gt 4 ] && continue
      t=$(python3 -c "print({'loop':round(350/$a+300),'whorl':round(400/$a+300),'arch':round(455/$a+400)}['$p'])")
      echo "$p $a $t $seed"
    done
  done
done; done | xargs -P 8 -L 1 bash -c 'job $0 $1 $2 $3'
