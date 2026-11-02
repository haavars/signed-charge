#!/bin/bash
# The island screen of E-041: islands.sh OUT_DIR. Run from the repo root. Finished runs are skipped.
OUT=$1; mkdir -p "$OUT"; export OUT
job() { [ -f "$OUT/$1_a0.6_s$4_$6.npz" ] || OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.generate.competency_model loop "$OUT/$1_a0.6_s$4_$6.npz" --speed 0.6 --t 883 --K $2 --noise $3 --sigma $4 --ramp $5 --seed $6 > "$OUT/$1_a0.6_s$4_$6.log" 2>&1; }
export -f job
seed=2000
for k in 1 2 3 4; do
  #    name     K      noise  sigma ramp
  for cfg in "K15 0.0015 0.01 0.1 10" "noise3 0.001 0.003 0.1 10" "sigma3 0.001 0.01 0.03 10" "ramp40 0.001 0.01 0.1 40"; do
    seed=$((seed+1)); echo "$cfg $seed"
  done
done | xargs -P 8 -L 1 bash -c 'job $0 $1 $2 $3 $4 $5'
