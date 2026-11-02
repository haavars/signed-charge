#!/bin/bash
# A population of fingers, each with its own geometry (D-016, D-017, D-020): [SIGMA=s] [ANISO=e] population.sh OUT_DIR LOOPS WHORLS ARCHES [FIRST_SEED]. Run from the repo root.
# Finished runs are skipped. Runs each pattern at speed 0.7, noise 0.003, 300 past the last arrival, plus 60 for a late pad.
OUT=$1; mkdir -p "$OUT"; export OUT
export SIGMA=${SIGMA:-0.03}
export ANISO=${ANISO:--0.1}
job() { [ -f "$OUT/$1_a0.7_s${SIGMA}_$3.npz" ] || OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.generate.competency_model $1 "$OUT/$1_a0.7_s${SIGMA}_$3.npz" --speed 0.7 --sigma $SIGMA --noise 0.003 --vary 1 --aniso $ANISO --t $2 --seed $3 > "$OUT/$1_a0.7_s${SIGMA}_$3.log" 2>&1; }
export -f job
seed=${5:-4000}
{ for i in $(seq 1 $2); do seed=$((seed+1)); echo "loop 860 $seed"; done
  for i in $(seq 1 $3); do seed=$((seed+1)); echo "whorl 930 $seed"; done
  for i in $(seq 1 $4); do seed=$((seed+1)); echo "arch 1110 $seed"; done; } | shuf --random-source=<(yes) | xargs -P 8 -L 1 bash -c 'job $0 $1 $2'
