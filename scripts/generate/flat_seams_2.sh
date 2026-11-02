#!/bin/bash
# D-047's flat runs: flat_seams_2.sh OUT_DIR. Run from the repo root. Finished runs are skipped. Seeds from 11, none shared with D-045's.
OUT=$1; mkdir -p "$OUT"; export OUT
job() { [ -f "$OUT/$1_$2.npz" ] || OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.exploratory.flat_seams "$OUT/$1_$2.npz" --seed $2 ${@:3}; }
export -f job
{ for seed in 11 12 13 14 15 16 17 18; do
    for a in 75 100 110 120 130 140 150 160 170; do echo "a$a $seed --alpha $a"; done
  done
  for seed in 11 12 13 14; do
    for e in 0.1 0.2 0.3 0.4; do for m in 0 15 30 45 60 75 90; do echo "m${m}_e$e $seed --alpha 180 --eps -$e --cue $m"; done; done
    for a in 90 120 150; do for e in 0.15 0.2 0.3 0.4; do for c in 45 90; do echo "a${a}_e${e}_c$c $seed --alpha $a --eps -$e --cue $c"; done; done; done
    for v in 0.5 0.7 0.9; do echo "a150_e0.3_c90_v$v $seed --alpha 150 --eps -0.3 --cue 90 --speed $v"; done
  done; } | xargs -P ${JOBS:-16} -L 1 bash -c 'job $0 $@'
