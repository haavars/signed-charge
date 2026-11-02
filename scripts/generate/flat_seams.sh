#!/bin/bash
# D-045's flat runs: flat_seams.sh OUT_DIR. Run from the repo root. Finished runs are skipped.
OUT=$1; mkdir -p "$OUT"; export OUT
job() { [ -f "$OUT/$1_$2.npz" ] || OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.exploratory.flat_seams "$OUT/$1_$2.npz" --seed $2 ${@:3}; }
export -f job
{ for seed in 1 2 3 4; do
    for a in 20 30 45 60 90 120 150 180; do echo "a$a $seed --alpha $a"; done
    for a in 45 90 150; do for e in 0.1 0.2; do for c in 0 45 90; do echo "a${a}_e${e}_c$c $seed --alpha $a --eps -$e --cue $c"; done; done; done
    for v in 0.5 0.9; do echo "a90_v$v $seed --alpha 90 --speed $v"; echo "a150_v$v $seed --alpha 150 --speed $v"; done
    for f in 30 100; do echo "a90_soft$f $seed --alpha 90 --soft $f"; echo "a150_soft$f $seed --alpha 150 --soft $f"; done
    for r in 40 100; do echo "a90_ramp$r $seed --alpha 90 --ramp $r"; echo "a150_ramp$r $seed --alpha 150 --ramp $r"; done
    for d in 25 50 100; do echo "a90_d$d $seed --alpha 90 --delay $d"; echo "a150_d$d $seed --alpha 150 --delay $d"; done
    for d in 0 50 100; do echo "disc_d$d $seed --alpha 90 --disc 60 --delay $d"; done
    for e in 0.1 0.2; do for c in 0 45 90; do echo "disc_d50_e${e}_c$c $seed --alpha 90 --disc 60 --delay 50 --eps -$e --cue $c"; done; done
  done; } | xargs -P ${JOBS:-16} -L 1 bash -c 'job $0 $@'
