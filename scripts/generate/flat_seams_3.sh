#!/usr/bin/env bash
# D-051's flat runs: flat_seams_3.sh OUT_DIR. Run from the repo root. JOBS workers, default 4.
OUT=$1
mkdir -p "$OUT"
JOBS=${JOBS:-4}
for seed in 11 12 13 14; do
    for n in 0.00003 0.0001 0.0003 0.001 0.003; do
        echo "$OUT/m45_e0.1_n${n}_$seed.npz --seed $seed --alpha 180 --eps -0.1 --cue 45 --noise $n"
        echo "$OUT/m90_e0.3_n${n}_$seed.npz --seed $seed --alpha 180 --eps -0.3 --cue 90 --noise $n"
    done
    for m in 45 75; do
        echo "$OUT/m${m}_e0.3_v0.5_$seed.npz --seed $seed --alpha 180 --eps -0.3 --cue $m --speed 0.5"
    done
done | OMP_NUM_THREADS=1 xargs -P "$JOBS" -L 1 .venv/bin/python -m signed_charge.exploratory.flat_seams
