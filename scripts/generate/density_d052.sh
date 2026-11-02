#!/bin/bash
# The density law on the corrected geometry (D-052): [JOBS=n] density_d052.sh OUT_DIR. Run from the repo root. Finished runs are skipped, and a lock lets a second copy widen a run in progress.
# Arms noise0.003 and noise0.01: loops and whorls as D-015 and D-010, for E-075's arches to stand beside. Arm population: all three patterns on the population's settings.
OUT=$1; mkdir -p "$OUT/noise0.003" "$OUT/noise0.01" "$OUT/population"; export OUT
job() { # arm pattern speed t seed sigma noise extra...
  arm=$1; p=$2; a=$3; t=$4; seed=$5; sigma=$6; noise=$7; shift 7; f="$OUT/$arm/${p}_a${a}_s${sigma}_$seed"
  [ -f "$f.npz" ] || ! mkdir "$f.lock" 2>/dev/null || OMP_NUM_THREADS=1 .venv/bin/python -m signed_charge.generate.competency_model $p "$f.npz" --speed $a --sigma $sigma --noise $noise --t $t --seed $seed "$@" > "$f.log" 2>&1
}
export -f job
python3 - <<'P' | xargs -P ${JOBS:-16} -L 1 bash -c 'job "$0" "$@"'
base = {"loop": (350, 300), "whorl": (400, 300), "arch": (455, 400)}
seed = 7000
for _ in range(4):
    for arm, noise, speeds, patterns in (("noise0.003", 0.003, (0.5, 0.6, 0.7, 0.8, 0.9), ("loop", "whorl")), ("noise0.01", 0.01, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), ("loop", "whorl")), ("population", 0.003, (0.5, 0.6, 0.7, 0.8, 0.9), ("loop", "whorl", "arch"))):
        for speed in speeds:
            for p in patterns:
                seed += 1
                a, b = base[p]
                if arm == "population":
                    print(arm, p, speed, round(a / speed + b + 60), seed, 0.03, noise, "--vary 1 --aniso -0.1")
                else:
                    print(arm, p, speed, round(a / speed + b), seed, 0.1, noise)
P
