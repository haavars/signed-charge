#!/usr/bin/env bash
# The prints read from outside generators, as links: outside_links.sh OUT_DIR. Needs ANGULI_ROOT and SYNFING_ROOT.
# Anguli's prints 1 to 90, and SynFing's first impressions at ids 0, 1111, 2222 and so on, 45 with each artifact type.
set -euo pipefail
OUT=$1
mkdir -p "$OUT"
for i in $(seq 1 90); do
  ln -sf "$ANGULI_ROOT/$i.jpg" "$OUT/$(printf 'anguli_%04d.jpg' "$i")"
done
for k in $(seq 0 89); do
  id=$(printf '%05d' $((k * 1111)))
  ln -sf "$SYNFING_ROOT/impression_1/F_$id.jpg" "$OUT/synfing_$id.jpg"
done
