# Verification

What has been rerun from this repository, and how it compared with the stored results. "Identical" means every number equal, with nothing rounded. Updated as runs finish.

## Machine

One Linux workstation: 24 cores, an RTX 4090, Python 3.12.14, with the packages in `uv.lock`. Data at the paths in `scripts/env.example.sh`, matching `manifests/`: all 960 FVC B files and LEADER's file hash as listed.

## Results so far

| What | How | Outcome |
|---|---|---|
| The main read, `signed_false` (Tables 2 and 4) | Rerun in this layout, 2026-11-02 | 1,118 of 1,118 values identical |
| Printed numbers in that read | `scripts/check_claims.py results/verify` | 56 of 56 match the paper |
| Printed numbers in all stored results | `scripts/check_claims.py` | 157 of 157 match the paper |
| Population prints | All regenerated from seeds, `scripts/compare_npz.py` | 200 of 200 identical |
| Fronts on model flows | All regenerated | 200 of 200 identical |
| Fronts on real flows | All 180 regenerated | 180 of 180 identical |
| Outside generators' selection | `scripts/generate/outside_links.sh` | All 180 links as used |

Prints are compared array by array, leaving out their run metadata (`provenance` and `settings`, which hold paths and commit hashes).

## Still to run

- The rest of the regeneration: 200 slower fronts, the minutiae cache and the matcher's training prints.
- Every other read in `scripts/reproduce.py reads`.
- The matcher test, on the GPU.
- The diffusion render can't be regenerated from here (DATA.md), so `signed_curves` reruns against the stored render.

## Not rerun, by design

The held-out reads (`held_out_signed`, `signed_false_a`) read FVC A fingers under `docs/held-out-protocol.md`, which allows each look once. Their stored results stand as first run.
