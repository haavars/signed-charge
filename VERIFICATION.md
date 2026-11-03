# Verification

What has been rerun from this repository, and how it compared with the stored results. "Identical" means every number equal, with nothing rounded.

## Machine

One Linux workstation: 24 cores, an RTX 4090, Python 3.12.14, with the packages in `uv.lock`. Data at the paths in `scripts/env.example.sh`, matching `manifests/`: all 960 FVC B files and LEADER's file hash as listed.

## The full rerun, 2026-11-02 to 2026-11-03

`scripts/reproduce.py generate`, then `reads`, then `matcher`, each with `--jobs 22`. Every stage exited 0, apart from one failure, described below. Each read's exact comparison with its stored result is in `results/verify/summary.jsonl`.

### Regenerated data

| What | Compared with | Outcome |
|---|---|---|
| Population prints | `scripts/compare_npz.py` | 200 of 200 identical |
| Fronts on model flows | `scripts/compare_npz.py` | 200 of 200 identical |
| Fronts on real flows | `scripts/compare_npz.py` | 180 of 180 identical |
| Slower, noisier fronts | `scripts/compare_npz.py` | 200 of 200 identical |
| Outside generators' selection | `scripts/generate/outside_links.sh` | All 180 links as used |
| Minutiae cache | Row by row: set, finger, mask, and both extractors' minutiae | 1,788 of 1,788 rows identical |
| Matcher training prints | Byte by byte | 18,000 of 18,000 identical |

Prints are compared array by array, leaving out their run metadata (`provenance` and `settings`, which hold paths and commit hashes).

**One failure, fixed**: the minutiae cache's first build stopped after 30 minutes, because its output folder didn't exist and the script didn't make it. The folder was made by hand and the cache built again, identical to the original. The script now makes the folder itself (commit 6daf95f).

### Reads

39 reads and 9 matcher runs compared, 36,074 values.

- **33 reads are identical**, 33,185 values.
- **6 reads differ, all for known reasons**:

| Read | Values that differ | Why |
|---|---|---|
| `agreement_total_d057` | 52 of 132 | The stored result predates a fix to the edge reflection in `flow_charge.spreading` (D-062), and this repository has the fixed code. The project's own rerun after the fix found the results stand |
| `extractor_free_d062` | 148 of 202 | Expected: the stored file holds only the mean flow, and a rerun writes the charge too (`CLAIMS.md`, "Table 6's two files") |
| `quality_lift_d063` | 30 of 335 | Random draws seeded from each generated print's full file path, which differs in this layout. Only the generated sets differ; real prints' values are identical |
| `quality_degrade_d064` | 102 of 107 | The same: seeds from full file paths of generated prints |
| `flux_scale_d070` | 742 of 1106 | Only confidence-interval bounds differ, by a median of 0.08 % of their value. The stored file was written before the rms flux by scale was added (it has no `rms_F` or `rms_Q`). Those bootstraps take draws from the same stream, so every later interval moves |
| `ridge_robust_d077` | 203 of 449 | Only confidence-interval bounds differ. The stored file was written before E-117's `shared` reader was added. Its bootstrap takes draws from the same stream before the sensors are summarised |

- **The 9 matcher runs differ** in 38 to 41 of 62 values each, because training on the GPU isn't repeatable bit for bit. Mean equal error rate by arm, three seeds each:

| Arm | Stored | Rerun |
|---|---|---|
| ZP | .237 | .235 |
| ZP+P | .308 | .309 |
| SynFing | .244 | .228 |

Each seed's rerun lies within the stored run's confidence interval for that seed. ZP and SynFing swap order, and their seeds' ranges overlap in both runs.

### After merging the cleanup branch, 2026-11-03

The quick stage and `cross_signs` were rerun on the merged code, which reformatted every module and changed no program structure.

| Read | Outcome |
|---|---|
| `signed_minutiae` | 6,688 of 6,688 values identical |
| `signed_reinked` | 1,672 of 1,672 values identical |
| `signed_false` | 1,118 of 1,118 values identical |
| `cross_signs` | 28 of 28 values identical |

### Printed numbers

`scripts/check_claims.py results/verify`, covering 300 numbers: **276 of 276 checked match the paper**. The other 24 are the held-out look's, which isn't rerun by design (below). The differences listed above are all below the paper's rounding.

## Earlier runs

| What | How | Outcome |
|---|---|---|
| The main read, `signed_false` (Tables 2 and 4) | Rerun in this layout, 2026-11-02 | 1,118 of 1,118 values identical |
| Printed numbers in that read | `scripts/check_claims.py results/verify` | 56 of 56 match the paper |
| Printed numbers in all stored results | `scripts/check_claims.py` | 157 of 157 match the paper |

## Known limits

- Two reads seed random draws from full file paths, so they repeat exactly only with the data at the original paths. Seeding from file names would fix that, but it would change their stored results, so it isn't done here.
- The matcher test isn't repeatable bit for bit on a GPU.
- The diffusion render can't be regenerated from here (DATA.md), so `signed_curves` reruns against the stored render.

## Not rerun, by design

The held-out reads (`held_out_signed`, `signed_false_a`) read FVC A fingers under `docs/held-out-protocol.md`, which allows each look once. Their stored results stand as first run.
