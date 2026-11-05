# Verification

A full rerun from the raw data, following [Reproducing the results](README.md#reproducing-the-results), was done on 2 and 3 November 2026 on one Linux workstation (24 threads, an RTX 4090, the packages in `uv.lock`).

**Outcome.** All 276 of the paper's numbers that a rerun can check matched (`scripts/check_claims.py results/verify`). The other 24 belong to the held-out look, which is read only once, by design, and isn't rerun. Regenerated prints, the minutiae cache and the matcher's training prints were identical to the originals.

**Where your rerun will differ, and why.** 33 of 39 reads repeat exactly. Six differ below the paper's rounding:

| Read | Why it differs |
|---|---|
| `agreement_total_d057` | The stored result predates a fix to edge reflection in `flow_charge.spreading`; this repository has the fixed code, and the results stand |
| `extractor_free_d062` | The stored file holds only the mean flow; a rerun also writes the charge (`CLAIMS.md`, "Table 6's two files") |
| `quality_lift_d063`, `quality_degrade_d064` | Random draws are seeded from each generated print's full file path, so they repeat exactly only with the data at the original paths. Real prints' values are identical |
| `flux_scale_d070`, `ridge_robust_d077` | Only confidence-interval bounds move, by a median of 0.08 % in `flux_scale_d070`: the stored files predate a later addition that takes draws from the same bootstrap stream |

**The matcher test** trains on a GPU, which doesn't repeat bit for bit. Mean equal error rates, three seeds each:

| Arm | Stored | Rerun |
|---|---|---|
| ZP | .237 | .235 |
| ZP+P | .308 | .309 |
| SynFing | .244 | .228 |

Each seed's rerun lies within the stored run's confidence interval for that seed. ZP and SynFing swap order, and their ranges overlap in both runs.
