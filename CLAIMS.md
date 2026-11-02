# Claims

Each table and numbered result in the paper, the result file behind it, and how to rerun it. Result files are in `results/expected/`. A rerun writes to `results/verify/` under the same name.

To rerun one read:

    .venv/bin/python scripts/reproduce.py reads --only NAME

Here NAME is the step name in the third column. `scripts/reproduce.py reads --list` lists them all.

To check the paper's printed numbers against a set of results:

    .venv/bin/python scripts/check_claims.py results/verify

The last column says whether `check_claims.py` covers a table, number by number. Where it doesn't, compare by hand, or compare the whole file with `scripts/compare_results.py`, which is exact.

| Paper | What | Result file and step | Script | Checked by `check_claims.py` |
|---|---|---|---|---|
| §5.1, Table 1 | Planted controls: the measure reads what is planted | `signed_minutiae`, key `controls` | `signed_charge.measure.signed_minutiae` | No |
| §5.2, Table 2 | All minutiae: real against the formula generators | `signed_false`, key `all` | `signed_charge.reads.signed_false` | Yes |
| §5.3, Table 3 | C(60) on held-out fingers and on B | `held_out_look1` | `signed_charge.reads.held_out_signed` | Yes, stored result only |
| §5.4, Table 4 | Under `confirmed`, false minutiae removed | `signed_false`, key `confirmed` | `signed_charge.reads.signed_false` | Yes |
| §5.5, Table 5 | C(60) by what is left out or changed | `signed_curves`, keys `all`, `confirmed`, `no_singular`, `scale6`, `scale14` | `signed_charge.reads.signed_curves` | No |
| §5.6 | Real prints don't screen | `classifier_floor`, `review_controls_d071` | `signed_charge.reads.classifier_floor`, `signed_charge.reads.review_controls` | Partly: `review_controls_d071` |
| §5.7, Table 6 | C(60) by phase winding, and by extractors with close opposite pairs cancelled | `extractor_free_d062`, `cancel_sweep_e102` | `signed_charge.reads.extractor_free`, `signed_charge.reads.cancel_sweep` | No |
| §5.8, Table 7 | C(60) by the coherence of each pair's segment | `quality_lift_d063`, with `quality_degrade_d064` in the text | `signed_charge.reads.quality_lift`, `signed_charge.reads.quality_degrade` | No |
| §5.9, Table 8 | Anguli and SynFing through real windows | `outside_d066` | `signed_charge.reads.outside_read` | Yes |
| §5.10, Table 9 | Net charge and the flow in 96 px patches | `flux_scale_d070`, with the variants `flux_d078_*` and `flux_e124_outside_q0` | `signed_charge.reads.flux_scale` | Yes |
| §5.11, Table 10 | h on one ridge, by finger, and by sensor | `ridge_frame_d072`, `ridge_robust_d077`, `ridge_robust_shared_e117` | `signed_charge.measure.ridge_frame`, `signed_charge.reads.ridge_robust` | Yes: `ridge_frame_d072` |
| §5.10 and §6 | Most of the charge comes with the flow: fronts on given flows | `flow_charge_d054`, `field_d049`, `flow_read_d055`, `flow_agree_*` | `signed_charge.measure.flow_charge`, `signed_charge.reads.field_read`, `signed_charge.reads.flow_read`, `signed_charge.reads.flow_agree` | No |
| §6 | Whether less charge helps a matcher | `tstr_d082_{zp,zpp,synfing}_{1,2,3}_e200`, run by `scripts/reproduce.py matcher` on the GPU | `signed_charge.reads.tstr` | No |

## What can't be rerun from here

- **`held_out_look1` and `signed_false_a`** read the sealed FVC A fingers. The held-out protocol (`docs/held-out-protocol.md`) allows each look once, so their stored results stand as first run.
- **The diffusion render**, the "Phantom" row of some tables, can't be regenerated from this repository (DATA.md). `signed_curves` rereads the stored render.

## Notebook entries

Each table's caption in the paper names the lab-notebook entries behind it, for example E-068 for Table 2. The notebook isn't part of this repository. The file names above carry the same numbers where a read was made for one entry, such as `_d070` for D-070.
