# Signed minutiae: real fingerprints arrange ridge insertions differently from synthetic generators

Code and stored results for a draft paper, shared so that its numbers can be checked independently. The work hasn't been peer reviewed.

## The research in brief

A minutia is a place where a ridge is added or lost, so it carries a sign, and a ridge pattern can be read as a phase in which each minutia is a charge of +1 or -1 (Larkin & Fletcher 2007). Whether neighbouring charges are arranged in any particular way is a plain question. For friction ridge skin we found no prior work that answers it. Charge correlations of phase singularities are well studied in optics: neighbouring vortices of a random wave anticorrelate, and their charge is screened (Halperin 1981; Freund & Shvartsman 1994; Shvartsman & Freund 1994; Berry & Dennis 2000; Dennis 2003). Champod (1996) tabulated a fixed-frame sign by region, and his one-sign regions need comparing with what's below. Stoney and Thornton's minutia-pair studies (1986, 1987) haven't been read yet.

We define a pair correlation of signs. Ridge flow has no global direction, so a sign is only defined for a pair. For minutiae i and j whose straight segment lies on the print, the ridge direction is lifted continuously along that segment, from the print's orientation field smoothed over one ridge period (9 px). s_i and s_j are each minutia's sign against the lifted direction at its end. Pairs whose segment leaves the print are dropped. The sign rule was calibrated on planted spirals, 96 of 96 right, for forks and endings alike.

    h(r) = mean of s_i s_j over pairs of minutiae at separation r
    C(R) = sum over a minutia's neighbours within R of s_i s_j, averaged over minutiae

h(r) is the non-normalised mark correlation function c_mm(r) of a point process with marks ±1 (Illian et al. 2008), except that here the marks are defined only pairwise, through the lift. The normalised k_mm(r) divides by the squared mean mark, which is near 0 here, so it isn't used.

- **h** is 0 for independent signs with no net bias: planted spirals of random sign on straight ridges read about 0. It is negative for opposite neighbours and positive for one-sign domains.
- **The flow sets a bias.** Where the flow fans out, it demands minutiae of one sign. So independent signs that follow that local mean give h > 0, and part of every positive h below is the flow's. Separating the two needs the flow's predicted mean sign. That isn't done yet (see below).
- **C(R) = -1** would mean a charge is exactly screened within R. C is a sum, so it grows with the number of neighbours as h does not; minutiae a print are given beside it.

All numbers below are mindtct / LEADER, plain impressions at 500 ppi, where 60 px is 3 mm, about six ridge periods.

### Findings

**Within 3 mm, neighbouring minutiae on real prints share sign only weakly, and every generator we tested shares it more.** The ordering is the same on both extractors.

| | minutiae a print | C(60) | h(15-45 px) |
|---|---|---|---|
| Real, fingers set aside, 270 fingers | | 0.81 / 1.07 | |
| Real, tuning data, 90 fingers | 31.5 / 27.9 | 0.70 / 1.03 | 0.13 / 0.23 |
| SFinGe, fingers set aside, 90 fingers | | 2.38 / 2.73 | |
| SFinGe, tuning data, 30 fingers | 25.0 / 24.8 | 2.42 / 2.76 | 0.45 / 0.53 |
| ZP, a zero-pole flow at fixed spacing | 49.1 / 51.2 | 4.15 / 3.88 | 0.51 / 0.48 |
| ZP+P, the same flow with placed minutiae | 35.0 / 28.3 | 3.39 / 2.88 | 0.54 / 0.58 |
| Anguli, through real windows | 25.8 / 25.2 | 2.12 / 2.10 | 0.47 / 0.46 |
| SynFing, a GAN, through real windows | 28.7 / 25.5 | 1.67 / 1.92 | 0.33 / 0.38 |

mindtct / LEADER, all minutiae. Confidence intervals, by finger, are in the paper and in `results/expected/`.

- **The held-out rows have no h or count.** The held-out look read C(60) only, and the protocol allows each look once (`docs/held-out-protocol.md`).
- **Density doesn't explain the gap.** SFinGe has fewer minutiae a print than real and still about three times its C(60), and h, which doesn't scale with the number of neighbours, separates it as well.
- **Where these come from.** Anguli and SynFing are read through real prints' windows here (the paper's Table 8). Read on each image's own print, they give 2.23 / 2.22 and 1.66 / 2.03 (Table 7), and the minutiae column counts minutiae a window.

What else the paper finds:

- **It survives false minutiae.** Keeping only minutiae that recur across impressions moves real's C(60) to 0.89 / 1.00, and generators stay well above it.
- **It survives having no extractor against ZP.** Counting spirals in the phase gives real 1.03, against 4.22 for ZP.
- **Real doesn't screen.** Charges of a random field read -0.88 through the same windows, and real's sum stays positive at about +1.0 to +1.6.
- **Net charge closes on the flow,** with r about 0.90 on real, as the divergence theorem requires. What separates real is the excess X = ⟨Q²⟩ − ⟨N⟩, which adding independent minutiae can't move: 2.8 / 2.7 on real, 5.3 / 5.2 on Turing fronts laid on the same prints' low-passed flow, and 10 to 19 on the other generators.
- **On one ridge at 1.5 to 3 mm, real's sign has almost no memory:** by the extractors h is 0.01 / 0.05, against 0.24 or more on every generator. With no extractor (phase winding) this holds against ZP (0.61) but not against SFinGe (0.04, real −0.01), so against SFinGe it depends on which marks count as minutiae.
- **For zero-pole flows the cause lies in the flow.** At fixed spacing a flow that spreads forces net charge, and swapping the ridge mechanism took C(60) down by only a third. What real prints do instead isn't settled.

### What this doesn't show

- **Not new sensors.** The held-out fingers come from the same nine sensors as the tuning data. C(60) ranges from 0.03 to 1.51 across them. For real prints in general the unit is the sensor, so n is 9, and each sensor's mean rests on 10 fingers.
- **A generator at real's image quality isn't tested.** On clear ridges real reads about 1.3, so part of the low end of the band is quality. Two registered recipes for degrading generators failed their own fairness gates.
- **A registered long-range band failed.** On LEADER, h(60-100 px) overlaps between real and SFinGe, so no long-range difference from SFinGe is claimed.
- **The size of the gap is less certain than its direction.** With no extractor, SFinGe reads 1.76 against real's 1.03, under the 2.4 to 2.7 the extractors give, and at a matched ridge count SFinGe's excess is a third smaller. That extractor-free read is firm against ZP only: it behaves differently on SFinGe, for reasons we can't explain.
- **Whether charge matters to recognition is open.** Less charge helped no comparison in a matcher test, but those sets differ in far more than charge.
- **Plain prints only.** Real's numbers come from plain prints, which show little of the area beside the delta. Champod's rolled prints show one sign at up to 83 % there.
- **The flow's own bias isn't taken out.** h and C are read on raw signs. A null that keeps the mean sign the flow demands at each point, or correlations of the residual signs, isn't done, so how much of each positive h is the flow's and how much is arrangement is open. The paper's reads of flow agreement and of excess charge over independent signs (X) bear on it, but don't settle it.
- **Two parts of the lift aren't checked.** Pairs whose segment passes close to a core or delta, where the lift is least certain, are kept. And there is no edge correction for C(R): near the print's edge a minutia's disc is cut off, which pulls C toward 0. h, a mean over pairs, is far less affected.
- **The prior-art search isn't complete.** It missed Champod's tables, and the older literature was reached largely through secondary sources.

## What's here

    src/signed_charge/measure/      the measures: minutiae, signs, h(r), C(R), flux, winding
    src/signed_charge/generate/     the generators: Turing fronts, placed minutiae
    src/signed_charge/reads/        one script for each result behind the paper (CLAIMS.md)
    src/signed_charge/exploratory/  other experiments, not behind any paper claim and not checked
    src/ridgegen/                   the generator behind ZP and ZP+P
    scripts/                        setup, generation, reproduction and comparison
    results/expected/               every result as first run
    manifests/                      hashes of the data, and fingers left out of the matcher test
    docs/                           the held-out protocol
    CLAIMS.md                       each table and claim, with its command and result file
    DATA.md                         where to get what isn't included
    VERIFICATION.md                 what was rerun, and how it came out

## Reproducing the results

You need Linux, [uv](https://docs.astral.sh/uv/), gcc, make and curl, and about 20 GB of disk. The matcher test needs a CUDA 13 GPU. The rest runs on CPU, and a many-core machine helps: a full run takes most of a day on 24 threads.

**1. Set up.** This installs pinned Python packages and builds NBIS into `tools/`.

    scripts/setup.sh

**2. Get the data.** Follow [DATA.md](DATA.md), then:

    cp scripts/env.example.sh scripts/env.sh    # set the paths
    source scripts/env.sh
    .venv/bin/python scripts/check_data.py

**3. Run.** Results land in `results/verify/`, and each is compared with `results/expected/`. Finished steps are skipped, so a run can be restarted.

    .venv/bin/python scripts/reproduce.py quick      # controls and the main read, about 40 minutes
    .venv/bin/python scripts/reproduce.py generate   # regenerate the Turing fronts, some hours
    .venv/bin/python scripts/reproduce.py reads      # the rest of the paper's reads
    .venv/bin/python scripts/reproduce.py matcher    # the matcher test, GPU, about 6 hours

A single read runs on its own, for example:

    .venv/bin/python -m signed_charge.reads.signed_false results/verify/signed_false.json
    .venv/bin/python scripts/compare_results.py results/expected/signed_false.json results/verify/signed_false.json

**4. Compare with the paper.** [CLAIMS.md](CLAIMS.md) gives the field behind each table.

Runs are deterministic given the same data and tools, with seeds in the code. Results here came from one machine, so numbers elsewhere may differ in the last digits. The stored results carry a `_provenance` block; their commit hashes refer to the private research tree this was copied from.

## Entry numbers

Scripts and results are labelled E-nnn, D-nnn and O-nnn. These are entries in a dated lab notebook that isn't part of this repository: E is an experiment, D a decision, often with criteria set before a read, and O an observation.

## Licence

No licence has been chosen yet. All rights reserved, shared for verification only. Third-party items keep their own terms: see [DATA.md](DATA.md).
