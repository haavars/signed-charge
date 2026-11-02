# Held-out protocol for the FVC A sets

The rule under which the paper's held-out fingers were read. It was adopted on 2026-10-31, before any A image was read, and the look below ran the same day. This is the part of the project's protocol that concerns the paper. The rest covers later work that isn't in this repository.

## Why

Everything in the paper was developed on FVC B: 90 real fingers, in the open for weeks, with settings chosen against them. A set nobody has looked at is the only way to show a claim holds beyond the prints that shaped it. Such a set can be spent only once, so what each look may cost was fixed first.

## Data

The A sets of FVC2000, FVC2002 and FVC2004, from the supplementary material of *Handbook of Fingerprint Recognition*, 3rd edition (`74034_3_En_4_MOESM1_ESM.zip`, see DATA.md), checked against its reported sha256 on extraction.

- **Real**: DB1-3 A of each year, 9 sets, 100 fingers × 8 impressions, 7,200 prints.
- **Synthetic**: DB4 A of each year (SFinGe), 2,400 prints, a baseline only.
- FVC2000 DB1_A's image 70_5 is mislabelled and stays as it is, as FVC advises.
- These are new fingers from the same nine sensors as B. So a pass says "new fingers", not "new sensors".

## The split

Within each set, fingers are sorted by the sha256 of the text `D-030/<year>/<db>/<finger>`. The first 30 are **checking**, the other 70 **proving**. The rule is fixed by this text, so nobody chooses it after seeing an image. All of a finger's impressions go the same way. `signed_charge.reads.held_out_split` computes it.

- **Checking**: 30 × 8 × 9 = 2,160 real prints. These may be read a counted number of times, each look named and its criteria committed before it runs.
- **Proving**: 70 × 8 × 9 = 5,040 real prints. These are kept sealed for a later, frozen generator, to be scored once. No result in this repository reads them.

Nobody looks at A images by eye. Only summary numbers come out, and no A image appears in this repository.

## The paper's look: real prints' signed band, on the checking fingers

This is the only look the paper uses. It involves no model, so it could run first. Its criteria were set before the run (`signed_charge.reads.held_out_signed`, result in `results/expected/held_out_look1.json`):

- C(60) on both extractors, mindtct and LEADER, per sensor and pooled, with CIs by finger.
- **It passes only if all three hold, on both extractors**:
  1. the pooled CI on A overlaps the pooled CI on B;
  2. at least 8 of the 9 sensors lie in the band B's sensors span: 0.03 to 1.40 on mindtct, 0.25 to 1.51 on LEADER;
  3. SFinGe's pooled CI on DB4 A's checking fingers lies wholly above real's.
- A fail would have counted as a fail of the paper's claim, and wouldn't have been retried on these fingers.

It passed: real 0.81 / 1.07 on 270 fingers, SFinGe 2.38 / 2.73 on 90.

## If the seal is broken

If anything from the proving fingers reaches a decision by accident, it's written up, and that part of a claim is stated as not held out.

## Two notes

- **LEADER's training data**: LEADER was trained on all images of FVC2002 DB2-A, FVC2002 DB3-A, FVC2004 DB1-A and FVC2004 DB3-A, and tuned on FVC2000 DB2-A, checking fingers included. That was learned after this look ran. mindtct, which wasn't trained on FVC, passed the look on its own as well.
- **Rerunning**: `scripts/reproduce.py` doesn't rerun the look, since each look is allowed once. Its stored result stands as first run (VERIFICATION.md).
