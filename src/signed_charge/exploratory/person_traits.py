"""A person's fingers alike as real ones are (step 5): person_traits.py [PEOPLE]

Model 2 draws each finger's pattern independently (by hand and finger, `finger.PATTERNS`) and its ridge count with
a 0.6 SD person share (`COUNT_PERSON`): its fingers' ridge counts correlate 0.11-0.29 between the same finger of the
two hands and 0.14-0.29 between others, against 0.76-0.82 and 0.41-0.70 real (Medland et al. 2007, about 2,100
Australian twins and siblings, arches counting 0). A design fit, simulated here without ridgegen:

- **Patterns**, a liability threshold model: finger f's liability z = a·P + b·H_name + √(1 − a² − b²)·e (P the
  person's, H shared by the finger's two hands, e its own, all N(0, 1)); its pattern the class Φ(z) falls in, in
  the order arches, radial loop, ulnar loop, whorls, with the finger's own frequencies from `PATTERNS`, so every
  finger's pattern frequencies stay exactly as published.
- **Ridge counts**, as model 2 (`finger.ridge_count`) but with the same three levels: c·P_count + d·H_count + √(1 −
  c² − d²)·e.

For each (a, b, c, d) on a grid: the counts' correlations, same finger of the two hands and different fingers, and
the total ridge count's mean and SD. Prints the grid's closest fits to Medland's.

Run: python -m signed_charge.exploratory.person_traits 20000
"""
import itertools
import sys

import numpy as np
from scipy.stats import norm

from ridgegen import finger as fingers

ORDER = ("arches", "radial_loop", "ulnar_loop", "whorl", "central_pocket_loop", "double_loop", "accidental")
TARGET_SAME, TARGET_OTHER = (0.76, 0.82), (0.41, 0.70)


def simulate(people, a, b, c, d, seed=0):
    rng = np.random.default_rng(seed)
    p, pc = rng.standard_normal(people), rng.standard_normal(people)
    shared = {n: rng.standard_normal(people) for n in fingers.LOOP_COUNT}
    shared_count = {n: rng.standard_normal(people) for n in fingers.LOOP_COUNT}
    counts = np.zeros((people, 10))
    for fgp in range(1, 11):
        name, hand = fingers.name_of(fgp), fingers.hand_of(fgp)
        z = a * p + b * shared[name] + np.sqrt(max(0.0, 1 - a * a - b * b)) * rng.standard_normal(people)
        weights = dict(zip(fingers.PATTERN_KINDS, fingers.PATTERNS[(hand, name)]))
        cum = np.cumsum([weights[k] for k in ORDER]) / sum(weights.values())
        kind = np.searchsorted(cum, norm.cdf(z))
        kind = np.minimum(kind, len(ORDER) - 1)
        mean, sd = fingers.LOOP_COUNT[name]
        own = np.sqrt(max(0.0, 1 - c * c - d * d)) * rng.standard_normal(people)
        level = c * pc + d * shared_count[name] + own
        loop_mean = mean - np.where(np.array(ORDER)[kind] == "radial_loop", fingers.RADIAL_LESS, 0.0)
        whorl = kind >= 3
        value = np.where(whorl, mean + fingers.WHORL_MORE + fingers.WHORL_SD * level, loop_mean + sd * level)
        counts[:, fgp - 1] = np.where(kind == 0, 0.0, np.clip(value, *fingers.COUNT_RANGE))
    return counts


def correlations(counts):
    same = [np.corrcoef(counts[:, f], counts[:, f + 5])[0, 1] for f in range(5)]
    other = [np.corrcoef(counts[:, i], counts[:, j])[0, 1] for i, j in itertools.combinations(range(10), 2)
             if j - i != 5]
    return float(np.mean(same)), float(np.median(other)), float(min(other)), float(max(other))


def main(people=20000):
    rows = []
    for a, b, c, d in itertools.product((0.0, 0.4, 0.55, 0.7), (0.0, 0.3, 0.45, 0.6), (0.6, 0.7, 0.8), (0.0, 0.3, 0.45)):
        if a * a + b * b > 1 or c * c + d * d > 1:
            continue
        counts = simulate(people, a, b, c, d)
        same, other, lo, hi = correlations(counts)
        tfrc = counts.sum(1)
        miss = abs(same - np.mean(TARGET_SAME)) + abs(other - np.mean(TARGET_OTHER))
        rows.append((miss, a, b, c, d, same, other, lo, hi, tfrc.mean(), tfrc.std()))
    print("model 2 (a=b=d=0, c=0.6):", [round(x, 2) for x in rows[[r[1:5] for r in rows].index((0.0, 0.0, 0.6, 0.0))][5:]])
    print("a     b     c     d     same  other(med, min-max)   TFRC mean, SD")
    for r in sorted(rows)[:8]:
        print(f"{r[1]:.2f}  {r[2]:.2f}  {r[3]:.2f}  {r[4]:.2f}  {r[5]:.2f}  {r[6]:.2f} ({r[7]:.2f}-{r[8]:.2f})   {r[9]:.0f}, {r[10]:.0f}")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
