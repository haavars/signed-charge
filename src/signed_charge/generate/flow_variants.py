"""Bending the flow and roughening it, as candidates for master model 3 (step 4): flow_variants.py OUT [PEOPLE] [VARIANT...]

Step 2's screens found model 2's flow more regular than real: at high curvature it lacks real prints'
non-conformality (Imdahl et al. 2017), as SFinGe's does, and where it's straight it's smoother than every real
sensor's. Two changes, on master model 2's masters (everything else as it draws it, the same people):

- **Bending** (Vizcaya & Gerhardt 1996, as SFinGe uses): each singular point's contribution ½ arg(z − z_k) becomes
  ½ g_k(arg(z − z_k)), g_k piecewise linear through 8 knots α_i = −π + iπ/4 with g_k(α_i) = α_i + Δ_k,i, Δ drawn
  per finger, N(0, `bend`) radians, clipped to ±π/8 so g stays increasing. Not conformal (g(arg z) isn't
  harmonic), so the field bends near the singular points as real ones do, and two fingers with the same singular
  points no longer share a field.
- **Fine wander**: smooth noise at `scale` px (model 2's wobble is at 180 px) with amplitude `fine` radians, added to
  the orientation before it's smoothed.
- **Ears**: smooth noise at `EARS_SCALE` px, weighted by how fast the model's field turns (full from `EARS_TURN`),
  so it bends the flow where it curves most, near the singular points (Imdahl et al.'s "rabbit ears" of extreme
  non-conformality at high curvature).
- **Flat**: how much of model 2's far-field flattening (towards horizontal above the core and towards the crease)
  to keep: it straightens ridges real fingers arc.

Each from its own random stream, so a variant changes only the flow. Plain fingertips inked flat at 500 ppi (as
`signed_charge.measure.field_similarity --procedural`), against FVC 2000/2002/2004 DB1-3 B: the conformality screen on
mindtct's binarised images (`conformality.py --binarise`: the high- and low-curvature extreme shares and the
classifier) and the flow tail (`field_similarity.py`: between-people distances below FVC's 1st and 5th percentiles).

Run: python -m signed_charge.generate.flow_variants OUT.json 30 model2 bend0.15 ...
"""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.generate import minutiae_growth as growth
from signed_charge.measure.conformality import fs

# (bend: Δ's SD in radians, fine: amplitude in radians, fine's scale in px[, ears: amplitude in radians where the
# field turns fastest, flat: how much of model 2's far-field flattening to keep])
VARIANTS = {"model2": (0.0, 0.0, 60), "bend0.15": (0.15, 0.0, 60), "bend0.3": (0.3, 0.0, 60),
            "fine0.08": (0.0, 0.08, 60), "fine0.15": (0.0, 0.15, 60), "bend0.3_fine0.08": (0.3, 0.08, 60),
            "fine0.06": (0.0, 0.06, 60), "bend0.1_fine0.06": (0.1, 0.06, 60), "fine0.08_flat0.5": (0.0, 0.08, 60, 0.0, 0.5),
            "fine0.08_ears0.25": (0.0, 0.08, 60, 0.25, 1.0), "bend0.1_fine0.06_ears0.25_flat0.5": (0.1, 0.06, 60, 0.25, 0.5)}
BEND_SALT, FINE_SALT, EARS_SALT = 31, 32, 33
EARS_SCALE = 40  # px
EARS_TURN = 0.02  # rad/px: where the field turns this fast or faster (about 25 px from a core), the full amplitude
KNOTS = -np.pi + np.arange(9) * np.pi / 4  # the 8 knots and −π's repeat at π
JOBS = 8


def bent_orientation(shape, cores, deltas, bends, base=0.0):
    """`synthesis.zero_pole_orientation` with each singular point's angle through its own g (`bends`: one array of 8
    knot offsets per core, then per delta)."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = np.zeros(shape, np.float32) + base
    for (px, py), delta, sign in zip(list(cores) + list(deltas), bends, [1] * len(cores) + [-1] * len(deltas)):
        a = np.arctan2(y - py, x - px)
        g = np.interp(a, KNOTS, KNOTS + np.append(delta, delta[0]))
        theta += sign * 0.5 * g
    return theta


def use(variant, seed, fgp):
    """Draw masters in this process as `variant` does, for the person `seed`'s finger `fgp`."""
    from ridgegen import finger as fingers
    from ridgegen import synthesis
    from ridgegen.synthesis import rng_for

    growth.use("base")  # one thread, model 2's growth
    if not hasattr(use, "zero_pole"):
        use.zero_pole, use.noise, use.flatten = fingers.zero_pole_orientation, fingers.smooth_noise, fingers.flatten_far_field
    bend, fine, fine_scale, ears, flat = (VARIANTS[variant] + (0.0, 1.0))[:5]

    def flatten(theta, cores, deltas):
        flattened = use.flatten(theta, cores, deltas)
        if flat != 1.0:  # keep `flat` of the pull: interpolate the doubled angle
            c = flat * np.cos(2 * flattened) + (1 - flat) * np.cos(2 * theta)
            s_ = flat * np.sin(2 * flattened) + (1 - flat) * np.sin(2 * theta)
            flattened = np.mod(0.5 * np.arctan2(s_, c), np.pi).astype(np.float32)
        if ears:
            # How fast the model's field turns, rad/px, from its doubled angle's gradient.
            c, s_ = np.cos(2 * theta), np.sin(2 * theta)
            turn = 0.5 * np.sqrt(sum(g**2 for g in np.gradient(c)) + sum(g**2 for g in np.gradient(s_)))
            weight = np.clip(turn / EARS_TURN, 0, 1)
            flattened = flattened + weight * use.noise(theta.shape, rng_for(seed, fgp, EARS_SALT), EARS_SCALE, ears)
        return flattened

    def zero_pole(shape, cores, deltas, base=0.0):
        if not bend:
            return use.zero_pole(shape, cores, deltas, base)
        rng = rng_for(seed, fgp, BEND_SALT)
        bends = [np.clip(rng.normal(0, bend, 8), -np.pi / 8, np.pi / 8) for _ in list(cores) + list(deltas)]
        return bent_orientation(shape, cores, deltas, bends, base)

    def noise(shape, rng, scale, amplitude=1.0):
        field = use.noise(shape, rng, scale, amplitude)
        if scale == 180 and fine:  # the master's wobble: add the finer wander from its own stream
            field = field + use.noise(shape, rng_for(seed, fgp, FINE_SALT), fine_scale, fine)
        return field

    fingers.zero_pole_orientation, fingers.smooth_noise, fingers.flatten_far_field = zero_pole, noise, flatten
    synthesis.use_device("cpu")
    if getattr(use, "drawing", None) != (variant, seed, fgp):
        fingers.master.cache_clear()  # its masters were drawn for another variant, or by these patches for another finger
        use.drawing = (variant, seed, fgp)


def tip(task):
    """A plain fingertip inked flat at 500 ppi, its flow drawn as `variant` does; and the two screens' descriptions."""
    from ridgegen import fingerprints
    from ridgegen.synthesis import rng_for

    variant, seed, fgp, capture = task
    use(variant, seed, fgp)
    rng = rng_for(seed, fgp, fingerprints.PLAIN, capture - 1, 13 if fgp <= 5 else 14)
    _, ridges, contact, _, _, pasted = fingerprints.plain_finger(seed, fgp, capture - 1, rng)
    im = np.where((contact > 0.5) & ~pasted & (ridges > 0), 40, 235).astype(np.uint8)
    return describe(((variant, f"PH{seed}/{fgp}", capture), im))


def describe(item):
    key, im = item
    _, conformal = cf.describe((key, cf.binarised(im)))
    # The flow tail compares first impressions only; what `fs.distance` needs of them.
    flow = fs.describe(im) if key[2] == 1 else None
    if flow and flow.get("cores") == 1:
        flow = {"cores": 1, "core": flow["core"], "theta": flow["theta"], "inner": flow["inner"]}
    else:
        flow = None
    return key, conformal, flow


def tail(described):
    """Between-people flow distances (first impressions, different people), as `field_similarity.run`."""
    firsts = {k[1]: f for k, _, f in described if f and k[2] == 1}
    out = []
    for a, b in itertools.combinations(sorted(firsts), 2):
        if a.split("/")[0] == b.split("/")[0] and a.startswith("PH"):
            continue
        d = fs.distance(firsts[a], firsts[b])
        if d is not None:
            out.append(d)
    return np.array(out)


def shares(xs):
    h = np.sum([np.array(v["h10"]).reshape(10, 10) * v["n"] for v in xs], 0)
    low, high = h[:3], h[5:]
    return {"extreme_low_curvature": float(low[:, [0, -1]].sum() / low.sum()),
            "extreme_high_curvature": float(high[:, [0, -1]].sum() / max(high.sum(), 1)),
            "lowest_curvature_share": float(h[0].sum() / h.sum())}


def main(out, people=30, variants=("model2",)):
    real_items = [((k[0], k[1], k[2]), im) for k, im in cf.fvc(("DB1", "DB2", "DB3"))]
    tasks = [(v, s, f, c) for v in variants for s in range(3000, 3000 + people) for f in (2, 3, 4, 5, 7, 8, 9, 10)
             for c in (1, 2)]
    with ProcessPoolExecutor(JOBS) as pool:
        real = list(pool.map(describe, real_items, chunksize=4))
        mine = list(pool.map(tip, tasks, chunksize=2))
    real_tail = tail([(("FVC", k[1], k[2]), c, f) for k, c, f in real])
    p1, p5 = np.percentile(real_tail, 1), np.percentile(real_tail, 5)
    real_c = [(k, c) for k, c, _ in real if c]
    report = {"real": {**shares([c for _, c in real_c]), "between_median": float(np.median(real_tail))}}
    for v in variants:
        xs = [(k, c, f) for k, c, f in mine if k[0] == v]
        t = tail(xs)
        cs = [(k, c) for k, c, _ in xs if c]
        report[v] = {**shares([c for _, c in cs]), "accuracy_h10": cf.accuracy(real_c, cs, "h10"),
                     "between_median": float(np.median(t)), "below_fvc_p1": float(np.mean(t < p1)),
                     "below_fvc_p5": float(np.mean(t < p5)), "prints": len(xs)}
        print(json.dumps({v: report[v]}), flush=True)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[2:]
    main(sys.argv[1], int(args[0]) if args else 30, tuple(args[1:]) or ("model2",))
