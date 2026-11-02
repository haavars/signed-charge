"""Masters whose minutiae are chosen, not grown (step 3, route b): phase_masters.py OUT
[PEOPLE] [VARIANT...]

`phase_edit.py` showed that a grown master's minutiae can be removed and added two at a time as spiral-phase
dipoles, cleanly. A master here is master model 2's, then:

1. **Cleared**: the growth's own minutiae removed in pairs of opposite spiral up to `MAX_SEPARATION` periods
   apart, nearest first, in rounds of edits that don't overlap, until no more go; none within reach of a core or
   delta (where ψ's convention turns). What's left is mostly the minutiae the flow needs (where ridges must start
   or stop because they spread or close up) and same-spiral clusters.
2. **Pairs added**, at `pairs` per 100 mm² of master: an ending facing a fork across the ridges (a ridge shifted,
   the commonest real close pair), two forks or a fork and an ending on a diagonal, or a break along a ridge (two
   endings facing), in the shares `mix`; placed uniformly, clear of other minutiae.
3. **Polished**: two passes of the growth's Gabor filtering from the edited pattern, so its texture is the
   growth's.

The ground truth is extracted from the final master as always (`minutiae.master`), so it stays exact. Plain
fingertips inked flat at 500 ppi from these masters (as `minutiae_stats.procedural_tip`), against FVC 2000/2002/2004
DB1-3 B: density, bifurcations, close pairs (`minutiae_pairs.py`) and the minutiae-histogram classifier
(`minutiae_stats.py`).

Run: python -m signed_charge.generate.phase_masters OUT.json 20 model2 cleared ...
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import provenance
from signed_charge.generate import minutiae_growth as growth
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.generate import phase_edit as pe

# (clear, pairs per 100 mm² of master, mix of across / diagonal / along)
# Masters drawn from one phase (`phase_global.py`): (smoothing of the flow's spirals, pairs per 100 mm²).
GLOBAL = {"global_flow": (2.0, 0.0), "global_add1": (2.0, 1.0), "global_add2": (2.0, 2.0), "global_add3": (2.0, 3.0), "global_add2_s3": (3.0, 2.0)}
# (smoothing, pairs per 100 mm², the flow's spirals' least spacing in periods, mix across / diagonal / along)
GLOBAL.update(
    {
        "global_free2_sp2": (2.0, 2.0, 2.0, (0, 0, 0, 1)),
        "global_free3_sp2": (2.0, 3.0, 2.0, (0, 0, 0, 1)),
        "global_free4_sp2": (2.0, 4.0, 2.0, (0, 0, 0, 1)),
        "global_free4_sp3": (2.0, 4.0, 3.0, (0, 0, 0, 1)),
    }
)
GLOBAL.update(
    {
        "global_s4_free4_sp3": (4.0, 4.0, 3.0, (0, 0, 0, 1)),
        "global_s4_free5_sp3": (4.0, 5.0, 3.0, (0, 0, 0, 1)),
        "global_s6_free5_sp3": (6.0, 5.0, 3.0, (0, 0, 0, 1)),
    }
)
# (…, regional clumping, pairs below the core against above[, error diffusion, spacing weight])
FREE = (0, 0, 0, 1)
GLOBAL.update(
    {
        "d_s0_f2": (0.0, 2.0, 0.0, FREE, 0.5, 1.6, False, 0.05),
        "d_s0_sp2_f2": (0.0, 2.0, 2.0, FREE, 0.5, 1.6, False, 0.05),
        "d_s1_sp2_f3": (1.0, 3.0, 2.0, FREE, 0.5, 1.6, False, 0.05),
    }
)
GLOBAL.update(
    {
        "e_vary": (1.0, 3.0, 2.0, FREE, 0.5, 1.6, False, 0.05, (3.0, 9.0)),  # E-044: spacing at real's spread
        "e_far": (1.0, 3.0, 2.0, FREE, 0.5, 1.6, False, 0.05, None, (8.0, 16.0)),
    }
)  # E-044: its pairs far apart, so signs mix
GLOBAL.update(
    {
        "m3_clump0.5": (6.0, 5.0, 3.0, FREE, 0.5, 1.0),
        "m3_below1.6": (6.0, 5.0, 3.0, FREE, 0.0, 1.6),
        "m3_clump0.5_below1.6": (6.0, 5.0, 3.0, FREE, 0.5, 1.6),
        "m3_clump0.8_below1.6": (6.0, 5.0, 3.0, FREE, 0.8, 1.6),
    }
)
GLOBAL.update({"global_add2_sp2": (2.0, 2.0, 2.0), "global_add2_sp3": (2.0, 2.0, 3.0), "global_add3_sp3_across": (2.0, 3.0, 3.0, (0.8, 0.1, 0.1))})
VARIANTS = {
    "model2": (False, 0, None),
    "cleared": (True, 0, None),
    "cleared_add2": (True, 2.0, (0.5, 0.3, 0.2)),
    "cleared_add4": (True, 4.0, (0.5, 0.3, 0.2)),
    "cleared_add6": (True, 6.0, (0.5, 0.3, 0.2)),
}
MAX_SEPARATION = 6  # periods
EDGE = 30  # px: the master's edge band, left as grown (prints' contact starts about 28 px in)
ROUNDS = 14
PASSES = 2
HOLD = 1.0  # periods
MAX_TURN = np.radians(85)  # the convention folds where the flow has turned 90° from the centre
MIN_WINDOW = 2.0  # periods: the least an edit's half window may shrink to near a core or delta
CLEARANCE = 3.0  # periods: an added pair from any minutia already there
PAIR_SALT = 41
MM2 = (25.4 / 500) ** 2
JOBS = 8


def inner(points, w, h):
    return [p for p in points if min(p["x"], p["y"], w - p["x"], h - p["y"]) > EDGE]


def fit(m, mid, half, least):
    """An edit's window at `mid`: `half` px, or less to stay a period clear of every core and delta, but not under
    `least` (None if it can't)."""
    clear_of = min([np.hypot(mid[0] - c[0], mid[1] - c[1]) - m.period for c in m.cores + m.deltas], default=1e9)
    half = int(min(half, clear_of))
    return half if half >= least else None


def near_singular(m, mid, half, period, beta2=None):
    """Whether an edit at `mid` would reach a core or delta, or (given `beta2`) a part of the flow turning more than
    `MAX_TURN` from its centre's, where ψ's convention can fold."""
    if any(np.hypot(mid[0] - c[0], mid[1] - c[1]) < half + period for c in m.cores + m.deltas):
        return True
    if beta2 is None:
        return False
    h, w = beta2.shape
    cx, cy = int(np.clip(mid[0], 0, w - 1)), int(np.clip(mid[1], 0, h - 1))
    window = beta2[max(0, cy - half) : cy + half + 1 : 4, max(0, cx - half) : cx + half + 1 : 4]
    turn = np.abs((window - beta2[cy, cx] + np.pi) % (2 * np.pi) - np.pi) / 2
    return float(turn.max()) > MAX_TURN


def clear(m, ridges):
    """Remove the growth's minutiae in opposite-spiral pairs, nearest first, until none go."""
    period = m.period
    h, w = ridges.shape
    for _ in range(ROUNDS):
        v = pe.vortex(ridges)
        beta2 = pe.doubled_direction(v)
        points = inner(pe.found(ridges), w, h)
        candidates = []
        for i, p in enumerate(points):
            for q in points[i + 1 :]:
                d = np.hypot(p["x"] - q["x"], p["y"] - q["y"])
                if d < MAX_SEPARATION * period:
                    candidates.append((d, p, q))
        candidates.sort(key=lambda c: c[0])
        used, edits = [], []
        for _, p, q in candidates:
            mid = np.array([(p["x"] + q["x"]) / 2, (p["y"] + q["y"]) / 2])
            sep = np.hypot(p["x"] - q["x"], p["y"] - q["y"])
            half = fit(m, mid, pe.reach((p["x"], p["y"]), (q["x"], q["y"]), period), max(2.5 * sep, MIN_WINDOW * period))
            if half is None or any(np.hypot(*(mid - u)) < half + hu for u, hu in used):
                continue
            if near_singular(m, mid, half, period, beta2):
                continue
            psi, (ys, xs) = pe.local_phase(ridges, v, beta2, mid, half)
            res = pe.residues(psi)
            wp, zp = pe.spiral_at(res, p["x"] - xs.start, p["y"] - ys.start, 0.6 * period)
            wq, zq = pe.spiral_at(res, q["x"] - xs.start, q["y"] - ys.start, 0.6 * period)
            if wp and wp == -wq and zp != zq:
                zp, zq = (zp[0] + xs.start, zp[1] + ys.start), (zq[0] + xs.start, zq[1] + ys.start)
                edits.append((zq, zp) if wp > 0 else (zp, zq))  # a − on the +, a + on the −
                used.append((mid, half))
        if not edits:
            break
        for (plus, minus), (_, half) in zip(edits, used):
            ridges = pe.apply_dipole(ridges, v, beta2, plus, minus, period, half)
    return ridges


def add_pairs(m, ridges, rate, mix, rng):
    """`rate` dipoles per 100 mm² of master, laid out across / diagonal / along in the shares `mix`."""
    period = m.period
    h, w = ridges.shape
    area = (w - 2 * EDGE) * (h - 2 * EDGE) * MM2
    wanted = rng.poisson(rate * area / 100)
    placed, tries, spirals = 0, 0, []
    while placed < wanted and tries < 20:
        tries += 1
        v = pe.vortex(ridges)
        beta2 = pe.doubled_direction(v)
        points = pe.found(ridges)
        used = []
        for _ in range(400):
            if placed >= wanted:
                break
            x, y = rng.uniform(EDGE + 10, w - EDGE - 10), rng.uniform(EDGE + 10, h - EDGE - 10)
            half = fit(m, (x, y), int(pe.WINDOW * period), MIN_WINDOW * period)
            if half is None or any(np.hypot(x - u[0], y - u[1]) < half + hu for u, hu in used):
                continue
            if near_singular(m, (x, y), half, period, beta2):
                continue
            if any((p["x"] - x) ** 2 + (p["y"] - y) ** 2 < (CLEARANCE * period) ** 2 for p in points):
                continue
            beta = beta2[int(y), int(x)] / 2  # across the ridges
            layout = rng.choice(3, p=mix)
            a, sep = [(beta, 1.0), (beta + rng.choice([-1, 1]) * np.pi / 4, rng.uniform(1.0, 1.5)), (beta + np.pi / 2, rng.uniform(1.5, 3.0))][layout]
            d = sep * period * np.array([np.cos(a), np.sin(a)]) / 2
            sign = rng.choice([-1, 1])
            ridges = pe.apply_dipole(ridges, v, beta2, (x, y) + sign * d, (x, y) - sign * d, period, half)
            spirals += [(x, y) + d, (x, y) - d]
            used.append(((x, y), half))
            placed += 1
    return ridges, spirals


def use(variant):
    """Draw masters in this process as `variant` does."""
    from ridgegen import finger as fingers
    from ridgegen import minutiae, synthesis
    from ridgegen.synthesis import rng_for

    growth.use("base")
    if not hasattr(use, "master"):
        use.master = fingers.master
    cleared, rate, mix = VARIANTS.get(variant, (True, 0, None))

    def master(seed, finger, model=fingers.MODEL):
        m = use.master(seed, finger, model)
        if variant in GLOBAL:
            key = (variant, seed, finger)
            if key not in use.cache:
                from signed_charge.generate import phase_global as pg

                defaults = (2.0, 0.0, 0.0, (0.5, 0.3, 0.2), 0.0, 1.0, False, None, None, pg.FREE_SEPARATION)
                smooth, pairs, spacing, mix, clump, below, dither, weight, vary, free = GLOBAL[variant] + defaults[len(GLOBAL[variant]) :]
                ridges, _ = pg.master_ridges(m, seed, finger, smooth, pairs, mix, spacing, clump, below, dither, weight, vary, free)
                use.cache.clear()
                use.cache[key] = type(m)(**{**m.__dict__, "ridges": ridges, "minutiae": minutiae.master(finger, ridges)})
            return use.cache[key]
        if not cleared:
            return m
        key = (variant, seed, finger)
        if key not in use.cache:
            ridges = clear(m, m.ridges.astype(np.float64))
            spirals = []
            if rate:
                ridges, spirals = add_pairs(m, ridges, rate, mix, rng_for(seed, finger, PAIR_SALT))
            # The added pairs are held through the polish, which would anneal them (a disc of `HOLD` periods).
            held = np.zeros(ridges.shape, bool)
            yy, xx = np.mgrid[0 : ridges.shape[0], 0 : ridges.shape[1]]
            for x, y in spirals:
                r = int(HOLD * m.period) + 1
                y0, y1, x0, x1 = max(0, int(y) - r), int(y) + r + 1, max(0, int(x) - r), int(x) + r + 1
                held[y0:y1, x0:x1] |= (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2 < (HOLD * m.period) ** 2

            class Start:  # the growth's first image is standard_normal × 0.2: here, the edited pattern
                def standard_normal(self, shape):
                    return (ridges / 0.2).astype(np.float32)

            fixed = (held, ridges.astype(np.float32)) if held.any() else None
            ridges = synthesis.grow_ridges(m.orientation, m.period, Start(), iterations=PASSES, fixed=fixed)
            use.cache.clear()
            use.cache[key] = type(m)(**{**m.__dict__, "ridges": ridges, "minutiae": minutiae.master(finger, ridges)})
        return use.cache[key]

    use.cache = getattr(use, "cache", {})
    fingers.master = master


def tip(task):
    from ridgegen import fingerprints
    from ridgegen.synthesis import rng_for

    variant, seed, fgp, capture = task
    use(variant)
    rng = rng_for(seed, fgp, fingerprints.PLAIN, capture - 1, 13 if fgp <= 5 else 14)
    _, ridges, contact, _, _, pasted = fingerprints.plain_finger(seed, fgp, capture - 1, rng)
    im = np.where((contact > 0.5) & ~pasted & (ridges > 0), 40, 235).astype(np.uint8)
    from ridgegen import finger as fingers

    m = fingers.master(seed, fgp)
    h, w = m.ridges.shape
    key = (variant, f"PH{seed}/{fgp}", capture)
    _, stats = ms.describe((key, im))
    _, pairs = mp.describe((key, im))
    return key, stats, pairs, len(inner(m.minutiae, w, h)) if capture == 1 else None


def main(out, people=20, variants=("model2",)):
    started = provenance.start()
    real = list(cf.fvc(("DB1", "DB2", "DB3")))
    tasks = [(v, s, f, c) for v in variants for s in range(3000, 3000 + people) for f in (2, 3, 4, 5, 7, 8, 9, 10) for c in (1, 2)]
    with ProcessPoolExecutor(JOBS) as pool:
        real_stats = [(k, v) for k, v in pool.map(ms.describe, real, chunksize=4) if v]
        mine = list(pool.map(tip, tasks, chunksize=2))
    report = {"real": ms.summary([v for _, v in real_stats])}
    for v in variants:
        xs = [x for x in mine if x[0][0] == v]
        stats = [(k, s) for k, s, _, _ in xs if s]
        pairs = [p for _, _, p, _ in xs if p]
        report[v] = {
            **ms.summary([s for _, s in stats]),
            "accuracy_mh": cf.accuracy(real_stats, stats, "mh"),
            "pairs": {k: val for k, val in mp.summary(pairs).items() if k != "nn_share"},
            "master_minutiae_median": float(np.median([n for *_, n in xs if n is not None])),
        }
        print(json.dumps({v: report[v]}), flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[2:]
    main(sys.argv[1], int(args[0]) if args else 20, tuple(args[1:]) or ("model2",))
