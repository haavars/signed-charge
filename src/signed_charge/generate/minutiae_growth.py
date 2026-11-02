"""Where a master's minutiae come from in the ridge growth (step 3, route a): minutiae_growth.py OUT

Phantom's rolled prints carry about 179 minutiae (ground truth; mindtct finds 161) against 85-115 real, and its
fingertips about 1.4 times FVC's density by the same extractor (`minutiae_stats.py`). Masters grow ridges from
white noise by repeated oriented Gabor filtering (`synthesis.grow_ridges`, 14 passes, a round envelope of
0.55 periods); every defect the noise leaves that the passes don't anneal away is a minutia. Two levers, varied here
on master model 2's masters: more passes (defects annihilate in pairs as the pattern coarsens), and an envelope
longer along the ridges than across (`ELONGATION`: ridges continue further before a defect starts one).

For each variant and master: minutiae within the rolled print's reach (an ellipse about the master's centre, as
`fingerprints.rolled_capture` places its contact), the bifurcation share, and below against above the core. CPU.

Run: python -m signed_charge.generate.minutiae_growth OUT.jsonl [MASTERS]
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

VARIANTS = {"base": (14, 1.0), "passes30": (30, 1.0), "passes60": (60, 1.0), "long1.5": (14, 1.5),
            "long2": (14, 2.0), "long1.5_passes30": (30, 1.5), "long2_passes30": (30, 2.0),
            "long2_passes45": (45, 2.0), "long2.5_passes30": (30, 2.5)}
# With breaks planted after growth (`plant_breaks`): (passes, elongation, breaks per 100 mm², share that are a short
# ridge's two ends).
VARIANTS.update({"long2_passes30_breaks0.75": (30, 2.0, 0.75, 0.4), "long2_passes30_breaks1.5": (30, 2.0, 1.5, 0.4),
                 "long2_passes30_breaks1.5_frag7": (30, 2.0, 1.5, 0.7), "long2_passes30_breaks4": (30, 2.0, 4.0, 0.4)})
MM2 = (25.4 / 500) ** 2
RUN = ("base", "passes30", "passes60", "long1.5", "long2", "long1.5_passes30")  # the first run's
FINGERS = (2, 3, 7, 8, 1)
JOBS = 8  # processes, one thread each (`OneThread`)


def elongated(elongation):
    from ridgegen import synthesis

    def gabor_kernel(bin_index, period):
        theta = np.pi * bin_index / synthesis.ORIENTATION_BINS
        across_sigma, along_sigma = 0.55 * period, 0.55 * period * elongation
        radius = int(np.ceil(2.6 * along_sigma))
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
        across = -x * np.sin(theta) + y * np.cos(theta)
        along = x * np.cos(theta) + y * np.sin(theta)
        kernel = np.exp(-across**2 / (2 * across_sigma**2) - along**2 / (2 * along_sigma**2))
        kernel = kernel * np.cos(2 * np.pi * across / period)
        kernel -= kernel.mean()
        kernel /= np.abs(kernel).sum()
        return kernel.astype(np.float32)

    return gabor_kernel


class OneThread:
    """scipy.fft as `synthesis` uses it, on one thread (it asks for every core), so 8 masters at once take 8."""

    def __init__(self, fft):
        self.fft = fft

    def __getattr__(self, name):
        f = getattr(self.fft, name)
        return lambda *a, **kw: f(*a, **{**kw, "workers": 1})


def plant_breaks(ridges, theta, period, rng, rate, fragments):
    """Breaks in the grown ridges: at `rate` per 100 mm², a ridge cut across for 0.5-1.2 periods, its two ends
    facing each other; a `fragments` share of them with a second cut 1-3 periods along the same ridge, leaving a
    short ridge (or, at under a period, a dot) between. Only the cut ridge's own pixels change."""
    out = ridges.copy()
    h, w = ridges.shape
    p = period if np.isscalar(period) else None
    centres = np.argwhere(ridges > 0.8)
    n = rng.poisson(rate * h * w * MM2 / 100)
    for y, x in centres[rng.choice(len(centres), size=min(n, len(centres)), replace=False)]:
        per = p or float(period[y, x])
        t = np.array([np.cos(theta[y, x]), np.sin(theta[y, x])])
        cuts = [0.0]
        if rng.random() < fragments:
            cuts.append(float(rng.choice([-1, 1]) * rng.uniform(1.0, 3.0) * per))
        for offset in cuts:
            cx, cy = x + offset * t[0], y + offset * t[1]
            half = rng.uniform(0.25, 0.6) * per
            r = int(np.ceil(half + per))
            y0, y1, x0, x1 = int(max(0, cy - r)), int(min(h, cy + r + 1)), int(max(0, cx - r)), int(min(w, cx + r + 1))
            if y1 <= y0 or x1 <= x0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            along = (xx - cx) * t[0] + (yy - cy) * t[1]
            across = -(xx - cx) * t[1] + (yy - cy) * t[0]
            # Soft-edged, within this ridge only: across less than half a period, and on ridge pixels.
            m = np.clip((half - np.abs(along)) / 1.5 + 0.5, 0, 1) * (np.abs(across) < 0.45 * per)
            m *= ridges[y0:y1, x0:x1] > 0
            block = out[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = block * (1 - m) - np.abs(block) * m
    return out


def use(variant):
    """Grow masters in this process as `variant` does (passes, envelope), on one thread, forgetting any grown
    before."""
    from ridgegen import finger as fingers
    from ridgegen import synthesis

    if not isinstance(synthesis.fft, OneThread):
        synthesis.fft = OneThread(synthesis.fft)
        use.grow, use.kernel = synthesis.grow_ridges, synthesis.gabor_kernel
    passes, elongation, *breaks = VARIANTS[variant]
    synthesis.use_device("cpu")
    synthesis.gabor_kernel = elongated(elongation) if elongation != 1.0 else use.kernel

    def grow(theta, period, rng, iterations=14, **kw):
        ridges = use.grow(theta, period, rng, passes if iterations == 14 else iterations, **kw)
        # Breaks on masters only (an extended master holds the master's ridges, `fixed`).
        if breaks and "fixed" not in kw:
            ridges = plant_breaks(ridges, theta, period, rng, *breaks)
        return ridges

    fingers.grow_ridges = grow
    if getattr(use, "variant", None) != variant:
        fingers.master.cache_clear()
        fingers.extended.cache_clear()
        use.variant = variant


def one(task):
    variant, seed, fgp = task
    from ridgegen import finger as fingers

    use(variant)
    m = fingers.master(seed, fgp)
    h, w = m.ridges.shape
    # The rolled contact's reach on the master (`fingerprints.rolled_capture`): half axes 0.43 and 0.525 of the
    # print's frame, its centre 0.2 of the height below the master's centre at 0.47, at the print's scale.
    from ridgegen import fingerprints

    fw, fh = fingerprints.ROLLED_SIZE
    scale = fingers.print_scale(seed, fgp, m.scale)
    rx, ry = 0.43 * fw / scale, 0.525 * fh / scale
    cx, cy = fingers.MASTER_W / 2, fingers.MASTER_H * 0.47 + 0.2 * ry
    ms = [p for p in m.minutiae if ((p["x"] - cx) / rx) ** 2 + ((p["y"] - cy) / ry) ** 2 < 1]
    row = {"variant": variant, "seed": seed, "fgp": fgp, "pattern": m.pattern, "period": round(m.period, 2),
           "count": len(ms), "all": len(m.minutiae),
           "bifurcations": float(np.mean([p["type"] == "bifurcation" for p in ms])) if ms else None}
    if m.cores:
        top = min(c[1] for c in m.cores)
        row["above"] = sum(1 for p in ms if p["y"] < top)
        row["below"] = sum(1 for p in ms if p["y"] >= top)
    return row


def main(out, masters=40):
    seeds = range(4000, 4000 + (masters + len(FINGERS) - 1) // len(FINGERS))
    done = set()
    if os.path.exists(out):
        done = {(r["variant"], r["seed"], r["fgp"]) for r in map(json.loads, open(out))}
    tasks = [(v, s, f) for v in RUN for s in seeds for f in FINGERS if (v, s, f) not in done]
    with ProcessPoolExecutor(JOBS) as pool, open(out, "a") as f:
        for row in pool.map(one, tasks):
            f.write(json.dumps(row) + "\n")
            f.flush()
    rows = [json.loads(line) for line in open(out)]
    for v in RUN:
        x = [r for r in rows if r["variant"] == v]
        ratio = [r["below"] / r["above"] for r in x if r.get("above")]
        print(json.dumps({"variant": v, "masters": len(x), "count_median": float(np.median([r["count"] for r in x])),
                          "bifurcations": float(np.median([r["bifurcations"] for r in x if r["bifurcations"] is not None])),
                          "below_over_above": float(np.median(ratio)) if ratio else None}))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
