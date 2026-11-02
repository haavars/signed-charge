"""How alike two people's ridge flow is around the core, real (FVC) against Phantom: field_similarity.py OUT

Each print, at 500 ppi: its contact (local ridge contrast), its orientation
field (smoothed doubled-angle gradients) and core (Poincaré index; prints with one core kept, loops mostly), then
the orientation sampled on a polar grid around the core, 0.5-5 mm out. Two prints' distance is the mean of
(1 - cos 2Δθ) / 2 over the samples both have, the smallest over turns of ±20° and a mirror (so a left loop meets
a right one fairly); 0 is the same flow, 0.5 unrelated. Same finger: two impressions (FVC) or two captures
(Phantom); different fingers: different people (Phantom) or fingers (FVC, whose B sets are a few people's fingers).
Also each print's ridge spacing near the core (mm), from the local spectrum.

Real: the FVC B sets (see DATA.md). Phantom: the
`impressions-slaps-value-02` slap fingertips, cut along `tip_polygon`, turned upright by `angle`, 1000 to 500 ppi.

Or, with `--procedural MODEL [PEOPLE] [FIRST_SEED]`, Phantom's plain fingertips drawn procedurally from master model
MODEL (1, ridgegen to 0.24; 2 since 0.25), two captures each, without the GPU: their ridge map inked flat, the
phalanges left out, at 500 ppi.

Run from python_biometrics: .venv/bin/python -I python -m signed_charge.measure.field_similarity OUT
or python -m signed_charge.measure.field_similarity OUT --procedural 2 60 3000
"""

import glob
import itertools
import json
import os
import sys

import cv2
import numpy as np
from signed_charge import paths

FVC = str(paths.FVC)
FVC_DBS = {"FVC2004/DB1_B": 500, "FVC2004/DB2_B": 500, "FVC2002/DB1_B": 500, "FVC2002/DB2_B": 569, "FVC2000/DB3_B": 500}
PHANTOM = str(paths.RENDERED_02)
MM = 500 / 25.4
B = 8
RADII = np.arange(0.5, 5.01, 0.5) * MM
ANGLES = np.deg2rad(np.arange(0, 360, 15))
TURNS = np.deg2rad(np.arange(-20, 21, 5))


def foreground(im):
    g = im.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 12)
    var = cv2.GaussianBlur(g * g, (0, 0), 8)
    var = np.clip(var / var.max() * 255, 0, 255).astype(np.uint8)
    t = cv2.threshold(var, 0, 255, cv2.THRESH_OTSU)[0]
    m = (var > t * 0.5).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def field(im):
    g = im.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gxx, gyy, gxy = [cv2.boxFilter(a, -1, (2 * B, 2 * B)) for a in (gx * gx, gy * gy, gx * gy)]
    c, s = cv2.GaussianBlur(gxx - gyy, (0, 0), 3 * B), cv2.GaussianBlur(2 * gxy, (0, 0), 3 * B)
    # Ridge direction: perpendicular to the gradient.
    return np.mod(0.5 * np.arctan2(s, c) + np.pi / 2, np.pi)


def cores(theta, fg):
    th = theta[B // 2 :: B, B // 2 :: B]
    f = cv2.erode(fg.astype(np.uint8), np.ones((5 * B, 5 * B), np.uint8))[B // 2 :: B, B // 2 :: B].astype(bool)

    def d(a, b):
        return (b - a + np.pi / 2) % np.pi - np.pi / 2

    a, b, c, e = th[:-1, :-1], th[:-1, 1:], th[1:, 1:], th[1:, :-1]
    idx = (d(a, b) + d(b, c) + d(c, e) + d(e, a)) / (2 * np.pi)
    ok = f[:-1, :-1] & f[1:, 1:] & f[:-1, 1:] & f[1:, :-1]
    ys, xs = np.nonzero((np.abs(idx) > 0.4) & ok & (idx > 0))
    merged = []
    for x, y in zip((xs + 1) * B, (ys + 1) * B):
        for q in merged:
            if np.hypot(x - q[0] / q[2], y - q[1] / q[2]) < 3 * B:
                q[0] += x
                q[1] += y
                q[2] += 1
                break
        else:
            merged.append([x, y, 1])
    return [(q[0] / q[2], q[1] / q[2]) for q in merged]


def spacing_mm(im, core):
    """Ridge spacing near the core: the peak of the local spectrum in a 64 px window, mm."""
    x, y = int(core[0]), int(core[1])
    w = im[max(0, y - 32) : y + 32, max(0, x - 32) : x + 32].astype(np.float32)
    if w.shape != (64, 64):
        return None
    w = (w - w.mean()) * np.outer(np.hanning(64), np.hanning(64))
    p = np.abs(np.fft.fftshift(np.fft.fft2(w)))
    yy, xx = np.mgrid[-32:32, -32:32]
    r = np.hypot(xx, yy)
    p[(r < 3) | (r > 16)] = 0
    k = r.flat[int(np.argmax(p))]
    return float(64 / k / MM) if k else None


def describe(im):
    fg = foreground(im)
    if fg is None:
        return None
    theta = field(im)
    cs = cores(theta, fg)
    if len(cs) != 1:
        return {"cores": len(cs)}
    cx, cy = cs[0]
    inner = cv2.erode(fg.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    return {"cores": 1, "core": (cx, cy), "theta": theta, "inner": inner, "spacing": spacing_mm(im, (cx, cy))}


def samples(d, turn=0.0, mirror=False):
    """The orientation (doubled angle, unit vectors) on the polar grid, turned by `turn` and mirrored."""
    cx, cy = d["core"]
    ang = ANGLES + turn
    if mirror:
        ang = np.pi - ang
    xs = cx + RADII[:, None] * np.cos(ang)[None, :]
    ys = cy + RADII[:, None] * np.sin(ang)[None, :]
    h, w = d["theta"].shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xi, yi = np.clip(xs, 0, w - 1).astype(int), np.clip(ys, 0, h - 1).astype(int)
    ok &= d["inner"][yi, xi]
    th = d["theta"][yi, xi] - turn
    if mirror:
        th = np.pi - th
    return np.cos(2 * th), np.sin(2 * th), ok


def distance(a, b):
    ca, sa, oka = samples(a)
    best = None
    for turn, mirror in itertools.product(TURNS, (False, True)):
        cb, sb, okb = samples(b, turn, mirror)
        ok = oka & okb
        if ok.sum() < 0.5 * ok.size:
            continue
        dist = float(np.mean((1 - (ca * cb + sa * sb)[ok]) / 2))
        best = dist if best is None else min(best, dist)
    return best


def at_500(im, ppi):
    if ppi == 500:
        return im
    return cv2.resize(im, None, fx=500 / ppi, fy=500 / ppi, interpolation=cv2.INTER_AREA)


def fvc():
    for db, ppi in FVC_DBS.items():
        for f in sorted(glob.glob(f"{FVC}/{db}/*.tif")):
            finger, impression = os.path.basename(f)[:-4].split("_")
            if impression in ("1", "2"):
                yield f"{db}/{finger}", int(impression), at_500(cv2.imread(f, 0), ppi)


def phantom(people=60):
    for person in sorted(glob.glob(f"{PHANTOM}/PH-*"))[:people]:
        for slap in ("fgp13_Right_four", "fgp14_Left_four"):
            for capture, suffix in ((1, ""), (2, "_c2")):
                png = f"{person}/fingerprints/slaps/{slap}{suffix}.png"
                truth = json.load(open(f"{person}/ground_truth/{slap}{suffix}.json"))
                im = cv2.imread(png, 0)
                for fing in truth["fingers"]:
                    if "tip_polygon" not in fing:
                        continue
                    yield (f"{os.path.basename(person)}/{fing['fgp']}", capture, at_500(upright(im, fing), 1000))


def procedural(model, people, first):
    import functools

    from ridgegen import finger as fingers
    from ridgegen import fingerprints, synthesis
    from ridgegen.synthesis import rng_for

    synthesis.use_device("cpu")
    fingers.master = functools.partial(fingers.master, model=model)
    for seed in range(first, first + people):
        for fgp in (2, 3, 4, 5, 7, 8, 9, 10):
            for capture in (1, 2):
                rng = rng_for(seed, fgp, fingerprints.PLAIN, capture - 1, 13 if fgp <= 5 else 14)
                _, ridges, contact, _, _, pasted = fingerprints.plain_finger(seed, fgp, capture - 1, rng)
                tip = (contact > 0.5) & ~pasted
                im = np.where(tip & (ridges > 0), 40, 235).astype(np.uint8)
                yield f"PH{seed}-{model}/{fgp}", capture, im


def upright(im, fing):
    poly = np.array(fing["tip_polygon"], np.float32)
    cx, cy = poly.mean(0)
    w = np.hypot(*(poly[1] - poly[0]))
    h = np.hypot(*(poly[3] - poly[0]))
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), fing["angle"], 1.0)
    m[:, 2] += (w / 2 + 40 - cx, h / 2 + 40 - cy)
    return cv2.warpAffine(im, m, (int(w + 80), int(h + 80)), borderValue=255)


def run(source):
    described = {}
    counts = {"prints": 0, "one_core": 0}
    for key, impression, im in source:
        counts["prints"] += 1
        d = describe(im)
        if d and d.get("cores") == 1:
            counts["one_core"] += 1
            described[(key, impression)] = d
    fingers = sorted({k for k, _ in described})
    same = [distance(described[(k, 1)], described[(k, 2)]) for k in fingers if (k, 1) in described and (k, 2) in described]
    firsts = [k for k in fingers if (k, 1) in described]
    between = []
    for a, b in itertools.combinations(firsts, 2):
        if a.split("/")[0] == b.split("/")[0] and a.startswith("PH"):
            continue  # Phantom: different people only
        dist = distance(described[(a, 1)], described[(b, 1)])
        if dist is not None:
            between.append(dist)
    spacing = [d["spacing"] for (k, i), d in described.items() if i == 1 and d["spacing"]]
    return counts, [x for x in same if x is not None], between, spacing


def summary(name, counts, same, between, spacing):
    q = lambda xs, p: float(np.percentile(xs, p)) if len(xs) else None
    return {
        "set": name,
        **counts,
        "same_n": len(same),
        "same_median": q(same, 50),
        "same_p90": q(same, 90),
        "between_n": len(between),
        "between_p1": q(between, 1),
        "between_p5": q(between, 5),
        "between_median": q(between, 50),
        "spacing_median_mm": q(spacing, 50),
        "spacing_sd_mm": float(np.std(spacing)) if spacing else None,
        "spacing_n": len(spacing),
    }


if __name__ == "__main__":
    out = sys.argv[1]
    results = {}
    if "--procedural" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--procedural") + 1 :]]
        model, people, first = (args + [60, 3000])[:3] if len(args) >= 1 else (2, 60, 3000)
        mine = procedural(model, people, first)
    else:
        mine = phantom()
    for name, source in (("fvc", fvc()), ("phantom", mine)):
        counts, same, between, spacing = run(source)
        results[name] = {"same": same, "between": between, "spacing": spacing}
        print(json.dumps(summary(name, counts, same, between, spacing)), flush=True)
    real = np.array(results["fvc"]["between"])
    for name in ("fvc", "phantom"):
        b = np.array(results[name]["between"])
        print(
            json.dumps(
                {
                    "set": name,
                    "between_below_fvc_p1": float(np.mean(b < np.percentile(real, 1))),
                    "between_below_fvc_p5": float(np.mean(b < np.percentile(real, 5))),
                }
            )
        )
    json.dump(results, open(out, "w"))
