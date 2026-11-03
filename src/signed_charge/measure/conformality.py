"""Curvature and conformality of the ridge flow, real against synthetic (step 2): conformality.py OUT

Imdahl, Gottschlich, Huckemann & Ohshika 2017 (arXiv 1708.02158, "Möbius moduli for fingerprint orientation
fields"): real orientation fields deviate from conformality where they curve most, above cores and above and below
whorls, and analytic models (zero-pole, quadratic differentials, SFinGe's) don't; a linear classifier on the 2D
histogram of curvature and conformality index tells FVC's real prints from its SFinGe ones (DB4) at 70-80 % from
one impression. Here, for every 4th pixel q0 of a print's foreground at 500 ppi:

- A tetraquadrilateral (§4): from q0 along the flow and across it, `C` px each way (q1, q3 along; q2, q4 across);
  from those four ends, along and across the flow until the curves meet, the corners p1-p4. None where a curve
  leaves the foreground, two don't meet, or a corner is degenerate (nearer than C / 2 or further than 2C from its
  cross's ends; near singular points).
- Its Möbius modulus (eq. 6), M = (p1-q2)/(q2-p2) · (p2-q3)/(q3-p3) · (p3-q4)/(q4-p4) · (p4-q1)/(q1-p1), 1 for
  a conformal field, and the conformality index log |M|.
- Curvature: how far the flow turns along the cross's arm through q0, q1 to q3 (2C px), radians. (Imdahl et al.
  use Gottschlich 2012's curved regions; this is the same idea, not necessarily its exact measure.)

Each print's 2D histogram (conformality index in [-0.33, 0.33], curvature in [0, 2.6], the outer bins holding what's
beyond, as their Fig. 8), then a linear classifier (logistic regression, L2) real against each synthetic set, from
one impression, cross-validated by finger (no finger in both training and test). The SFinGe sets check this
implementation against the paper's 70-80 %; Phantom's should be near 50 % to be realistic.

Sets: FVC 2000, 2002, 2004 DB1-3 B (real) and DB4 B (SFinGe), 10 fingers × 8 each
(see DATA.md); Phantom's rendered slap fingertips
(`impressions-slaps-value-03`, ridgegen 0.25, cut and turned upright as `field_similarity.py` does);
with `--procedural MODEL PEOPLE`, fingertips inked flat from master model MODEL, without the GPU.

Run: python -m signed_charge.measure.conformality OUT.json [PEOPLE] [--binarise]
[--procedural 2 40]
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.measure import field_similarity as fs
from signed_charge import paths

FVC = str(paths.FVC)
PPI = {"FVC2002/DB2_B": 569, "FVC2004/DB3_B": 512}
C = 40  # px at 500 ppi, about 3 ridge periods (§4)
STRIDE = 4
SIGMA = 12  # px: how widely the orientation's gradients are averaged
INDEX_MAX, CURVATURE_MAX = 0.33, 2.6
JOBS = 8
# With `--binarise`, every print's field is estimated from NBIS mindtct's binarised image instead of its grey one, so
# sensors' and renderers' texture and noise (which move the index where the flow is straight) count less.
BINARISE = False
NBIS = paths.NBIS


def binarised(im):
    """mindtct's binarised image (ridges black, 0 and 255)."""
    with tempfile.TemporaryDirectory() as tmp:
        h, w = im.shape
        np.ascontiguousarray(im, dtype=np.uint8).tofile(f"{tmp}/i.raw")
        subprocess.run([NBIS / "cjpegl", "jpl", f"{tmp}/i.raw", "-raw_in", f"{w},{h},8,500"], check=True, capture_output=True)
        subprocess.run([NBIS / "mindtct", "-b", "-m1", f"{tmp}/i.jpl", f"{tmp}/b"], check=True, capture_output=True)
        b = np.fromfile(f"{tmp}/b.brw", np.uint8).reshape(h, w)
    return np.where(b > 0, 255, 0).astype(np.uint8) if b.max() <= 1 else b


def orientation(im, fg):
    """The flow's direction as unit doubled-angle components (cos 2θ, sin 2θ), θ along the ridges."""
    g = im.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    c = cv2.GaussianBlur(gx * gx - gy * gy, (0, 0), SIGMA)
    s = cv2.GaussianBlur(2 * gx * gy, (0, 0), SIGMA)
    # The gradient's doubled angle; the ridges run across it (θ + 90°: doubled, + 180°).
    norm = np.hypot(c, s) + 1e-6
    return -c / norm, -s / norm


def sample(c2, s2, p):
    """Bilinear (cos 2θ, sin 2θ) at points p (N, 2) as x, y; returns the direction (N, 2) along θ."""
    h, w = c2.shape
    x, y = np.clip(p[:, 0], 0, w - 1.001), np.clip(p[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0

    def at(a):
        return a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy) + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy

    theta = 0.5 * np.arctan2(at(s2), at(c2))
    return np.stack([np.cos(theta), np.sin(theta)], 1)


def trace(c2, s2, inside, start, direction, steps, across):
    """Follow the flow (or, `across`, its orthogonal) from `start` (N, 2), first along `direction` (N, 2), 1 px a
    step. Returns the path (N, steps + 1, 2), whether it has stayed inside up to each step (N, steps + 1), and its
    total turning (radians)."""
    h, w = inside.shape
    p = start.astype(np.float64).copy()
    prev = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9)
    path = [p.copy()]
    ok = np.ones(len(p), bool)
    oks = [ok.copy()]
    turning = np.zeros(len(p))
    for _ in range(steps):
        d = sample(c2, s2, p)
        if across:
            d = np.stack([-d[:, 1], d[:, 0]], 1)
        d *= np.where((d * prev).sum(1) < 0, -1, 1)[:, None]
        turning += np.arccos(np.clip((d * prev).sum(1), -1, 1))
        prev = d
        p = p + d
        xi, yi = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        ok &= (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        ok &= inside[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
        path.append(p.copy())
        oks.append(ok.copy())
    return np.stack(path, 1), np.stack(oks, 1), turning


def meet(a, b, oka, okb, tol=1.5):
    """Where paths a and b (N, L, 2) meet: the midpoint of their closest points, and whether they come within `tol`
    px, both inside up to there (`oka`, `okb`, as `trace` returns them)."""
    out, ok = np.zeros((len(a), 2)), np.zeros(len(a), bool)
    for i in range(0, len(a), 256):
        d = np.linalg.norm(a[i : i + 256, :, None, :] - b[i : i + 256, None, :, :], axis=3)
        flat = d.reshape(len(d), -1).argmin(1)
        ia, ib = np.unravel_index(flat, d.shape[1:])
        rows = np.arange(len(d))
        out[i : i + 256] = (a[i + rows, ia] + b[i + rows, ib]) / 2
        ok[i : i + 256] = (d[rows, ia, ib] < tol) & oka[i + rows, ia] & okb[i + rows, ib]
    return out, ok


def moduli(im, fg):
    """Conformality index and curvature at every `STRIDE`th foreground pixel where a tetraquadrilateral exists."""
    c2, s2 = orientation(im, fg)
    inside = cv2.erode(fg.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ys, xs = np.nonzero(inside[::STRIDE, ::STRIDE])
    q0 = np.stack([xs, ys], 1).astype(np.float64) * STRIDE
    if len(q0) == 0:
        return np.zeros(0), np.zeros(0)
    t = sample(c2, s2, q0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    along_f, ok1, turn_f = trace(c2, s2, inside, q0, t, C, False)
    along_b, ok2, turn_b = trace(c2, s2, inside, q0, -t, C, False)
    across_f, ok3, _ = trace(c2, s2, inside, q0, n, C, True)
    across_b, ok4, _ = trace(c2, s2, inside, q0, -n, C, True)
    q1, q3 = along_b[:, -1], along_f[:, -1]
    q2, q4 = across_b[:, -1], across_f[:, -1]
    # Cyclic order q1, q2, q3, q4 one way round for every TQL (counterclockwise on the image).
    a, b = q3 - q1, q4 - q2
    flip = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 0
    q2, q4 = np.where(flip[:, None], q4, q2), np.where(flip[:, None], q2, q4)
    ok = ok1[:, -1] & ok2[:, -1] & ok3[:, -1] & ok4[:, -1]

    def corner(qa, qb):
        """The corner between qa (on the flow through q0: its edge runs across the flow) and qb (across: its edge
        runs along the flow): across the flow from qa towards qb, along it from qb towards qa."""
        a, oka, _ = trace(c2, s2, inside, qa, qb - q0, 2 * C, True)
        b, okb, _ = trace(c2, s2, inside, qb, qa - q0, 2 * C, False)
        p, ok = meet(a, b, oka, okb)
        # Not degenerate (§3): the corner between C / 2 and 2C from both its cross's ends, as a TQL's corners are
        # where the flow is smooth (C).
        for q in (qa, qb):
            d = np.linalg.norm(p - q, axis=1)
            ok &= (d > C / 2) & (d < 2 * C)
        return p, ok

    p1, k1 = corner(q1, q2)
    p2, k2 = corner(q3, q2)
    p3, k3 = corner(q3, q4)
    p4, k4 = corner(q1, q4)
    ok &= k1 & k2 & k3 & k4
    z = lambda v: v[:, 0] + 1j * v[:, 1]  # noqa: E731
    P1, P2, P3, P4, Q1, Q2, Q3, Q4 = map(z, (p1, p2, p3, p4, q1, q2, q3, q4))
    with np.errstate(divide="ignore", invalid="ignore"):
        m = ((P1 - Q2) / (Q2 - P2)) * ((P2 - Q3) / (Q3 - P3)) * ((P3 - Q4) / (Q4 - P4)) * ((P4 - Q1) / (Q1 - P1))
        index = np.log(np.abs(m))
    ok &= np.isfinite(index)
    return index[ok], (turn_f + turn_b)[ok]


def histogram(index, curvature, bins):
    i = np.clip(((np.clip(index, -INDEX_MAX, INDEX_MAX) + INDEX_MAX) / (2 * INDEX_MAX) * bins).astype(int), 0, bins - 1)
    k = np.clip((np.clip(curvature, 0, CURVATURE_MAX) / CURVATURE_MAX * bins).astype(int), 0, bins - 1)
    h = np.zeros((bins, bins))
    np.add.at(h, (k, i), 1)
    return h / max(1, h.sum())


def describe(item):
    key, im = item
    fg = fs.foreground(im)
    if fg is None:
        return key, None
    index, curvature = moduli(binarised(im) if BINARISE else im, fg)
    if len(index) < 200:
        return key, None
    return key, {
        "n": int(len(index)),
        "h10": histogram(index, curvature, 10).ravel().tolist(),
        "h20": histogram(index, curvature, 20).ravel().tolist(),
        "index_sd": float(np.std(index)),
        "index_p99": float(np.percentile(np.abs(index), 99)),
        "extreme_at_high_curvature": float(np.mean((np.abs(index) > INDEX_MAX)[curvature > 1.3])) if np.any(curvature > 1.3) else None,
    }


def fvc(dbs):
    for year in ("FVC2000", "FVC2002", "FVC2004"):
        for db in dbs:
            name = f"{year}/{db}_B"
            for f in sorted(glob.glob(f"{FVC}/{name}/*.tif")):
                finger, impression = os.path.basename(f)[:-4].split("_")
                yield (name, f"{name}/{finger}", int(impression)), fs.at_500(cv2.imread(f, 0), PPI.get(name, 500))


def phantom(people):
    fs.PHANTOM = str(paths.RENDERED)
    for key, capture, im in fs.phantom(people):
        yield ("phantom-0.25", key, capture), im


def procedural(model, people):
    for key, capture, im in fs.procedural(model, people, 3000):
        yield (f"procedural-model{model}", key, capture), im


def logistic(x, y, l2=1.0, steps=3000, rate=0.5):
    """L2 logistic regression by gradient descent, standardised features; returns a predictor."""
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = (x - mu) / sd
    w, b = np.zeros(z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        g = p - y
        w -= rate * (z.T @ g / len(y) + l2 * w / len(y))
        b -= rate * g.mean()
    return lambda xs: ((xs - mu) / sd) @ w + b > 0


def accuracy(real, synthetic, feature, folds=5, seed=0):
    """Real against synthetic from one impression, cross-validated by finger."""
    rows = [(k[1], 1, v[feature]) for k, v in real] + [(k[1], 0, v[feature]) for k, v in synthetic]
    fingers = sorted({r[0] for r in rows})
    rng = np.random.default_rng(seed)
    fold = dict(zip(fingers, rng.permutation(len(fingers)) % folds))
    correct = []
    for f in range(folds):
        train = [r for r in rows if fold[r[0]] != f]
        test = [r for r in rows if fold[r[0]] == f]
        # Balance the classes in training.
        n = min(sum(r[1] for r in train), sum(1 - r[1] for r in train))
        pos = [r for r in train if r[1]]
        neg = [r for r in train if not r[1]]
        neg = [neg[i] for i in rng.permutation(len(neg))[:n]]
        pos = [pos[i] for i in rng.permutation(len(pos))[:n]]
        tr = pos + neg
        predict = logistic(np.array([r[2] for r in tr]), np.array([r[1] for r in tr], float))
        for label in (0, 1):
            group = [r for r in test if r[1] == label]
            if group:
                correct.append(np.mean(predict(np.array([r[2] for r in group])) == label))
    return float(np.mean(correct))  # balanced: the mean of each class's accuracy per fold


def main(out, args):
    global BINARISE
    BINARISE = "--binarise" in args
    args = [a for a in args if a != "--binarise"]
    sources = list(fvc(("DB1", "DB2", "DB3"))) + list(fvc(("DB4",)))
    if "--procedural" in args:
        model, people = (int(a) for a in args[args.index("--procedural") + 1 :][:2])
        sources += list(procedural(model, people))
    else:
        sources += list(phantom(int(args[0]) if args else 60))
    with ProcessPoolExecutor(JOBS) as pool:
        described = [(k, v) for k, v in pool.map(describe, sources, chunksize=4) if v]
    sets = {}
    for k, v in described:
        sets.setdefault("real" if "DB4" not in k[0] and k[0].startswith("FVC") else k[0], []).append((k, v))
    sfinge = [x for name, xs in sets.items() if "DB4" in name for x in xs]
    real = sets["real"]
    report = {"prints": {name: len(xs) for name, xs in sets.items()}}
    for name, xs in [("sfinge", sfinge)] + [(n, xs) for n, xs in sets.items() if not n.startswith("FVC") and n != "real"]:
        report[name] = {f"accuracy_{f}": accuracy(real, xs, f) for f in ("h10", "h20")}
    for name, xs in [("real", real), ("sfinge", sfinge)] + [(n, xs) for n, xs in sets.items() if not n.startswith("FVC") and n != "real"]:
        e = [v["extreme_at_high_curvature"] for _, v in xs if v["extreme_at_high_curvature"] is not None]
        report.setdefault(name, {}).update(
            {
                "index_sd_median": float(np.median([v["index_sd"] for _, v in xs])),
                "index_p99_median": float(np.median([v["index_p99"] for _, v in xs])),
                "extreme_at_high_curvature_median": float(np.median(e)) if e else None,
                "mean_h10": np.mean([v["h10"] for _, v in xs], 0).round(4).tolist(),
            }
        )
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "mean_h10"} if isinstance(v, dict) else v for k, v in report.items()}, indent=1))
    json.dump({"report": report, "prints": [[list(k), v] for k, v in described]}, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
