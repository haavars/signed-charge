"""Editing a master's minutiae as spiral phase (step 3, route b): phase_edit.py OUT [MASTERS]

A ridge pattern is b cos ψ, and every minutia is a spiral in ψ, + or − (Larkin & Fletcher 2007). Moving,
adding or removing minutiae two at a time, + and − together, changes ψ by a dipole, p·arg((z − z1)/(z − z2)),
whose effect falls off as 1/r: ridge counts far off stay as they were, and so do the cores and deltas. So a
master's grown pattern can carry the minutiae a point process chooses: remove the growth's own in pairs, add
the chosen ones in pairs, move single ones. This tests the operations on master model 2's masters before a point
process is built on them:

- **ψ of a grown master**: the spiral-phase (vortex) quadrature transform V of the ridges (Larkin, Bone &
  Oldfield 2001) is i e^{iβ} b sin ψ, β the direction of ψ's gradient; β is taken from V² up to π, and the half
  turn chosen to match the edit's centre, so each edit works in one convention.
- **Add a pair**: + and − one ridge period apart, along the ridge, across it, or diagonal. How many minutiae the
  extractor (`minutiae.extract`, the masters' ground truth) then finds new, of what types, and whether they face
  each other.
- **Remove a grown pair**: two of the master's minutiae under 20 px apart, of opposite spiral (their winding in
  ψ), cancelled. How many of the two go, and how many new ones appear.

Run: python -m signed_charge.generate.phase_edit OUT.json
[MASTERS]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.generate import minutiae_growth as growth

WINDOW = 6  # ridge periods: an edit's reach, past which its phase is tapered off
SATURATION = 4.0  # the grown ridges' tanh: tanh(a cos ψ) against them
EDITS = 12  # of each kind, per master
EDGE = 60  # px
JOBS = 8


def vortex(ridges):
    """The spiral-phase quadrature transform of `ridges` (its mean removed)."""
    h, w = ridges.shape
    u, v = np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h))
    spiral = np.exp(1j * np.arctan2(v, u))
    spiral[0, 0] = 0
    return np.fft.ifft2(np.fft.fft2(ridges - ridges.mean()) * spiral)


def doubled_direction(v, sigma=3):
    """e^{2iβ} from V (= i e^{iβ} b sin ψ: −V² = e^{2iβ} b² sin² ψ), averaged so it holds at sin ψ = 0 too."""
    d = -v**2
    re = cv2.GaussianBlur(d.real.astype(np.float32), (0, 0), sigma)
    im = cv2.GaussianBlur(d.imag.astype(np.float32), (0, 0), sigma)
    return np.arctan2(im, re)


def local_phase(ridges, v, beta2, centre, half):
    """ψ about `centre` (x, y) within `half` px, in the convention whose β is nearest the centre's; and the
    window's slices."""
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    h, w = ridges.shape
    ys, xs = slice(max(0, cy - half), min(h, cy + half + 1)), slice(max(0, cx - half), min(w, cx + half + 1))
    b2 = beta2[ys, xs]
    beta0 = beta2[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)] / 2
    beta = b2 / 2
    flip = np.cos(beta - beta0) < 0
    beta = np.where(flip, beta + np.pi, beta)
    q = np.real(-1j * np.exp(-1j * beta) * v[ys, xs])
    return np.arctan2(q, ridges[ys, xs]), (ys, xs)


def residues(psi):
    """ψ's winding round each 2 × 2 plaquette (Goldstein's residues): +1 or −1 at a spiral, at the plaquette's
    centre (x + 0.5, y + 0.5)."""
    def wrap(d):
        return (d + np.pi) % (2 * np.pi) - np.pi

    a, b, c, d = psi[:-1, :-1], psi[:-1, 1:], psi[1:, 1:], psi[1:, :-1]
    return np.round((wrap(b - a) + wrap(c - b) + wrap(d - c) + wrap(a - d)) / (2 * np.pi)).astype(int)


def spiral_at(res, x, y, reach):
    """The sign and place of the spiral nearest (x, y) within `reach` px (the minutia's vortex: at an ending's tip
    or a fork's join, not always on its skeleton pixel), or (0, None)."""
    ys, xs = np.nonzero(res)
    if not len(xs):
        return 0, None
    d = np.hypot(xs + 0.5 - x, ys + 0.5 - y)
    i = int(np.argmin(d))
    return (int(res[ys[i], xs[i]]), (xs[i] + 0.5, ys[i] + 0.5)) if d[i] <= reach else (0, None)


def reach(z_plus, z_minus, period):
    """An edit's half window, px: `WINDOW` periods, or three times the pair's spacing if that's more."""
    return int(max(WINDOW * period, 3 * np.hypot(*(np.asarray(z_plus) - np.asarray(z_minus)))))


def apply_dipole(ridges, v, beta2, z_plus, z_minus, period, half=None):
    """The ridges with a + spiral at `z_plus` and a − at `z_minus` (x, y) added to ψ, in the convention of their
    midpoint (a − at a grown + cancels it), within `half` px (default `reach`)."""
    mid = (np.asarray(z_plus) + np.asarray(z_minus)) / 2
    half = half or reach(z_plus, z_minus, period)
    psi, (ys, xs) = local_phase(ridges, v, beta2, mid, half)
    yy, xx = np.mgrid[ys, xs].astype(np.float64)
    z = xx + 1j * yy
    zp, zm = complex(*z_plus), complex(*z_minus)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.angle((z - zp) / (z - zm))
    delta = np.nan_to_num(delta)
    r = np.abs(z - complex(*mid)) / half
    taper = np.clip(1.5 - 1.5 * r, 0, 1) ** 2 * (3 - 2 * np.clip(1.5 - 1.5 * r, 0, 1))  # smoothstep, off at r = 1
    edited = ridges.copy()
    # Re-drawn from ψ within the window, blended into the grown ridges towards its edge (re-drawing unedited ψ
    # gives the grown ridges back closely; adding only the difference left the growth's profile errors as smears).
    edited[ys, xs] = (1 - taper) * ridges[ys, xs] + taper * np.tanh(SATURATION * np.cos(psi + delta))
    return np.clip(edited, -1, 1)


def found(ridges):
    from ridgegen import minutiae

    return minutiae.extract(ridges > 0, np.ones(ridges.shape, bool))


def changes(before, after, near, radius=40):
    """Minutiae within `radius` of `near` that are new in `after` and gone from `before` (3 px tolerance)."""
    def close(p, q):
        return (p["x"] - q["x"]) ** 2 + (p["y"] - q["y"]) ** 2 <= 9 and p["type"] == q["type"]

    def local(ms):
        return [p for p in ms if (p["x"] - near[0]) ** 2 + (p["y"] - near[1]) ** 2 < radius**2]

    b, a = local(before), local(after)
    return [p for p in a if not any(close(p, q) for q in b)], [q for q in b if not any(close(p, q) for p in a)]


def one(task):
    seed, fgp = task
    from ridgegen import finger as fingers

    growth.use("base")
    m = fingers.master(seed, fgp)
    ridges, period = m.ridges.astype(np.float64), m.period
    v = vortex(ridges)
    beta2 = doubled_direction(v)
    base = found(ridges)
    h, w = ridges.shape
    rng = np.random.default_rng(seed * 100 + fgp)
    rows = []

    # Add pairs, one at a time on the unedited master, away from its own minutiae and singular points.
    tries = 0
    while sum(r["op"] == "add" for r in rows) < 3 * EDITS and tries < 400:
        tries += 1
        x, y = rng.uniform(120, w - 120), rng.uniform(120, h - 120)
        if any((p["x"] - x) ** 2 + (p["y"] - y) ** 2 < (2.5 * period) ** 2 for p in base):
            continue
        if any((c[0] - x) ** 2 + (c[1] - y) ** 2 < (5 * period) ** 2 for c in m.cores + m.deltas):
            continue
        beta = beta2[int(y), int(x)] / 2  # across the ridges
        layout = ("along", "across", "diagonal")[sum(r["op"] == "add" for r in rows) % 3]
        a = {"along": beta + np.pi / 2, "across": beta, "diagonal": beta + np.pi / 4}[layout]
        d = period * np.array([np.cos(a), np.sin(a)]) / 2
        edited = apply_dipole(ridges, v, beta2, (x, y) + d, (x, y) - d, period)
        new, gone = changes(base, found(edited), (x, y))
        angle = None
        if len(new) == 2:
            angle = abs((new[0]["angle"] - new[1]["angle"] + 180) % 360 - 180)
        rows.append({"op": "add", "layout": layout, "new": len(new), "gone": len(gone),
                     "types": sorted(p["type"][0] for p in new), "angle": angle})

    # Remove grown pairs: two minutiae under 20 px apart whose windings are opposite.
    # Not in the master's edge band (half its minutiae, where the growth met the edge; no print shows it).
    inner = [p for p in base if min(p["x"], p["y"], w - p["x"], h - p["y"]) > EDGE]
    pairs = []
    for i, p in enumerate(inner):
        for q in inner[i + 1:]:
            if (p["x"] - q["x"]) ** 2 + (p["y"] - q["y"]) ** 2 < 400:
                pairs.append((p, q))
    rng.shuffle(pairs)
    for p, q in pairs[:EDITS * 2]:
        mid = ((p["x"] + q["x"]) / 2, (p["y"] + q["y"]) / 2)
        psi, (ys, xs) = local_phase(ridges, v, beta2, mid, int(WINDOW * period))
        res = residues(psi)
        wp, zp = spiral_at(res, p["x"] - xs.start, p["y"] - ys.start, 0.6 * period)
        wq, zq = spiral_at(res, q["x"] - xs.start, q["y"] - ys.start, 0.6 * period)
        angle = abs((p["angle"] - q["angle"] + 180) % 360 - 180)
        row = {"op": "remove", "windings": [wp, wq], "angle": angle, "types": sorted(t["type"][0] for t in (p, q))}
        if wp and wq and wp == -wq and zp != zq:
            at = {id(p): (zp[0] + xs.start, zp[1] + ys.start), id(q): (zq[0] + xs.start, zq[1] + ys.start)}
            plus, minus = (p, q) if wp < 0 else (q, p)  # adding the opposite spiral at each cancels it
            edited = apply_dipole(ridges, v, beta2, at[id(plus)], at[id(minus)], period)
            new, gone = changes(base, found(edited), mid)
            row.update(new=len(new), gone=len(gone))
        rows.append(row)
        if sum(r["op"] == "remove" and "gone" in r for r in rows) >= EDITS:
            break
    return {"seed": seed, "fgp": fgp, "period": period, "minutiae": len(inner), "rows": rows}


def main(out, masters=16):
    tasks = [(s, f) for s in range(5000, 5000 + masters // 4) for f in (2, 3, 7, 8)]
    with ProcessPoolExecutor(JOBS) as pool:
        results = list(pool.map(one, tasks))
    json.dump(results, open(out, "w"))
    rows = [r for res in results for r in res["rows"]]
    for layout in ("along", "across", "diagonal"):
        xs = [r for r in rows if r["op"] == "add" and r["layout"] == layout]
        two = [r for r in xs if r["new"] == 2 and r["gone"] == 0]
        print(json.dumps({"add": layout, "tries": len(xs), "exactly_two": len(two) / len(xs),
                          "types": {t: sum(r["types"] == list(t) for r in two) for t in ("bb", "be", "ee")},
                          "facing_over_144": sum(r["angle"] > 144 for r in two) / max(len(two), 1),
                          "new_hist": np.bincount([r["new"] for r in xs]).tolist()}))
    rem = [r for r in rows if r["op"] == "remove"]
    opposite = [r for r in rem if "gone" in r]
    print(json.dumps({"remove_candidates": len(rem), "opposite_windings": len(opposite) / max(len(rem), 1),
                      "both_gone_none_new": sum(r["gone"] == 2 and r["new"] == 0 for r in opposite) / max(len(opposite), 1),
                      "gone_hist": np.bincount([r["gone"] for r in opposite]).tolist(),
                      "new_hist": np.bincount([r["new"] for r in opposite]).tolist(),
                      "windings": {str(k): sum(tuple(r["windings"]) == k for r in rem)
                                   for k in {tuple(r["windings"]) for r in rem}}}))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
