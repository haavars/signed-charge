"""Net charge by phase winding (Larkin-Fletcher residues of a vortex-demodulated phase), against mindtct, LEADER,
and flow_charge's smoothed divergence d. Scratch for a peer check of E-090/E-091; writes
nothing in the repo.

Three reads, each frame-free:
  cells    on a stride-8 grid in the guarded disc, every charge (residue, or an extractor's minutia) is lifted to the
           cell's frame along the straight segment (as signed_minutiae.charge_pairs), so rho_sigma(c) = sum q G_sigma
           is a blurred net charge density seen from c, and Phi d_sigma(c) in the same frame is what the budget says
           it should equal. Slope, correlation, mean |.|, split by distance to the nearest core or delta.
  signs    net residue charge within MATCH px of each extractor minutia, in the minutia's heading frame.
  patches  side x side squares with no singular point within MARGIN: net charge by winding against each extractor's
           signed count, both lists with opposite pairs under DIPOLE_FAR px cancelled.
Run from python_biometrics:
  RIDGEGEN_TOOLS=$PWD/tools PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python winding.py OUT.json [SETS...]
LIMIT=n takes the first n prints of each set. S is the residue-to-heading handedness, calibrated on one-sign sheets.
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from signed_charge.reads import agreement_null as an
from signed_charge.measure import fanout as fo
from signed_charge.reads import field_read as fr
from signed_charge.measure import field_similarity as fs
from signed_charge.measure import flow_charge as fc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import minutiae_stats as ms
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge import paths

JOBS = 4
LIMIT = int(os.environ.get("LIMIT", 0))
S = float(os.environ.get("S", -1))  # +1 gave agreement -0.94 on one-sign sheets and 0 % heading match with the extractors
SCALES = (30, 50, 70, 100, 150)
SIDES = (64, 96, 128)
MARGIN, DIPOLE, DIPOLE_FAR, MATCH, SPIRALS, SHEETS, EDGE, STRIDE, SAMPLES = 16, 6.0, 15.0, 8.0, 36, 30, 20, 8, 64
CACHE = str(paths.CACHE)
FLOW = paths.GENERATED / "flow_d055"
RESAMPLES = 1000
BAND, AMPLITUDE, ALONG, DIRECTIONS = (
    (1.5, 6.0),
    0.4,
    5.0,
    16,
)  # the band-pass round the ridge frequency, and the least local amplitude (of the median) a residue may sit in
ROWS = []


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(nx, ny):
    """Ridge tangent from the wave vector direction, handedness S."""
    return -S * ny, S * nx


def along_ridges(b, th2):
    """b smoothed along the local ridge direction (ALONG px), from DIRECTIONS fixed line kernels picked per pixel."""
    theta = np.angle(th2) / 2 + np.pi / 2  # ridge tangent
    half = int(3 * ALONG)
    t = np.arange(-half, half + 1)
    weight = np.exp(-(t**2) / (2 * ALONG**2))
    weight /= weight.sum()
    stack = []
    for k in range(DIRECTIONS):
        a = np.pi * k / DIRECTIONS
        kernel = np.zeros((2 * half + 1, 2 * half + 1), np.float32)
        for w_, dt in zip(weight, t):
            x, y = half + dt * np.cos(a), half + dt * np.sin(a)
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            fx, fy = x - x0, y - y0
            for dx, dy, f in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                if 0 <= x0 + dx <= 2 * half and 0 <= y0 + dy <= 2 * half:
                    kernel[y0 + dy, x0 + dx] += w_ * f
        stack.append(cv2.filter2D(b, -1, kernel))
    stack = np.stack(stack)
    index = np.rint((theta % np.pi) / np.pi * DIRECTIONS).astype(int) % DIRECTIONS
    return np.take_along_axis(stack, index[None], 0)[0]


def demodulate(im, fg, th2=None):
    """Band-passed image b (smoothed along the ridges when th2 is given) and its spiral-phase (Riesz) transform V."""
    g = im.astype(np.float32)
    g -= g[fg].mean()
    b = cv2.GaussianBlur(g, (0, 0), BAND[0]) - cv2.GaussianBlur(g, (0, 0), BAND[1])
    if th2 is not None:
        b = along_ridges(b, th2)
    b *= cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 4)
    h, w = b.shape
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    spiral = (fx + 1j * fy) / (np.abs(fx + 1j * fy) + 1e-12)
    spiral[0, 0] = 0
    return b, np.fft.ifft2(spiral * np.fft.fft2(b))


def residues(b, V, z2, mask):
    """Plaquette residues of the demodulated phase: (x, y, q, beta), q = ±1 relative to n = (cos beta, sin beta),
    the wave vector direction at the plaquette's first corner. Plaquettes holding a half-index point are left out."""
    th = np.angle(z2)
    A, B, C, D = (np.s_[:-1, :-1], np.s_[:-1, 1:], np.s_[1:, 1:], np.s_[1:, :-1])
    t = [th[A]]
    for X in (B, C, D, A):
        t.append(t[-1] + wrap(th[X] - t[-1]))
    w2 = t[-1] - t[0]
    psi = [np.arctan2(np.imag(np.exp(-0.5j * tt) * V[X]), b[X]) for X, tt in zip((A, B, C, D, A), t)]
    res = sum(wrap(psi[i + 1] - psi[i]) for i in range(4)) / (2 * np.pi)
    q = np.rint(res)
    hit = mask[A] & mask[B] & mask[C] & mask[D] & (np.abs(w2) < 1) & (q != 0)
    ys, xs = np.nonzero(hit)
    return xs + 0.5, ys + 0.5, q[hit], t[0][hit] / 2


def heading_of(q, beta):
    kx, ky = rot(np.cos(beta), np.sin(beta))
    return np.arctan2(q * ky, q * kx)


def cancel(x, y, hd, dist):
    """Drop opposite-heading pairs closer than dist, nearest first. Returns a keep mask and the pairs dropped."""
    keep = np.ones(len(x), bool)
    if len(x) < 2:
        return keep, 0
    pts = np.c_[x, y]
    pairs = cKDTree(pts).query_pairs(dist, output_type="ndarray")
    if not len(pairs):
        return keep, 0
    opp = pairs[np.cos(hd[pairs[:, 0]] - hd[pairs[:, 1]]) < 0]
    d = np.hypot(*(pts[opp[:, 0]] - pts[opp[:, 1]]).T)
    dropped = 0
    for i, j in opp[np.argsort(d)]:
        if keep[i] and keep[j]:
            keep[i] = keep[j] = False
            dropped += 1
    return keep, dropped


def unwrap_patch(th2, i0, i1, j0, j1):
    p = th2[i0:i1, j0:j1].astype(float)
    col = np.unwrap(p[:, 0])
    rows = np.unwrap(p, axis=1)
    rows = rows - rows[:, [0]] + col[:, None]
    last = np.unwrap(p[:, -1])
    last = last - last[0] + rows[0, -1]
    return rows / 2, float(rows[-1, -1] - last[-1])


def patches(mask, singular, side):
    h, w = mask.shape
    integral = cv2.integral(mask.astype(np.uint8))
    out = []
    for i0 in range(0, h - side + 1, side // 2):
        for j0 in range(0, w - side + 1, side // 2):
            full = integral[i0 + side, j0 + side] - integral[i0, j0 + side] - integral[i0 + side, j0] + integral[i0, j0]
            if full < side * side:
                continue
            if any(j0 - MARGIN <= x <= j0 + side + MARGIN and i0 - MARGIN <= y <= i0 + side + MARGIN for x, y in singular):
                continue
            out.append((i0, j0))
    return out


def lifted(th2, cells, src, q_src):
    """Charges q_src (relative to the wave vector frame at their own pixel) seen from each cell, lifted along straight
    segments: (cells, sources) array of ±1 times q."""
    h, w = th2.shape
    t = np.linspace(0, 1, SAMPLES)
    out = np.zeros((len(cells), len(src)))
    for r, (sx, sy) in enumerate(src):
        xs = np.clip(np.rint(sx + t[None] * (cells[:, 0] - sx)[:, None]).astype(int), 0, w - 1)
        ys = np.clip(np.rint(sy + t[None] * (cells[:, 1] - sy)[:, None]).astype(int), 0, h - 1)
        u = np.unwrap(th2[ys, xs], axis=1)
        out[:, r] = q_src[r] * np.sign(np.cos((u[:, -1] - u[:, 0]) / 2) + 1e-12)
    return out


def analyse(im, points, inner, finger, truth=None):
    """One print. points: {reader: (n, 4) rows (x, y, angle, bif)} inside inner."""
    fg = fs.foreground(im)
    h, w = im.shape
    yy, xx = np.mgrid[0:h, 0:w]
    inner = inner & (xx >= EDGE) & (xx < w - EDGE) & (yy >= EDGE) & (yy < h - EDGE)
    ys, xs = np.nonzero(inner)
    disc = inner & (np.hypot(xx - xs.mean(), yy - ys.mean()) <= fc.DISC)
    singular = sc.singular(fs.field(im), fg)
    guarded = disc.copy()
    near = np.full(im.shape, np.inf)
    for x, y in singular:
        r = np.hypot(xx - x, yy - y)
        guarded &= r >= sc.GUARD
        near = np.minimum(near, r)
    z2 = sm.direction(im)
    z2 = z2 / (np.abs(z2) + 1e-9)
    th2 = np.angle(z2)
    b, V = demodulate(im, fg, z2)
    qmap = sf.quality(im)
    amp = cv2.GaussianBlur(np.abs(b + 1j * np.imag(np.exp(-0.5j * th2) * V)).astype(np.float32), (0, 0), 4)
    strong = inner & (amp > AMPLITUDE * np.median(amp[inner]))
    rx, ry, rq, rb = residues(b, V, z2, strong)
    rh = heading_of(rq, rb)
    keep6, dip6 = cancel(rx, ry, rh, DIPOLE)
    keep15, dip15 = cancel(rx, ry, rh, DIPOLE_FAR)
    res6 = np.c_[rx, ry, rh, np.zeros(len(rx))][keep6]
    res15 = np.c_[rx, ry, rh, np.zeros(len(rx))][keep15]
    spread = {s: fc.spreading(z2, float(s)) for s in SCALES}
    _, _, period = fo.fields(im, fg)
    phi = 1.0 / np.maximum(period, 1.0)
    out = {
        "finger": finger,
        "residues": int(len(rx)),
        "cleaned6": int(keep6.sum()),
        "cleaned15": int(keep15.sum()),
        "singular": len(singular),
        "weak_share": float(1 - strong.sum() / max(inner.sum(), 1)),
        "guarded_px": int(guarded.sum()),
        "phi_disc": float(phi[guarded].mean()) if guarded.any() else float("nan"),
    }
    # Agreement with d at each scale: residues (cleaned at 6 and 15 px) and the extractors, in the guarded disc.
    out["agree"], out["count"] = {}, {}
    for name, p, y_sign in [("res6", sf.inside(res6, guarded), 1.0), ("res15", sf.inside(res15, guarded), 1.0)] + [
        (reader, sf.inside(points[reader], guarded), y_sign) for reader, (_, y_sign) in sm.EXTRACTORS.items()
    ]:
        out["agree"][name] = {s: an.agreement(p, y_sign, *spread[s]) if len(p) else (0.0, 0) for s in SCALES}
        out["count"][name] = int(len(p))
    # Per-minutia: net residue charge within MATCH px, in the minutia's heading frame.
    out["signs"] = {}
    tree = cKDTree(np.c_[rx, ry]) if len(rx) else None
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = sf.inside(points[reader], guarded)
        hist = {"-": 0, "0": 0, "+": 0, "++": 0, "n": int(len(p)), "c-": 0, "c0": 0, "c+": 0, "c++": 0, "cn": 0}
        if tree is not None and len(p):
            hd = sf.heading(p, y_sign)
            good = qmap[p[:, 1].astype(int), p[:, 0].astype(int)] >= 4
            hist["cn"] = int(good.sum())
            for k, near_r in enumerate(tree.query_ball_point(p[:, :2], MATCH)):
                net = int(np.sign(np.cos(hd[k] - rh[near_r])).sum()) if near_r else 0
                key = "-" if net < 0 else "0" if net == 0 else "+" if net == 1 else "++"
                hist[key] += 1
                if good[k]:
                    hist["c" + key] += 1
        out["signs"][reader] = hist
    # Cells: blurred charge seen from each cell against Phi d_sigma in the cell's frame.
    cy, cx = np.nonzero(guarded[::STRIDE, ::STRIDE])
    cells = np.c_[cx * STRIDE, cy * STRIDE].astype(float)
    out["cells"] = {}
    if len(cells) >= 10:
        ci_, cj_ = cells[:, 1].astype(int), cells[:, 0].astype(int)
        beta_c = th2[ci_, cj_] / 2
        kcx, kcy = rot(np.cos(beta_c), np.sin(beta_c))
        sources = {}
        rxi, ryi = ry.astype(int), rx.astype(int)
        qr_pix = rq * np.sign(np.cos(rb - th2[rxi, ryi] / 2) + 1e-12)
        sources["res"] = (np.c_[rx, ry], qr_pix)
        for reader, (_, y_sign) in sm.EXTRACTORS.items():
            p = sf.inside(points[reader], inner)
            if len(p):
                hd = sf.heading(p, y_sign)
                beta_m = th2[p[:, 1].astype(int), p[:, 0].astype(int)] / 2
                kmx, kmy = rot(np.cos(beta_m), np.sin(beta_m))
                sources[reader] = (p[:, :2], np.sign(np.cos(hd) * kmx + np.sin(hd) * kmy + 1e-12))
            else:
                sources[reader] = (np.zeros((0, 2)), np.zeros(0))
        if truth is not None:
            tx, ty, ts, grad = truth
            beta_t = th2[ty.astype(int), tx.astype(int)] / 2
            n_dot = np.cos(beta_t) * grad[0] + np.sin(beta_t) * grad[1]
            sources["truth"] = (np.c_[tx, ty], ts * np.sign(n_dot) * TRUTH_SIGN)
        lifts = {name: lifted(th2, cells, xy, q) if len(xy) else np.zeros((len(cells), 0)) for name, (xy, q) in sources.items()}
        dist = {
            name: np.hypot(cells[:, 0][:, None] - xy[:, 0][None], cells[:, 1][:, None] - xy[:, 1][None]) if len(xy) else np.zeros((len(cells), 0))
            for name, (xy, q) in sources.items()
        }
        near_c = near[ci_, cj_]
        phi_c = phi[ci_, cj_]
        clean_c = qmap[ci_, cj_] >= 3
        for s in SCALES:
            ang, d = spread[s]
            dc = d[ci_, cj_] * np.sign(np.cos(ang[ci_, cj_]) * kcx + np.sin(ang[ci_, cj_]) * kcy + 1e-12) * phi_c
            rho = {name: (lifts[name] * np.exp(-(dist[name] ** 2) / (2 * s * s)) / (2 * np.pi * s * s)).sum(1) for name in sources}
            entry = {}
            for part, pick in (("near", near_c < s), ("far", near_c >= s), ("farclean", (near_c >= s) & clean_c), ("all", np.ones(len(cells), bool))):
                if pick.sum() < 3:
                    continue
                e = {
                    "n": int(pick.sum()),
                    "abs_d": float(np.abs(dc[pick]).sum()),
                    "dd": float(dc[pick] @ dc[pick]),
                    "cross": {reader: float(rho["res"][pick] @ rho[reader][pick]) for reader in sm.EXTRACTORS},
                }
                for name in sources:
                    r_ = rho[name][pick]
                    e[name] = (float(np.abs(r_).sum()), float(r_ @ r_), float(r_ @ dc[pick]))
                entry[part] = e
            out["cells"][s] = entry
    # Patches: winding against the extractors' counts, both with opposite pairs under DIPOLE_FAR cancelled.
    out["patches"] = []
    ext = {}
    for reader, (_, y_sign) in sm.EXTRACTORS.items():
        p = points[reader]
        if len(p):
            hd = sf.heading(p, y_sign)
            k15, _ = cancel(p[:, 0], p[:, 1], hd, DIPOLE_FAR)
            ext[reader] = (p[k15], hd[k15])
        else:
            ext[reader] = (p, np.zeros(0))
    r15 = (rx[keep15], ry[keep15], rq[keep15], rb[keep15])
    for side in SIDES:
        for i0, j0 in patches(inner, singular, side):
            i1, j1 = i0 + side, j0 + side
            beta, mismatch = unwrap_patch(th2, i0, i1, j0, j1)
            if abs(mismatch) > 1:
                continue
            nx, ny = np.cos(beta), np.sin(beta)
            kx, ky = rot(nx, ny)
            inside = lambda x, y: (x >= j0) & (x < j1) & (y >= i0) & (y < i1)  # noqa: E731
            m = inside(r15[0], r15[1])
            xi, yi = (r15[0][m] - j0).astype(int), (r15[1][m] - i0).astype(int)
            qw = float((r15[2][m] * np.sign(np.cos(r15[3][m]) * nx[yi, xi] + np.sin(r15[3][m]) * ny[yi, xi])).sum())
            qe, ne = {}, {}
            for reader, (p, hd) in ext.items():
                mm = inside(p[:, 0], p[:, 1]) if len(p) else np.zeros(0, bool)
                if mm.any():
                    xi, yi = (p[mm, 0] - j0).astype(int), (p[mm, 1] - i0).astype(int)
                    qe[reader] = float(np.sign(np.cos(hd[mm]) * kx[yi, xi] + np.sin(hd[mm]) * ky[yi, xi]).sum())
                else:
                    qe[reader] = 0.0
                ne[reader] = int(mm.sum())
            out["patches"].append(
                {
                    "side": side,
                    "qw": qw,
                    "nres": int(m.sum()),
                    "qe": qe,
                    "ne": ne,
                    "qmin": int(qmap[i0:i1, j0:j1].min()),
                    "qmean": float(qmap[i0:i1, j0:j1].mean()),
                }
            )
    if truth is not None:
        tx, ty, ts, grad = truth
        found = same = 0
        if len(res6):
            d, idx = cKDTree(res6[:, :2]).query(np.c_[tx, ty], distance_upper_bound=4.0)
            got = np.isfinite(d)
            found = int(got.sum())
            for k in np.flatnonzero(got):
                r = idx[k]
                n_dot = np.cos(rb[keep6][r]) * grad[0] + np.sin(rb[keep6][r]) * grad[1]
                same += int(rq[keep6][r] == ts[k] * np.sign(n_dot) * TRUTH_SIGN)
        out["truth"] = {"planted": int(len(tx)), "found": found, "same_sign": same, "extra": int(len(res6) - found)}
    return out


TRUTH_SIGN = 1.0


def sheet(seed, kind="independent"):
    rng = np.random.default_rng(seed)
    h, w = sm.SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rot_ = rng.uniform(0, np.pi)
    psi = 2 * np.pi * (-(xx - w / 2) * np.sin(rot_) + (yy - h / 2) * np.cos(rot_)) / sm.PERIOD + rng.uniform(0, 2 * np.pi)
    tx, ty, ts = [], [], []
    for _ in range(SPIRALS):
        cx, cy = rng.uniform(40, w - 40), rng.uniform(40, h - 40)
        s = 1 if kind == "one_sign" else rng.choice([-1, 1])
        psi += s * np.arctan2(yy - cy, xx - cx)
        tx.append(cx), ty.append(cy), ts.append(s)
    im = np.where(np.cos(psi) > 0, 40, 235).astype(np.uint8)
    return im, psi, (np.array(tx), np.array(ty), np.array(ts), (-np.sin(rot_), np.cos(rot_)))


def fresh_points(im, inner):
    return {reader: sf.inside(extract(im), inner) for reader, (extract, _) in sm.EXTRACTORS.items()}


def inner_of(im):
    fg = fs.foreground(im)
    return cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)


def task(t):
    kind, payload = t
    if kind == "cache":
        index, source = payload
        row = ROWS[index]
        return analyse(place.image(source), row["points"], parts.mask_of(row), row["finger"])
    if kind == "flow":
        im = fr.sheet(payload)
        inner = inner_of(im)
        return analyse(im, fresh_points(im, inner), inner, Path(payload).stem)
    if kind == "sheet":
        seed, sk = payload
        im, _, truth = sheet(seed, sk)
        inner = inner_of(im)
        return analyse(im, fresh_points(im, inner), inner, f"{sk}/{seed}", truth)


def ci(table, fingers, f, rng):
    uniq = sorted(set(fingers))
    where = {u: i for i, u in enumerate(uniq)}
    idx = np.array([where[x] for x in fingers])
    groups = [np.flatnonzero(idx == i) for i in range(len(uniq))]
    full = f(np.arange(len(fingers)))
    draws = [f(np.concatenate([groups[i] for i in rng.integers(len(uniq), size=len(uniq))])) for _ in range(RESAMPLES)]
    return {"mean": float(full), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]}


def summary(rows, rng):
    fingers = [r["finger"].rsplit("_", 1)[0] if "/" not in r["finger"] else r["finger"] for r in rows]
    rep = {"prints": len(rows), "fingers": len(set(fingers))}
    rep["mean_counts"] = {k: float(np.mean([r["count"][k] for r in rows])) for k in rows[0]["count"]}
    rep["mean_residues_raw"] = float(np.mean([r["residues"] for r in rows]))
    rep["weak_share"] = float(np.mean([r["weak_share"] for r in rows]))
    rep["phi"] = float(np.nanmean([r["phi_disc"] for r in rows]))
    rep["agree"] = {}
    for who in rows[0]["agree"]:
        rep["agree"][who] = {}
        for s in SCALES:
            tab = np.array([r["agree"][who][s] for r in rows], float)
            rep["agree"][who][s] = ci(tab, fingers, lambda i: tab[i, 0].sum() / max(tab[i, 1].sum(), 1), rng)
    rep["signs"] = {}
    for reader in sm.EXTRACTORS:
        tab = np.array([[r["signs"][reader][k] for k in ("-", "0", "+", "++", "n")] for r in rows], float)
        tot = tab[:, 4].sum()
        rep["signs"][reader] = {
            "minutiae": float(tot),
            "share": {k: float(tab[:, i].sum() / max(tot, 1)) for i, k in enumerate(("-", "0", "+", "++"))},
            "opposite_given_signed": ci(tab, fingers, lambda i: tab[i, 0].sum() / max(tab[i, 0].sum() + tab[i, 2].sum() + tab[i, 3].sum(), 1), rng),
        }
        tab = np.array([[r["signs"][reader][k] for k in ("c-", "c0", "c+", "c++", "cn")] for r in rows], float)
        tot = tab[:, 4].sum()
        rep["signs"][reader]["clean"] = {
            "minutiae": float(tot),
            "share": {k: float(tab[:, i].sum() / max(tot, 1)) for i, k in enumerate(("-", "0", "+", "++"))},
            "opposite_given_signed": ci(tab, fingers, lambda i: tab[i, 0].sum() / max(tab[i, 0].sum() + tab[i, 2].sum() + tab[i, 3].sum(), 1), rng),
        }
    rep["cells"] = {}
    for s in SCALES:
        rep["cells"][s] = {}
        for part in ("near", "far", "farclean", "all"):
            have = [r for r in rows if str(s) in r["cells"] or s in r["cells"]]
            have = [r for r in have if part in r["cells"].get(s, r["cells"].get(str(s), {}))]
            if len(have) < 3:
                continue
            get = lambda r: r["cells"].get(s, r["cells"].get(str(s)))[part]  # noqa: E731
            hf = [fingers[rows.index(r)] for r in have]
            n = np.array([get(r)["n"] for r in have], float)
            absd = np.array([get(r)["abs_d"] for r in have])
            dd = np.array([get(r)["dd"] for r in have])
            e = {"prints": len(have), "cells": float(n.sum()), "mean_abs_phi_d": float(absd.sum() / n.sum())}
            rr = np.array([get(r)["res"][1] for r in have]).sum()
            for reader in sm.EXTRACTORS:
                e[f"corr_res_{reader}"] = float(
                    np.array([get(r)["cross"][reader] for r in have]).sum() / np.sqrt(max(rr * np.array([get(r)[reader][1] for r in have]).sum(), 1e-24))
                )
            for name in get(have[0]):
                if name in ("n", "abs_d", "dd", "cross"):
                    continue
                t = np.array([get(r)[name] for r in have])  # abs, rr, rd
                e[name] = {
                    "mean_abs_rho": float(t[:, 0].sum() / n.sum()),
                    "slope_phi_d_on_rho": ci(t, hf, lambda i: t[i, 2].sum() / max(t[i, 1].sum(), 1e-12), rng),
                    "slope_rho_on_phi_d": ci(t, hf, lambda i: t[i, 2].sum() / max(dd[i].sum(), 1e-12), rng),
                    "corr": float(t[:, 2].sum() / np.sqrt(max(t[:, 1].sum() * dd.sum(), 1e-24))),
                }
            rep["cells"][s][part] = e
    rep["patches"] = {}
    for side, grade in [(sd, g) for sd in SIDES for g in ("all", "clean")]:
        P = [(k, p) for k, r in enumerate(rows) for p in r["patches"] if p["side"] == side and (grade == "all" or p["qmin"] >= 3)]
        if len(P) < 5:
            continue
        pf = [fingers[k] for k, _ in P]
        qw = np.array([p["qw"] for _, p in P])
        entry = {
            "n": len(P),
            "mean_abs_qw": float(np.abs(qw).mean()),
            "mean_nres": float(np.mean([p["nres"] for _, p in P])),
            "mean_n": {rd: float(np.mean([p["ne"][rd] for _, p in P])) for rd in sm.EXTRACTORS},
        }
        for rd in sm.EXTRACTORS:
            arr = np.array([p["qe"][rd] for _, p in P])
            entry[rd] = {
                "mean_abs": float(np.abs(arr).mean()),
                "slope_on_qw": ci(arr, pf, lambda i: float(arr[i] @ qw[i] / max(qw[i] @ qw[i], 1e-9)), rng),
                "corr": float(np.corrcoef(arr, qw)[0, 1]) if arr.std() > 0 and qw.std() > 0 else float("nan"),
                "exact": float(np.mean(arr == qw)),
                "rms_diff": float(np.sqrt(np.mean((arr - qw) ** 2))),
            }
        rep["patches"][f"{side}/{grade}"] = entry
    if "truth" in rows[0]:
        t = np.array([(r["truth"]["planted"], r["truth"]["found"], r["truth"]["same_sign"], r["truth"]["extra"]) for r in rows], float).sum(0)
        rep["truth"] = {
            "planted": float(t[0]),
            "found": float(t[1] / t[0]),
            "same_sign": float(t[2] / max(t[1], 1)),
            "extra_per_sheet": float(t[3] / len(rows)),
        }
    return rep


def show(name, rep):
    print(
        "==",
        name,
        rep["prints"],
        "prints; counts",
        {k: round(v, 1) for k, v in rep["mean_counts"].items()},
        "raw residues",
        round(rep["mean_residues_raw"], 1),
        "weak share",
        round(rep["weak_share"], 3),
        "phi",
        round(rep["phi"], 4),
    )
    print("  agree", {who: {s: round(v["mean"], 2) for s, v in a.items()} for who, a in rep["agree"].items()})
    print(
        "  signs",
        {
            rd: (
                {k: round(v, 3) for k, v in e["share"].items()},
                round(e["opposite_given_signed"]["mean"], 3),
                [round(c, 3) for c in e["opposite_given_signed"]["ci"]],
            )
            for rd, e in rep["signs"].items()
        },
    )
    print(
        "  signs clean",
        {
            rd: (
                int(e["clean"]["minutiae"]),
                {k: round(v, 3) for k, v in e["clean"]["share"].items()},
                round(e["clean"]["opposite_given_signed"]["mean"], 3),
                [round(c, 3) for c in e["clean"]["opposite_given_signed"]["ci"]],
            )
            for rd, e in rep["signs"].items()
        },
    )
    for s, parts_ in rep["cells"].items():
        for part, e in parts_.items():
            print(
                f"  cells s={s} {part}: prints={e['prints']} cells={int(e['cells'])} |phi d|={e['mean_abs_phi_d']:.2e} res~mindtct={e['corr_res_mindtct']:.2f} res~leader={e['corr_res_leader']:.2f}",
                {
                    name: (
                        f"|rho|={v['mean_abs_rho']:.2e}",
                        f"d/rho={v['slope_phi_d_on_rho']['mean']:.2f}",
                        f"rho/d={v['slope_rho_on_phi_d']['mean']:.2f}",
                        f"corr={v['corr']:.2f}",
                    )
                    for name, v in e.items()
                    if isinstance(v, dict)
                },
            )
    for side, e in rep["patches"].items():
        print(
            f"  L={side} n={e['n']} |qw|={e['mean_abs_qw']:.2f} nres={e['mean_nres']:.1f} n={ {k: round(v, 1) for k, v in e['mean_n'].items()} }",
            {
                k: (
                    round(e[k]["mean_abs"], 2),
                    round(e[k]["slope_on_qw"]["mean"], 2),
                    [round(c, 2) for c in e[k]["slope_on_qw"]["ci"]],
                    round(e[k]["corr"], 2),
                    round(e[k]["exact"], 2),
                )
                for k in sm.EXTRACTORS
            },
        )
    if "truth" in rep:
        print("  truth", rep["truth"])


def main(out, sets):
    base = pickle.load(open(CACHE, "rb"))["rows"]
    where = place.sources(base)
    tasks, seen = [], {}
    for row in base:
        name = next((n for n in ("real", "sfinge", "model2") if parts.member(row, n)), None)
        if name is None or name not in sets or id(row) not in where or (LIMIT and seen.get(name, 0) >= LIMIT):
            continue
        seen[name] = seen.get(name, 0) + 1
        ROWS.append(row)
        tasks.append((name, ("cache", (len(ROWS) - 1, where[id(row)]))))
    for arm in ("realflow", "readflow"):
        if arm in sets:
            for path in sorted(FLOW.glob(f"{arm}_*.npz"))[: LIMIT or None]:
                tasks.append((arm, ("flow", str(path))))
    for sk in ("independent", "one_sign"):
        if sk in sets:
            for seed in range(LIMIT or SHEETS):
                tasks.append((sk, ("sheet", (seed, sk))))
    print(len(tasks), "tasks", flush=True)
    with ProcessPoolExecutor(JOBS) as pool:
        read = list(pool.map(task, [t[1] for t in tasks], chunksize=2))
    rng = np.random.default_rng(0)
    report = {name: summary([r for t, r in zip(tasks, read) if t[0] == name], rng) for name in sorted({t[0] for t in tasks})}
    json.dump(report, open(out, "w"), indent=1)
    for name, rep in report.items():
        show(name, rep)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["real", "sfinge", "model2", "realflow", "readflow", "independent", "one_sign"])
