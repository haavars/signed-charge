"""One phase for a whole master, its minutiae chosen (step 3, route b): phase_global.py OUT

Local spiral-phase edits (`phase_edit.py`, `phase_masters.py`) can't reach a core's surroundings, where ψ's sign
convention folds. Feng & Jain 2011 (TPAMI 33(2):209-218) reconstruct a whole print's phase: unwrap the orientation
with branch cuts from each singular point, take the phase gradient G = 2π/period across the ridges, subtract the
chosen spirals' gradient, integrate what's left (the continuous phase ψc) and add the spirals back. Their
integration goes block by block and leaves spurious minutiae near singular points. Here ψc is one least-squares
solve over every pixel, and across a branch cut the sides are conjugate (the direction flips, so ψ on one side is
−ψ on the other), which the solve holds as an equation, ψc_i + ψc_j = −(ψs_i + ψs_j) − G_i·Δ:

- **Branch cuts** run straight up from each core and delta to the master's top edge, and each spiral's own arg
  branch runs straight up too, so no spiral's 2π jump crosses a cut (parallel lines don't meet).
- **The unwrapped direction**: θ + π/2 unwrapped breadth first, the cuts not crossed; an edge whose directions
  differ by about π is a seam edge.
- **The flow's minutiae**: where the ridges spread or close up (period and orientation), the gradient field has
  curl: ρ = curl G / 2π is the density of net spirals the flow needs. Without them the solve spreads the mismatch
  as spacing errors. Drawn here as a Poisson field of ±1 spirals with intensity |ρ|.

Test on master model 2's masters: ψ with no spirals; with the flow's spirals; with those and facing pairs. Each
rendered tanh(4 cos ψ), its minutiae extracted (`minutiae.extract`, the ground truth's extractor) and matched to the
spirals placed: how many found, how many spurious, and where (near a singular point, a cut, or elsewhere).

Run: python -m signed_charge.generate.phase_global OUT.json
"""

import json
import sys
import time
from collections import deque

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import map_coordinates

from signed_charge.generate import minutiae_growth as growth

SATURATION = 4.0
HEALING = 100  # conjugate-gradient steps after the half-size solve
NEAR_SPIRAL = 1.5  # px: edges this close to a spiral carry little weight (its gradient is singular there)
FLOW_SALT, PAIR_SALT, CLUMP_SALT = 51, 52, 53
CLUMP_SCALE = 100  # px


def arg_up(dx, dy):
    """arg(dx + i dy) with its branch cut straight up (towards −y), in (−π/2, 3π/2]."""
    return np.angle((dx + 1j * dy) * -1j) + np.pi / 2


def cuts(shape, singular):
    """Branch cuts: from each singular point straight up to the top edge, 1 px wide."""
    barrier = np.zeros(shape, bool)
    for x, y in singular:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < shape[1]:
            barrier[: max(0, min(shape[0], yi + 1)), xi] = True
    return barrier


def unwrap_direction(theta, barrier):
    """θ + π/2, unwrapped breadth first across 4-neighbours, not crossing `barrier`; barrier pixels from their left
    neighbour."""
    h, w = theta.shape
    beta = np.mod(theta + np.pi / 2, np.pi).astype(np.float64)
    out = np.full((h, w), np.nan)
    start = (h - 1, w // 2)
    out[start] = beta[start]
    queue = deque([start])
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and np.isnan(out[ny, nx]) and not barrier[ny, nx]:
                out[ny, nx] = beta[ny, nx] + np.pi * np.round((out[y, x] - beta[ny, nx]) / np.pi)
                queue.append((ny, nx))
    ys, xs = np.nonzero(np.isnan(out))
    for y, x in sorted(zip(ys, xs), key=lambda p: p[1]):
        ref = out[y, x - 1] if x > 0 and not np.isnan(out[y, x - 1]) else beta[y, x]
        out[y, x] = beta[y, x] + np.pi * np.round((ref - beta[y, x]) / np.pi)
    return out


def spiral_phase(shape, spirals):
    """ψs and its gradient (gx, gy) for spirals [(x, y, p)], each arg's branch straight up."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    psi, gx, gy = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for x, y, p in spirals:
        dx, dy = xx - x, yy - y
        r2 = dx**2 + dy**2 + 1e-9
        psi += p * arg_up(dx, dy)
        gx += p * -dy / r2
        gy += p * dx / r2
    return psi, gx, gy


def direct(normal, b):
    """The sparse direct solve: symmetric ordering, no pivoting (6.5 s and 1.6 GB at full size, against 11 s and 1.8
    GB for `spsolve`, the same answer)."""
    return spla.splu(normal, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)).solve(b)


def assemble_anisotropic(beta_u, gx, gy, cgx, cgy, psi_s, weight, spacing_weight):
    """Rows for ψc at each pixel, from forward differences (a seam difference as in `solve_continuous`): along the
    ridges, t·∇ψ = t·G, weight 1, so the ridges keep the flow's direction; across, n·∇ψ = n·G, weight `spacing_weight`,
    so where the flow can't be met exactly the spacing gives (as real ridges' does), not the direction."""
    h, w = beta_u.shape
    idx = np.arange(h * w).reshape(h, w)
    i, right, down = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()
    b = beta_u.ravel()
    seam_x, seam_y = np.abs(b[right] - b[i]) > np.pi / 2, np.abs(b[down] - b[i]) > np.pi / 2
    ps = psi_s.ravel()
    # ψ's differences along x and y: on a seam the neighbour is seen conjugate, its coefficient −1.
    rx = np.where(seam_x, gx.ravel()[i] + ps[right] + ps[i], 0.5 * (cgx.ravel()[i] + cgx.ravel()[right]))
    ry = np.where(seam_y, gy.ravel()[i] + ps[down] + ps[i], 0.5 * (cgy.ravel()[i] + cgy.ravel()[down]))
    sx, sy = np.where(seam_x, -1.0, 1.0), np.where(seam_y, -1.0, 1.0)
    rows, cols, vals, rhs, wts = [], [], [], [], []
    n = len(i)
    for k, (cx, cy, row_weight) in enumerate(((-np.sin(b[i]), np.cos(b[i]), 1.0), (np.cos(b[i]), np.sin(b[i]), spacing_weight))):
        e = np.arange(n) + k * n
        rows += [e, e, e]
        cols += [i, right, down]
        vals += [-cx - cy, cx * sx, cy * sy]
        rhs.append(cx * rx + cy * ry)
        wts.append(weight.ravel()[i] * row_weight)
    a = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, h * w))
    return a, np.concatenate(rhs), np.concatenate(wts)


def solve_continuous(beta_u, period, spirals, exact=False, spacing_weight=None):
    """ψc by least squares over every 4-neighbour edge: ψc_j − ψc_i = Gc·Δ within a sheet, ψc_i + ψc_j =
    −(ψs_i + ψs_j) − G_i·Δ across a seam.

    Solved directly at half size (ψc is smooth), brought up, and `HEALING` conjugate-gradient steps heal the strip
    along each cut, where the interpolation mixes the two sheets: about 5 s and 0.3 GB for a master against 6.5 s and
    1.2 GB exactly (`exact`), minutiae within 1 px at the median, and a slight smooth shift of the ridges (E-025).
    """
    h, w = beta_u.shape
    k = 2 * np.pi / period
    gx, gy = k * np.cos(beta_u), k * np.sin(beta_u)
    psi_s, sgx, sgy = spiral_phase((h, w), spirals)
    cgx, cgy = gx - sgx, gy - sgy
    weight = np.ones((h, w))
    if spirals:
        yy, xx = np.mgrid[0:h, 0:w]
        for x, y, _ in spirals:
            weight[(xx - x) ** 2 + (yy - y) ** 2 < NEAR_SPIRAL**2 * 4] = 0.01
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, vals, rhs, wts = [], [], [], [], []
    n = 0
    for axis, (dy, dx) in enumerate(((0, 1), (1, 0))):
        i = idx[: h - dy, : w - dx].ravel()
        j = idx[dy:, dx:].ravel()
        bi, bj = beta_u.ravel()[i], beta_u.ravel()[j]
        seam = np.abs(bj - bi) > np.pi / 2
        g_i = (cgx if axis == 0 else cgy).ravel()
        g_j = g_i
        ew = np.minimum(weight.ravel()[i], weight.ravel()[j])
        m = len(i)
        e = np.arange(n, n + m)
        # Within a sheet: ψj − ψi = (Gc_i + Gc_j)/2.
        within = ~seam
        rows += [e[within], e[within]]
        cols += [j[within], i[within]]
        vals += [np.ones(within.sum()), -np.ones(within.sum())]
        b = np.zeros(m)
        b[within] = 0.5 * (g_i[i[within]] + g_j[j[within]])
        # Across a seam: ψi + ψj = −(ψs_i + ψs_j) − (G_i − G_j)/2 (G total, its sign flips across).
        tg = (gx if axis == 0 else gy).ravel()
        rows += [e[seam], e[seam]]
        cols += [i[seam], j[seam]]
        vals += [np.ones(seam.sum()), np.ones(seam.sum())]
        b[seam] = -(psi_s.ravel()[i[seam]] + psi_s.ravel()[j[seam]]) - 0.5 * (tg[i[seam]] - tg[j[seam]])
        rhs.append(b)
        wts.append(ew)
        n += m
    if spacing_weight is None:
        a = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, h * w))
        wv = sp.diags(np.concatenate(wts))
        b = a.T @ (wv @ np.concatenate(rhs))
    else:
        a, right_side, row_weights = assemble_anisotropic(beta_u, gx, gy, cgx, cgy, psi_s, weight, spacing_weight)
        wv = sp.diags(row_weights)
        b = a.T @ (wv @ right_side)
    normal = (a.T @ wv @ a + 1e-8 * sp.identity(h * w)).tocsc()
    t = time.time()
    if exact or min(h, w) < 200:
        psi_c = direct(normal, b)
    else:
        half = period[::2, ::2] / 2 if np.ndim(period) else period / 2
        coarse, _, _ = solve_continuous(beta_u[::2, ::2], half, [(x / 2, y / 2, p) for x, y, p in spirals], exact=True, spacing_weight=spacing_weight)
        yy, xx = np.mgrid[0:h, 0:w]
        start = map_coordinates(coarse, [yy / 2, xx / 2], order=1, mode="nearest").ravel()
        jacobi = 1 / normal.diagonal()
        preconditioner = spla.LinearOperator(normal.shape, matvec=lambda v: jacobi * v)
        psi_c, _ = spla.cg(normal, b, x0=start, M=preconditioner, rtol=1e-12, maxiter=HEALING)
    return psi_c.reshape(h, w), psi_s, time.time() - t


def dithered(rho, beta_u, period, rng):
    """±1 spirals meeting the demand `rho` by error diffusion over cells of a ridge period: a spiral where a cell's
    demand, with what's carried in, reaches about ½, the rest carried on to its neighbours (Floyd-Steinberg, the rows
    serpentine), so the net charge matches at every scale. The demand's sign is its sheet's, so it flips when carried
    across a branch cut."""
    c = int(round(period))
    h, w = rho.shape[0] // c, rho.shape[1] // c
    cells = rho[: h * c, : w * c].reshape(h, c, w, c).sum((1, 3))
    beta = beta_u[c // 2 :: c, c // 2 :: c][:h, :w]
    spirals = []
    for i in range(h):
        d = 1 if i % 2 == 0 else -1
        for j in range(w) if d == 1 else range(w - 1, -1, -1):
            value = cells[i, j]
            sign = 0
            if abs(value) >= rng.uniform(0.4, 0.6):
                sign = int(np.sign(value))
                spirals.append((float((j + rng.random()) * c), float((i + rng.random()) * c), sign))
            rest = value - sign
            for di, dj, share in ((0, d, 7 / 16), (1, -d, 3 / 16), (1, 0, 5 / 16), (1, d, 1 / 16)):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    across = np.cos(beta[ni, nj] - beta[i, j]) < 0
                    cells[ni, nj] += (-1 if across else 1) * share * rest
    return spirals


def period_field(beta_u, period, singular, smoothness, bound, step=4, clear=2.5):
    """Ridge spacing that pays for some of the flow's fan-out before spirals have to (D-014, E-042). With u = ln k, curl(k n) = 0 is
    transport along the ridges, F·∇u = -n·∇β, F = (sin β, -cos β), so minimise ∫w(F·∇u + n·∇β)² + `smoothness`|∇u|² on a coarse grid: F·∇u
    alone has no solution round a loop, and `clear` periods about a singular point (w = 0), where the fan-out diverges, are left to the
    singular point itself. u is limited to ±`bound`, from real prints' spread (E-042). Returns the period, same shape as beta_u."""
    b = beta_u[::step, ::step]
    h, w = b.shape
    idx = np.arange(h * w).reshape(h, w)
    wrapped = lambda d: np.where(np.abs(d) > np.pi / 2, np.nan, d)  # across a seam: dropped
    bx, by = np.full((h, w), np.nan), np.full((h, w), np.nan)
    bx[:, 1:-1] = wrapped(b[:, 2:] - b[:, :-2]) / (2 * step)
    by[1:-1, :] = wrapped(b[2:, :] - b[:-2, :]) / (2 * step)
    cos, sin = np.cos(b), np.sin(b)
    target = -(cos * bx + sin * by)
    yy, xx = np.mgrid[0:h, 0:w] * step
    keep = ~np.isnan(target)
    for x, y in singular:
        keep &= np.hypot(xx - x, yy - y) > clear * period
    keep[[0, -1], :] = False
    keep[:, [0, -1]] = False
    i, j = np.nonzero(keep)
    n = len(i)
    e = np.arange(n)
    rows = np.tile(e, 4)
    cols = np.concatenate([idx[i, j + 1], idx[i, j - 1], idx[i + 1, j], idx[i - 1, j]])
    fx, fy = sin[i, j] / (2 * step), -cos[i, j] / (2 * step)
    vals = np.concatenate([fx, -fx, fy, -fy])
    transport = sp.csr_matrix((vals, (rows, cols)), shape=(n, h * w))
    root = np.sqrt(smoothness) / step
    right = sp.csr_matrix(
        (
            np.r_[np.full((w - 1) * h, root), np.full((w - 1) * h, -root)],
            (np.r_[np.arange((w - 1) * h), np.arange((w - 1) * h)], np.r_[idx[:, 1:].ravel(), idx[:, :-1].ravel()]),
        ),
        shape=((w - 1) * h, h * w),
    )
    down = sp.csr_matrix(
        (
            np.r_[np.full((h - 1) * w, root), np.full((h - 1) * w, -root)],
            (np.r_[np.arange((h - 1) * w), np.arange((h - 1) * w)], np.r_[idx[1:, :].ravel(), idx[:-1, :].ravel()]),
        ),
        shape=((h - 1) * w, h * w),
    )
    a = sp.vstack([transport, right, down]).tocsc()
    rhs = np.r_[target[i, j], np.zeros(a.shape[0] - n)]
    u = spla.spsolve((a.T @ a + 1e-9 * sp.identity(h * w)).tocsc(), a.T @ rhs).reshape(h, w)
    u = cv2.resize((u - u.mean()).astype(np.float32), beta_u.shape[::-1], interpolation=cv2.INTER_CUBIC)
    return period * np.exp(-np.clip(u, -bound, bound))


def flow_spirals(beta_u, period, barrier, rng, edge=30, smooth=0.0, spacing=0.0, dither=False):
    """The net spirals the flow needs: curl of G / 2π per pixel (computed within sheets), drawn as a Poisson field
    of ±1 spirals."""
    k = 2 * np.pi / period
    gx, gy = k * np.cos(beta_u), k * np.sin(beta_u)
    # curl = ∂gy/∂x − ∂gx/∂y, by differences within a sheet (seam pairs dropped).
    dgy = np.diff(gy, axis=1)[:-1, :]
    dgx = np.diff(gx, axis=0)[:, :-1]
    ok = (np.abs(np.diff(beta_u, axis=1))[:-1, :] < np.pi / 2) & (np.abs(np.diff(beta_u, axis=0))[:, :-1] < np.pi / 2)
    rho = np.where(ok, dgy - dgx, 0) / (2 * np.pi)
    if dither:
        h, w = rho.shape
        rho[:edge], rho[-edge:], rho[:, :edge], rho[:, -edge:] = 0, 0, 0, 0
        return dithered(rho, beta_u, float(np.mean(period)), rng), float(rho.sum()), float(np.abs(rho).sum())
    if smooth:  # the net over a few periods: ± fluctuations that cancel within that need no spirals
        rho = cv2.GaussianBlur(rho.astype(np.float32), (0, 0), smooth * float(np.mean(period))).astype(np.float64)
    h, w = rho.shape
    rho[:edge], rho[-edge:], rho[:, :edge], rho[:, -edge:] = 0, 0, 0, 0
    spirals = []
    period_px = float(np.mean(period))
    for sign in (1, -1):
        lam = np.clip(sign * rho, 0, None)
        n = rng.poisson(lam.sum())
        if not n:
            continue
        # Drawn from the intensity, but none within `spacing` periods of another of either sign (redrawn, up to
        # 50 tries each): the flow's minutiae spread out rather than clump where its demand peaks.
        p = (lam / lam.sum()).ravel()
        placed, tries = 0, 0
        while placed < n and tries < 50 * n:
            tries += 1
            y, x = np.unravel_index(rng.choice(lam.size, p=p), lam.shape)
            x, y = float(x) + rng.random(), float(y) + rng.random()
            if spacing and spirals and np.min(np.hypot(*(np.array([(a, b) for a, b, _ in spirals]) - (x, y)).T)) < spacing * period_px:
                continue
            spirals.append((x, y, sign))
            placed += 1
    return spirals, float(rho.sum()), float(np.abs(rho).sum())


LAYOUTS = ("across", "diagonal", "along", "free")
FREE_SEPARATION = (1.0, 4.5)  # periods: real facing pairs lie 5-40 px apart, median 27 (E-019), two ridges or more


def placement(shape, cores, clump, below, seed, fgp):
    """Where pairs land, up to 1: regional clumping (log-normal over smooth noise at `CLUMP_SCALE` px, as real
    minutiae cluster by region, Zhu, Dass & Jain 2007) and `below` times as many below the top core as above it."""
    from ridgegen.synthesis import rng_for, smooth_noise

    h, w = shape
    weight = np.ones(shape)
    if clump:
        noise = smooth_noise(shape, rng_for(seed, fgp, CLUMP_SALT), CLUMP_SCALE, 1.0)
        weight *= np.exp(clump * noise / (noise.std() or 1.0))
    if below != 1.0 and cores:
        top = min(c[1] for c in cores)
        yy = np.arange(h)[:, None]
        weight = weight * np.where(yy >= top, below, 1.0)
    return weight / weight.max()


def pairs(beta_u, period, spirals, singular, rate, mix, rng, edge=30, labels=None, where=None, free=FREE_SEPARATION):
    """`rate` dipoles per 100 mm² of master: + and − across the ridges one period apart (an ending facing a fork),
    on a diagonal (1-1.5 periods: two forks, or a fork and an ending), along a ridge (1.5-3: a break) or free (any
    direction, `FREE_SEPARATION` periods apart), in the shares `mix`; uniform, the centre at least 2 periods from any
    other spiral and singular point (a free pair's ends 1.5 each). Each pair's layout is appended to `labels` if
    given; `where`, up to 1 at each pixel, thins where they land (`placement`)."""
    h, w = beta_u.shape
    area = (w - 2 * edge) * (h - 2 * edge) * (25.4 / 500) ** 2
    wanted = rng.poisson(rate * area / 100)
    taken = [(x, y) for x, y, _ in spirals] + list(singular)
    out = []
    for _ in range(wanted * (20 if where is None else 200)):
        if len(out) >= 2 * wanted:
            break
        x, y = rng.uniform(edge, w - edge), rng.uniform(edge, h - edge)
        if where is not None and rng.random() > where[int(y), int(x)]:
            continue
        if taken and np.min(np.hypot(*(np.array(taken) - (x, y)).T)) < 2 * period:
            continue
        beta = beta_u[int(y), int(x)]  # across the ridges
        mix4 = tuple(mix) + (0.0,) * (4 - len(mix))
        layout = rng.choice(4, p=mix4)
        a, sep = [
            (beta, 1.0),
            (beta + rng.choice([-1, 1]) * np.pi / 4, rng.uniform(1.0, 1.5)),
            (beta + np.pi / 2, rng.uniform(1.5, 3.0)),
            (rng.uniform(0, np.pi), rng.uniform(*free)),
        ][layout]
        d = sep * period * np.array([np.cos(a), np.sin(a)]) / 2
        sign = rng.choice([-1, 1])
        ends = [(x + d[0], y + d[1]), (x - d[0], y - d[1])]
        if LAYOUTS[layout] == "free":  # its ends can be far from the centre: each clear of the rest, and inside
            if any(not (edge <= ex < w - edge and edge <= ey < h - edge) for ex, ey in ends):
                continue
            if taken and min(np.min(np.hypot(*(np.array(taken) - e).T)) for e in ends) < 1.5 * period:
                continue
            taken += ends
        out += [(ends[0][0], ends[0][1], sign), (ends[1][0], ends[1][1], -sign)]
        taken.append((x, y))
        if labels is not None:
            labels.append(LAYOUTS[layout])
    return out


def master_ridges(
    m,
    seed,
    fgp,
    smooth=2.0,
    rate=0.0,
    mix=(0.5, 0.3, 0.2),
    spacing=0.0,
    clump=0.0,
    below=1.0,
    dither=False,
    spacing_weight=None,
    vary=None,
    free=FREE_SEPARATION,
):
    """A master's ridges drawn from one phase: the flow's spirals (density smoothed over `smooth` periods) and
    `rate` added pairs per 100 mm²; and the spirals. `vary` is (smoothness, bound) for `period_field`."""
    from ridgegen.synthesis import rng_for

    shape, period = m.ridges.shape, m.period
    singular = m.cores + m.deltas
    barrier = cuts(shape, singular)
    beta_u = unwrap_direction(m.orientation, barrier)
    field = period_field(beta_u, period, singular, *vary) if vary else period
    spirals, _, _ = flow_spirals(beta_u, field, barrier, rng_for(seed, fgp, FLOW_SALT), smooth=smooth, spacing=spacing, dither=dither)
    if rate:
        where = placement(shape, m.cores, clump, below, seed, fgp) if clump or below != 1.0 else None
        spirals += pairs(beta_u, float(np.mean(field)), spirals, singular, rate, mix, rng_for(seed, fgp, PAIR_SALT), where=where, free=free)
    psi_c, psi_s, _ = solve_continuous(beta_u, field, spirals, spacing_weight=spacing_weight)
    return render(psi_c + psi_s), spirals


def render(psi):
    return np.tanh(SATURATION * np.cos(psi)).astype(np.float32)


def match(found, spirals, singular, barrier, period):
    """Found minutiae near a placed spiral (within 0.7 periods), and the rest by where they are."""
    placed = np.array([(x, y) for x, y, _ in spirals]).reshape(-1, 2)
    near_sp = near_cut = elsewhere = matched = 0
    for p in found:
        if len(placed) and np.min(np.hypot(placed[:, 0] - p["x"], placed[:, 1] - p["y"])) < 0.7 * period:
            matched += 1
        elif any(np.hypot(p["x"] - x, p["y"] - y) < 4 * period for x, y in singular):
            near_sp += 1
        elif barrier[:, max(0, p["x"] - 6) : p["x"] + 7].any():
            near_cut += 1
        else:
            elsewhere += 1
    return {"found": len(found), "matched": matched, "spurious_near_singular": near_sp, "spurious_near_cut": near_cut, "spurious_elsewhere": elsewhere}


def one(seed, fgp, out_images=None):
    from ridgegen import finger as fingers
    from ridgegen import minutiae
    from ridgegen.synthesis import rng_for

    growth.use("base")
    m = fingers.master(seed, fgp)
    shape, period = m.ridges.shape, m.period
    singular = m.cores + m.deltas
    barrier = cuts(shape, singular)
    beta_u = unwrap_direction(m.orientation, barrier)
    rng = rng_for(seed, fgp, FLOW_SALT)
    flow, net, gross = flow_spirals(beta_u, period, barrier, rng)
    rows = {
        "seed": seed,
        "fgp": fgp,
        "period": period,
        "pattern": m.pattern,
        "flow_net": net,
        "flow_gross": gross,
        "grown": len(minutiae.extract(m.ridges > 0, np.ones(shape, bool))),
    }
    images = {}
    for name, spirals in (("none", []), ("flow", flow)):
        psi_c, psi_s, seconds = solve_continuous(beta_u, period, spirals)
        ridges = render(psi_c + psi_s)
        found = minutiae.extract(ridges > 0, np.ones(shape, bool))
        rows[name] = {**match(found, spirals, singular, barrier, period), "placed": len(spirals), "solve_s": round(seconds, 1)}
        images[name] = ridges
    if out_images:
        np.save(out_images, np.stack([m.ridges, images["none"], images["flow"]]))
    return rows


if __name__ == "__main__":
    from signed_charge.measure import provenance

    out = sys.argv[1]
    started = provenance.start()
    results = []
    for seed, fgp in ((3000, 2), (3001, 8)):
        r = one(seed, fgp, out.replace(".json", f"_{seed}_{fgp}.npy"))
        print(json.dumps(r), flush=True)
        results.append(r)
    json.dump({"_provenance": provenance.record(started), "results": results}, open(out, "w"), indent=1)
