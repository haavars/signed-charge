"""Glover et al. 2023's Turing model of ridge formation, reimplemented (E-028): glover_model.py
PATTERN OUT.npz [--dx 2.0] [--dt 0.25] [--t 2500]

Their mathematical appendix (`glover_cell2023_model_details.pdf`, the paper's supplementary material; their code is COMSOL files,
which we can't run) gives a saturated Gierer-Meinhardt system on a fingertip:

    u_t = r (a + u² / (v (1 + K u²))) − μ u + ∇²u        (1)
    v_t = u² − ν v + D ∇²v                               (2)

The domain is a hemicylinder of height L and radius R = 2L/5 (y > 0) topped by a spherical quadrant. Stripe fronts start
from elevated activator inside (eqs 8-10, whorl and loop; none for an arch) and from boundaries (the nail bed and the
crease) where a time-dependent condition (13) turns on after T_a and T_b. Elsewhere no flux.

Here: a triangle mesh from a (φ, s) grid, s along the meridian, the cotangent Laplacian with lumped mass, and a
semi-implicit step with the diffusion and linear decay implicit. Two departures from the appendix, both to check against
its Figure Ax2:
- Eq 13's Robin condition is applied as a relaxation of boundary u to zero at rate A/(1 − A), not as a Robin condition
  (it says the pattern is insensitive to the details of this condition).
- It doesn't give where ∂Ω_S1 (the nail bed's side) ends or how far the arch's restricted ∂Ω_T reaches: `NAIL_FROM` and
  `ARCH_TOP` are our reading of the figure.

Run: python -m signed_charge.generate.glover_model
loop OUT.npz [--sigma S --scale PX --seed N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from signed_charge.measure import provenance

L, R = 600.0, 240.0
A_, K = 0.001, 0.001
D, r_, MU, NU = 180.0, 0.75, 0.4, 5.5
NAIL_FROM = 0.65  # of L: the lateral boundary above this is the nail bed's side (∂Ω_S1)
ARCH_TOP = 0.5  # of the top's arc around the tip: the arch's restricted ∂Ω_T, about the tip
GATE = 0.3  # of u*: departure from the steady state that marks the front's passing
TIMING = {"whorl": (50.0, 50.0), "loop": (50.0, 50.0), "arch": (100.0, 0.0)}  # T_a (nail bed), T_b (crease)


def steady_state():
    roots = np.roots([K, -A_ * K * r_ / MU, 1.0, -r_ * (NU + A_) / MU])
    u = float(roots[np.abs(roots.imag) < 1e-9].real.max())
    return u, u * u / NU


def mesh(dx):
    """Vertices (x, y, z), triangles and the grid they came from: φ across (0-π), s along the meridian from the crease
    (s = 0) over the shoulder to the top pole, which is one vertex."""
    s_max = L + R * np.pi / 2
    ns, nphi = int(round(s_max / dx)) + 1, int(round(np.pi * R / dx)) + 1
    s = np.linspace(0, s_max, ns)
    phi = np.linspace(0, np.pi, nphi)
    alpha = np.clip((s - L) / R, 0, np.pi / 2)
    ring = R * np.cos(alpha)
    z = np.where(s <= L, s, L + R * np.sin(alpha))
    x = np.outer(ring, np.cos(phi))
    y = np.outer(ring, np.sin(phi))
    zz = np.repeat(z[:, None], nphi, 1)
    index = np.arange(ns * nphi).reshape(ns, nphi)
    index[-1, :] = index[-1, 0]  # the pole
    a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
    tris = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
    points = np.stack([x.ravel(), y.ravel(), zz.ravel()], 1)
    # Only the vertices triangles use: the pole's row is one vertex.
    used, tris = np.unique(tris, return_inverse=True)
    tris = tris.reshape(-1, 3)
    remap = np.full(ns * nphi, -1)
    remap[used] = np.arange(len(used))
    return points[used], tris, s, phi, remap[index]


def laplacian(points, tris):
    """The cotangent stiffness matrix and lumped mass."""
    n = len(points)
    p = points[tris]
    rows, cols, vals = [], [], []
    mass = np.zeros(n)
    for k in range(3):
        i, j, o = tris[:, k], tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
        u, v = p[:, k] - p[:, (k + 2) % 3], p[:, (k + 1) % 3] - p[:, (k + 2) % 3]
        cross = np.linalg.norm(np.cross(u, v), axis=1)
        cot = np.sum(u * v, 1) / np.maximum(cross, 1e-12)
        w = 0.5 * cot
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]
        np.add.at(mass, o, cross / 6)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)), mass


def bump(x):
    out = np.zeros_like(x)
    inside = np.abs(x) < 1
    out[inside] = np.exp(1 - 1 / (1 - x[inside] ** 2))
    return out


def initial(points, pattern, rng=None, jitter=0.0):
    y, x, z = points.T * np.array([[1], [-1], [1]])  # into Glover's frame, whose hemicylinder is x < 0 (notebook E-051)
    ustar, vstar = steady_state()
    u = np.full(len(points), ustar)
    if pattern == "whorl":
        u = ustar * (1 + 0.1 * bump(10 * np.sqrt(((x + R) / R) ** 2 + (y / R) ** 2 + ((z - 0.8 * L) / R) ** 2)))
    elif pattern == "loop":
        a = 3 * z / (2 * L)
        u = ustar * (1 + 0.1 * bump(10 * np.sqrt(((x + R * np.sin(a)) / R) ** 2 + ((y - R * np.cos(a)) / R) ** 2)))
    return u, np.full(len(points), vstar)


def boundaries(points, s, phi, index, pattern):
    """Vertex sets of the crease and the nail bed."""
    crease = np.unique(index[0, :])
    side = np.unique(np.concatenate([index[s >= NAIL_FROM * L, 0], index[s >= NAIL_FROM * L, -1]]))
    if pattern == "arch":
        keep = s >= L + (1 - ARCH_TOP) * (s[-1] - L)
        nail = np.unique(np.concatenate([index[keep, 0], index[keep, -1]]))
    else:
        nail = side
    return crease, nail


def ramp(t, start):
    return float(np.clip((t - start) / 100, 0, 1))


def variation(points, sigma, scale, seed):
    """Smooth multiplicative noise in activator production (EDAR, the model's r), 1 ± `sigma`, at `scale` units along
    the surface: a nearest-grid sample of a smooth 3D field, so it doesn't see the mesh's seams."""
    if not sigma:
        return 1.0
    from scipy.ndimage import gaussian_filter, map_coordinates

    rng = np.random.default_rng(seed)
    lo = points.min(0) - 2 * scale
    shape = np.ceil((points.max(0) - lo) / (scale / 2)).astype(int) + 4
    field = gaussian_filter(rng.standard_normal(shape), 2.0, mode="wrap")
    field /= field.std()
    return 1 + sigma * map_coordinates(field, ((points - lo) / (scale / 2)).T, order=1, mode="nearest")


def run(pattern, dx, dt, t_end, snapshots=(150, 300, 450, 900, 2500), sigma=0.0, scale=40.0, seed=0, gated=True):
    points, tris, s, phi, index = mesh(dx)
    # Variation in r takes the state off its Turing-unstable steady state, so applied everywhere it starts the
    # pattern everywhere at once and the initiation sites hardly matter (E-030). `gated`: only where the front has
    # reached, so the state ahead stays steady and the waves lay the ridges.
    noise = variation(points, sigma, scale, seed) - 1.0
    ustar = steady_state()[0]
    stiff, mass = laplacian(points, tris)
    n = len(points)
    m = sp.diags(mass)
    lu_u = spla.splu((m * (1 + dt * MU) + dt * stiff).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                     options=dict(SymmetricMode=True))
    lu_v = spla.splu((m * (1 + dt * NU) + dt * D * stiff).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                     options=dict(SymmetricMode=True))
    u, v = initial(points, pattern)
    reached = np.zeros(n, bool) if gated else np.ones(n, bool)
    crease, nail = boundaries(points, s, phi, index, pattern)
    ta, tb = TIMING[pattern]
    frames, t, step = {}, 0.0, 0
    started = time.time()
    while t < t_end:
        if gated:  # one way, on once the front has clearly passed: exactly zero ahead of it (E-031)
            reached = reached | (np.abs(u - ustar) > GATE * ustar)
        production = (1 + noise * reached) * r_ * (A_ + u * u / (np.maximum(v, 1e-9) * (1 + K * u * u)))
        u = lu_u.solve(mass * (u + dt * production))
        v = lu_v.solve(mass * (v + dt * u * u))
        t += dt
        step += 1
        for where, start in ((nail, ta), (crease, tb)):
            a = ramp(t, start)
            if a >= 1:
                u[where] = 0.0
            elif a > 0:
                u[where] *= np.exp(-dt * a / (1 - a))
        if any(abs(t - f) < dt / 2 for f in snapshots):
            frames[int(round(t))] = u[index].astype(np.float32)
            print(json.dumps({"t": round(t), "u_max": float(u.max()), "u_min": float(u.min()),
                              "seconds": round(time.time() - started)}), flush=True)
    return {"s": s, "phi": phi, "frames": frames, "n": n, "u": u[index].astype(np.float32)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", choices=sorted(TIMING))
    parser.add_argument("out")
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--t", type=float, default=2500)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--everywhere", action="store_true", help="E-029's noise, not gated on the front")
    args = parser.parse_args()
    started = provenance.start()
    result = run(args.pattern, args.dx, args.dt, args.t, sigma=args.sigma, scale=args.scale, seed=args.seed,
                 gated=not args.everywhere)
    np.savez_compressed(args.out, s=result["s"], phi=result["phi"], u=result["u"],
                        **{f"t{k}": v for k, v in result["frames"].items()},
                        provenance=json.dumps(provenance.record(started)))


if __name__ == "__main__":
    main()
