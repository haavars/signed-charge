"""Glover's model behind a wave of competency, with noise (E-033):
competency_model.py PATTERN OUT.npz --speed A [--noise N] [--sigma S] [--seed K] [--t T]

Glover's steady state is Turing-unstable, so with any noise the pattern forms everywhere at once (E-030, E-032). Their data
show waves of signalling ahead of the ridges, and Liu, Maini & Baker 2022 show that patterning behind such a wave survives
noise. Here cells react only once the wave has reached them:

    u_t = c(x, t) f(u, v) + ∇²u + noise,    v_t = c(x, t) g(u, v) + D ∇²v

with f and g the kinetics of Glover's eqs 1-2, c = 0 ahead of the wave and 1 behind it (raising activator decay, as Liu's
W does, makes this model more unstable: notebook E-033). Ahead of the wave the chemicals only diffuse: nothing restores u*,
so the noise wanders, by about 1 % of u* at the default strength.

- **The wave**: arrival time T(x) = min over sites (t_site + distance(x, site) / speed), on the mesh. The sites are Glover's:
  the pad (their eqs 9-10's regions, at t = 0), the nail bed at T_a, the crease at T_b. Distances are shortest paths on a
  16-neighbour graph of the (s, φ) grid, which keeps fronts round to a few per cent.
- **Noise**: everywhere from t = 0: `noise` × u* × √dt, white, added to u each step, and `sigma` smooth variation in r
  (`glover_model.variation`) at `scale`.
- No Dirichlet boundary here: the sites start the wave, and everything else is no-flux.

Run: python -m signed_charge.generate.competency_model loop OUT.npz --speed 1.0
"""

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra

from signed_charge.generate import glover_model as g
from signed_charge.measure import provenance

RAMP = 10.0  # time units over which a cell's reactions turn on
OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (1, -2), (2, 1), (2, -1))
PERSON = (50.0, 0.15, 25.0, 25.0)  # sd of the pad along the finger (units) and round it (rad), and of the nail's and crease's delays


def graph(points, index):
    """Edges of the 16-neighbour graph on the (s, φ) grid, weighted by distance; the pole's merged row is skipped."""
    ns, nphi = index.shape
    rows, cols = [], []
    for di, dj in OFFSETS:
        a = index[max(0, -di) : ns - max(0, di), max(0, -dj) : nphi - max(0, dj)]
        b = index[max(0, di) : ns - max(0, -di), max(0, dj) : nphi - max(0, -dj)]
        rows.append(a.ravel())
        cols.append(b.ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    weights = np.linalg.norm(points[rows] - points[cols], axis=1)
    return rows, cols, weights


def site_vertices(points, index, s, pattern, person=(0.0, 0.0, 0.0, 0.0)):
    """The sites, (vertices, delay), nail bed, crease and pad. `person` is one finger's departure from Glover's geometry: the pad along the finger and
    round it, and the nail bed's and crease's delays."""
    along, around, nail_delay, crease_delay = person
    y0, x0, z0 = points.T * np.array([[1], [-1], [1]])  # into Glover's frame, whose hemicylinder is x < 0: the ventral midline is (-R, 0), our (0, R) (E-051)
    x = x0 * np.cos(around) + y0 * np.sin(around)
    y = -x0 * np.sin(around) + y0 * np.cos(around)
    z = z0 + along
    crease, nail = g.boundaries(points, s, None, index, pattern)
    ta, tb = g.TIMING[pattern]
    ta, tb = max(ta + nail_delay, 0.0), max(tb + crease_delay, 0.0)
    sites = [(nail, ta), (crease, tb)]
    if pattern == "whorl":
        pad = g.bump(10 * np.sqrt(((x + g.R) / g.R) ** 2 + (y / g.R) ** 2 + ((z - 0.8 * g.L) / g.R) ** 2)) > 0
        sites.append((np.nonzero(pad)[0], 0.0))
    elif pattern == "loop":
        a = 3 * z / (2 * g.L)
        pad = g.bump(10 * np.sqrt(((x + g.R * np.sin(a)) / g.R) ** 2 + ((y - g.R * np.cos(a)) / g.R) ** 2)) > 0
        sites.append((np.nonzero(pad)[0], 0.0))
    return sites


def site_arrivals(points, index, s, pattern, speed, person=(0.0, 0.0, 0.0, 0.0)):
    """Each site's wave alone: arrival times, one row a site, from a virtual node joined to the site's vertices by its delay."""
    n = len(points)
    rows, cols, weights = graph(points, index)
    times = []
    for vertices, delay in site_vertices(points, index, s, pattern, person):
        r = np.concatenate([rows, np.full(len(vertices), n)])
        c = np.concatenate([cols, vertices])
        w = np.concatenate([weights, np.full(len(vertices), delay * speed + 1e-6)])
        times.append(dijkstra(sp.csr_matrix((w, (r, c)), shape=(n + 1, n + 1)), directed=False, indices=n)[:n] / speed)
    return np.array(times)


def guide(stiff, sites, pattern):
    """A smooth field from the finger's geometry, in the spirit of Kücken & Newell 2005: harmonic, 0 along the crease and nail bed, 1 at the pad (for an arch, 1 at
    the nail bed), free at the lateral edge, so its level sets run along the creases and meet the free edge square. Ridges are to follow them."""
    n = stiff.shape[0]
    psi, fixed = np.zeros(n), np.zeros(n, bool)
    for i, (vertices, _) in enumerate(sites):
        fixed[vertices] = True
        psi[vertices] = (i == 0) if pattern == "arch" else (i == 2)
    free = ~fixed
    stiff = stiff.tocsr()
    psi[free] = spla.spsolve(stiff[free][:, free].tocsc(), -stiff[free][:, fixed] @ psi[fixed])
    return psi


def anisotropy(points, tris, psi, epsilon):
    """The extra stiffness of diffusion stronger by `epsilon` along the level sets of ψ: ∫ε(d·∇φ_i)(d·∇φ_j), faded out where ∇ψ vanishes, at the guide's own
    singular points, where a direction means nothing."""
    p = points[tris]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    twice = np.linalg.norm(normal, axis=1)
    normal = normal / twice[:, None]
    edges = [p[:, 2] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 0]]
    grads = np.stack([np.cross(normal, e) / twice[:, None] for e in edges], 1)  # ∇φ_k, each triangle
    gradient = np.sum(psi[tris][:, :, None] * grads, 1)
    size = np.linalg.norm(gradient, axis=1)
    along = np.cross(normal, gradient) / np.maximum(size, 1e-12)[:, None]
    weight = epsilon * size / (size + 0.25 * np.median(size)) * twice / 2
    c = np.sum(along[:, None] * grads, 2)
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    vals = (weight[:, None, None] * c[:, :, None] * c[:, None, :]).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(points),) * 2)


def arrival(points, index, s, pattern, speed, person=(0.0, 0.0, 0.0, 0.0), soft=0.0):
    """T(x), the wave's arrival time: the earliest site's, or with `soft` a smooth minimum over that many time units, so where two waves
    meet their fronts bend into each other rather than meeting at a kink (E-049)."""
    times = site_arrivals(points, index, s, pattern, speed, person)
    if not soft:
        return times.min(0)
    first = times.min(0)
    return np.maximum(first - soft * np.log(np.exp(-(times - first) / soft).sum(0)), 0.0)


def person_of(seed, vary):
    """One finger's geometry: normal, within 2.5 sd, `vary` × `PERSON`."""
    return tuple(float(v) for v in vary * np.array(PERSON) * np.clip(np.random.default_rng([seed, 7]).standard_normal(4), -2.5, 2.5))


def run(pattern, speed, noise, sigma, scale, seed, dx, dt, t_end, vary=0.0, soft=0.0, aniso=0.0):
    points, tris, s, phi, index = g.mesh(dx)
    stiff, mass = g.laplacian(points, tris)
    n = len(points)
    m = sp.diags(mass)
    solve = lambda a: spla.splu(a.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    stiff_v = stiff
    if aniso:
        sites = site_vertices(points, index, s, pattern, person_of(seed, vary))
        stiff_v = stiff + anisotropy(points, tris, guide(stiff, sites, pattern), aniso)
    lu_u, lu_v = solve(m + dt * stiff), solve(m + dt * g.D * stiff_v)
    rng = np.random.default_rng(seed)
    ustar, vstar = g.steady_state()
    u, v = np.full(n, ustar), np.full(n, vstar)
    production_scale = g.variation(points, sigma, scale, seed)
    t_arrive = arrival(points, index, s, pattern, speed, person_of(seed, vary), soft)
    frames, t, started = {}, 0.0, time.time()
    snapshots = (150, 300, 450, 900, 1800)
    while t < t_end:
        c = np.clip((t - t_arrive) / RAMP, 0, 1)
        f = production_scale * g.r_ * (g.A_ + u * u / (np.maximum(v, 1e-9) * (1 + g.K * u * u))) - g.MU * u
        h = u * u - g.NU * v
        u1 = u + dt * c * f + noise * ustar * np.sqrt(dt) * rng.standard_normal(n)
        v1 = v + dt * c * h
        u = lu_u.solve(mass * u1)
        v = lu_v.solve(mass * v1)
        t += dt
        if any(abs(t - f_) < dt / 2 for f_ in snapshots):
            frames[int(round(t))] = u[index].astype(np.float32)
            print(json.dumps({"t": round(t), "u_max": float(u.max()), "seconds": round(time.time() - started)}), flush=True)
    return {"s": s, "phi": phi, "frames": frames, "u": u[index].astype(np.float32), "arrival": t_arrive[index].astype(np.float32)}


def main():
    global RAMP
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", choices=sorted(g.TIMING))
    parser.add_argument("out")
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--K", type=float, default=g.K)
    parser.add_argument("--ramp", type=float, default=RAMP)
    parser.add_argument("--vary", type=float, default=0.0)
    parser.add_argument("--soft", type=float, default=0.0)
    parser.add_argument("--aniso", type=float, default=0.0)
    parser.add_argument("--t", type=float, default=900)
    args = parser.parse_args()
    started = provenance.start()
    g.K, RAMP = args.K, args.ramp
    r = run(args.pattern, args.speed, args.noise, args.sigma, args.scale, args.seed, args.dx, args.dt, args.t, args.vary, args.soft, args.aniso)
    np.savez_compressed(
        args.out,
        s=r["s"],
        phi=r["phi"],
        u=r["u"],
        arrival=r["arrival"],
        **{f"t{k}": v for k, v in r["frames"].items()},
        provenance=json.dumps(provenance.record(started)),
    )


if __name__ == "__main__":
    main()
