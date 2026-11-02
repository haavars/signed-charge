"""The competency model with its cue from growth stress, not the harmonic guide (D-027 to D-029): stress_run.py PATTERN OUT.npz --speed A --noise N --sigma S --vary V --seed K --t T [--aniso E] [--core-site]

`competency_model`'s run, but for what sets the anisotropy: ridges are to follow d, the least compressed direction, with strength E × a, a being the stress's own anisotropy. E is -0.2, as a's median is about 0.45, so the
median strength is the -0.1 of E-051.

Kept apart from `competency_model.py` while D-020's population starts new processes from it.

Run: python -m signed_charge.exploratory.stress_run loop OUT.npz --speed 0.7 ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from signed_charge.generate import competency_model as cm
from signed_charge.generate import glover_model as g
from signed_charge.exploratory import growth_stress as gs
from signed_charge.measure import provenance
from signed_charge.exploratory import stress_read as sr


def stiffness(points, tris, pattern, person, epsilon, dx):
    """The extra stiffness ∫ε a (d·∇φ_i)(d·∇φ_j) for the stress on this finger, as `competency_model.anisotropy`."""
    _, _, s, phi, index = g.mesh(dx)
    tensor = np.zeros((len(points), 3, 3))
    tensor[index] = gs.carry(gs.solve(pattern, person, **gs.settings(pattern)), s, phi)
    p = points[tris]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    twice = np.linalg.norm(normal, axis=1)
    normal = normal / twice[:, None]
    d, a, _, _ = gs.along(tensor[tris].mean(1), normal)
    edges = [p[:, 2] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 0]]
    grads = np.stack([np.cross(normal, e) / twice[:, None] for e in edges], 1)
    weight = epsilon * a * twice / 2
    c = np.sum(d[:, None] * grads, 2)
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    vals = (weight[:, None, None] * c[:, :, None] * c[:, None, :]).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(points),) * 2)


def core_arrivals(points, index, s, pattern, speed, person):
    """`competency_model.site_arrivals`, but the pad's wave starts from a disc at the stress field's cores (D-032). The pad's load stays on the band, as `growth_stress` calls `site_vertices` itself."""
    sites = cm.site_vertices(points, index, s, pattern, person)
    report, (_, _, ss, pp) = sr.read(pattern, person, **gs.settings(pattern))
    cores = [(ss[i], pp[k]) for i, k, q, inside in report["points"] if q > 0 and inside]
    if cores:
        s0, p0 = np.mean(cores, axis=0)
        i, j = int(round(s0 / (s[1] - s[0]))), int(round(p0 / np.pi * (index.shape[1] - 1)))
        sites[2] = (np.nonzero(np.linalg.norm(points - points[index[min(i, len(s) - 1), j]], axis=1) < g.R / 10)[0], 0.0)
    n = len(points)
    rows, cols, weights = cm.graph(points, index)
    times = []
    for vertices, delay in sites:
        r = np.concatenate([rows, np.full(len(vertices), n)])
        c = np.concatenate([cols, vertices])
        w = np.concatenate([weights, np.full(len(vertices), delay * speed + 1e-6)])
        times.append(cm.dijkstra(sp.csr_matrix((w, (r, c)), shape=(n + 1, n + 1)), directed=False, indices=n)[:n] / speed)
    return np.array(times)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", choices=sorted(g.TIMING))
    parser.add_argument("out")
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--noise", type=float, default=0.003)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--vary", type=float, default=1.0)
    parser.add_argument("--aniso", type=float, default=-0.2)
    parser.add_argument("--t", type=float, default=900)
    parser.add_argument("--core-site", action="store_true")
    args = parser.parse_args()
    started = provenance.start()
    person = cm.person_of(args.seed, args.vary)
    cm.guide = lambda *a: None
    if args.core_site:
        cm.site_arrivals = core_arrivals
    cm.anisotropy = lambda points, tris, psi, epsilon: stiffness(points, tris, args.pattern, person, epsilon, args.dx)
    r = cm.run(args.pattern, args.speed, args.noise, args.sigma, args.scale, args.seed, args.dx, args.dt, args.t, args.vary, 0.0, args.aniso)
    np.savez_compressed(args.out, s=r["s"], phi=r["phi"], u=r["u"], arrival=r["arrival"], **{f"t{k}": v for k, v in r["frames"].items()}, provenance=json.dumps(provenance.record(started)))


if __name__ == "__main__":
    main()
