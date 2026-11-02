"""The ridges' direction from growth stress, after Kücken & Newell 2005 (D-027, D-028): imported by `stress_read.py`.

A linear membrane of flat facets on Glover's volar surface, their section 6.1: inward line forces across the crease and nail furrow, nothing at the margin, normal and tangential springs, and an inward normal load at the
finger's pad site for the pad's regression:

    K u = F(edges) + F(load),    K = membrane + springs,    N = C ε(u)

Ridges run along the least compressed direction, d, the eigenvector of N's largest eigenvalue. Its weight a = (σ1 − σ2) / (|σ1| + |σ2|) falls to 0 where stress is equal both ways, at cores and deltas.
Solved on a coarse mesh, as the stress is smooth, then carried to the model's mesh through the (s, φ) grid.

Lengths are model units, the membrane's stiffness Et is 1 and the nail's force is 1: only ratios matter.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from signed_charge.generate import competency_model as cm
from signed_charge.generate import glover_model as g
from signed_charge.measure import provenance

POISSON = 0.3
COARSE = 6.0  # model units
SPRING, NORMAL, CREASE, WIDTH = 480.0, 60.0, 0.75, 140.0  # D-029: tangential and normal spring lengths, the crease's force, the pad's width
PADS = {"arch": (0.0, 0.0), "loop": (20.0, 0.04), "whorl": (40.0, 0.1)}  # the pad's height and load


RAMP = float(os.environ.get("STRESS_RAMP", 0.0))  # model units either side of s = L over which curvature rises (E-059); 0 is Glover's jump


def settings(pattern):
    height, load = PADS[pattern]
    return dict(spring=SPRING, normal_spring=NORMAL, crease_force=CREASE, width=WIDTH, height=height, load=load, ramp=RAMP)


def meridian(s, ramp):
    """The surface's profile on an evenly spaced s: radius, height and slope α. Curvature rises smoothly from 0 to 1 / R over L ± ramp, and the tip still closes at s's end, a little off R."""
    x = np.clip((s - (g.L - ramp)) / (2 * ramp), 0, 1)
    step = s[1] - s[0]
    alpha = np.concatenate([[0.0], np.cumsum((x * x * (3 - 2 * x))[:-1] / g.R * step)])
    alpha = alpha * (np.pi / 2) / alpha[-1]
    ring = (np.sin(alpha) * step)[::-1].cumsum()[::-1] - np.sin(alpha[-1]) * step
    return ring, np.concatenate([[0.0], np.cumsum(np.cos(alpha)[:-1] * step)]), alpha


def facets(points, tris):
    """Each triangle's frame e1, e2, its normal, area, and the shape gradients as (triangle, vertex, xyz)."""
    p = points[tris]
    e1 = p[:, 1] - p[:, 0]
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    twice = np.linalg.norm(normal, axis=1)
    normal = normal / twice[:, None]
    e2 = np.cross(normal, e1)
    edges = [p[:, 2] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 0]]
    grads = np.stack([np.cross(normal, e) / twice[:, None] for e in edges], 1)
    return e1, e2, normal, twice / 2, grads


def strain_matrix(e1, e2, grads):
    """B, (triangle, 3, 9): the facet's strains (ε11, ε22, γ12) from its vertices' 3D displacements."""
    g1 = np.sum(grads * e1[:, None], 2)  # (triangle, vertex)
    g2 = np.sum(grads * e2[:, None], 2)
    n = len(e1)
    b = np.zeros((n, 3, 9))
    for k in range(3):
        for axis in range(3):
            b[:, 0, 3 * k + axis] = g1[:, k] * e1[:, axis]
            b[:, 1, 3 * k + axis] = g2[:, k] * e2[:, axis]
            b[:, 2, 3 * k + axis] = g2[:, k] * e1[:, axis] + g1[:, k] * e2[:, axis]
    return b


def elasticity(poisson=POISSON):
    return np.array([[1, poisson, 0], [poisson, 1, 0], [0, 0, (1 - poisson) / 2]]) / (1 - poisson**2)


def vertex_normals(points, tris, normal, area):
    out = np.zeros_like(points)
    for k in range(3):
        np.add.at(out, tris[:, k], normal * area[:, None])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def outward(points, tris):
    """Facets wound outward: the surface's normals point away from the finger's axis."""
    p = points[tris]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    centre = p.mean(1)
    middle = centre * np.array([1, 1, 0])
    middle[:, 2] = np.minimum(centre[:, 2], g.L)
    flip = np.sum(normal * (centre - middle), 1) < 0
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def solve(pattern, person=(0.0, 0.0, 0.0, 0.0), spring=240.0, normal_spring=240.0, load=0.0, width=120.0, height=0.0, crease_force=0.75, ramp=0.0, dx=COARSE):
    """The stress on a coarse mesh: 3D tensors at its vertices, and the mesh's (s, φ) grid. `spring` and `normal_spring` are the lengths √(Et / k) of the springs."""
    points, tris, s, phi, index = g.mesh(dx)
    tris = outward(points, tris)
    n = len(points)
    profile = pad_profile(points, index, s, pattern, person, width) if load or height else np.zeros(n)
    if ramp:
        ring, z, _ = meridian(s, ramp)
        grid = np.stack([np.outer(ring, np.cos(phi)), np.outer(ring, np.sin(phi)), np.repeat(z[:, None], len(phi), 1)], -1)
        points = points.copy()
        points[index] = grid
    if height:
        e1, e2, normal, area, grads = facets(points, tris)
        points = points + height * profile[:, None] * vertex_normals(points, tris, normal, area)
    e1, e2, normal, area, grads = facets(points, tris)
    b = strain_matrix(e1, e2, grads)
    c = elasticity()
    ke = area[:, None, None] * np.einsum("tki,kl,tlj->tij", b, c, b)
    dofs = (3 * tris[:, :, None] + np.arange(3)).reshape(len(tris), 9)
    rows = np.repeat(dofs, 9, axis=1).ravel()
    cols = np.tile(dofs, (1, 9)).ravel()
    stiff = sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(3 * n, 3 * n))

    forces = np.zeros(3 * n)
    crease, nail = g.boundaries(points, s, None, index, "loop")  # the furrow is anatomy, the same whatever the pattern
    ds, dphi = s[1] - s[0], phi[1] - phi[0]
    forces[3 * crease + 2] += crease_force * g.R * dphi  # up the finger; both of the row's ends are half a step, near enough
    for column in (0, -1):
        for i in np.nonzero(s >= g.NAIL_FROM * g.L)[0]:
            forces[3 * index[i, column] + 1] += np.clip((s[i] - g.NAIL_FROM * g.L) / ((1 - g.NAIL_FROM) * g.L), 0, 1) * ds  # the furrow lies in y = 0 and the surface leaves it along +y

    mass = np.zeros(n)
    for k in range(3):
        np.add.at(mass, tris[:, k], area / 3)
    vn = vertex_normals(points, tris, normal, area)
    tangential, perpendicular = 1 / spring**2, 1 / normal_spring**2
    block = mass[:, None, None] * (tangential * np.eye(3)[None] + (perpendicular - tangential) * vn[:, :, None] * vn[:, None, :])
    i = (3 * np.arange(n)[:, None] + np.arange(3)).astype(int)
    springs = sp.csr_matrix((block.ravel(), (np.repeat(i, 3, axis=1).ravel(), np.tile(i, (1, 3)).ravel())), shape=(3 * n, 3 * n))

    forces -= (load * mass * profile)[:, None].repeat(3, 1).ravel() * vn.ravel()
    u = spla.spsolve((stiff + springs).tocsc(), forces)

    stress = np.einsum("tij,tj->ti", b, u[dofs]) @ c.T
    frame = e1[:, :, None] * e1[:, None] * stress[:, 0, None, None] + e2[:, :, None] * e2[:, None] * stress[:, 1, None, None]
    frame = frame + (e1[:, :, None] * e2[:, None] + e2[:, :, None] * e1[:, None]) * stress[:, 2, None, None]
    tensor, weight = np.zeros((n, 3, 3)), np.zeros(n)
    for k in range(3):
        np.add.at(tensor, tris[:, k], frame * area[:, None, None])
        np.add.at(weight, tris[:, k], area)
    return {"tensor": tensor / weight[:, None, None], "index": index, "s": s, "phi": phi, "u": u.reshape(-1, 3), "points": points}


def pad_profile(points, index, s, pattern, person, width):
    """exp(−(d / width)²), d the distance to the finger's pad site: a disc for a whorl, a slanted band for a loop. An arch has none."""
    sites = cm.site_vertices(points, index, s, pattern, person)
    if len(sites) < 3 or not len(sites[2][0]):
        return np.zeros(len(points))
    distance, _ = cKDTree(points[sites[2][0]]).query(points)
    return np.exp(-((distance / width) ** 2))


def carry(result, s, phi):
    """The tensor on another (s, φ) grid: bilinear, each component. Returns (ns, nphi, 3, 3)."""
    ss, pp = np.meshgrid((s - result["s"][0]) / (result["s"][1] - result["s"][0]), (phi - result["phi"][0]) / (result["phi"][1] - result["phi"][0]), indexing="ij")
    grid = result["tensor"][result["index"]]
    return np.stack([np.stack([map_coordinates(grid[:, :, a, b], [ss, pp], order=1, mode="nearest") for b in range(3)], -1) for a in range(3)], -2)


def principal(tensor, e1, e2):
    """In the plane of (e1, e2): σ1 ≥ σ2, and the angle of σ1's direction from e1."""
    a = np.einsum("...i,...ij,...j->...", e1, tensor, e1)
    b = np.einsum("...i,...ij,...j->...", e2, tensor, e2)
    c = np.einsum("...i,...ij,...j->...", e1, tensor, e2)
    middle, radius = (a + b) / 2, np.hypot((a - b) / 2, c)
    return middle + radius, middle - radius, 0.5 * np.arctan2(2 * c, a - b)


def along(tensor, normal):
    """Tensors at points of a surface with this normal: unit d along the least compression, with its weight."""
    seed = np.where(np.abs(normal[..., :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(normal, e1)
    s1, s2, angle = principal(tensor, e1, e2)
    d = np.cos(angle)[..., None] * e1 + np.sin(angle)[..., None] * e2
    return d, (s1 - s2) / np.maximum(np.abs(s1) + np.abs(s2), 1e-12), s1, s2
