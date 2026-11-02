"""Two fronts meeting on a flat sheet (D-045, E-076): flat_seams.py OUT.npz --alpha DEGREES [--speed A] [--seed K] [--soft S] [--ramp R] [--eps E --cue DEGREES] [--disc R --delay T]

`competency_model`'s kinetics, gate, noise and variation on a periodic square of `N` cells of 2 units, by Fourier, without a finger. Two straight fronts cross at `alpha`, the angle between their lines, and meet along x = 0:

    T = (X - |x|) sin(α / 2) / a + y cos(α / 2) / a

so 180 is head-on, and small angles are fronts nearly in line, their kink running up the seam. `delay` makes the left front late, moving the seam, and `disc` swaps the straight pair for a ring from a disc against a straight front, as a pad's wave meets a crease's. `soft` is `competency_model.arrival`'s smooth minimum. `eps` is the cue as there, inhibitor diffusion stronger by that share along direction `cue`, degrees from the seam, x = 0: it's uniform here, with no fade. Edges are margins, read by nothing: the wrap makes a second seam at x = ±X and a start at y = 0.

Run: python -m signed_charge.exploratory.flat_seams OUT.npz --alpha 45
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from signed_charge.generate import glover_model as g
from signed_charge.measure import provenance

N, DX, DT = 384, 2.0, 0.1
RELAX = 300.0


def fronts(alpha, speed, disc, delay):
    """Each front's arrival. With `disc`, as on a finger: a ring from a disc of that radius near the top meets a straight front from below, late by `delay`."""
    y, x = (np.mgrid[0:N, 0:N] - N / 2 + 0.5) * DX
    extent = N * DX / 2
    if disc:
        return np.maximum(np.hypot(x, y - 0.35 * extent) - disc, 0) / speed, (y + extent) / speed + delay
    half = np.deg2rad(alpha) / 2
    return ((extent + x) * np.sin(half) + (y + extent) * np.cos(half)) / speed + delay, ((extent - x) * np.sin(half) + (y + extent) * np.cos(half)) / speed


def arrival(left, right, soft):
    first = np.minimum(left, right)
    if not soft:
        return first
    return np.maximum(first - soft * np.log(np.exp(-(left - first) / soft) + np.exp(-(right - first) / soft)), 0.0)


def seam_distance(left, right, speed):
    """How far each cell is from where the fronts meet, in cells: the gap between them over its gradient."""
    gap = (left - right) * speed
    gy, gx = np.gradient(gap, DX)
    return np.abs(gap) / np.maximum(np.hypot(gx, gy), 1e-3) / DX


def laplacians(eps, cue):
    """Symbols of -∇², and of the cue's -∂²/∂d², on the grid's five-point stencil."""
    ky = 2 * np.pi * np.fft.fftfreq(N, DX)
    kx = 2 * np.pi * np.fft.rfftfreq(N, DX)
    ky, kx = np.meshgrid(ky, kx, indexing="ij")
    lap = (2 - 2 * np.cos(kx * DX) + 2 - 2 * np.cos(ky * DX)) / DX**2
    angle = np.deg2rad(cue)
    along = (np.sin(angle) * kx + np.cos(angle) * ky) ** 2  # the cue's direction, degrees from the seam, which runs along y
    return lap, lap + eps * along


def run(alpha, speed, noise, sigma, scale, seed, soft, ramp, eps, cue, disc=0.0, delay=0.0):
    left, right = fronts(alpha, speed, disc, delay)
    t_arrive = arrival(left, right, soft).astype(np.float64)
    rng = np.random.default_rng(seed)
    ustar, vstar = g.steady_state()
    u, v = np.full((N, N), ustar), np.full((N, N), vstar)
    field = gaussian_filter(rng.standard_normal((N, N)), scale / (2 * DX) * 2, mode="wrap") if sigma else None
    production = 1 + sigma * field / field.std() if sigma else 1.0
    lap_u, lap_v = laplacians(eps, cue)
    implicit_u, implicit_v = 1 / (1 + DT * lap_u), 1 / (1 + DT * g.D * lap_v)
    t, t_end = 0.0, float(t_arrive.max()) + RELAX
    while t < t_end:
        c = np.clip((t - t_arrive) / ramp, 0, 1)
        f = production * g.r_ * (g.A_ + u * u / (np.maximum(v, 1e-9) * (1 + g.K * u * u))) - g.MU * u
        h = u * u - g.NU * v
        u1 = u + DT * c * f + noise * ustar * np.sqrt(DT) * rng.standard_normal((N, N))
        v1 = v + DT * c * h
        u = np.fft.irfft2(np.fft.rfft2(u1) * implicit_u, s=(N, N))
        v = np.fft.irfft2(np.fft.rfft2(v1) * implicit_v, s=(N, N))
        t += DT
    return u.astype(np.float32), t_arrive.astype(np.float32), seam_distance(left, right, speed).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--speed", type=float, default=0.7)
    parser.add_argument("--noise", type=float, default=0.003)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--soft", type=float, default=0.0)
    parser.add_argument("--ramp", type=float, default=10.0)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--cue", type=float, default=0.0)
    parser.add_argument("--disc", type=float, default=0.0)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    started = provenance.start()
    u, t_arrive, seam = run(args.alpha, args.speed, args.noise, args.sigma, args.scale, args.seed, args.soft, args.ramp, args.eps, args.cue, args.disc, args.delay)
    np.savez_compressed(args.out, u=u, arrival=t_arrive, seam=seam, settings=json.dumps(vars(args)), provenance=json.dumps(provenance.record(started)))


if __name__ == "__main__":
    main()
