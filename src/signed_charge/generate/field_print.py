"""A planar print: fronts lay the ridges, a cue strong enough to win sets their direction (D-047): field_print.py OUT.npz --flow SOURCE [--eps E] [--speed A] [--front FRONT] [--seed K]

`flat_seams`'s kinetics, gate, noise and variation on a periodic square of `N` cells, with inhibitor diffusion D(1 + `eps`) along a given flow, `eps` negative, as stripes then run along it (Shoji et al. 2002), so the cue is the orientation field itself and not a uniform direction. The flow is smoothed as a double angle, so at a core or delta the cue fades to isotropic by itself.

    --flow model2:SEED:FINGER   ridgegen's master for that person and finger
    --flow real:PATH            the orientation read from a real print on FVC B (not for figures, D-004)
    --flow read:SEED:FINGER     the same read of model 2's own print of that finger, capture 1 (D-055)
                                A read flow carries each minutia's own bend, so a front could copy the print back: `--smooth` px takes it out, as a double angle (E-084).
    --flow uniform:DEGREES      calibration against `flat_seams`, degrees from y

    --front base                a straight front from the bottom of the sheet
    --front ring                a ring from the flow's first core
    --front sites               Glover's three: ring from the core, line from the base, line from the tip, the first to arrive

Run: python -m signed_charge.generate.field_print OUT.npz --flow model2:9000:7
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.fft as sfft
from scipy.ndimage import distance_transform_edt, gaussian_filter

from signed_charge.generate import glover_model as g
from signed_charge.measure import provenance

N, DT = 432, 0.1
DX = 2.0  # units a cell: the pattern's period is 18 units, so 2 is 9 px, and `--period` scales it
RELAX, BLUR = 300.0, 3.0
SMOOTH = 0.0


def crop(theta, centre):
    """`N` square of a flow, its centre at `centre`, padded by the edge's own values."""
    cy, cx = int(centre[0]), int(centre[1])
    top, left = cy - N // 2, cx - N // 2
    pad = max(0, -top, -left, top + N - theta.shape[0], left + N - theta.shape[1])
    wide = cv2.copyMakeBorder(theta.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return wide[top + pad : top + pad + N, left + pad : left + pad + N]


def model2(seed, finger):
    from ridgegen import finger as fingers

    m = fingers.master(seed, finger)
    core = m.cores[0] if len(m.cores) else (m.orientation.shape[1] / 2, m.orientation.shape[0] / 2)
    return crop(m.orientation, (core[1], core[0])), (N / 2, N / 2), m.period


def read(im):
    """A print's read flow, centred on its topmost core or else its centroid, with each cell off the print taking the nearest cell on it."""
    from signed_charge.measure import field_similarity as fs

    fg = fs.foreground(im)
    theta = fs.field(im)
    cores = fs.cores(theta, fg)
    ys, xs = np.nonzero(fg)
    centre = min(((c[1], c[0]) for c in cores), default=(ys.mean(), xs.mean()))
    near = distance_transform_edt(~fg, return_distances=False, return_indices=True)
    theta = theta[near[0], near[1]]
    if SMOOTH:
        c, s = cv2.GaussianBlur(np.cos(2 * theta).astype(np.float32), (0, 0), SMOOTH), cv2.GaussianBlur(np.sin(2 * theta).astype(np.float32), (0, 0), SMOOTH)
        theta = np.arctan2(s, c) / 2 % np.pi
    return crop(theta, centre), (N / 2, N / 2), 9.0


def real(path):
    from signed_charge.measure import conformality as cf
    from signed_charge.measure import field_similarity as fs

    ppi = next((v for k, v in cf.PPI.items() if k in path), 500)
    return read(fs.at_500(cv2.imread(path, 0), ppi))


def read_model2(seed, finger):
    from signed_charge.measure import flow_check as fc

    return read(fc.image(("model2", seed, finger, 1))[1])


def flow_of(source):
    kind, *rest = source.split(":")
    if kind == "model2":
        return model2(int(rest[0]), int(rest[1]))
    if kind == "real":
        return real(rest[0])
    if kind == "read":
        return read_model2(int(rest[0]), int(rest[1]))
    return np.full((N, N), np.pi / 2 - np.deg2rad(float(rest[0])), np.float32), (N / 2, N / 2), 9.0  # degrees from y, the seam's direction in `flat_seams`


def tensor(theta):
    """d dᵀ, from the flow smoothed as a double angle: (Dxx, Dyy, Dxy), in cells of x right and y down."""
    c = gaussian_filter(np.cos(2 * theta), BLUR, mode="wrap")
    s = gaussian_filter(np.sin(2 * theta), BLUR, mode="wrap")
    return (1 + c) / 2, (1 - c) / 2, s / 2


def along(v, xx, yy, xy):
    """∇·(d dᵀ ∇v), periodic, conservative."""
    r = lambda a, k, axis: np.roll(a, k, axis)  # noqa: E731
    fx, fy = (r(v, -1, 1) - v) / DX, (r(v, -1, 0) - v) / DX
    cx, cy = (r(v, -1, 1) - r(v, 1, 1)) / (2 * DX), (r(v, -1, 0) - r(v, 1, 0)) / (2 * DX)
    px = (xx + r(xx, -1, 1)) / 2 * fx + (xy + r(xy, -1, 1)) / 2 * (cy + r(cy, -1, 1)) / 2
    py = (yy + r(yy, -1, 0)) / 2 * fy + (xy + r(xy, -1, 0)) / 2 * (cx + r(cx, -1, 0)) / 2
    return (px - r(px, 1, 1) + py - r(py, 1, 0)) / DX


def arrival(front, core, speed):
    y, x = (np.mgrid[0:N, 0:N] + 0.5) * DX
    extent = N * DX
    cy, cx = core[0] * DX, core[1] * DX
    ring = np.maximum(np.hypot(x - cx, y - cy) - 30.0, 0) / speed
    base, tip = (extent - y) / speed, y / speed
    if front == "base":
        return base
    if front == "ring":
        return ring
    return np.minimum.reduce([ring, base, tip])


def run(theta, core, eps, speed, noise, sigma, scale, seed, front, ramp):
    t_arrive = arrival(front, core, speed).astype(np.float32)
    rng = np.random.default_rng(seed)
    ustar, vstar = g.steady_state()
    u, v = np.full((N, N), ustar, np.float32), np.full((N, N), vstar, np.float32)
    field = gaussian_filter(rng.standard_normal((N, N)), scale / (2 * DX) * 2, mode="wrap")
    production = (1 + sigma * field / field.std()).astype(np.float32)
    ky = 2 * np.pi * sfft.fftfreq(N, DX)
    kx = 2 * np.pi * sfft.rfftfreq(N, DX)
    ky, kx = np.meshgrid(ky, kx, indexing="ij")
    lap = (2 - 2 * np.cos(kx * DX) + 2 - 2 * np.cos(ky * DX)) / DX**2
    implicit_u, implicit_v = (1 / (1 + DT * lap)).astype(np.float32), (1 / (1 + DT * g.D * lap)).astype(np.float32)
    xx, yy, xy = [a.astype(np.float32) for a in tensor(theta)]
    t, t_end = 0.0, float(t_arrive.max()) + RELAX
    while t < t_end:
        c = np.clip((t - t_arrive) / ramp, 0, 1).astype(np.float32)
        f = production * g.r_ * (g.A_ + u * u / (np.maximum(v, 1e-9) * (1 + g.K * u * u))) - g.MU * u
        h = u * u - g.NU * v
        u1 = u + DT * c * f + noise * ustar * np.sqrt(DT) * rng.standard_normal((N, N), np.float32)
        v1 = v + DT * c * h
        u = sfft.irfft2(sfft.rfft2(u1) * implicit_u, s=(N, N))
        # The cue is the last step's: v follows u fast, so the lag costs nothing, and at rest this is the implicit solve's own fixed point.
        v = sfft.irfft2(sfft.rfft2(v1 + (DT * g.D * eps * along(v, xx, yy, xy) if eps else 0.0)) * implicit_v, s=(N, N))
        t += DT
    return u.astype(np.float32), t_arrive.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--flow", required=True)
    parser.add_argument("--eps", type=float, default=-0.3)
    parser.add_argument("--speed", type=float, default=0.7)
    parser.add_argument("--noise", type=float, default=0.003)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--front", default="sites", choices=("base", "ring", "sites"))
    parser.add_argument("--ramp", type=float, default=10.0)
    parser.add_argument("--period", type=float, default=0.0, help="ridge period in px; 0 takes the flow's own")
    parser.add_argument("--smooth", type=float, default=0.0, help="px, on a read flow")
    args = parser.parse_args()
    started = provenance.start()
    global DX, SMOOTH
    SMOOTH = args.smooth
    theta, core, period = flow_of(args.flow)
    DX = 18.0 / (args.period or period)
    u, t_arrive = run(theta, core, args.eps, args.speed, args.noise, args.sigma, args.scale, args.seed, args.front, args.ramp)
    np.savez_compressed(args.out, u=u, arrival=t_arrive, theta=theta, settings=json.dumps(vars(args)), provenance=json.dumps(provenance.record(started)))


if __name__ == "__main__":
    main()
