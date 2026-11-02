"""The natural front speed of Glover's kinetics, by linear theory and in 1D (E-036): front_speed.py OUT.json

Theory: the pulled-front speed, the saddle of the dispersion relation about the Turing-unstable steady state: with
s(q) the growth rate at complex wavenumber q, ds/dq = -iv and Re(s + iqv) = 0. 1D: the same kinetics and semi-implicit
step as competency_model.py, no competency gate and no noise, front started by a bump at one end and followed by the last
point 25 % from u*.

Run: python -m signed_charge.exploratory.front_speed OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import fsolve

from signed_charge.generate import glover_model as g
from signed_charge.measure import provenance

LENGTH, T_END = 2600.0, 1800
TRIALS = ((2.0, 0.1), (1.0, 0.1), (0.5, 0.02))  # dx, dt


def f(u, v):
    return g.r_ * (g.A_ + u * u / (v * (1 + g.K * u * u))) - g.MU * u


def h(u, v):
    return u * u - g.NU * v


def jacobian(u, v, e=1e-6):
    return np.array([[(f(u + e, v) - f(u - e, v)) / (2 * e), (f(u, v + e) - f(u, v - e)) / (2 * e)],
                     [(h(u + e, v) - h(u - e, v)) / (2 * e), (h(u, v + e) - h(u, v - e)) / (2 * e)]])


def pulled(u, v):
    """The saddle: wavenumber k, decay κ and speed v* of the leading edge."""
    jac = jacobian(u, v)

    def s(q):
        w = np.linalg.eigvals(jac - np.diag([1, g.D]) * q * q)
        return w[np.argmax(w.real)]

    def conditions(x):
        q = x[0] + 1j * x[1]
        ds = (s(q + 1e-6) - s(q - 1e-6)) / 2e-6
        return [ds.real, ds.imag + x[2], (s(q) + 1j * q * x[2]).real]

    k, kappa, speed = fsolve(conditions, [0.3, -0.1, 0.8], xtol=1e-11)
    wavenumbers = np.linspace(0.01, 1, 4000)
    growth = np.array([s(q).real for q in wavenumbers])
    return {"k": abs(float(k)), "kappa": abs(float(kappa)), "speed": abs(float(speed)), "max_growth": float(growth.max()),
            "wavelength": float(2 * np.pi / wavenumbers[growth.argmax()]), "front_wavelength": float(2 * np.pi / abs(k))}


def front(ustar, vstar, dx, dt):
    """Position of the front every 100 time units."""
    n = int(LENGTH / dx)
    lap = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="lil")
    lap[0, 0], lap[-1, -1] = -1.0, -1.0
    lap = lap.tocsc() / dx**2
    identity = sp.identity(n, format="csc")
    lu_u, lu_v = spla.splu(identity - dt * lap), spla.splu(identity - dt * g.D * lap)
    u, v = np.full(n, ustar), np.full(n, vstar)
    u[: int(40 / dx)] *= 1.2
    every = int(round(100 / dt))
    positions = []
    for step in range(1, int(T_END / dt) + 1):
        u, v = lu_u.solve(u + dt * f(u, v)), lu_v.solve(v + dt * h(u, v))
        if step % every == 0:
            ahead = np.nonzero(np.abs(u - ustar) > 0.25 * ustar)[0]
            positions.append(float(ahead[-1] * dx) if len(ahead) else 0.0)
    return positions


def main(out):
    started = provenance.start()
    ustar, vstar = g.steady_state()
    report = {"theory": pulled(ustar, vstar), "runs": []}
    print(report["theory"], flush=True)
    for dx, dt in TRIALS:
        p = front(ustar, vstar, dx, dt)
        run = {"dx": dx, "dt": dt, "every": 100, "positions": p, "speed_t300_1800": (p[17] - p[2]) / 1500}
        report["runs"].append(run)
        print(dx, dt, round(run["speed_t300_1800"], 3), flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
