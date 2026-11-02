"""D-068 step 1: how much of a print's net charge does its flow account for, at what scale? flux_scale.py OUT.json [SETS...]

Patches of `SIDE` px with no singular point near, as `winding`. In each, the net charge Q of mindtct, LEADER and phase winding, lifted to the patch's frame, beside the flux of the ridge field round the patch's edge, whose exact value is Q:
    F(s)      mean ridge frequency times the flux of the unit ridge direction, smoothed at s px
    G(s)      the same with the local frequency on the edge, so spacing counts too
Per set, by finger: slope of Q on F and G, their correlation, and X = <Q²> - <N>, which independent neutral minutiae leave alone, where <N> is minutiae a patch.

Sets: planted (flat sheets, independent signs), real, sfinge, model2, realflow, readflow, lead. Run: python -m signed_charge.reads.flux_scale OUT.json [SETS...]
LIMIT=n takes the first n prints of each set.
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import fanout as fo
from signed_charge.reads import field_read as fr
from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import provenance
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge.measure import winding as w

JOBS = int(os.environ.get("READ_JOBS", 8))
LIMIT = int(os.environ.get("LIMIT", 0))
SIDE, SCALES, SHEETS, RESAMPLES = int(os.environ.get("SIDE", 96)), (9, 18, 30, 50), 60, 1000
QUALITY = int(os.environ.get("QUALITY", 3))
FLOWS = {"realflow": "realflow_*.npz", "readflow": "readflow_*.npz"}
FLOW_DIR = {"realflow": w.FLOW, "readflow": w.FLOW, "lead": Path(w.FLOW).parent / "lead_d067"}
ROWS = []


def smooth(z2, fg, s):
    weight = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), s, borderType=cv2.BORDER_CONSTANT)
    out = sum(c * cv2.GaussianBlur(p.astype(np.float32) * fg, (0, 0), s, borderType=cv2.BORDER_CONSTANT) for c, p in ((1, z2.real), (1j, z2.imag)))
    return out / np.maximum(weight, 1e-3)


def flux(z2, phi, i0, j0):
    """Flux round the patch's edge: with the patch's mean frequency, and with the local one."""
    beta, mismatch = w.unwrap_patch(np.angle(z2), i0, i0 + SIDE, j0, j0 + SIDE)
    if abs(mismatch) > 1:
        return None
    tx, ty = w.rot(np.cos(beta), np.sin(beta))
    p = phi[i0 : i0 + SIDE, j0 : j0 + SIDE]
    plain = tx[:, -1].sum() - tx[:, 0].sum() + ty[-1, :].sum() - ty[0, :].sum()
    local = (tx * p)[:, -1].sum() - (tx * p)[:, 0].sum() + (ty * p)[-1, :].sum() - (ty * p)[0, :].sum()
    return float(plain * p.mean()), float(local)


def read(im, points, inner, finger):
    fg = fs.foreground(im)
    singular = sc.singular(fs.field(im), fg)
    z2 = sm.direction(im)
    z2 = z2 / (np.abs(z2) + 1e-9)
    b, V = w.demodulate(im, fg, z2)
    amp = cv2.GaussianBlur(np.abs(b + 1j * np.imag(np.exp(-0.5j * np.angle(z2)) * V)).astype(np.float32), (0, 0), 4)
    strong = inner & (amp > w.AMPLITUDE * np.median(amp[inner]))
    rx, ry, rq, rb = w.residues(b, V, z2, strong)
    keep, _ = w.cancel(rx, ry, w.heading_of(rq, rb), w.DIPOLE_FAR)
    rx, ry, rq, rb = rx[keep], ry[keep], rq[keep], rb[keep]
    _, _, period = fo.fields(im, fg)
    phi = 1.0 / np.maximum(period, 1.0)
    quality = sf.quality(im)
    smoothed = {s: smooth(z2, fg, s) for s in SCALES}
    heads = {
        reader: (points[reader], sf.heading(points[reader], y_sign) if len(points[reader]) else np.zeros(0)) for reader, (_, y_sign) in sm.EXTRACTORS.items()
    }
    out = []
    for i0, j0 in w.patches(inner, singular, SIDE):
        if quality[i0 : i0 + SIDE, j0 : j0 + SIDE].min() < QUALITY:
            continue
        beta, mismatch = w.unwrap_patch(np.angle(z2), i0, i0 + SIDE, j0, j0 + SIDE)
        if abs(mismatch) > 1:
            continue
        nx, ny = np.cos(beta), np.sin(beta)
        tx, ty = w.rot(nx, ny)
        inside = lambda x, y: (x >= j0) & (x < j0 + SIDE) & (y >= i0) & (y < i0 + SIDE)  # noqa: E731
        m = inside(rx, ry)
        xi, yi = (rx[m] - j0).astype(int), (ry[m] - i0).astype(int)
        row = {"finger": finger, "winding": float((rq[m] * np.sign(np.cos(rb[m]) * nx[yi, xi] + np.sin(rb[m]) * ny[yi, xi])).sum()), "n_winding": int(m.sum())}
        for reader, (p, hd) in heads.items():
            mm = inside(p[:, 0], p[:, 1]) if len(p) else np.zeros(0, bool)
            xi, yi = (p[mm, 0] - j0).astype(int), (p[mm, 1] - i0).astype(int)
            row[reader] = float(np.sign(np.cos(hd[mm]) * tx[yi, xi] + np.sin(hd[mm]) * ty[yi, xi]).sum()) if mm.any() else 0.0
            row["n_" + reader] = int(mm.sum())
        # Each smoothed flow takes the patch frame's sign, so a flip between scales can't cancel.
        row["F"], row["G"] = {}, {}
        for s in SCALES:
            got = flux(smoothed[s], phi, i0, j0)
            if got is None:
                row = None
                break
            bs, _ = w.unwrap_patch(np.angle(smoothed[s]), i0, i0 + SIDE, j0, j0 + SIDE)
            turn = 1.0 if np.cos(bs - beta).mean() >= 0 else -1.0
            row["F"][s], row["G"][s] = got[0] * turn, got[1] * turn
        if row:
            out.append(row)
    return out


def task(t):
    kind, payload = t
    if kind == "cache":
        index, source = payload
        row = ROWS[index]
        return read(place.image(source), row["points"], parts.mask_of(row), row["finger"])
    if kind in ("flow", "image"):
        im = fr.sheet(payload) if kind == "flow" else cv2.imread(payload, 0)
        inner = w.inner_of(im)
        return read(im, w.fresh_points(im, inner), inner, Path(payload).stem)
    im, _, _ = w.sheet(payload)
    inner = w.inner_of(im)
    return read(im, w.fresh_points(im, inner), inner, f"sheet/{payload}")


def boot(rows, stat, rng):
    fingers = sorted({r["finger"] for r in rows})
    groups = {f: [r for r in rows if r["finger"] == f] for f in fingers}
    draws = []
    for _ in range(RESAMPLES):
        pick = []
        for i in rng.integers(len(fingers), size=len(fingers)):
            pick += groups[fingers[i]]
        draws.append(stat(pick))
    return {"mean": float(stat(rows)), "ci": [float(np.nanpercentile(draws, q)) for q in (2.5, 97.5)]}


def slope(rows, q, key, s):
    x, y = np.array([r[key][s] for r in rows]), np.array([r[q] for r in rows])
    return float(x @ y / max(x @ x, 1e-9))


def correlation(rows, q, key, s):
    x, y = np.array([r[key][s] for r in rows]), np.array([r[q] for r in rows])
    return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")


def excess(rows, q):
    return float(np.mean([r[q] ** 2 - r["n_" + q] for r in rows]))


def summary(rows, rng):
    out = {"patches": len(rows), "fingers": len({r["finger"] for r in rows})}
    out["rms_F"] = {str(s): boot(rows, lambda p: float(np.sqrt(np.mean([r["F"][s] ** 2 for r in p]))), rng) for s in SCALES}
    for q in ("mindtct", "leader", "winding"):
        e = {"X": boot(rows, lambda p: excess(p, q), rng), "minutiae": float(np.mean([r["n_" + q] for r in rows]))}
        e["rms_Q"] = boot(rows, lambda p: float(np.sqrt(np.mean([r[q] ** 2 for r in p]))), rng)
        for key in ("F", "G"):
            e[key] = {
                str(s): {"slope": boot(rows, lambda p: slope(p, q, key, s), rng), "r": boot(rows, lambda p: correlation(p, q, key, s), rng)} for s in SCALES
            }
        out[q] = e
    return out


def main(out, names):
    started = provenance.start()
    names = names or ["planted", "real", "sfinge", "model2", "realflow", "readflow", "lead"]
    tasks = []
    if set(names) & {"real", "sfinge", "model2"}:
        rows = pickle.load(open(w.CACHE, "rb"))["rows"]
        where = place.sources(rows)
        seen = {}
        for row in rows:
            name = next((n for n in names if n in ("real", "sfinge", "model2") and parts.member(row, n)), None)
            if name and id(row) in where and (not LIMIT or seen.get(name, 0) < LIMIT):
                seen[name] = seen.get(name, 0) + 1
                ROWS.append(row)
                tasks.append((name, ("cache", (len(ROWS) - 1, where[id(row)]))))
    for name in names:
        if name in FLOW_DIR:
            pattern = "lead_*.npz" if name == "lead" else FLOWS[name]
            paths = sorted(str(p) for p in FLOW_DIR[name].glob(pattern))
            tasks += [(name, ("flow", p)) for p in paths[: LIMIT or None]]
        if name in ("synfing", "anguli"):
            tasks += [(name, ("image", str(p))) for p in sorted((Path(w.FLOW).parent / "outside_d066").glob(name + "_*"))[: LIMIT or None]]
        if name == "planted":
            tasks += [(name, ("sheet", seed)) for seed in range(LIMIT or SHEETS)]
    with ProcessPoolExecutor(JOBS) as pool:
        read_rows = list(pool.map(task, [t[1] for t in tasks], chunksize=2))
    rng = np.random.default_rng(0)
    report = {}
    for name in names:
        rows = [r for t, got in zip(tasks, read_rows) if t[0] == name for r in got]
        report[name] = summary(rows, rng) if rows else {"patches": 0}
        print(name, report[name]["patches"], flush=True)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
