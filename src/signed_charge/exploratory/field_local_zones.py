"""Exploratory, written after D-050's score (E-080): field_local_zones.py OUT.json RUN.npz...

Minutiae and h(15-45 px) by mismatch, 0-30, 30-60 and 60-90, on the seam, off it and in the 12 px between, pooled over the prints with no CI. Run from python_biometrics as `field_local`.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as nd
from signed_charge.measure import provenance
from signed_charge.reads import field_local as fl
from signed_charge.measure import field_similarity as fs
from signed_charge.reads import minutiae_seams as seams
from signed_charge.measure import minutiae_stats as ms
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
EDGE = (sm.EDGES[:-1] >= sf.SHORT[0]) & (sm.EDGES[1:] <= sf.SHORT[1])
BANDS = {"0-30": (0, 30), "30-60": (30, 60), "60-90": (60, 90.01)}
def one(path):
    d = np.load(path); u, arrival = d["u"], d["arrival"]
    im = np.where(u > np.median(u), 40, 235).astype(np.uint8)
    fg = fs.foreground(im)
    inner = cv2.erode(fg.astype(np.uint8), np.ones((2 * ms.BORDER + 1, 2 * ms.BORDER + 1), np.uint8)).astype(bool)
    guarded = inner.copy()
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    for x, y in sc.singular(fs.field(im), fg):
        guarded &= np.hypot(xx - x, yy - y) >= sc.GUARD
    k = fl.kink(arrival)
    on, off = guarded & (k >= seams.SEAM), guarded & (nd.maximum_filter(k, 25) < seams.SEAM)
    near = guarded & ~on & ~off
    angle = fl.mismatch(d["theta"], arrival)
    field = sm.direction(im)
    out = {"near": int(near.sum())}
    for reader, (extract, y_sign) in sm.EXTRACTORS.items():
        pts = extract(im)
        for zone, m in (("on", on), ("off", off), ("near", near)):
            for name, (a, b) in BANDS.items():
                mask = m & (angle >= a) & (angle < b)
                got = sf.inside(pts, mask)
                t, c, n = sf.binned(got, y_sign, field, inner)
                out[f"{reader}/{zone}/{name}"] = (int(mask.sum()), len(got), float(t[EDGE].sum()), float(c[EDGE].sum()))
    return out


if __name__ == "__main__":
    started = provenance.start()
    rows = list(ProcessPoolExecutor(4).map(one, sorted(sys.argv[2:]), chunksize=2))
    report = {}
    for reader in sm.EXTRACTORS:
        total = sum(r[f"{reader}/{z}/{b}"][0] for r in rows for z in ("on", "near", "off") for b in BANDS)
        for zone in ("on", "near", "off"):
            for name in BANDS:
                a = np.array([r[f"{reader}/{zone}/{name}"] for r in rows], float).sum(0)
                report[f"{reader}/{zone}/{name}"] = {"share_of_cells": float(a[0] / total), "density": float(a[1] / max(a[0] * ms.MM2, 1e-6) * 100), "minutiae": int(a[1]), "h": float(a[2] / max(a[3], 1)), "pairs": int(a[3])}
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(sys.argv[1], "w"), indent=1)
