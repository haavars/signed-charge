"""(E-089) Does a realflow print follow the flow it was given? Geometry only, no minutiae. Controls: turned 90°, flipped top to bottom, another finger's flow."""
import sys
import glob
import os
from pathlib import Path
import cv2
import numpy as np
from signed_charge.measure import conformality as cf
from signed_charge.measure import field_similarity as fs
from signed_charge.generate import field_print as fp
from signed_charge import paths
D = str(paths.GENERATED)
ROOT = str(paths.FVC)
def src(f):
    _, year, db, finger = Path(f).stem.split("_")
    return f"{ROOT}/{year}/{db}_B/{finger}_1.tif"
def agree(a, b):
    c = 216; w = 150
    d = 2 * (a - b)[c - w:c + w, c - w:c + w]
    return float(np.cos(d).mean())
files = sorted(glob.glob(f"{D}/flow_d055/realflow_*.npz"))[2::7]
flows = []
for f in files:
    fp.SMOOTH = 50.0
    smooth = fp.real(src(f))[0]
    fp.SMOOTH = 0.0
    raw = fp.real(src(f))[0]
    flows.append((smooth, raw))
rows = []
for i, f in enumerate(files):
    u = np.load(f)["u"]
    u = cv2.normalize(u.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t = fs.field(u)
    smooth, raw = flows[i]
    other = flows[(i + 3) % len(flows)][0]
    fg = np.ones(smooth.shape, bool)
    rows.append((agree(t, smooth), agree(t, smooth + np.pi / 2), agree(t, smooth[::-1]), agree(t, other), agree(smooth, raw), len(fs.cores(raw, fg)), len(fs.cores(smooth, fg)), len(fs.cores(t, fg))))
    print(Path(f).stem, " ".join(f"{x:.2f}" for x in rows[-1][:5]), *rows[-1][5:], flush=True)
print("mean: given, turned, flipped, other, smooth-vs-raw", np.round(np.mean(rows, 0)[:5], 3), "cores raw/smooth/grown", np.round(np.mean(rows, 0)[5:], 2))
