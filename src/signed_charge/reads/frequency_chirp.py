"""(E-089) Does fanout.fields read a known change in ridge spacing? Straight stripes with period p(y) known exactly, through the same estimator and 50 px smoothing as agreement_total."""
import sys
from pathlib import Path
import numpy as np
import cv2
from signed_charge.measure import fanout as fan
H = 560
y = np.arange(H, dtype=float)
def stripes(p):
    phase = 2 * np.pi * np.cumsum(1 / p)
    im = 128 + 110 * np.cos(phase)
    return np.tile(im, (H, 1)).T.astype(np.uint8)
for lo, hi in ((9, 9), (8, 10), (7, 12), (6, 14)):
    for shape in ("linear", "sine"):
        if shape == "linear":
            lnp = np.linspace(np.log(lo), np.log(hi), H)
        else:
            lnp = np.log(np.sqrt(lo * hi)) + 0.5 * np.log(hi / lo) * np.sin(2 * np.pi * y / 250)
        p = np.exp(lnp)
        im = stripes(p)
        fg = np.ones(im.shape, bool)
        _, _, period = fan.fields(im, fg)
        read = cv2.GaussianBlur(np.log(np.maximum(period, 1.0)).astype(np.float32), (0, 0), 50.0)
        c = (60, H - 60)
        true = cv2.GaussianBlur(np.tile(lnp, (H, 1)).T.astype(np.float32), (0, 0), 50.0)
        g_read, g_true = np.gradient(read, axis=0)[c[0]:c[1], H // 2], np.gradient(true, axis=0)[c[0]:c[1], H // 2]
        slope = (g_read @ g_true) / max(g_true @ g_true, 1e-12)
        print(f"{lo}-{hi} {shape:6s} true |grad| {np.abs(g_true).mean():.5f} read {np.abs(g_read).mean():.5f} gain {slope:.2f}  read period {np.exp(read[c[0]:c[1], H//2]).min():.2f}-{np.exp(read[c[0]:c[1], H//2]).max():.2f} true {np.exp(true[c[0]:c[1], H//2]).min():.2f}-{np.exp(true[c[0]:c[1], H//2]).max():.2f}")
