"""Figures 1 and 2 of the signed-charge paper, from `signed_curves.json`: signed_curves_figure.py CURVES.json OUT_DIR

h(r) and C(R) under `all`, with 95 % CIs by finger, mindtct and LEADER side by side. Real prints appear only as numbers.

Run with python_biometrics/.venv-figures.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = {"real": ("Real, FVC B", "#2a78d6"), "sfinge": ("SFinGe", "#eb6834"), "model2": ("ZP", "#1baf7a"), "d_s1_sp2_f3": ("ZP+P", "#eda100"), "phantom": ("Rendered ZP", "#e87ba4")}
READERS = {"mindtct": "mindtct", "leader": "LEADER"}
RADII = (30, 60, 90, 120, 150)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def axes(n):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "sans-serif"})
    fig, ax = plt.subplots(1, n, figsize=(7.2, 3.0), sharey=True, constrained_layout=True)
    for a in ax:
        a.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.axhline(0, color=MUTED, linewidth=0.8)
    return fig, ax


def h_figure(report, out):
    fig, ax = axes(2)
    for a, (reader, title) in zip(ax, READERS.items()):
        for name, (label, colour) in SERIES.items():
            v = report[reader][name]["all"]
            r, h = np.array(v["r"]), np.array(v["h"])
            keep = r >= 15  # under it, few pairs and an extractor's own dipoles
            a.fill_between(r[keep], np.array(v["h_low"])[keep], np.array(v["h_high"])[keep], color=colour, alpha=0.15, linewidth=0)
            a.plot(r[keep], h[keep], color=colour, linewidth=2, label=label)
        a.set_title(title, loc="left", color=INK, fontsize=9)
        a.set_xlabel("Separation r, px at 500 ppi")
    ax[0].set_ylabel("h(r), mean sign product")
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(out, dpi=200)


def c_figure(report, out):
    fig, ax = axes(2)
    for a, (reader, title) in zip(ax, READERS.items()):
        for name, (label, colour) in SERIES.items():
            c = report[reader][name]["all"]["c"]
            mean = [c[str(R)]["mean"] for R in RADII]
            low = [c[str(R)]["mean"] - c[str(R)]["ci"][0] for R in RADII]
            high = [c[str(R)]["ci"][1] - c[str(R)]["mean"] for R in RADII]
            a.errorbar(RADII, mean, yerr=[low, high], color=colour, linewidth=2, marker="o", markersize=4, capsize=2, label=label)
        a.axhline(-1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        a.text(150, -0.85, "sum rule in random fields", color=MUTED, fontsize=8, ha="right", va="bottom")
        a.set_title(title, loc="left", color=INK, fontsize=9)
        a.set_xlabel("Radius R, px at 500 ppi")
        a.set_xticks(RADII)
    ax[0].set_ylabel("C(R), charge seen within R")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(out, dpi=200)


def main(path, out):
    report = json.load(open(path))
    h_figure(report, Path(out) / "paper-signed-h.png")
    c_figure(report, Path(out) / "paper-signed-c.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
