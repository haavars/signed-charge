"""A speed scan's prints, one line a print and a contact sheet (E-037): scan_read.py OUT.json SHEET.jpg RUN.npz...

Run names are `{pattern}_a{speed}_s{sigma}_{seed}`. For each: `screen.print_of`, mindtct's screens as `screen.py`, the ridge
period (peak of the print's radial power spectrum, px), the share of the contact the wave never reached, the median time since
it arrived, and the density at t = 900 where the run went on past it. Too few prints a setting for the classifier.

Run: python -m signed_charge.exploratory.scan_read OUT.json SHEET.jpg RUN.npz...
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import screen


def period(im):
    """Peak of the radial power spectrum."""
    h, w = im.shape
    a = (im == 40).astype(float) - (im == 40).mean()
    power = np.abs(np.fft.fft2(a, (512, 512))) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(512), np.fft.fftfreq(512), indexing="ij")
    f = np.hypot(fx, fy)
    bins = np.arange(0.04, 0.25, 0.002)
    radial = [power[(f >= lo) & (f < lo + 0.002)].mean() for lo in bins]
    return float(1 / (bins[int(np.argmax(radial))] + 0.001))


def main(out, sheet, paths):
    started = provenance.start()
    report = {}
    tiles = {}
    for path in sorted(paths):
        name = Path(path).stem
        pattern, speed, sigma, seed = re.fullmatch(r"(\w+?)_a([\d.]+)_s([\d.]+)_(\d+)", name).groups()
        run = np.load(path)
        im = screen.print_of(run["u"])
        command = json.loads(str(run["provenance"]))["command"]
        end = float(command[command.index("--t") + 1])
        arrival, contact = screen.flattened(run["arrival"])
        tiles.setdefault((speed, sigma), {}).setdefault(pattern, []).append(cv2.resize(im, (150, 200), interpolation=cv2.INTER_AREA))
        s = ms.describe((name, im))[1] or {}
        p = mp.describe((name, im))[1] or {}
        close = p.get("close", {})
        n = max(sum(close.get(k, 0) for k in ("same", "offset", "facing", "other")), 1)
        report[name] = {"pattern": pattern, "speed": float(speed), "sigma": float(sigma), "seed": int(seed), "period_px": period(im),
                        "density": s.get("density"), "bifurcations": s.get("bifurcations"), "count": s.get("count"), "close": n,
                        "same": close.get("same", 0) / n, "offset": close.get("offset", 0) / n, "facing": close.get("facing", 0) / n,
                        "end": end, "unreached": float(np.mean(arrival[contact] > end)),
                        "age_median": float(end - np.median(arrival[contact])),
                        "density_t900": (ms.describe((name, screen.print_of(run["t900"])))[1] or {}).get("density")
                        if "t900" in run and end > 900 else None}
        print(name.ljust(24), {k: round(v, 2) if isinstance(v, float) else v for k, v in report[name].items()
                               if k in ("density", "bifurcations", "period_px", "same", "offset", "facing", "unreached", "age_median")}, flush=True)
    rows = []
    for key in sorted(tiles):
        cells = [t for pattern in ("loop", "whorl", "arch") for t in tiles[key].get(pattern, [])]
        row = np.hstack(cells)
        cv2.putText(row, f"a{key[0]} s{key[1]}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)
        rows.append(row)
    cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
