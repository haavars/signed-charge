"""Developmental prints through ridgegen's diffusion pass, beside Phantom's, unchanged (O-002, E-046): diffused.py SHEET.jpg OUT.json RUN.npz...

Ink as drawn, white paper outside the contact, then `diffusion.render` at its standard strength; model 2 through the same steps, same size. A few prints, to see what the pass does.

Run: python -m signed_charge.exploratory.diffused SHEET.jpg OUT.json RUN.npz...
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from signed_charge.measure import flow_check as fc
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.measure import validate_masters as vm
from ridgegen import diffusion

KEEP = 6  # px: a minutia kept if one lies within this after the pass
STRENGTH = float(os.environ.get("DIFFUSED_STRENGTH", diffusion.STRENGTH))


def on_paper(im):
    h, w = im.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    contact = ((xx - w / 2) / (w / 2 - 6)) ** 2 + ((yy - h / 2) / (h / 2 - 6)) ** 2 < 1
    return np.where(contact, cv2.GaussianBlur(im, (0, 0), 0.8), 255).astype(np.uint8)


def kept(before, after):
    if not len(before):
        return 0.0
    if not len(after):
        return 0.0
    d = np.hypot(before[:, 0, None] - after[:, 0], before[:, 1, None] - after[:, 1]).min(1)
    return float(np.mean(d < KEEP))


def treat(name, im, seed):
    out = diffusion.render(im, seed, STRENGTH)
    row = {"name": name}
    for label, extract in (("mindtct", ms.mindtct), ("leader", vm.leader)):
        a, b = extract(im), extract(out)
        row[label] = {"before": len(a), "after": len(b), "kept": kept(a, b)}
    print(row, flush=True)
    return out, row


def main(sheet, out, paths):
    started = provenance.start()
    rows, report = [], []
    drawn = [(Path(p).stem, on_paper(screen.print_of(np.load(p)["u"]))) for p in paths]
    done = [treat(n, im, 100 + i) for i, (n, im) in enumerate(drawn)]
    rows.append(np.hstack([im for _, im in drawn]))
    rows.append(np.hstack([d for d, _ in done]))
    report += [r for _, r in done]
    phantom = []
    for i, f in enumerate((2, 3, 7, 8)[: len(paths)]):
        key, im = fc.image(("model2", 7000 + i, f, 1))
        ys, xs = np.nonzero(im == 40)  # the inked pad, centred on a canvas of the developmental prints' size
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        canvas = np.full((screen.HEIGHT, screen.WIDTH), 235, np.uint8)
        top, left = cy - screen.HEIGHT // 2, cx - screen.WIDTH // 2
        cut = im[max(top, 0) : top + screen.HEIGHT, max(left, 0) : left + screen.WIDTH]
        y, x = max(-top, 0), max(-left, 0)
        canvas[y : y + cut.shape[0], x : x + cut.shape[1]] = cut[: screen.HEIGHT - y, : screen.WIDTH - x]
        phantom.append((f"model2_{key[1]}", on_paper(canvas)))
    done = [treat(n, im, 200 + i) for i, (n, im) in enumerate(phantom)]
    rows.append(np.hstack([im for _, im in phantom]))
    rows.append(np.hstack([d for d, _ in done]))
    report += [r for _, r in done]
    cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    json.dump({"prints": report, "strength": STRENGTH, "_provenance": provenance.record(started)}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
