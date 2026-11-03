"""A sheet for the eye from fine_ridges_experiment.py's results: fine_ridges_sheet.py RESULTS.jsonl OUT.png

Four rolled prints' centres at 500 ppi (two fine ones that failed under today's pass, one fine that passed, one
coarse), each as `base`, `strength25` and `rescaled` render it, the failing attempt where there is one.

Run: python -m signed_charge.exploratory.fine_ridges_sheet ../research/realism/fine_ridges_experiment.jsonl OUT.png
"""

import sys
import json
import cv2
import numpy as np
from signed_charge.exploratory import fine_ridges_experiment as fx
from ridgegen import fingerprints, jp2
from ridgegen import impression as imp

rows = [json.loads(l) for l in open(sys.argv[1])]
# two fine fingers that failed in base, one fine that passed, one coarse
fails = sorted({(r["seed"], r["fgp"], r["period"]) for r in rows if r["variant"] == "base" and not r["accepted"] and r["period"] < 7.6})[:2]
ok = sorted({(r["seed"], r["fgp"], r["period"]) for r in rows if r["variant"] == "base" and r["period"] < 7.2} - set(fails))[:1]
coarse = sorted({(r["seed"], r["fgp"], r["period"]) for r in rows if r["period"] >= 9.2})[:1]
tiles = []
for seed, fgp, period in fails + ok + coarse:
    cap = fingerprints.rolled_capture(seed, fgp, 0)
    drawn = imp.upscale(cap, 2)
    attempt = next((r["attempt"] for r in rows if r["seed"] == seed and r["fgp"] == fgp and r["variant"] == "base" and not r["accepted"]), 0)
    row = []
    for v in ["base", "strength25", "rescaled"]:
        img, _ = fx.rendered(cap, drawn, attempt, v, period)
        small = jp2.downsample(img)
        ys, xs = np.nonzero(cap.contact > 0.8)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        crop = small[cy - 100 : cy + 100, cx - 100 : cx + 100]
        crop = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_NEAREST)
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        cv2.putText(crop, f"{v} {seed}/{fgp} p={period}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        row.append(crop)
    tiles.append(np.hstack(row))
cv2.imwrite(sys.argv[2], np.vstack(tiles))
