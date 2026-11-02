"""Wobble after the diffusion pass, like against like with real prints (E-047): wobble_diffused.py OUT.json RUN.npz...

The prints of `wobble.py` as drawn have no capture in them, and real ones do. So both models through `diffusion.render` at standard strength, then `wobble.measure`, on 8 fingers of each.

Run: python -m signed_charge.exploratory.wobble_diffused OUT.json RUN.npz...
"""
import json
import sys
from pathlib import Path

import numpy as np

from signed_charge.exploratory import diffused as df
from signed_charge.measure import flow_check as fc
from signed_charge.measure import provenance
from signed_charge.measure import screen
from signed_charge.exploratory import wobble as wb
from ridgegen import diffusion


def main(out, paths):
    started = provenance.start()
    report = {"developmental": [], "developmental_drawn": [], "model2": [], "model2_drawn": []}
    for i, p in enumerate(paths[:8]):
        im = df.on_paper(screen.print_of(np.load(p)["u"]))
        report["developmental_drawn"].append(wb.measure((("d", i, 1), im))[1])
        report["developmental"].append(wb.measure((("d", i, 1), diffusion.render(im, 300 + i)))[1])
    for i, (seed, f) in enumerate(zip(range(7000, 7008), (2, 3, 4, 5, 7, 8, 9, 10))):
        _, im = fc.image(("model2", seed, f, 1))
        ys, xs = np.nonzero(im == 40)
        crop = im[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        im = np.where(crop == 40, 40, np.where(crop == 235, 235, 255)).astype(np.uint8)
        pad = -im.shape[0] % 16, -im.shape[1] % 16
        im = np.pad(im, ((16, 16 + pad[0]), (16, 16 + pad[1])), constant_values=235)
        report["model2_drawn"].append(wb.measure((("m", i, 1), im))[1])
        report["model2"].append(wb.measure((("m", i, 1), diffusion.render(im, 400 + i)))[1])
    summary = {k: {"prints": sum(1 for r in v if r), "fine_coarse": float(np.median([r["fine_coarse"] for r in v if r])), "fine_mid": float(np.median([r["fine_mid"] for r in v if r]))} for k, v in report.items()}
    print(summary, flush=True)
    summary["_provenance"] = provenance.record(started)
    json.dump(summary, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
