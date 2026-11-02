"""Fine ridges in slaps and tips, before and after 0.25.1 (step 1): fine_ridges_slaps.py OUT [N] [ATTEMPTS]

fine_ridges_experiment.py measured rolled prints; slaps and tips have less contact and failed more often in the
`impressions-slaps-*-03` runs. Here N tips of fine-ridged fingers (print period under 8 px at 500 ppi) and N slaps
with at least one, rendered as the service renders them at 1000 ppi (`server.render_capture`) and verified on a 500
ppi copy, each attempt on its own: `before`, every finger at 0.35 (0.25.0); `after`, each finger's strength by its
period (`diffusion.strength_for`). Also N coarse tips (8 px and above) as a control, which 0.25.1 renders as before.

One JSON line per print, variant and attempt: kind, seed, code, the fingers' periods, recall, spurious share,
NFIQ 2, whether it passed. Needs the GPU and NBIS and NFIQ 2.

Run: python -m signed_charge.exploratory.fine_ridges_slaps OUT.jsonl 40 3
"""

import json
import sys

from ridgegen import diffusion, fingerprints, jp2, server, verify
from ridgegen import impression as imp

from signed_charge.exploratory.fine_ridges_experiment import print_period

SLAPS = {13: (2, 3, 4, 5), 14: (7, 8, 9, 10), 15: (1, 6)}


def pick(n, first=8000):
    tips, coarse, slaps = [], [], []
    seed = first
    while len(tips) < n or len(slaps) < n or len(coarse) < n:
        periods = {fgp: print_period(seed, fgp) for fgp in range(1, 11)}
        for fgp, p in periods.items():
            if p < 8.0 and len(tips) < n:
                tips.append((seed, fgp, [p]))
            elif p >= 8.0 and len(coarse) < n and seed % 7 == 0:
                coarse.append((seed, fgp, [p]))
        for code, members in SLAPS.items():
            if any(periods[f] < 8.0 for f in members) and len(slaps) < n:
                slaps.append((seed, code, [periods[f] for f in members]))
        seed += 1
    return [("tip", *t) for t in tips] + [("tip_coarse", *t) for t in coarse] + [("slap", *s) for s in slaps]


def main(out, n=40, attempts=3):
    request = server.RenderRequest(kind="finger", code=1, seed=0, renderer="diffusion", ppi=1000)
    with open(out, "w") as f:
        for kind, seed, code, periods in pick(n):
            for variant in ("before", "after"):
                # A fresh capture for each: attempt 0 continues the capture's own generator (`Capture.rng_for`).
                capture = fingerprints.slap_capture(seed, code, 0) if kind == "slap" else fingerprints.tip_capture(seed, code, 0)
                drawn = imp.upscale(capture, 2)
                diffusion.FINE_STRENGTH = diffusion.STRENGTH if variant == "before" else 0.25
                for attempt in range(attempts):
                    image = server.render_capture(drawn, request, attempt)
                    m = verify.verify(jp2.downsample(image), capture)
                    f.write(
                        json.dumps(
                            {
                                "kind": kind,
                                "seed": seed,
                                "code": code,
                                "periods": [round(p, 2) for p in periods],
                                "variant": variant,
                                "attempt": attempt,
                                "recall": m["minutiae_recall"],
                                "spurious": m["minutiae_spurious"],
                                "nfiq2": m["nfiq2"],
                                "accepted": m["accepted"],
                            }
                        )
                        + "\n"
                    )
                    f.flush()
    diffusion.FINE_STRENGTH = 0.25


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
