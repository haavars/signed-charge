"""Do mindtct and LEADER give the same sign to the same minutia? From the cache alone, no images: cross_signs.py [OUT.json]
Mutual nearest neighbours within MATCH px; heading agreement is cos(h_m - h_l) > 0. CI by finger.
Section 5.7's "opposite signs to 0.7 % of 15,424 shared minutiae" is the real row."""

import json
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

from signed_charge import paths
from signed_charge.measure import provenance
from signed_charge.measure import signed_minutiae as sm
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import signed_false as sf

MATCH = 8.0
SETS = ("real", "sfinge", "model2", "e027")


def main(out=None):
    started = provenance.start()
    rows = pickle.load(open(paths.CACHE, "rb"))["rows"]
    rng = np.random.default_rng(0)
    report = {}
    for name in SETS:
        per_finger = {}
        angles = []
        for row in rows:
            if not parts.member(row, name):
                continue
            m, l = row["points"]["mindtct"], row["points"]["leader"]
            if len(m) < 1 or len(l) < 1:
                continue
            hm, hl = sf.heading(m, sm.EXTRACTORS["mindtct"][1]), sf.heading(l, sm.EXTRACTORS["leader"][1])
            dm, im_ = cKDTree(l[:, :2]).query(m[:, :2], distance_upper_bound=MATCH)
            dl, il = cKDTree(m[:, :2]).query(l[:, :2], distance_upper_bound=MATCH)
            same = opp = 0
            for i in np.flatnonzero(np.isfinite(dm)):
                j = im_[i]
                if np.isfinite(dl[j]) and il[j] == i:
                    c = np.cos(hm[i] - hl[j])
                    angles.append(c)
                    same += c > 0
                    opp += c < 0
            f = per_finger.setdefault(row["finger"], [0, 0, 0])
            f[0] += same
            f[1] += opp
            f[2] += len(m)
        tab = np.array(list(per_finger.values()), float)
        value = lambda i: tab[i, 1].sum() / max(tab[i, 0].sum() + tab[i, 1].sum(), 1)  # noqa: E731
        draws = [value(rng.integers(len(tab), size=len(tab))) for _ in range(1000)]
        angles = np.array(angles)
        report[name] = {
            "fingers": len(tab),
            "mindtct_minutiae": int(tab[:, 2].sum()),
            "matches": int(tab[:, :2].sum()),
            "opposite": {"mean": float(value(np.arange(len(tab)))), "ci": [float(np.percentile(draws, q)) for q in (2.5, 97.5)]},
            "across_share": float(np.mean(np.abs(angles) < 0.5)),
        }
        print(
            f"{name}: fingers {len(tab)}, mindtct minutiae {int(tab[:, 2].sum())}, mutual matches {int(tab[:, :2].sum())} ({tab[:, :2].sum() / tab[:, 2].sum():.2f} of mindtct's), "
            f"opposite sign {value(np.arange(len(tab))):.3f} ({np.percentile(draws, 2.5):.3f}-{np.percentile(draws, 97.5):.3f}); |cos|<0.5 share {np.mean(np.abs(angles) < 0.5):.3f}"
        )
    if out:
        report["_provenance"] = provenance.record(started)
        json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
