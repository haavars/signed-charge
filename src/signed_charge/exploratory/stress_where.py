"""Where on a print the flow templates go wrong, a diagnosis of E-057, no test (E-058): stress_where.py OUT.json FIGURE.jpg

Every fourth print of `stress_flow`'s, each at its best pose for each candidate: the angle by block, averaged on the template's canvas, so "where" is in the finger's frame. Also by the template each print chose.

Run: python -m signed_charge.exploratory.stress_where OUT.json FIGURE.jpg
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from signed_charge.measure import conformality as cf
from signed_charge.measure import provenance
from signed_charge.exploratory import stress_flow as sf

SHARE = 4


def pose(z, mask, poses):
    """The best (score, pose, y0, x0)."""
    h, w = z.shape
    ph, pw = poses.shape[1:]
    total, top = mask.sum(), None
    for dy in sf.SHIFTS:
        for dx in sf.SHIFTS:
            y0, x0 = 220 // sf.B - h // 2 + dy, (pw - w) // 2 + dx
            if y0 + h > ph or x0 + w > pw:
                continue
            cut = np.zeros((len(poses), h, w), complex)
            a, c = max(y0, 0), max(x0, 0)
            cut[:, a - y0 :, c - x0 :] = poses[:, a : y0 + h, c : x0 + w]
            valid = (cut != 0) & mask
            count = valid.sum((1, 2))
            gap = np.abs(np.angle(z * np.conj(cut))) / 2
            score = (gap * valid).sum((1, 2)) / np.maximum(count, 1)
            score[count < sf.OVERLAP * total] = np.inf
            k = int(np.argmin(score))
            if top is None or score[k] < top[0]:
                top = (float(score[k]), k, y0, x0, gap[k], valid[k])
    return top


def one(item):
    key, im = item
    b = sf.blocks(im)
    if b is None:
        return key, None
    z, mask = b
    out = {}
    for name in ("stress", "guide"):
        found = {p: pose(z, mask, poses) for p, poses in sf.SETS[name].items()}
        found = {p: v for p, v in found.items() if v}
        if not found:
            continue
        p = min(found, key=lambda q: found[q][0])
        _, k, y0, x0, gap, valid = found[p]
        out[name] = (p, {q: float(np.degrees(v[0])) for q, v in found.items()}, y0, x0, gap.astype(np.float32), valid, k)
    return key, out


def main(out, figure):
    started = provenance.start()
    sf.SETS["stress"] = {k: sf.posed(v) for k, v in sf.stress_templates().items()}
    sf.SETS["guide"] = {k: sf.posed(v) for k, v in sf.guide_templates().items()}
    items = [x for i, x in enumerate(cf.fvc(("DB1", "DB2", "DB3"))) if i % SHARE == 0]
    with ProcessPoolExecutor(2) as pool:
        rows = [v for _, v in pool.map(one, items, chunksize=2) if v and len(v) == 2]
    h, w = sf.CANVAS[0] // sf.B, sf.CANVAS[1] // sf.B
    report, tiles = {}, []
    for name in ("stress", "guide"):
        total, count = np.zeros((h, w)), np.zeros((h, w))
        for r in rows:
            _, _, y0, x0, gap, valid, _ = r[name]
            for (y, x), g, v in zip(np.ndindex(gap.shape), gap.ravel(), valid.ravel()):
                if v and 0 <= y + y0 < h and 0 <= x + x0 < w:
                    total[y + y0, x + x0] += g
                    count[y + y0, x + x0] += 1
        mean = np.degrees(total / np.maximum(count, 1))
        tiles.append(cv2.applyColorMap(np.clip(mean / 45 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS) * (count[..., None] > 5))
        third = h // 3
        report[name] = {
            "by_template": {p: float(np.mean([r[name][1][p] for r in rows if p in r[name][1]])) for p in sf.SETS[name]},
            "chosen_as": {p: float(np.mean([r[name][1][p] for r in rows if r[name][0] == p])) for p in sf.SETS[name] if any(r[name][0] == p for r in rows)},
            "tip_third": float(total[:third].sum() / count[:third].sum() * 180 / np.pi),
            "middle_third": float(total[third : 2 * third].sum() / count[third : 2 * third].sum() * 180 / np.pi),
            "base_third": float(total[2 * third :].sum() / max(count[2 * third :].sum(), 1) * 180 / np.pi),
        }
    cv2.imwrite(figure, cv2.resize(np.hstack(tiles), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST), [cv2.IMWRITE_JPEG_QUALITY, 90])
    report["prints"] = len(rows)
    report["_provenance"] = provenance.record(started)
    print(json.dumps({k: v for k, v in report.items() if k != "_provenance"}, indent=0), flush=True)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
