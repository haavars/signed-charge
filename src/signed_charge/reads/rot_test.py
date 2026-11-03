"""Known answer: a print against itself rotated by a known angle, both ways of comparing d; and against another finger. Exploratory."""

import pickle
import cv2
import numpy as np
from signed_charge.measure import flow_charge as fc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.measure import signed_minutiae as sm
from signed_charge import paths

base = pickle.load(open(paths.CACHE, "rb"))["rows"]
where = place.sources(base)
rows = [r for r in base if parts.member(r, "real") and id(r) in where][::60][:6]


def unit(im):
    z = sm.direction(im)
    return z / (np.abs(z) + 1e-9)


def corr(a, b):
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


for deg in (0.0, 5.0, 15.0):
    res = {50: [[], []], 100: [[], []], 150: [[], []]}
    for row in rows:
        im = place.image(where[id(row)])
        h, w = im.shape
        turn = np.radians(deg)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, 1.0)  # x' = co x - si y + ..., y' = si x + co y + ...
        imb = cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        za, zb = unit(im), unit(imb)
        co, si = np.cos(turn), np.sin(turn)
        rotate = np.array([[co, si], [-si, co]])
        shift = np.array([M[0, 2], M[1, 2]])
        yy, xx = np.mgrid[60 : h - 60 : 4, 60 : w - 60 : 4]
        pts = np.c_[xx.ravel(), yy.ravel()]
        back = (pts - shift) @ rotate.T
        xa, ya = np.rint(back[:, 0]).astype(int), np.rint(back[:, 1]).astype(int)
        ok = (xa >= 0) & (xa < w) & (ya >= 0) & (ya < h)
        pts, xa, ya = pts[ok], xa[ok], ya[ok]
        Y, X = np.mgrid[0:h, 0:w]
        bk = (np.c_[X.ravel(), Y.ravel()] - shift) @ rotate.T
        xw, yw = np.clip(np.rint(bk[:, 0]).astype(int), 0, w - 1), np.clip(np.rint(bk[:, 1]).astype(int), 0, h - 1)
        warped = (za[yw, xw] * np.exp(2j * turn)).reshape(h, w)
        for s in res:
            ab, db = fc.spreading(zb, float(s))
            aw, dw = fc.spreading(warped, float(s))
            aa, da = fc.spreading(za, float(s))
            x, y = pts[:, 0], pts[:, 1]
            vb = np.c_[db[y, x] * np.cos(ab[y, x]), db[y, x] * np.sin(ab[y, x])]
            vw = np.c_[dw[y, x] * np.cos(aw[y, x]), dw[y, x] * np.sin(aw[y, x])]
            va = np.c_[da[ya, xa] * np.cos(aa[ya, xa]), da[ya, xa] * np.sin(aa[ya, xa])] @ rotate
            res[s][0].append(corr(vb, vw))
            res[s][1].append(corr(vb, va))
    print(deg, {s: (round(float(np.mean(v[0])), 2), round(float(np.mean(v[1])), 2)) for s, v in res.items()}, "(warped, moved)")
