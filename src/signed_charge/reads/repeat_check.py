"""Is E-096's repeat test right? Two impressions: (a) its way, splay vectors moved between frames; (b) the other impression's double angle warped into this frame first, then d read there. Exploratory."""
import sys
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from signed_charge.reads import extractor_free as ef
from signed_charge.measure import flow_charge as fc
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import minutiae_place as place
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
SCALES = (50, 100, 150)
MODE = sys.argv[1]
def pair(task):
    (ia, sa), (ib, sb) = task
    ra, rb = ef.ROWS[ia], ef.ROWS[ib]
    pa, pb = ra["points"]["mindtct"], rb["points"]["mindtct"]
    y = sm.EXTRACTORS["mindtct"][1]
    if len(pa) < 5 or len(pb) < 5: return None
    move = sf.align(pa[:, :2], sf.heading(pa, y), pb[:, :2], sf.heading(pb, y))
    if move is None: return None
    turn, shift, count = move
    ga, za, fa = ef.state(ra, sa); gb, zb, fb = ef.state(rb, sb)
    co, si = np.cos(turn), np.sin(turn)
    rotate = np.array([[co, si], [-si, co]])
    h, w = zb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    back = (np.c_[xx.ravel(), yy.ravel()] - shift) @ rotate.T
    xa, ya = np.rint(back[:, 0]).astype(int), np.rint(back[:, 1]).astype(int)
    ok = (xa >= 0) & (xa < za.shape[1]) & (ya >= 0) & (ya < za.shape[0])
    ok[ok] &= fa[ya[ok], xa[ok]]
    warped = np.zeros((h, w), complex)
    warped.ravel()[np.flatnonzero(ok)] = za[ya[ok], xa[ok]] * np.exp(2j * turn)
    covered = np.zeros((h, w), bool); covered.ravel()[np.flatnonzero(ok)] = True
    cells = gb & covered
    xc, yc = np.nonzero(cells)[1], np.nonzero(cells)[0]
    if len(xc) < 2000: return None
    bk = (np.c_[xc, yc] - shift) @ rotate.T
    xk, yk = np.rint(bk[:, 0]).astype(int), np.rint(bk[:, 1]).astype(int)
    keep = (xk >= 0) & (xk < za.shape[1]) & (yk >= 0) & (yk < za.shape[0])
    xc, yc, xk, yk = xc[keep], yc[keep], xk[keep], yk[keep]
    out = {"turn": abs(float(np.degrees(turn))), "orient": float(np.abs(np.mean(zb[yc, xc] * np.conj(warped[yc, xc]))))}
    for s in SCALES:
        ab, db = fc.spreading(zb, float(s)); aw, dw = fc.spreading(warped, float(s)); aa, da = fc.spreading(za, float(s))
        am, dm = fc.spreading(zb * fb, float(s)); ac, dc = fc.spreading(za * fa, float(s))
        vm = np.c_[dm[yc, xc] * np.cos(am[yc, xc]), dm[yc, xc] * np.sin(am[yc, xc])]
        vc = np.c_[dc[yk, xk] * np.cos(ac[yk, xk]), dc[yk, xk] * np.sin(ac[yk, xk])] @ rotate
        vbg = np.c_[db[yc, xc] * np.cos(ab[yc, xc]), db[yc, xc] * np.sin(ab[yc, xc])]
        extra = ((vm * vc).sum(), (vm * vm).sum(), (vc * vc).sum(), (vm * vbg).sum(), (vbg * vbg).sum(), np.abs(db[yc, xc]).sum(), np.abs(dm[yc, xc]).sum(), len(xc))
        vb = np.c_[db[yc, xc] * np.cos(ab[yc, xc]), db[yc, xc] * np.sin(ab[yc, xc])]
        vw = np.c_[dw[yc, xc] * np.cos(aw[yc, xc]), dw[yc, xc] * np.sin(aw[yc, xc])]
        va = np.c_[da[yk, xk] * np.cos(aa[yk, xk]), da[yk, xk] * np.sin(aa[yk, xk])] @ rotate
        out[s] = ((vb * vw).sum(), (vb * vb).sum(), (vw * vw).sum(), (vb * va).sum(), (va * va).sum()) + extra + ((va * vw).sum(),)
    return out
def main():
    base = pickle.load(open(ef.CACHE, "rb"))["rows"]
    where = place.sources(base)
    groups = {}
    for row in base:
        name = next((n for n in ("real", "sfinge", "model2") if parts.member(row, n)), None)
        if name is None or id(row) not in where: continue
        ef.ROWS.append(row)
        groups.setdefault(name, {}).setdefault(row["finger"], []).append((len(ef.ROWS) - 1, where[id(row)]))
    for name, fingers in groups.items():
        items = sorted(fingers.items())[:40]
        tasks = [(v[0], v[1]) for f, v in items if len(v) > 1] if MODE == "same" else [(items[k][1][0], items[(k + 1) % len(items)][1][1]) for k in range(len(items))]
        with ProcessPoolExecutor(3) as pool:
            got = [r for r in pool.map(pair, tasks, chunksize=1) if r]
        print(name, len(got), "of", len(tasks), "mean |turn|", round(float(np.mean([r["turn"] for r in got])), 1), "orient", round(float(np.mean([r["orient"] for r in got])), 3))
        for s in SCALES:
            t = np.array([r[s] for r in got]).sum(0)
            print(f"  {s}: warped {t[0] / np.sqrt(t[1] * t[2]):.2f}  moved, background in {t[3] / np.sqrt(t[1] * t[4]):.2f}  moved, background zeroed {t[5] / np.sqrt(t[6] * t[7]):.2f}  own flow, zeroed vs not {t[8] / np.sqrt(t[6] * t[9]):.2f}  a-warped vs a-moved {t[13] / np.sqrt(t[2] * t[4]):.2f}")
main()
