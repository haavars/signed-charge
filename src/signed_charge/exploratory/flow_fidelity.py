"""How far phase-drawn ridges depart from the intended flow (E-025): flow_fidelity.py

Orientation of the drawn ridges (structure tensor, sigma 8 px) against the master's within 200 px of the core, and the
ridges crossed on the straight line from core to delta, 16 masters (seeds 7000-7003, FGP 2, 3, 7, 8), grown and
phase-drawn. Run as flow_check.py.
"""
import sys
import numpy as np
import cv2
import json
from signed_charge.generate import phase_global as pg
from signed_charge.generate import minutiae_growth as g
from concurrent.futures import ProcessPoolExecutor
FREE=(0,0,0,1)
W=0.05
V={'grown':None}
for name,(sm,sp_,pairs) in {'s0_sp0_f0':(0,0,0),'s0_sp2_f0':(0,2,0),'s0_sp0_f2':(0,0,2),'s0_sp2_f2':(0,2,2),'s0_sp2_f3':(0,2,3),'s1_sp2_f3':(1,2,3),'s2_sp2_f3':(2,2,3)}.items():
    V[name]=(float(sm),float(pairs),FREE,float(sp_),0.5,1.6,False,W)
def orient(r,sig=8):
    gx=cv2.Sobel(r,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(r,cv2.CV_32F,0,1,ksize=3)
    c=cv2.GaussianBlur(gx*gx-gy*gy,(0,0),sig); s=cv2.GaussianBlur(2*gx*gy,(0,0),sig)
    return 0.5*np.arctan2(-s,-c)  # along the ridges
def count(r,a,b):
    n=int(np.hypot(*(np.array(b)-a)))*2; xs=np.linspace(a[0],b[0],n).astype(int); ys=np.linspace(a[1],b[1],n).astype(int)
    v=(r[np.clip(ys,0,r.shape[0]-1),np.clip(xs,0,r.shape[1]-1)]>0).astype(int)
    return int(np.sum(np.diff(v)==1))
def one(t):
    seed,fgp=t
    g.use('base')
    from ridgegen import finger as F
    m=F.master(seed,fgp); th=m.orientation; h,w=th.shape
    yy,xx=np.mgrid[0:h,0:w]
    cx,cy=(m.cores[0] if m.cores else (410,380))
    near=np.hypot(xx-cx,yy-cy)<200
    out={'seed':seed,'fgp':fgp,'pattern':m.pattern}
    for name,a in V.items():
        if a is None:
            r,n=m.ridges,0
        else:
            r,sp=pg.master_ridges(m,seed,fgp,*a); n=len(sp)
        o=orient(np.float32(r))
        e=np.abs(np.degrees(0.5*np.angle(np.exp(2j*(o-th)))))
        out[name]={'err_med':float(np.median(e[near])),'err_p95':float(np.percentile(e[near],95)),'spirals':n}
        if m.cores and m.deltas: out[name]['rc']=count(r,m.cores[0],m.deltas[0])
    return out
if __name__=="__main__":
    tasks=[(s,f) for s in (7000,7001,7002,7003) for f in (2,3,7,8)]
    with ProcessPoolExecutor(8) as p: rows=list(p.map(one,tasks))
    for n in V:
        print(n.ljust(16),'err med %.1f p95 %.1f | spirals %.0f | ridge count sum %d'%(np.median([r[n]['err_med'] for r in rows]),np.median([r[n]['err_p95'] for r in rows]),np.median([r[n]['spirals'] for r in rows]),sum(r[n].get('rc') or 0 for r in rows)), [r[n].get('rc') for r in rows])
    print([r['pattern'] for r in rows])
