"""What minutiae alone do to spacing and splay: stripes of fixed period with planted spirals, read by fanout.py (E-042): splay_null.py SCALE. Run from python_biometrics."""
import os
import sys
import numpy as np
import cv2
sc=sys.argv[1]; os.environ['FANOUT_SCALE']=sc
from signed_charge.measure import fanout as fo
from signed_charge.measure import signed_minutiae as sm
h,w=480,360
yy,xx=np.mgrid[0:h,0:w].astype(float)
mask=((xx-w/2)/(w/2-6))**2+((yy-h/2)/(h/2-6))**2<1
area_mm2=mask.sum()*fo.MM2
def make(n,kind,seed):
    rng=np.random.default_rng(seed)
    rot=rng.uniform(0,np.pi); v=-(xx-w/2)*np.sin(rot)+(yy-h/2)*np.cos(rot)
    psi=2*np.pi*v/10.4+rng.uniform(0,6.28)
    pts=[]
    if kind=='pairs':
        for _ in range(n//2):
            c=rng.uniform(40,w-40),rng.uniform(40,h-40); a=rng.uniform(0,6.28); d=rng.uniform(18,36)
            for p in (1,-1): psi+=p*np.arctan2(yy-c[1]-p*d/2*np.sin(a),xx-c[0]-p*d/2*np.cos(a))
    else:
        for _ in range(n):
            c=rng.uniform(30,w-30),rng.uniform(30,h-30)
            psi+=rng.choice([-1,1])*np.arctan2(yy-c[1],xx-c[0])
    return np.where(mask&(np.cos(psi)>0),40,235).astype(np.uint8)
for n,kind in ((0,'indep'),(int(.13*area_mm2),'indep'),(int(.20*area_mm2),'indep'),(int(.30*area_mm2),'indep'),(int(.20*area_mm2),'pairs')):
    rows=[]
    for s in range(10):
        im=make(n,kind,s)
        k,v=fo.measure((('x',s,1),im))
        if v: rows.append(v)
    print('scale',sc,kind,'density per 100mm2',round(n/area_mm2*100,1),'prints',len(rows),{k:round(float(np.median([r[k] for r in rows])),3) for k in ('splay_per_ridge','bend_per_ridge','ln_period_sd','coherence')},flush=True)
