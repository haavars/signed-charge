"""How much of a master's flow demand `phase_global.period_field` absorbs, by smoothness (E-042). Run from python_biometrics."""
import sys
import numpy as np
import time
from signed_charge.generate import phase_global as pg
from ridgegen import finger as fingers
from ridgegen.synthesis import rng_for
rows=[]
for seed,fgp in ((7000,2),(7000,3),(7001,4),(7002,7),(7003,8),(7004,2)):
    m=fingers.master(seed,fgp); period=m.period; singular=m.cores+m.deltas
    barrier=pg.cuts(m.ridges.shape,singular); beta=pg.unwrap_direction(m.orientation,barrier)
    base=None
    out=[]
    for lam,bound in ((None,None),(1,9),(2,9),(3,9),(10,9)):
        t=time.time()
        field=period if lam is None else pg.period_field(beta,period,singular,lam,bound)
        k=2*np.pi/field
        gx,gy=k*np.cos(beta),k*np.sin(beta)
        ok=(np.abs(np.diff(beta,axis=1))[:-1,:]<np.pi/2)&(np.abs(np.diff(beta,axis=0))[:,:-1]<np.pi/2)
        rho=np.where(ok,np.diff(gy,axis=1)[:-1,:]-np.diff(gx,axis=0)[:,:-1],0)/(2*np.pi)
        import cv2
        r=cv2.GaussianBlur(rho.astype(np.float32),(0,0),2*period)  # net over a few periods
        e=40; r=r[e:-e,e:-e]
        g=float(np.abs(r).sum()); base=base or g
        sd=0 if lam is None else float(np.log(field/period).std())
        out.append((lam,bound,round(sd,3),round(g/base,2)))
    print(seed,fgp,m.ridges.shape,round(float(period),2),out,round(time.time()-t,2),flush=True)
    rows.append(out)
a=np.array([[x[2:] for x in r] for r in rows],float)
print('mean sd and demand ratio by setting',[(r[0],r[1],round(float(a[:,i,0].mean()),3),round(float(a[:,i,1].mean()),2)) for i,r in enumerate(rows[0])])
