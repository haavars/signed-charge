"""Exploratory, on planar prints only (E-084): minutiae against the flow they were laid on, and against the flow read from them, by smoothing scale; and K(L). Run from python_biometrics as `flow_charge`, no arguments."""
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from pathlib import Path
from signed_charge.measure import flow_charge as fc
from signed_charge.measure import field_similarity as fs
from signed_charge.reads import signed_curves as sc
from signed_charge.reads import signed_false as sf
from signed_charge.measure import signed_minutiae as sm
from signed_charge.reads import field_read as fr
from signed_charge.reads import minutiae_parts as parts
from signed_charge.reads import real_mismatch as rm
from signed_charge import paths
SCALES=(12,30,50)
def one(path):
    row=rm.field_row(path)
    im=fr.sheet(path); d=np.load(path)
    inner=parts.mask_of(row); fg=fs.foreground(im)
    ys,xs=np.nonzero(inner); yy,xx=np.mgrid[0:im.shape[0],0:im.shape[1]]
    disc=inner&(np.hypot(xx-xs.mean(),yy-ys.mean())<=200)
    guarded=disc.copy()
    for x,y in sc.singular(fs.field(im),fg): guarded&=np.hypot(xx-x,yy-y)>=sc.GUARD
    read=sm.direction(im)
    th=d["theta"]; true=np.exp(2j*th)*-1  # unit double angle of the gradient, as sm.direction's
    out={}
    p=sf.inside(row["points"]["mindtct"],guarded); h=sf.heading(p,-1.0); cy,cx=p[:,1].astype(int),p[:,0].astype(int)
    for name,f in (("read",read),("true",true)):
        for s in SCALES:
            a,dv=fc.spreading(f,float(s))
            al=np.sign(np.cos(h)*np.cos(a[cy,cx])+np.sin(h)*np.sin(a[cy,cx]))
            out[f"{name}{s}"]=(float(np.sign(dv[cy,cx]*al).sum()),len(p))
    pts=sf.inside(row["points"]["mindtct"],disc)
    for L in (40,60,80,100):
        out[f"K{L}"]=fc.box_pairs(pts,disc,L)
    return out
rows=list(ProcessPoolExecutor(4).map(one,sorted(glob.glob(str(paths.GENERATED / "field_d049/field_*.npz")))[::2],chunksize=2))
for k in ("read12","read30","read50","true12","true30","true50"):
    a=np.array([r[k] for r in rows]).sum(0); print(k, round(a[0]/a[1],3), int(a[1]))
for L in (40,60,80,100):
    v=[r[f"K{L}"][0]/r[f"K{L}"][1]-1 for r in rows if r[f"K{L}"] and r[f"K{L}"][1]>0]
    print("K",L,round(float(np.mean(v)),3),round(float(np.std(v)/np.sqrt(len(v))),3),len(v))
