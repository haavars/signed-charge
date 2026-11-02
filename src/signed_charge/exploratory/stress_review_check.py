"""Checks of a reviewer's predictions, front against field on the model alone (D-033). Run: python -m signed_charge.exploratory.stress_review_check"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import map_coordinates
from signed_charge.exploratory import stress_read as sr
from signed_charge.exploratory import growth_stress as gs
from signed_charge.generate import competency_model as cm
from signed_charge.generate import glover_model as g
points,tris,s,phi,index=g.mesh(6.0)
rows,cols,weights=cm.graph(points,index)
n=len(points); mid=len(phi)//2
def arrive(sites,w0,speed=0.7):
    times=[]
    for v,d in sites:
        r=np.concatenate([rows,np.full(len(v),n)]); c=np.concatenate([cols,v]); w=np.concatenate([w0,np.full(len(v),d*speed+1e-6)])
        times.append(dijkstra(sp.csr_matrix((w,(r,c)),shape=(n+1,n+1)),directed=False,indices=n)[:n]/speed)
    return np.array(times)
def crossing(arr,angle,ss,pp,mask):
    I,J=np.meshgrid(ss/(s[1]-s[0]),pp/(phi[1]-phi[0]),indexing='ij')
    a=map_coordinates(arr,[I,J],order=1,mode='nearest')
    ring=g.R*np.cos(np.clip((ss-g.L)/g.R,0,np.pi/2))
    ts,tp=np.gradient(a,ss,axis=0),np.gradient(a,pp,axis=1)/np.maximum(ring,1.0)[:,None]
    c=np.abs((np.arctan2(tp,ts)+np.pi/2-angle+np.pi/2)%np.pi-np.pi/2)
    return round(float(np.degrees(np.median(c[mask]))),1),round(float((c[mask]>np.pi/4).mean()),2)
def seam(times):
    """midline: where nail and crease meet, kink across the midline of the nail wave, and the crossing angle of fronts at the meeting"""
    nail,crease=times[0][index],times[1][index]
    i=int(np.argmin(np.abs(nail[:,mid]-crease[:,mid])))
    kink=nail[:,mid-1]-2*nail[:,mid]+nail[:,mid+1]
    def grad(t):
        ring=g.R*np.cos(np.clip((s-g.L)/g.R,0,np.pi/2))
        return np.gradient(t,s,axis=0),np.gradient(t,phi,axis=1)/np.maximum(ring,1.0)[:,None]
    a,b=grad(nail),grad(crease)
    def angle(i,j):
        u=np.array([a[0][i,j],a[1][i,j]]); v=np.array([b[0][i,j],b[1][i,j]])
        return np.degrees(np.arccos(np.clip(abs(u@v)/np.linalg.norm(u)/np.linalg.norm(v),0,1)))
    side=[int(np.argmin(np.abs(nail[:,j]-crease[:,j]))) for j in (mid-30,mid+30)]
    return {'seam_s_mid':float(s[i]),'seam_s_side':[float(s[k]) for k in side],'kink_min':round(float(kink[:i+10].min()),1),'angle_off_mid':[round(angle(side[0],mid-30),1),round(angle(side[1],mid+30),1)]}
# 1. the arch, nail arc over the tip only, at several reaches
sites=cm.site_vertices(points,index,s,'arch',(0,0,0,0))
for top in (0.5,0.75,1.0):
    keep=s>=g.L+(1-top)*(s[-1]-g.L)
    nail=np.unique(np.concatenate([index[keep,0],index[keep,-1]]))
    print('arch arc reach',top,len(nail),seam(arrive([(nail,sites[0][1]),sites[1]],weights)),flush=True)
# 2. metric saturated, so a small a doesn't switch it off
for pattern,seed in (('loop',None),('loop',4502)):
    person=(0,0,0,0) if seed is None else cm.person_of(seed,1.0)
    r,(angle,w,ss,pp)=sr.read(pattern,person,**gs.settings(pattern))
    mask=np.array([[sr.contact(a,b,ss.max()) for b in pp] for a in ss])
    res=gs.solve(pattern,person,**gs.settings(pattern))
    tt=gs.outward(points,tris); pp2=points[tt]; fn=np.cross(pp2[:,1]-pp2[:,0],pp2[:,2]-pp2[:,0])
    vn=np.zeros_like(points)
    for k in range(3): np.add.at(vn,tt[:,k],fn)
    vn/=np.linalg.norm(vn,axis=1,keepdims=True)
    d,a,_,_=gs.along(res['tensor'],vn)
    e=(points[cols]-points[rows])/weights[:,None]
    along=0.5*(np.sum(e*d[rows],1)**2+np.sum(e*d[cols],1)**2)
    sites=cm.site_vertices(points,index,s,pattern,person)
    print(pattern,seed,'plain',crossing(arrive(sites,weights).min(0)[index],angle,ss,pp,mask),flush=True)
    for sat in (0.3,0.1):
      am=np.minimum(0.5*(a[rows]+a[cols])/sat,1.0)
      for gamma in (3,6,12):
        w1=weights*np.sqrt(np.maximum(1-am*(1-gamma**-2)*along,1e-3))
        t1=arrive(sites,w1)
        print('  sat',sat,'gamma',gamma,'all',crossing(t1.min(0)[index],angle,ss,pp,mask),'pad alone',crossing(t1[2][index],angle,ss,pp,mask),flush=True)
