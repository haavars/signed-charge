"""The last line of `extractor_forks.py` (LEADER on Phantom's rendered fingertips), which hit a time limit: extractor_forks_phantom.py OUT"""
import sys
import json
from concurrent.futures import ProcessPoolExecutor
from signed_charge.measure import conformality as cf
from signed_charge.reads import minutiae_pairs as mp
from signed_charge.measure import minutiae_stats as ms
from signed_charge.measure import validate_masters as vm
from signed_charge.measure import provenance
if __name__=="__main__":
    started=provenance.start()
    items=list(cf.phantom(30))
    with ProcessPoolExecutor(4,initializer=vm.set_extractor,initargs=("leader",)) as pool:
        stats=[v for _,v in pool.map(ms.describe,items,chunksize=4) if v]
        pairs=[v for _,v in pool.map(mp.describe,items,chunksize=4) if v]
    s,p=ms.summary(stats),mp.summary(pairs); c=p["clustering"]
    print("leader phantom_rendered",len(stats),"dens %.1f forks %.2f"%(s["density_per_100mm2"],s["bifurcations"]),"same/off/face %.2f %.2f %.2f"%(p["close_same"],p["close_offset"],p["close_facing"]),"EE/EB/BB %.2f %.2f %.2f"%(p["close_EE"],p["close_EB"],p["close_BB"]),"c10/20/40 %.2f %.2f %.2f"%(c[10],c[20],c[40]))
    json.dump({"leader_phantom_rendered":{"stats":s,"pairs":{k:v for k,v in p.items() if k!="nn_share"}},"_provenance":provenance.record(started)},open(sys.argv[1],"w"),indent=1)
