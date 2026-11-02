"""D-067's population against real prints (D-067): lead_read.py CACHE.pkl OUT.json RUN.npz...

`flow_read`'s charge through real windows and classifier against FVC DB1-3 B, on one arm, `lead`.

Run: python -m signed_charge.reads.lead_read CACHE.pkl OUT.json RUN.npz...
"""
import sys
from pathlib import Path

from signed_charge.reads import flow_read as fl

fl.ARMS = ("lead",)
fl.SHOWN = (*fl.ARMS, "population", "sfinge", "model2", "e027")

if __name__ == "__main__":
    fl.main(sys.argv[1], sys.argv[2], sys.argv[3:])
