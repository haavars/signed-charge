"""Two generators from outside this project against real prints (D-066): outside_read.py CACHE.pkl OUT.json IMAGE...

`flow_read`'s charge through real windows and classifier against FVC DB1-3 B, on image files named SET_ID: SynFing, a GAN's rolled prints, and Anguli, a formula generator. Both taken as 500 ppi, greyscale, with nothing rescaled or filtered.

Run: python -m signed_charge.reads.outside_read CACHE.pkl OUT.json IMAGE...
"""
import sys
from pathlib import Path

import cv2

from signed_charge.reads import field_read as fr
from signed_charge.reads import flow_read as fl

fl.ARMS = ("synfing", "anguli")
fl.SHOWN = (*fl.ARMS, "population", "sfinge", "model2", "e027")
fr.sheet = lambda path: cv2.imread(str(path), 0)

if __name__ == "__main__":
    fl.main(sys.argv[1], sys.argv[2], sys.argv[3:])
