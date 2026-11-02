"""The FVC A sets' split, by the rule in docs/held-out-protocol.md: held_out_split.py OUT.json

Within each set, fingers sorted by sha256 of `D-030/<year>/<db>/<finger>` as text: the first 30 are checking, the other 70 proving. Writes the checking fingers only.

Run anywhere: python3 research/realism/held_out_split.py research/realism/held_out_checking.json
"""

import hashlib
import json
import sys

YEARS = ("FVC2000", "FVC2002", "FVC2004")
DBS = ("DB1", "DB2", "DB3", "DB4")
CHECKING = 30


def checking(year, db):
    rank = lambda finger: hashlib.sha256(f"D-030/{year}/{db}/{finger}".encode()).hexdigest()
    return sorted(sorted(range(1, 101), key=rank)[:CHECKING])


if __name__ == "__main__":
    json.dump({f"{y}/{d}_A": checking(y, d) for y in YEARS for d in DBS}, open(sys.argv[1], "w"), indent=0)
