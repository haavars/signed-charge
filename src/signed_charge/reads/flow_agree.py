"""(E-089) Exploratory, unregistered: flow_charge's agreement on D-055's arms, on the read flow, as E-084 did for field prints."""

import sys
from pathlib import Path
from signed_charge.measure import flow_charge as fc
from signed_charge.reads import field_read as fr
from signed_charge.measure import minutiae_cache as mc


def row(path):
    got = mc.row("field", "PH" + Path(path).stem, fr.sheet(path))
    return dict(got, path=str(path)) if got else None


fc.rm.field_row = row
fc.SETS = ()
fc.main(sys.argv[1], sys.argv[2], sys.argv[3:])
