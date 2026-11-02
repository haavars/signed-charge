"""Are two runs' arrays the same? compare_npz.py DIR_A DIR_B [GLOB]: each .npz in A against its namesake in B, exactly."""
import sys
from pathlib import Path

import numpy as np


def main(a, b, pattern="*.npz"):
    same = differ = missing = 0
    for path in sorted(Path(a).glob(pattern)):
        other = Path(b) / path.name
        if not other.exists():
            missing += 1
            continue
        x, y = np.load(path, allow_pickle=True), np.load(other, allow_pickle=True)
        keys = set(x.files) | set(y.files)
        bad = [k for k in keys if k not in x.files or k not in y.files or not np.array_equal(x[k], y[k])]
        if bad:
            differ += 1
            print(f"DIFFERS {path.name}: {sorted(bad)}")
        else:
            same += 1
    print(f"{same} identical, {differ} differ, {missing} without a namesake")
    return 1 if differ or missing else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
