"""What data and tools are in place? check_data.py [--hash]

With --hash, every FVC B file is hashed against manifests/fvc_b.sha256, which takes a minute or so.
"""

import hashlib
import sys
from pathlib import Path

from signed_charge import paths


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = [line.split() for line in (paths.ROOT / "manifests/fvc_b.sha256").read_text().splitlines()]
    present = [name for _, name in manifest if (paths.FVC / name).exists()]
    print(f"FVC B images    {len(present)} of {len(manifest)} under {paths.FVC}")
    if "--hash" in sys.argv:
        wrong = [name for digest, name in manifest if (paths.FVC / name).exists() and sha256(paths.FVC / name) != digest]
        print(f"                {len(wrong)} differ from the manifest")
    expected = (paths.ROOT / "manifests/leader.sha256").read_text().split()[0]
    state = "missing" if not paths.LEADER.exists() else "as used here" if sha256(paths.LEADER) == expected else "differs from the one used here"
    print(f"LEADER          {state}: {paths.LEADER}")
    print(f"mindtct         {'present' if (paths.NBIS / 'mindtct').exists() else 'missing: run scripts/setup.sh'}")
    print(f"Anguli          {'present' if (paths.ANGULI / '1.jpg').exists() else 'missing'}: {paths.ANGULI}")
    print(f"SynFing         {'present' if (paths.SYNFING / 'impression_1/F_00000.jpg').exists() else 'missing'}: {paths.SYNFING}")
    print(f"Diffusion render {'present' if paths.RENDERED.exists() else 'missing'}: {paths.RENDERED}")
    return 0 if len(present) == len(manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
