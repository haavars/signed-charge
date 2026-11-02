"""Where the external tools live: NIST NBIS, NFIQ 2 and OpenLQM, and the diffusion renderer's IMPOSE code and
weights. `ridgegen install-tools` puts them there.

`RIDGEGEN_TOOLS` names the directory; without it, `ridgegen/tools` in the user's data directory
(`$XDG_DATA_HOME`, by default `~/.local/share`). Read on every call, so it can be set after import. Single tools
can still be pointed at one by one (`NBIS_BIN`, `NFIQ2_BIN`, `OPENLQM_BIN`, `IMPOSE_CHECKPOINT`).
"""

import os
from pathlib import Path


def root():
    configured = os.environ.get("RIDGEGEN_TOOLS")
    if configured:
        return Path(configured).expanduser()
    data = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(data) / "ridgegen" / "tools"


def path(*parts):
    """A path inside the tools directory."""
    return root().joinpath(*parts)
