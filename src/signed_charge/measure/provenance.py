"""Where a results file came from, written into it (README.md): commit, uncommitted changes, command,
ridgegen version, host and times. `started = provenance.start()` at the beginning of a run, then
`report["_provenance"] = provenance.record(started)` before writing it.
"""
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from signed_charge import paths

REPO = paths.ROOT


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def start():
    paths.require_fvc()
    return time.time()


def record(started):
    """The run's provenance, as a dict for a results file."""
    try:
        import ridgegen

        version = ridgegen.__version__
    except Exception:  # noqa: BLE001 - provenance mustn't fail a run
        version = None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": (_git("rev-parse", "HEAD") or "").strip() or None,
        # Tracked files changed since the commit: the run's code may not be exactly the commit's.
        "uncommitted_changes": sorted(line[3:] for line in dirty.splitlines() if line.strip()) if dirty else [],
        "command": [os.path.basename(sys.executable)] + sys.argv,
        "cwd": os.getcwd(),
        "ridgegen": version,
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "env": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "RIDGEGEN_TOOLS", "PYTHONPATH") if k in os.environ},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
