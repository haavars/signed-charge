"""Where the data and tools live. Everything can be set from the environment; see DATA.md."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(name, default):
    return Path(os.environ.get(name, default)).expanduser()


DATA = _path("SIGNED_CHARGE_DATA", ROOT / "data")
FVC = _path("FVC_ROOT", DATA / "fvc")  # holds FVC2000/DB1_B/ and so on
LEADER = _path("LEADER_ONNX", DATA / "leader.onnx")
SYNFING = _path("SYNFING_ROOT", DATA / "synfing")
ANGULI = _path("ANGULI_ROOT", DATA / "anguli")
RENDERED = _path("RENDERED_ROOT", DATA / "rendered")  # the diffusion render of ZP's masters
RENDERED_02 = _path("RENDERED_02_ROOT", DATA / "rendered_02")  # an earlier render, exploratory reads only
GENERATED = _path("GENERATED_ROOT", DATA / "generated")  # what scripts/generate_*.sh write
CACHE = GENERATED / "minutiae_parts" / "cache.pkl"

RESULTS = ROOT / "results" / "expected"  # results as first run, inputs to a few reads
EXCLUDED_FINGERS = ROOT / "manifests" / "excluded_fingers.json"

TOOLS = _path("RIDGEGEN_TOOLS", ROOT / "tools")
os.environ.setdefault("RIDGEGEN_TOOLS", str(TOOLS))
NBIS = Path(os.environ.get("NBIS_BIN", TOOLS / "nbis" / "bin"))


def require_fvc():
    """Stop if the FVC images aren't there: an empty glob would otherwise read as a run on no fingers."""
    if not any(FVC.glob("FVC*/DB*_B/*.tif")):
        raise SystemExit(f"No FVC images under {FVC}. Set FVC_ROOT; see DATA.md.")
