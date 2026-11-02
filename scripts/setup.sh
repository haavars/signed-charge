#!/usr/bin/env bash
# Installs the pinned Python packages into .venv and builds NBIS, NFIQ 2 and OpenLQM into tools/. Safe to re-run.
# EXTRAS="--extra matcher" adds PyTorch for the matcher test, on a CUDA 13 machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv sync --frozen ${EXTRAS:-}
RIDGEGEN_TOOLS=${RIDGEGEN_TOOLS:-$PWD/tools} RIDGEGEN_PYTHON=$PWD/.venv/bin/python bash src/ridgegen/install_tools.sh "$@"
