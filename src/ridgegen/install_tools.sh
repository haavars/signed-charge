#!/usr/bin/env bash
# The tools ridgegen uses, into RIDGEGEN_TOOLS: NIST NBIS (mindtct, bozorth3, cjpegl, and
# cwsq/dwsq/an2ktool for WSQ and ANSI/NIST-ITL files), NIST NFIQ 2 and NIST OpenLQM. Run it as
# `ridgegen install-tools`, which sets RIDGEGEN_TOOLS (ridgegen.tools) and RIDGEGEN_PYTHON (the
# Python ridgegen is installed in). Safe to re-run: what's there already is skipped.
#
# --diffusion adds the diffusion renderer's (ridgegen.diffusion): IMPOSE and taming-transformers
# at pinned commits, and IMPOSE's rolled-print checkpoint (about 320 MB, from the authors' Google
# Drive), checked against its pinned hash and converted once to safetensors, which is all
# ridgegen.diffusion loads (the checkpoint is a pickle, which can run code). It needs the
# `diffusion` extra installed, with CUDA torch.
#
# Building NBIS needs gcc, make, curl and unzip, and CMake older than 4 (fetched with uv when the
# system's is newer); NFIQ 2 and OpenLQM are unpacked from NIST's Ubuntu and Debian packages
# (dpkg-deb).
set -euo pipefail

TOOLS=${RIDGEGEN_TOOLS:?run this as ridgegen install-tools}
PYTHON=${RIDGEGEN_PYTHON:-python3}
NBIS_URL=${NBIS_URL:-https://nigos.nist.gov/nist/nbis/nbis_v5_0_0.zip}
NFIQ2_VERSION=${NFIQ2_VERSION:-2.3.0}
# NIST OpenLQM, the latent quality metric (Apache-2.0): its amd64 package, pinned.
OPENLQM_URL=https://github.com/usnistgov/openlqm/releases/download/v1.0.1/openlqm_1.0.1-1_amd64.deb
OPENLQM_SHA256=f8c1aabb463979a68e409d755286ff973c734b2f83aa6aa0a737089d1434e658
# mindtct, bozorth3 and cjpegl verify prints; cwsq, dwsq and an2ktool make and check WSQ and ANSI/NIST-ITL files.
NBIS_TOOLS=(mindtct bozorth3 cjpegl cwsq dwsq an2ktool)
IMPOSE_COMMIT=c46f36baa32b61eb2e4fadd3b3797d2a57365054
TAMING_COMMIT=3ba01b241669f5ade541ce990f7650a3b8f65318
IMPOSE_ROLLED_FOLDER=https://drive.google.com/drive/folders/1p6hCoPb1xrYsKLMbxDxQ6bPqnTmhKeVM
# The authors' checkpoint as downloaded on 2026-09-23, and the weights converted from it.
IMPOSE_CKPT_SHA256=bf29d6aeaf901b5cf1b8399ca1e3cf4a2cefc69ab5edcb17b3fbf8f42c70dd48
IMPOSE_WEIGHTS_SHA256=cab6ea2ded7df5407a8fc5736dae682e142cd45c100a6c0f3e7e34a01a22a0f1

DIFFUSION=0
for arg in "$@"; do
  case "$arg" in
    --diffusion) DIFFUSION=1 ;;
    *) echo "Unknown option: $arg (see ridgegen install-tools --help)"; exit 1 ;;
  esac
done

mkdir -p "$TOOLS"
TOOLS="$(cd "$TOOLS" && pwd)"
echo "==> Tools directory: $TOOLS"

# A CMake older than 4 on PATH: NBIS bundles libraries whose CMakeLists predate CMake 4.
old_cmake() {
  local build=$1 major
  # A cmake on PATH may be missing or broken (a pip shim without its package): then fetch one.
  major="$( (cmake --version 2> /dev/null || true) | sed -n 's/^cmake version \([0-9]*\).*/\1/p')"
  if [ -n "$major" ] && [ "$major" -lt 4 ]; then
    return
  fi
  command -v uv > /dev/null || { echo "!! NBIS needs CMake older than 4, or uv to fetch one"; exit 1; }
  echo "==> CMake 3 for NBIS (with uv)"
  uv pip install -q --target "$build/cmake" "cmake<4" \
    || { echo "!! Couldn't fetch CMake 3 with uv"; exit 1; }
  # The wheel's own binaries, not its Python wrappers, which need the wheel on PYTHONPATH.
  export PATH="$build/cmake/cmake/data/bin:$PATH"
}

install_nbis() {
  local tool missing=0
  for tool in "${NBIS_TOOLS[@]}"; do
    [ -x "$TOOLS/nbis/bin/$tool" ] || missing=1
  done
  if [ "$missing" = 0 ]; then
    echo "==> NBIS already in $TOOLS/nbis"
    return
  fi
  echo "==> Building NIST NBIS 5.0 (${NBIS_TOOLS[*]}) - a few minutes"
  local build="$TOOLS/build"
  rm -rf "$build" && mkdir -p "$build"
  curl -fsSL -o "$build/nbis.zip" "$NBIS_URL"
  unzip -q "$build/nbis.zip" -d "$build"
  (
    old_cmake "$build"
    # GCC 14+ makes old C's implicit declarations errors, and GCC 15 defaults to C23: the bundled libraries
    # built with CMake (OpenJPEG and others) take CFLAGS.
    export CFLAGS="${CFLAGS:-} -std=gnu17 -fpermissive"
    cd "$build/Rel_5.0.0"
    step() { local log="$build/$1.log"; shift; "$@" > "$log" 2>&1 || { echo "!! NBIS $* failed, see $log"; exit 1; }; }
    mkdir -p "$build/install"
    step setup ./setup.sh "$build/install" --without-X11 --64
    # GCC 10+ defaults to -fno-common, which NBIS's tentative definitions don't link with; NBIS's own code is
    # built as -ansi, and -fpermissive keeps GCC 14+ from failing on its implicit declarations.
    sed -i 's/^\(ARCH_FLAG[[:space:]]*:=.*\)$/\1 -fcommon -fpermissive/' rules.mak
    step config make config
    step build make it
    step install make install LIBNBIS=no
  )
  mkdir -p "$TOOLS/nbis/bin"
  for tool in "${NBIS_TOOLS[@]}"; do
    cp "$build/install/bin/$tool" "$TOOLS/nbis/bin/"
  done
  rm -rf "$build"
}

install_nfiq2() {
  if [ -x "$TOOLS/nfiq2/bin/nfiq2" ]; then
    echo "==> NFIQ 2 already in $TOOLS/nfiq2"
    return
  fi
  # NIST publishes NFIQ 2 builds as Ubuntu packages; unpack one without installing it.
  local release
  release="$(. /etc/os-release 2>/dev/null && [ "${ID:-}" = ubuntu ] && echo "${VERSION_ID%%.*}" || true)"
  if [ -z "$release" ] || ! command -v dpkg-deb > /dev/null; then
    echo "!! NFIQ 2: no prebuilt package for this OS. Build https://github.com/usnistgov/NFIQ2 and set"
    echo "   NFIQ2_BIN and NFIQ2_MODEL (see ridgegen.verify). Verification stays off until then."
    return
  fi
  # NIST builds for Ubuntu 20, 22 and 24; a newer Ubuntu takes the newest of those.
  local tmp candidate got=""
  tmp="$(mktemp -d)"
  for candidate in "$release" 24 22 20; do
    [ "$candidate" -le "$release" ] || continue
    if curl -fsSL -o "$tmp/nfiq2.deb" \
      "https://github.com/usnistgov/NFIQ2/releases/download/v$NFIQ2_VERSION/nfiq2_$NFIQ2_VERSION-1_ubuntu${candidate}_amd64.deb" \
      2> /dev/null; then
      got=$candidate
      break
    fi
  done
  if [ -z "$got" ]; then
    echo "!! NFIQ 2: no package of $NFIQ2_VERSION for Ubuntu $release or older. Verification stays off."
    rm -rf "$tmp"
    return
  fi
  echo "==> Installed NIST NFIQ 2 $NFIQ2_VERSION (the Ubuntu $got package)"
  dpkg-deb -x "$tmp/nfiq2.deb" "$tmp/root"
  rm -rf "$TOOLS/nfiq2"
  mv "$tmp/root/usr/local/nfiq2" "$TOOLS/nfiq2"
  rm -rf "$tmp"
}

install_openlqm() {
  if [ -x "$TOOLS/openlqm/bin/openlqm" ]; then
    echo "==> OpenLQM already in $TOOLS/openlqm"
    return
  fi
  if ! command -v dpkg-deb > /dev/null; then
    echo "!! OpenLQM: dpkg-deb is needed to unpack NIST's package. Latents are scored without it."
    return
  fi
  echo "==> Installing NIST OpenLQM 1.0.1 (latent quality, about 16 MB)"
  local tmp
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/openlqm.deb" "$OPENLQM_URL"
  echo "$OPENLQM_SHA256  $tmp/openlqm.deb" | sha256sum --check --quiet \
    || { echo "!! OpenLQM's package doesn't match its pinned hash"; rm -rf "$tmp"; exit 1; }
  dpkg-deb -x "$tmp/openlqm.deb" "$tmp/root"
  rm -rf "$TOOLS/openlqm"
  mv "$tmp/root/usr/local/openlqm" "$TOOLS/openlqm"
  rm -rf "$tmp"
}

checkout() {  # checkout <url> <dir> <commit>
  if [ ! -d "$2/.git" ]; then
    git clone --quiet "$1" "$2"
  fi
  git -C "$2" fetch --quiet --depth 1 origin "$3" 2> /dev/null || true
  git -C "$2" checkout --quiet "$3"
}

install_diffusion() {
  "$PYTHON" -c "import torch, gdown, safetensors" 2> /dev/null \
    || { echo "!! The diffusion renderer needs ridgegen's diffusion extra (with CUDA torch) installed first"; exit 1; }
  echo "==> IMPOSE and taming-transformers at pinned commits"
  checkout https://github.com/Yu-Yy/IMPOSE.git "$TOOLS/impose" "$IMPOSE_COMMIT"
  checkout https://github.com/CompVis/taming-transformers.git "$TOOLS/taming-src" "$TAMING_COMMIT"
  local models="$TOOLS/impose/models/fingerprint_ldm_rolled_512"
  local ckpt="$models/model_ldm_rolled.ckpt" weights="$models/model_ldm_rolled.safetensors"
  if [ ! -f "$weights" ]; then
    if [ ! -f "$ckpt" ]; then
      echo "==> Downloading IMPOSE's rolled-print checkpoint"
      "$PYTHON" -m gdown --quiet --folder "$IMPOSE_ROLLED_FOLDER" -O "$models"
    fi
    echo "$IMPOSE_CKPT_SHA256  $ckpt" | sha256sum --check --quiet \
      || { echo "!! $ckpt isn't the checkpoint this was set up with; not loading it"; exit 1; }
    echo "==> Converting it to safetensors (read without running code)"
    "$PYTHON" -c "import sys; from ridgegen import diffusion; diffusion.convert(sys.argv[1], sys.argv[2])" "$ckpt" "$weights"
  fi
  echo "$IMPOSE_WEIGHTS_SHA256  $weights" | sha256sum --check --quiet \
    || { echo "!! $weights doesn't match its pinned hash; delete it and run again"; exit 1; }
}

install_nbis
install_nfiq2
install_openlqm
if [ "$DIFFUSION" = 1 ]; then
  install_diffusion
fi

echo ""
echo "Tools installed in $TOOLS"
