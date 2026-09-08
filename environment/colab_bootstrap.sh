#!/usr/bin/env bash
# Bootstrap a neural-network-verification toolchain (Vehicle + Marabou) in a fresh Python 3.11
# virtual environment. Written for Google Colab (whose kernel is Python 3.12, for which Marabou has
# no wheels) but works on any x86_64 Linux or x86_64 macOS machine.
#
# Usage (Colab cell):   !bash colab_bootstrap.sh
# Then either add $VENV/bin to PATH or call $VENV/bin/vehicle directly.
set -euo pipefail

VENV="${VENV:-/content/venv}"
PYVER="${PYVER:-3.11}"
VEHICLE_VERSION="${VEHICLE_VERSION:-0.27.1}"
MARABOU_VERSION="${MARABOU_VERSION:-2.0.0}"

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  pip install --quiet uv
fi

if [ ! -x "$VENV/bin/python" ]; then
  log "creating Python $PYVER environment at $VENV"
  uv venv "$VENV" --python "$PYVER" --quiet
else
  log "reusing existing environment at $VENV"
fi

log "installing vehicle-lang==$VEHICLE_VERSION, maraboupy==$MARABOU_VERSION and helpers"
uv pip install --python "$VENV/bin/python" --quiet \
  "vehicle-lang==$VEHICLE_VERSION" \
  "maraboupy==$MARABOU_VERSION" \
  "numpy<2" idx2numpy onnx onnxruntime

log "checking installation"
"$VENV/bin/vehicle" --version
"$VENV/bin/python" -c "import maraboupy; print('maraboupy import OK')"
test -x "$VENV/bin/Marabou" && log "Marabou executable: $VENV/bin/Marabou"

log "done. Add to PATH with:  export PATH=\"$VENV/bin:\$PATH\""
