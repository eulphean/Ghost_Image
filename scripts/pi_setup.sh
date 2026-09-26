#!/usr/bin/env bash
# One-time setup on a Raspberry Pi (64-bit Raspberry Pi OS).
# Installs system packages, creates the virtual environment and installs the
# pinned Python dependencies. Safe to re-run.
#
# Usage:  ./scripts/pi_setup.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "warning: expected a 64-bit (aarch64) Raspberry Pi OS; found $(uname -m)." >&2
    echo "         MediaPipe does not ship wheels for 32-bit ARM." >&2
fi

echo "==> Installing system packages"
sudo apt-get update
sudo apt-get install -y \
    python3 python3-venv python3-dev \
    libgl1 libglib2.0-0 libatlas3-base \
    v4l-utils

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "==> System Python is ${PY_VERSION}"
python3 - <<'EOF'
import sys
if sys.version_info < (3, 11):
    sys.exit("error: Python 3.11 or newer is required.")
EOF

if [[ ! -d .venv ]]; then
    echo "==> Creating virtual environment in .venv"
    python3 -m venv .venv
fi

echo "==> Installing Python dependencies"
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> Downloading segmentation models"
./scripts/download_models.sh

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Run with:             python -m ghost_image"
