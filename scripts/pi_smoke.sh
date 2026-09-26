#!/usr/bin/env bash
# Confirm a fresh Pi clone can see the USB camera and read frames.
# Run after ./scripts/pi_setup.sh. Prints the baseline FPS (no window, no
# segmentation) so processing resolution can be chosen later.
#
# Usage:  ./scripts/pi_smoke.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
    echo "error: .venv is missing. Run ./scripts/pi_setup.sh first." >&2
    exit 1
fi

echo "==> Cameras"
.venv/bin/python -m ghost_image --list-cameras

echo "==> Baseline capture (5s, no window)"
.venv/bin/python -m ghost_image --benchmark-seconds 5

echo
echo "Baseline recorded above. Open the live view with:"
echo "    source .venv/bin/activate && python -m ghost_image"
echo "Press q to quit."
