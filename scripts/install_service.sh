#!/usr/bin/env bash
# Install a systemd user service that starts Ghost Image fullscreen on
# login, restarts it if it exits, and waits until the graphical session exists.
#
# Usage:  ./scripts/install_service.sh
# The Pi also needs display.fullscreen: true in config.local.yaml.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

sed "s|@ROOT@|${ROOT}|g" "$ROOT/scripts/ghost-image.service.in" > "$UNIT_DIR/ghost-image.service"

systemctl --user daemon-reload
systemctl --user enable --now ghost-image.service
# Keep the user service running after logout / at boot, once the user has logged in once.
loginctl enable-linger "${USER}" || true

echo "Installed $UNIT_DIR/ghost-image.service"
echo "Logs: journalctl --user -u ghost-image -f"
