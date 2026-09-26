"""The kiosk unit file. systemd itself is not invoked."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_service_template_restarts_and_waits_for_the_display():
    text = (ROOT / "scripts" / "ghost-image.service.in").read_text(encoding="utf-8")
    assert "After=graphical-session.target" in text
    assert "Restart=on-failure" in text
    assert "WantedBy=graphical-session.target" in text
    assert "@ROOT@/.venv/bin/python -m ghost_image" in text
    assert "disable_blanking.sh" in text


def test_installer_substitutes_the_project_root():
    text = (ROOT / "scripts" / "install_service.sh").read_text(encoding="utf-8")
    assert "ghost-image.service.in" in text
    assert "systemctl --user enable --now" in text
    assert "loginctl enable-linger" in text
