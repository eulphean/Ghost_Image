"""The Windows setup script. PowerShell is not invoked."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_windows_setup_creates_venv_and_fetches_models():
    text = (ROOT / "scripts" / "windows_setup.ps1").read_text(encoding="utf-8")
    assert "python" in text
    assert ".venv" in text
    assert "requirements.txt" in text
    assert "selfie_segmenter.tflite" in text
    assert "3, 11" in text


def test_windows_setup_bat_launches_the_powershell_script():
    text = (ROOT / "scripts" / "windows_setup.bat").read_text(encoding="utf-8")
    assert "windows_setup.ps1" in text
    assert "ExecutionPolicy Bypass" in text


def test_windows_run_uses_the_venv_python():
    text = (ROOT / "scripts" / "windows_run.bat").read_text(encoding="utf-8")
    assert r".venv\Scripts\python.exe" in text
    assert "ghost_image" in text
