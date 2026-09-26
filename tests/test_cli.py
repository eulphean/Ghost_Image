"""Smoke tests for ``python -m ghost_image``."""

from __future__ import annotations

import pytest

from ghost_image import __version__
from ghost_image.__main__ import main
from ghost_image.camera import CameraDevice


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_default_run_reports_version(capsys, monkeypatch):
    monkeypatch.setattr("ghost_image.__main__.run", lambda _config: 0)
    assert main([]) == 0
    assert f"Ghost Image v{__version__}" in capsys.readouterr().out


def test_show_config_prints_yaml(capsys):
    assert main(["--show-config"]) == 0
    out = capsys.readouterr().out
    assert "camera:" in out
    assert "ghost:" in out


def test_list_cameras_prints_discovery(monkeypatch, capsys):
    monkeypatch.setattr(
        "ghost_image.__main__.list_cameras",
        lambda _config: [CameraDevice(1, "Logitech", "/dev/video1", True)],
    )
    assert main(["--list-cameras"]) == 0
    assert "Logitech" in capsys.readouterr().out


def test_list_cameras_empty_is_an_error(monkeypatch, capsys):
    monkeypatch.setattr("ghost_image.__main__.list_cameras", lambda _config: [])
    assert main(["--list-cameras"]) == 1
    assert "no cameras found" in capsys.readouterr().out


def test_bad_config_path_returns_error(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
    assert "not found" in capsys.readouterr().err
