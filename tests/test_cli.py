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


def test_benchmark_seconds_prints_fps(monkeypatch, capsys):
    class Cam:
        def release(self) -> None:
            self.released = True

    cam = Cam()
    monkeypatch.setattr("ghost_image.__main__.open_camera", lambda _config: cam)
    monkeypatch.setattr(
        "ghost_image.__main__.benchmark_capture",
        lambda _camera, _seconds: {
            "frames": 10.0,
            "elapsed": 1.0,
            "fps": 10.0,
            "cpu_seconds": 0.2,
        },
    )
    assert main(["--benchmark-seconds", "1"]) == 0
    assert "fps=10.0" in capsys.readouterr().out
    assert cam.released


def test_no_restore_held_overrides_config(monkeypatch):
    seen: dict[str, bool] = {}

    def fake_run(config):
        seen["restore"] = config.paths.restore_held
        return 0

    monkeypatch.setattr("ghost_image.__main__.run", fake_run)
    assert main(["--no-restore-held"]) == 0
    assert seen["restore"] is False


def test_benchmark_seconds_rejects_zero():
    assert main(["--benchmark-seconds", "0"]) == 2


def test_list_cameras_empty_is_an_error(monkeypatch, capsys):
    monkeypatch.setattr("ghost_image.__main__.list_cameras", lambda _config: [])
    assert main(["--list-cameras"]) == 1
    assert "no cameras found" in capsys.readouterr().out


def test_bad_config_path_returns_error(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
    assert "not found" in capsys.readouterr().err
