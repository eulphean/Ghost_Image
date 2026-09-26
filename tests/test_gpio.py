"""GPIO edge detection. gpiozero is not imported."""

from __future__ import annotations

import numpy as np

from ghost_image.app import run
from ghost_image.config import Config, GpioConfig
from ghost_image.gpio import GpioControls, rising_press
from tests.test_app import FakeCamera, FakeDisplay


def test_rising_press_is_only_the_first_down_frame():
    assert not rising_press(False, False)
    assert rising_press(False, True)
    assert not rising_press(True, True)
    assert not rising_press(True, False)


def test_poll_reports_hold_once_per_press():
    states = iter([(False, False), (True, False), (True, False), (False, True)])
    controls = GpioControls(GpioConfig(enabled=True), levels=lambda: next(states))
    assert controls.poll() is None
    assert controls.poll() == "hold"
    assert controls.poll() is None
    assert controls.poll() == "release"


def test_disabled_gpio_never_polls():
    controls = GpioControls(GpioConfig(enabled=False), levels=lambda: (True, True))
    assert controls.poll() is None


def test_hold_button_freezes_the_frame_like_space(tmp_path, monkeypatch):
    monkeypatch.setattr("ghost_image.app.held_frame_path", lambda _config: tmp_path / "held.png")
    states = [(False, False), (True, False), (True, False)]
    index = {"i": 0}

    def levels() -> tuple[bool, bool]:
        value = states[min(index["i"], len(states) - 1)]
        index["i"] += 1
        return value

    controls = GpioControls(GpioConfig(enabled=True), levels=levels)
    frames = [
        np.full((120, 160, 3), 10, np.uint8),
        np.full((120, 160, 3), 10, np.uint8),
        np.full((120, 160, 3), 80, np.uint8),
    ]
    display = FakeDisplay([-1, -1, ord("q")])
    run(Config(), camera=FakeCamera(frames), display=display, buttons=controls)
    assert (tmp_path / "held.png").is_file()
    assert int(display.shown[2][110, 140, 0]) == 10
