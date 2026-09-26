"""Main-loop tests with a fake camera and display. No window is opened."""

from __future__ import annotations

import numpy as np

from ghost_image.app import present, run
from ghost_image.config import Config


class FakeCamera:
    def __init__(self, frames: list[np.ndarray | None]) -> None:
        self._frames = list(frames)
        self.released = False

    def read(self) -> np.ndarray | None:
        if not self._frames:
            return np.zeros((8, 8, 3), np.uint8)
        return self._frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeDisplay:
    def __init__(self, keys: list[int]) -> None:
        self._keys = list(keys)
        self.shown: list[np.ndarray] = []
        self.fullscreen = False
        self.closed = False
        self.toggles = 0

    def show(self, frame: np.ndarray) -> int:
        self.shown.append(frame)
        if not self._keys:
            return -1
        return self._keys.pop(0)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.toggles += 1

    def close(self) -> None:
        self.closed = True


def test_present_draws_fps():
    frame = np.zeros((40, 80, 3), np.uint8)
    out = present(frame, fps=15.0, show_fps=True)
    assert not np.array_equal(out, frame)


def test_present_can_hide_fps():
    frame = np.zeros((40, 80, 3), np.uint8)
    assert np.array_equal(present(frame, fps=15.0, show_fps=False), frame)


def test_quit_key_stops_and_closes():
    camera = FakeCamera([np.zeros((8, 8, 3), np.uint8)])
    display = FakeDisplay([ord("q")])
    assert run(Config(), camera=camera, display=display) == 0
    assert display.closed
    assert len(display.shown) == 1
    assert not camera.released  # caller-supplied cameras stay open


def test_fullscreen_key_toggles():
    camera = FakeCamera([np.zeros((8, 8, 3), np.uint8)] * 2)
    display = FakeDisplay([ord("f"), ord("q")])
    run(Config(), camera=camera, display=display)
    assert display.toggles == 1
    assert display.fullscreen


def test_keyboard_interrupt_closes_display():
    class Boom:
        def read(self) -> np.ndarray:
            raise KeyboardInterrupt

        def release(self) -> None:
            self.released = True

    camera = Boom()
    display = FakeDisplay([])
    assert run(Config(), camera=camera, display=display) == 0
    assert display.closed


def test_owned_camera_is_released(monkeypatch):
    released = {"yes": False}

    class Owned:
        def read(self) -> np.ndarray:
            return np.zeros((4, 4, 3), np.uint8)

        def release(self) -> None:
            released["yes"] = True

    monkeypatch.setattr("ghost_image.app.open_camera", lambda _config: Owned())
    display = FakeDisplay([ord("q")])
    run(Config(), display=display)
    assert released["yes"]
