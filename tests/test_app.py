"""Main-loop tests with a fake camera and display. No window is opened."""

from __future__ import annotations

import numpy as np
import pytest

from ghost_image.app import (
    Session,
    average_frames,
    benchmark_capture,
    format_benchmark,
    present,
    run,
)
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


def test_benchmark_capture_counts_frames_on_a_fake_clock():
    now = {"t": 0.0}

    class Cam:
        def read(self) -> np.ndarray:
            now["t"] += 0.1
            return np.zeros((2, 2, 3), np.uint8)

        def release(self) -> None:
            pass

    stats = benchmark_capture(Cam(), 0.35, clock=lambda: now["t"])
    assert stats["frames"] == 4
    assert stats["elapsed"] == pytest.approx(0.4)
    assert stats["fps"] == pytest.approx(10.0)
    text = format_benchmark(stats)
    assert "fps=10.0" in text
    assert "frames=4" in text


def test_present_draws_fps():
    frame = np.zeros((40, 80, 3), np.uint8)
    out = present(frame, fps=15.0, show_fps=True)
    assert not np.array_equal(out, frame)


def test_present_can_hide_fps_but_keeps_the_mode():
    frame = np.zeros((40, 80, 3), np.uint8)
    out = present(frame, fps=15.0, show_fps=False, mode="held")
    assert not np.array_equal(out, frame)


def test_live_and_held_indicators_differ():
    frame = np.zeros((80, 200, 3), np.uint8)
    live = present(frame, fps=0.0, show_fps=False, mode="live")
    held = present(frame, fps=0.0, show_fps=False, mode="held")
    assert not np.array_equal(live, held)


def test_average_frames_means_pixels():
    dark = np.zeros((2, 2, 3), np.uint8)
    bright = np.full((2, 2, 3), 100, np.uint8)
    assert int(average_frames([dark, bright])[0, 0, 0]) == 50


def test_space_freezes_the_frame_and_r_releases_it():
    def solid(value: int) -> np.ndarray:
        return np.full((80, 120, 3), value, np.uint8)

    frames = [solid(10), solid(20), solid(30), solid(40)]
    camera = FakeCamera(frames)
    display = FakeDisplay([32, -1, ord("r"), ord("q")])
    run(Config(), camera=camera, display=display)
    # Frame 0 is still live. Frame 1 is the held average of the first frame.
    assert int(display.shown[1][-1, -1, 0]) == 10
    # Release is applied after frame 2 is drawn, so frame 3 is live again.
    assert int(display.shown[3][-1, -1, 0]) == 40


def test_hold_averages_the_recent_buffer():
    session = Session()
    session.remember(np.zeros((2, 2, 3), np.uint8), hold_frames=2)
    session.remember(np.full((2, 2, 3), 100, np.uint8), hold_frames=2)
    session.hold()
    assert session.mode == "held"
    assert int(session.held[0, 0, 0]) == 50
    session.release()
    assert session.mode == "live"
    assert session.held is None


def test_hold_is_restored_on_the_next_run(tmp_path, monkeypatch):
    path = tmp_path / "held.png"
    monkeypatch.setattr("ghost_image.app.held_frame_path", lambda _config: path)
    held = np.full((80, 120, 3), 10, np.uint8)
    live = np.full((80, 120, 3), 90, np.uint8)
    run(Config(), camera=FakeCamera([held]), display=FakeDisplay([32, ord("q")]))
    assert path.is_file()

    display = FakeDisplay([ord("q")])
    run(Config(), camera=FakeCamera([live]), display=display)
    assert int(display.shown[0][-1, -1, 0]) == 10


def test_release_deletes_the_saved_frame(tmp_path, monkeypatch):
    path = tmp_path / "held.png"
    monkeypatch.setattr("ghost_image.app.held_frame_path", lambda _config: path)
    frame = np.full((80, 120, 3), 10, np.uint8)
    run(
        Config(),
        camera=FakeCamera([frame, frame, frame]),
        display=FakeDisplay([32, ord("r"), ord("q")]),
    )
    assert not path.exists()


def test_debug_view_shows_the_mask_once_a_frame_is_held():
    white = np.full((80, 120, 3), 200, np.uint8)
    camera = FakeCamera([white, white, white])
    display = FakeDisplay([ord("d"), 32, ord("q")])
    run(Config(), camera=camera, display=display)
    # Debug is on and the third frame is held, so the view is the (empty) mask.
    assert int(display.shown[2][-1, -1, 0]) == 0
    assert int(display.shown[0][-1, -1, 0]) == 200


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
