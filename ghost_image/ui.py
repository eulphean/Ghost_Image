"""Window, on-screen text, and keyboard decoding.

The OpenCV window lives here so the rest of the app can be tested with a fake
display. Key names in ``config.yaml`` are matched against ``cv2.waitKey`` codes.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np

from ghost_image.config import DisplayConfig

_NAMED_KEYS = {
    "space": 32,
    "esc": 27,
    "enter": 13,
    "return": 13,
}


def normalize_key(key: int) -> int | None:
    """Collapse a ``waitKey`` result to one byte, or ``None`` when nothing was pressed."""
    if key < 0:
        return None
    if key == 27:
        return 27
    return key & 0xFF


def key_code(name: str) -> int:
    cleaned = name.lower()
    if cleaned in _NAMED_KEYS:
        return _NAMED_KEYS[cleaned]
    if len(cleaned) == 1:
        return ord(cleaned)
    raise ValueError(f"unsupported key name: {name!r}")


def key_matches(key: int, names: list[str]) -> bool:
    code = normalize_key(key)
    if code is None:
        return False
    return any(key_code(name) == code for name in names)


def camera_lost_frame(
    width: int,
    height: int,
    message: str = "camera lost — reconnecting",
) -> np.ndarray:
    """A black frame with a single line, shown while discovery runs again."""
    image = np.zeros((height, width, 3), np.uint8)
    cv2.putText(
        image,
        message,
        (24, max(height // 2, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw ``lines`` in the top-left of a copy of ``frame``."""
    out = frame.copy()
    y = 28
    for line in lines:
        origin = (16, y)
        cv2.putText(out, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            out, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 28
    return out


class FpsCounter:
    """Frames per second, updated about twice a second."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._t = clock()
        self._count = 0
        self.fps = 0.0

    def tick(self) -> float:
        self._count += 1
        now = self._clock()
        elapsed = now - self._t
        if elapsed >= 0.5:
            self.fps = self._count / elapsed
            self._count = 0
            self._t = now
        return self.fps


class OpenCVDisplay:
    """A named OpenCV window. ``show`` returns the key pressed this frame."""

    def __init__(self, config: DisplayConfig) -> None:
        self.name = config.window_name
        self.fullscreen = config.fullscreen
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self._apply_fullscreen()

    def show(self, frame: np.ndarray) -> int:
        cv2.imshow(self.name, frame)
        return int(cv2.waitKey(1))

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self._apply_fullscreen()

    def close(self) -> None:
        cv2.destroyWindow(self.name)

    def _apply_fullscreen(self) -> None:
        flag = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, float(flag))
