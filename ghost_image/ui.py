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
    """Draw each status line once, in white, on one black panel."""
    out = frame.copy()
    if not lines:
        return out
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    margin = 8
    pad = 10
    gap = 6
    measured = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    text_width = max(size[0] for size, _baseline in measured)
    block = sum(size[1] + baseline for size, baseline in measured) + gap * (len(lines) - 1)
    x0, y0 = margin, margin
    x1 = min(out.shape[1], x0 + text_width + pad * 2)
    y1 = min(out.shape[0], y0 + block + pad * 2)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), cv2.FILLED)
    y = y0 + pad
    for line, (size, baseline) in zip(lines, measured, strict=True):
        y += size[1]
        cv2.putText(out, line, (x0 + pad, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += baseline + gap
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
