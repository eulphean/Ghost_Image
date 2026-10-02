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


def tile_vertical(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Stack copies of ``frame`` at its own size until ``height`` is covered.

    Nothing is scaled. A wide frame is centre-cropped to ``width``. A narrow
    frame is centred on black. The last tile is cut off at the bottom of the
    screen when it does not land on a tile boundary.
    """
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0 or width <= 0 or height <= 0:
        return frame
    count = max(1, (height + src_h - 1) // src_h)
    stacked = np.tile(frame, (count, 1, 1))
    view_h = min(height, stacked.shape[0])
    view_w = min(width, stacked.shape[1])
    x0 = max(0, (stacked.shape[1] - width) // 2)
    cropped = stacked[0:view_h, x0 : x0 + view_w]
    if cropped.shape[0] == height and cropped.shape[1] == width:
        return cropped
    canvas = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    x_off = (width - cropped.shape[1]) // 2
    canvas[0 : cropped.shape[0], x_off : x_off + cropped.shape[1]] = cropped
    return canvas


def frame_for_window(frame: np.ndarray, width: int, height: int, *, fullscreen: bool) -> np.ndarray:
    """Picture to put in the window.

    A landscape fullscreen window is scaled to cover it. A portrait window
    (width < height) is filled by tiling the camera frame at its original
    resolution, with no scaling.
    """
    if not fullscreen:
        return frame
    if width < height:
        return tile_vertical(frame, width, height)
    return fit_to_screen(frame, width, height)


def fit_to_screen(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale ``frame`` so it covers ``width`` x ``height``, cropping the overflow.

    A 16:9 camera frame becomes exactly 1920×1080 on a 1080p screen. A different
    aspect is cropped at the centre rather than leaving a grey border.
    """
    src_h, src_w = frame.shape[:2]
    if src_w == width and src_h == height:
        return frame
    if src_w <= 0 or src_h <= 0 or width <= 0 or height <= 0:
        return frame
    scale = max(width / src_w, height / src_h)
    resized_w = max(width, int(round(src_w * scale)))
    resized_h = max(height, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (resized_w - width) // 2)
    y0 = max(0, (resized_h - height) // 2)
    cropped = resized[y0 : y0 + height, x0 : x0 + width]
    if cropped.shape[1] != width or cropped.shape[0] != height:
        cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    return cropped


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
        self.output_width = config.width
        self.output_height = config.height
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        self._apply_fullscreen()

    def show(self, frame: np.ndarray) -> int:
        image = frame_for_window(
            frame,
            self.output_width,
            self.output_height,
            fullscreen=self.fullscreen,
        )
        cv2.imshow(self.name, image)
        return int(cv2.waitKey(1))

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self._apply_fullscreen()

    def close(self) -> None:
        cv2.destroyWindow(self.name)

    def _apply_fullscreen(self) -> None:
        # The frame passed to imshow is already this size, so the window does
        # not stretch it. Portrait windows are tiled; landscape ones are scaled.
        cv2.setWindowProperty(self.name, cv2.WND_PROP_ASPECT_RATIO, float(cv2.WINDOW_FREERATIO))
        if self.fullscreen:
            cv2.resizeWindow(self.name, self.output_width, self.output_height)
            cv2.moveWindow(self.name, 0, 0)
            flag = cv2.WINDOW_FULLSCREEN
        else:
            flag = cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, float(flag))
