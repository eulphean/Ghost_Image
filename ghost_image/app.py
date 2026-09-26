"""Main loop.

Grabs a frame, draws the FPS readout, and shows it until the operator quits.
Later phases plug segmentation and compositing into ``present``.
"""

from __future__ import annotations

import resource
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ghost_image.camera import open_camera
from ghost_image.compositor import composite
from ghost_image.config import Config
from ghost_image.segmentation import (
    SegmentationError,
    Segmenter,
    create_segmenter,
    mask_to_bgr,
)
from ghost_image.store import delete_held_frame, held_frame_path, load_held_frame, save_held_frame
from ghost_image.ui import FpsCounter, OpenCVDisplay, draw_overlay, key_matches


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...

    def release(self) -> None: ...


class Display(Protocol):
    fullscreen: bool

    def show(self, frame: np.ndarray) -> int: ...

    def toggle_fullscreen(self) -> None: ...

    def close(self) -> None: ...


@dataclass
class Session:
    """LIVE shows the camera. HELD freezes the averaged reference frame."""

    mode: str = "live"
    held: np.ndarray | None = None
    debug: bool = False
    segment_error: bool = False
    alpha: float = 0.45
    recent: deque[np.ndarray] = field(default_factory=deque)

    def remember(self, frame: np.ndarray, hold_frames: int) -> None:
        if self.recent.maxlen != hold_frames:
            self.recent = deque(self.recent, maxlen=hold_frames)
        self.recent.append(frame.copy())

    def hold(self) -> None:
        if not self.recent:
            return
        self.held = average_frames(list(self.recent))
        self.mode = "held"

    def release(self) -> None:
        self.held = None
        self.mode = "live"

    def output_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == "held" and self.held is not None:
            return self.held
        return frame


def average_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Mean of BGR frames, used to denoise the held reference."""
    stacked = np.stack([frame.astype(np.float32) for frame in frames], axis=0)
    return np.clip(stacked.mean(axis=0), 0, 255).astype(np.uint8)


def present(frame: np.ndarray, *, fps: float, show_fps: bool, mode: str = "live") -> np.ndarray:
    """Build the image shown for this frame, with the LIVE/HELD indicator."""
    lines = [mode.upper()]
    if show_fps:
        lines.append(f"{fps:4.1f} fps")
    return draw_overlay(frame, lines)


def benchmark_capture(
    camera: FrameSource,
    seconds: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, float]:
    """Read frames for ``seconds`` and report FPS and process CPU time.

    Used by the Pi smoke test (``--benchmark-seconds``) before any segmentation
    is added, so the number is the camera's baseline.
    """
    start = clock()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    frames = 0
    while clock() - start < seconds:
        if camera.read() is not None:
            frames += 1
    elapsed = max(clock() - start, 1e-9)
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (usage_after.ru_utime - usage.ru_utime) + (usage_after.ru_stime - usage.ru_stime)
    return {
        "frames": float(frames),
        "elapsed": elapsed,
        "fps": frames / elapsed,
        "cpu_seconds": cpu,
    }


def format_benchmark(stats: dict[str, float]) -> str:
    return (
        f"frames={int(stats['frames'])} elapsed={stats['elapsed']:.2f}s "
        f"fps={stats['fps']:.1f} cpu={stats['cpu_seconds']:.2f}s"
    )


def run(
    config: Config,
    *,
    camera: FrameSource | None = None,
    display: Display | None = None,
    max_frames: int | None = None,
) -> int:
    """Show the live feed until quit, Ctrl-C, or ``max_frames`` frames."""
    owns_camera = camera is None
    if camera is None:
        camera = open_camera(config.camera)
    if display is None:
        display = OpenCVDisplay(config.display)

    fps = FpsCounter()
    session = Session()
    session.debug = config.display.debug
    session.alpha = config.ghost.alpha
    path = held_frame_path(config)
    segmenter = None
    if config.paths.restore_held:
        restored = load_held_frame(path)
        if restored is not None:
            session.held = restored
            session.mode = "held"
    shown = 0
    try:
        while max_frames is None or shown < max_frames:
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            session.remember(frame, config.camera.hold_frames)
            mask, segmenter = _person_mask(session, frame, config, segmenter)
            visual = session.output_frame(frame)
            if session.debug and mask is not None:
                visual = mask_to_bgr(mask)
            elif mask is not None and session.held is not None:
                visual = composite(session.held, frame, mask, config.ghost, session.alpha)
            image = present(
                visual,
                fps=fps.tick(),
                show_fps=config.display.show_fps,
                mode=session.mode,
            )
            key = display.show(image)
            shown += 1
            if key_matches(key, config.keys.quit):
                return 0
            if key_matches(key, config.keys.fullscreen):
                display.toggle_fullscreen()
            elif key_matches(key, config.keys.hold):
                session.hold()
                if session.held is not None:
                    save_held_frame(path, session.held)
            elif key_matches(key, config.keys.release):
                session.release()
                delete_held_frame(path)
            elif key_matches(key, config.keys.debug):
                session.debug = not session.debug
            elif key_matches(key, config.keys.opacity_down):
                session.alpha = max(0.0, session.alpha - config.ghost.alpha_step)
            elif key_matches(key, config.keys.opacity_up):
                session.alpha = min(1.0, session.alpha + config.ghost.alpha_step)
    except KeyboardInterrupt:
        return 0
    finally:
        if segmenter is not None:
            segmenter.close()
        if owns_camera:
            camera.release()
        display.close()
    return 0


def _person_mask(
    session: Session,
    frame: np.ndarray,
    config: Config,
    segmenter: Segmenter | None,
) -> tuple[np.ndarray | None, Segmenter | None]:
    """Segment the current frame once a reference is held.

    The segmenter is created on the first held frame. A missing model leaves
    the held frame on screen and reports the error once.
    """
    if session.mode != "held":
        return None, segmenter
    if segmenter is None and not session.segment_error:
        try:
            segmenter = create_segmenter(config)
        except SegmentationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            session.segment_error = True
            return None, None
    if segmenter is None:
        return None, None
    try:
        return segmenter.mask(frame, session.held), segmenter
    except SegmentationError as exc:
        if not session.segment_error:
            print(f"error: {exc}", file=sys.stderr)
            session.segment_error = True
        return None, segmenter
