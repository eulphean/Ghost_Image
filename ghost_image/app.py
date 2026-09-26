"""Main loop.

Grabs a frame, draws the FPS readout, and shows it until the operator quits.
Later phases plug segmentation and compositing into ``present``.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

from ghost_image.camera import open_camera
from ghost_image.config import Config
from ghost_image.ui import FpsCounter, OpenCVDisplay, draw_overlay, key_matches


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...

    def release(self) -> None: ...


class Display(Protocol):
    fullscreen: bool

    def show(self, frame: np.ndarray) -> int: ...

    def toggle_fullscreen(self) -> None: ...

    def close(self) -> None: ...


def present(frame: np.ndarray, *, fps: float, show_fps: bool) -> np.ndarray:
    """Build the image shown for this frame. No ghost processing yet."""
    if not show_fps:
        return frame
    return draw_overlay(frame, [f"{fps:4.1f} fps"])


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
    shown = 0
    try:
        while max_frames is None or shown < max_frames:
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            image = present(frame, fps=fps.tick(), show_fps=config.display.show_fps)
            key = display.show(image)
            shown += 1
            if key_matches(key, config.keys.quit):
                return 0
            if key_matches(key, config.keys.fullscreen):
                display.toggle_fullscreen()
    except KeyboardInterrupt:
        return 0
    finally:
        if owns_camera:
            camera.release()
        display.close()
    return 0
