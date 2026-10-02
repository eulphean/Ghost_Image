"""Main loop.

Grabs a frame, draws the FPS readout, and shows it until the operator quits.
Later phases plug segmentation and compositing into ``present``.
"""

from __future__ import annotations

import time

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from ghost_image.camera import open_camera
from ghost_image.compositor import add_glow, composite, glow_layer
from ghost_image.config import Config
from ghost_image.gpio import GpioControls
from ghost_image.log import StageTimer, report_error
from ghost_image.segmentation import (
    DiffSegmenter,
    ProcessedSegmenter,
    SegmentationError,
    Segmenter,
    create_segmenter,
    mask_to_bgr,
)
from ghost_image.store import (
    delete_held_frame,
    held_frame_path,
    load_held_frame,
    save_held_frame,
    save_snapshot,
    snapshot_path,
)
from ghost_image.ui import FpsCounter, OpenCVDisplay, camera_lost_frame, draw_overlay, key_matches


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
    view: str = "composite"
    segmenter_name: str = "none"
    segment_error: bool = False
    loop_error: bool = False
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


VIEWS = ("composite", "live", "mask", "held")


def cycle_view(view: str) -> str:
    """Step D through live, mask, held frame, and the final composite."""
    try:
        index = VIEWS.index(view)
    except ValueError:
        index = 0
    return VIEWS[(index + 1) % len(VIEWS)]


def present(
    frame: np.ndarray,
    *,
    fps: float,
    show_fps: bool,
    show_status: bool = True,
    mode: str = "live",
    lines: list[str] | None = None,
) -> np.ndarray:
    """Build the image shown for this frame, with one status row per value."""
    if not show_status:
        return frame.copy()
    rows = [mode.upper()]
    if show_fps:
        rows.append(f"{fps:4.1f} fps")
    if lines:
        rows.extend(lines)
    return draw_overlay(frame, rows)


def status_lines(session: Session, config: Config, timings: dict[str, float]) -> list[str]:
    """One label per row: opacity, glow, segmenter, view, and optional timings."""
    segmenter = "off" if session.segmenter_name == "none" else session.segmenter_name
    rows = [
        f"opacity {session.alpha:.2f}",
        f"glow {config.glow.intensity:.1f}",
        f"segmenter {segmenter}",
        f"view {session.view}",
    ]
    if session.view != "composite" or config.display.debug:
        rows.append(f"segment {timings.get('segment', 0):.0f} ms")
        rows.append(f"composite {timings.get('composite', 0):.0f} ms")
    return rows


def _view_frame(
    session: Session,
    frame: np.ndarray,
    mask: np.ndarray | None,
    config: Config,
) -> np.ndarray:
    """Pick the image for the current debug view."""
    blended = None
    if mask is not None and session.held is not None:
        blended = composite(session.held, frame, mask, config.ghost, session.alpha)
        blended = add_glow(
            blended,
            glow_layer(mask, config.glow, config.processing.mask_threshold, time.perf_counter()),
        )
    if session.view == "live":
        return frame
    if session.view == "mask":
        if mask is None:
            return np.zeros_like(frame)
        return mask_to_bgr(mask)
    if session.view == "held":
        if session.held is None:
            return np.zeros_like(frame)
        held = session.held
        if held.shape[:2] != frame.shape[:2]:
            held = cv2.resize(held, (frame.shape[1], frame.shape[0]))
        return held
    if blended is not None:
        return blended
    return session.output_frame(frame)


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
    cpu_before = _process_cpu_seconds()
    frames = 0
    while clock() - start < seconds:
        if camera.read() is not None:
            frames += 1
    elapsed = max(clock() - start, 1e-9)
    cpu = _process_cpu_seconds() - cpu_before
    return {
        "frames": float(frames),
        "elapsed": elapsed,
        "fps": frames / elapsed,
        "cpu_seconds": cpu,
    }


def _process_cpu_seconds() -> float:
    """CPU seconds used by this process.

    ``resource`` is Unix-only. Windows reports the same quantity through
    ``time.process_time``.
    """
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime
    return time.process_time()


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
    buttons: GpioControls | None = None,
) -> int:
    """Show the live feed until quit, Ctrl-C, or ``max_frames`` frames."""
    owns_camera = camera is None
    if camera is None:
        camera = open_camera(config.camera)
    if display is None:
        display = OpenCVDisplay(config.display)

    fps = FpsCounter()
    timer = StageTimer()
    session = Session()
    session.view = "mask" if config.display.debug else "composite"
    session.alpha = config.ghost.alpha
    path = held_frame_path(config)
    segmenter = None
    controls = buttons if buttons is not None else GpioControls(config.gpio)
    if config.paths.restore_held:
        restored = load_held_frame(path)
        if restored is not None:
            session.held = restored
            session.mode = "held"
    shown = 0
    try:
        while max_frames is None or shown < max_frames:
            try:
                frame = camera.read()
            except Exception as exc:  # noqa: BLE001 - one bad read must not kill the installation
                _note_loop_error(session, exc)
                time.sleep(0.05)
                continue
            if frame is None:
                lost = camera_lost_frame(config.camera.width, config.camera.height)
                key = display.show(lost)
                if key_matches(key, config.keys.quit):
                    return 0
                time.sleep(0.05)
                continue
            try:
                session.remember(frame, config.camera.hold_frames)
                with timer.measure("segment"):
                    mask, segmenter = _person_mask(session, frame, config, segmenter)
                with timer.measure("composite"):
                    visual = _view_frame(session, frame, mask, config)
                image = present(
                    visual,
                    fps=fps.tick(),
                    show_fps=config.display.show_fps,
                    show_status=config.display.show_status,
                    mode=session.mode,
                    lines=status_lines(session, config, timer.ms),
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
                    session.view = cycle_view(session.view)
                elif key_matches(key, config.keys.hide):
                    config.display.show_status = not config.display.show_status
                elif key_matches(key, config.keys.snapshot):
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    save_snapshot(snapshot_path(config, stamp), visual)
                elif key_matches(key, config.keys.opacity_down):
                    session.alpha = max(0.0, session.alpha - config.ghost.alpha_step)
                elif key_matches(key, config.keys.opacity_up):
                    session.alpha = min(1.0, session.alpha + config.ghost.alpha_step)
                action = controls.poll()
                if action == "hold":
                    session.hold()
                    if session.held is not None:
                        save_held_frame(path, session.held)
                elif action == "release":
                    session.release()
                    delete_held_frame(path)
            except Exception as exc:  # noqa: BLE001 - keep the installation up
                _note_loop_error(session, exc)
                time.sleep(0.05)
    except KeyboardInterrupt:
        return 0
    finally:
        controls.close()
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

    If the configured segmenter cannot be built or fails on a frame, switch to
    frame differencing and report that once.
    """
    if session.mode != "held":
        return None, segmenter
    if segmenter is None:
        try:
            segmenter = create_segmenter(config)
            session.segmenter_name = getattr(segmenter, "name", "segmenter")
        except SegmentationError as exc:
            segmenter = _fallback_segmenter(session, config, str(exc))
    try:
        return segmenter.mask(frame, session.held), segmenter
    except SegmentationError as exc:
        if session.segmenter_name == "diff":
            _note_loop_error(session, exc)
            return None, segmenter
        segmenter = _fallback_segmenter(session, config, str(exc), previous=segmenter)
        return segmenter.mask(frame, session.held), segmenter


def _fallback_segmenter(
    session: Session,
    config: Config,
    reason: str,
    previous: Segmenter | None = None,
) -> Segmenter:
    if not session.segment_error:
        report_error(f"{reason}; falling back to the diff segmenter")
        session.segment_error = True
    if previous is not None:
        previous.close()
    segmenter = ProcessedSegmenter(
        DiffSegmenter(
            config.processing.diff_threshold,
            adapt=config.processing.background_adapt,
        ),
        config.processing,
    )
    session.segmenter_name = segmenter.name
    return segmenter


def _note_loop_error(session: Session, exc: BaseException) -> None:
    if session.loop_error:
        return
    report_error(str(exc))
    session.loop_error = True
