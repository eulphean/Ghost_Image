"""Person masks.

``Segmenter.mask`` returns a float32 array in ``[0, 1]`` at the frame's
resolution, where 1 is "person" (body, arms, hands, fingers) and 0 is
everything else. ``create_segmenter`` picks the implementation named in
``processing.segmenter`` so the engines can be compared without code changes.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from ghost_image.config import PROJECT_ROOT, Config, ProcessingConfig


class SegmentationError(RuntimeError):
    """Raised when a segmenter cannot be built or cannot produce a mask."""


class Segmenter:
    """Base class. Subclasses fill in ``mask``."""

    name: str = "base"

    def mask(self, frame: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
        """Return a float32 person mask the same height and width as ``frame``."""
        raise NotImplementedError

    def close(self) -> None:
        """Release models or threads. The default does nothing."""
        return None


def create_segmenter(config: Config) -> Segmenter:
    """Build the segmenter selected by ``config.processing.segmenter``."""
    kind = config.processing.segmenter
    if kind == "diff":
        inner: Segmenter = DiffSegmenter(
            config.processing.diff_threshold,
            adapt=config.processing.background_adapt,
        )
    elif kind == "mediapipe":
        inner = _mediapipe(config)
    elif kind == "hybrid":
        inner = HybridSegmenter(
            _mediapipe(config),
            config.processing.hybrid_mode,
            config.processing.diff_threshold,
            adapt=config.processing.background_adapt,
        )
    else:
        raise SegmentationError(f"unknown segmenter {kind!r}")
    return ProcessedSegmenter(inner, config.processing)


def _mediapipe(config: Config) -> MediaPipeSegmenter:
    path = PROJECT_ROOT / config.paths.models_dir / config.paths.model_file
    return MediaPipeSegmenter(path, config.processing.width)


def ensure_mask(mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Resize, cast, and clip a mask so it matches the contract."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)


def empty_mask(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return np.zeros((height, width), np.float32)


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Scale ``frame`` so its width is ``width``, keeping the aspect ratio."""
    height, current = frame.shape[:2]
    if current == width:
        return frame
    new_height = max(1, int(round(height * (width / current))))
    interpolation = cv2.INTER_AREA if width < current else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, new_height), interpolation=interpolation)


def person_from_confidences(masks: list[np.ndarray]) -> np.ndarray:
    """Turn model outputs into one person mask.

    The selfie model returns a single confidence map (high = person). The
    multiclass model returns background first, then hair, skin, clothes and
    others, so the person is everything that is not background.
    """
    if not masks:
        raise SegmentationError("segmenter returned no confidence mask")
    if len(masks) == 1:
        person = np.squeeze(masks[0])
    else:
        person = 1.0 - np.squeeze(masks[0])
    return np.clip(np.array(person, dtype=np.float32, copy=True), 0.0, 1.0)


def postprocess(
    mask: np.ndarray,
    previous: np.ndarray | None,
    processing: ProcessingConfig,
) -> np.ndarray:
    """Binarise, clean, feather, and temporally smooth a person mask.

    Morphology and blob removal run on a hard mask. Feathering and the
    exponential average put the soft edge back, which is what the compositor
    blends with.
    """
    hard = (mask >= processing.mask_threshold).astype(np.uint8) * 255
    if processing.morph_px:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (processing.morph_px, processing.morph_px)
        )
        hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, kernel)
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel)
    if processing.min_blob_area > 0:
        hard = _drop_small_blobs(hard, processing.min_blob_area)
    soft = hard.astype(np.float32) / 255.0
    if processing.feather_px:
        soft = cv2.GaussianBlur(soft, (processing.feather_px, processing.feather_px), 0)
    if previous is not None and processing.mask_smoothing > 0:
        weight = processing.mask_smoothing
        soft = previous * weight + soft * (1.0 - weight)
    return np.clip(soft, 0.0, 1.0).astype(np.float32, copy=False)


def _drop_small_blobs(hard: np.ndarray, min_area: int) -> np.ndarray:
    binary = (hard > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return hard
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0  # background label
    keep = (areas >= min_area).astype(np.uint8)
    return hard * keep[labels]


class ProcessedSegmenter(Segmenter):
    """Applies :func:`postprocess` to another segmenter's mask."""

    def __init__(self, inner: Segmenter, processing: ProcessingConfig) -> None:
        self._inner = inner
        self._processing = processing
        self._previous: np.ndarray | None = None
        self.name = inner.name

    def mask(self, frame: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
        raw = self._inner.mask(frame, held)
        cleaned = postprocess(raw, self._previous, self._processing)
        self._previous = cleaned
        return cleaned

    def close(self) -> None:
        self._inner.close()


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Gray BGR image of a mask, for the debug view."""
    gray = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class _Engine:
    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - protocol
        return None


class MediaPipeEngine:
    """MediaPipe image segmenter in VIDEO mode. Not safe to call from two threads."""

    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        options = ImageSegmenterOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._mp = mp
        self._segmenter = ImageSegmenter.create_from_options(options)
        self._timestamp_ms = 0

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self._segmenter.segment_for_video(image, self._timestamp_ms)
        views = [mask.numpy_view() for mask in result.confidence_masks or []]
        return person_from_confidences(views)

    def close(self) -> None:
        self._segmenter.close()


class MediaPipeSegmenter(Segmenter):
    """Runs MediaPipe on a worker thread and upscales the latest mask.

    ``engine`` is injectable so tests do not load the model. Capture never
    waits on inference: ``mask`` returns the previous result, or zeros until
    the first one is ready.
    """

    name = "mediapipe"

    def __init__(self, model_path: Path, width: int, *, engine: _Engine | None = None) -> None:
        self.width = width
        if engine is None:
            if not Path(model_path).is_file():
                raise SegmentationError(
                    f"model not found: {model_path}. Run scripts/download_models.sh"
                )
            engine = MediaPipeEngine(model_path)
        self._engine = engine
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending: np.ndarray | None = None
        self._latest: np.ndarray | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="segmenter", daemon=True)
        self._thread.start()

    def mask(self, frame: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
        small = resize_to_width(frame, self.width)
        with self._lock:
            if self._error is not None:
                raise SegmentationError(str(self._error)) from self._error
            self._pending = small
            self._wake.set()
            latest = self._latest
        if latest is None:
            return empty_mask(frame)
        return ensure_mask(latest, frame)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2)
        self._engine.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.05)
            with self._lock:
                self._wake.clear()
                frame = self._pending
                self._pending = None
            if frame is None:
                continue
            try:
                mask = self._engine.segment(frame)
            except Exception as exc:  # noqa: BLE001 - surfaced on the next mask() call
                with self._lock:
                    self._error = exc
                return
            with self._lock:
                self._latest = mask


def difference_mask(frame: np.ndarray, held: np.ndarray, threshold: int) -> np.ndarray:
    """Person mask from how much ``frame`` differs from ``held``.

    Compared in Lab after scaling the live lightness to the reference, so a
    camera-wide brightness shift does not light up the whole frame. 1 where
    the scene changed.
    """
    if held.shape[:2] != frame.shape[:2]:
        held = cv2.resize(held, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    live = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference = cv2.cvtColor(held, cv2.COLOR_BGR2LAB).astype(np.float32)
    live[..., 0] *= (float(reference[..., 0].mean()) + 1.0) / (float(live[..., 0].mean()) + 1.0)
    delta = np.max(np.abs(live - reference), axis=2)
    return (delta >= float(threshold)).astype(np.float32)


def combine_masks(model: np.ndarray, diff: np.ndarray, mode: str) -> np.ndarray:
    """Merge a model mask and a frame-difference mask.

    ``refine`` keeps the model mask and adds difference pixels that touch it,
    which recovers thin fingers without letting a lighting change fill the frame.
    """
    if mode == "union":
        return np.maximum(model, diff)
    if mode == "intersect":
        return np.minimum(model, diff)
    dilated = cv2.dilate(model, np.ones((7, 7), np.uint8))
    nearby = np.where(dilated > 0.2, diff, 0.0)
    return np.maximum(model, nearby).astype(np.float32, copy=False)


class DiffSegmenter(Segmenter):
    """Classical difference against the held frame. No model required.

    ``adapt`` slowly updates an internal background where no person is seen, so
    daylight drift does not fill the mask. The image shown to the audience
    stays the original held frame; only this comparison copy moves.
    """

    name = "diff"

    def __init__(self, threshold: int, adapt: float = 0.0) -> None:
        self.threshold = threshold
        self.adapt = adapt
        self._background: np.ndarray | None = None

    def mask(self, frame: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
        if held is None:
            return empty_mask(frame)
        if self.adapt <= 0:
            return difference_mask(frame, held, self.threshold)
        reference = held
        if reference.shape[:2] != frame.shape[:2]:
            reference = cv2.resize(
                reference, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        if self._background is None or self._background.shape != frame.shape:
            self._background = reference.astype(np.float32)
        mask = difference_mask(
            frame, np.clip(self._background, 0, 255).astype(np.uint8), self.threshold
        )
        still = mask < 0.5
        live = frame.astype(np.float32)
        self._background[still] = (
            self._background[still] * (1.0 - self.adapt) + live[still] * self.adapt
        )
        return mask


class HybridSegmenter(Segmenter):
    """MediaPipe mask refined with frame difference."""

    name = "hybrid"

    def __init__(self, model: Segmenter, mode: str, threshold: int, adapt: float = 0.0) -> None:
        self._model = model
        self._diff = DiffSegmenter(threshold, adapt=adapt)
        self.mode = mode

    def mask(self, frame: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
        model_mask = self._model.mask(frame, held)
        if held is None:
            return model_mask
        return combine_masks(model_mask, self._diff.mask(frame, held), self.mode)

    def close(self) -> None:
        self._model.close()
