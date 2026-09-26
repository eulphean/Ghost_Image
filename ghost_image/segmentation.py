"""Person masks.

``Segmenter.mask`` returns a float32 array in ``[0, 1]`` at the frame's
resolution, where 1 is "person" (body, arms, hands, fingers) and 0 is
everything else. ``create_segmenter`` picks the implementation named in
``processing.segmenter`` so the engines can be compared without code changes.
"""

from __future__ import annotations

import cv2
import numpy as np

from ghost_image.config import SEGMENTERS, Config


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
    """Build the segmenter selected by ``config.processing.segmenter``.

    Concrete engines are registered by later steps. Asking for one that has
    not been added yet is an error, not a silent empty mask.
    """
    kind = config.processing.segmenter
    if kind not in SEGMENTERS:
        raise SegmentationError(f"unknown segmenter {kind!r}")
    raise SegmentationError(f"segmenter {kind!r} is not available yet")


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
