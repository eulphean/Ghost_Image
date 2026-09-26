"""Blend a segmented person over the held frame as a translucent ghost."""

from __future__ import annotations

import cv2
import numpy as np

from ghost_image.config import GhostConfig


def stylise(frame: np.ndarray, ghost: GhostConfig) -> np.ndarray:
    """Desaturate, tint, lift, and optionally blur the person pixels."""
    image = frame.astype(np.float32)
    if ghost.desaturate > 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
        image = image * (1.0 - ghost.desaturate) + gray_bgr * ghost.desaturate
    if ghost.tint_strength > 0:
        tint = np.array(ghost.tint, dtype=np.float32)
        image = image * (1.0 - ghost.tint_strength) + tint * ghost.tint_strength
    image *= ghost.brightness
    if ghost.blur_px:
        image = cv2.GaussianBlur(image, (ghost.blur_px, ghost.blur_px), 0)
    return np.clip(image, 0, 255)


def composite(
    held: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
    ghost: GhostConfig,
    alpha: float,
) -> np.ndarray:
    """``held * (1 - α·mask) + ghost(frame) * (α·mask)``.

    ``alpha`` is the live opacity (0 invisible, 1 solid). Where the mask is 0
    the held frame is unchanged, so the room stays visible through the person.
    """
    if held.shape[:2] != frame.shape[:2]:
        held = cv2.resize(held, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    coverage = np.clip(mask.astype(np.float32), 0.0, 1.0) * float(np.clip(alpha, 0.0, 1.0))
    coverage = coverage[..., None]
    ghost_px = stylise(frame, ghost)
    blended = held.astype(np.float32) * (1.0 - coverage) + ghost_px * coverage
    return np.clip(blended, 0, 255).astype(np.uint8)
