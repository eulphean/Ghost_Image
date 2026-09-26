"""Blend a segmented person over the held frame as a translucent ghost."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ghost_image.config import GhostConfig, GlowConfig


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


def glow_layer(mask: np.ndarray, glow: GlowConfig, threshold: float, now: float) -> np.ndarray:
    """A soft coloured halo around the person contour. Additive, float BGR."""
    height, width = mask.shape[:2]
    if not glow.enabled:
        return np.zeros((height, width, 3), np.float32)
    hard = (mask >= threshold).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge = np.zeros((height, width), np.uint8)
    if contours:
        cv2.drawContours(edge, contours, -1, 255, thickness=glow.thickness)
    if glow.blur_px:
        edge = cv2.GaussianBlur(edge, (glow.blur_px, glow.blur_px), 0)
    intensity = glow.intensity
    if glow.pulse:
        wave = math.sin(2.0 * math.pi * now / glow.pulse_period_s)
        intensity *= 0.65 + 0.35 * wave
    color = np.array(glow.color, dtype=np.float32) * intensity
    return (edge.astype(np.float32) / 255.0)[..., None] * color


def add_glow(image: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Add an HDR-style halo onto an 8-bit image."""
    return np.clip(image.astype(np.float32) + layer, 0, 255).astype(np.uint8)
