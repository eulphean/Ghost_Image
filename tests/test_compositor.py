"""Ghost blend math on synthetic frames. No camera."""

from __future__ import annotations

import numpy as np
import pytest

from ghost_image.compositor import composite, stylise
from ghost_image.config import GhostConfig


def plain_ghost() -> GhostConfig:
    return GhostConfig(desaturate=0.0, tint_strength=0.0, brightness=1.0, blur_px=0)


def test_zero_alpha_leaves_the_held_frame():
    held = np.full((4, 4, 3), 20, np.uint8)
    frame = np.full((4, 4, 3), 200, np.uint8)
    mask = np.ones((4, 4), np.float32)
    out = composite(held, frame, mask, plain_ghost(), alpha=0.0)
    assert np.array_equal(out, held)


def test_solid_mask_replaces_with_the_person():
    held = np.zeros((4, 4, 3), np.uint8)
    frame = np.full((4, 4, 3), 80, np.uint8)
    mask = np.ones((4, 4), np.float32)
    out = composite(held, frame, mask, plain_ghost(), alpha=1.0)
    assert np.array_equal(out, frame)


def test_half_alpha_is_the_mean_where_the_mask_is_set():
    held = np.zeros((4, 4, 3), np.uint8)
    frame = np.full((4, 4, 3), 100, np.uint8)
    mask = np.zeros((4, 4), np.float32)
    mask[:, 2:] = 1.0
    out = composite(held, frame, mask, plain_ghost(), alpha=0.5)
    assert int(out[0, 0, 0]) == 0
    assert int(out[0, 3, 0]) == 50


def test_stylise_desaturates_a_pure_channel():
    frame = np.zeros((2, 2, 3), np.uint8)
    frame[:, :] = (0, 0, 200)  # red in BGR
    ghost = GhostConfig(desaturate=1.0, tint_strength=0.0, brightness=1.0, blur_px=0)
    out = stylise(frame, ghost)
    # Grey of pure red is equal channels, below the original red.
    assert out[0, 0, 0] == pytest.approx(out[0, 0, 1])
    assert out[0, 0, 2] < 200


def test_composite_resizes_a_smaller_held_frame():
    held = np.full((2, 2, 3), 10, np.uint8)
    frame = np.full((4, 6, 3), 10, np.uint8)
    mask = np.zeros((4, 6), np.float32)
    out = composite(held, frame, mask, plain_ghost(), alpha=1.0)
    assert out.shape == frame.shape
