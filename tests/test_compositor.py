"""Ghost blend math on synthetic frames. No camera."""

from __future__ import annotations

import numpy as np
import pytest

from ghost_image.compositor import EchoBuffer, add_glow, composite, echo_tiles, glow_layer, stylise
from ghost_image.config import GhostConfig, GlowConfig


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


def test_glow_traces_the_contour_and_not_the_interior():
    mask = np.zeros((40, 40), np.float32)
    mask[10:30, 10:30] = 1.0
    glow = GlowConfig(
        enabled=True,
        color=[255, 255, 255],
        thickness=1,
        blur_px=0,
        intensity=1.0,
        pulse=False,
    )
    layer = glow_layer(mask, glow, threshold=0.5, now=0.0)
    assert float(layer[20, 20].sum()) == 0.0
    assert float(layer[10, 20, 0]) > 200
    assert float(layer[0, 0].sum()) == 0.0
    image = add_glow(np.zeros((40, 40, 3), np.uint8), layer)
    assert int(image[10, 20, 0]) > 200
    assert int(image[20, 20, 0]) == 0


def test_glow_pulse_changes_intensity():
    mask = np.ones((8, 8), np.float32)
    glow = GlowConfig(blur_px=0, thickness=1, pulse=True, pulse_period_s=4.0, intensity=1.0)
    quiet = glow_layer(mask, glow, 0.5, now=0.0)
    loud = glow_layer(mask, glow, 0.5, now=1.0)
    assert float(loud.max()) > float(quiet.max())


def test_echo_tiles_use_an_older_frame_and_a_different_tint():
    held = np.zeros((4, 4, 3), np.uint8)
    live = np.full((4, 4, 3), 100, np.uint8)
    past = np.full((4, 4, 3), 40, np.uint8)
    mask = np.ones((4, 4), np.float32)
    echo = EchoBuffer()
    echo.add(0.0, past, mask, keep_s=1.0)
    echo.add(0.3, live, mask, keep_s=1.0)
    glow = GlowConfig(enabled=False)
    plain = echo_tiles(
        echo,
        held,
        GhostConfig(desaturate=0.0, tint_strength=0.0, brightness=1.0, blur_px=0),
        glow,
        alpha=1.0,
        threshold=0.5,
        now=0.3,
        count=2,
        delay_s=0.3,
        fade_step=0.0,
        tints=[[10, 10, 10], [200, 0, 0]],
    )
    assert np.all(plain[0] == 100)
    assert np.all(plain[1] == 40)
    tinted = echo_tiles(
        echo,
        held,
        GhostConfig(desaturate=0.0, tint_strength=1.0, brightness=1.0, blur_px=0),
        glow,
        alpha=1.0,
        threshold=0.5,
        now=0.3,
        count=2,
        delay_s=0.0,
        fade_step=0.0,
        tints=[[10, 10, 10], [200, 0, 0]],
    )
    assert np.all(tinted[0] == 10)
    assert np.all(tinted[1][:, :, 0] == 200)


def test_composite_resizes_a_smaller_held_frame():
    held = np.full((2, 2, 3), 10, np.uint8)
    frame = np.full((4, 6, 3), 10, np.uint8)
    mask = np.zeros((4, 6), np.float32)
    out = composite(held, frame, mask, plain_ghost(), alpha=1.0)
    assert out.shape == frame.shape
