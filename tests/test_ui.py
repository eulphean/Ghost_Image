"""UI helpers. No window is opened."""

from __future__ import annotations

import numpy as np

from ghost_image.ui import (
    FpsCounter,
    camera_lost_frame,
    draw_overlay,
    fit_to_screen,
    frame_for_window,
    key_matches,
    normalize_key,
    tile_vertical,
)


def test_normalize_key_ignores_no_key():
    assert normalize_key(-1) is None
    assert normalize_key(27) == 27
    assert normalize_key(ord("q")) == ord("q")


def test_key_matches_quit_and_space():
    assert key_matches(ord("q"), ["q", "esc"])
    assert key_matches(27, ["q", "esc"])
    assert key_matches(32, ["space"])
    assert not key_matches(-1, ["q"])
    assert not key_matches(ord("f"), ["q", "esc"])


def test_camera_lost_frame_is_a_labelled_black_image():
    image = camera_lost_frame(80, 40)
    assert image.shape == (40, 80, 3)
    assert int(image.sum()) > 0
    assert int(image[0, 0, 0]) == 0


def test_overlay_is_white_text_on_one_black_panel():
    frame = np.full((80, 220, 3), 180, np.uint8)
    out = draw_overlay(frame, ["LIVE", "opacity 0.45"])
    assert int(out[8, 8, 0]) == 0
    panel = out[8:70, 8:160]
    assert np.any(np.all(panel == 255, axis=2))
    assert int(out[8, 8, 1]) == int(out[8, 8, 2]) == 0


def test_draw_overlay_puts_each_line_on_its_own_row():
    frame = np.zeros((80, 200, 3), np.uint8)
    out = draw_overlay(frame, ["opacity 0.45", "glow 1.0"])
    rows = np.flatnonzero(out.any(axis=(1, 2)))
    assert np.diff(rows).max() > 1


def test_fit_to_screen_fills_1080p_from_720p():
    frame = np.full((720, 1280, 3), 40, np.uint8)
    out = fit_to_screen(frame, 1920, 1080)
    assert out.shape == (1080, 1920, 3)
    assert int(out[0, 0, 0]) == 40
    assert int(out[-1, -1, 0]) == 40


def test_portrait_window_scales_to_the_screen_width_then_tiles():
    frame = np.full((4, 8, 3), 40, np.uint8)
    out = tile_vertical(frame, 4, 10)
    # 8x4 scaled to width 4 is 4x2. ceil(10 / 2) == 5 copies, edge to edge.
    assert out.shape == (10, 4, 3)
    assert np.all(out == 40)
    assert np.array_equal(out[0], out[2])
    assert np.array_equal(out[2], out[4])


def test_landscape_fullscreen_scales_and_portrait_tiles():
    frame = np.full((4, 8, 3), 40, np.uint8)
    landscape = frame_for_window(frame, 16, 8, fullscreen=True)
    assert landscape.shape == (8, 16, 3)
    portrait = frame_for_window(frame, 4, 10, fullscreen=True)
    assert portrait.shape == (10, 4, 3)
    assert np.all(portrait == 40)
    windowed = frame_for_window(frame, 4, 10, fullscreen=False)
    assert windowed.shape == frame.shape


def test_fit_to_screen_crops_a_taller_frame_instead_of_leaving_a_border():
    frame = np.full((480, 640, 3), 40, np.uint8)
    out = fit_to_screen(frame, 1920, 1080)
    assert out.shape == (1080, 1920, 3)
    assert int(out[0, 0, 0]) == 40
    assert int(out[0, -1, 0]) == 40


def test_draw_overlay_changes_pixels():
    frame = np.zeros((80, 200, 3), np.uint8)
    out = draw_overlay(frame, ["12.0 fps"])
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
    assert np.array_equal(frame, np.zeros_like(frame))


def test_fps_counter_updates_after_half_second():
    now = {"t": 0.0}

    def clock() -> float:
        return now["t"]

    counter = FpsCounter(clock)
    assert counter.tick() == 0.0
    now["t"] = 0.5
    assert counter.tick() == 4.0  # 2 frames over 0.5 s
