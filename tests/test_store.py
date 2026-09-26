"""Held-frame persistence. Uses temporary files, never the project captures/ dir."""

from __future__ import annotations

import numpy as np

from ghost_image.store import delete_held_frame, load_held_frame, save_held_frame


def test_roundtrip(tmp_path):
    path = tmp_path / "captures" / "held.png"
    frame = np.zeros((6, 8, 3), np.uint8)
    frame[1, 2] = (10, 20, 30)
    save_held_frame(path, frame)
    loaded = load_held_frame(path)
    assert loaded is not None
    assert loaded.shape == frame.shape
    assert tuple(int(v) for v in loaded[1, 2]) == (10, 20, 30)


def test_missing_file_loads_as_none(tmp_path):
    assert load_held_frame(tmp_path / "missing.png") is None


def test_corrupt_file_loads_as_none(tmp_path):
    path = tmp_path / "held.png"
    path.write_bytes(b"not a png")
    assert load_held_frame(path) is None


def test_delete_is_idempotent(tmp_path):
    path = tmp_path / "held.png"
    save_held_frame(path, np.zeros((2, 2, 3), np.uint8))
    delete_held_frame(path)
    delete_held_frame(path)
    assert not path.exists()
