"""Keep the app loop off the real segmentation model."""

from __future__ import annotations

import numpy as np
import pytest

from ghost_image.segmentation import Segmenter, empty_mask


class _ZeroSegmenter(Segmenter):
    name = "fake"

    def mask(self, frame: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
        return empty_mask(frame)

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _no_real_segmenter(monkeypatch, tmp_path):
    """App tests must not load MediaPipe or write the project's captures/ folder."""
    monkeypatch.setattr("ghost_image.app.create_segmenter", lambda _config: _ZeroSegmenter())
    monkeypatch.setattr("ghost_image.app.held_frame_path", lambda _config: tmp_path / "held.png")
