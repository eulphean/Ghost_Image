"""Segmenter contract. No model and no camera."""

from __future__ import annotations

import numpy as np
import pytest

from ghost_image.config import Config, config_from_dict
from ghost_image.segmentation import (
    SegmentationError,
    create_segmenter,
    empty_mask,
    ensure_mask,
)


def test_ensure_mask_clips_and_casts():
    frame = np.zeros((4, 6, 3), np.uint8)
    mask = np.array(
        [
            [-1.0, 0.0, 0.2, 0.4, 0.6, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        np.float64,
    )
    out = ensure_mask(mask, frame)
    assert out.dtype == np.float32
    assert out.shape == (4, 6)
    assert float(out[0, 0]) == 0.0
    assert float(out[0, -1]) == 1.0


def test_ensure_mask_upsamples_to_the_frame():
    frame = np.zeros((8, 10, 3), np.uint8)
    small = np.ones((2, 2), np.float32)
    out = ensure_mask(small, frame)
    assert out.shape == (8, 10)
    assert out.dtype == np.float32
    assert float(out.min()) == pytest.approx(1.0)


def test_empty_mask_matches_frame():
    frame = np.zeros((3, 5, 3), np.uint8)
    mask = empty_mask(frame)
    assert mask.shape == (3, 5)
    assert mask.dtype == np.float32
    assert float(mask.sum()) == 0.0


@pytest.mark.parametrize("kind", ["mediapipe", "diff", "hybrid"])
def test_factory_rejects_engines_until_they_exist(kind):
    config = config_from_dict({"processing": {"segmenter": kind}})
    with pytest.raises(SegmentationError, match="not available yet"):
        create_segmenter(config)


def test_factory_uses_the_configured_name():
    assert Config().processing.segmenter == "mediapipe"
