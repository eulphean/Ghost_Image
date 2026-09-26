"""Segmenter contract. No model and no camera."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ghost_image.config import Config, config_from_dict
from ghost_image.segmentation import (
    MediaPipeSegmenter,
    SegmentationError,
    create_segmenter,
    empty_mask,
    ensure_mask,
    person_from_confidences,
    resize_to_width,
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


@pytest.mark.parametrize("kind", ["diff", "hybrid"])
def test_factory_rejects_engines_until_they_exist(kind):
    config = config_from_dict({"processing": {"segmenter": kind}})
    with pytest.raises(SegmentationError, match="not available yet"):
        create_segmenter(config)


def test_mediapipe_factory_requires_the_model(tmp_path):
    config = config_from_dict(
        {
            "processing": {"segmenter": "mediapipe"},
            "paths": {"models_dir": str(tmp_path), "model_file": "missing.tflite"},
        }
    )
    with pytest.raises(SegmentationError, match="model not found"):
        create_segmenter(config)


def test_factory_uses_the_configured_name():
    assert Config().processing.segmenter == "mediapipe"


def test_resize_to_width_keeps_aspect():
    frame = np.zeros((10, 20, 3), np.uint8)
    out = resize_to_width(frame, 10)
    assert out.shape == (5, 10, 3)


def test_person_from_single_and_multiclass_masks():
    single = [np.array([[0.2, 1.5], [0.0, 0.4]], np.float32)]
    person = person_from_confidences(single)
    assert float(person[0, 1]) == 1.0
    background = np.full((2, 2), 0.25, np.float32)
    other = np.zeros((2, 2), np.float32)
    combined = person_from_confidences([background, other, other])
    assert float(combined[0, 0]) == pytest.approx(0.75)


class FakeEngine:
    def __init__(self) -> None:
        self.closed = False
        self.seen: list[tuple[int, int]] = []

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.seen.append(frame_bgr.shape[:2])
        return np.ones(frame_bgr.shape[:2], np.float32)

    def close(self) -> None:
        self.closed = True


def test_async_segmenter_upsamples_on_a_worker_thread(tmp_path):
    engine = FakeEngine()
    segmenter = MediaPipeSegmenter(tmp_path / "unused.tflite", width=8, engine=engine)
    frame = np.zeros((16, 20, 3), np.uint8)
    try:
        deadline = time.monotonic() + 2
        mask = empty_mask(frame)
        while time.monotonic() < deadline and float(mask.max()) == 0.0:
            mask = segmenter.mask(frame)
            time.sleep(0.01)
    finally:
        segmenter.close()
    assert mask.shape == (16, 20)
    assert float(mask.max()) == pytest.approx(1.0)
    assert engine.closed
    assert engine.seen  # inference ran on the resized frame
    assert engine.seen[0][1] == 8
