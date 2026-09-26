"""Segmenter contract. No model and no camera."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ghost_image.config import Config, ProcessingConfig, config_from_dict
from ghost_image.segmentation import (
    DiffSegmenter,
    MediaPipeSegmenter,
    SegmentationError,
    combine_masks,
    create_segmenter,
    difference_mask,
    empty_mask,
    ensure_mask,
    person_from_confidences,
    postprocess,
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


def test_diff_segmenter_needs_no_model():
    config = config_from_dict({"processing": {"segmenter": "diff", "feather_px": 0, "morph_px": 0}})
    segmenter = create_segmenter(config)
    try:
        frame = np.zeros((8, 8, 3), np.uint8)
        mask = segmenter.mask(frame, frame)
    finally:
        segmenter.close()
    assert mask.shape == (8, 8)
    assert float(mask.max()) == 0.0


def test_hybrid_factory_requires_the_model(tmp_path):
    config = config_from_dict(
        {
            "processing": {"segmenter": "hybrid"},
            "paths": {"models_dir": str(tmp_path), "model_file": "missing.tflite"},
        }
    )
    with pytest.raises(SegmentationError, match="model not found"):
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


def _clean(**overrides: object) -> ProcessingConfig:
    processing = ProcessingConfig(
        feather_px=0,
        morph_px=0,
        mask_smoothing=0.0,
        min_blob_area=0,
    )
    for key, value in overrides.items():
        setattr(processing, key, value)
    return processing


def test_postprocess_drops_specks_and_keeps_the_body():
    mask = np.zeros((40, 40), np.float32)
    mask[0:2, 0:2] = 1.0
    mask[10:30, 10:30] = 1.0
    out = postprocess(mask, None, _clean(min_blob_area=50))
    assert float(out[0, 0]) == 0.0
    assert float(out[20, 20]) == 1.0


def test_postprocess_closes_small_holes():
    mask = np.ones((21, 21), np.float32)
    mask[9:12, 9:12] = 0.0
    out = postprocess(mask, None, _clean(morph_px=5))
    assert float(out[10, 10]) == 1.0


def test_postprocess_feathers_the_edge():
    mask = np.zeros((31, 31), np.float32)
    mask[8:24, 8:24] = 1.0
    out = postprocess(mask, None, _clean(feather_px=7))
    assert float(out[15, 15]) == pytest.approx(1.0, abs=0.05)
    assert 0.0 < float(out[7, 15]) < 1.0


def test_postprocess_smooths_toward_the_previous_mask():
    previous = np.ones((4, 4), np.float32)
    current = np.zeros((4, 4), np.float32)
    out = postprocess(current, previous, _clean(mask_smoothing=0.5))
    assert float(out[0, 0]) == pytest.approx(0.5)


def test_difference_mask_is_zero_for_identical_frames_and_one_on_a_change():
    held = np.full((20, 20, 3), 80, np.uint8)
    same = difference_mask(held, held, threshold=30)
    assert float(same.max()) == 0.0
    changed = held.copy()
    changed[5:15, 5:15] = 255
    mask = difference_mask(changed, held, threshold=30)
    assert float(mask[10, 10]) == 1.0
    assert float(mask[0, 0]) == 0.0


def test_background_adapt_follows_empty_scene_and_leaves_the_held_frame():
    segmenter = DiffSegmenter(threshold=255, adapt=0.5)
    held = np.zeros((4, 4, 3), np.uint8)
    frame = np.full((4, 4, 3), 100, np.uint8)
    mask = segmenter.mask(frame, held)
    assert float(mask.max()) == 0.0
    assert int(held[0, 0, 0]) == 0
    assert float(segmenter._background[0, 0, 0]) == pytest.approx(50.0)


def test_refine_adds_a_nearby_finger_and_drops_a_distant_speck():
    model = np.zeros((40, 40), np.float32)
    model[10:20, 10:20] = 1.0
    diff = np.zeros((40, 40), np.float32)
    diff[10:20, 20:23] = 1.0
    diff[35:37, 35:37] = 1.0
    refined = combine_masks(model, diff, "refine")
    assert float(refined[15, 21]) == 1.0
    assert float(refined[15, 15]) == 1.0
    assert float(refined[36, 36]) == 0.0
    assert float(combine_masks(model, diff, "union")[36, 36]) == 1.0
    assert float(combine_masks(model, diff, "intersect")[15, 21]) == 0.0


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
