#!/usr/bin/env python3
"""Benchmark person segmenters on synthetic frames.

Run this on the machine you care about (the Mac, or the Pi) and paste the
lines into docs/BENCHMARKS.md. It does not open a camera. MediaPipe is timed
synchronously, which is the cost the worker thread has to keep up with.

Usage:
    python scripts/bench_segmenters.py
    python scripts/bench_segmenters.py --frames 40 --widths 256 320
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ghost_image.bench import benchmark_callable  # noqa: E402
from ghost_image.config import PROJECT_ROOT, ProcessingConfig  # noqa: E402
from ghost_image.segmentation import (  # noqa: E402
    DiffSegmenter,
    MediaPipeEngine,
    combine_masks,
    difference_mask,
    postprocess,
    resize_to_width,
)


def synthetic_frames(count: int, width: int = 640, height: int = 480) -> list[np.ndarray]:
    frames = []
    for index in range(count):
        frame = np.full((height, width, 3), 40, np.uint8)
        origin = (index * 17) % (width - 80)
        frame[80:400, origin : origin + 80] = (180, 160, 140)
        frames.append(frame)
    return frames


def upscale(mask: np.ndarray, width: int) -> np.ndarray:
    height = max(1, int(round(mask.shape[0] * (width / mask.shape[1]))))
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--widths", type=int, nargs="+", default=[256, 320])
    args = parser.parse_args(argv)
    if args.frames < 2:
        print("error: --frames must be >= 2", file=sys.stderr)
        return 2

    frames = synthetic_frames(args.frames)
    held = frames[0]
    model = PROJECT_ROOT / "models" / "selfie_segmenter.tflite"
    print(
        f"platform={platform.system()} {platform.machine()} "
        f"python={platform.python_version()} frames={args.frames} size=640x480"
    )

    for width in args.widths:
        processing = ProcessingConfig(width=width)
        diff = DiffSegmenter(processing.diff_threshold)

        def run_diff(
            frame: np.ndarray,
            diff: DiffSegmenter = diff,
            processing: ProcessingConfig = processing,
        ) -> None:
            postprocess(diff.mask(frame, held), None, processing)

        stats = benchmark_callable(run_diff, frames, warmup=1)
        print(f"diff width={width} fps={stats['fps']:.1f}")
        diff.close()

        if not model.is_file():
            print(f"mediapipe width={width} skipped (model missing)")
            print(f"hybrid width={width} skipped (model missing)")
            continue

        engine = MediaPipeEngine(model)

        def run_mp(
            frame: np.ndarray,
            engine: MediaPipeEngine = engine,
            width: int = width,
            processing: ProcessingConfig = processing,
        ) -> None:
            raw = engine.segment(resize_to_width(frame, width))
            postprocess(upscale(raw, frame.shape[1]), None, processing)

        stats = benchmark_callable(run_mp, frames, warmup=2)
        print(f"mediapipe width={width} fps={stats['fps']:.1f}")

        def run_hybrid(
            frame: np.ndarray,
            engine: MediaPipeEngine = engine,
            width: int = width,
            processing: ProcessingConfig = processing,
        ) -> None:
            raw = upscale(engine.segment(resize_to_width(frame, width)), frame.shape[1])
            diff_mask = difference_mask(frame, held, processing.diff_threshold)
            postprocess(combine_masks(raw, diff_mask, "refine"), None, processing)

        stats = benchmark_callable(run_hybrid, frames, warmup=1)
        print(f"hybrid width={width} fps={stats['fps']:.1f}")
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
