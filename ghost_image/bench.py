"""Timing helper for segmenter benchmarks. No camera required."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def benchmark_callable(
    fn: Callable[[T], object],
    items: list[T],
    *,
    warmup: int = 1,
) -> dict[str, float]:
    """Time ``fn`` over ``items``. The first ``warmup`` calls are not counted."""
    if not items:
        raise ValueError("benchmark needs at least one item")
    for item in items[:warmup]:
        fn(item)
    start = time.perf_counter()
    for item in items:
        fn(item)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {"frames": float(len(items)), "elapsed": elapsed, "fps": len(items) / elapsed}
