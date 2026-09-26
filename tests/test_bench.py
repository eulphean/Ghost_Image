"""Benchmark helper. Does not load a model."""

from __future__ import annotations

import pytest

from ghost_image.bench import benchmark_callable


def test_benchmark_callable_skips_warmup_in_the_count():
    calls: list[int] = []

    def fn(item: int) -> None:
        calls.append(item)

    stats = benchmark_callable(fn, [1, 2, 3, 4], warmup=1)
    assert calls == [1, 1, 2, 3, 4]
    assert stats["frames"] == 4
    assert stats["fps"] > 0


def test_benchmark_callable_rejects_an_empty_list():
    with pytest.raises(ValueError, match="at least one"):
        benchmark_callable(lambda _item: None, [])
