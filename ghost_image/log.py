"""Rotating log file and per-stage timings."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ghost_image.config import PROJECT_ROOT, Config

LOGGER = logging.getLogger("ghost_image")
LOGGER.addHandler(logging.NullHandler())


def setup_logging(config: Config) -> Path:
    """Send ``ghost_image`` logs to a rotating file. Safe to call again."""
    path = PROJECT_ROOT / config.paths.log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(LOGGER.handlers):
        if isinstance(handler, RotatingFileHandler):
            LOGGER.removeHandler(handler)
            handler.close()
    handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    return path


def report_error(message: str) -> None:
    """Print an error for the operator and keep a copy in the log."""
    print(f"error: {message}", file=sys.stderr)
    LOGGER.error(message)


class StageTimer:
    """Milliseconds spent in named stages of the last frame."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = (time.perf_counter() - start) * 1000.0
