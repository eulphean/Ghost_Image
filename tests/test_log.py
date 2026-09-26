"""Logging and stage timing. The log file stays in a temporary directory."""

from __future__ import annotations

from ghost_image.config import Config
from ghost_image.log import StageTimer, report_error, setup_logging


def test_setup_logging_records_errors(tmp_path):
    config = Config()
    config.paths.log_file = str(tmp_path / "ghost.log")
    path = setup_logging(config)
    report_error("disk full")
    assert "disk full" in path.read_text(encoding="utf-8")


def test_stage_timer_records_a_name():
    timer = StageTimer()
    with timer.measure("segment"):
        sum(range(1000))
    assert timer.ms["segment"] >= 0.0
