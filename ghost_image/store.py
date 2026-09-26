"""Persisted held frame, so a power cycle does not forget the reference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ghost_image.config import PROJECT_ROOT, Config


def held_frame_path(config: Config) -> Path:
    return PROJECT_ROOT / config.paths.captures_dir / config.paths.held_frame_file


def save_held_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"could not write held frame to {path}")


def load_held_frame(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return frame


def delete_held_frame(path: Path) -> None:
    path.unlink(missing_ok=True)
