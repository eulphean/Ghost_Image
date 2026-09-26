"""Configuration loading and validation.

All tunables live in ``config.yaml`` at the project root. An optional
``config.local.yaml`` beside it is deep-merged on top, so a machine-specific
file (e.g. lower processing resolution on the Raspberry Pi) can override a
few values without editing the shared config. Unknown keys are rejected so
typos are caught at startup rather than silently ignored.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOCAL_CONFIG_NAME = "config.local.yaml"

BACKENDS = ("auto", "v4l2", "avfoundation", "any")
SEGMENTERS = ("mediapipe", "diff", "hybrid")
HYBRID_MODES = ("union", "intersect", "refine")


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed, or invalid."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class CameraConfig:
    """How to find and open the camera."""

    preferred_index: int | None = None  # force a specific device index
    preferred_name: str | None = None  # substring match on device name (Linux)
    prefer_usb: bool = True  # skip built-in cameras when a USB one exists
    backend: str = "auto"  # auto | v4l2 | avfoundation | any
    width: int = 1280
    height: int = 720
    fps: int = 30
    reconnect_failures: int = 30  # consecutive read failures before rediscovery
    hold_frames: int = 4  # frames averaged into the held reference


@dataclass
class ProcessingConfig:
    """Segmentation and mask clean-up."""

    segmenter: str = "mediapipe"  # mediapipe | diff | hybrid
    width: int = 320  # frame width fed to the segmenter
    frame_skip: int = 0  # 0 = segment every frame, 1 = every other frame, ...
    mask_threshold: float = 0.5  # binarisation threshold for contours
    mask_smoothing: float = 0.5  # EMA weight of previous mask (0 = off)
    feather_px: int = 7  # Gaussian feathering of mask edges (odd, 0 = off)
    morph_px: int = 5  # open/close kernel (odd, 0 = off)
    min_blob_area: int = 400  # remove connected components smaller than this
    diff_threshold: int = 30  # frame-difference segmenter threshold (0-255)
    hybrid_mode: str = "refine"  # union | intersect | refine


@dataclass
class GhostConfig:
    """Appearance of the translucent person."""

    alpha: float = 0.45  # 0 = invisible, 1 = fully opaque
    alpha_step: float = 0.05  # change per key press
    desaturate: float = 0.7  # 0 = original colour, 1 = greyscale
    tint: list[int] = field(default_factory=lambda: [255, 230, 200])  # BGR
    tint_strength: float = 0.25
    brightness: float = 1.08  # slight lift so the ghost reads lighter than the room
    blur_px: int = 0  # soften the ghost body (odd, 0 = off)


@dataclass
class GlowConfig:
    """Luminous outline around the ghost."""

    enabled: bool = True
    color: list[int] = field(default_factory=lambda: [255, 240, 200])  # BGR
    thickness: int = 3
    blur_px: int = 21  # halo softness (odd)
    intensity: float = 1.0
    pulse: bool = True
    pulse_period_s: float = 4.0


@dataclass
class DisplayConfig:
    window_name: str = "Ghost Image"
    fullscreen: bool = False
    show_fps: bool = True
    debug: bool = False


@dataclass
class PathsConfig:
    models_dir: str = "models"
    model_file: str = "selfie_segmenter.tflite"
    captures_dir: str = "captures"
    held_frame_file: str = "held.png"
    restore_held: bool = True  # reload the held frame at startup if present


@dataclass
class KeysConfig:
    """Keyboard bindings. Each entry is a list of key names accepted by ``ui``."""

    hold: list[str] = field(default_factory=lambda: ["space"])
    release: list[str] = field(default_factory=lambda: ["r"])
    debug: list[str] = field(default_factory=lambda: ["d"])
    fullscreen: list[str] = field(default_factory=lambda: ["f"])
    opacity_down: list[str] = field(default_factory=lambda: ["["])
    opacity_up: list[str] = field(default_factory=lambda: ["]"])
    snapshot: list[str] = field(default_factory=lambda: ["s"])
    quit: list[str] = field(default_factory=lambda: ["q", "esc"])


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ghost: GhostConfig = field(default_factory=GhostConfig)
    glow: GlowConfig = field(default_factory=GlowConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    keys: KeysConfig = field(default_factory=KeysConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_config(path: Path | str | None = None) -> Config:
    """Load ``config.yaml`` (plus ``config.local.yaml`` if present) and validate.

    If *path* is ``None`` or the default path and the file does not exist,
    built-in defaults are used. An explicitly given path that does not exist
    is an error.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if path.exists():
        data = _read_yaml(path)
    elif path == DEFAULT_CONFIG_PATH:
        data = {}
    else:
        raise ConfigError(f"config file not found: {path}")

    local_path = path.parent / LOCAL_CONFIG_NAME
    if local_path.exists():
        data = _deep_merge(data, _read_yaml(local_path))

    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Build and validate a :class:`Config` from a (possibly partial) mapping."""
    if not isinstance(data, dict):
        raise ConfigError("top level of config must be a mapping")

    config = Config()
    section_names = {f.name for f in fields(Config)}
    unknown = set(data) - section_names
    if unknown:
        raise ConfigError(f"unknown config section(s): {', '.join(sorted(unknown))}")

    for f in fields(Config):
        if f.name in data:
            section_data = data[f.name]
            if section_data is None:
                continue
            if not isinstance(section_data, dict):
                raise ConfigError(f"section '{f.name}' must be a mapping")
            _apply_section(getattr(config, f.name), f.name, section_data)

    _validate(config)
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_section(section: Any, name: str, data: dict[str, Any]) -> None:
    known = {f.name: f for f in fields(section)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"unknown key(s) in '{name}': {', '.join(sorted(unknown))}")
    for key, value in data.items():
        setattr(section, key, _coerce(f"{name}.{key}", value, getattr(section, key)))


def _coerce(label: str, value: Any, default: Any) -> Any:
    """Check *value* against the type of *default* and return it."""
    # Optional fields: default None means int|str|None depending on the field.
    if default is None:
        if value is None or isinstance(value, (int, str)) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{label}: expected int, str or null, got {type(value).__name__}")
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{label}: expected true/false, got {value!r}")
    if isinstance(default, int):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{label}: expected integer, got {value!r}")
    if isinstance(default, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ConfigError(f"{label}: expected number, got {value!r}")
    if isinstance(default, str):
        if isinstance(value, str):
            return value
        raise ConfigError(f"{label}: expected string, got {value!r}")
    if isinstance(default, list):
        if isinstance(value, list):
            return list(value)
        raise ConfigError(f"{label}: expected list, got {value!r}")
    return value  # pragma: no cover - no other field types are defined


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate(cfg: Config) -> None:
    cam, proc, ghost, glow, keys = cfg.camera, cfg.processing, cfg.ghost, cfg.glow, cfg.keys

    _check(cam.backend in BACKENDS, f"camera.backend must be one of {BACKENDS}")
    _check(cam.width > 0 and cam.height > 0, "camera.width and camera.height must be > 0")
    _check(cam.fps > 0, "camera.fps must be > 0")
    _check(cam.reconnect_failures >= 1, "camera.reconnect_failures must be >= 1")
    _check(cam.hold_frames >= 1, "camera.hold_frames must be >= 1")
    _check(
        cam.preferred_index is None or cam.preferred_index >= 0,
        "camera.preferred_index must be >= 0 or null",
    )

    _check(proc.segmenter in SEGMENTERS, f"processing.segmenter must be one of {SEGMENTERS}")
    _check(proc.width > 0, "processing.width must be > 0")
    _check(proc.frame_skip >= 0, "processing.frame_skip must be >= 0")
    _check(0.0 <= proc.mask_threshold <= 1.0, "processing.mask_threshold must be in [0, 1]")
    _check(0.0 <= proc.mask_smoothing < 1.0, "processing.mask_smoothing must be in [0, 1)")
    _check_kernel("processing.feather_px", proc.feather_px)
    _check_kernel("processing.morph_px", proc.morph_px)
    _check(proc.min_blob_area >= 0, "processing.min_blob_area must be >= 0")
    _check(0 <= proc.diff_threshold <= 255, "processing.diff_threshold must be in [0, 255]")
    _check(
        proc.hybrid_mode in HYBRID_MODES,
        f"processing.hybrid_mode must be one of {HYBRID_MODES}",
    )

    _check(0.0 <= ghost.alpha <= 1.0, "ghost.alpha must be in [0, 1]")
    _check(0.0 < ghost.alpha_step <= 1.0, "ghost.alpha_step must be in (0, 1]")
    _check(0.0 <= ghost.desaturate <= 1.0, "ghost.desaturate must be in [0, 1]")
    _check(0.0 <= ghost.tint_strength <= 1.0, "ghost.tint_strength must be in [0, 1]")
    _check(ghost.brightness > 0.0, "ghost.brightness must be > 0")
    _check_color("ghost.tint", ghost.tint)
    _check_kernel("ghost.blur_px", ghost.blur_px)

    _check_color("glow.color", glow.color)
    _check(glow.thickness >= 1, "glow.thickness must be >= 1")
    _check_kernel("glow.blur_px", glow.blur_px)
    _check(glow.intensity >= 0.0, "glow.intensity must be >= 0")
    _check(glow.pulse_period_s > 0.0, "glow.pulse_period_s must be > 0")

    for f in fields(keys):
        value = getattr(keys, f.name)
        _check(
            len(value) > 0 and all(isinstance(k, str) and k for k in value),
            f"keys.{f.name} must be a non-empty list of key names",
        )


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_kernel(label: str, value: int) -> None:
    """Blur/feather sizes must be 0 (off) or a positive odd number (OpenCV requirement)."""
    _check(value == 0 or (value > 0 and value % 2 == 1), f"{label} must be 0 or a positive odd int")


def _check_color(label: str, value: list[Any]) -> None:
    _check(
        len(value) == 3
        and all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in value),
        f"{label} must be a list of three integers in [0, 255] (BGR)",
    )
