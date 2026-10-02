"""Tests for ghost_image.config. No camera or display required."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ghost_image.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    config_from_dict,
    load_config,
)


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_defaults_are_valid():
    cfg = config_from_dict({})
    assert isinstance(cfg, Config)
    assert cfg.camera.prefer_usb is True
    assert cfg.processing.segmenter == "mediapipe"
    assert 0.0 <= cfg.ghost.alpha <= 1.0
    assert cfg.keys.quit == ["q", "esc"]


def test_shipped_config_yaml_loads_and_matches_defaults():
    """config.yaml in the repo must parse and agree with the code defaults."""
    assert DEFAULT_CONFIG_PATH.exists()
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = config_from_dict(data)
    assert cfg.to_dict() == Config().to_dict()


def test_missing_default_path_uses_defaults(tmp_path, monkeypatch):
    import ghost_image.config as config_module

    missing = tmp_path / "config.yaml"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", missing)
    cfg = config_module.load_config(None)
    assert cfg.to_dict() == Config().to_dict()


def test_missing_explicit_path_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file_uses_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path).to_dict() == Config().to_dict()


# --------------------------------------------------------------------------- #
# Overrides and merging
# --------------------------------------------------------------------------- #


def test_partial_override_keeps_other_defaults(tmp_path):
    path = write_yaml(tmp_path / "config.yaml", {"ghost": {"alpha": 0.8}})
    cfg = load_config(path)
    assert cfg.ghost.alpha == 0.8
    assert cfg.ghost.desaturate == Config().ghost.desaturate
    assert cfg.camera.width == Config().camera.width


def test_int_accepted_for_float_field(tmp_path):
    cfg = config_from_dict({"ghost": {"alpha": 1}})
    assert cfg.ghost.alpha == 1.0
    assert isinstance(cfg.ghost.alpha, float)


def test_optional_fields_accept_values_and_null():
    cfg = config_from_dict({"camera": {"preferred_index": 2, "preferred_name": "Logitech"}})
    assert cfg.camera.preferred_index == 2
    assert cfg.camera.preferred_name == "Logitech"
    cfg = config_from_dict({"camera": {"preferred_index": None}})
    assert cfg.camera.preferred_index is None


def test_local_overlay_is_deep_merged(tmp_path):
    write_yaml(
        tmp_path / "config.yaml",
        {"camera": {"width": 1920, "height": 1080}, "display": {"fullscreen": False}},
    )
    write_yaml(
        tmp_path / "config.local.yaml",
        {"camera": {"width": 640}, "display": {"fullscreen": True}},
    )
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.camera.width == 640  # overridden
    assert cfg.camera.height == 1080  # kept from base
    assert cfg.display.fullscreen is True


def test_null_section_is_ignored():
    cfg = config_from_dict({"glow": None})
    assert cfg.glow.enabled is True


def test_roundtrip_to_yaml():
    cfg = config_from_dict({"ghost": {"alpha": 0.3}})
    again = config_from_dict(yaml.safe_load(cfg.to_yaml()))
    assert again.to_dict() == cfg.to_dict()


# --------------------------------------------------------------------------- #
# Invalid input
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "data, match",
    [
        ({"ghots": {}}, "unknown config section"),
        ({"ghost": {"alhpa": 0.5}}, "unknown key"),
        ({"ghost": "not a mapping"}, "must be a mapping"),
        ({"ghost": {"alpha": "high"}}, "expected number"),
        ({"ghost": {"alpha": 1.5}}, r"alpha must be in \[0, 1\]"),
        ({"ghost": {"brightness": 0}}, "brightness"),
        ({"camera": {"width": 0}}, "camera.width"),
        ({"display": {"tile_speed": -1}}, "tile_speed"),
        ({"camera": {"hold_frames": 0}}, "hold_frames"),
        ({"camera": {"width": 12.5}}, "expected integer"),
        ({"camera": {"prefer_usb": "yes"}}, "expected true/false"),
        ({"camera": {"prefer_usb": 1}}, "expected true/false"),
        ({"camera": {"backend": "directshow"}}, "camera.backend"),
        ({"camera": {"preferred_index": -1}}, "preferred_index"),
        ({"camera": {"preferred_index": 1.5}}, "expected int, str or null"),
        ({"processing": {"segmenter": "magic"}}, "processing.segmenter"),
        ({"processing": {"feather_px": 4}}, "positive odd"),
        ({"processing": {"morph_px": 4}}, "positive odd"),
        ({"processing": {"mask_smoothing": 1.0}}, r"mask_smoothing must be in \[0, 1\)"),
        ({"processing": {"diff_threshold": 300}}, "diff_threshold"),
        ({"processing": {"hybrid_mode": "magic"}}, "hybrid_mode"),
        ({"processing": {"background_adapt": 1}}, "background_adapt"),
        ({"glow": {"color": [255, 255]}}, "three integers"),
        ({"glow": {"color": [255, 255, 256]}}, "three integers"),
        ({"glow": {"color": "white"}}, "expected list"),
        ({"glow": {"blur_px": 20}}, "positive odd"),
        ({"glow": {"thickness": 0}}, "glow.thickness"),
        ({"display": {"window_name": 42}}, "expected string"),
        ({"gpio": {"hold_pin": 40}}, "hold_pin"),
        ({"keys": {"quit": []}}, "non-empty list"),
        ({"keys": {"quit": ["q", 7]}}, "non-empty list"),
    ],
)
def test_invalid_values_raise(data, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict(data)


def test_top_level_not_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_config(path)


def test_malformed_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("camera: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config(path)
