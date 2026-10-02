"""Command-line entry point: ``python -m ghost_image``."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from ghost_image import __version__
from ghost_image.app import benchmark_capture, format_benchmark, run
from ghost_image.camera import (
    CameraError,
    format_camera_list,
    list_cameras,
    list_windows_camera_names,
    open_camera,
    windows_camera_hint,
)
from ghost_image.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from ghost_image.log import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ghost_image",
        description="Ghost Image: render people as translucent ghosts over a held camera frame.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"path to config.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="print the resolved configuration and exit",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="print discovered cameras and exit",
    )
    parser.add_argument(
        "--restore-held",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reload captures/held.png at startup (default: config paths.restore_held)",
    )
    parser.add_argument(
        "--benchmark-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        help="read the camera for SECONDS and print FPS, without opening a window",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.show_config:
        print(config.to_yaml())
        return 0

    if args.list_cameras:
        try:
            devices = list_cameras(config.camera)
        except CameraError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_camera_list(devices))
        if not devices and platform.system() == "Windows":
            print(windows_camera_hint(list_windows_camera_names()), file=sys.stderr)
        return 0 if devices else 1

    if args.benchmark_seconds is not None:
        if args.benchmark_seconds <= 0:
            print("error: --benchmark-seconds must be > 0", file=sys.stderr)
            return 2
        try:
            camera = open_camera(config.camera)
        except CameraError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            stats = benchmark_capture(camera, args.benchmark_seconds)
        finally:
            camera.release()
        print(format_benchmark(stats))
        return 0

    if args.restore_held is not None:
        config.paths.restore_held = args.restore_held

    setup_logging(config)
    print(f"Ghost Image v{__version__}")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    try:
        return run(config)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
