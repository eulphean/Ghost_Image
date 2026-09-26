"""Command-line entry point: ``python -m ghost_image``."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from ghost_image import __version__
from ghost_image.camera import CameraError, format_camera_list, list_cameras
from ghost_image.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


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
        return 0 if devices else 1

    # Phase 1 scaffold: report environment and exit. The camera/live view
    # main loop is added in Phase 2 (see docs/PLAN.md).
    print(f"Ghost Image v{__version__}")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"Config loaded from {args.config if args.config.exists() else '<defaults>'}")
    print("Nothing to run yet: camera and live view arrive in Phase 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
