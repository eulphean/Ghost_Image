"""Optional GPIO buttons for hold and release.

``gpiozero`` ships with Raspberry Pi OS and is imported only when
``gpio.enabled`` is true, so the Mac does not need the package. Buttons are
wired to BCM pins with the internal pull-up, and a press is a falling edge
that ``gpiozero`` reports as ``is_pressed``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ghost_image.config import GpioConfig
from ghost_image.log import report_error

Levels = Callable[[], tuple[bool, bool]]


def rising_press(was_down: bool, is_down: bool) -> bool:
    """True on the frame a button goes from released to pressed."""
    return is_down and not was_down


class GpioControls:
    """Polls two buttons. ``levels`` injects readings in tests."""

    def __init__(self, config: GpioConfig, levels: Levels | None = None) -> None:
        self.enabled = config.enabled
        self._levels = levels
        self._hold_was = False
        self._release_was = False
        self._buttons: tuple[Any, Any] | None = None
        if config.enabled and levels is None:
            self._buttons = _open_buttons(config)

    def poll(self) -> str | None:
        """Return ``"hold"``, ``"release"``, or ``None``."""
        if not self.enabled:
            return None
        if self._levels is not None:
            hold_down, release_down = self._levels()
        elif self._buttons is None:
            return None
        else:
            hold_down = bool(self._buttons[0].is_pressed)
            release_down = bool(self._buttons[1].is_pressed)
        action = None
        if rising_press(self._hold_was, hold_down):
            action = "hold"
        elif rising_press(self._release_was, release_down):
            action = "release"
        self._hold_was = hold_down
        self._release_was = release_down
        return action

    def close(self) -> None:
        if self._buttons is None:
            return
        for button in self._buttons:
            close = getattr(button, "close", None)
            if close is not None:
                close()


def _open_buttons(config: GpioConfig) -> tuple[Any, Any] | None:
    try:
        from gpiozero import Button
    except ImportError:
        report_error("gpio is enabled but gpiozero is not installed; use the keyboard")
        return None
    return (
        Button(config.hold_pin, pull_up=True),
        Button(config.release_pin, pull_up=True),
    )
