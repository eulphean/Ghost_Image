#!/usr/bin/env bash
# Best-effort: stop the screen blanking and hide the pointer.
# Safe when xset or unclutter are missing (Wayland sessions, or a Mac).

export DISPLAY="${DISPLAY:-:0}"

if command -v xset >/dev/null 2>&1; then
    xset s off || true
    xset s noblank || true
    xset -dpms || true
fi

if command -v unclutter >/dev/null 2>&1; then
    if ! pgrep -x unclutter >/dev/null 2>&1; then
        unclutter -idle 0 -root >/dev/null 2>&1 &
    fi
fi

exit 0
