"""Phone hold page. Binds to localhost only; no camera or display."""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

from ghost_image.app import run
from ghost_image.config import Config, RemoteConfig
from ghost_image.remote import (
    PhoneRequest,
    RemoteControls,
    local_ipv4_addresses,
    page_urls,
    pin_matches,
)
from tests.test_app import FakeCamera, FakeDisplay


def test_pin_matches_requires_the_same_value():
    assert pin_matches("2468", "2468")
    assert not pin_matches("2468", "2469")
    assert not pin_matches("2468", "24680")


def test_page_urls_use_lan_addresses_when_listening_everywhere(monkeypatch):
    monkeypatch.setattr("ghost_image.remote.local_ipv4_addresses", lambda: ["10.0.0.8"])
    assert page_urls("0.0.0.0", 8080) == ["http://10.0.0.8:8080"]
    assert page_urls("127.0.0.1", 9) == ["http://127.0.0.1:9"]


def test_local_ipv4_addresses_skip_loopback():
    for ip in local_ipv4_addresses():
        assert isinstance(ip, str)
        assert not ip.startswith("127.")


def test_disabled_phone_does_not_listen():
    phone = RemoteControls(RemoteConfig(enabled=False))
    assert phone.poll() is None
    phone.close()


def test_wrong_pin_is_refused_and_five_failures_lock_the_page():
    phone = RemoteControls(RemoteConfig(enabled=True, host="127.0.0.1", port=0, pin="2468"))
    try:
        for _ in range(5):
            _post(phone.port, "0000", expect=403)
        held = _post(phone.port, "2468")
        assert "Too many tries" in held
        assert phone.poll() is None
    finally:
        phone.close()


def test_correct_pin_asks_the_camera_loop_to_hold():
    phone = RemoteControls(RemoteConfig(enabled=True, host="127.0.0.1", port=0, pin="2468"))

    def acknowledge() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            request = phone.poll()
            if request is not None:
                request.ok = True
                request.done.set()
                return
            time.sleep(0.01)

    threading.Thread(target=acknowledge, daemon=True).start()
    try:
        body = _post(phone.port, "2468")
        assert "Held." in body
    finally:
        phone.close()


def test_phone_address_is_shown_on_the_first_frames():
    class Phone:
        url = "http://10.0.0.8:8080"

        def poll(self) -> None:
            return None

        def close(self) -> None:
            return

    display = FakeDisplay([ord("q")])
    run(
        Config(),
        camera=FakeCamera([np.full((8, 8, 3), 10, np.uint8)]),
        display=display,
        phone=Phone(),
    )
    assert display.notices[0] == "http://10.0.0.8:8080"


def test_phone_hold_saves_the_current_picture(tmp_path, monkeypatch):
    monkeypatch.setattr("ghost_image.app.held_frame_path", lambda _config: tmp_path / "held.png")
    request = PhoneRequest()
    run(
        Config(),
        camera=FakeCamera([np.full((8, 8, 3), 10, np.uint8)]),
        display=FakeDisplay([-1, ord("q")]),
        phone=_OneRequest(request),
    )
    assert request.ok
    assert request.done.is_set()
    held = cv2.imread(str(tmp_path / "held.png"))
    assert held is not None
    assert int(held[0, 0, 0]) == 10


def test_phone_release_deletes_the_saved_frame(tmp_path, monkeypatch):
    path = tmp_path / "held.png"
    monkeypatch.setattr("ghost_image.app.held_frame_path", lambda _config: path)
    cv2.imwrite(str(path), np.full((8, 8, 3), 10, np.uint8))
    request = PhoneRequest(action="release")
    run(
        Config(),
        camera=FakeCamera([np.full((8, 8, 3), 40, np.uint8)]),
        display=FakeDisplay([-1, ord("q")]),
        phone=_OneRequest(request),
    )
    assert request.ok
    assert request.done.is_set()
    assert not path.is_file()


def test_correct_pin_can_release():
    phone = RemoteControls(RemoteConfig(enabled=True, host="127.0.0.1", port=0, pin="2468"))

    def acknowledge() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            request = phone.poll()
            if request is not None:
                assert request.action == "release"
                request.ok = True
                request.done.set()
                return
            time.sleep(0.01)

    threading.Thread(target=acknowledge, daemon=True).start()
    try:
        body = _post(phone.port, "2468", path="/release")
        assert "Released." in body
    finally:
        phone.close()


class _OneRequest:
    def __init__(self, request: PhoneRequest) -> None:
        self._request: PhoneRequest | None = request

    def poll(self) -> PhoneRequest | None:
        request = self._request
        self._request = None
        return request

    def close(self) -> None:
        return


def _post(port: int, pin: str, expect: int = 200, path: str = "/hold") -> str:
    data = urllib.parse.urlencode({"pin": pin}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            body = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        status = exc.code
    assert status == expect
    return body
