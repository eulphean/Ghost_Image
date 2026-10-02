"""Camera discovery tests. No real camera is opened."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ghost_image.camera import (
    V4L2_CAP_VIDEO_CAPTURE,
    Camera,
    CameraDevice,
    CameraError,
    backend_flags,
    choose_device,
    format_camera_list,
    largest_capture_size,
    list_v4l2_cameras,
    open_capture,
    parse_avfoundation_listing,
    probe_indices,
    windows_camera_hint,
)
from ghost_image.config import CameraConfig


def usb(index: int, name: str = "USB Cam") -> CameraDevice:
    return CameraDevice(index, name, f"/dev/video{index}", True)


def builtin(index: int = 0, name: str = "FaceTime") -> CameraDevice:
    return CameraDevice(index, name, None, False)


def test_choose_prefers_lowest_usb_index():
    chosen = choose_device([builtin(0), usb(2), usb(1)], CameraConfig())
    assert chosen.index == 1


def test_choose_falls_back_to_builtin_when_no_usb():
    chosen = choose_device([builtin(0)], CameraConfig(prefer_usb=True))
    assert chosen.index == 0


def test_choose_keeps_builtin_when_usb_not_preferred():
    chosen = choose_device([builtin(0), usb(1)], CameraConfig(prefer_usb=False))
    assert chosen.index == 0


def test_avfoundation_listing_marks_external_cameras_usb():
    text = "\n".join(
        [
            "0\tUSB Video\tAVCaptureDeviceTypeExternal",
            "1\tMacBook Pro Camera\tAVCaptureDeviceTypeBuiltInWideAngleCamera",
            "2\tMacBook Pro Desk View Camera\tAVCaptureDeviceTypeDeskViewCamera",
        ]
    )
    devices = parse_avfoundation_listing(text)
    chosen = choose_device(devices, CameraConfig(prefer_usb=True))
    assert chosen.index == 0
    assert chosen.name == "USB Video"
    assert chosen.usb is True
    assert devices[1].usb is False


def test_preferred_index_overrides_usb_heuristic():
    chosen = choose_device([builtin(0), usb(1)], CameraConfig(preferred_index=0))
    assert chosen.index == 0


def test_preferred_index_missing():
    with pytest.raises(CameraError, match="not found"):
        choose_device([usb(1)], CameraConfig(preferred_index=3))


def test_preferred_name_matches_substring():
    chosen = choose_device(
        [usb(0, "Built-in"), usb(2, "Logitech C920")],
        CameraConfig(preferred_name="logi"),
    )
    assert chosen.index == 2


def test_preferred_name_miss_raises():
    with pytest.raises(CameraError, match="no camera name"):
        choose_device([usb(0, "FaceTime")], CameraConfig(preferred_name="Logitech"))


def test_empty_device_list_raises():
    with pytest.raises(CameraError, match="no cameras"):
        choose_device([], CameraConfig())


def test_format_empty_list():
    assert format_camera_list([]) == "no cameras found"
    assert "Logitech" in format_camera_list([usb(1, "Logitech")])


class FakeCap:
    def __init__(self, frames: list[np.ndarray | None], opened: bool = True) -> None:
        self._frames = list(frames)
        self.opened = opened
        self.released = False
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self):
        if not self._frames:
            return False, None
        frame = self._frames.pop(0)
        if frame is None:
            return False, None
        return True, frame

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def release(self) -> None:
        self.released = True


def test_probe_indices_marks_zero_builtin_and_keeps_frames():
    good = np.zeros((2, 2, 3), np.uint8)
    caps = {
        0: FakeCap([good]),
        1: FakeCap([None]),
        2: FakeCap([good]),
    }

    def opener(index: int) -> FakeCap:
        return caps.get(index, FakeCap([], opened=False))

    found = probe_indices(opener, 4)
    assert [(d.index, d.usb) for d in found] == [(0, False), (2, True)]
    assert all(cap.released for cap in caps.values())


def _v4l_node(root: Path, index: int, name: str, *, usb: bool) -> Path:
    dev = root / "dev"
    dev.mkdir(exist_ok=True)
    (dev / f"video{index}").touch()
    node = root / "sys" / "class" / "video4linux" / f"video{index}"
    node.mkdir(parents=True)
    (node / "name").write_text(name + "\n", encoding="utf-8")
    kind = "usb1/1-1" if usb else "platform/soc"
    target = root / "sys" / "devices" / kind / f"video{index}"
    target.mkdir(parents=True)
    (node / "device").symlink_to(target)
    return dev / f"video{index}"


def test_v4l2_keeps_usb_capture_and_skips_metadata(tmp_path: Path):
    usb_dev = _v4l_node(tmp_path, 0, "Logitech C920", usb=True)
    meta = _v4l_node(tmp_path, 1, "Logitech C920", usb=True)
    pci = _v4l_node(tmp_path, 2, "Built-in", usb=False)
    caps = {
        usb_dev: V4L2_CAP_VIDEO_CAPTURE,
        meta: 0x00800000,  # metadata only
        pci: V4L2_CAP_VIDEO_CAPTURE,
    }
    found = list_v4l2_cameras(
        tmp_path / "dev",
        tmp_path / "sys" / "class" / "video4linux",
        caps.get,
    )
    assert [(d.index, d.usb, d.name) for d in found] == [
        (0, True, "Logitech C920"),
        (2, False, "Built-in"),
    ]


def test_v4l2_missing_sysfs_is_empty(tmp_path: Path):
    assert list_v4l2_cameras(tmp_path / "dev", tmp_path / "sys", lambda _path: None) == []


def test_windows_camera_hint_names_devices_windows_can_see():
    assert "USB Video" in windows_camera_hint(["USB Video"])
    assert "does not see a camera" in windows_camera_hint([])


def test_largest_capture_size_keeps_the_biggest_mode_the_camera_accepts():
    class Cap:
        def __init__(self) -> None:
            self.w = 0
            self.h = 0
            self.supported = {(1920, 1080), (1280, 720), (640, 480)}

        def set(self, prop: int, value: float) -> bool:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self.w = int(value)
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self.h = int(value)
            return True

        def get(self, prop: int) -> float:
            if (self.w, self.h) not in self.supported:
                return 0.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.w)
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.h)
            return 0.0

    assert largest_capture_size(Cap()) == (1920, 1080)


def test_windows_auto_tries_directshow_then_media_foundation():
    assert backend_flags(CameraConfig(), "Windows") == [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    assert backend_flags(CameraConfig(backend="msmf"), "Windows") == [cv2.CAP_MSMF]


def test_open_capture_uses_the_backend_recorded_on_the_device(monkeypatch):
    frame = np.zeros((4, 4, 3), np.uint8)
    opened: list[tuple[int, int]] = []

    def fake_capture(index: int, backend: int) -> FakeCap:
        opened.append((index, backend))
        return FakeCap([frame])

    monkeypatch.setattr("ghost_image.camera.cv2.VideoCapture", fake_capture)
    device = CameraDevice(0, "USB Video", None, True, backend=cv2.CAP_MSMF)
    cap, chosen = open_capture(
        CameraConfig(),
        system="Windows",
        lister=lambda: [device],
    )
    assert opened == [(0, cv2.CAP_MSMF)]
    assert chosen.backend == cv2.CAP_MSMF
    cap.release()


def test_open_capture_uses_lister_and_requests_size():
    frame = np.zeros((4, 4, 3), np.uint8)
    opened: list[int] = []

    def opener(index: int) -> FakeCap:
        opened.append(index)
        return FakeCap([frame])

    config = CameraConfig(width=640, height=480, fps=15)
    cap, device = open_capture(
        config,
        system="Darwin",
        opener=opener,
        lister=lambda: [builtin(0), usb(1, "USB")],
    )
    assert device.index == 1
    assert opened == [1]
    assert cap.props  # resolution / fps were requested
    cap.release()


def test_open_capture_raises_when_no_frame():
    config = CameraConfig(preferred_index=0)
    with pytest.raises(CameraError, match="did not return a frame"):
        open_capture(
            config,
            system="Darwin",
            opener=lambda _index: FakeCap([None]),
            lister=lambda: [builtin(0)],
        )


def test_reconnect_after_consecutive_failures():
    bad = FakeCap([None, None, None])
    good_frame = np.full((2, 2, 3), 7, np.uint8)
    good = FakeCap([good_frame])

    def open_fn():
        return good, usb(1, "replugged")

    camera = Camera(bad, usb(0), CameraConfig(reconnect_failures=2), open_fn)
    assert camera.read() is None
    frame = camera.read()
    assert frame is not None and int(frame[0, 0, 0]) == 7
    assert camera.device.name == "replugged"
    assert bad.released


def test_reconnect_failure_does_not_raise():
    bad = FakeCap([None])

    def open_fn():
        raise CameraError("still gone")

    camera = Camera(bad, usb(0), CameraConfig(reconnect_failures=1), open_fn)
    assert camera.read() is None
    assert camera.read() is None
