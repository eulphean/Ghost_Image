"""Cross-platform USB camera discovery and capture.

Linux (Raspberry Pi) enumerates V4L2 nodes under ``/dev/video*`` and keeps the
ones that sit on a USB bus and advertise video capture. macOS asks AVFoundation
for device names and sorts them the same way OpenCV does (by unique id), so
index 0 is not assumed to be the built-in camera. ``preferred_index`` overrides
auto-selection on either platform.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ghost_image.config import CameraConfig

V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
PROBE_COUNT = 6
SYSFS_V4L = Path("/sys/class/video4linux")
DEV_DIR = Path("/dev")

Opener = Callable[[int], cv2.VideoCapture]


class CameraError(RuntimeError):
    """Raised when no usable camera can be opened."""


@dataclass(frozen=True)
class CameraDevice:
    """One capture device discovered on this machine."""

    index: int
    name: str
    path: str | None
    usb: bool

    @property
    def description(self) -> str:
        kind = "usb" if self.usb else "builtin"
        where = f" {self.path}" if self.path else ""
        return f"{self.index}: {self.name} [{kind}]{where}"


Lister = Callable[[], list[CameraDevice]]


def list_cameras(config: CameraConfig, *, system: str | None = None) -> list[CameraDevice]:
    """Return capture devices on this machine, lowest index first."""
    system = system if system is not None else platform.system()
    if system == "Linux":
        return list_v4l2_cameras(DEV_DIR, SYSFS_V4L, read_v4l2_device_caps)
    if system == "Darwin":
        named = list_avfoundation_cameras()
        if named:
            return named
        backend = cv2.CAP_AVFOUNDATION
        return probe_indices(lambda index: cv2.VideoCapture(index, backend), PROBE_COUNT)
    backend = backend_flag(config, system)
    return probe_indices(lambda index: cv2.VideoCapture(index, backend), PROBE_COUNT)


def list_v4l2_cameras(
    dev_dir: Path,
    sysfs_dir: Path,
    caps_reader: Callable[[Path], int | None],
) -> list[CameraDevice]:
    """List V4L2 capture nodes. Metadata-only nodes are dropped.

    ``caps_reader`` returns the device's capability bitfield, or ``None`` when
    the node cannot be queried. A node is kept when those bits include video
    capture. USB is decided from the sysfs device symlink: a real path
    containing ``usb`` is a USB camera.
    """
    if not sysfs_dir.is_dir():
        return []

    devices: list[CameraDevice] = []
    for node in sorted(sysfs_dir.glob("video*"), key=lambda p: _video_index(p.name)):
        if not node.is_dir():
            continue
        try:
            index = _video_index(node.name)
        except ValueError:
            continue
        dev_path = dev_dir / node.name
        caps = caps_reader(dev_path)
        if caps is None or not _is_capture(caps):
            continue
        name_file = node / "name"
        name = name_file.read_text(encoding="utf-8").strip() if name_file.is_file() else node.name
        devices.append(
            CameraDevice(
                index=index,
                name=name or node.name,
                path=str(dev_path),
                usb=_sysfs_is_usb(node / "device"),
            )
        )
    return devices


_AVFOUNDATION_LIST = """
import AVFoundation
let types: [AVCaptureDevice.DeviceType] = [
  .builtInWideAngleCamera, .external, .deskViewCamera, .continuityCamera
]
let session = AVCaptureDevice.DiscoverySession(
  deviceTypes: types, mediaType: .video, position: .unspecified)
let devices = session.devices.sorted { $0.uniqueID < $1.uniqueID }
for (i, device) in devices.enumerated() {
  let name = device.localizedName.replacingOccurrences(of: "\\t", with: " ")
  print("\\(i)\\t\\(name)\\t\\(device.deviceType.rawValue)")
}
"""


def list_avfoundation_cameras() -> list[CameraDevice]:
    """Cameras in OpenCV's index order, with real names.

    OpenCV sorts AVFoundation devices by unique id. That order is not
    "built-in first", so a USB camera can be index 0 and the Mac camera
    index 1. Returns an empty list when ``swift`` cannot list devices.
    """
    try:
        completed = subprocess.run(
            ["swift", "-e", _AVFOUNDATION_LIST],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_avfoundation_listing(completed.stdout)


def parse_avfoundation_listing(text: str) -> list[CameraDevice]:
    """Parse ``index<tab>name<tab>deviceType`` lines from the Swift lister."""
    devices: list[CameraDevice] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        index_text, name, device_type = parts
        try:
            index = int(index_text)
        except ValueError:
            continue
        devices.append(
            CameraDevice(
                index=index,
                name=name,
                path=None,
                usb="External" in device_type,
            )
        )
    return devices


def probe_indices(opener: Opener, count: int) -> list[CameraDevice]:
    """Open indices ``0 .. count-1`` and keep those that return a frame.

    Index 0 is marked built-in. Any higher index that returns a frame is marked
    USB, which matches how macOS orders the built-in camera and a USB webcam.
    """
    devices: list[CameraDevice] = []
    for index in range(count):
        cap = opener(index)
        try:
            if cap is None or not cap.isOpened():
                continue
            if not _read_frame(cap):
                continue
            devices.append(
                CameraDevice(
                    index=index,
                    name=f"camera {index}",
                    path=None,
                    usb=index != 0,
                )
            )
        finally:
            if cap is not None:
                cap.release()
    return devices


def choose_device(devices: list[CameraDevice], config: CameraConfig) -> CameraDevice:
    """Pick the device to open.

    ``preferred_index`` wins. Otherwise a ``preferred_name`` substring wins.
    Otherwise, when ``prefer_usb`` is set and a USB device is present, built-in
    cameras are skipped. The lowest index in the remaining set is used.
    """
    if not devices:
        raise CameraError("no cameras found")

    if config.preferred_index is not None:
        for device in devices:
            if device.index == config.preferred_index:
                return device
        found = ", ".join(str(d.index) for d in devices)
        raise CameraError(f"camera index {config.preferred_index} not found (have {found})")

    pool = list(devices)
    if config.preferred_name:
        needle = config.preferred_name.lower()
        pool = [d for d in pool if needle in d.name.lower()]
        if not pool:
            raise CameraError(f"no camera name contains {config.preferred_name!r}")
    elif config.prefer_usb and any(d.usb for d in pool):
        pool = [d for d in pool if d.usb]

    return min(pool, key=lambda d: d.index)


def format_camera_list(devices: list[CameraDevice]) -> str:
    if not devices:
        return "no cameras found"
    return "\n".join(device.description for device in devices)


def backend_flag(config: CameraConfig, system: str) -> int:
    name = config.backend
    if name == "auto":
        name = {"Linux": "v4l2", "Darwin": "avfoundation"}.get(system, "any")
    return {
        "v4l2": cv2.CAP_V4L2,
        "avfoundation": cv2.CAP_AVFOUNDATION,
        "any": cv2.CAP_ANY,
    }[name]


def open_camera(config: CameraConfig) -> Camera:
    """Open the best matching camera and verify that a frame arrives."""

    def open_fn() -> tuple[cv2.VideoCapture, CameraDevice]:
        return open_capture(config)

    cap, device = open_fn()
    return Camera(cap, device, config, open_fn)


def open_capture(
    config: CameraConfig,
    *,
    system: str | None = None,
    opener: Opener | None = None,
    lister: Lister | None = None,
) -> tuple[cv2.VideoCapture, CameraDevice]:
    """Open a capture device. ``opener`` and ``lister`` exist so tests inject fakes."""
    system = system if system is not None else platform.system()
    backend = backend_flag(config, system)
    if opener is None:

        def opener(index: int, backend: int = backend) -> cv2.VideoCapture:
            return cv2.VideoCapture(index, backend)

    if config.preferred_index is not None and lister is None:
        device = CameraDevice(
            index=config.preferred_index,
            name=f"camera {config.preferred_index}",
            path=f"/dev/video{config.preferred_index}" if system == "Linux" else None,
            usb=False,
        )
    else:
        devices = lister() if lister is not None else list_cameras(config, system=system)
        device = choose_device(devices, config)

    cap = opener(device.index)
    if cap is None or not _configure_capture(cap, config, mjpg=backend == cv2.CAP_V4L2):
        if cap is not None:
            cap.release()
        raise CameraError(f"camera {device.index} ({device.name}) did not return a frame")
    return cap, device


class Camera:
    """A capture device that re-discovers itself after repeated read failures."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        device: CameraDevice,
        config: CameraConfig,
        open_fn: Callable[[], tuple[cv2.VideoCapture, CameraDevice]],
    ) -> None:
        self._cap = cap
        self.device = device
        self.config = config
        self._open_fn = open_fn
        self._failures = 0

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or ``None`` when the camera is down."""
        return self._read(allow_reconnect=True)

    def _read(self, *, allow_reconnect: bool) -> np.ndarray | None:
        if self._cap is None:
            failed = True
            frame = None
        else:
            ok, frame = self._cap.read()
            failed = not ok or frame is None or getattr(frame, "size", 0) == 0
        if not failed:
            self._failures = 0
            return frame

        self._failures += 1
        if allow_reconnect and self._failures >= self.config.reconnect_failures:
            self._failures = 0
            if self.reconnect():
                return self._read(allow_reconnect=False)
        return None

    def reconnect(self) -> bool:
        """Close the current device and run discovery again."""
        self.release()
        try:
            self._cap, self.device = self._open_fn()
        except CameraError:
            self._cap = None
            return False
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def read_v4l2_device_caps(device: Path) -> int | None:
    """Return V4L2 capability bits for ``device``, or ``None`` if it isn't capture."""
    import fcntl  # Linux only; a top-level import breaks Windows.

    if not device.exists():
        return None
    caps = _V4L2Capability()
    request = (2 << 30) | (ctypes.sizeof(caps) << 16) | (ord("V") << 8) | 0
    try:
        fd = os.open(str(device), os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        fcntl.ioctl(fd, request, caps)
    except OSError:
        return None
    finally:
        os.close(fd)
    return int(caps.device_caps or caps.capabilities)


class _V4L2Capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_char * 16),
        ("card", ctypes.c_char * 32),
        ("bus_info", ctypes.c_char * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


def _configure_capture(cap: cv2.VideoCapture, config: CameraConfig, *, mjpg: bool) -> bool:
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(config.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(config.height))
    cap.set(cv2.CAP_PROP_FPS, float(config.fps))
    return _read_frame(cap)


def _read_frame(cap: cv2.VideoCapture) -> bool:
    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return True
    return False


def _is_capture(caps: int) -> bool:
    return bool(caps & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE))


def _video_index(name: str) -> int:
    if not name.startswith("video"):
        raise ValueError(name)
    return int(name.removeprefix("video"))


def _sysfs_is_usb(device_link: Path) -> bool:
    """True when the device's real sysfs path sits on a USB bus.

    A path component such as ``usb1`` marks the bus. Matching the substring
    ``usb`` anywhere would also match unrelated directories that merely contain
    those letters.
    """
    if not device_link.exists():
        return False
    return any(part.lower().startswith("usb") for part in Path(os.path.realpath(device_link)).parts)
