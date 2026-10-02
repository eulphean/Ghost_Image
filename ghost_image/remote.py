"""A phone page on the local network that holds or releases the background.

The OpenCV window only hears the keyboard. This server lets a phone on the
same network send the same hold and release, after a PIN. It is off unless
``remote.enabled`` is true.
"""

from __future__ import annotations

import hmac
import html
import queue
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol

from ghost_image.config import RemoteConfig
from ghost_image.log import LOGGER, report_error

_MAX_FAILURES = 5
_LOCKOUT_S = 15.0
_BODY_LIMIT = 1024
_HOLD_WAIT_S = 2.0

_MESSAGES = {
    "held": "Held. This picture is the new background.",
    "released": "Released. The window is showing the live camera.",
    "empty": "The camera has no picture yet. Try again.",
    "slow": "The piece did not answer. Try again.",
    "locked": "Too many tries. Wait a moment.",
    "bad": "Wrong PIN.",
}

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ghost Image</title>
<style>
  body {
    margin: 0; min-height: 100vh; display: flex;
    align-items: center; justify-content: center;
    background: #111; color: #f2f2f2; font-family: system-ui, sans-serif;
  }
  form { width: min(24rem, 92vw); display: flex; flex-direction: column; gap: 1rem; }
  h1 { font-size: 1.6rem; font-weight: 600; margin: 0; }
  p { margin: 0; line-height: 1.4; }
  input, button {
    font: inherit; font-size: 1.4rem; padding: 1rem; border: 0; border-radius: 0.6rem;
  }
  input { background: #222; color: #f2f2f2; }
  button { background: #f2f2f2; color: #111; font-weight: 650; }
  button.release { background: transparent; color: #f2f2f2; border: 1px solid #555; }
  .note { min-height: 1.4em; color: #d0d0d0; }
</style>
</head>
<body>
<form method="post" action="/hold">
  <h1>Ghost Image</h1>
  <p>Clear the room, then hold a new background.</p>
  <input name="pin" type="password" autocomplete="current-password"
         placeholder="PIN" autofocus required>
  <button type="submit">Hold</button>
  <button type="submit" class="release" formaction="/release">Release</button>
  <p class="note">__MESSAGE__</p>
</form>
</body>
</html>
"""


@dataclass
class PhoneRequest:
    """One hold or release from the phone, completed by the camera loop."""

    action: str = "hold"
    done: threading.Event = field(default_factory=threading.Event)
    ok: bool = False


def pin_matches(given: str, expected: str) -> bool:
    """True when the submitted PIN equals the configured one."""
    given_b = given.encode()
    expected_b = expected.encode()
    if len(given_b) != len(expected_b):
        return False
    return hmac.compare_digest(given_b, expected_b)


def local_ipv4_addresses() -> list[str]:
    """LAN addresses a phone can use to open the page."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            found.append(sock.getsockname()[0])
    except OSError:
        pass
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)
    except socket.gaierror:
        infos = []
    for info in infos:
        ip = info[4][0]
        if ip not in found:
            found.append(ip)
    return [
        ip for ip in found if ip and not ip.startswith("127.") and not ip.startswith("169.254.")
    ]


def page_urls(host: str, port: int) -> list[str]:
    """Addresses to print when the page starts listening."""
    if host in ("0.0.0.0", "::", ""):
        ips = local_ipv4_addresses() or ["127.0.0.1"]
    else:
        ips = [host]
    return [f"http://{ip}:{port}" for ip in ips]


def page_html(message: str) -> str:
    return _PAGE.replace("__MESSAGE__", html.escape(message))


class PhoneServer(ThreadingHTTPServer):
    """Serves the hold page. ``daemon_threads`` so shutdown does not wait on a phone."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        pin: str,
        requests: queue.Queue[PhoneRequest],
    ) -> None:
        super().__init__(address, PhoneHandler)
        self.pin = pin
        self.requests: queue.Queue[PhoneRequest] = requests
        self.failures: dict[str, tuple[int, float]] = {}
        self.lock = threading.Lock()

    def locked(self, ip: str) -> bool:
        now = time.monotonic()
        with self.lock:
            _fails, locked_until = self.failures.get(ip, (0, 0.0))
            if locked_until and now >= locked_until:
                self.failures.pop(ip, None)
                return False
            return now < locked_until

    def note_failure(self, ip: str) -> None:
        now = time.monotonic()
        with self.lock:
            fails, locked_until = self.failures.get(ip, (0, 0.0))
            if locked_until and now >= locked_until:
                fails = 0
            fails += 1
            locked_until = now + _LOCKOUT_S if fails >= _MAX_FAILURES else 0.0
            self.failures[ip] = (fails, locked_until)

    def note_success(self, ip: str) -> None:
        with self.lock:
            self.failures.pop(ip, None)


class PhoneHandler(BaseHTTPRequestHandler):
    server: PhoneServer

    def log_message(self, format: str, *args: object) -> None:
        """Keep the gallery console free of one line per phone refresh."""
        return

    def do_GET(self) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        result = query.get("result", [""])[0]
        self._send_html(200, page_html(_MESSAGES.get(result, "")))

    def do_POST(self) -> None:
        action = urllib.parse.urlparse(self.path).path
        if action not in ("/hold", "/release"):
            self._send_html(404, page_html(""))
            return
        pin = self._read_pin()
        ip = self.client_address[0]
        if self.server.locked(ip):
            self._redirect("/?result=locked")
            return
        if not pin_matches(pin, self.server.pin):
            self.server.note_failure(ip)
            self._send_html(403, page_html(_MESSAGES["bad"]))
            return
        self.server.note_success(ip)
        request = PhoneRequest(action="release" if action == "/release" else "hold")
        self.server.requests.put(request)
        if not request.done.wait(_HOLD_WAIT_S):
            self._redirect("/?result=slow")
            return
        if request.action == "release":
            self._redirect("/?result=released")
            return
        self._redirect("/?result=held" if request.ok else "/?result=empty")

    def _read_pin(self) -> str:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return ""
        if length < 0 or length > _BODY_LIMIT:
            return ""
        body = self.rfile.read(length).decode("utf-8", "replace")
        values = urllib.parse.parse_qs(body, keep_blank_values=True).get("pin", [""])
        return values[0] if values else ""

    def _send_html(self, status: int, page: str) -> None:
        payload = page.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


class PhoneControls(Protocol):
    """What the camera loop needs from the phone page."""

    def poll(self) -> PhoneRequest | None: ...

    def close(self) -> None: ...


class RemoteControls:
    """Queues phone holds for the camera loop. Disabled configs do not bind a port."""

    def __init__(self, config: RemoteConfig) -> None:
        self.enabled = config.enabled
        self._requests: queue.Queue[PhoneRequest] = queue.Queue()
        self._httpd: PhoneServer | None = None
        self._thread: threading.Thread | None = None
        self.port = config.port
        if not config.enabled:
            return
        address = (config.host, config.port)
        try:
            self._httpd = PhoneServer(address, config.pin.strip(), self._requests)
        except OSError as exc:
            report_error(f"phone page could not listen on {config.host}:{config.port}: {exc}")
            return
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="ghost-phone",
            daemon=True,
        )
        self._thread.start()
        for url in page_urls(config.host, self.port):
            message = f"Phone reset: {url}"
            print(message)
            LOGGER.info(message)

    def poll(self) -> PhoneRequest | None:
        """Return a hold or release the phone is waiting on, or ``None``."""
        try:
            return self._requests.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._httpd = None
        self._thread = None
