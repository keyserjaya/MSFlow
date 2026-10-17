"""Push exported motion files to the Blender bridge add-on over HTTP.

The add-on listens on localhost only; `push_motion` posts a JSON body naming
a shared file path (the repo lives on the Windows drive, so WSL2 can hand
Blender a Windows-visible path). Stdlib only.
"""
from __future__ import annotations

import json
import socket
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_PORT = 8765
DEFAULT_TIMEOUT = 3.0


def _gateway_from_route_table() -> str:
    """WSL2: the Windows host answers on the default-gateway IP of eth0."""
    try:
        for line in Path("/proc/net/route").read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) < 3 or fields[1] != "00000000":
                continue
            address = socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
            if address and address != "0.0.0.0":
                return address
    except Exception:
        pass
    return ""


def windows_visible(path: Path) -> str:
    """Translate WSL2's /mnt/<drive>/... form into a Windows path the
    Blender side can open; other forms pass through."""
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6 and text[5].isalpha() and text[6] == "/":
        return f"{text[5].upper()}:\\{text[7:].replace('/', chr(92))}"
    return text


def candidate_hosts(host_text: str) -> list[str]:
    """Ordered hostnames to try for one push ("auto" = loopback, then WSL2
    gateway — mirrored-mode WSL2 answers on loopback, NAT needs the gateway
    with the receiver bound to all interfaces)."""
    token = (host_text or "auto").strip().lower()
    if token not in ("", "auto"):
        return [token]
    hosts = ["127.0.0.1"]
    host = _gateway_from_route_table()
    if host and host not in hosts:
        hosts.append(host)
    return hosts


def resolve_host_label(host_text: str) -> str:
    """Readable summary of where pushes will try to connect."""
    hosts = candidate_hosts(host_text)
    token = (host_text or "auto").strip().lower()
    note = " (auto: loopback + WSL2 gateway)" if token in ("", "auto") else ""
    return " and ".join(hosts) + note


def push_motion(
    path: str | Path,
    *,
    port: int = DEFAULT_PORT,
    host: str = "auto",
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[bool, str]:
    """POST `{"path": ...}` to the add-on receiver. Returns (ok, message)."""
    file_path = Path(path)
    if not file_path.is_file():
        return False, f"file not found: {file_path}"
    body: dict[str, Any] = {"path": windows_visible(file_path.resolve())}
    errors: list[str] = []
    for candidate in candidate_hosts(host):
        url = f"http://{candidate}:{int(port)}/import"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            ok = bool(payload.get("ok"))
            if ok:
                return True, str(payload.get("result", "accepted"))
            errors.append(f"{candidate}: {payload.get('error', 'refused')}")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            errors.append(f"{candidate}: {exc}")
    return False, "; ".join(errors) if errors else "no host answered"
