"""Localhost HTTP receiver + main-thread import dispatch (bpy timers).

The receiver only ENQUEUES payloads from its worker thread; the timer
(`receiver_tick`) runs in Blender's main thread and keeps calling the import
callback installed at start(), so bpy is never touched off-thread.
Accepted payloads: {"path": "<msflow.json>"} or a raw msflow-motion document.
"""
from __future__ import annotations

import json
import queue
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from . import io_msflow


class Receiver:
    def __init__(self) -> None:
        self.queue: queue.Queue[Any] = queue.Queue()
        self.callback: Callable[[str], None] | None = None
        self.running = False
        self.error = ""
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(
        self, port: int, bind_all: bool, callback: Callable[[str], None]
    ) -> str:
        if self.running:
            self.stop()
        self.callback = callback
        address = ("0.0.0.0" if bind_all else "127.0.0.1", int(port))
        handler = _make_handler(self)
        try:
            self._server = ThreadingHTTPServer(address, handler)
            self._server.daemon_threads = True
        except OSError as exc:
            self.error = f"cannot listen on port {port}: {exc}"
            return self.error
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="msflow-receiver"
        )
        self.running = True
        self._thread.start()
        return ""

    def stop(self) -> None:
        self.running = False
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except OSError:
                pass
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def enqueue(self, payload: Any) -> None:
        self.queue.put(payload)


def _make_handler(receiver: Receiver) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/import"):
                self._respond(404, "unknown route")
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                self._respond(400, f"bad JSON body: {exc}")
                return
            if not isinstance(payload, dict):
                self._respond(400, "payload must be a JSON object")
                return
            if "path" not in payload and "format" not in payload:
                self._respond(400, "need 'path' or a msflow-motion document")
                return
            receiver.enqueue(payload)
            self._respond(200, "queued")

        def _respond(self, code: int, result: str) -> None:
            key = "result" if code == 200 else "error"
            body = json.dumps({"ok": code == 200, key: result}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            pass  # keep the Blender console clean

    return Handler


def payload_to_file(payload: dict[str, Any]) -> str:
    """Materialize a payload as a .msflow.json file path (documents are
    written to the OS temp dir so all imports flow through one path API)."""
    if "path" in payload:
        return str(payload["path"])
    motion = io_msflow.parse_motion_dict(payload, source="<push>")
    label = io_msflow.default_action_label(motion.meta)
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", label)
    destination = Path(tempfile.gettempdir()) / f"{safe}.msflow.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return str(destination)


def receiver_tick() -> float:
    """Main-thread poller: import queued pushes (a few per tick)."""
    drained = 0
    while drained < 4:
        try:
            payload = RECEIVER.queue.get_nowait()
        except queue.Empty:
            break
        drained += 1
        try:
            path = payload_to_file(payload)
        except Exception as exc:
            path = ""
            _report(f"⛔ push rejected: {exc}")
        if path and RECEIVER.callback is not None:
            RECEIVER.callback(path)
    return 0.2


def _report(message: str) -> None:
    try:
        import bpy

        config = getattr(bpy.context.scene, "msflow_bridge_settings", None)
        if config is not None:
            config.status = message
    except Exception:
        print(f"[msflow] {message}", flush=True)


RECEIVER = Receiver()
