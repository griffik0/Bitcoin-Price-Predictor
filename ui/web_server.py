"""Lightweight HTTP server for the web dashboard (stdlib only)."""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.web_state import WebDashboardState

logger = logging.getLogger("btc_bot")

WEB_ROOT = Path(__file__).resolve().parent / "web"


class _DashboardHandler(BaseHTTPRequestHandler):
    state: WebDashboardState | None = None

    def log_message(self, format: str, *args) -> None:
        logger.debug("Web %s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == "/api/state":
            self._send_json(self.state.snapshot() if self.state else {})
            return
        if path in ("", "/"):
            path = "/index.html"
        file_path = self._safe_file(path)
        if file_path is None:
            self.send_error(404)
            return
        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _safe_file(url_path: str) -> Path | None:
        rel = url_path.lstrip("/") or "index.html"
        candidate = (WEB_ROOT / rel).resolve()
        try:
            candidate.relative_to(WEB_ROOT.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class WebDashboardServer:
    def __init__(self, state: WebDashboardState, port: int = 8765) -> None:
        self.state = state
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        handler = type(
            "BoundDashboardHandler",
            (_DashboardHandler,),
            {"state": self.state},
        )
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="web-dashboard",
            daemon=True,
        )
        self._thread.start()
        logger.info("Web dashboard at %s", self.url)

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd = None
