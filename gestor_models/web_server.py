"""Lightweight HTTP server and REST API for the gestor_models Web UI."""
from __future__ import annotations

from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

from .config import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_WINDOW_MINUTES, RECENT_REQUESTS_LIMIT, STATIC_DIR
from .process_controller import (
    find_running_processes,
    get_runtime_overview,
    start_background_operation,
    stop_models,
)
from .registry import discover_models
from .telemetry import fetch_rawtree_telemetry, read_model_logs

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


class GestorRequestHandler(BaseHTTPRequestHandler):
    """Serve static assets and real-time REST endpoints for model control."""

    server_version = "GestorModels/2.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Silence per-poll access logs to keep terminal clean
        return

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", raw)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(min(length, 65536))
        try:
            obj = json.loads(raw.decode("utf-8"))
            return obj if isinstance(obj, dict) else {}
        except ValueError:
            return {}

    def _serve_static_file(self, filename: str) -> None:
        safe_name = Path(filename).name
        target = (STATIC_DIR / safe_name).resolve()
        if not target.is_file() or target.parent != STATIC_DIR.resolve():
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Recurs no trobat"})
            return
        ctype = _CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        self._send_bytes(HTTPStatus.OK, ctype, target.read_bytes())

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in {"/", "/index.html"}:
            self._serve_static_file("index.html")
            return
        if path.startswith("/static/"):
            self._serve_static_file(path[len("/static/") :])
            return
        if path == "/api/state":
            try:
                window = max(1, min(60, int(qs.get("window", [str(DEFAULT_WINDOW_MINUTES)])[0])))
            except ValueError:
                window = DEFAULT_WINDOW_MINUTES
            try:
                req_limit = max(10, min(100, int(qs.get("limit", [str(RECENT_REQUESTS_LIMIT)])[0])))
            except ValueError:
                req_limit = RECENT_REQUESTS_LIMIT
            log_model = (qs.get("log_model", [""])[0] or "").strip()

            registry = discover_models()
            overview = get_runtime_overview(registry)
            procs = find_running_processes(registry)
            telemetry = fetch_rawtree_telemetry(window_minutes=window, req_limit=req_limit)
            logs = read_model_logs(registry, procs, log_model=log_model or None, n_lines=45)

            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    **overview,
                    "telemetry": telemetry,
                    "logs": logs,
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Ruta desconeguda"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        body = self._read_json_body()
        registry = discover_models()

        if parsed.path in {"/api/train_activate", "/api/switch"}:
            model = str(body.get("model") or "").strip()
            if not model:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": "Falta el camp 'model'."})
                return
            train_first = parsed.path == "/api/train_activate"
            force = bool(body.get("force", False))
            ok, msg = start_background_operation(
                registry,
                model,
                train_first=train_first,
                force=force,
            )
            status = HTTPStatus.OK if ok else HTTPStatus.CONFLICT
            self._send_json(status, {"ok": ok, "message": msg})
            return

        if parsed.path == "/api/stop":
            model = str(body.get("model") or "").strip() or None
            ok, details = stop_models(registry, model_filter=model)
            self._send_json(HTTPStatus.OK, {"ok": ok, "message": " | ".join(details), "details": details})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Ruta POST desconeguda"})


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Instantiate the threaded HTTP dashboard server."""
    ThreadingHTTPServer.allow_reuse_address = True
    handler_cls = partial(GestorRequestHandler)
    return ThreadingHTTPServer((host, port), handler_cls)


def run_web_ui(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> int:
    """Start the Web UI server and block until Ctrl+C."""
    server = create_server(host=host, port=port)
    url = f"http://{host}:{port}"
    print(f"[GESTOR MODELS UI] Servidor web actiu a {url}")
    print("Prem Ctrl+C per aturar el dashboard (els models en viu seguiran funcionant).")
    if open_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[GESTOR MODELS UI] Dashboard tancat.")
    finally:
        server.server_close()
    return 0
