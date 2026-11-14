"""Central configuration, filesystem paths, and shared helpers for gestor_models.

Edita aquest fitxer si vols canviar:
- El port del servidor web (`DEFAULT_PORT`)
- La finestra temporal d'estadistiques (`DEFAULT_WINDOW_MINUTES`)
- El nombre de sol.licituds recents visibles (`RECENT_REQUESTS_LIMIT`)
- L'ordre o nom de les carpetes historiques (`LEGACY_STAGE_ORDER`)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

GESTOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = GESTOR_DIR.parent
STATE_DIR = GESTOR_DIR / "state"
STATIC_DIR = GESTOR_DIR / "static"

STATE_FILE = REPO_ROOT / ".active_model.json"
OPERATION_STATUS_FILE = STATE_DIR / "operation_status.json"
TRANSITIONS_FILE = STATE_DIR / "model_transitions.json"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_WINDOW_MINUTES = 3
RECENT_REQUESTS_LIMIT = 40
TIMELINE_MINUTES = 15
TELEMETRY_CACHE_TTL_S = 1.0
HTTP_QUERY_TIMEOUT_S = 5.0

LEGACY_STAGE_ORDER: tuple[tuple[str, str, str], ...] = (
    ("etapa_4", "Legacy Stage 4", "calypso-2.1"),
    ("etapa_3", "Legacy Stage 3", "calypso-2.1"),
    ("etapa_2_version_especial", "Legacy Stage 2 Especial", "e2esp"),
    ("etapa_2", "Legacy Stage 2", "calypso-2.1"),
    ("etapa_1_version_especial", "Legacy Stage 1 Especial", "etapa1-especial"),
    ("etapa_1", "Legacy Stage 1", "etapa1-hybrid-v1"),
)


def configure_stdio() -> None:
    """Configure UTF-8 output safely on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except OSError:
                pass


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_json_file(path: Path | None) -> Any | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_env_credentials() -> tuple[str, str, str]:
    """Load RAWTREE_API_URL, TEAM_ID, RAWTREE_API_KEY from env or project env files."""
    search_dirs = [REPO_ROOT, REPO_ROOT / "calypso_3_1", REPO_ROOT / "datathon-fme-2026-main"]
    for directory in search_dirs:
        for fname in (".env", "group_API.env"):
            env_path = directory / fname
            if not env_path.is_file():
                continue
            try:
                lines = env_path.read_text(encoding="utf-8-sig").splitlines()
            except OSError:
                continue
            for raw in lines:
                line = raw.strip().replace("\r", "")
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip().removeprefix("export ").strip()
                if key:
                    os.environ.setdefault(key, val.strip().strip("'\""))

    api_url = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com").strip().rstrip("/")
    team_id = os.environ.get("TEAM_ID", "").strip()
    api_key = os.environ.get("RAWTREE_API_KEY", "").strip()
    return api_url, team_id, api_key
