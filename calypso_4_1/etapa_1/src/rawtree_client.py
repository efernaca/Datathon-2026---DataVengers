"""Cliente HTTP de RawTree: consultas SQL e inserciones por lotes."""
from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)


class RawTreeError(RuntimeError):
    pass


def sql_str(value: str) -> str:
    """Literal SQL seguro para valores que vienen de nuestros propios datos."""
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


class RawTreeClient:
    def __init__(self, api_url: str, api_key: str, team_id: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.team_id = team_id
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def query(self, sql: str, timeout: float = 10.0) -> list[dict[str, Any]]:
        resp = self.session.post(f"{self.api_url}/v1/query", json={"sql": sql}, timeout=timeout)
        if not resp.ok:
            raise RawTreeError(f"Query failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json().get("data", [])

    def query_safe(self, sql: str, timeout: float = 10.0) -> list[dict[str, Any]] | None:
        """[] si la tabla aún no existe (stream vacío); None si hay error real."""
        try:
            return self.query(sql, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Unknown expression identifier" in msg or "UNKNOWN_IDENTIFIER" in msg:
                return []
            logger.warning("query_safe: %s", msg[:200])
            return None

    def push_table(self, table: str, records: list[dict[str, Any]], timeout: float = 1.5) -> None:
        if not records:
            return
        if not self.team_id:
            raise RawTreeError("TEAM_ID es obligatorio para insertar.")
        resp = self.session.post(
            f"{self.api_url}/v1/tables/{table}",
            params={"database": self.team_id},
            json=records,
            timeout=timeout,
        )
        if not resp.ok:
            raise RawTreeError(f"Insert {table} failed ({resp.status_code}): {resp.text[:300]}")

    def push_submissions(self, answers: list[dict[str, Any]], timeout: float = 1.2) -> None:
        self.push_table("submissions", answers, timeout=timeout)

    def push_scratch(self, records: list[dict[str, Any]], timeout: float = 10.0) -> None:
        self.push_table("scratch", records, timeout=timeout)

    def query_paged(self, sql_template: str, n_pages: int, timeout: float = 300.0,
                    log_every: int = 1) -> list[dict[str, Any]]:
        """Ejecuta `sql_template.format(page=i, n_pages=n)` y concatena (descargas grandes)."""
        out: list[dict[str, Any]] = []
        for i in range(n_pages):
            t0 = time.perf_counter()
            rows = self.query(sql_template.format(page=i, n_pages=n_pages), timeout=timeout)
            out.extend(rows)
            if (i + 1) % log_every == 0:
                logger.info("  página %d/%d: %d filas (%.1fs)", i + 1, n_pages, len(rows),
                            time.perf_counter() - t0)
        return out
