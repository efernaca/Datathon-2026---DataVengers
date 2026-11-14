"""RawTree local con ClickHouse embebido (chdb) para probar el SQL y el bucle completo.

Las tablas usan columnas `Dynamic` (como RawTree) y `context` como `JSON`. Expone la
misma interfaz que `RawTreeClient` (`query`, `query_safe`, `query_paged`, `push_table`,
`push_submissions`, `push_scratch`). Solo para tests: `pip install chdb`.
"""
from __future__ import annotations

import json
import tempfile
import threading
from typing import Any

SHOP_TABLES = {
    "catalog": ["product_id", "category", "subcategory", "brand", "price_eur", "description",
                "created_at", "is_active"],
    "history_sessions": ["request_id", "ts", "user_id", "context", "recent_views", "shown",
                         "propensity", "clicked", "purchased", "revenue_eur"],
    "history_events": ["ts", "event_type", "user_id", "session_id", "product_id", "price_eur",
                       "device", "region"],
    "requests": ["request_id", "ts", "user_id", "is_new_user", "context", "recent_views"],
    "events": ["ts", "event_type", "user_id", "session_id", "product_id", "price_eur", "device", "region"],
    "catalog_updates": ["ts", "change", "product_id", "category", "subcategory", "brand", "price_eur",
                        "description", "created_at", "is_active"],
}
TEAM_TABLES = {
    "submissions": ["request_id", "items", "model_version"],
    "feedback": ["request_id", "ts", "shown", "served_by", "fallback_reason", "latency_ms",
                 "clicked", "purchased", "revenue_eur"],
    "scratch": ["kind", "model_version", "src", "dst", "w"],
}


class LocalRawTree:
    def __init__(self, team_id: str = "team_00", path: str | None = None) -> None:
        from chdb import session

        self.team_id = team_id
        self.dir = path or tempfile.mkdtemp(prefix="chdb_")
        self.s = session.Session(self.dir)
        self._lock = threading.Lock()
        self.n_queries = 0
        self._q("SET session_timezone = 'UTC'")
        self._q("CREATE DATABASE IF NOT EXISTS shop")
        self._q(f"CREATE DATABASE IF NOT EXISTS {team_id}")
        for db, tables in (("shop", SHOP_TABLES), (team_id, TEAM_TABLES)):
            for t, cols in tables.items():
                defs = ", ".join(f"{c} {'JSON' if c == 'context' else 'Dynamic'}" for c in cols)
                self._q(f"CREATE TABLE IF NOT EXISTS {db}.{t} ({defs}, _ins DateTime64(3) DEFAULT now64(3)) "
                        f"ENGINE = MergeTree ORDER BY _ins")

    def _q(self, sql: str, fmt: str = "CSV"):
        with self._lock:
            self.n_queries += 1
            return self.s.query(sql, fmt)

    def insert(self, db: str, table: str, rows: list[dict[str, Any]], chunk: int = 5000) -> None:
        cols = (SHOP_TABLES if db == "shop" else TEAM_TABLES)[table]
        for i in range(0, len(rows), chunk):
            body = "\n".join(json.dumps({c: r.get(c) for c in cols}, default=str) for r in rows[i:i + chunk])
            self._q(f"INSERT INTO {db}.{table} ({', '.join(cols)}) FORMAT JSONEachRow {body}")

    # ---- interfaz de RawTreeClient
    def query(self, sql: str, timeout: float = 10.0) -> list[dict[str, Any]]:
        res = self._q(sql.strip().rstrip(";") + " FORMAT JSON", fmt="JSON")
        data = res.bytes() if hasattr(res, "bytes") else bytes(res)
        if not data:
            return []
        return json.loads(data).get("data", [])

    def query_safe(self, sql: str, timeout: float = 10.0):
        try:
            return self.query(sql, timeout)
        except Exception:  # noqa: BLE001
            return None

    def query_paged(self, tmpl: str, n_pages: int, timeout: float = 300.0, log_every: int = 1):
        out = []
        for i in range(n_pages):
            out.extend(self.query(tmpl.format(page=i, n_pages=n_pages), timeout))
        return out

    def push_table(self, table: str, records: list[dict[str, Any]], timeout: float = 1.5) -> None:
        self.insert(self.team_id, table, records)

    def push_submissions(self, answers, timeout: float = 1.2) -> None:
        self.push_table("submissions", answers)

    def push_scratch(self, records, timeout: float = 10.0) -> None:
        self.push_table("scratch", records)
