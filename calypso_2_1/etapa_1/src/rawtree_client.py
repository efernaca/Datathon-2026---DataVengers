"""Low-latency HTTP client for RawTree SQL queries and batch JSON ingestion."""
from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)


class RawTreeError(RuntimeError):
    """Raised when RawTree API returns an unexpected error."""


class RawTreeClient:
    """Persistent HTTP client with connection pooling and fast-fail guarantees."""

    def __init__(self, api_url: str, api_key: str, team_id: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.team_id = team_id
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def query(self, sql: str, timeout: float = 10.0) -> list[dict[str, Any]]:
        """Execute a SQL query against RawTree and return rows as dictionaries."""
        resp = self.session.post(
            f"{self.api_url}/v1/query",
            json={"sql": sql},
            timeout=timeout,
        )
        if not resp.ok:
            raise RawTreeError(f"Query failed ({resp.status_code}): {resp.text[:300]}")
        payload = resp.json()
        return payload.get("data", [])

    def query_safe(self, sql: str, timeout: float = 10.0) -> list[dict[str, Any]] | None:
        """Run a SQL query returning None on expected empty-stream errors instead of raising."""
        try:
            return self.query(sql, timeout=timeout)
        except Exception as exc:
            msg = str(exc)
            if "Unknown expression identifier" in msg or "UNKNOWN_IDENTIFIER" in msg:
                logger.debug("Empty/uninitialized table ignored during query: %s", msg[:120])
                return []
            logger.warning("RawTree query_safe error: %s", msg[:200])
            return None

    def push_table(
        self,
        table: str,
        records: list[dict[str, Any]],
        timeout: float = 1.5,
        retries: int = 1,
    ) -> None:
        """Batch-insert JSON records into a team_NN table."""
        if not records:
            return
        if not self.team_id:
            raise RawTreeError("TEAM_ID is required before inserting records.")

        url = f"{self.api_url}/v1/tables/{table}"
        params = {"database": self.team_id}
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = self.session.post(url, params=params, json=records, timeout=timeout)
                if resp.ok:
                    return
                raise RawTreeError(f"Insert into {table} failed ({resp.status_code}): {resp.text[:300]}")
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(0.05)
        assert last_exc is not None
        raise last_exc

    def push_submissions(self, answers: list[dict[str, Any]], timeout: float = 1.4) -> None:
        """Insert recommendation answers into team_NN.submissions."""
        self.push_table("submissions", answers, timeout=timeout, retries=1)

    def push_scratch(self, records: list[dict[str, Any]], timeout: float = 5.0) -> None:
        """Insert features, embeddings, or diagnostics into team_NN.scratch."""
        self.push_table("scratch", records, timeout=timeout, retries=1)
