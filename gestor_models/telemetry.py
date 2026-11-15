"""Real-time RawTree request stream, model KPI aggregation, timeline & live logs."""
from __future__ import annotations

from collections import defaultdict
import threading
import time
from typing import Any

import requests

from .config import (
    HTTP_QUERY_TIMEOUT_S,
    TELEMETRY_CACHE_TTL_S,
    TIMELINE_MINUTES,
    load_env_credentials,
    safe_float,
)
from .process_controller import RunningProcess, tail_lines
from .registry import ModelSpec

_CACHE_LOCK = threading.Lock()
_CACHE_DATA: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}


def _parse_items(raw_items: Any) -> list[str]:
    if isinstance(raw_items, list):
        return [str(x) for x in raw_items[:10]]
    if isinstance(raw_items, tuple):
        return [str(x) for x in raw_items[:10]]
    if isinstance(raw_items, str) and raw_items.strip():
        cleaned = raw_items.strip().strip("[]")
        return [part.strip().strip("'\"") for part in cleaned.split(",") if part.strip()][:10]
    return []


def _query_rawtree_sql(api_url: str, headers: dict[str, str], sql: str) -> list[dict[str, Any]]:
    resp = requests.post(
        f"{api_url}/v1/query",
        headers=headers,
        json={"sql": sql},
        timeout=HTTP_QUERY_TIMEOUT_S,
    )
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:180]}")
    data = resp.json().get("data", [])
    return data if isinstance(data, list) else []


def _summarize_feedback(rows: list[dict[str, Any]], window_minutes: int) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "on_time": 0, "clicks": 0, "purchases": 0, "rev": 0.0, "lats": []}
    )
    for row in rows:
        mv = str(row.get("model_version") or "fallback").strip() or "fallback"
        b = buckets[mv]
        b["n"] += 1
        lat = safe_float(row.get("latency_ms"), -1.0)
        if lat >= 0.0:
            b["lats"].append(lat)
        served_by = str(row.get("served_by") or "").strip().lower()
        if served_by == "team" or (not served_by and mv != "fallback" and 0.0 <= lat <= 3000.0):
            b["on_time"] += 1
        if str(row.get("clicked") or "").strip():
            b["clicks"] += 1
        if str(row.get("purchased") or "").strip():
            b["purchases"] += 1
        b["rev"] += max(safe_float(row.get("revenue_eur"), 0.0), 0.0)

    total_all = sum(b["n"] for b in buckets.values()) or 1
    out: list[dict[str, Any]] = []
    for mv, b in buckets.items():
        n = b["n"]
        lats = sorted(b["lats"])
        p50 = lats[int(0.50 * (len(lats) - 1))] if lats else 0.0
        p95 = lats[int(0.95 * (len(lats) - 1))] if lats else 0.0
        out.append(
            {
                "model_version": mv,
                "reqs": n,
                "traffic_pct": round(100.0 * n / total_all, 1),
                "req_per_min": round(n / max(window_minutes, 1), 1),
                "on_time_pct": round(100.0 * b["on_time"] / max(n, 1), 2),
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "clicks": b["clicks"],
                "purchases": b["purchases"],
                "ctr_pct": round(100.0 * b["clicks"] / max(n, 1), 2),
                "cvr_pct": round(100.0 * b["purchases"] / max(n, 1), 2),
                "revenue_eur": round(b["rev"], 2),
                "eur_per_req": round(b["rev"] / max(n, 1), 4),
            }
        )
    out.sort(key=lambda r: r["reqs"], reverse=True)
    return out


def fetch_rawtree_telemetry(window_minutes: int = 3, req_limit: int = 40) -> dict[str, Any]:
    """Fetch recent requests, per-model KPI summary, and 15-min timeline with 1s TTL cache."""
    key = (int(window_minutes), int(req_limit))
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE_DATA.get(key)
        if cached and (now - cached[0]) < TELEMETRY_CACHE_TTL_S:
            return cached[1]

    api_url, team_id, api_key = load_env_credentials()
    if not team_id or not api_key:
        return {
            "error": "Falten TEAM_ID o RAWTREE_API_KEY a .env / group_API.env",
            "fetched_at": now,
            "window_minutes": window_minutes,
            "models_summary": [],
            "recent_requests": [],
            "timeline": [],
            "totals": {"reqs": 0, "on_time_pct": 0.0, "revenue_eur": 0.0, "p50_ms": 0.0},
        }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    sql_window = f"""
SELECT
    coalesce(nullIf(s.model_version::String, ''), 'fallback') AS model_version,
    f.served_by::String AS served_by,
    f.latency_ms::Float64 AS latency_ms,
    f.clicked::String AS clicked,
    f.purchased::String AS purchased,
    f.revenue_eur::Float64 AS revenue_eur
FROM {team_id}.feedback AS f
LEFT JOIN {team_id}.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
  AND f.ts::DateTime64(3) >= now64(3) - INTERVAL {int(window_minutes)} MINUTE
ORDER BY f.ts DESC
LIMIT 5000
"""
    sql_recent = f"""
SELECT
    toString(f.ts) AS ts,
    f.request_id::String AS request_id,
    coalesce(nullIf(s.model_version::String, ''), 'fallback') AS model_version,
    f.served_by::String AS served_by,
    round(f.latency_ms::Float64, 1) AS latency_ms,
    f.clicked::String AS clicked,
    f.purchased::String AS purchased,
    round(f.revenue_eur::Float64, 2) AS revenue_eur,
    s.items AS items
FROM {team_id}.feedback AS f
LEFT JOIN {team_id}.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
ORDER BY f.ts DESC
LIMIT {int(req_limit)}
"""
    sql_timeline = f"""
SELECT
    toString(toStartOfMinute(f.ts::DateTime64(3))) AS minute_ts,
    coalesce(nullIf(s.model_version::String, ''), 'fallback') AS model_version,
    count() AS reqs,
    round(avgIf(1.0, lowerUTF8(f.served_by::String) = 'team') * 100.0, 1) AS on_time_pct,
    round(avg(f.latency_ms::Float64), 0) AS avg_ms,
    round(sum(f.revenue_eur::Float64), 2) AS revenue_eur
FROM {team_id}.feedback AS f
LEFT JOIN {team_id}.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
  AND f.ts::DateTime64(3) >= now64(3) - INTERVAL {int(TIMELINE_MINUTES)} MINUTE
GROUP BY minute_ts, model_version
ORDER BY minute_ts DESC, reqs DESC
LIMIT 90
"""

    err: str | None = None
    window_rows: list[dict[str, Any]] = []
    recent_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    try:
        window_rows = _query_rawtree_sql(api_url, headers, sql_window)
        recent_rows = _query_rawtree_sql(api_url, headers, sql_recent)
        timeline_rows = _query_rawtree_sql(api_url, headers, sql_timeline)
    except Exception as exc:
        err = f"Error consultant RawTree: {exc}"

    models_summary = _summarize_feedback(window_rows, window_minutes=window_minutes)
    recent_requests: list[dict[str, Any]] = []
    for r in recent_rows:
        mv = str(r.get("model_version") or "fallback").strip() or "fallback"
        served_by = str(r.get("served_by") or "").strip().lower() or "unknown"
        lat = safe_float(r.get("latency_ms"), 0.0)
        ts_raw = str(r.get("ts") or "")
        hhmmss = ts_raw[11:19] if len(ts_raw) >= 19 else ts_raw
        recent_requests.append(
            {
                "ts": ts_raw,
                "time": hhmmss,
                "request_id": str(r.get("request_id") or ""),
                "model_version": mv,
                "served_by": served_by,
                "on_time": (served_by == "team" and mv != "fallback"),
                "latency_ms": round(lat, 1),
                "clicked": str(r.get("clicked") or ""),
                "purchased": str(r.get("purchased") or ""),
                "revenue_eur": round(safe_float(r.get("revenue_eur"), 0.0), 2),
                "items": _parse_items(r.get("items")),
            }
        )

    timeline_out: list[dict[str, Any]] = []
    for row in timeline_rows:
        m_ts = str(row.get("minute_ts") or "")
        timeline_out.append(
            {
                "minute_ts": m_ts,
                "minute": m_ts[11:16] if len(m_ts) >= 16 else m_ts,
                "model_version": str(row.get("model_version") or "fallback").strip() or "fallback",
                "reqs": int(safe_float(row.get("reqs"), 0)),
                "on_time_pct": round(safe_float(row.get("on_time_pct"), 0.0), 1),
                "avg_ms": round(safe_float(row.get("avg_ms"), 0.0), 0),
                "revenue_eur": round(safe_float(row.get("revenue_eur"), 0.0), 2),
            }
        )

    all_lats = sorted(
        safe_float(r.get("latency_ms"), -1.0)
        for r in window_rows
        if safe_float(r.get("latency_ms"), -1.0) >= 0.0
    )
    total_reqs = len(window_rows)
    total_on_time = sum(1 for r in window_rows if str(r.get("served_by") or "").lower() == "team")
    total_rev = sum(max(safe_float(r.get("revenue_eur"), 0.0), 0.0) for r in window_rows)
    p50_all = all_lats[int(0.50 * (len(all_lats) - 1))] if all_lats else 0.0

    by_req: dict[str, set[str]] = defaultdict(set)
    for req_item in recent_requests:
        mv_item = str(req_item.get("model_version") or "")
        rid_item = str(req_item.get("request_id") or "")
        if rid_item and mv_item and mv_item != "fallback":
            by_req[rid_item].add(mv_item)
    colliding_rids = sorted(rid for rid, mvs in by_req.items() if len(mvs) > 1)
    colliding_versions = sorted({mv for rid in colliding_rids for mv in by_req[rid]})

    result = {
        "error": err,
        "fetched_at": now,
        "window_minutes": window_minutes,
        "models_summary": models_summary,
        "recent_requests": recent_requests,
        "timeline": timeline_out,
        "collision_request_ids": colliding_rids,
        "colliding_versions": colliding_versions,
        "totals": {
            "reqs": total_reqs,
            "req_per_min": round(total_reqs / max(window_minutes, 1), 1),
            "on_time_pct": round(100.0 * total_on_time / max(total_reqs, 1), 2) if total_reqs else 0.0,
            "revenue_eur": round(total_rev, 2),
            "p50_ms": round(p50_all, 1),
        },
    }
    if err is None:
        with _CACHE_LOCK:
            _CACHE_DATA[key] = (now, result)
    return result


def read_model_logs(
    registry: dict[str, ModelSpec],
    procs: list[RunningProcess],
    log_model: str | None = None,
    n_lines: int = 50,
) -> dict[str, Any]:
    """Read live.log tail for requested model (or currently running model)."""
    folder = (log_model or "").strip()
    if folder not in registry:
        serves = [p for p in procs if p.role == "serve" and p.folder in registry]
        if serves:
            folder = serves[0].folder
        elif "calypso_3_1" in registry:
            folder = "calypso_3_1"
        else:
            folder = next(iter(registry.keys()), "")

    spec = registry.get(folder)
    if spec is None:
        return {"folder": "", "path": "", "lines": []}
    return {
        "folder": spec.folder,
        "path": str(spec.log_path),
        "lines": tail_lines(spec.log_path, n=n_lines),
    }
