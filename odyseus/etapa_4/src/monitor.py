"""Live RawTree feedback monitor tracking ON TIME %, latency p50/p95, CTR, CVR, and revenue."""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
import time
from typing import Any

from .config import ConfigV4
from .rawtree_client import RawTreeClient


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def summarize_feedback_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate RawTree feedback/submission rows by model_version cleanly."""
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_reqs": 0,
            "team_on_time": 0,
            "clicks": 0,
            "purchases": 0,
            "revenue_eur": 0.0,
            "latencies": [],
        }
    )

    for row in rows:
        mv = str(row.get("model_version") or "fallback_or_unknown").strip() or "fallback_or_unknown"
        b = buckets[mv]
        b["total_reqs"] += 1

        served_by = str(row.get("served_by") or "").strip().lower()
        lat_raw = row.get("latency_ms")
        lat_val = _safe_float(lat_raw, -1.0) if lat_raw is not None else -1.0
        if lat_val >= 0.0:
            b["latencies"].append(lat_val)

        if served_by == "team" or (not served_by and mv != "fallback_or_unknown" and lat_val <= 3000.0):
            b["team_on_time"] += 1

        if str(row.get("clicked") or "").strip():
            b["clicks"] += 1
        if str(row.get("purchased") or "").strip():
            b["purchases"] += 1
        b["revenue_eur"] += max(_safe_float(row.get("revenue_eur"), 0.0), 0.0)

    summary: list[dict[str, Any]] = []
    for mv, b in buckets.items():
        n = b["total_reqs"]
        lats = sorted(b["latencies"])
        if lats:
            p50 = lats[int(0.50 * (len(lats) - 1))]
            p95 = lats[int(0.95 * (len(lats) - 1))]
        else:
            p50 = 0.0
            p95 = 0.0
        summary.append(
            {
                "model_version": mv,
                "total_reqs": n,
                "team_on_time": b["team_on_time"],
                "on_time_pct": round(100.0 * b["team_on_time"] / max(n, 1), 2),
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "ctr_pct": round(100.0 * b["clicks"] / max(n, 1), 3),
                "cvr_pct": round(100.0 * b["purchases"] / max(n, 1), 3),
                "revenue_eur": round(b["revenue_eur"], 2),
                "eur_per_req": round(b["revenue_eur"] / max(n, 1), 4),
            }
        )

    summary.sort(key=lambda r: r["total_reqs"], reverse=True)
    return summary


def _sql_monitor_window(team_id: str, window_minutes: int) -> str:
    return f"""
SELECT
    coalesce(nullIf(s.model_version::String, ''), 'fallback_or_unknown') AS model_version,
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


def _sql_monitor_fallback(team_id: str) -> str:
    return f"""
SELECT
    coalesce(nullIf(s.model_version::String, ''), 'fallback_or_unknown') AS model_version,
    f.clicked::String AS clicked,
    f.purchased::String AS purchased,
    f.revenue_eur::Float64 AS revenue_eur
FROM {team_id}.feedback AS f
LEFT JOIN {team_id}.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
ORDER BY f.ts DESC
LIMIT 2000
"""


def format_report(summary: list[dict[str, Any]], window_minutes: int) -> str:
    if not summary:
        return f"[MONITOR] Sense dades de feedback als últims {window_minutes} min."
    header = (
        f"=== Monitor En Viu · Últims {window_minutes} min ===\n"
        f"{'model_version':<22} {'reqs':>6} {'on_time%':>9} {'p50_ms':>8} "
        f"{'p95_ms':>8} {'CTR%':>8} {'CVR%':>8} {'€/req':>8}"
    )
    lines = [header]
    for r in summary:
        lines.append(
            f"{r['model_version']:<22} {r['total_reqs']:>6} {r['on_time_pct']:>8.2f}% "
            f"{r['p50_ms']:>8.1f} {r['p95_ms']:>8.1f} {r['ctr_pct']:>7.3f}% "
            f"{r['cvr_pct']:>7.3f}% {r['eur_per_req']:>8.4f}"
        )
    return "\n".join(lines)


def run_dry_run() -> None:
    """Validate monitor aggregation locally with realistic mixed feedback."""
    sample_rows = [
        {
            "model_version": "odyseus-4.1e",
            "served_by": "team",
            "latency_ms": 420.0,
            "clicked": "p_0001",
            "purchased": "p_0001",
            "revenue_eur": 79.9,
        },
        {
            "model_version": "odyseus-4.1e",
            "served_by": "team",
            "latency_ms": 610.0,
            "clicked": "p_0002",
            "purchased": "",
            "revenue_eur": 0.0,
        },
        {
            "model_version": "fallback_or_unknown",
            "served_by": "baseline",
            "latency_ms": 3450.0,
            "clicked": "",
            "purchased": "",
            "revenue_eur": 0.0,
        },
    ]
    summary = summarize_feedback_rows(sample_rows)
    print(format_report(summary, window_minutes=30))
    print(f"[MONITOR DRY-RUN OK] Tracked {len(summary)} model buckets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 Live RawTree Health Monitor")
    parser.add_argument("--window-min", type=int, default=30, help="Lookback window in minutes")
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh interval seconds (0=once)")
    parser.add_argument("--dry-run", action="store_true", help="Run offline monitor verification")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    cfg = ConfigV4.from_env()
    if not cfg.team_id or not cfg.api_key:
        raise SystemExit("TEAM_ID i RAWTREE_API_KEY són obligatoris per consultar RawTree.")
    client = RawTreeClient(api_url=cfg.api_url, api_key=cfg.api_key, team_id=cfg.team_id)

    while True:
        rows = client.query_safe(_sql_monitor_window(cfg.team_id, args.window_min), timeout=8.0)
        if rows is None:
            rows = client.query_safe(_sql_monitor_fallback(cfg.team_id), timeout=8.0) or []
        summary = summarize_feedback_rows(rows)
        print(format_report(summary, args.window_min))
        if args.watch <= 0.0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
