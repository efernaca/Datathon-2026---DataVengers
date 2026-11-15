"""Exporta el feedback en vivo de RawTree para los paneles 1, 5 y 6 del dashboard.

Ejecutar en una máquina con red a api.rawtree.com (la del equipo, no la sesión de Claude):

    python dashboard/export_live.py                       # desde las 10:00 de hoy (hora local)
    python dashboard/export_live.py --since "2026-11-15 09:00:00"   # ts tal como lo guarda RawTree
    python dashboard/build_data.py                        # reconstruye index.html con los datos

Lee TEAM_ID y RAWTREE_API_KEY de group_API.env (o variables de entorno). Solo librería estándar.
Escribe dashboard/live/live_export.json (ignorado por git: contiene datos del feedback privado).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = HERE / "live"
PAGE = 20000


def load_env() -> None:
    for cand in (ROOT / "group_API.env", ROOT / ".env", ROOT / "calypso_6_2_b" / ".env"):
        if not cand.exists():
            continue
        for line in cand.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip().lstrip("﻿")
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def query(sql: str, timeout: float = 60.0) -> list[dict]:
    url = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com").rstrip("/") + "/v1/query"
    req = urllib.request.Request(
        url,
        data=json.dumps({"sql": sql}).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.environ['RAWTREE_API_KEY']}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")).get("data", [])
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"RawTree {exc.code}: {exc.read()[:400]!r}\nSQL:\n{sql}")
        except Exception as exc:  # timeouts / red
            if attempt == 2:
                raise SystemExit(f"Sin conexión con RawTree: {exc}")
            time.sleep(2 * (attempt + 1))
    return []


def main() -> int:
    load_env()
    team = os.environ.get("TEAM_ID", "").strip()
    if not team or not os.environ.get("RAWTREE_API_KEY"):
        print("Falta TEAM_ID o RAWTREE_API_KEY (group_API.env)", file=sys.stderr)
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="ts mínimo del feedback (por defecto, hoy 10:00 hora local -> UTC)")
    ap.add_argument("--bucket-minutes", type=int, default=10)
    ap.add_argument("--max-rows", type=int, default=400000)
    args = ap.parse_args()

    if args.since:
        since = args.since
    else:
        local10 = datetime.now().astimezone().replace(hour=10, minute=0, second=0, microsecond=0)
        since = local10.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[export] equipo {team}, feedback desde {since} (UTC)")

    base = f"""
SELECT f.request_id::String                 AS request_id,
       f.ts::String                         AS ts,
       f.served_by::String                  AS served_by,
       f.fallback_reason::String            AS fallback_reason,
       f.latency_ms::Float64                AS latency_ms,
       f.clicked::String                    AS clicked,
       f.purchased::String                  AS purchased,
       coalesce(f.revenue_eur::Float64, 0)  AS revenue_eur,
       r.is_new_user::Bool                  AS is_new_user,
       r.context.page::String               AS page,
       r.context.device::String             AS device,
       r.recent_views                       AS recent_views,
       s.model_version::String              AS model_version
FROM {team}.feedback AS f
LEFT JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
LEFT JOIN {team}.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
  AND f.ts::DateTime64(3) >= toDateTime64('{since}', 3)
ORDER BY f.ts::DateTime64(3) ASC"""
    rows: list[dict] = []
    while len(rows) < args.max_rows:
        chunk = query(f"{base}\nLIMIT {PAGE} OFFSET {len(rows)}")
        rows.extend(chunk)
        print(f"[export] {len(rows)} filas…")
        if len(chunk) < PAGE:
            break

    new_products = [r["product_id"] for r in query(
        "SELECT DISTINCT product_id::String AS product_id FROM shop.catalog_updates WHERE change::String = 'new_product'"
    )]
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / ".gitignore").write_text("*\n", encoding="utf-8")
    out = {
        "exported_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "team": team,
        "since": since,
        "bucket_minutes": args.bucket_minutes,
        "new_products": new_products,
        "rows": rows,
    }
    path = OUT_DIR / "live_export.json"
    path.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[export] {len(rows)} peticiones y {len(new_products)} productos nuevos -> {path.relative_to(ROOT)}")
    print("[export] ahora: python dashboard/build_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
