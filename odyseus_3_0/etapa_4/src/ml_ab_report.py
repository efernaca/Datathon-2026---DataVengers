"""Odyseus 2.0 · resultado del A/B en vivo por rama (model_version), total y en peticiones frías.

    python ml_ab_report.py              # desde que hay ramas -A / -B
    python ml_ab_report.py --hours 4    # solo las últimas 4 horas
"""
from __future__ import annotations

import argparse
import math

from .config import ConfigV4
from .rawtree_client import RawTreeClient

SQL = """
SELECT s.model_version::String AS model_version,
       if(r.is_new_user::Bool, 1, 0) AS is_new,
       count() AS n,
       countIf(f.served_by::String = 'team') AS on_time,
       countIf(f.purchased IS NOT NULL AND f.purchased::String != '') AS buys,
       countIf(f.clicked IS NOT NULL AND f.clicked::String != '') AS clicks,
       sum(ifNull(f.revenue_eur::Float64, 0.0)) AS revenue
FROM {team}.feedback AS f
INNER JOIN {team}.submissions AS s ON s.request_id::String = f.request_id::String
LEFT JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
  AND (endsWith(s.model_version::String, '-A') OR endsWith(s.model_version::String, '-B'))
  {window}
GROUP BY model_version, is_new
ORDER BY model_version, is_new
"""


SQL_SIMPLE = """
SELECT s.model_version::String AS model_version,
       count() AS n,
       countIf(f.served_by::String = 'team') AS on_time,
       countIf(f.purchased IS NOT NULL AND f.purchased::String != '') AS buys,
       countIf(f.clicked IS NOT NULL AND f.clicked::String != '') AS clicks,
       sum(ifNull(f.revenue_eur::Float64, 0.0)) AS revenue
FROM {team}.feedback AS f
INNER JOIN {team}.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
  AND (endsWith(s.model_version::String, '-A') OR endsWith(s.model_version::String, '-B'))
  {window}
GROUP BY model_version
ORDER BY model_version
"""


def _z(b1: int, n1: int, b2: int, n2: int) -> float:
    if min(n1, n2) == 0:
        return float("nan")
    p = (b1 + b2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-18))
    return (b2 / n2 - b1 / n1) / se


def ab_summary(client: RawTreeClient, team_id: str, minutes: float = 0.0) -> dict[tuple[str, str], list[float]]:
    """{(model_version, segmento): [peticiones, a_tiempo, compras, clics, ingresos]}; segmento = total|nuevos|conocidos.

    Si la consulta con usuarios nuevos falla, repite sin el cruce con shop.requests (solo 'total').
    """
    window = f"AND f.ts::DateTime64(3) >= now64(3) - INTERVAL {max(int(minutes), 1)} MINUTE" if minutes > 0 else ""
    with_new = True
    try:
        rows = client.query(SQL.format(team=team_id, window=window), timeout=120.0)
    except Exception:
        with_new = False
        rows = client.query(SQL_SIMPLE.format(team=team_id, window=window), timeout=120.0)
    agg: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        mv = str(r["model_version"])
        segs = ("total", "nuevos" if int(r.get("is_new") or 0) else "conocidos") if with_new else ("total",)
        for seg in segs:
            a = agg.setdefault((mv, seg), [0, 0, 0, 0, 0.0])
            a[0] += int(r["n"]); a[1] += int(r["on_time"]); a[2] += int(r["buys"]); a[3] += int(r["clicks"]); a[4] += float(r["revenue"] or 0)
    return agg


def arm_pair(agg: dict[tuple[str, str], list[float]], seg: str = "total") -> tuple[list[float] | None, list[float] | None]:
    arms = sorted({mv for mv, _ in agg})
    A = [m for m in arms if m.endswith("-A")]
    B = [m for m in arms if m.endswith("-B")]
    return (agg.get((A[-1], seg)) if A else None, agg.get((B[-1], seg)) if B else None)


def format_table(agg: dict[tuple[str, str], list[float]]) -> str:
    lines = [f"{'rama':<22} {'segmento':<10} {'peticiones':>10} {'a tiempo':>9} {'compras':>8} {'compras/100':>12} {'clics/100':>10} {'€/petición':>11}"]
    for (mv, seg), a in sorted(agg.items(), key=lambda kv: (kv[0][1] != "total", kv[0][1], kv[0][0])):
        n = max(a[0], 1)
        lines.append(f"{mv:<22} {seg:<10} {int(a[0]):>10,} {100 * a[1] / n:>8.2f}% {int(a[2]):>8,} {100 * a[2] / n:>12.3f} {100 * a[3] / n:>10.3f} {a[4] / n:>11.4f}")
    for seg in ("total", "nuevos"):
        a, b = arm_pair(agg, seg)
        if a and b and a[0] and b[0]:
            z = _z(int(a[2]), int(a[0]), int(b[2]), int(b[0]))
            lift = (b[2] / b[0]) / max(a[2] / a[0], 1e-12) - 1
            lines.append(f"[{seg}] B frente a A: {100 * lift:+.1f} % en compras/100 | z = {z:+.2f} "
                         f"({'señal clara' if abs(z) >= 2 else 'dentro del ruido'}; hace falta |z| >= 2)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=0.0)
    args = ap.parse_args()
    cfg = ConfigV4.from_env()
    client = RawTreeClient(api_url=cfg.api_url, api_key=cfg.api_key, team_id=cfg.team_id)
    agg = ab_summary(client, cfg.team_id, args.hours * 60.0)
    if not agg:
        print("Sin filas con ramas -A / -B todavía (¿serve.py lanzado con --ab-share?).")
        return
    print(format_table(agg))
    return
    print(f"{'rama':<22} {'segmento':<10} {'peticiones':>10} {'a tiempo':>9} {'compras':>8} {'compras/100':>12} {'clics/100':>10} {'€/petición':>11}")
    for (mv, seg), a in sorted(agg.items(), key=lambda kv: (kv[0][1] != "total", kv[0][1], kv[0][0])):
        n = max(a[0], 1)
        print(f"{mv:<22} {seg:<10} {int(a[0]):>10,} {100 * a[1] / n:>8.2f}% {int(a[2]):>8,} {100 * a[2] / n:>12.3f} {100 * a[3] / n:>10.3f} {a[4] / n:>11.4f}")
    arms = sorted({mv for mv, _ in agg})
    A = [m for m in arms if m.endswith("-A")]; B = [m for m in arms if m.endswith("-B")]
    if A and B:
        for seg in ("total", "nuevos"):
            a, b = agg.get((A[-1], seg)), agg.get((B[-1], seg))
            if a and b and a[0] and b[0]:
                z = _z(int(a[2]), int(a[0]), int(b[2]), int(b[0]))
                lift = (b[2] / b[0]) / max(a[2] / a[0], 1e-12) - 1
                print(f"\n[{seg}] B frente a A: {100 * lift:+.1f} % en compras/100 | z = {z:+.2f} "
                      f"({'señal clara' if abs(z) >= 2 else 'dentro del ruido'}; hace falta |z| >= 2)")


if __name__ == "__main__":
    main()
