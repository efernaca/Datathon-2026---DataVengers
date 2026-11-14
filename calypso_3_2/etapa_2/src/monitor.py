"""Alarma (F7): cada minuto lee `team.feedback` y avisa si el sistema se degrada.

    python -m src.monitor                 # en otra terminal, junto al servicio

Avisa si en los últimos 5 minutos:
  * `served_by = 'team'` < 98 %   (la luz del leaderboard se pone roja por debajo del 90 %)
  * p95 de `latency_ms` > 2.000 ms
  * no llega feedback desde hace > 3 min (servicio parado o stream caído)
  * el latido local (`cache/heartbeat`) lleva > 10 s sin actualizarse
Canales: pitido en consola (y `winsound` en Windows) y, si existen las variables
TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID, un mensaje de Telegram.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
import logging
import os
import sys
import time

from .config import Config
from .rawtree_client import RawTreeClient

logger = logging.getLogger("e2esp.monitor")

SQL = """
SELECT count() AS n,
       countIf(served_by::String = 'team') / greatest(count(), 1) AS team_share,
       quantile(0.95)(latency_ms::Float64) AS p95_latency,
       countIf(fallback_reason::String = 'late') AS late,
       countIf(fallback_reason::String = 'missing') AS missing,
       countIf(fallback_reason::String = 'invalid') AS invalid,
       countIf(ifNull(clicked::String, '') != '') / greatest(count(), 1) AS ctr,
       countIf(ifNull(purchased::String, '') != '') / greatest(count(), 1) AS cvr,
       dateDiff('second', max(ts::DateTime64(3)), now64(3)) AS secs_since_last
FROM {team}.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) > now64(3) - INTERVAL 5 MINUTE
"""


def alert(msg: str) -> None:
    logger.error("ALERTA: %s", msg)
    sys.stdout.write("\a")
    sys.stdout.flush()
    try:
        import winsound  # type: ignore

        winsound.Beep(1200, 600)
    except Exception:  # noqa: BLE001
        pass
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat:
        try:
            import requests

            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": f"[DataVengers e2esp] {msg}"}, timeout=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram falló: %s", exc)


def check(client: RawTreeClient, cfg: Config, min_share: float, max_p95: float) -> list[str]:
    problems = []
    rows = client.query_safe(SQL.format(team=cfg.team_id), timeout=10)
    if rows is None:
        return ["no se puede consultar feedback (¿red o RawTree caídos?)"]
    r = rows[0] if rows else {}
    n = int(float(r.get("n", 0) or 0))
    if n == 0:
        problems.append("sin feedback en los últimos 5 minutos")
    else:
        share = float(r["team_share"])
        p95 = float(r["p95_latency"] or 0)
        logger.info("5 min: n=%d team=%.1f%% p95=%.0f ms late=%s missing=%s invalid=%s ctr=%.3f cvr=%.4f",
                    n, 100 * share, p95, r["late"], r["missing"], r["invalid"], float(r["ctr"]), float(r["cvr"]))
        if share < min_share:
            problems.append(f"served_by=team {100 * share:.1f}% < {100 * min_share:.0f}% "
                            f"(late={r['late']}, missing={r['missing']}, invalid={r['invalid']})")
        if p95 > max_p95:
            problems.append(f"p95 latency {p95:.0f} ms > {max_p95:.0f} ms")
        if float(r.get("secs_since_last") or 0) > 180:
            problems.append(f"último feedback hace {r['secs_since_last']} s")
    try:
        age = time.time() - os.path.getmtime(cfg.heartbeat_file)
        if age > 10:
            problems.append(f"latido local parado hace {age:.0f} s (¿proceso muerto?)")
    except OSError:
        pass
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=60)
    ap.add_argument("--min-share", type=float, default=0.98)
    ap.add_argument("--max-p95", type=float, default=2000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = Config.from_env()
    client = RawTreeClient(cfg.api_url, cfg.api_key, cfg.team_id)
    while True:
        for p in check(client, cfg, args.min_share, args.max_p95):
            alert(p)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
