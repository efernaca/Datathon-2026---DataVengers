"""Odyseus 2.0 · piloto automático del A/B (para dejarlo solo durante la noche).

    python ml_autopilot.py --decide-at 07:30

No arranca ni reinicia serve.py: solo escribe `ab_control.json`, que serve.py relee cada 5 s.
Todo queda registrado en logs/ab_autopilot.log.

Qué hace:
  1. Rampa: empieza con `--start-share` (20 %). Pasados `--ramp-min` minutos, si la rama B responde
     a tiempo (>= 98 % y como mucho 1,5 puntos por debajo de A) sube a `--full-share` (50 %).
  2. Vigilancia cada `--every-min` minutos. APAGA la rama B (reparto 0 = todo Calypso 5.1) si:
       - B responde a tiempo 3 puntos peor que A en la última ventana, o
       - B va claramente peor en compras/100 con todo lo acumulado (z <= -2.5).
  3. Decisión final a la hora `--decide-at` (hora local del ordenador), si se indica:
       - todo a B (reparto 1) si B no va por detrás (z >= -0.5) y no pierde claramente en usuarios nuevos;
       - todo a A (reparto 0) en caso contrario.
     Sin `--decide-at` mantiene el reparto y sigue vigilando hasta que lo pares.
Si RawTree no responde o no hay datos suficientes, NO cambia nada y lo vuelve a intentar.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from .config import ConfigV4
from .ml_ab_report import _z, ab_summary, arm_pair, format_table
from .rawtree_client import RawTreeClient

MODEL_ROOT = Path(__file__).resolve().parents[2]
CONTROL = MODEL_ROOT / "ab_control.json"
LOG = MODEL_ROOT / "logs" / "ab_autopilot.log"


def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def set_control(share: float, rebuy: bool, reason: str) -> None:
    payload = {"ab_share": float(share), "rebuy_in_b": bool(rebuy), "reason": reason,
               "updated_at": datetime.now().isoformat(timespec="seconds")}
    tmp = CONTROL.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, CONTROL)
    _log(f"REPARTO -> rama B {100 * share:.0f} % | recompra en B: {'sí' if rebuy else 'no'} | {reason}")


def _pct(a: list[float]) -> float:
    return 100.0 * a[1] / max(a[0], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Odyseus 2.0 · piloto automático del A/B")
    ap.add_argument("--start-share", type=float, default=0.2)
    ap.add_argument("--full-share", type=float, default=0.5)
    ap.add_argument("--ramp-min", type=float, default=30.0)
    ap.add_argument("--every-min", type=float, default=30.0)
    ap.add_argument("--no-rebuy", action="store_true", help="Rama B sin recompra (solo la red)")
    ap.add_argument("--decide-at", default="", help="Hora local HH:MM de la decisión final (p. ej. 07:30)")
    ap.add_argument("--min-requests", type=int, default=300, help="Peticiones mínimas de B en la ventana para juzgar")
    args = ap.parse_args()

    cfg = ConfigV4.from_env()
    client = RawTreeClient(api_url=cfg.api_url, api_key=cfg.api_key, team_id=cfg.team_id)
    rebuy = not args.no_rebuy
    started = time.time()
    decide_ts: float | None = None
    if args.decide_at:
        hh, mm = (int(x) for x in args.decide_at.split(":"))
        d = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if d <= datetime.now():
            d += timedelta(days=1)
        decide_ts = d.timestamp()
        _log(f"Decisión final programada para {d.strftime('%Y-%m-%d %H:%M')}")
    if not (MODEL_ROOT / "ml_model.json").is_file():
        _log("AVISO: no existe ml_model.json; las ramas A y B responderán lo mismo hasta que se exporte.")

    set_control(args.start_share, rebuy, "inicio de la rampa")
    phase = "ramp"          # ramp -> full -> (off | final)
    next_check = started + args.ramp_min * 60.0

    while True:
        now = time.time()
        if decide_ts is not None and now >= decide_ts and phase != "final":
            try:
                agg = ab_summary(client, cfg.team_id, (now - started) / 60.0)
                _log("Decisión final. Acumulado desde el inicio:\n" + format_table(agg))
                a, b = arm_pair(agg, "total")
                if phase == "off":
                    set_control(0.0, rebuy, "final: la rama B ya estaba apagada por seguridad -> todo Calypso 5.1")
                elif not (a and b and a[0] >= args.min_requests and b[0] >= args.min_requests):
                    set_control(0.0, rebuy, "final: datos insuficientes para juzgar -> todo Calypso 5.1")
                else:
                    z = _z(int(a[2]), int(a[0]), int(b[2]), int(b[0]))
                    an, bn = arm_pair(agg, "nuevos")
                    z_new = _z(int(an[2]), int(an[0]), int(bn[2]), int(bn[0])) if (an and bn and an[0] and bn[0]) else 0.0
                    ok_time = _pct(b) >= _pct(a) - 1.5
                    if z >= -0.5 and z_new > -2.0 and ok_time:
                        set_control(1.0, rebuy, f"final: B no va por detrás (z={z:+.2f}, nuevos z={z_new:+.2f}) -> todo a B")
                    else:
                        set_control(0.0, rebuy, f"final: B por detrás o menos fiable (z={z:+.2f}, nuevos z={z_new:+.2f}, a tiempo B {_pct(b):.1f} % / A {_pct(a):.1f} %) -> todo Calypso 5.1")
                phase = "final"
                _log("Piloto automático terminado. serve.py sigue sirviendo con el reparto final.")
                return
            except Exception as exc:
                _log(f"Decisión final: RawTree no responde ({str(exc)[:160]}); reintento en 60 s")
                time.sleep(60.0)
                continue

        if now < next_check:
            time.sleep(min(20.0, max(next_check - now, 1.0)))
            continue

        window_min = args.ramp_min if phase == "ramp" else args.every_min
        try:
            recent = ab_summary(client, cfg.team_id, window_min)
            total = ab_summary(client, cfg.team_id, (now - started) / 60.0 + 1.0)
        except Exception as exc:
            _log(f"RawTree no responde ({str(exc)[:160]}); no se cambia nada, reintento en 2 min")
            next_check = now + 120.0
            continue
        _log(f"Últimos {window_min:.0f} min:\n" + (format_table(recent) if recent else "(sin filas con ramas -A/-B)"))
        _log("Acumulado:\n" + (format_table(total) if total else "(sin filas)"))
        a, b = arm_pair(recent, "total")
        ta, tb = arm_pair(total, "total")

        if phase == "off":
            next_check = now + args.every_min * 60.0
            continue
        if not (a and b and b[0] >= args.min_requests * (0.3 if phase == "ramp" else 1.0)):
            _log("Datos insuficientes de la rama B en la ventana (¿serve.py en marcha?); no se cambia nada.")
            next_check = now + 300.0
            continue

        if _pct(b) < _pct(a) - 3.0 or (phase == "ramp" and (_pct(b) < 98.0 and _pct(b) < _pct(a) - 1.5)):
            set_control(0.0, rebuy, f"SEGURIDAD: B responde a tiempo {_pct(b):.1f} % frente a {_pct(a):.1f} % de A -> rama B apagada")
            phase = "off"
        elif ta and tb and ta[0] >= 2000 and tb[0] >= 2000 and _z(int(ta[2]), int(ta[0]), int(tb[2]), int(tb[0])) <= -2.5:
            set_control(0.0, rebuy, "SEGURIDAD: B claramente peor en compras/100 (z <= -2.5) -> rama B apagada")
            phase = "off"
        elif phase == "ramp":
            set_control(args.full_share, rebuy, f"rampa superada: B a tiempo {_pct(b):.1f} % (A {_pct(a):.1f} %)")
            phase = "full"
        next_check = now + args.every_min * 60.0


if __name__ == "__main__":
    main()
