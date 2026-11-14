"""Terminal live monitor (`python model_manager.py live`) showing active model and request feed."""
from __future__ import annotations

import os
import time

from .process_controller import get_runtime_overview
from .registry import discover_models
from .telemetry import fetch_rawtree_telemetry


def run_live_tui(*, window_minutes: int = 3, refresh_s: float = 2.0, req_limit: int = 15) -> int:
    """Continuously render active model status and real-time requests in terminal."""
    print("Arrencant monitor en viu (Ctrl+C per sortir)...")
    try:
        while True:
            registry = discover_models()
            overview = get_runtime_overview(registry)
            telemetry = fetch_rawtree_telemetry(window_minutes=window_minutes, req_limit=req_limit)

            os.system("cls" if os.name == "nt" else "clear")
            print("=== GESTOR DE MODELS · MONITOR EN TEMPS REAL ===")
            serves = [p for p in overview["processes"] if p["role"] == "serve"]
            if serves:
                for p in serves:
                    hb = p.get("heartbeat") or {}
                    hb_txt = (
                        f"snap=v{hb.get('snapshot_version')} ml_alpha={hb.get('ml_alpha')} enviats={hb.get('submitted_total')}"
                        if hb
                        else "sense heartbeat JSON"
                    )
                    print(
                        f"ACTIU -> {p['folder']} ({p['model_version']}) | PID {p['pid']} | "
                        f"uptime {p['uptime_str']} | {p['rss_mb']} MB | {hb_txt}"
                    )
            else:
                print("ACTIU -> CAP PROCES serve.py EN MARXA")

            if telemetry.get("error"):
                print(f"\n[AVIS RAWTREE] {telemetry['error']}")
            else:
                print(f"\n--- Rendiment per model (ultims {window_minutes} min) ---")
                print(f"  {'MODEL_VERSION':<18} {'REQS':>6} {'ON_TIME%':>9} {'P50_MS':>8} {'CTR%':>7} {'CVR%':>7} {'EUR/REQ':>8}")
                for row in telemetry["models_summary"]:
                    print(
                        f"  {row['model_version']:<18} {row['reqs']:>6} {row['on_time_pct']:>8.2f}% "
                        f"{row['p50_ms']:>8.1f} {row['ctr_pct']:>6.2f}% {row['cvr_pct']:>6.2f}% "
                        f"{row['eur_per_req']:>8.4f}"
                    )

                print(f"\n--- Ultimes {req_limit} sol.licituds al servidor ---")
                print(f"  {'HORA':<9} {'REQUEST_ID':<20} {'MODEL':<15} {'ORIGEN':<10} {'LATENCIA':>9} {'RESULTAT'}")
                for req in telemetry["recent_requests"][:req_limit]:
                    outcome = "-"
                    if req["purchased"]:
                        outcome = f"COMPRA +{req['revenue_eur']:.2f} EUR"
                    elif req["clicked"]:
                        outcome = f"CLIC ({req['clicked']})"
                    print(
                        f"  {req['time']:<9} {req['request_id']:<20} {req['model_version']:<15} "
                        f"{req['served_by']:<10} {req['latency_ms']:>7.1f}ms {outcome}"
                    )

            print("\nActualitzant cada %.1fs · Ctrl+C per aturar" % refresh_s)
            time.sleep(refresh_s)
    except KeyboardInterrupt:
        print("\n[LIVE MONITOR] Aturat.")
        return 0
