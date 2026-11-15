"""Unified CLI entrypoint for gestor_models."""
from __future__ import annotations

import argparse
from collections import defaultdict
import time

import psutil

from .config import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_WINDOW_MINUTES, configure_stdio
from .process_controller import (
    _launch_serve_process,
    _wait_for_readiness,
    execute_transition,
    find_running_processes,
    get_runtime_overview,
    record_active_state,
    stop_models,
    tail_lines,
    terminate_pid,
)
from .registry import ModelSpec, discover_models, resolve_model
from .telemetry import fetch_rawtree_telemetry
from .tui import run_live_tui
from .web_server import run_web_ui


def _cmd_list(registry: dict[str, ModelSpec]) -> int:
    overview = get_runtime_overview(registry)
    state = overview["active_state"] or {}
    recorded_folder = state.get("active_folder")

    print("Models disponibles al projecte:")
    print(f"  {'CARPETA':<26} {'TIPUS':<14} {'MODEL_VERSION':<18} {'ESTAT'}")
    print("  " + "-" * 78)
    for m in overview["models"]:
        tag = "[ENCAPSULAT]" if m["kind"] == "encapsulated" else "[HISTORIC]"
        if m["running_serves"]:
            pids = ",".join(str(p["pid"]) for p in m["running_serves"])
            status_txt = f"ACTIU EN VIU (PID {pids})"
        elif m["running_trainers"]:
            pids = ",".join(str(x) for x in m["running_trainers"])
            status_txt = f"ENTRENANT (PID {pids})"
        elif m["folder"] == recorded_folder:
            status_txt = "Ultim registrat (aturat)"
        else:
            status_txt = "Aturat"
        print(f"  {m['folder']:<26} {tag:<14} {m['model_version']:<18} {status_txt}")
    return 0


def _cmd_status(registry: dict[str, ModelSpec], window_minutes: int) -> int:
    overview = get_runtime_overview(registry)
    state = overview["active_state"]
    print("=== ESTAT DEL SERVEI EN VIU (LOCAL + RAWTREE) ===")
    if state:
        alive_lbl = "ACTIU" if state.get("pid_alive") else "NO ACTIU"
        print(
            f"Registre (.active_model.json): {state.get('active_folder')} "
            f"({state.get('model_name')} / model_version={state.get('model_version')}, "
            f"PID={state.get('pid')} [{alive_lbl}])"
        )
    else:
        print("Registre (.active_model.json): encara no inicialitzat")

    print("\n1) Processos locals detectats (psutil):")
    serves = [p for p in overview["processes"] if p["role"] == "serve"]
    trainers = [p for p in overview["processes"] if p["role"] == "trainer"]
    if not serves and not trainers:
        print("   Cap proces serve.py en execucio ara mateix.")
    for p in serves:
        print(
            f"   - PID {p['pid']:<7} | carpeta={p['folder']:<24} | model_version={p['model_version']:<16} | "
            f"uptime={p['uptime_str']:<10} | RAM={p['rss_mb']:.1f} MB"
        )
        hb = p.get("heartbeat")
        if hb:
            print(
                f"     Heartbeat: ready={hb.get('ready')} | snap=v{hb.get('snapshot_version')} | "
                f"ml_alpha={hb.get('ml_alpha')} | regime={hb.get('regime')} | "
                f"enviats={hb.get('submitted_total')} | fa {hb.get('age_s')}s"
            )
    for t in trainers:
        print(f"   - [ENTRENANT] PID {t['pid']:<7} | carpeta={t['folder']} | RAM={t['rss_mb']:.1f} MB")
    for w in overview["watchdogs"]:
        print(f"   - [WATCHDOG]  PID {w['pid']} vigilant '{w['folder']}'")

    print(f"\n2) Telemetria RawTree en directe (ultims {window_minutes} min):")
    telemetry = fetch_rawtree_telemetry(window_minutes=window_minutes, req_limit=10)
    if telemetry.get("error"):
        print(f"   {telemetry['error']}")
    elif telemetry["models_summary"]:
        print("   Feedback real (team_XX.feedback):")
        print(
            f"     {'MODEL_VERSION':<20} {'REQS':>6} {'ON_TIME%':>9} "
            f"{'P50_MS':>8} {'P95_MS':>8} {'CTR%':>7} {'CVR%':>7} {'EUR/REQ':>8}"
        )
        for row in telemetry["models_summary"]:
            print(
                f"     {row['model_version']:<20} {row['reqs']:>6} "
                f"{row['on_time_pct']:>8.2f}% {row['p50_ms']:>8.1f} {row['p95_ms']:>8.1f} "
                f"{row['ctr_pct']:>6.2f}% {row['cvr_pct']:>6.2f}% {row['eur_per_req']:>8.4f}"
            )
    else:
        print(f"   Cap resposta registrada als ultims {window_minutes} minuts.")

    recent = telemetry.get("recent_requests", [])[:5]
    if recent:
        print("\n3) Ultimes 5 sol.licituds al servidor:")
        for r in recent:
            res = f"COMPRA +{r['revenue_eur']:.2f}€" if r["purchased"] else (f"CLIC {r['clicked']}" if r["clicked"] else "-")
            print(
                f"   - {r['time']} | {r['request_id']} | model={r['model_version']} | "
                f"{r['served_by']} | {r['latency_ms']:.0f}ms | {res}"
            )

    local_mvs = {p["model_version"] for p in serves}
    ext_rows = [
        row["model_version"]
        for row in (telemetry.get("models_summary") or [])
        if row["model_version"] != "fallback" and row["model_version"] not in local_mvs
    ]
    colliding = telemetry.get("collision_request_ids") or []
    if ext_rows:
        print("\n[ALERTA CONFLICTE EXTERN]")
        print(
            f"   Hi ha models EXTERNS enviant respostes amb el mateix TEAM_ID des d'un ALTRE PC/servidor: "
            f"{', '.join(ext_rows)}"
        )
        if colliding:
            print(
                f"   Col.lisions directes detectades (mateix request_id respost per dos models alhora): "
                f"{', '.join(colliding[:5])}"
            )
        print("   Accio requerida: atura aquests processos a la maquina externa on corren o rota la RAWTREE_API_KEY.")
    return 0


def _cmd_logs(registry: dict[str, ModelSpec], target: str | None, n_lines: int) -> int:
    overview = get_runtime_overview(registry)
    if target:
        spec = resolve_model(registry, target)
    else:
        serves = [p for p in overview["processes"] if p["role"] == "serve" and p["folder"] in registry]
        state = overview["active_state"] or {}
        if serves:
            spec = registry[serves[0]["folder"]]
        elif state.get("active_folder") in registry:
            spec = registry[str(state["active_folder"])]
        else:
            spec = next(iter(registry.values()))

    lines = tail_lines(spec.log_path, n=n_lines)
    if not lines:
        print(f"No hi ha logs disponibles a {spec.log_path}")
        return 0
    print(f"=== Ultimes {len(lines)} linies de {spec.log_path} ===")
    for line in lines:
        print(line)
    return 0


def _cmd_watchdog(registry: dict[str, ModelSpec], target: str | None, interval_s: float) -> int:
    if target:
        spec = resolve_model(registry, target)
    else:
        overview = get_runtime_overview(registry)
        state = overview["active_state"] or {}
        folder = str(state.get("active_folder") or "calypso_3_1")
        spec = resolve_model(registry, folder)

    print(f"[WATCHDOG] Supervisant '{spec.folder}' ({spec.model_version}) cada {interval_s:.1f}s...")
    try:
        while True:
            serves = [
                p for p in find_running_processes(registry)
                if p.role == "serve" and p.folder == spec.folder
            ]
            needs_restart = len(serves) == 0
            if needs_restart:
                print(f"[WATCHDOG] Engegant '{spec.folder}'...")
                proc, log_offset = _launch_serve_process(spec)
                ok, msg = _wait_for_readiness(spec, proc, proc.pid, log_offset, timeout_s=120.0)
                if ok:
                    record_active_state(spec, proc.pid)
                    print(f"[WATCHDOG] Actiu PID {proc.pid}: {msg}")
                else:
                    print(f"[WATCHDOG] Error d'arrencada: {msg}")
                    terminate_pid(proc.pid, wait_s=3.0)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("[WATCHDOG] Aturat per l'usuari.")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gestor modular de models RawTree (Web UI + CLI)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ui = sub.add_parser("ui", help="Obre la interfaç web en temps real")
    p_ui.add_argument("--host", default=DEFAULT_HOST, help="Host HTTP (per defecte 127.0.0.1)")
    p_ui.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port HTTP (per defecte 8765)")
    p_ui.add_argument("--no-browser", action="store_true", help="No obrir el navegador automaticament")

    p_live = sub.add_parser("live", help="Monitor en temps real a la terminal")
    p_live.add_argument("--window", type=int, default=DEFAULT_WINDOW_MINUTES, help="Finestra en minuts")
    p_live.add_argument("--refresh", type=float, default=2.0, help="Segons entre actualitzacions")
    p_live.add_argument("--limit", type=int, default=15, help="Sol.licituds visibles")

    sub.add_parser("list", help="Llista tots els models encapsulats i historics")

    p_status = sub.add_parser("status", help="Mostra el model actiu, processos i ultimes sol.licituds")
    p_status.add_argument("--window", type=int, default=DEFAULT_WINDOW_MINUTES, help="Finestra de minuts")

    p_train = sub.add_parser("train", help="Entrena un model en segon pla i l'activa sense conflictes")
    p_train.add_argument("model", help="Nom de la carpeta o model (ex: calypso_3_1)")
    p_train.add_argument("--timeout", type=float, default=180.0, help="Temps maxim d'espera (s)")

    p_switch = sub.add_parser("switch", help="Canvia a un altre model amb Zero-Downtime")
    p_switch.add_argument("model", help="Nom de la carpeta o model (ex: calypso_3_1)")
    p_switch.add_argument("--timeout", type=float, default=120.0, help="Temps maxim d'espera (s)")
    p_switch.add_argument("--force", action="store_true", help="Forca reinici fins i tot si ja es actiu")

    p_start = sub.add_parser("start", help="Engega un model si no n'hi ha cap altre actiu")
    p_start.add_argument("model", help="Nom de la carpeta o model")
    p_start.add_argument("--timeout", type=float, default=120.0, help="Temps maxim d'espera (s)")

    p_stop = sub.add_parser("stop", help="Atura els processos de model en execucio")
    p_stop.add_argument("--model", type=str, default="", help="Opcional: nomes atura aquesta carpeta")

    p_logs = sub.add_parser("logs", help="Mostra les ultimes linies del log del model")
    p_logs.add_argument("model", nargs="?", default="", help="Carpeta del model (opcional)")
    p_logs.add_argument("-n", "--lines", type=int, default=40, help="Nombre de linies")

    p_wd = sub.add_parser("watchdog", help="Supervisa el model actiu i el reinicia si cau")
    p_wd.add_argument("model", nargs="?", default="", help="Carpeta del model a supervisar")
    p_wd.add_argument("--interval", type=float, default=3.0, help="Segons entre comprovacions")

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    registry = discover_models()

    if args.command == "ui":
        return run_web_ui(host=args.host, port=args.port, open_browser=not args.no_browser)
    if args.command == "live":
        return run_live_tui(
            window_minutes=max(args.window, 1),
            refresh_s=max(args.refresh, 1.0),
            req_limit=max(args.limit, 5),
        )
    if args.command == "list":
        return _cmd_list(registry)
    if args.command == "status":
        return _cmd_status(registry, window_minutes=max(args.window, 1))
    if args.command == "train":
        ok, msg = execute_transition(registry, args.model, train_first=True, timeout_s=args.timeout, force=True)
        print(msg)
        return 0 if ok else 1
    if args.command == "switch":
        ok, msg = execute_transition(registry, args.model, train_first=False, timeout_s=args.timeout, force=args.force)
        print(msg)
        return 0 if ok else 1
    if args.command == "start":
        serves = [p for p in find_running_processes(registry) if p.role == "serve"]
        if serves:
            active_desc = ", ".join(f"{p.folder} (PID {p.pid})" for p in serves)
            print(
                f"[AVIS] Ja hi ha model(s) en execucio: {active_desc}. "
                f"Usa 'switch {args.model}' o 'train {args.model}' per evitar conflictes."
            )
            return 1
        ok, msg = execute_transition(registry, args.model, train_first=False, timeout_s=args.timeout, force=True)
        print(msg)
        return 0 if ok else 1
    if args.command == "stop":
        _, details = stop_models(registry, model_filter=args.model or None)
        for line in details:
            print(f"  - {line}")
        return 0
    if args.command == "logs":
        return _cmd_logs(registry, target=args.model or None, n_lines=max(args.lines, 1))
    if args.command == "watchdog":
        return _cmd_watchdog(registry, target=args.model or None, interval_s=max(args.interval, 1.0))
    return 1
