"""Root-level model manager for encapsulated models and legacy stages (Datathon 2026).

Usage:
    python model_manager.py list
    python model_manager.py status [--window 3]
    python model_manager.py switch calypso_3_1 [--timeout 120] [--force]
    python model_manager.py start calypso_3_1 [--timeout 120]
    python model_manager.py stop [--model calypso_3_1]
    python model_manager.py logs [calypso_3_1] [-n 40]
    python model_manager.py watchdog [calypso_3_1]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import psutil
import requests

REPO_ROOT = Path(__file__).resolve().parent
STATE_FILE = REPO_ROOT / ".active_model.json"

LEGACY_STAGE_ORDER: tuple[tuple[str, str, str], ...] = (
    ("etapa_4", "Legacy Stage 4", "calypso-2.1"),
    ("etapa_3", "Legacy Stage 3", "calypso-2.1"),
    ("etapa_2_version_especial", "Legacy Stage 2 Especial", "etapa2-especial"),
    ("etapa_2", "Legacy Stage 2", "calypso-2.1"),
    ("etapa_1_version_especial", "Legacy Stage 1 Especial", "etapa1-especial"),
    ("etapa_1", "Legacy Stage 1", "etapa1-hybrid-v1"),
)

_MODEL_VERSION_RE = re.compile(r'model_version\s*:\s*str\s*=\s*["\']([^"\']+)["\']')
_READY_LOG_MARKERS = (
    "Snapshot v",
    "FastLoop active",
    "Submitted ",
    "Calypso 3.1 LIVE",
    "Live serving loop",
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Descriptor for an encapsulated model folder or legacy stage directory."""

    folder: str
    name: str
    model_version: str
    kind: str  # "encapsulated" or "legacy"
    root_dir: Path
    entrypoint: Path
    self_check_args: tuple[str, ...]
    heartbeat_path: Path | None
    log_path: Path
    description: str


@dataclass(slots=True)
class RunningProcess:
    """Metadata describing a live Python process detected via psutil."""

    pid: int
    role: str  # "serve" or "watchdog"
    folder: str | None
    create_time: float
    rss_mb: float
    cmdline: list[str]


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except OSError:
                pass


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_json_file(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _load_env_credentials() -> tuple[str, str, str]:
    """Load RAWTREE_API_URL, TEAM_ID, RAWTREE_API_KEY from env or project env files."""
    search_dirs = [REPO_ROOT, REPO_ROOT / "calypso_3_1", REPO_ROOT / "datathon-fme-2026-main"]
    for directory in search_dirs:
        for fname in (".env", "group_API.env"):
            env_path = directory / fname
            if not env_path.is_file():
                continue
            try:
                lines = env_path.read_text(encoding="utf-8-sig").splitlines()
            except OSError:
                continue
            for raw in lines:
                line = raw.strip().replace("\r", "")
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip().removeprefix("export ").strip()
                if key:
                    os.environ.setdefault(key, val.strip().strip("'\""))

    api_url = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com").strip().rstrip("/")
    team_id = os.environ.get("TEAM_ID", "").strip()
    api_key = os.environ.get("RAWTREE_API_KEY", "").strip()
    return api_url, team_id, api_key


def _detect_legacy_model_version(folder_dir: Path, fallback: str) -> str:
    cfg_path = folder_dir / "src" / "config.py"
    if not cfg_path.is_file():
        return fallback
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return fallback
    match = _MODEL_VERSION_RE.search(text)
    return match.group(1).strip() if match else fallback


def discover_models() -> dict[str, ModelSpec]:
    """Discover encapsulated models (with model_manifest.json) and legacy stage folders."""
    registry: dict[str, ModelSpec] = {}

    for child in sorted(REPO_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        manifest_path = child / "model_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json_file(manifest_path) or {}
        entry_rel = str(manifest.get("entrypoint") or "serve.py")
        hb_rel = manifest.get("heartbeat_file")
        log_rel = str(manifest.get("log_file") or "logs/live.log")
        self_check_raw = manifest.get("self_check_args") or []
        registry[child.name] = ModelSpec(
            folder=child.name,
            name=str(manifest.get("name") or child.name),
            model_version=str(manifest.get("model_version") or child.name),
            kind="encapsulated",
            root_dir=child,
            entrypoint=child / entry_rel,
            self_check_args=tuple(str(a) for a in self_check_raw),
            heartbeat_path=(child / str(hb_rel)) if hb_rel else None,
            log_path=child / log_rel,
            description=str(manifest.get("description") or ""),
        )

    for folder, default_name, default_mv in LEGACY_STAGE_ORDER:
        if folder in registry:
            continue
        folder_dir = REPO_ROOT / folder
        entrypoint = folder_dir / "src" / "serve.py"
        if not entrypoint.is_file():
            continue
        hb_candidate = folder_dir / "cache" / "heartbeat.json"
        registry[folder] = ModelSpec(
            folder=folder,
            name=default_name,
            model_version=_detect_legacy_model_version(folder_dir, default_mv),
            kind="legacy",
            root_dir=folder_dir,
            entrypoint=entrypoint,
            self_check_args=(),
            heartbeat_path=hb_candidate if hb_candidate.parent.exists() else None,
            log_path=folder_dir / "logs" / "live.log",
            description=f"Carpeta històrica ({folder}/src/serve.py)",
        )

    return registry


def resolve_model(registry: dict[str, ModelSpec], query: str) -> ModelSpec:
    """Resolve user-provided folder, name, or unique model_version."""
    norm = query.strip().lower()
    for folder, spec in registry.items():
        if folder.lower() == norm:
            return spec

    canon = re.sub(r"[\s\-_]+", "", norm)
    name_matches = [
        spec
        for spec in registry.values()
        if re.sub(r"[\s\-_]+", "", spec.name.lower()) == canon
        or re.sub(r"[\s\-_]+", "", spec.folder.lower()) == canon
    ]
    if len(name_matches) == 1:
        return name_matches[0]

    mv_matches = [spec for spec in registry.values() if spec.model_version.lower() == norm]
    if len(mv_matches) == 1:
        return mv_matches[0]
    if len(mv_matches) > 1:
        folders = ", ".join(s.folder for s in mv_matches)
        raise SystemExit(
            f"Hi ha múltiples carpetes amb model_version='{query}': {folders}. "
            f"Especifica directament el nom de la carpeta."
        )

    available = ", ".join(registry.keys())
    raise SystemExit(f"Model '{query}' no trobat. Disponibles: {available}")


def _extract_folder_from_cmdline(
    cmdline: list[str],
    proc_cwd: str | None,
    known_folders: set[str],
) -> tuple[str | None, str | None]:
    """Identify (role, model_folder) from a Python process command line."""
    if not cmdline:
        return None, None
    joined = " ".join(cmdline).replace("\\", "/")
    if "model_manager.py" in joined:
        if "watchdog" in cmdline:
            for token in cmdline:
                if token in known_folders:
                    return "watchdog", token
            return "watchdog", None
        return None, None

    is_serve = any(
        arg.replace("\\", "/").endswith("serve.py") or arg == "src.serve" or arg.endswith(".src.serve")
        for arg in cmdline[1:]
    )
    if not is_serve:
        return None, None

    base_cwd = Path(proc_cwd) if proc_cwd else REPO_ROOT
    for arg in cmdline[1:]:
        if arg.startswith("-"):
            continue
        cleaned = arg.strip().strip('"').strip("'")
        if not cleaned:
            continue
        candidates = [Path(cleaned)]
        if not Path(cleaned).is_absolute():
            candidates.append(base_cwd / cleaned)
            candidates.append(REPO_ROOT / cleaned)
        for cand in candidates:
            try:
                rel = cand.resolve().relative_to(REPO_ROOT.resolve())
                if rel.parts and rel.parts[0] in known_folders:
                    return "serve", rel.parts[0]
            except (OSError, ValueError):
                pass
        for folder in known_folders:
            norm_arg = cleaned.replace("\\", "/")
            if norm_arg == folder or norm_arg.startswith(f"{folder}/") or f"/{folder}/" in norm_arg:
                return "serve", folder

    if proc_cwd:
        try:
            rel_cwd = Path(proc_cwd).resolve().relative_to(REPO_ROOT.resolve())
            if rel_cwd.parts and rel_cwd.parts[0] in known_folders:
                return "serve", rel_cwd.parts[0]
        except (OSError, ValueError):
            pass

    return "serve", "desconegut"


def find_running_processes(registry: dict[str, ModelSpec]) -> list[RunningProcess]:
    """Inspect host processes via psutil for live serve.py and watchdog processes."""
    known = set(registry.keys())
    found: list[RunningProcess] = []
    my_pid = os.getpid()

    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            pid = int(proc.info["pid"])
            if pid == my_pid:
                continue
            pname = str(proc.info.get("name") or "").lower()
            if "python" not in pname:
                continue
            cmdline = list(proc.info.get("cmdline") or ())
            try:
                proc_cwd = proc.cwd()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
                proc_cwd = None
            role, folder = _extract_folder_from_cmdline(cmdline, proc_cwd, known)
            if role is None:
                continue
            rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
            ctime = float(proc.info.get("create_time") or time.time())
            found.append(
                RunningProcess(
                    pid=pid,
                    role=role,
                    folder=folder,
                    create_time=ctime,
                    rss_mb=rss_mb,
                    cmdline=cmdline,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    found.sort(key=lambda r: r.create_time)
    return found


def _terminate_pid(pid: int, wait_s: float = 5.0) -> bool:
    """Terminate a process gracefully, escalating to kill() only if needed."""
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    try:
        proc.terminate()
        proc.wait(timeout=wait_s)
        return True
    except psutil.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=3.0)
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _format_duration(seconds: float) -> str:
    sec = max(int(seconds), 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _tail_lines(path: Path, n: int = 30) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def _query_rawtree_live(window_minutes: int) -> tuple[list[dict[str, Any]], str | None]:
    """Query RawTree feedback joined with submissions by model_version."""
    api_url, team_id, api_key = _load_env_credentials()
    if not team_id or not api_key:
        return [], "Falten TEAM_ID o RAWTREE_API_KEY a .env / group_API.env"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    sql_fb = f"""
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
    try:
        r_fb = requests.post(f"{api_url}/v1/query", headers=headers, json={"sql": sql_fb}, timeout=5.0)
        if not r_fb.ok:
            return [], f"Error HTTP RawTree ({r_fb.status_code}): {r_fb.text[:160]}"
        fb_rows = r_fb.json().get("data", [])
    except Exception as exc:
        return [], f"No s'ha pogut connectar a RawTree: {exc}"

    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "on_time": 0, "clicks": 0, "purchases": 0, "rev": 0.0, "lats": []}
    )
    for row in fb_rows:
        mv = str(row.get("model_version") or "fallback_or_unknown").strip() or "fallback_or_unknown"
        b = buckets[mv]
        b["n"] += 1
        lat = _safe_float(row.get("latency_ms"), -1.0)
        if lat >= 0.0:
            b["lats"].append(lat)
        served_by = str(row.get("served_by") or "").strip().lower()
        if served_by == "team" or (not served_by and mv != "fallback_or_unknown" and 0.0 <= lat <= 3000.0):
            b["on_time"] += 1
        if str(row.get("clicked") or "").strip():
            b["clicks"] += 1
        if str(row.get("purchased") or "").strip():
            b["purchases"] += 1
        b["rev"] += max(_safe_float(row.get("revenue_eur"), 0.0), 0.0)

    fb_summary: list[dict[str, Any]] = []
    for mv, b in buckets.items():
        n = b["n"]
        lats = sorted(b["lats"])
        p50 = lats[int(0.50 * (len(lats) - 1))] if lats else 0.0
        p95 = lats[int(0.95 * (len(lats) - 1))] if lats else 0.0
        fb_summary.append(
            {
                "model_version": mv,
                "reqs": n,
                "on_time_pct": round(100.0 * b["on_time"] / max(n, 1), 2),
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "ctr_pct": round(100.0 * b["clicks"] / max(n, 1), 2),
                "cvr_pct": round(100.0 * b["purchases"] / max(n, 1), 2),
                "eur_per_req": round(b["rev"] / max(n, 1), 4),
            }
        )
    fb_summary.sort(key=lambda x: x["reqs"], reverse=True)
    return fb_summary, None


def cmd_list(registry: dict[str, ModelSpec]) -> int:
    procs = [p for p in find_running_processes(registry) if p.role == "serve"]
    running_by_folder: dict[str, list[int]] = defaultdict(list)
    for p in procs:
        if p.folder:
            running_by_folder[p.folder].append(p.pid)

    state = _read_json_file(STATE_FILE) or {}
    recorded_folder = state.get("active_folder")

    print("Models disponibles al projecte:")
    print(f"  {'CARPETA':<26} {'TIPUS':<14} {'MODEL_VERSION':<18} {'ESTAT'}")
    print("  " + "-" * 78)
    for folder, spec in registry.items():
        tag = "[ENCAPSULAT]" if spec.kind == "encapsulated" else "[HISTORIC]"
        pids = running_by_folder.get(folder, [])
        if pids:
            status_txt = f"ACTIU EN VIU (PID {','.join(str(x) for x in pids)})"
        elif folder == recorded_folder:
            status_txt = "Ultim registrat (aturat)"
        else:
            status_txt = "Aturat"
        print(f"  {folder:<26} {tag:<14} {spec.model_version:<18} {status_txt}")
    return 0


def cmd_status(registry: dict[str, ModelSpec], window_minutes: int) -> int:
    procs = find_running_processes(registry)
    serve_procs = [p for p in procs if p.role == "serve"]
    watchdogs = [p for p in procs if p.role == "watchdog"]
    state = _read_json_file(STATE_FILE)

    print("=== ESTAT DEL SERVEI EN VIU (LOCAL + RAWTREE) ===")
    if state:
        pid_alive = isinstance(state.get("pid"), int) and psutil.pid_exists(int(state["pid"]))
        alive_lbl = "ACTIU" if pid_alive else "NO ACTIU"
        print(
            f"Registre (.active_model.json): {state.get('active_folder')} "
            f"({state.get('model_name')} / model_version={state.get('model_version')}, "
            f"PID={state.get('pid')} [{alive_lbl}])"
        )
    else:
        print("Registre (.active_model.json): encara no inicialitzat")

    print("\n1) Processos locals detectats (psutil):")
    if not serve_procs:
        print("   Cap proces serve.py en execucio ara mateix.")
    for p in serve_procs:
        spec = registry.get(p.folder or "")
        mv = spec.model_version if spec else "desconegut"
        up = _format_duration(time.time() - p.create_time)
        print(
            f"   - PID {p.pid:<7} | carpeta={p.folder:<24} | model_version={mv:<16} | "
            f"uptime={up:<10} | RAM={p.rss_mb:.1f} MB"
        )
        hb = _read_json_file(spec.heartbeat_path if spec else None)
        if hb and hb.get("pid") == p.pid:
            age_s = max(time.time() - float(hb.get("last_heartbeat_ts", 0.0)), 0.0)
            print(
                f"     Heartbeat: ready={hb.get('ready')} | snap=v{hb.get('snapshot_version')} | "
                f"ml_alpha={hb.get('ml_alpha')} | regime={hb.get('regime')} | "
                f"enviats={hb.get('submitted_total')} | fa {age_s:.1f}s"
            )
    if watchdogs:
        for w in watchdogs:
            print(f"   - Watchdog PID {w.pid} vigilant '{w.folder or 'auto'}'")

    print(f"\n2) Telemetria RawTree en directe (ultims {window_minutes} min):")
    fb_summary, err = _query_rawtree_live(window_minutes)
    if err:
        print(f"   {err}")
    elif fb_summary:
        print("   Feedback real (team_XX.feedback):")
        print(
            f"     {'MODEL_VERSION':<20} {'REQS':>6} {'ON_TIME%':>9} "
            f"{'P50_MS':>8} {'P95_MS':>8} {'CTR%':>7} {'CVR%':>7} {'EUR/REQ':>8}"
        )
        for row in fb_summary:
            print(
                f"     {row['model_version']:<20} {row['reqs']:>6} "
                f"{row['on_time_pct']:>8.2f}% {row['p50_ms']:>8.1f} {row['p95_ms']:>8.1f} "
                f"{row['ctr_pct']:>6.2f}% {row['cvr_pct']:>6.2f}% {row['eur_per_req']:>8.4f}"
            )
    else:
        print(f"   Cap resposta registrada als ultims {window_minutes} minuts.")
    return 0


def _run_self_check(spec: ModelSpec) -> bool:
    if not spec.self_check_args:
        return True
    cmd = [sys.executable, str(spec.entrypoint), *spec.self_check_args]
    print(f"[PRE-CHECK] Verificant '{spec.folder}' abans de tocar el model en viu...")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, timeout=90)
    if res.returncode != 0:
        print(f"[ABORTAT] --self-check ha fallat (codi {res.returncode}). El model actual NO s'ha tocat.")
        return False
    return True


def _wait_for_readiness(
    spec: ModelSpec,
    proc: subprocess.Popen[Any],
    new_pid: int,
    log_start_offset: int,
    timeout_s: float,
) -> tuple[bool, str]:
    """Wait until new process reports ready heartbeat (or healthy live activity) with rollback on error."""
    deadline = time.time() + timeout_s
    started = time.time()
    legacy_min_wait = 18.0

    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            return False, f"El proces nou (PID {new_pid}) s'ha tancat inesperadament amb codi {rc}."

        if spec.heartbeat_path is not None:
            hb = _read_json_file(spec.heartbeat_path)
            if (
                hb is not None
                and hb.get("pid") == new_pid
                and hb.get("ready") is True
                and (time.time() - float(hb.get("last_heartbeat_ts", 0.0))) < 15.0
            ):
                snap_v = hb.get("snapshot_version", "?")
                return True, f"Heartbeat llest (PID {new_pid}, snapshot v{snap_v})"
        else:
            # Legacy stages without JSON heartbeat: inspect incremental live.log output
            try:
                if spec.log_path.is_file() and spec.log_path.stat().st_size > log_start_offset:
                    with spec.log_path.open("rb") as fh:
                        fh.seek(log_start_offset)
                        new_txt = fh.read().decode("utf-8", errors="replace")
                    for marker in _READY_LOG_MARKERS:
                        if marker in new_txt:
                            return True, f"Activitat confirmada al log ('{marker}')"
            except OSError:
                pass
            if time.time() - started >= legacy_min_wait and psutil.pid_exists(new_pid):
                return True, f"Proces historic estable durant {legacy_min_wait:.0f}s"

        time.sleep(0.5)

    return False, f"Timeout de {timeout_s:.0f}s esperant que '{spec.folder}' quedes llest."


def _launch_serve_process(spec: ModelSpec) -> tuple[subprocess.Popen[Any], int]:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_offset = spec.log_path.stat().st_size if spec.log_path.is_file() else 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    creationflags = 0
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )

    banner = (
        f"\n===== [model_manager] Launching {spec.folder} ({spec.model_version}) "
        f"at {datetime.now().isoformat(timespec='seconds')} =====\n"
    )
    with spec.log_path.open("a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(banner)
        log_handle.flush()
        log_offset = spec.log_path.stat().st_size
        proc = subprocess.Popen(
            [sys.executable, str(spec.entrypoint)],
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            creationflags=creationflags,
        )
    return proc, log_offset


def _record_active_state(spec: ModelSpec, pid: int) -> None:
    payload = {
        "active_folder": spec.folder,
        "model_name": spec.name,
        "model_version": spec.model_version,
        "pid": pid,
        "log_path": str(spec.log_path),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _atomic_write_json(STATE_FILE, payload)


def cmd_switch(registry: dict[str, ModelSpec], target: str, timeout_s: float, force: bool) -> int:
    spec = resolve_model(registry, target)
    running = find_running_processes(registry)
    old_serves = [p for p in running if p.role == "serve"]
    old_watchdogs = [p for p in running if p.role == "watchdog"]

    if not force and len(old_serves) == 1 and old_serves[0].folder == spec.folder:
        print(
            f"[JA ACTIU] '{spec.folder}' ({spec.model_version}) ja es l'unic model en execucio "
            f"(PID {old_serves[0].pid}). Usa --force si vols reiniciar-lo."
        )
        _record_active_state(spec, old_serves[0].pid)
        return 0

    if not _run_self_check(spec):
        return 1

    print(
        f"[HOT-SWAP] Engegant '{spec.folder}' ({spec.model_version}) en segon pla "
        f"mentre {len(old_serves)} proces(os) antic(s) segueixen servint..."
    )
    t0 = time.perf_counter()
    proc, log_offset = _launch_serve_process(spec)
    new_pid = proc.pid

    ready, reason = _wait_for_readiness(spec, proc, new_pid, log_offset, timeout_s=timeout_s)
    if not ready:
        print(f"[ROLLBACK] {reason}")
        _terminate_pid(new_pid, wait_s=3.0)
        tail = _tail_lines(spec.log_path, 25)
        if tail:
            print("--- Ultimes linies del log del proces fallit ---")
            for line in tail:
                print(f"  {line}")
        print("[ROLLBACK COMPLETAT] El model anterior s'ha mantingut intacte.")
        return 1

    _record_active_state(spec, new_pid)

    stopped_pids: list[int] = []
    for wd in old_watchdogs:
        if wd.pid != new_pid and _terminate_pid(wd.pid, wait_s=3.0):
            stopped_pids.append(wd.pid)
    for old in old_serves:
        if old.pid != new_pid and _terminate_pid(old.pid, wait_s=5.0):
            stopped_pids.append(old.pid)

    elapsed = time.perf_counter() - t0
    print(f"[CANVI COMPLETAT en {elapsed:.1f}s]")
    print(f"  Model actiu ara : {spec.folder} ({spec.name} | model_version={spec.model_version})")
    print(f"  Nou PID en viu  : {new_pid}")
    print(f"  Motiu llest     : {reason}")
    print(f"  Logs            : {spec.log_path}")
    if stopped_pids:
        print(f"  PIDs antics parats: {', '.join(str(x) for x in stopped_pids)}")
    return 0


def cmd_start(registry: dict[str, ModelSpec], target: str, timeout_s: float) -> int:
    running_serves = [p for p in find_running_processes(registry) if p.role == "serve"]
    if running_serves:
        active_desc = ", ".join(f"{p.folder} (PID {p.pid})" for p in running_serves)
        print(
            f"[AVIS] Ja hi ha model(s) en execucio: {active_desc}. "
            f"Fes servir 'python model_manager.py switch {target}' per canviar sense talls."
        )
        return 1
    return cmd_switch(registry, target, timeout_s=timeout_s, force=True)


def cmd_stop(registry: dict[str, ModelSpec], model_filter: str | None) -> int:
    running = find_running_processes(registry)
    target_folder = resolve_model(registry, model_filter).folder if model_filter else None

    targets = [
        p
        for p in running
        if target_folder is None or p.folder == target_folder
    ]
    if not targets:
        print("No s'ha trobat cap proces actiu per aturar.")
        return 0

    # Stop watchdogs before serve processes so they do not respawn
    watchdogs = [p for p in targets if p.role == "watchdog"]
    serves = [p for p in targets if p.role == "serve"]
    stopped: list[str] = []
    for p in [*watchdogs, *serves]:
        ok = _terminate_pid(p.pid, wait_s=5.0)
        stopped.append(f"PID {p.pid} ({p.role}/{p.folder}) -> {'OK' if ok else 'FALLAT'}")

    print("Processos aturats:")
    for line in stopped:
        print(f"  - {line}")
    return 0


def cmd_logs(registry: dict[str, ModelSpec], target: str | None, n_lines: int) -> int:
    if target:
        spec = resolve_model(registry, target)
    else:
        state = _read_json_file(STATE_FILE) or {}
        folder = str(state.get("active_folder") or "")
        if folder and folder in registry:
            spec = registry[folder]
        else:
            running = [p for p in find_running_processes(registry) if p.role == "serve"]
            if running and running[0].folder in registry:
                spec = registry[running[0].folder]
            else:
                spec = next(iter(registry.values()))

    lines = _tail_lines(spec.log_path, n=n_lines)
    if not lines:
        print(f"No hi ha logs disponibles a {spec.log_path}")
        return 0
    print(f"=== Ultimes {len(lines)} linies de {spec.log_path} ===")
    for line in lines:
        print(line)
    return 0


def cmd_watchdog(registry: dict[str, ModelSpec], target: str | None, interval_s: float) -> int:
    if target:
        spec = resolve_model(registry, target)
    else:
        state = _read_json_file(STATE_FILE) or {}
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
            if not needs_restart and spec.heartbeat_path is not None:
                hb = _read_json_file(spec.heartbeat_path)
                if hb and hb.get("pid") == serves[0].pid and hb.get("ready") is True:
                    age = time.time() - float(hb.get("last_heartbeat_ts", 0.0))
                    if age > 45.0:
                        print(f"[WATCHDOG] Heartbeat congelat ({age:.0f}s). Reiniciant PID {serves[0].pid}...")
                        _terminate_pid(serves[0].pid, wait_s=4.0)
                        needs_restart = True

            if needs_restart:
                print(f"[WATCHDOG] Engegant '{spec.folder}'...")
                proc, log_offset = _launch_serve_process(spec)
                ok, msg = _wait_for_readiness(spec, proc, proc.pid, log_offset, timeout_s=120.0)
                if ok:
                    _record_active_state(spec, proc.pid)
                    print(f"[WATCHDOG] Actiu PID {proc.pid}: {msg}")
                else:
                    print(f"[WATCHDOG] Error d'arrencada: {msg}")
                    _terminate_pid(proc.pid, wait_s=3.0)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("[WATCHDOG] Aturat per l'usuari.")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gestor de models per al projecte RawTree")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Llista tots els models encapsulats i carpetes historiques")

    p_status = sub.add_parser("status", help="Mostra el proces actiu i la telemetria RawTree en directe")
    p_status.add_argument("--window", type=int, default=3, help="Finestra de minuts RawTree (per defecte 3)")

    p_switch = sub.add_parser("switch", help="Canvia a un altre model amb Zero-Downtime i rollback automatic")
    p_switch.add_argument("model", help="Nom de la carpeta o model (ex: calypso_3_1)")
    p_switch.add_argument("--timeout", type=float, default=120.0, help="Temps maxim d'espera fins llest (s)")
    p_switch.add_argument("--force", action="store_true", help="Forca reinici fins i tot si ja es l'unic actiu")

    p_start = sub.add_parser("start", help="Engega un model si no n'hi ha cap altre actiu")
    p_start.add_argument("model", help="Nom de la carpeta o model (ex: calypso_3_1)")
    p_start.add_argument("--timeout", type=float, default=120.0, help="Temps maxim d'espera fins llest (s)")

    p_stop = sub.add_parser("stop", help="Atura els processos serve.py en execucio")
    p_stop.add_argument("--model", type=str, default="", help="Opcional: nomes atura aquesta carpeta")

    p_logs = sub.add_parser("logs", help="Mostra les ultimes linies del log del model")
    p_logs.add_argument("model", nargs="?", default="", help="Carpeta del model (opcional)")
    p_logs.add_argument("-n", "--lines", type=int, default=40, help="Nombre de linies a mostrar")

    p_wd = sub.add_parser("watchdog", help="Supervisa el model actiu i el reinicia si cau")
    p_wd.add_argument("model", nargs="?", default="", help="Carpeta del model a supervisar")
    p_wd.add_argument("--interval", type=float, default=3.0, help="Segons entre comprovacions")

    return parser


def main() -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    registry = discover_models()

    if args.command == "list":
        return cmd_list(registry)
    if args.command == "status":
        return cmd_status(registry, window_minutes=max(args.window, 1))
    if args.command == "switch":
        return cmd_switch(registry, args.model, timeout_s=args.timeout, force=args.force)
    if args.command == "start":
        return cmd_start(registry, args.model, timeout_s=args.timeout)
    if args.command == "stop":
        return cmd_stop(registry, model_filter=args.model or None)
    if args.command == "logs":
        return cmd_logs(registry, target=args.model or None, n_lines=max(args.lines, 1))
    if args.command == "watchdog":
        return cmd_watchdog(registry, target=args.model or None, interval_s=max(args.interval, 1.0))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
