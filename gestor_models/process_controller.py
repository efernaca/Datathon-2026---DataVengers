"""Process discovery, training orchestration, and zero-downtime hot-swap controller."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import psutil

from .config import (
    OPERATION_STATUS_FILE,
    REPO_ROOT,
    STATE_FILE,
    TRANSITIONS_FILE,
    atomic_write_json,
    read_json_file,
)
from .registry import ModelSpec, resolve_model

_READY_LOG_MARKERS = (
    "Snapshot v",
    "FastLoop active",
    "Submitted ",
    "Calypso 3.1 LIVE",
    "Live serving loop",
)

_OP_LOCK = threading.Lock()
_STATUS_LOCK = threading.Lock()
_CURRENT_OP: dict[str, Any] = {
    "state": "idle",
    "action": "",
    "target_folder": "",
    "target_name": "",
    "progress": 0,
    "message": "Sistema preparat.",
    "started_at": None,
    "updated_at": None,
}


@dataclass(slots=True)
class RunningProcess:
    """Metadata describing a live Python model process detected via psutil."""

    pid: int
    role: str  # "serve", "trainer", "watchdog"
    folder: str
    create_time: float
    rss_mb: float
    cmdline: list[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    sec = max(int(seconds), 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def tail_lines(path: Path, n: int = 40) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def get_operation_status() -> dict[str, Any]:
    with _STATUS_LOCK:
        if _CURRENT_OP["updated_at"] is None:
            saved = read_json_file(OPERATION_STATUS_FILE)
            if isinstance(saved, dict) and saved.get("state"):
                # Avoid stale "in-progress" state after UI process restarts
                if saved["state"] in {"self_check", "training", "starting", "switching"}:
                    return dict(_CURRENT_OP)
                return saved
        return dict(_CURRENT_OP)


def _set_operation_status(
    *,
    state: str,
    action: str,
    target_folder: str,
    target_name: str,
    progress: int,
    message: str,
    started_at: float | None = None,
) -> None:
    now = time.time()
    with _STATUS_LOCK:
        _CURRENT_OP.update(
            {
                "state": state,
                "action": action,
                "target_folder": target_folder,
                "target_name": target_name,
                "progress": progress,
                "message": message,
                "started_at": started_at or _CURRENT_OP.get("started_at") or now,
                "updated_at": now,
            }
        )
        snapshot = dict(_CURRENT_OP)
    try:
        atomic_write_json(OPERATION_STATUS_FILE, snapshot)
    except OSError:
        pass


def _append_transition(entry: dict[str, Any]) -> None:
    history = read_json_file(TRANSITIONS_FILE)
    if not isinstance(history, list):
        history = []
    history.append(entry)
    history = history[-50:]
    try:
        atomic_write_json(TRANSITIONS_FILE, history)
    except OSError:
        pass


def get_transitions(limit: int = 15) -> list[dict[str, Any]]:
    history = read_json_file(TRANSITIONS_FILE)
    if not isinstance(history, list):
        return []
    return list(reversed(history[-limit:]))


def _resolve_folder_from_args(cmdline: list[str], proc_cwd: str | None, known_folders: set[str]) -> str:
    base_cwd = Path(proc_cwd) if proc_cwd else REPO_ROOT
    for arg in cmdline[1:]:
        if arg.startswith("-"):
            continue
        cleaned = arg.strip().strip('"').strip("'")
        if not cleaned:
            continue
        for cand in (Path(cleaned), base_cwd / cleaned, REPO_ROOT / cleaned):
            try:
                rel = cand.resolve().relative_to(REPO_ROOT.resolve())
                if rel.parts and rel.parts[0] in known_folders:
                    return rel.parts[0]
            except (OSError, ValueError):
                pass
        norm_arg = cleaned.replace("\\", "/")
        for folder in known_folders:
            if norm_arg == folder or norm_arg.startswith(f"{folder}/") or f"/{folder}/" in norm_arg:
                return folder

    if proc_cwd:
        try:
            rel_cwd = Path(proc_cwd).resolve().relative_to(REPO_ROOT.resolve())
            if rel_cwd.parts and rel_cwd.parts[0] in known_folders:
                return rel_cwd.parts[0]
        except (OSError, ValueError):
            pass
    return "desconegut"


def _extract_process_info(
    cmdline: list[str],
    proc_cwd: str | None,
    known_folders: set[str],
) -> tuple[str | None, str | None]:
    """Classify Python processes into (serve|trainer|watchdog, folder)."""
    if not cmdline:
        return None, None
    joined = " ".join(cmdline).replace("\\", "/")
    if "model_manager.py" in joined or "gestor_models" in joined:
        if "watchdog" in cmdline:
            for token in cmdline:
                if token in known_folders:
                    return "watchdog", token
            return "watchdog", "auto"
        return None, None

    is_train_flag = "--train-only" in cmdline
    is_builder_or_train = any(
        arg.replace("\\", "/").endswith(("builder.py", "train.py"))
        or arg in {"src.builder", "src.train"}
        for arg in cmdline[1:]
    )
    if is_train_flag or is_builder_or_train:
        return "trainer", _resolve_folder_from_args(cmdline, proc_cwd, known_folders)

    is_serve = any(
        arg.replace("\\", "/").endswith("serve.py") or arg == "src.serve" or arg.endswith(".src.serve")
        for arg in cmdline[1:]
    )
    if is_serve:
        return "serve", _resolve_folder_from_args(cmdline, proc_cwd, known_folders)

    return None, None


def find_running_processes(registry: dict[str, ModelSpec]) -> list[RunningProcess]:
    """Inspect host Python processes via psutil for serve/trainer/watchdog workers."""
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
            role, folder = _extract_process_info(cmdline, proc_cwd, known)
            if role is None or folder is None:
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


def terminate_pid(pid: int, wait_s: float = 5.0) -> bool:
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


def record_active_state(spec: ModelSpec, pid: int) -> None:
    payload = {
        "active_folder": spec.folder,
        "model_name": spec.name,
        "model_version": spec.model_version,
        "pid": pid,
        "log_path": str(spec.log_path),
        "updated_at": _now_iso(),
    }
    atomic_write_json(STATE_FILE, payload)


def _run_self_check(spec: ModelSpec) -> tuple[bool, str]:
    if not spec.self_check_args:
        return True, "Sense self-check configurat"
    cmd = [sys.executable, str(spec.entrypoint), *spec.self_check_args]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=90,
        )
    except Exception as exc:
        return False, f"Error executant --self-check: {exc}"
    if res.returncode != 0:
        detail = (res.stderr or res.stdout or "").strip()[-300:]
        return False, f"--self-check ha fallat (codi {res.returncode}): {detail}"
    return True, "Pre-check superat correctament"


def _run_training_step(spec: ModelSpec, timeout_s: float) -> tuple[bool, str]:
    """Run offline pre-training command streaming output to spec.log_path while current server runs."""
    if not spec.train_cmd:
        return True, "El model construeix i entrena la snapshot directament en arrencar."

    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    banner = (
        f"\n===== [gestor_models] Training {spec.folder} ({spec.model_version}) "
        f"at {datetime.now().isoformat(timespec='seconds')} =====\n"
    )
    with spec.log_path.open("a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(banner)
        log_handle.flush()
        proc = subprocess.Popen(
            list(spec.train_cmd),
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )

    deadline = time.time() + max(timeout_s, 60.0)
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            if rc == 0:
                return True, f"Entrenament completat per '{spec.folder}'"
            tail = " | ".join(tail_lines(spec.log_path, 4))
            return False, f"Entrenament fallat (codi {rc}): {tail[-260:]}"
        time.sleep(0.5)

    terminate_pid(proc.pid, wait_s=3.0)
    return False, f"Timeout entrenant '{spec.folder}' despres de {timeout_s:.0f}s"


def _launch_serve_process(spec: ModelSpec) -> tuple[subprocess.Popen[Any], int]:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
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
        f"\n===== [gestor_models] Launching {spec.folder} ({spec.model_version}) "
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


def _wait_for_readiness(
    spec: ModelSpec,
    proc: subprocess.Popen[Any],
    new_pid: int,
    log_start_offset: int,
    timeout_s: float,
) -> tuple[bool, str]:
    """Wait until new process emits ready heartbeat or activity marker in logs."""
    deadline = time.time() + timeout_s
    started = time.time()
    legacy_min_wait = 18.0

    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            return False, f"El proces nou (PID {new_pid}) s'ha tancat amb codi {rc}."

        if spec.heartbeat_path is not None:
            hb = read_json_file(spec.heartbeat_path)
            if (
                isinstance(hb, dict)
                and hb.get("pid") == new_pid
                and hb.get("ready") is True
                and (time.time() - float(hb.get("last_heartbeat_ts", 0.0))) < 15.0
            ):
                snap_v = hb.get("snapshot_version", "?")
                return True, f"Heartbeat llest (PID {new_pid}, snapshot v{snap_v})"
        else:
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
                return True, f"Proces estable durant {legacy_min_wait:.0f}s"

        time.sleep(0.5)

    return False, f"Timeout de {timeout_s:.0f}s esperant que '{spec.folder}' quedes llest."


def _cutover_old_processes(running: list[RunningProcess], keep_pid: int) -> list[int]:
    """Stop all prior watchdogs, trainers, and serve processes cleanly after cutover."""
    stopped: list[int] = []
    ordered = sorted(
        (p for p in running if p.pid != keep_pid),
        key=lambda p: 0 if p.role == "watchdog" else (1 if p.role == "trainer" else 2),
    )
    for proc in ordered:
        wait_s = 3.0 if proc.role != "serve" else 5.0
        if terminate_pid(proc.pid, wait_s=wait_s):
            stopped.append(proc.pid)
    return stopped


def execute_transition(
    registry: dict[str, ModelSpec],
    target: str,
    *,
    train_first: bool,
    timeout_s: float = 180.0,
    force: bool = False,
) -> tuple[bool, str]:
    """Run optional pre-training + zero-downtime hot-swap + rollback safety."""
    try:
        spec = resolve_model(registry, target)
    except SystemExit as exc:
        return False, str(exc)

    if not _OP_LOCK.acquire(blocking=False):
        return False, "Ja hi ha un entrenament o canvi de model en curs. Espera que acabi."

    action = "train_and_activate" if train_first else "switch"
    t0 = time.time()
    running_before = find_running_processes(registry)
    old_serves = [p for p in running_before if p.role == "serve"]
    from_folders = sorted({p.folder for p in old_serves if p.folder})

    try:
        if not train_first and not force and len(old_serves) == 1 and old_serves[0].folder == spec.folder:
            record_active_state(spec, old_serves[0].pid)
            msg = (
                f"'{spec.folder}' ({spec.model_version}) ja es l'unic model actiu "
                f"(PID {old_serves[0].pid})."
            )
            _set_operation_status(
                state="completed",
                action=action,
                target_folder=spec.folder,
                target_name=spec.name,
                progress=100,
                message=msg,
                started_at=t0,
            )
            return True, msg

        _set_operation_status(
            state="self_check",
            action=action,
            target_folder=spec.folder,
            target_name=spec.name,
            progress=15,
            message=f"Verificant '{spec.folder}' (--self-check) mentre l'actual segueix servint...",
            started_at=t0,
        )
        ok_check, check_msg = _run_self_check(spec)
        if not ok_check:
            _set_operation_status(
                state="failed",
                action=action,
                target_folder=spec.folder,
                target_name=spec.name,
                progress=100,
                message=f"[ROLLBACK] {check_msg}",
                started_at=t0,
            )
            return False, check_msg

        if train_first:
            _set_operation_status(
                state="training",
                action=action,
                target_folder=spec.folder,
                target_name=spec.name,
                progress=40,
                message=f"Entrenant '{spec.folder}' en segon pla (el model en viu no s'atura)...",
                started_at=t0,
            )
            ok_train, train_msg = _run_training_step(spec, timeout_s=max(timeout_s, 180.0))
            if not ok_train:
                _set_operation_status(
                    state="failed",
                    action=action,
                    target_folder=spec.folder,
                    target_name=spec.name,
                    progress=100,
                    message=f"[ROLLBACK] {train_msg}",
                    started_at=t0,
                )
                return False, train_msg

        _set_operation_status(
            state="starting",
            action=action,
            target_folder=spec.folder,
            target_name=spec.name,
            progress=70,
            message=f"Engegant '{spec.folder}' ({spec.model_version}) i esperant confirmacio ready...",
            started_at=t0,
        )
        proc, log_offset = _launch_serve_process(spec)
        new_pid = proc.pid
        ok_ready, ready_msg = _wait_for_readiness(spec, proc, new_pid, log_offset, timeout_s=timeout_s)
        if not ok_ready:
            terminate_pid(new_pid, wait_s=3.0)
            tail = " | ".join(tail_lines(spec.log_path, 5))
            err = f"[ROLLBACK] {ready_msg}" + (f" ({tail[-220:]})" if tail else "")
            _set_operation_status(
                state="failed",
                action=action,
                target_folder=spec.folder,
                target_name=spec.name,
                progress=100,
                message=err,
                started_at=t0,
            )
            return False, err

        _set_operation_status(
            state="switching",
            action=action,
            target_folder=spec.folder,
            target_name=spec.name,
            progress=90,
            message=f"Tall net: activant '{spec.folder}' (PID {new_pid}) i aturant processos antics...",
            started_at=t0,
        )
        record_active_state(spec, new_pid)
        stopped_pids = _cutover_old_processes(find_running_processes(registry), keep_pid=new_pid)
        elapsed_s = round(time.time() - t0, 1)

        done_msg = (
            f"Model '{spec.folder}' ({spec.model_version}) actiu amb PID {new_pid} "
            f"en {elapsed_s}s ({ready_msg}). Aturats: {stopped_pids or 'cap'}."
        )
        _append_transition(
            {
                "ts": _now_iso(),
                "action": action,
                "from_folders": from_folders,
                "to_folder": spec.folder,
                "model_version": spec.model_version,
                "pid": new_pid,
                "stopped_pids": stopped_pids,
                "elapsed_s": elapsed_s,
                "status": "ok",
                "detail": ready_msg,
            }
        )
        _set_operation_status(
            state="completed",
            action=action,
            target_folder=spec.folder,
            target_name=spec.name,
            progress=100,
            message=done_msg,
            started_at=t0,
        )
        return True, done_msg
    finally:
        _OP_LOCK.release()


def start_background_operation(
    registry: dict[str, ModelSpec],
    target: str,
    *,
    train_first: bool,
    force: bool = False,
) -> tuple[bool, str]:
    """Launch execute_transition asynchronously for Web UI responsiveness."""
    try:
        spec = resolve_model(registry, target)
    except SystemExit as exc:
        return False, str(exc)

    if _OP_LOCK.locked():
        return False, "Ja hi ha una operacio en curs. Espera que acabi."

    _set_operation_status(
        state="starting",
        action="train_and_activate" if train_first else "switch",
        target_folder=spec.folder,
        target_name=spec.name,
        progress=5,
        message=f"Iniciant {'entrenament + activacio' if train_first else 'canvi rapid'} de '{spec.folder}'...",
        started_at=time.time(),
    )
    th = threading.Thread(
        target=execute_transition,
        args=(registry, spec.folder),
        kwargs={"train_first": train_first, "force": force},
        name=f"GestorOp-{spec.folder}",
        daemon=True,
    )
    th.start()
    return True, f"Operacio iniciada en segon pla per '{spec.folder}'."


def stop_models(registry: dict[str, ModelSpec], model_filter: str | None = None) -> tuple[bool, list[str]]:
    running = find_running_processes(registry)
    target_folder = resolve_model(registry, model_filter).folder if model_filter else None
    targets = [p for p in running if target_folder is None or p.folder == target_folder]
    if not targets:
        return False, ["No s'ha trobat cap proces actiu per aturar."]

    ordered = sorted(
        targets,
        key=lambda p: 0 if p.role == "watchdog" else (1 if p.role == "trainer" else 2),
    )
    out: list[str] = []
    for p in ordered:
        ok = terminate_pid(p.pid, wait_s=5.0)
        out.append(f"PID {p.pid} ({p.role}/{p.folder}) -> {'OK' if ok else 'FALLAT'}")
    return True, out


def get_runtime_overview(registry: dict[str, ModelSpec]) -> dict[str, Any]:
    """Return full process, heartbeat, active-model, and operation state snapshot."""
    procs = find_running_processes(registry)
    serve_by_folder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trainer_by_folder: dict[str, list[int]] = defaultdict(list)
    watchdogs: list[dict[str, Any]] = []
    now = time.time()

    all_procs_out: list[dict[str, Any]] = []
    for p in procs:
        up_s = max(now - p.create_time, 0.0)
        spec = registry.get(p.folder)
        hb = read_json_file(spec.heartbeat_path if spec else None)
        hb_out = None
        if isinstance(hb, dict) and hb.get("pid") == p.pid:
            hb_out = {
                "ready": bool(hb.get("ready")),
                "snapshot_version": hb.get("snapshot_version"),
                "active_products": hb.get("active_products"),
                "ml_alpha": hb.get("ml_alpha"),
                "regime": hb.get("regime"),
                "submitted_total": hb.get("submitted_total"),
                "last_batch_size": hb.get("last_batch_size"),
                "age_s": round(max(now - float(hb.get("last_heartbeat_ts", 0.0)), 0.0), 1),
            }
        item = {
            "pid": p.pid,
            "role": p.role,
            "folder": p.folder,
            "model_version": spec.model_version if spec else "desconegut",
            "uptime_s": int(up_s),
            "uptime_str": format_duration(up_s),
            "rss_mb": round(p.rss_mb, 1),
            "heartbeat": hb_out,
        }
        all_procs_out.append(item)
        if p.role == "serve":
            serve_by_folder[p.folder].append(item)
        elif p.role == "trainer":
            trainer_by_folder[p.folder].append(p.pid)
        elif p.role == "watchdog":
            watchdogs.append(item)

    state = read_json_file(STATE_FILE)
    if not isinstance(state, dict):
        state = None
    elif isinstance(state.get("pid"), int):
        state = dict(state)
        state["pid_alive"] = psutil.pid_exists(int(state["pid"]))

    models_out: list[dict[str, Any]] = []
    for folder, spec in registry.items():
        d = spec.to_dict()
        d["running_serves"] = serve_by_folder.get(folder, [])
        d["running_trainers"] = trainer_by_folder.get(folder, [])
        d["is_active"] = len(d["running_serves"]) > 0
        d["is_training"] = len(d["running_trainers"]) > 0
        models_out.append(d)

    return {
        "active_state": state,
        "processes": all_procs_out,
        "watchdogs": watchdogs,
        "models": models_out,
        "operation": get_operation_status(),
        "transitions": get_transitions(limit=12),
    }
