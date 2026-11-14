"""4.2 · Aprender los pesos de las señales con el feedback propio (y por segmento, 4.3).

    python -m src.learner                 # una vez (el constructor lo lanza cada hora)

1. Lee el registro de señales que escribe el servidor (`logs/signals_*.jsonl`): por cada
   lista servida, las señales de los 10 productos mostrados, su posición, el segmento y
   los interruptores.
2. Une con `team.feedback` por `request_id` (solo `served_by = 'team'`): etiqueta = clic.
3. Por segmento con datos suficientes (si no, global): regresión logística
   `clic ~ señales + posición`. Los coeficientes de las señales son los pesos nuevos.
4. Solo se aceptan si mejoran la log-loss de validación (partición por hash de
   `request_id`) frente a los pesos actuales recalibrados; si no, se conservan.
5. Escribe `state/weights.json` (el servidor lo recarga solo) y una copia en `scratch`.

Necesita la aleatorización de 4.1 (interruptores y barajado) para no confirmar sus propias
decisiones: con el barajado, la posición deja de estar confundida con la puntuación.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
import glob
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import parse_ts
from .config import SEGMENTS, SIGNALS, Config
from .policy import Weights, unit_hash

logger = logging.getLogger("e2esp.learner")

SQL_FEEDBACK_SINCE = """
SELECT request_id::String AS request_id, served_by::String AS served_by,
       ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased,
       ifNull(revenue_eur::Float64, 0) AS revenue_eur
FROM {team}.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '{since}'
"""


def load_signal_logs(cfg: Config, hours: float | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t_min = time.time() - hours * 3600 if hours else 0.0
    for path in sorted(glob.glob(str(Path(cfg.log_dir) / "signals_*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if t_min and parse_ts(r.get("ts")) < t_min:
                    continue
                rows.append(r)
    return rows


def join_feedback(client: Any, rows: list[dict[str, Any]], team: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    t0 = min(parse_ts(r.get("ts")) for r in rows) - 60
    from datetime import datetime, timezone

    since = datetime.fromtimestamp(max(t0, 0), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    fb = {r["request_id"]: r for r in client.query(SQL_FEEDBACK_SINCE.format(team=team, since=since), timeout=300)}
    out = []
    for r in rows:
        f = fb.get(r["request_id"])
        if f is None or f.get("served_by") != "team":
            continue
        out.append({**r, "clicked": f.get("clicked") or "", "purchased": f.get("purchased") or "",
                    "revenue": float(f.get("revenue_eur") or 0)})
    return out


def to_table(joined: list[dict[str, Any]], target: str = "clicked") -> dict[str, np.ndarray]:
    X, slot, y, seg, rid = [], [], [], [], []
    for r in joined:
        lab = r.get(target) or ""
        for k, (pid, sig) in enumerate(zip(r["items"], r["signals"])):
            X.append(sig)
            slot.append(k)
            y.append(pid == lab)
            seg.append(r["segment"])
            rid.append(r["request_id"])
    return {"X": np.array(X, dtype=float).reshape(-1, len(SIGNALS)), "slot": np.array(slot),
            "y": np.array(y, dtype=bool), "seg": np.array(seg), "rid": np.array(rid)}


def _design(X: np.ndarray, slot: np.ndarray) -> np.ndarray:
    D = np.zeros((len(slot), 9))
    for k in range(1, 10):
        D[:, k - 1] = slot == k
    return np.hstack([X, D])


def _fit(Xd: np.ndarray, y: np.ndarray, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(Xd, y)
    return clf


def _logloss(clf, Xd, y) -> float:
    from sklearn.metrics import log_loss

    return float(log_loss(y, np.clip(clf.predict_proba(Xd)[:, 1], 1e-7, 1 - 1e-7), labels=[False, True]))


def learn(t: dict[str, np.ndarray], current: Weights, cfg: Config) -> tuple[Weights, list[dict[str, Any]]]:
    is_val = np.array([unit_hash(f"{r}:val") < cfg.learn_val_share for r in t["rid"]])
    report = []
    new_by_seg: dict[str, dict[str, float]] = {}
    groups = {"global": np.ones(len(t["y"]), bool), **{s: t["seg"] == s for s in SEGMENTS}}
    global_w: dict[str, float] | None = None
    for seg, m in groups.items():
        tr, va = m & ~is_val, m & is_val
        n_pos = int(t["y"][tr].sum())
        if (seg != "global" and tr.sum() < cfg.learn_min_rows_segment) or n_pos < 30 or t["y"][va].sum() < 10:
            report.append({"segmento": seg, "filas": int(m.sum()), "clics": int(t["y"][m].sum()),
                           "decisión": "pocos datos → global/actuales"})
            continue
        Xd_tr, Xd_va = _design(t["X"][tr], t["slot"][tr]), _design(t["X"][va], t["slot"][va])
        new = _fit(Xd_tr, t["y"][tr])
        wcur = current.vector(seg if seg != "global" else SEGMENTS[0])
        s_tr = (t["X"][tr] @ wcur)[:, None]
        s_va = (t["X"][va] @ wcur)[:, None]
        base = _fit(_design(s_tr, t["slot"][tr]), t["y"][tr])
        ll_new, ll_base = _logloss(new, Xd_va, t["y"][va]), _logloss(base, _design(s_va, t["slot"][va]), t["y"][va])
        w = {s: float(c) for s, c in zip(SIGNALS, new.coef_[0][: len(SIGNALS)])}
        accept = ll_new < ll_base
        report.append({"segmento": seg, "filas": int(m.sum()), "clics": int(t["y"][m].sum()),
                       "logloss_actual": round(ll_base, 5), "logloss_nuevo": round(ll_new, 5),
                       "decisión": "ACEPTADO" if accept else "se mantienen los actuales",
                       "pesos": {k: round(v, 3) for k, v in w.items()},
                       "atencion_slot": [1.0] + [round(float(np.exp(c)), 3) for c in new.coef_[0][len(SIGNALS):]]})
        if accept:
            if seg == "global":
                global_w = w
            else:
                new_by_seg[seg] = w
    by_seg = {}
    for seg in SEGMENTS:
        if seg in new_by_seg:
            by_seg[seg] = new_by_seg[seg]
        elif global_w is not None:
            by_seg[seg] = global_w
        else:
            by_seg[seg] = current.by_segment.get(seg, current.by_segment.get("global", {}))
    changed = bool(new_by_seg or global_w)
    return Weights(by_seg, f"w{int(time.time())}" if changed else current.version), report


def write_report(report: list[dict[str, Any]], path: Path, n_lists: int) -> None:
    lines = [f"# Aprendizaje de pesos (4.2) · {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Listas con feedback: {n_lists}. Objetivo: clic. Validación: 20 % por hash de request_id.", ""]
    for r in report:
        lines.append(f"## {r['segmento']}: {r['decisión']}")
        lines.append("")
        lines.append(f"filas={r['filas']} clics={r['clics']}" + (
            f" · log-loss actual {r['logloss_actual']} → nuevo {r['logloss_nuevo']}" if "logloss_nuevo" in r else ""))
        if "pesos" in r:
            lines += ["", "| señal | peso |", "|---|---|"] + [f"| {k} | {v} |" for k, v in r["pesos"].items()]
            lines += ["", f"Atención relativa por slot (exp(coef)): {r['atencion_slot']}"]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_learner(client: Any, cfg: Config, hours: float | None = None, publish: bool = True) -> Weights | None:
    rows = load_signal_logs(cfg, hours)
    joined = join_feedback(client, rows, client.team_id)
    logger.info("Registro: %d listas, %d con feedback 'team'", len(rows), len(joined))
    if len(joined) < 500:
        logger.info("Menos de 500 listas con feedback: no se reentrena todavía")
        return None
    wpath = Path(cfg.state_dir) / "weights.json"
    current = Weights.load(wpath)
    new, report = learn(to_table(joined), current, cfg)
    write_report(report, Path(cfg.log_dir) / "learn_report.md", len(joined))
    if new.version != current.version:
        new.save(wpath)
        logger.info("Pesos nuevos guardados: %s", new.version)
        if publish:
            try:
                client.push_scratch([{"kind": "weights", "version": new.version, "weights": new.by_segment,
                                      "metrics": report}])
            except Exception as exc:  # noqa: BLE001
                logger.warning("scratch weights: %s", exc)
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=None, help="solo las últimas N horas del registro")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .rawtree_client import RawTreeClient

    cfg = Config.from_env()
    run_learner(RawTreeClient(cfg.api_url, cfg.api_key, cfg.team_id), cfg, args.hours)


if __name__ == "__main__":
    main()
