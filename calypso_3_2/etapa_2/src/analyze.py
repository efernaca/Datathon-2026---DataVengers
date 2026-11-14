"""Análisis del experimento factorial (4.1) y del ensayo de la final (4.4).

    python -m src.analyze            # lee logs/signals_*.jsonl + team.feedback

Produce `experiments/EFFECTS.md`, `experiments/effects.csv` y `state/switches.json`
(lo que `serve.py --freeze` encenderá el domingo):

* Efecto de cada interruptor sobre el CTR por lista (y la CVR): diferencia de medias con
  su error estándar (válida porque cada interruptor se asigna al azar e independiente) y
  regresión logística `clic ~ interruptores + segmento` con intervalos (IRLS).
* Por segmento (sesión vacía / con vistas, nuevo / conocido).
* Curva de atención medida con el 10 % barajado (CTR por slot) → `eda/attention.json`.
* Modo final: caída de CTR al tratar como nuevo a un usuario conocido.
* N8: listas con producto nuevo, posición media y CTR de esos productos.
Regla de decisión: encender si el IC95 % del efecto está por encima de 0, apagar si está
por debajo; si no es concluyente, valor por defecto (todo encendido salvo `rev`).
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ETAPA_DIR, SWITCHES, Config
from .learner import join_feedback, load_signal_logs

logger = logging.getLogger("e2esp.analyze")
DEFAULT_SWITCHES = {s: (0 if s == "rev" else 1) for s in SWITCHES}


def irls_logit(X: np.ndarray, y: np.ndarray, iters: int = 25, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Regresión logística por Newton/IRLS; devuelve coeficientes y errores estándar."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
    H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
    return beta, np.sqrt(np.diag(np.linalg.inv(H)))


def lists_frame(joined: list[dict[str, Any]]) -> pd.DataFrame:
    recs = []
    for r in joined:
        sw = r.get("switches", {})
        click_slot = r["items"].index(r["clicked"]) + 1 if r.get("clicked") in r["items"] else 0
        new_pos = r.get("new_pos") or []
        recs.append({"request_id": r["request_id"], "segment": r["segment"], "shuf": r.get("shuf", 0),
                     "fin": r.get("fin", 0), "new_user": r.get("new_user", 0),
                     "click": int(bool(r.get("clicked"))), "buy": int(bool(r.get("purchased"))),
                     "revenue": r.get("revenue", 0.0), "click_slot": click_slot,
                     "has_new": int(bool(new_pos)), "new_pos_mean": float(np.mean(new_pos)) if new_pos else np.nan,
                     "click_on_new": int(click_slot in new_pos) if new_pos else 0,
                     **{s: int(sw.get(s, DEFAULT_SWITCHES[s])) for s in SWITCHES}})
    return pd.DataFrame(recs)


def switch_effects(df: pd.DataFrame, metric: str = "click") -> pd.DataFrame:
    rows = []
    d0 = df[(df["shuf"] == 0)]
    segs = {"TODOS": d0, "con_sesion": d0[d0["segment"].str.endswith("_sesion") & ~d0["segment"].str.contains("sin")],
            "sin_sesion": d0[d0["segment"].str.contains("sin_sesion")],
            "nuevo": d0[d0["segment"].str.startswith("nuevo")], "conocido": d0[d0["segment"].str.startswith("conocido")],
            "producto_nuevo": d0[d0["segment"] == "producto_nuevo"]}
    for seg, d in segs.items():
        for s in SWITCHES:
            a, b = d[d[s] == 1][metric], d[d[s] == 0][metric]
            if len(a) < 30 or len(b) < 30:
                continue
            p1, p0 = a.mean(), b.mean()
            se = math.sqrt(p1 * (1 - p1) / len(a) + p0 * (1 - p0) / len(b)) or 1e-9
            rows.append({"segmento": seg, "interruptor": s, "n_on": len(a), "n_off": len(b),
                         f"{metric}_on": p1, f"{metric}_off": p0, "efecto": p1 - p0,
                         "ic95_lo": p1 - p0 - 1.96 * se, "ic95_hi": p1 - p0 + 1.96 * se,
                         "efecto_rel": (p1 - p0) / p0 if p0 > 0 else np.nan})
    return pd.DataFrame(rows)


def logistic_effects(df: pd.DataFrame, metric: str = "click") -> pd.DataFrame:
    d = df[df["shuf"] == 0]
    if len(d) < 200 or d[metric].sum() < 20:
        return pd.DataFrame()
    segs = sorted(d["segment"].unique())
    cols = list(SWITCHES) + [f"seg_{s}" for s in segs[1:]] + ["fin"]
    X = np.column_stack([np.ones(len(d))] + [d[s].to_numpy(float) for s in SWITCHES]
                        + [(d["segment"] == s).to_numpy(float) for s in segs[1:]] + [d["fin"].to_numpy(float)])
    beta, se = irls_logit(X, d[metric].to_numpy(float))
    return pd.DataFrame({"variable": ["intercepto", *cols], "coef_logit": beta, "se": se,
                         "z": beta / np.maximum(se, 1e-12), "odds_ratio": np.exp(beta)})


def attention_curve(df: pd.DataFrame) -> list[float] | None:
    d = df[df["shuf"] == 1]
    if len(d) < 300 or (d["click_slot"] > 0).sum() < 30:
        return None
    ctr = np.array([(d["click_slot"] == k).mean() for k in range(1, 11)])
    if ctr[0] <= 0:
        return None
    e = np.maximum.accumulate((ctr / ctr[0])[::-1])[::-1]
    return [round(float(x), 4) for x in e]


def decide(eff: pd.DataFrame) -> dict[str, int]:
    out = dict(DEFAULT_SWITCHES)
    if eff.empty:
        return out
    allr = eff[eff["segmento"] == "TODOS"].set_index("interruptor")
    for s in SWITCHES:
        if s in allr.index:
            if allr.loc[s, "ic95_lo"] > 0:
                out[s] = 1
            elif allr.loc[s, "ic95_hi"] < 0:
                out[s] = 0
    return out


def run_analysis(client: Any, cfg: Config, out_dir: Path | None = None, hours: float | None = None,
                 joined: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir or ETAPA_DIR / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    if joined is None:
        joined = join_feedback(client, load_signal_logs(cfg, hours), client.team_id)
    df = lists_frame(joined)
    res: dict[str, Any] = {"n_lists": len(df)}
    if df.empty:
        return res
    eff_c = switch_effects(df, "click")
    eff_b = switch_effects(df, "buy")
    logit = logistic_effects(df, "click")
    att = attention_curve(df)
    known = df[df["new_user"] == 0]
    fin = {"ctr_normal": known[known["fin"] == 0]["click"].mean(), "ctr_modo_final": known[known["fin"] == 1]["click"].mean(),
           "n_normal": int((known["fin"] == 0).sum()), "n_final": int((known["fin"] == 1).sum())}
    n8 = {"pct_listas_con_nuevo": 100 * df["has_new"].mean(), "posicion_media": df["new_pos_mean"].mean(),
          "ctr_en_nuevos_por_lista_con_nuevo": df.loc[df["has_new"] == 1, "click_on_new"].mean()}
    sw = decide(eff_c)
    eff_c.to_csv(out_dir / "effects_click.csv", index=False)
    eff_b.to_csv(out_dir / "effects_buy.csv", index=False)
    Path(cfg.state_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.state_dir) / "switches.json").write_text(json.dumps({"switches": sw, "n_lists": len(df)}, indent=1))
    if att:
        (out_dir / "attention_measured.json").write_text(json.dumps({"attention": att}))
        (ETAPA_DIR / "eda").mkdir(exist_ok=True)
        (ETAPA_DIR / "eda" / "attention.json").write_text(json.dumps({"attention": att, "source": "barajado 4.1"}))

    def md(d: pd.DataFrame) -> str:
        if d is None or d.empty:
            return "_(sin datos suficientes)_"
        try:
            return d.round(4).to_markdown(index=False)
        except ImportError:
            return "```\n" + d.round(4).to_string(index=False) + "\n```"

    txt = f"""# Efectos del experimento factorial (4.1) y ensayo de la final (4.4)

Listas analizadas (servidas a tiempo, con feedback): **{len(df)}** · barajadas: {int(df['shuf'].sum())}

## Interruptores recomendados para el domingo (`state/switches.json`)

`{sw}`

Regla: encender si el IC95 % del efecto sobre el CTR está por encima de 0; apagar si está
por debajo; si no es concluyente, valor por defecto (todo encendido salvo `rev`).

## Efecto sobre el CTR por lista (diferencia de medias, sin listas barajadas)

{md(eff_c)}

## Efecto sobre la CVR por lista

{md(eff_b)}

## Regresión logística `clic ~ interruptores + segmento + modo final`

{md(logit)}

## Curva de atención medida con el barajado

`{att}` {"(escrita en eda/attention.json)" if att else "(faltan datos: ≥ 300 listas barajadas)"}

## Modo final (usuarios conocidos tratados como nuevos)

{fin}

Si `ctr_modo_final` es mucho menor que `ctr_normal`, la personalización por historial
pesa y el domingo (compradores nuevos) se perderá esa parte: priorizar sesión, contenido y vivo.

## N8 · productos nuevos en las listas

{n8}
"""
    (out_dir / "EFFECTS.md").write_text(txt, encoding="utf-8")
    res.update({"effects_click": eff_c, "logit": logit, "attention": att, "final_mode": fin, "n8": n8,
                "switches": sw})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .rawtree_client import RawTreeClient

    cfg = Config.from_env()
    res = run_analysis(RawTreeClient(cfg.api_url, cfg.api_key, cfg.team_id), cfg, hours=args.hours)
    print(f"{res['n_lists']} listas · interruptores recomendados: {res.get('switches')}")
    print(f"Informe: {ETAPA_DIR / 'experiments' / 'EFFECTS.md'}")


if __name__ == "__main__":
    main()
