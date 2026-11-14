"""Entrenamiento del modelo de respuesta (enfoque A: aprender el simulador).

    python -m src.train                       # datos reales de RawTree
    python -m src.train --local-sim           # mundo sintético + RawTree embebido (demo/tests)

Pasos (A1–A3):
 1. Corte temporal: los últimos `--val-days` días de `history_sessions` son validación.
 2. Histórico calculado SOLO con datos anteriores al corte (sin fuga hacia validación).
 3. Una fila por (petición, posición): todas las positivas y un `--neg-rate` de negativas,
    con peso 1/neg_rate y, por defecto, peso IPS min(1/propensity, 200).
 4. Modelos: logística (rápida y explicable) y GBM (LightGBM si está instalado; si no,
    HistGradientBoosting de sklearn). Variante SIN variables de identidad (riesgo 2).
 5. Métricas por segmento: log-loss, AUC, calibración; valor de política offline
    (IPS ítem-posición) contra popularidad y heurística; test C4 de cold start.
 6. Guarda `artifacts/model.json` o `artifacts/model_gbm.pkl`, `selected.txt` y
    `train_report.md` (con importancia de variables para la presentación).
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
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .catalog import parse_ts
from .config import Config
from .eval import (SQL_TS_RANGE, as_logged, cold_start_test, evaluate_segments, evaluate_synthetic,
                   exposure_rates, load_sessions, model_dm_scorer, popularity_policy, recommender_policy,
                   snapshot_from_history, synthetic_truth)
from .features import FEATURES, IDENTITY_FEATURES, compute_features
from .history import build_history, get_or_build_history
from .model import GBMModel, LinearModel, sigmoid
from .policy import Recommender
from .serve import fmt_ts

logger = logging.getLogger("etapa1esp.train")


# ----------------------------------------------------------------------------- dataset (A1)


def build_table(rows: list[dict[str, Any]], snap, h, neg_rate: float, ips_clip: float,
                seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cat = snap.catalog
    Xs, yc, yb, wns, wips, cold, ts, rid = [], [], [], [], [], [], [], []
    t0 = time.perf_counter()
    for k, row in enumerate(rows):
        req = as_logged(row, h)
        shown = list(row["shown"])[:10]
        props = list(row["propensity"])[:10]
        pos = [(s, cat.idx[p]) for s, p in enumerate(shown) if p in cat.idx]
        if not pos:
            continue
        slots = np.array([s + 1 for s, _ in pos])
        idx = np.array([j for _, j in pos])
        X, _ = compute_features(req, snap, item_idx=idx, slots=slots)
        c, b = row.get("clicked") or "", row.get("purchased") or ""
        for r, (s, j) in enumerate(pos):
            item = shown[s]
            is_c, is_b = item == c, item == b
            if not (is_c or is_b) and rng.random() >= neg_rate:
                continue
            Xs.append(X[r])
            yc.append(is_c)
            yb.append(is_b)
            wns.append(1.0 if (is_c or is_b) else 1.0 / neg_rate)
            wips.append(min(1.0 / max(float(props[s]), 1e-6), ips_clip))
            cold.append(req.is_new_user)
            ts.append(req.ts)
            rid.append(k)
        if (k + 1) % 20000 == 0:
            logger.info("  %d/%d filas (%.0fs)", k + 1, len(rows), time.perf_counter() - t0)
    return {"X": np.array(Xs, dtype=np.float32), "click": np.array(yc, bool), "buy": np.array(yb, bool),
            "w_ns": np.array(wns), "w_ips": np.array(wips), "cold": np.array(cold, bool),
            "ts": np.array(ts), "row": np.array(rid)}


def subset(t: dict[str, np.ndarray], m: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[m] for k, v in t.items()}


# ----------------------------------------------------------------------------- modelos (A3)


def fit_linear(t, target: str, feats: list[str], use_ips: bool, C: float = 1.0, name: str = "linear"):
    from sklearn.linear_model import LogisticRegression

    cols = [FEATURES.index(f) for f in feats]
    X = t["X"][:, cols].astype(np.float64)
    w = t["w_ns"] * (t["w_ips"] if use_ips else 1.0)
    w = w / w.mean()
    mu = np.average(X, axis=0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=w))
    sd[sd < 1e-6] = 1.0
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit((X - mu) / sd, t[target], sample_weight=w)
    coef = {f: float(c) for f, c in zip(feats, clf.coef_[0])}
    lo, hi = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
    return LinearModel(coef, float(clf.intercept_[0]), name=name,
                       mean={f: float(m) for f, m in zip(feats, mu)},
                       scale={f: float(s) for f, s in zip(feats, sd)},
                       meta={"target": target, "ips": use_ips},
                       clip_lo={f: float(v) for f, v in zip(feats, lo)},
                       clip_hi={f: float(v) for f, v in zip(feats, hi)})


def fit_gbm(t, target: str, feats: list[str], use_ips: bool, name: str = "gbm"):
    cols = [FEATURES.index(f) for f in feats]
    X = t["X"][:, cols]
    w = t["w_ns"] * (t["w_ips"] if use_ips else 1.0)
    w = w / w.mean()
    try:
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(n_estimators=200, max_depth=6, num_leaves=31, learning_rate=0.05,
                                 min_child_samples=100, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        lib = "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05,
                                             min_samples_leaf=100, l2_regularization=1.0)
        lib = "sklearn-hgb"
    clf.fit(X, t[target], sample_weight=w)
    return GBMModel(clf, feats, name=name, meta={"target": target, "ips": use_ips, "lib": lib})


def metrics(model, t, target: str) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    out = {}
    for seg, m in (("ALL", np.ones(len(t[target]), bool)), ("WARM", ~t["cold"]), ("COLD", t["cold"])):
        y = t[target][m]
        if len(y) < 50 or y.all() or not y.any():
            continue
        p = np.clip(model.predict(t["X"][m]), 1e-7, 1 - 1e-7)
        w = t["w_ns"][m]
        out[f"logloss_{seg}"] = float(log_loss(y, p, sample_weight=w))
        out[f"auc_{seg}"] = float(roc_auc_score(y, p, sample_weight=w))
        out[f"calib_ratio_{seg}"] = float(np.average(p, weights=w) / max(np.average(y, weights=w), 1e-12))
    return out


def calibration_table(model, t, target: str, bins: int = 10) -> pd.DataFrame:
    p = model.predict(t["X"])
    w = t["w_ns"]
    y = t[target].astype(float)
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        m = b == k
        if m.any():
            rows.append({"bin": k + 1, "p_pred": float(np.average(p[m], weights=w[m])),
                         "p_obs": float(np.average(y[m], weights=w[m])), "peso": float(w[m].sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- pipeline


def run(client: Any, cfg: Config, *, val_days: float = 10.0, train_days: float = 25.0, sample_pages: int = 4, sample_mod: int = 40,
        neg_rate: float = 0.1, ips_clip: float = 200.0, use_ips: bool = True, target: str = "buy",
        rebuild: bool = False, n_policy_eval: int = 2000, world=None, out_dir: Path | None = None,
        cache: bool = True) -> dict[str, Any]:
    out_dir = Path(out_dir or cfg.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_row = client.query(SQL_TS_RANGE, timeout=cfg.query_timeout_slow_sec)[0]
    t_max = parse_ts(rng_row["t_max"])
    split = fmt_ts(t_max - val_days * 86400)[:19]
    t_min = parse_ts(rng_row["t_min"])
    stats_cut = fmt_ts(max(t_max - (val_days + train_days) * 86400, t_min + 0.3 * (t_max - t_min)))[:19]
    logger.info("Sesiones: %s → %s (%s filas). Variables < %s · entrenamiento [%s, %s) · validación ≥ %s",
                rng_row["t_min"], rng_row["t_max"], rng_row["n"], stats_cut, stats_cut, split, split)

    # Variables "a fecha de": las filas de entrenamiento solo ven el histórico anterior a
    # `stats_cut` y las de validación el anterior a `split`. Así ni la popularidad por
    # contexto ni la CVR por ítem (que salen de las mismas sesiones) contienen la etiqueta.
    def hist(cut: str):
        return get_or_build_history(client, cfg, cutoff=cut, rebuild=rebuild) if cache else \
            build_history(client, cfg, cutoff=cut)

    h_tr = hist(stats_cut)
    h = hist(split)
    snap_tr = snapshot_from_history(h_tr, cfg)
    snap = snapshot_from_history(h, cfg)

    logger.info("Descargando muestra de sesiones (%d/%d)…", sample_pages, sample_mod)
    rows = load_sessions(client, n_pages=sample_pages, mod=sample_mod)
    tr_rows = [r for r in rows if parse_ts(stats_cut) <= parse_ts(r["ts"]) < parse_ts(split)]
    va_rows = [r for r in rows if parse_ts(r["ts"]) >= parse_ts(split)]
    logger.info("Filas: %d entrenamiento, %d validación", len(tr_rows), len(va_rows))
    t_tr = build_table(tr_rows, snap_tr, h_tr, neg_rate, ips_clip, seed=1)
    t_va = build_table(va_rows, snap, h, neg_rate, ips_clip, seed=2)
    n_pos = int(t_tr[target].sum())
    if target == "buy" and n_pos < 300:
        logger.warning("Solo %d compras en entrenamiento: uso clics como objetivo", n_pos)
        target = "click"
    logger.info("Tabla: %d ejemplos (%d positivos de %s)", len(t_tr[target]), int(t_tr[target].sum()), target)

    all_f = list(FEATURES)
    noid_f = [f for f in FEATURES if f not in IDENTITY_FEATURES]
    candidates = {
        "linear": fit_linear(t_tr, target, all_f, use_ips, name="linear"),
        "linear_noid": fit_linear(t_tr, target, noid_f, use_ips, name="linear_noid"),
        "gbm": fit_gbm(t_tr, target, all_f, use_ips, name="gbm"),
        "gbm_noid": fit_gbm(t_tr, target, noid_f, use_ips, name="gbm_noid"),
    }
    if use_ips:
        candidates["linear_noips"] = fit_linear(t_tr, target, all_f, False, name="linear_noips")
    met = {k: metrics(m, t_va, target) for k, m in candidates.items()}
    met["heuristic"] = metrics(LinearModel.default(), t_va, target)
    met_df = pd.DataFrame(met).T
    logger.info("Validación (%s):\n%s", target, met_df.round(4).to_string())

    # Selección: mejor log-loss; la lineal gana si está a < 1 % (más rápida y explicable)
    ll = {k: v.get("logloss_ALL", np.inf) for k, v in met.items() if k in candidates}
    best = min(ll, key=ll.get)
    if candidates[best].kind == "gbm":
        lin_best = min((k for k in ll if k.startswith("linear")), key=ll.get)
        if ll[lin_best] <= ll[best] * 1.01:
            best = lin_best
    chosen = candidates[best]
    logger.info("Modelo elegido: %s", best)

    # Valor de política offline sobre validación
    cfg_eval = cfg
    va_eval = va_rows[:n_policy_eval]
    pols = {"popularidad (baseline)": popularity_policy(snap)}
    for name, model in (("heurística", LinearModel.default()), ("linear", candidates["linear"]),
                        ("linear_noid", candidates["linear_noid"]), ("gbm", candidates["gbm"])):
        rec = Recommender(cfg_eval, model=model, seed=0)
        rec.cfg = cfg_eval
        pols[name] = recommender_policy(rec, snap, arm="hist")
    dm_ref = candidates["gbm"] if best.startswith("linear") else candidates["linear"]
    pol_df = evaluate_segments(pols, va_eval, h, cfg.attention, dm_scorer=model_dm_scorer(dm_ref, snap),
                               all_new_policies={k: v for k, v in pols.items()
                                                 if k in ("popularidad (baseline)", "heurística", best if best in pols else "linear")},
                               exposure=exposure_rates(rows))
    if world is not None:
        truth = synthetic_truth(world, pols, va_eval, h)
        pol_df["true_buy_per100(sintético)"] = [truth.get(p, np.nan) if s == "ALL" else np.nan
                                               for s, p in pol_df.index]
    logger.info("Valor de política (validación):\n%s", pol_df.round(3).to_string())

    c4 = cold_start_test(h, cfg)
    if len(c4):
        logger.info("Test C4:\n%s", c4.round(3).to_string())

    synth_df = None
    if world is not None:
        sh = build_history(client, cfg)          # histórico completo, como en servicio
        ssnap = snapshot_from_history(sh, cfg)
        spols = {"popularidad (baseline)": lambda r, s=ssnap: list(s.catalog.global_ranked[:10])}
        for name, model in (("heurística", LinearModel.default()), (best, chosen),
                            ("linear", candidates["linear"]), ("gbm", candidates["gbm"])):
            rec = Recommender(cfg, model=model, seed=0)
            spols[name] = (lambda r, rec=rec: rec.recommend(r, ssnap, arm="hist").items)
        synth_df = evaluate_synthetic(world, spols, n=1500)
        logger.info("Mundo sintético (compras esperadas reales del modelo oculto):\n%s",
                    synth_df.round(3).to_string())

    # Guardar
    for p in (out_dir / "model.json", out_dir / "model_gbm.pkl"):
        if p.exists():
            p.unlink()
    path = out_dir / ("model.json" if chosen.kind == "linear" else "model_gbm.pkl")
    chosen.save(path)
    (out_dir / "selected.txt").write_text(path.name)
    calib = calibration_table(chosen, t_va, target)
    imp = chosen.importance()
    write_report(out_dir / "train_report.md", split=split, target=target, n_tr=len(tr_rows), n_va=len(va_rows),
                 table_n=len(t_tr[target]), met_df=met_df, best=best, calib=calib, imp=imp, pol_df=pol_df,
                 c4=c4, synth=synth_df, use_ips=use_ips, neg_rate=neg_rate, comps=h.ok_components)
    (out_dir / "train_metrics.json").write_text(json.dumps({"best": best, "target": target, "metrics": met},
                                                           indent=1, default=float))
    return {"best": best, "path": str(path), "metrics": met_df, "policy": pol_df, "c4": c4, "synth": synth_df,
            "calib": calib, "importance": imp}


def _md(df: pd.DataFrame | None) -> str:
    if df is None or len(df) == 0:
        return "_(sin datos)_"
    try:
        return df.round(4).to_markdown()
    except ImportError:          # sin `tabulate`
        return "```\n" + df.round(4).to_string() + "\n```"


def write_report(path: Path, **kw: Any) -> None:
    imp = "\n".join(f"| {f} | {v:.3f} |" for f, v in kw["imp"][:20])
    txt = f"""# Informe de entrenamiento · Etapa 1 · versión especial

* Corte temporal: entrenamiento < `{kw['split']}` ≤ validación
* Objetivo: **{kw['target']}** · IPS: {kw['use_ips']} · negativas muestreadas al {kw['neg_rate']:.0%}
* Sesiones: {kw['n_tr']} entrenamiento, {kw['n_va']} validación → {kw['table_n']} ejemplos (petición × posición)
* Componentes del histórico: {kw['comps']}
* **Modelo elegido: `{kw['best']}`**

## Métricas en validación (por segmento)

COLD = usuario sin historial antes del corte (en `history_sessions` no hay `is_new_user`).
`calib_ratio` = media de p̂ / tasa observada (1,0 = bien calibrado).

{_md(kw['met_df'])}

## Calibración del modelo elegido (validación)

{_md(kw['calib'])}

## Importancia de variables

| Variable | Importancia |
|---|---|
{imp}

(Lineal: |coeficiente estandarizado|. GBM: nº de divisiones.)

## Valor de política offline (validación)

`ips_buy_per100`: compras por 100 peticiones estimadas con IPS ítem-posición (aproximado).
`ips_ess`: tamaño efectivo de muestra. `dm_buy_per100`: método directo con un modelo distinto.
`ALL_AS_NEW`: las mismas peticiones tratadas como usuarios nuevos (escenario de la final).

{_md(kw['pol_df'])}

## Test C4 · cold start de producto

{_md(kw['c4'])}

## Mundo sintético (solo en `--local-sim`)

{_md(kw['synth'])}
"""
    path.write_text(txt, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Etapa 1 · versión especial · entrenamiento del modelo de respuesta")
    ap.add_argument("--local-sim", action="store_true", help="mundo sintético + RawTree embebido (chdb)")
    ap.add_argument("--val-days", type=float, default=10.0)
    ap.add_argument("--sample-pages", type=int, default=4)
    ap.add_argument("--sample-mod", type=int, default=40, help="muestra = pages/mod de las sesiones")
    ap.add_argument("--neg-rate", type=float, default=0.1)
    ap.add_argument("--ips-clip", type=float, default=200.0)
    ap.add_argument("--train-days", type=float, default=25.0)
    ap.add_argument("--no-ips", action="store_true")
    ap.add_argument("--target", choices=["buy", "click"], default="buy")
    ap.add_argument("--rebuild-history", action="store_true")
    ap.add_argument("--policy-eval-rows", type=int, default=2000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    world = None
    if args.local_sim:
        import tempfile

        from .localdb import LocalRawTree
        from .sim import build_world

        W = build_world(seed=5, n_products=500, n_sessions_events=15000, n_sessions_log=20000)
        world = W["world"]
        client = LocalRawTree()
        client.insert("shop", "catalog", W["catalog"])
        client.insert("shop", "history_events", W["events"])
        client.insert("shop", "history_sessions", W["sessions"])
        tmp = tempfile.mkdtemp()
        cfg = Config(cache_dir=tmp, artifacts_dir=str(Path(tmp) / "artifacts"))
        args.sample_pages, args.sample_mod = 1, 1
    else:
        from .rawtree_client import RawTreeClient

        cfg = Config.from_env()
        client = RawTreeClient(cfg.api_url, cfg.api_key, cfg.team_id)
    res = run(client, cfg, val_days=args.val_days, train_days=args.train_days, sample_pages=args.sample_pages, sample_mod=args.sample_mod,
              neg_rate=args.neg_rate, ips_clip=args.ips_clip, use_ips=not args.no_ips, target=args.target,
              rebuild=args.rebuild_history, n_policy_eval=args.policy_eval_rows, world=world)
    print(f"\nModelo guardado en {res['path']} (elegido: {res['best']})")


_ = timedelta
if __name__ == "__main__":
    main()
