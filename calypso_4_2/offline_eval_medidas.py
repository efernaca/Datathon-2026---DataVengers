"""Evaluación offline de las MEDIDAS_CONVERSION sobre el log real (offline_eval_cache).

Compara, sobre las 25.000 peticiones reales de shop.history_sessions que descarga
run_offline_eval_official.py, tres políticas:

  * baseline            top-10 del ranking de ventas
  * 4.1e legacy         preset conversion_first (4.1e tal cual)
  * medidas_conversion  preset con M1, M3, M4, M5, M6, M7

Métricas:
  * hit@10 clic / compra   el producto clicado / comprado está entre los 10 recomendados
  * SNIPS ctr slot-1       métrica oficial del notebook (IPS normalizado sobre el slot 1)
  * compras esperadas      por 100 peticiones bajo el modelo logístico de MEDIDAS_CONVERSION §1.4
                           (ajustado sobre este mismo log; es una estimación, no una medida)

Limitaciones: la caché no trae catálogo, historial de usuario ni compras absolutas por producto.
El snapshot se construye solo con el orden de best_sellers.json (compras sintéticas ∝ puesto^-0,87),
las co-visitas y las vistas de la sesión. Sin afinidad, precio, carrito ni contexto.

    python calypso_4_2/offline_eval_medidas.py [--cache ../offline_eval_cache] [--n 25000]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_MODEL_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(_MODEL_ROOT), str(_MODEL_ROOT / "etapa_4")]

from etapa_4.src.config import ConfigV4  # noqa: E402
from etapa_4.src.recommender import CalypsoRecommenderV4, build_snapshot_v4_from_records  # noqa: E402
from etapa_4.src.weights_profile import get_preset  # noqa: E402

MAX_RANK = 3000


def _pid(v):
    return v["product_id"] if isinstance(v, dict) else str(v)


def load(cache: Path, n: int):
    df = pd.read_parquet(cache / "official_test_data.parquet")
    cov = pd.read_parquet(cache / "covisit.parquet")
    best = json.loads((cache / "best_sellers.json").read_text(encoding="utf-8"))
    for col in ("clicked", "purchased"):
        df[col] = df[col].where(df[col].notna() & (df[col] != ""), None)
    df["page"] = df.context.str["page"]
    df["rv"] = df.recent_views.apply(lambda a: [_pid(v) for v in a] if a is not None else [])
    if n and n < len(df):
        df = df.sample(n=n, random_state=0)
    return df.reset_index(drop=True), cov, best


def build_snapshot(df, cov, best, preset: str):
    wp = get_preset(preset)
    cfg = ConfigV4.from_profile(wp)
    pids = set(best) | set(cov.p) | set(cov.q)
    for r in df.itertuples(index=False):
        pids.update(r.rv)
        pids.update(r.shown)
    catalog = [{"product_id": p, "category": "", "subcategory": "", "brand": "", "price_eur": 0.0, "description": f"producto {p}",
                "is_active": True, "is_new_product": False} for p in sorted(pids)]
    pop = []
    for i, p in enumerate(best):
        buys = 1000.0 * (i + 1.0) ** -0.87
        pop.append({"product_id": p, "views": buys * 50.0, "carts": buys * 3.0, "purchases": buys,
                    "self_cvr": (buys + 0.5) / (buys * 50.0 + 25.0),
                    "weighted_pop": buys * 5.0 + buys * 3.0 * 2.0 + buys * 50.0 * 0.2})
    trans = [{"src_product": p, "dst_product": q, "raw_transition_score": float(nn)}
             for p, q, nn in cov[["p", "q", "n"]].itertuples(index=False)]
    snap, _ = build_snapshot_v4_from_records(
        catalog_rows=catalog, popularity_rows=pop, transition_rows=trans,
        config=cfg, version=1, weights_profile=wp,
    )
    snap.config_override = cfg
    return snap, CalypsoRecommenderV4(cfg)


def fit_logit(df, covd, rank):
    rows = []
    for r in df.itertuples(index=False):
        rvset = set(r.rv)
        nb = set()
        for v in r.rv:
            nb.update(covd.get(v, [])[:20])
        home = int(r.page == "home")
        for k, item in enumerate(r.shown):
            rows.append((k + 1, np.log(min(rank.get(item, 99999), MAX_RANK) + 1.0), int(item in rvset),
                         int(item in nb and item not in rvset), home, int(r.purchased == item)))
    L = pd.DataFrame(rows, columns=["slot", "log_rank", "visto", "cov", "home", "y"])
    X = pd.get_dummies(L.slot.astype(str)).drop(columns=["1"]).astype(float)[[str(k) for k in range(2, 11)]]
    for c in ("log_rank", "visto", "cov", "home"):
        X[c] = L[c].values
    m = LogisticRegression(C=1e4, max_iter=3000).fit(X, L.y)
    coef = dict(zip(X.columns, m.coef_[0]))
    slot_c = np.array([0.0] + [coef[str(k)] for k in range(2, 11)])
    return float(m.intercept_[0]), coef, slot_c


def main() -> None:
    logging.disable(logging.WARNING)
    os.environ.pop("CALYPSO_PRESET", None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(_MODEL_ROOT.parent / "offline_eval_cache"))
    ap.add_argument("--n", type=int, default=25000)
    args = ap.parse_args()

    df, cov, best = load(Path(args.cache), args.n)
    covd = cov.groupby("p")["q"].apply(list).to_dict()
    rank = {p: i for i, p in enumerate(best)}
    b0, coef, slot_c = fit_logit(df, covd, rank)

    def expected(items, rvset, nb, home):
        z = np.array([
            b0 + coef["log_rank"] * np.log(min(rank.get(i, 99999), MAX_RANK) + 1.0)
            + coef["visto"] * (i in rvset) + coef["cov"] * (i in nb and i not in rvset) + coef["home"] * home
            for i in items
        ]) + slot_c[: len(items)]
        return float((1.0 / (1.0 + np.exp(-z))).sum())

    snaps = {name: build_snapshot(df, cov, best, name) for name in ("conversion_first", "medidas_conversion")}
    policies = {
        "baseline top-10 ventas": lambda req: best[:10],
        "4.1e legacy (conversion_first)": lambda req: snaps["conversion_first"][1].recommend(
            req, snaps["conversion_first"][0], now_ms=0.0),
        "4.2 (medidas_conversion)": lambda req: snaps["medidas_conversion"][1].recommend(
            req, snaps["medidas_conversion"][0], now_ms=0.0),
    }

    prop1 = np.array([float(p[0]) if len(p) and float(p[0]) > 0 else 1.0 for p in df.propensity])
    first = df.shown.str[0].to_numpy()
    results = {}
    seg = np.where(df.page == "home", "home", "con sesión")
    for name, pol in policies.items():
        hit_c, hit_b, s1, val = [], [], [], []
        for r in df.itertuples(index=False):
            req = {"request_id": r.request_id, "user_id": r.user_id, "is_new_user": False,
                   "context": r.context, "recent_views": list(r.recent_views)}
            items = list(pol(req))[:10]
            rvset = set(r.rv)
            nb = set()
            for v in r.rv:
                nb.update(covd.get(v, [])[:20])
            hit_c.append(r.clicked in items if r.clicked else np.nan)
            hit_b.append(r.purchased in items if r.purchased else np.nan)
            s1.append(items[0] if items else "")
            val.append(expected(items, rvset, nb, int(r.page == "home")))
        match = np.array(s1) == first
        reward = (df.clicked.to_numpy() == first).astype(float)
        w = match / prop1
        val = np.array(val)
        results[name] = {
            "hit@10 clic": np.nanmean(np.array(hit_c, dtype=float)),
            "hit@10 compra": np.nanmean(np.array(hit_b, dtype=float)),
            "SNIPS ctr slot-1": (w * reward).sum() / max(w.sum(), 1e-9),
            "compras esp./100 (con sesión)": 100 * val[seg == "con sesión"].mean(),
            "compras esp./100 (home)": 100 * val[seg == "home"].mean(),
            "compras esp./100 (total)": 100 * val.mean(),
        }
    out = pd.DataFrame(results).T
    base = out.iloc[0]
    print(f"\nEvaluación offline sobre {len(df)} peticiones reales ({df.purchased.notna().sum()} compras)\n")
    print(out.round(4).to_string())
    print("\nRelativo al baseline:")
    print((out / base).round(3).to_string())


if __name__ == "__main__":
    main()
