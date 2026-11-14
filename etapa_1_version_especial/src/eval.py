"""Evaluación offline (sección 3.4 y A3 de la revisión).

* `load_sessions`: cargador de `shop.history_sessions` REAL (muestra por hash, paginado).
* Estimadores de valor de una política sobre el log:
    - IPS ítem-posición (PBM): aprovecha las 10 posiciones de cada fila. Para cada ítem i
      mostrado en el slot s que nuestra política coloca en la posición r:
          y · (e_r / e_s) / Q_i,     Q_i = nº medio de veces que el log muestra i por petición
      Como el sistema antiguo es "best sellers + aleatoriedad" (no depende del contexto),
      la exposición marginal empírica Q_i es una buena propensity. La `propensity` de la
      tabla es *condicional a los slots de arriba* y, usada como marginal, sobrecuenta a los
      populares (lo comprobamos en el mundo sintético: favorecía al baseline).
      Supone atención por posición e independencia entre slots: es aproximado.
    - Método directo (DM): Σ_k e_k · p̂(ítem_k) con un modelo **distinto** del evaluado.
    - HitRate@10 y SNIPS en slot 1 (se mantienen, pero no sirven para ordenar modelos).
* Segmentos ALL / WARM / COLD y escenario "todos los usuarios como nuevos" (sección 5).
* Test C4 de cold start: ocultar un 5 % de productos con historial y medir cuánto de su
  fila EASE real recupera cada método de vecinos.
* `evaluate_synthetic`: compras esperadas con el modelo oculto del mundo sintético.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .catalog import CatalogManager, parse_ts
from .cold_start import SimWeights, content_neighbors
from .config import Config
from .features import Req, compute_features, parse_request
from .state import HistoryBundle, Snapshot

logger = logging.getLogger(__name__)

SQL_SESSIONS = """
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
       user_id::String AS user_id,
       context.region::String AS region, context.device::String AS device,
       context.page::String AS page,
       toJSONString(recent_views) AS recent_views,
       shown::Array(String) AS shown, propensity::Array(Float64) AS propensity,
       ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased,
       ifNull(revenue_eur::Float64, 0) AS revenue_eur
FROM shop.history_sessions
WHERE cityHash64(request_id::String) % {mod} = {{page}} {extra}
"""

SQL_TS_RANGE = """
SELECT toString(min(ts::DateTime64(3))) AS t_min, toString(max(ts::DateTime64(3))) AS t_max,
       count() AS n FROM shop.history_sessions
"""


def load_sessions(client: Any, n_pages: int = 4, mod: int = 40, since: str | None = None,
                  until: str | None = None, timeout: float = 300.0) -> list[dict[str, Any]]:
    """Muestra de n_pages/mod de las sesiones (por defecto 10 % ≈ 200.000 filas)."""
    extra = ""
    if since:
        extra += f" AND ts::DateTime64(3) >= '{since}'"
    if until:
        extra += f" AND ts::DateTime64(3) < '{until}'"
    tmpl = SQL_SESSIONS.format(mod=mod, extra=extra)
    if hasattr(client, "query_paged"):
        return client.query_paged(tmpl, n_pages=n_pages, timeout=timeout)
    return [r for p in range(n_pages) for r in client.query(tmpl.format(page=p, n_pages=n_pages), timeout)]


def snapshot_from_history(h: HistoryBundle, cfg: Config) -> Snapshot:
    return Snapshot(history=h, catalog=CatalogManager(h, cfg).build(), live={})


def as_logged(row: Mapping[str, Any], h: HistoryBundle) -> Req:
    """En `history_sessions` no hay `is_new_user`: se aproxima por 'sin perfil histórico'."""
    req = parse_request(row)
    req.is_new_user = req.user_id not in h.users
    return req


# ----------------------------------------------------------------------------- estimadores

PolicyFn = Callable[[Req], list[str]]


def exposure_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Q_i: veces que el log muestra el producto i por petición (propensity marginal)."""
    cnt: dict[str, float] = {}
    for r in rows:
        for p in list(r["shown"])[:10]:
            cnt[p] = cnt.get(p, 0.0) + 1.0
    n = max(len(rows), 1)
    return {p: c / n for p, c in cnt.items()}


def evaluate_policy(policy: PolicyFn, rows: Sequence[Mapping[str, Any]], reqs: Sequence[Req],
                    attention: Sequence[float], clip: float = 50.0,
                    dm_scorer: Callable[[Req, list[str]], np.ndarray] | None = None,
                    exposure: Mapping[str, float] | None = None, max_inv_q: float = 500.0) -> dict[str, float]:
    e = np.asarray(attention, dtype=float)
    n = len(rows)
    if n == 0:
        return {"n": 0}
    Q = exposure if exposure is not None else exposure_rates(rows)
    ips_buy = ips_click = 0.0
    w_all: list[float] = []
    hit_c = hit_b = n_c = n_b = 0
    s1_w = s1_c = s1_b = 0.0
    dm = 0.0
    for row, req in zip(rows, reqs):
        lst = policy(req)
        rank = {p: r for r, p in enumerate(lst)}
        shown = list(row["shown"])
        props = list(row["propensity"])
        clicked, bought = row.get("clicked") or "", row.get("purchased") or ""
        for s, item in enumerate(shown[:10]):
            r = rank.get(item)
            if r is None:
                continue
            w = (e[r] / e[s]) * min(1.0 / max(Q.get(item, 0.0), 1e-9), max_inv_q)
            w_all.append(w)
            if item == clicked:
                ips_click += w
            if item == bought:
                ips_buy += w
        if clicked:
            n_c += 1
            hit_c += clicked in rank
        if bought:
            n_b += 1
            hit_b += bought in rank
        if lst and shown and lst[0] == shown[0]:
            w1 = min(1.0 / max(float(props[0]), 1e-6), clip)
            s1_w += w1
            s1_c += w1 * (clicked == shown[0])
            s1_b += w1 * (bought == shown[0])
        if dm_scorer is not None:
            p = dm_scorer(req, lst)
            dm += float(np.sum(e[: len(p)] * p))
    wa = np.array(w_all) if w_all else np.zeros(1)
    out = {
        "n": n,
        "ips_buy_per100": 100 * ips_buy / n,
        "ips_click_per100": 100 * ips_click / n,
        "ips_ess": float(wa.sum() ** 2 / max((wa ** 2).sum(), 1e-12)),
        "hit_click@10": hit_c / max(n_c, 1),
        "hit_buy@10": hit_b / max(n_b, 1),
        "snips_slot1_ctr": s1_c / s1_w if s1_w else float("nan"),
        "slot1_matches_w": s1_w,
    }
    if dm_scorer is not None:
        out["dm_buy_per100"] = 100 * dm / n
    return out


def evaluate_segments(policies: Mapping[str, PolicyFn], rows: Sequence[Mapping[str, Any]],
                      h: HistoryBundle, attention: Sequence[float],
                      dm_scorer=None, all_new_policies: Mapping[str, PolicyFn] | None = None,
                      exposure: Mapping[str, float] | None = None) -> pd.DataFrame:
    reqs = [as_logged(r, h) for r in rows]
    Q = exposure if exposure is not None else exposure_rates(rows)
    cold = np.array([q.is_new_user for q in reqs])
    segs = {"ALL": np.ones(len(rows), bool), "WARM": ~cold, "COLD": cold}
    out = []
    for seg, m in segs.items():
        sub_rows = [r for r, k in zip(rows, m) if k]
        sub_reqs = [q for q, k in zip(reqs, m) if k]
        for name, pol in policies.items():
            out.append({"segment": seg, "policy": name,
                        **evaluate_policy(pol, sub_rows, sub_reqs, attention, dm_scorer=dm_scorer, exposure=Q)})
    if all_new_policies:
        new_reqs = []
        for q in reqs:
            q2 = Req(**{**q.__dict__})
            q2.is_new_user = True
            new_reqs.append(q2)
        for name, pol in all_new_policies.items():
            out.append({"segment": "ALL_AS_NEW", "policy": name,
                        **evaluate_policy(pol, rows, new_reqs, attention, dm_scorer=dm_scorer, exposure=Q)})
    return pd.DataFrame(out).set_index(["segment", "policy"])


def popularity_policy(snap: Snapshot) -> PolicyFn:
    top = list(snap.catalog.global_ranked[:10])
    return lambda req: top


def recommender_policy(rec, snap: Snapshot, arm: str = "hist", variant: str = "model") -> PolicyFn:
    return lambda req: rec.recommend(req, snap, arm=arm, variant=variant,
                                     use_profile=not req.is_new_user).items


def model_dm_scorer(model, snap: Snapshot) -> Callable[[Req, list[str]], np.ndarray]:
    def score(req: Req, items: list[str]) -> np.ndarray:
        idx = np.array([snap.catalog.idx[p] for p in items if p in snap.catalog.idx])
        X, _ = compute_features(req, snap, item_idx=idx)
        return model.predict(X)
    return score


# ----------------------------------------------------------------------------- C4


def cold_start_test(h: HistoryBundle, cfg: Config, frac: float = 0.05, seed: int = 0, k: int = 20
                    ) -> pd.DataFrame:
    """Tratar como nuevos un 5 % de productos con fila EASE y comparar métodos de vecinos.

    Métrica: recall@k de los k destinos EASE reales del producto dentro de los k destinos
    que se obtienen heredando la fila de sus vecinos de contenido (se excluye el propio
    producto). Aproximación: la B no se reentrena sin esos productos.
    """
    if h.ease is None:
        return pd.DataFrame()
    cm = CatalogManager(h, cfg)
    cat = cm.build()
    B = h.ease.B
    eidx = h.ease.index
    warm = np.array([i for i, p in enumerate(cat.ids) if p in eidx])
    rng = np.random.default_rng(seed)
    test = rng.choice(warm, size=max(5, int(frac * len(warm))), replace=False)
    pool = np.setdiff1d(warm, test)
    pool_e = np.array([eidx[cat.ids[j]] for j in pool])
    variants = {
        "texto+estructura (actual)": SimWeights(),
        "solo texto": SimWeights(subcat=0, cat=0, brand=0, price=0),
        "solo estructura": SimWeights(text=0),
        "texto+estructura, sin exigir categoría": SimWeights(),
    }
    rows = []
    sub_pop: dict[int, np.ndarray] = {}
    for name, sw in variants.items():
        same_cat = name != "texto+estructura, sin exigir categoría"
        nbs = content_neighbors(cat.content[test], cat.category[test], cat.subcategory[test], cat.brand[test],
                                cat.log_price[test], cat.content[pool], cat.category[pool],
                                cat.subcategory[pool], cat.brand[pool], cat.log_price[pool],
                                top_k=cfg.cold_top_k, min_sim=0.0, same_category=same_cat, weights=sw)
        recalls = []
        for t, nb in zip(test, nbs):
            te = eidx[cat.ids[t]]
            true = B[te].copy()
            true[te] = -np.inf
            true_top = set(np.argsort(-true)[:k])
            if not nb:
                recalls.append(0.0)
                continue
            inh = sum(s * B[pool_e[j]] for j, s in nb)
            inh[te] = -np.inf
            pred_top = set(np.argsort(-inh)[:k])
            recalls.append(len(true_top & pred_top) / k)
        rows.append({"método": name, f"recall@{k}": float(np.mean(recalls)), "n": len(test)})
    # referencia: los más populares de su subcategoría
    pop = np.array([h.pop.get(p, 0.0) for p in h.ease.item_ids])
    e_sub = np.array([cat.subcategory[cat.idx[p]] if p in cat.idx else -1 for p in h.ease.item_ids])
    recalls = []
    for t in test:
        te = eidx[cat.ids[t]]
        true = B[te].copy()
        true[te] = -np.inf
        true_top = set(np.argsort(-true)[:k])
        sc = np.where(e_sub == cat.subcategory[t], pop, -1.0)
        sc[te] = -np.inf
        recalls.append(len(true_top & set(np.argsort(-sc)[:k])) / k)
    rows.append({"método": "populares de la subcategoría (referencia)", f"recall@{k}": float(np.mean(recalls)),
                 "n": len(test)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- sintético


def evaluate_synthetic(world, policies: Mapping[str, Callable[[dict], list[str]]], n: int = 1500,
                       new_user_rate: float = 0.5, seed: int = 11, new_products: np.ndarray | None = None
                       ) -> pd.DataFrame:
    """Compras esperadas por 100 peticiones según el modelo oculto + score 0–100."""
    rng_state = world.rng
    world.rng = np.random.default_rng(seed)
    reqs = [world.make_request(world.t0 + 61 * 86400 + i, new_user_rate=new_user_rate,
                               new_products=new_products, rid=f"ev_{i}") for i in range(n)]
    world.rng = rng_state
    idx = {p["product_id"]: i for i, p in enumerate(world.products)}
    active = np.ones(len(world.products), dtype=bool)
    res: dict[str, list[float]] = {k: [] for k in policies}
    res["_oracle"] = []
    cold = []
    for req, u, rec in reqs:
        a, b = world.relevance(u, rec, req["context"]["page"])
        res["_oracle"].append(world.expected_purchase(world.oracle(a, b, active), a, b))
        cold.append(req["is_new_user"] or (new_products is not None and any(r in set(new_products) for r in rec)))
        for name, pol in policies.items():
            items = [idx[p] for p in pol(req) if p in idx]
            res[name].append(world.expected_purchase(items, a, b))
    cold = np.array(cold)
    out = []
    base_name = next(iter(policies))
    for seg, m in (("ALL", np.ones(n, bool)), ("COLD", cold), ("WARM", ~cold)):
        if not m.any():
            continue
        base = np.mean(np.array(res[base_name])[m])
        orc = np.mean(np.array(res["_oracle"])[m])
        for name in policies:
            v = np.mean(np.array(res[name])[m])
            out.append({"segment": seg, "policy": name, "exp_buys_per100": 100 * v,
                        "score_0_100": float(np.clip(100 * (v - base) / max(orc - base, 1e-12), 0, 100)),
                        "oracle_per100": 100 * orc})
    return pd.DataFrame(out).set_index(["segment", "policy"])


_ = (math, parse_ts)


def synthetic_truth(world, policies: Mapping[str, PolicyFn], rows: Sequence[Mapping[str, Any]],
                    h: HistoryBundle) -> dict[str, float]:
    """Solo mundo sintético: compras esperadas REALES de cada política sobre las mismas filas
    del log, para medir el sesgo de los estimadores offline (IPS, DM)."""
    idx = {p["product_id"]: i for i, p in enumerate(world.products)}
    out = {}
    for name, pol in policies.items():
        vals = []
        for r in rows:
            q = as_logged(r, h)
            u = int(q.user_id[2:]) if q.user_id.startswith("u_") and q.user_id[2:].isdigit() else None
            rec = [idx[p] for p in q.recent if p in idx]
            a, b = world.relevance(u, rec, q.page)
            vals.append(world.expected_purchase([idx[p] for p in pol(q) if p in idx], a, b))
        out[name] = 100 * float(np.mean(vals))
    return out
