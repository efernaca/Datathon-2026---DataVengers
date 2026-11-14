"""Catálogo vivo: aplica `shop.catalog_updates` de forma incremental (C1, F4, V6).

`CatalogManager` lo usa un único hilo (el de catálogo y stock). Cada vez que entra
algún cambio construye un `CatalogState` nuevo e inmutable y lo publica.
Los productos nuevos tienen vecinos de contenido en el mismo segundo en que llegan.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp

from .cold_start import ContentIndex, content_neighbors
from .config import Config
from .state import CatalogState, HistoryBundle

logger = logging.getLogger(__name__)


def parse_ts(value: Any) -> float:
    """Timestamp RawTree/ISO → epoch en segundos (UTC)."""
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("T", " ").rstrip("Z")
    if "+" in s[10:]:
        s = s[: 10 + s[10:].index("+")]
    try:
        from datetime import datetime, timezone

        fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in s else "%Y-%m-%d %H:%M:%S"
        if "." in s:
            head, frac = s.split(".", 1)
            s = f"{head}.{frac[:6]}"
        return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def _f(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


class CatalogManager:
    def __init__(self, history: HistoryBundle, cfg: Config) -> None:
        self.h = history
        self.cfg = cfg
        self.rows: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        for r in history.catalog_rows:
            pid = str(r["product_id"])
            if pid in self.rows:
                continue
            row = dict(r)
            row["is_new"] = False
            row["initial_price_eur"] = _f(r.get("price_eur"))
            self.rows[pid] = row
            self.order.append(pid)
        self.content_index = ContentIndex().fit([self.rows[p] for p in self.order])
        self.content = self.content_index.transform([self.rows[p] for p in self.order])
        self.oos: set[str] = set()
        self.price_change: dict[str, tuple[float, float]] = {}
        self.cold_cache: dict[str, tuple[tuple[str, float], ...]] = {}
        self.version = 0
        self.ease_index = history.ease.index if history.ease is not None else {}
        # Prior de CVR/engagement por subcategoría
        self._stats_cache: dict[str, float] | None = None

    # ------------------------------------------------------------------ updates

    def apply_updates(self, updates: Iterable[Mapping[str, Any]]) -> int:
        """Aplica cambios en orden temporal. Devuelve el número de cambios aplicados."""
        n = 0
        new_rows: list[str] = []
        for u in sorted(updates, key=lambda r: parse_ts(r.get("ts"))):
            pid = str(u.get("product_id", ""))
            if not pid:
                continue
            change = str(u.get("change", "") or "")
            ts = parse_ts(u.get("ts"))
            prev = self.rows.get(pid)
            row = dict(prev) if prev else {"product_id": pid, "is_new": False}
            for k in ("category", "subcategory", "brand", "description"):
                if u.get(k) not in (None, ""):
                    row[k] = str(u[k])
            if u.get("price_eur") not in (None, ""):
                new_price = _f(u["price_eur"], row.get("price_eur", 0.0))
                old_price = _f(row.get("price_eur"), new_price)
                if change == "price_change" and old_price > 0 and abs(new_price - old_price) > 1e-9:
                    self.price_change[pid] = ((old_price - new_price) / old_price, ts)
                row["price_eur"] = new_price
            if "is_active" in u and u["is_active"] is not None:
                row["is_active"] = bool(u["is_active"]) if not isinstance(u["is_active"], str) \
                    else u["is_active"].lower() in ("1", "true")
            row.setdefault("is_active", True)
            if change == "new_product":
                row["is_new"] = True
            if change == "out_of_stock":
                self.oos.add(pid)
            elif change == "back_in_stock":
                self.oos.discard(pid)
            if prev is None:
                row.setdefault("initial_price_eur", _f(row.get("price_eur")))
                self.order.append(pid)
                new_rows.append(pid)
            self.rows[pid] = row
            n += 1
        if new_rows:
            extra = self.content_index.transform([self.rows[p] for p in new_rows])
            self.content = sp.vstack([self.content, extra], format="csr")
            logger.info("Productos nuevos en catálogo: %s", ", ".join(new_rows[:10]))
        return n

    # ------------------------------------------------------------------ build

    def build(self) -> CatalogState:
        t0 = time.perf_counter()
        cfg, h = self.cfg, self.h
        ids = tuple(self.order)
        n = len(ids)
        idx = {p: i for i, p in enumerate(ids)}
        rows = [self.rows[p] for p in ids]

        def codes(key: str) -> tuple[np.ndarray, tuple[str, ...], dict[str, int]]:
            names = sorted({str(r.get(key, "") or "") for r in rows})
            m = {v: i for i, v in enumerate(names)}
            return np.array([m[str(r.get(key, "") or "")] for r in rows], dtype=np.int32), tuple(names), m

        cat, cat_names, _ = codes("category")
        sub, sub_names, sub_codes = codes("subcategory")
        brand, _, brand_codes = codes("brand")
        price = np.array([max(_f(r.get("price_eur")), 0.0) for r in rows])
        logp = np.log1p(price)
        mu = np.zeros(len(sub_names))
        sd = np.ones(len(sub_names))
        for s in range(len(sub_names)):
            m = sub == s
            if m.any():
                mu[s] = logp[m].mean()
                sd[s] = max(logp[m].std(), 0.15)
        price_rel_z = np.clip((logp - mu[sub]) / sd[sub], -3, 3)
        is_new = np.array([bool(r.get("is_new")) for r in rows])
        active = np.array([bool(r.get("is_active", True)) and p not in self.oos for p, r in zip(ids, rows)])

        # --- vecinos de contenido para productos sin historial (C1/C3)
        warm_mask = np.array([(p in self.ease_index) or (h.pop.get(p, 0.0) > 0) for p in ids])
        need = [i for i, p in enumerate(ids) if (not warm_mask[i] or is_new[i]) and p not in self.cold_cache]
        pool = np.where(warm_mask & ~is_new)[0]
        if need and len(pool):
            q = np.array(need)
            res = content_neighbors(
                self.content[q], cat[q], sub[q], brand[q], logp[q],
                self.content[pool], cat[pool], sub[pool], brand[pool], logp[pool],
                top_k=cfg.cold_top_k, min_sim=cfg.cold_min_sim, same_category=cfg.cold_same_category,
            )
            for i, nb in zip(need, res):
                self.cold_cache[ids[i]] = tuple((ids[pool[j]], s) for j, s in nb)
        cold_neighbors = {ids[i]: self.cold_cache.get(ids[i], ()) for i in range(n)
                          if not warm_mask[i] or is_new[i]}

        def inherit(values: np.ndarray, known: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            """Valor de los productos sin dato = media ponderada de sus vecinos (encogida)."""
            out = values.copy()
            for p, nbs in cold_neighbors.items():
                i = idx[p]
                if known[i]:
                    continue
                ws = [(idx[q], s) for q, s in nbs if q in idx and known[idx[q]]]
                if ws:
                    tot = sum(s for _, s in ws)
                    v = sum(values[j] * s for j, s in ws) / tot
                    out[i] = cfg.cold_inherit_shrink * v + (1 - cfg.cold_inherit_shrink) * fallback[i]
                else:
                    out[i] = fallback[i]
            return out

        # --- popularidad prior (log share * N), heredada para los nuevos
        pop_raw = np.array([h.pop.get(p, 0.0) for p in ids])
        known_pop = pop_raw > 0
        share = (pop_raw + 1.0) / (pop_raw + 1.0).sum()
        log_pop = np.log(share * n)
        sub_mean = np.zeros(len(sub_names))
        for s in range(len(sub_names)):
            m = (sub == s) & known_pop
            sub_mean[s] = log_pop[m].mean() if m.any() else log_pop.min()
        log_pop = inherit(log_pop, known_pop, sub_mean[sub])
        pop_share = np.exp(log_pop)
        pop_share = pop_share / pop_share.sum()
        log_pop = np.clip(np.log(pop_share * n), -6, 6)
        subcat_log_pop = np.clip(np.log(np.bincount(sub, weights=pop_share, minlength=len(sub_names))
                                        / np.maximum(np.bincount(sub, minlength=len(sub_names)), 1) * n + 1e-9)[sub], -6, 6)

        # --- engagement IPS suavizado (identidad), heredado para los nuevos
        st = np.array([h.item_stats.get(p, (0.0, 0.0, 0.0, 0.0)) for p in ids], dtype=float).reshape(n, 4)
        imps, imps_w, clicks_w, buys_w = st.T
        known_st = imps > 0
        rate_num = buys_w + 0.1 * clicks_w
        g_rate = rate_num.sum() / max(imps_w.sum(), 1e-9) if known_st.any() else 0.01
        avg_w = imps_w.sum() / max(imps.sum(), 1.0)
        k = cfg.cvr_prior_strength * max(avg_w, 1.0)
        sub_rate = np.full(len(sub_names), g_rate)
        for s in range(len(sub_names)):
            m = (sub == s) & known_st
            if m.any():
                sub_rate[s] = (rate_num[m].sum() + k * g_rate) / (imps_w[m].sum() + k)
        rate = (rate_num + k * sub_rate[sub]) / (imps_w + k)
        rate = inherit(rate, known_st, sub_rate[sub])
        rate = np.clip(rate, 1e-5, 0.999)
        logit = np.log(rate / (1 - rate))
        cvr_logit = np.clip(logit - np.log(g_rate / (1 - g_rate)) if 0 < g_rate < 1 else logit, -4, 4)

        # --- prior por contexto (V4)
        ctx_arrays: dict[tuple[str, str, str], np.ndarray] = {}
        for key, d in h.ctx_prior.items():
            arr = np.zeros(n, dtype=np.float32)
            for p, v in d.items():
                j = idx.get(p)
                if j is not None:
                    arr[j] = v
            mx = arr.max()
            if mx > 0:
                ctx_arrays[key] = arr / mx

        # --- proyección EASE: identidad para productos con fila, vecinos para el resto
        ease_proj = None
        if h.ease is not None:
            rr, cc, vv = [], [], []
            for i, p in enumerate(ids):
                e = self.ease_index.get(p)
                if e is not None:
                    rr.append(i); cc.append(e); vv.append(1.0)
                    continue
                nbs = [(self.ease_index[q], s) for q, s in cold_neighbors.get(p, ()) if q in self.ease_index]
                tot = sum(s for _, s in nbs)
                for e2, s in nbs:
                    rr.append(i); cc.append(e2); vv.append(cfg.cold_inherit_shrink * s / tot)
            ease_proj = sp.csr_matrix((vv, (rr, cc)), shape=(n, len(h.ease.item_ids)), dtype=np.float32)

        pc_rel = np.zeros(n)
        pc_ts = np.zeros(n)
        for p, (rel, ts) in self.price_change.items():
            j = idx.get(p)
            if j is not None:
                pc_rel[j], pc_ts[j] = rel, ts

        active_ids = frozenset(p for p, a in zip(ids, active) if a)
        order = np.lexsort((np.arange(n), -pop_raw))
        global_ranked = tuple(ids[i] for i in order if active[i])

        self.version += 1
        state = CatalogState(
            version=self.version, ids=ids, idx=idx, category=cat, subcategory=sub, brand=brand,
            cat_names=cat_names, subcat_names=sub_names, brand_codes=brand_codes, subcat_codes=sub_codes,
            price=price, log_price=logp, price_rel_z=price_rel_z, active=active, is_new=is_new,
            active_ids=active_ids, global_ranked=global_ranked,
            log_pop_prior=log_pop, pop_share_prior=pop_share, subcat_log_pop=subcat_log_pop,
            cvr_logit=cvr_logit, ctx_arrays=ctx_arrays, ease_proj=ease_proj, content=self.content,
            cold_neighbors=cold_neighbors, price_change_rel=pc_rel, price_change_ts=pc_ts,
            created_at=time.time(),
        )
        logger.debug("CatalogState v%d (%d productos, %d activos) en %.0f ms", self.version, n,
                     len(active_ids), (time.perf_counter() - t0) * 1000)
        return state
