"""Las señales de la Etapa 2, vectorizadas sobre todo el catálogo.

Misma idea que la Etapa 2 (una docena de señales por producto que se combinan con
pesos), pero:
* se calculan para los ~5.000 productos a la vez (NumPy), no para ~100 candidatos;
* cada señal está en una escala fija e independiente de la petición, para que los
  pesos se puedan **aprender** con el feedback (4.2) y ser distintos por segmento (4.3);
* se corrigen N7 (similitud absoluta), N11 (intención con decaimiento), N12 (contexto
  bayesiano) y N13 (nombres y magnitudes: `visto_conv`, semivida real).
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .catalog import parse_ts
from .config import SIGNALS, Config
from .state import LiveView, Snapshot, UserProfile

SIDX = {s: i for i, s in enumerate(SIGNALS)}
LN2 = math.log(2.0)


@dataclass
class Req:
    request_id: str
    user_id: str
    is_new_user: bool
    region: str
    device: str
    page: str
    recent: list[str]
    recent_ts: list[float]
    ts: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_obj(x: Any) -> Any:
    if isinstance(x, str) and x[:1] in "[{":
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def parse_request(raw: Any) -> Req:
    ctx = _as_obj(_get(raw, "context")) or {}
    rv = _as_obj(_get(raw, "recent_views")) or []
    if isinstance(rv, np.ndarray):
        rv = list(rv)
    recent, recent_ts = [], []
    for it in rv:
        if isinstance(it, Mapping):
            if it.get("product_id"):
                recent.append(str(it["product_id"]))
                recent_ts.append(parse_ts(it.get("ts")))
        elif it:
            recent.append(str(it))
            recent_ts.append(0.0)
    nu = _get(raw, "is_new_user", False)
    if isinstance(nu, str):
        nu = nu.lower() in ("1", "true")
    if isinstance(nu, float) and math.isnan(nu):
        nu = False
    return Req(
        request_id=str(_get(raw, "request_id", "") or ""), user_id=str(_get(raw, "user_id", "") or ""),
        is_new_user=bool(nu),
        region=str(_get(ctx, "region", "") or _get(raw, "region", "") or ""),
        device=str(_get(ctx, "device", "") or _get(raw, "device", "") or ""),
        page=str(_get(ctx, "page", "") or _get(raw, "page", "") or ""),
        recent=recent, recent_ts=recent_ts, ts=parse_ts(_get(raw, "ts")),
    )


@dataclass
class Session:
    recent_idx: np.ndarray
    weights: np.ndarray
    profile: UserProfile | None
    has_new_product: bool


def session_weights(req: Req, cat, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Peso de cada vista: 0,72^k × (0,5 + 0,5·2^(−Δt / semivida)), k = posición desde el final."""
    pairs = [(cat.idx[p], t) for p, t in zip(req.recent, req.recent_ts) if p in cat.idx]
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    idx = np.array([i for i, _ in pairs], dtype=np.int64)
    k = len(idx)
    w = cfg.recent_view_decay ** np.arange(k - 1, -1, -1, dtype=float)
    if req.ts:
        dt = np.array([max(req.ts - t, 0.0) if t else np.nan for _, t in pairs])
        tw = np.where(np.isnan(dt), 1.0, 0.5 + 0.5 * np.exp(-LN2 * np.nan_to_num(dt) / cfg.recent_halflife_sec))
        w = w * tw
    return idx, w


def compute_signals(req: Req, snap: Snapshot, cfg: Config, live: LiveView | None = None,
                    use_profile: bool = True, item_idx: np.ndarray | None = None) -> tuple[np.ndarray, Session]:
    cat = snap.catalog
    h = snap.history
    n = cat.n
    S = np.zeros((n, len(SIGNALS)), dtype=np.float32)
    rec_idx, w = session_weights(req, cat, cfg)
    k = len(rec_idx)
    has_new = bool(k and cat.is_new[rec_idx].any())

    # popularidad normalizada dentro de cada subcategoría
    log_pop = live.log_pop if (live is not None and live.n_items == n) else cat.log_pop_prior
    share = np.exp(log_pop)
    nsub = len(cat.subcat_names)
    mx = np.zeros(nsub)
    np.maximum.at(mx, cat.subcategory, share)
    sub_pop_norm = share / np.maximum(mx[cat.subcategory], 1e-12)

    if k:
        page_last = 0.65 if req.page == "product" else 1.0
        in_w = np.zeros(n)
        for v, wv in zip(rec_idx, w):
            in_w[v] = max(in_w[v], wv)
        in_w[rec_idx[-1]] *= page_last
        vc = cat.visto_conv if cat.visto_conv is not None else np.full(n, 0.5)
        S[:, SIDX["visto_conv"]] = in_w * (0.4 + 0.6 * vc)

        lift = np.zeros(n)
        cold_nb = np.zeros(n)
        cold_gr = np.zeros(n)
        rev = np.zeros(n)
        sub_w = np.zeros(nsub)
        trans = h.transitions
        for v, wv in zip(rec_idx, w):
            pid = cat.ids[v]
            sub_w[cat.subcategory[v]] += wv
            lst = trans.get(pid, ())
            if lst:
                top = lst[0][1] or 1.0
                for dst, val in lst:
                    j = cat.idx.get(dst)
                    if j is not None:
                        lift[j] += wv * val / top
            if cat.is_new[v] or len(lst) < 3:
                for q, sim in cat.cold_neighbors.get(pid, ()):
                    j = cat.idx.get(q)
                    if j is None:
                        continue
                    cold_nb[j] += wv * sim
                    l2 = trans.get(q, ())
                    if l2:
                        top2 = l2[0][1] or 1.0
                        for dst, val in l2[:6]:
                            jj = cat.idx.get(dst)
                            if jj is not None and jj != v:
                                cold_gr[jj] += wv * sim * val / top2
            for q, sim in cat.reverse.get(pid, ()):
                j = cat.idx.get(q)
                if j is not None:
                    rev[j] += wv * sim
        S[:, SIDX["lift"]] = lift
        S[:, SIDX["cold_vecino"]] = cold_nb
        S[:, SIDX["cold_injerto"]] = cold_gr
        S[:, SIDX["injerto_inverso"]] = rev
        top2 = np.argsort(-sub_w)[:2]
        sw = np.zeros(nsub)
        if sub_w[top2[0]] > 0:
            sw[top2] = sub_w[top2] / sub_w[top2[0]]
        S[:, SIDX["subcat_sesion"]] = sw[cat.subcategory] * sub_pop_norm

        if cat.ease_proj is not None and h.ease is not None:
            u = np.asarray(cat.ease_proj[rec_idx].T @ w).ravel()
            nz = np.flatnonzero(u)
            if nz.size:
                s = u[nz] @ h.ease.B[nz]
                e = np.asarray(cat.ease_proj @ s).ravel()
                S[:, SIDX["ease"]] = np.clip(e / w.sum() * 10.0, -3, 3)

    prof = None
    if use_profile and not req.is_new_user and req.user_id:
        prof = h.users.get(req.user_id)
    if prof is not None:
        if prof.subcat:
            top = sorted(prof.subcat.items(), key=lambda x: -x[1])[:2]
            arr = np.zeros(nsub)
            for sname, v in top:
                c = cat.subcat_codes.get(sname)
                if c is not None:
                    arr[c] = v
            S[:, SIDX["user_subcat"]] = arr[cat.subcategory] * sub_pop_norm
        if prof.brand:
            col = np.zeros(n)
            for b, v in prof.brand.items():
                c = cat.brand_codes.get(b)
                if c is not None:
                    col[cat.brand == c] = v
            S[:, SIDX["user_marca"]] = col
        if prof.n_buys > 0 and math.isfinite(prof.price_mu):
            sig = max(prof.price_sigma if math.isfinite(prof.price_sigma) else 0.0, 0.35)
            S[:, SIDX["precio_fit"]] = np.exp(-0.5 * ((cat.log_price - prof.price_mu) / sig) ** 2)
    if use_profile and not req.is_new_user and req.user_id and h.intent.get(req.user_id):
        items = h.intent[req.user_id]
        gap = 0.0
        hend = h.meta.get("history_end") if h.meta else None
        if hend and req.ts:
            gap = max((req.ts - parse_ts(hend)) / 86400.0, 0.0)
        top = max(s for _, s, _ in items) or 1.0
        col = np.zeros(n)
        for pid, strength, age in items:
            j = cat.idx.get(pid)
            if j is not None:
                col[j] = strength / top * math.exp(-LN2 * (age + gap) / cfg.intent_halflife_days)
        S[:, SIDX["intencion"]] = col

    ctx = cat.ctx_arrays.get((req.region, req.device, req.page))
    if ctx is None:
        ctx = cat.ctx_arrays.get((req.region, "", req.page))
    if ctx is None:
        ctx = cat.ctx_arrays.get(("", "", req.page))
    if ctx is not None:
        S[:, SIDX["contexto"]] = ctx
    S[:, SIDX["pop"]] = share / share.max()
    if live is not None and live.n_items == n:
        S[:, SIDX["tendencia"]] = np.clip(live.trend, 0, 3) / 3.0
    now = (live.now_ts if live is not None and live.now_ts else 0.0) or req.ts
    if now:
        age = np.maximum(now - cat.price_change_ts, 0.0)
        decay = np.where(cat.price_change_ts > 0, np.exp(-age / (cfg.price_change_tau_min * 60)), 0.0)
        S[:, SIDX["bajada_precio"]] = cat.price_change_rel * decay

    sess = Session(recent_idx=rec_idx, weights=w, profile=prof, has_new_product=has_new)
    if item_idx is not None:
        return S[np.asarray(item_idx, dtype=np.int64)], sess
    return S, sess
