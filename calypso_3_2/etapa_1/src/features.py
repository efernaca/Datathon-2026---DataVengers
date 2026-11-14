"""Variables (contexto, producto, posición) del modelo de respuesta (enfoque A, A2).

La MISMA función se usa para entrenar (filas de `history_sessions`, solo los 10
productos mostrados) y para servir (los ~5.000 productos del catálogo en cada
petición, A4). Está vectorizada con NumPy: < 2 ms por petición con 5.000 productos.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .catalog import parse_ts
from .state import LiveView, Snapshot, UserProfile

FEATURES: tuple[str, ...] = (
    "slot_log",              # posición (solo en entrenamiento; 0 = slot 1 al predecir)
    "in_recent",             # el producto está en recent_views
    "recent_recency",        # 0,7^posición desde el final si está en recent_views
    "is_last_viewed",
    "same_subcat_last",
    "same_cat_last",
    "same_brand_last",
    "session_subcat_share",  # peso de la sesión en la subcategoría del producto
    "ease",                  # puntuación EASE de la sesión, normalizada
    "trans_lift",            # transiciones dirigidas con lift, normalizadas
    "content_sim_last",      # coseno TF-IDF con la última vista
    "price_dist_last",       # |log p - log p_última|
    "log_pop",               # popularidad (histórica o en vivo suavizada)
    "subcat_log_pop",
    "item_cvr",              # engagement IPS suavizado (identidad de producto)
    "price_rel_z",           # precio relativo dentro de la subcategoría
    "log_price",
    "is_new_product",
    "ctx_pop",               # prior por contexto con IPS (V4)
    "user_subcat_aff",       # V1
    "user_brand_aff",
    "user_price_fit",
    "user_has_profile",
    "already_bought",
    "is_new_user",
    "n_recent_log",
    "x_product_same_subcat", # cruces con la página
    "x_product_in_recent",
    "x_home_log_pop",
    "x_nosession_ctx_pop",
)
FIDX = {n: i for i, n in enumerate(FEATURES)}
IDENTITY_FEATURES = frozenset({"item_cvr", "user_subcat_aff", "user_brand_aff",
                               "user_price_fit", "user_has_profile"})
RECENT_DECAY = 0.7


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
    region = str(_get(ctx, "region", "") or _get(raw, "region", "") or "")
    device = str(_get(ctx, "device", "") or _get(raw, "device", "") or "")
    page = str(_get(ctx, "page", "") or _get(raw, "page", "") or "")
    rv = _as_obj(_get(raw, "recent_views")) or []
    if isinstance(rv, np.ndarray):
        rv = list(rv)
    recent, recent_ts = [], []
    for it in rv:
        if isinstance(it, Mapping):
            pid = it.get("product_id")
            if pid:
                recent.append(str(pid))
                recent_ts.append(parse_ts(it.get("ts")))
        elif it:
            recent.append(str(it))
            recent_ts.append(0.0)
    nu = _get(raw, "is_new_user", False)
    if isinstance(nu, str):
        nu = nu.lower() in ("1", "true")
    return Req(
        request_id=str(_get(raw, "request_id", "") or ""),
        user_id=str(_get(raw, "user_id", "") or ""),
        is_new_user=bool(nu) if nu is not None and not (isinstance(nu, float) and math.isnan(nu)) else False,
        region=region, device=device, page=page, recent=recent, recent_ts=recent_ts,
        ts=parse_ts(_get(raw, "ts")),
    )


@dataclass
class SessionInfo:
    recent_idx: np.ndarray        # índices en el catálogo (cronológico)
    weights: np.ndarray           # peso por vista (más reciente = 1)
    cat_weights: np.ndarray       # peso de la sesión por categoría
    profile: UserProfile | None
    is_cold: bool


def compute_features(
    req: Req,
    snap: Snapshot,
    live: LiveView | None = None,
    item_idx: np.ndarray | None = None,
    slots: np.ndarray | None = None,
    use_profile: bool = True,
) -> tuple[np.ndarray, SessionInfo]:
    cat = snap.catalog
    n = cat.n
    sel = np.arange(n) if item_idx is None else np.asarray(item_idx, dtype=np.int64)
    m = len(sel)
    X = np.zeros((m, len(FEATURES)), dtype=np.float32)

    # ---- sesión
    rec = [cat.idx[p] for p in req.recent if p in cat.idx]
    rec_idx = np.array(rec, dtype=np.int64)
    k = len(rec_idx)
    w = RECENT_DECAY ** np.arange(k - 1, -1, -1) if k else np.zeros(0)
    cat_w = np.bincount(cat.category[rec_idx], weights=w, minlength=len(cat.cat_names)) if k else \
        np.zeros(len(cat.cat_names))
    is_cold = req.is_new_user or bool(k and cat.is_new[rec_idx].any())
    page = req.page

    if k:
        last = rec_idx[-1]
        rec_pos = np.full(n, -1.0)
        rec_pos[rec_idx] = w          # si un producto aparece varias veces, queda la vista más reciente
        rp = rec_pos[sel]
        in_rec = rp > 0
        X[:, FIDX["in_recent"]] = in_rec
        X[:, FIDX["recent_recency"]] = np.where(in_rec, rp, 0.0)
        X[:, FIDX["is_last_viewed"]] = sel == last
        X[:, FIDX["same_subcat_last"]] = cat.subcategory[sel] == cat.subcategory[last]
        X[:, FIDX["same_cat_last"]] = cat.category[sel] == cat.category[last]
        X[:, FIDX["same_brand_last"]] = cat.brand[sel] == cat.brand[last]
        sub_w = np.bincount(cat.subcategory[rec_idx], weights=w, minlength=len(cat.subcat_names))
        X[:, FIDX["session_subcat_share"]] = (sub_w / w.sum())[cat.subcategory[sel]]
        X[:, FIDX["content_sim_last"]] = (cat.content[sel] @ cat.content[last].T).toarray().ravel()
        X[:, FIDX["price_dist_last"]] = np.abs(cat.log_price[sel] - cat.log_price[last])
        X[:, FIDX["n_recent_log"]] = math.log1p(k)

        # EASE: u = Σ w_v · proj[v];  s = u·B;  score = proj · s   (cubre nuevos en ambos sentidos)
        # Normalizado por Σw (no por el máximo de la petición) para que el valor de un
        # producto no dependa de qué otros productos se puntúan (entrenamiento = 10, servicio = todos).
        if cat.ease_proj is not None and snap.history.ease is not None:
            u = np.asarray(cat.ease_proj[rec_idx].T @ w).ravel()
            nz = np.flatnonzero(u)
            if nz.size:
                proj = cat.ease_proj[sel]
                cols = np.unique(proj.indices)
                if cols.size:
                    s_cols = u[nz] @ snap.history.ease.B[np.ix_(nz, cols)]
                    s = np.zeros(proj.shape[1], dtype=np.float64)
                    s[cols] = s_cols
                    e = np.asarray(proj @ s).ravel()
                    X[:, FIDX["ease"]] = np.clip(e / w.sum() * 10.0, -3, 3)

        # Transiciones dirigidas con lift (V3); los productos fríos usan las de sus vecinos
        tl = np.zeros(n)
        trans = snap.history.transitions
        for v, wv in zip(rec_idx, w):
            pid = cat.ids[v]
            srcs = [(pid, 1.0)] if pid in trans else \
                [(q, s * 0.6) for q, s in cat.cold_neighbors.get(pid, ())]
            for src, ws in srcs:
                for dst, lift in trans.get(src, ()):
                    j = cat.idx.get(dst)
                    if j is not None:
                        tl[j] += wv * ws * lift
        X[:, FIDX["trans_lift"]] = np.log1p(tl[sel] / w.sum())
        X[:, FIDX["x_product_same_subcat"]] = X[:, FIDX["same_subcat_last"]] * (page == "product")
        X[:, FIDX["x_product_in_recent"]] = X[:, FIDX["in_recent"]] * (page == "product")

    # ---- producto
    log_pop = (live.log_pop if (live is not None and live.n_items == n) else cat.log_pop_prior)[sel]
    X[:, FIDX["log_pop"]] = log_pop
    X[:, FIDX["subcat_log_pop"]] = cat.subcat_log_pop[sel]
    X[:, FIDX["item_cvr"]] = cat.cvr_logit[sel]
    X[:, FIDX["price_rel_z"]] = cat.price_rel_z[sel]
    X[:, FIDX["log_price"]] = cat.log_price[sel] / 5.0
    X[:, FIDX["is_new_product"]] = cat.is_new[sel]

    # ---- contexto
    ctx = cat.ctx_arrays.get((req.region, req.device, page))
    if ctx is None:
        ctx = cat.ctx_arrays.get(("", "", page))
    if ctx is not None:
        X[:, FIDX["ctx_pop"]] = ctx[sel]
        if not k:
            X[:, FIDX["x_nosession_ctx_pop"]] = ctx[sel]
    X[:, FIDX["x_home_log_pop"]] = log_pop * (page == "home")
    X[:, FIDX["is_new_user"]] = float(req.is_new_user)

    # ---- usuario (V1)
    prof = None
    if use_profile and not req.is_new_user and req.user_id:
        prof = snap.history.users.get(req.user_id)
    if prof is not None:
        X[:, FIDX["user_has_profile"]] = 1.0
        if prof.subcat:
            arr = np.zeros(len(cat.subcat_names))
            for s, v in prof.subcat.items():
                c = cat.subcat_codes.get(s)
                if c is not None:
                    arr[c] = v
            X[:, FIDX["user_subcat_aff"]] = arr[cat.subcategory[sel]]
        if prof.brand:
            bsel = cat.brand[sel]
            col = np.zeros(m)
            for b, v in prof.brand.items():
                c = cat.brand_codes.get(b)
                if c is not None:
                    col[bsel == c] = v
            X[:, FIDX["user_brand_aff"]] = col
        if prof.n_buys > 0 and math.isfinite(prof.price_mu):
            sig = max(prof.price_sigma if math.isfinite(prof.price_sigma) else 0.0, 0.35)
            X[:, FIDX["user_price_fit"]] = np.exp(-0.5 * ((cat.log_price[sel] - prof.price_mu) / sig) ** 2)
        if prof.bought:
            bidx = [cat.idx[p] for p in prof.bought if p in cat.idx]
            if bidx:
                mask = np.zeros(n, dtype=bool)
                mask[bidx] = True
                X[:, FIDX["already_bought"]] = mask[sel]

    if slots is not None:
        X[:, FIDX["slot_log"]] = np.log(np.maximum(np.asarray(slots, dtype=float), 1.0))

    return X, SessionInfo(recent_idx=rec_idx, weights=w, cat_weights=cat_w, profile=prof, is_cold=is_cold)
