"""Odyseus-ML · variables del modelo residual (mismo código para entrenar y para servir).

Vector por (petición, producto candidato) = las 14 señales del re-ranker (`reranker.FEATURE_NAMES`,
calculadas con `_extract_feature_matrix`, sin tocar) + 20 variables nuevas de contexto, sesión,
producto y posición. Ninguna variable es una identidad (user_id / product_id / marca / región):
todo se puede calcular para un comprador o un producto que acaba de aparecer.

También expone `prior_scores`, réplica de la puntuación de precisión de `_recommend_core`
(`score_p`), que se usa como offset fijo al entrenar la red: logit = b + prior + f(x).
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .recommender import (
    _extract_feature_matrix,
    _parse_request_fields,
    classify_user_segment,
)
from .reranker import FEATURE_NAMES as BASE_FEATURE_NAMES
from .reranker import PositionDebiasedRanker
from .state import RecommenderSnapshotV4

EXTRA_FEATURE_NAMES: tuple[str, ...] = (
    # contexto de la petición
    "page_home",
    "page_product",
    "page_category",
    "device_web",
    "device_ios",
    "device_android",
    # usuario / sesión
    "has_profile",
    "has_views",
    "n_recent_views",
    "last_view_gap",
    # producto x sesión («misma categoría que el último visto» NO entra: en el log antiguo los 10
    # productos mostrados son siempre de esa categoría, así que es constante y no se puede aprender)
    "in_recent_views",
    "is_last_viewed",
    "view_recency",
    "dlogprice_last",
    "abs_dlogprice_last",
    # producto / usuario x producto
    "self_cvr_raw",
    "in_cart",
    "is_recurrent",
    "already_bought",
    # posición (solo informativa al entrenar; al servir se fija en 0 = slot 1)
    "slot",
)

ML_FEATURE_NAMES: tuple[str, ...] = tuple(BASE_FEATURE_NAMES) + EXTRA_FEATURE_NAMES
NUM_BASE = len(BASE_FEATURE_NAMES)
NUM_ML_FEATURES = len(ML_FEATURE_NAMES)
_GAP_NORM = math.log1p(3600.0)


def parse_ts_ms(value: Any) -> float | None:
    """Convierte un ts de RawTree (ISO con T/Z o 'YYYY-MM-DD HH:MM:SS.fffffffff') a epoch ms (UTC)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v != v:  # NaN
            return None
        return v if v > 1e11 else v * 1000.0
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("T", " ")
    if s.endswith("Z"):
        s = s[:-1]
    plus = s.find("+", 10)
    if plus > 0:
        s = s[:plus]
    try:
        if "." in s:
            head, frac = s.split(".", 1)
            frac = (frac + "000000")[:6]
            dt = datetime.strptime(f"{head}.{frac}", "%Y-%m-%d %H:%M:%S.%f")
        elif len(s) <= 10:
            dt = datetime.strptime(s, "%Y-%m-%d")
        else:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc).timestamp() * 1000.0


def _recent_views_raw(request: Any) -> Any:
    if isinstance(request, Mapping):
        return request.get("recent_views")
    return getattr(request, "recent_views", None)


def _last_view_ts_ms(raw_recent: Any) -> float | None:
    """ts de la última vista de la sesión (la más reciente), si viene en la petición."""
    if raw_recent is None:
        return None
    try:
        if len(raw_recent) == 0:
            return None
    except TypeError:
        return None
    last = raw_recent[-1]
    if isinstance(last, Mapping):
        return parse_ts_ms(last.get("ts"))
    return None


def request_has_profile(snapshot: RecommenderSnapshotV4, user_id: str, is_new_user: bool) -> bool:
    """Misma definición de `has_profile` que `_recommend_core`."""
    prof = None if is_new_user else snapshot.user_profiles.get(user_id)
    return prof is not None and bool(
        prof.bought_products
        or prof.intent_items
        or prof.subcat_brand_affinity
        or prof.log_price_mean > 0.0
    )


def build_ml_features(
    snapshot: RecommenderSnapshotV4,
    request: Mapping[str, Any] | Any,
    candidate_ids: Sequence[str],
    *,
    slots: Sequence[int] | None = None,
    request_ts_ms: float | None = None,
    parsed: tuple[str, bool, str, str, str, list[str]] | None = None,
    weights_profile: Any = None,
    base_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Matriz (n_candidatos, NUM_ML_FEATURES) float32.

    `slots`: posición 0..9 en la que se mostró cada producto (entrenamiento). Al servir se deja
    en None y la columna vale 0 (se puntúa «como si fuera al slot 1»).
    `request_ts_ms`: instante de la petición (ts de la fila al entrenar, ahora al servir).
    """
    n = len(candidate_ids)
    if n == 0:
        return np.empty((0, NUM_ML_FEATURES), dtype=np.float32)
    p = parsed if parsed is not None else _parse_request_fields(request)
    user_id, is_new_user, _region, device, page, recent_views = p

    X = np.zeros((n, NUM_ML_FEATURES), dtype=np.float32)
    X[:, :NUM_BASE] = (
        base_matrix
        if base_matrix is not None
        else _extract_feature_matrix(snapshot, request, candidate_ids, p, weights_profile=weights_profile)
    )
    c = NUM_BASE

    # ── contexto / usuario / sesión (constantes para toda la petición) ──────────
    X[:, c + 0] = 1.0 if page == "home" else 0.0
    X[:, c + 1] = 1.0 if page == "product" else 0.0
    X[:, c + 2] = 1.0 if page == "category" else 0.0
    X[:, c + 3] = 1.0 if device == "web" else 0.0
    X[:, c + 4] = 1.0 if device == "ios" else 0.0
    X[:, c + 5] = 1.0 if device == "android" else 0.0
    X[:, c + 6] = 1.0 if request_has_profile(snapshot, user_id, is_new_user) else 0.0
    n_views = len(recent_views)
    X[:, c + 7] = 1.0 if n_views else 0.0
    X[:, c + 8] = min(n_views, 6) / 6.0
    gap = 0.0
    if n_views and request_ts_ms is not None:
        last_ts = _last_view_ts_ms(_recent_views_raw(request))
        if last_ts is not None:
            gap_s = max((request_ts_ms - last_ts) / 1000.0, 0.0)
            gap = min(math.log1p(gap_s) / _GAP_NORM, 1.5)
    X[:, c + 9] = gap

    # ── producto x sesión ───────────────────────────────────────────────────────
    wp_eff = weights_profile if weights_profile is not None else snapshot.weights_profile
    decay = float(getattr(wp_eff, "recency_decay", 0.65)) if wp_eff is not None else 0.65  # como _prepare_request_signals
    view_w: dict[str, float] = {}
    for rev_idx, pid in enumerate(reversed(recent_views)):
        if pid not in view_w:
            view_w[pid] = decay**rev_idx
    last_pid = recent_views[-1] if recent_views else ""
    last_info = snapshot.products.get(last_pid) if last_pid else None
    last_lp = math.log1p(max(last_info.price_eur, 0.0)) if last_info is not None else None

    prof = None if is_new_user else snapshot.user_profiles.get(user_id)
    bought = prof.bought_products if prof is not None else frozenset()
    live_bought = snapshot.live_user_bought.get(user_id, frozenset()) if user_id else frozenset()
    cart_ids = {pid for pid, _ in snapshot.user_carts.get(user_id, ())} if user_id else set()
    rec_ids = {pid for pid, _ in snapshot.user_recurrent.get(user_id, ())} if user_id else set()
    products = snapshot.products
    self_cvr = snapshot.self_cvr

    for i, pid in enumerate(candidate_ids):
        w = view_w.get(pid)
        if w is not None:
            X[i, c + 10] = 1.0
            X[i, c + 12] = w
            if pid == last_pid:
                X[i, c + 11] = 1.0
        info = products.get(pid)
        if info is not None:
            if last_lp is not None:
                d = math.log1p(max(info.price_eur, 0.0)) - last_lp
                d = 3.0 if d > 3.0 else (-3.0 if d < -3.0 else d)
                X[i, c + 13] = d
                X[i, c + 14] = abs(d)
        X[i, c + 15] = min(self_cvr.get(pid, 0.02) * 5.0, 2.0)
        if pid in cart_ids:
            X[i, c + 16] = 1.0
        if pid in rec_ids:
            X[i, c + 17] = 1.0
        if pid in bought or pid in live_bought:
            X[i, c + 18] = 1.0

    if slots is not None:
        X[:, c + 19] = np.asarray(slots, dtype=np.float32) / 9.0
    return X


def prior_scores(
    snapshot: RecommenderSnapshotV4,
    parsed: tuple[str, bool, str, str, str, list[str]],
    candidate_ids: Sequence[str],
    base_matrix: np.ndarray,
    *,
    weights_profile: Any = None,
    config: Any = None,
    now_ms: float | None = None,
) -> np.ndarray:
    """Réplica de `score_p` de `CalypsoRecommenderV4._recommend_core` de Calypso 5.1
    (incluye bonus de carrito, recurrentes, descuento y misma categoría; sin UCB, MMR ni bonus de slot 1).

    Recibe la matriz de 14 señales ya calculada. Se usa como offset del modelo residual.
    `now_ms`: instante de la petición (para la antigüedad del carrito).
    """
    wp = weights_profile if weights_profile is not None else snapshot.weights_profile
    cfg = config if config is not None else snapshot.config_override
    user_id, is_new_user, _, _, _, recent_views = parsed
    has_session_intent = bool(recent_views)
    segment = classify_user_segment(
        is_new_user, has_session_intent, request_has_profile(snapshot, user_id, is_new_user)
    )
    m_seq, m_int, m_aff, m_ctx, m_pop, m_trd, m_cld = wp.segment_multipliers(segment)

    overrides = snapshot.agent_overrides
    ranker = (
        snapshot.ranker
        if snapshot.ranker is not None
        else PositionDebiasedRanker.from_priors(
            prior_weights=wp.prior_weights_array(),
            min_prior_ratios=wp.min_prior_ratios_array(),
            max_prior_multiplier=wp.max_prior_multiplier,
        )
    )
    if wp.w_log_price_norm != 0.0:
        rev_tie_override = None
    elif wp.tie_break_mode == "cheap":
        rev_tie_override = -abs(wp.revenue_tie_weight or -0.08)
    elif wp.tie_break_mode == "conversion":
        rev_tie_override = 0.0
    else:
        rev_tie_override = overrides.revenue_tie_weight
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13 = ranker.resolve_weights_tuple(
        blend_alpha_scale=overrides.ml_blend_alpha_scale,
        reverse_graft_override=overrides.weight_reverse_cold_graft,
        forward_cold_override=overrides.weight_forward_cold,
        revenue_tie_override=rev_tie_override,
    )
    cfg_no_floor = bool(getattr(cfg, "v3_no_subcat_floor", False)) if cfg is not None else False
    cfg_bg = bool(getattr(cfg, "v3_bg_penalty", True)) if cfg is not None else True
    if wp.v3_no_subcat_floor or cfg_no_floor:
        subcat_w, brand_w = w12, w13
    else:
        subcat_w, brand_w = max(w12, wp.subcat_floor_min), max(w13, wp.brand_floor_min)
    v3_bg_penalty = wp.v3_bg_penalty and cfg_bg

    # (5.1 · M6) carrito y recurrentes compiten por puntuación; descuento reciente; misma categoría
    cart_bonus: dict[str, float] = {}
    rec_bonus: dict[str, float] = {}
    if user_id and not is_new_user and cfg is not None:
        cur_ms = time.time() * 1000.0 if now_ms is None else now_ms
        active_ms = cfg.v4_cart_active_minutes * 60_000.0
        max_forgot_ms = cfg.v4_forgotten_max_days * 86_400_000.0
        for c_pid, cart_ts in snapshot.user_carts.get(user_id, ()):
            age_ms = max(cur_ms - cart_ts, 0.0)
            if age_ms <= active_ms:
                if wp.cart_active_score_bonus > 0.0:
                    cart_bonus[c_pid] = wp.cart_active_score_bonus
            elif age_ms <= max_forgot_ms and wp.cart_forgotten_score_bonus > 0.0:
                cart_bonus[c_pid] = wp.cart_forgotten_score_bonus * math.exp(-(age_ms / 86_400_000.0) / 7.0)
        if wp.recurrent_score_bonus > 0.0:
            for r_pid, n_buys in snapshot.user_recurrent.get(user_id, ()):
                if n_buys >= cfg.v4_recurrent_min_purchases:
                    rec_bonus[r_pid] = wp.recurrent_score_bonus
    anchor_info = snapshot.products.get(recent_views[-1]) if recent_views else None
    anchor_cat = anchor_info.category if anchor_info is not None else ""
    same_cat_w = wp.same_cat_bonus_weight
    discount_w = wp.discount_boost_weight
    products = snapshot.products

    out = np.zeros(len(candidate_ids), dtype=np.float64)
    static = snapshot.product_static
    B = base_matrix
    for i, pid in enumerate(candidate_ids):
        s_seq = float(B[i, 0]) * m_seq
        s_self = float(B[i, 1])
        s_intent = float(B[i, 2]) * m_int
        s_ctx = float(B[i, 5]) * m_ctx
        s_rev = float(B[i, 9]) * m_cld
        s_fwd = float(B[i, 10]) * m_cld
        p_info = products.get(pid)
        p_cat = p_info.category if p_info is not None else ""
        rule_bonus = (
            cart_bonus.get(pid, 0.0) + rec_bonus.get(pid, 0.0)
            + discount_w * (p_info.discount_ratio if p_info is not None else 0.0)
        )
        st = static.get(pid)
        if st is None:
            out[i] = (
                w0 * s_seq + w1 * s_self + w2 * s_intent + w5 * s_ctx
                + w6 * float(B[i, 6]) * m_pop + w7 * float(B[i, 7]) * m_trd
                + w9 * s_rev + w10 * s_fwd + rule_bonus
            )
            continue
        exploit_d = st[6]
        same_sc = B[i, 12] > 0.5
        same_br = B[i, 13] > 0.5
        bg = w5 * s_ctx + w6 * (float(B[i, 6]) * m_pop) + w7 * (float(B[i, 7]) * m_trd) + exploit_d
        same_cat = bool(anchor_cat and p_cat == anchor_cat)
        if v3_bg_penalty and has_session_intent and s_seq == 0.0 and s_intent == 0.0 and s_fwd == 0.0 and s_self == 0.0:
            bg *= wp.bg_penalty_same_subcat if same_sc else wp.bg_penalty_diff_subcat
        out[i] = (
            w0 * s_seq + w1 * s_self + w2 * s_intent
            + w3 * (float(B[i, 3]) * m_aff) + w4 * float(B[i, 4])
            + bg + w8 * float(B[i, 8]) + w9 * s_rev + w10 * s_fwd + w11 * float(B[i, 11])
            + (subcat_w if same_sc else (same_cat_w if same_cat else 0.0)) + (brand_w if same_br else 0.0)
            + rule_bonus
        )
    return out
