"""Slot-Gated Calypso-2.1 recommender with precomputed static item vectors (<1 ms/req)."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .agent_scratch import CalypsoAgent
from .business_rules import (
    build_cart_and_recurrent_maps,
    guarantee_items,
    sort_by_price,
    split_cart_items,
)
from .bandits import PositionAwareBandit
from .cold_start import ColdStartIndexV4
from .config import ConfigV4
from .ease import fit_ease_from_transitions
from .guardrails import ensure_valid_submission
from .reranker import NUM_FEATURES, PositionDebiasedRanker, build_ips_training_dataset
from .state import ProductInfo, RecommenderSnapshotV4, UserProfile
from .weights_profile import WeightsProfile


def classify_user_segment(is_new_user: bool, has_views: bool, has_profile: bool) -> str:
    """Classify request into one of 5 canonical segments for modular signal weighting."""
    if is_new_user:
        return "new_with_views" if has_views else "new_no_views"
    if not has_profile:
        return "new_with_views" if has_views else "known_cold"
    return "known_with_views" if has_views else "known_with_history"


def _compute_price_compat(
    lp: float,
    mean_lp: float,
    std_lp: float,
    *,
    z2_clip: float = 6.0,
    cheap_asymmetry: float = 0.0,
) -> float:
    z = (lp - mean_lp) / std_lp
    z2 = z * z
    compat = math.exp(-0.5 * (z2 if z2 < z2_clip else z2_clip))
    if cheap_asymmetry > 0.0 and z < 0.0:
        compat *= 1.0 + cheap_asymmetry * min(-z, 2.0) * 0.5
    return compat


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        res = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(res) else res


def extract_recent_product_ids(raw_recent: Any) -> list[str]:
    """Parse recent_views accepting either RawTree dict items or plain string IDs."""
    out: list[str] = []
    if raw_recent is None:
        return out
    try:
        if len(raw_recent) == 0:
            return out
    except TypeError:
        pass
    for item in raw_recent:
        if isinstance(item, Mapping):
            pid = item.get("product_id")
            if pid:
                out.append(str(pid).strip())
        elif item:
            s_pid = str(item).strip()
            if s_pid:
                out.append(s_pid)
    return out


def _parse_request_fields(request: Any) -> tuple[str, bool, str, str, str, list[str]]:
    """Extract (user_id, is_new_user, region, device, page, recent_views) from dict or row."""
    if isinstance(request, Mapping):
        user_id = str(request.get("user_id") or "").strip()
        is_new_user = bool(request.get("is_new_user", False))
        ctx = request.get("context")
        raw_recent = request.get("recent_views")
        if isinstance(ctx, Mapping):
            region = str(ctx.get("region") or "").strip()
            device = str(ctx.get("device") or "").strip()
            page = str(ctx.get("page") or "").strip()
        else:
            region = str(request.get("region") or "").strip()
            device = str(request.get("device") or "").strip()
            page = str(request.get("page") or "").strip()
    else:
        user_id = str(getattr(request, "user_id", "") or "").strip()
        is_new_user = bool(getattr(request, "is_new_user", False))
        ctx = getattr(request, "context", None)
        raw_recent = getattr(request, "recent_views", None)
        if isinstance(ctx, Mapping):
            region = str(ctx.get("region") or "").strip()
            device = str(ctx.get("device") or "").strip()
            page = str(ctx.get("page") or "").strip()
        else:
            region = str(getattr(request, "region", "") or "").strip()
            device = str(getattr(request, "device", "") or "").strip()
            page = str(getattr(request, "page", "") or "").strip()

    return user_id, is_new_user, region, device, page, extract_recent_product_ids(raw_recent)


def _build_products(
    catalog_rows: Iterable[Mapping[str, Any]],
    catalog_updates_rows: Iterable[Mapping[str, Any]],
    out_of_stock_ids: frozenset[str],
) -> tuple[dict[str, ProductInfo], frozenset[str], frozenset[str], float]:
    """Merge shop.catalog with incremental shop.catalog_updates and stock state."""
    products: dict[str, ProductInfo] = {}
    max_log_price = 1.0

    for row in catalog_rows:
        pid = str(row.get("product_id") or "").strip()
        if not pid:
            continue
        is_active = bool(row.get("is_active", True)) and pid not in out_of_stock_ids
        price = max(_safe_float(row.get("price_eur"), 0.0), 0.0)
        if price > 0.0:
            max_log_price = max(max_log_price, math.log1p(price))
        products[pid] = ProductInfo(
            product_id=pid,
            category=str(row.get("category") or "").strip(),
            subcategory=str(row.get("subcategory") or "").strip(),
            brand=str(row.get("brand") or "").strip(),
            price_eur=price,
            description=str(row.get("description") or "").strip(),
            is_active=is_active,
            is_new=bool(row.get("is_new_product", False)),
        )

    for row in catalog_updates_rows:
        pid = str(row.get("product_id") or "").strip()
        if not pid:
            continue
        existing = products.get(pid)
        price_raw = row.get("price_eur")
        if price_raw is None:
            price_raw = row.get("initial_price")
        price = (
            max(_safe_float(price_raw, 0.0), 0.0)
            if price_raw is not None
            else (existing.price_eur if existing else 0.0)
        )
        if price > 0.0:
            max_log_price = max(max_log_price, math.log1p(price))

        is_active_flag = row.get("is_active")
        if is_active_flag is None:
            is_active = (existing.is_active if existing else True) and (pid not in out_of_stock_ids)
        else:
            is_active = bool(is_active_flag) and (pid not in out_of_stock_ids)

        discount_ratio = existing.discount_ratio if existing else 0.0
        if existing is not None and existing.price_eur > 0.0 and price < existing.price_eur:
            discount_ratio = max(0.0, min((existing.price_eur - price) / existing.price_eur, 0.70))

        products[pid] = ProductInfo(
            product_id=pid,
            category=str(row.get("category") or (existing.category if existing else "")).strip(),
            subcategory=str(row.get("subcategory") or (existing.subcategory if existing else "")).strip(),
            brand=str(row.get("brand") or (existing.brand if existing else "")).strip(),
            price_eur=price,
            description=str(row.get("description") or (existing.description if existing else "")).strip(),
            is_active=is_active,
            is_new=bool(row.get("is_new_product", existing.is_new if existing else True)),
            discount_ratio=discount_ratio,
        )

    active_ids = frozenset(pid for pid, p in products.items() if p.is_active)
    new_product_ids = frozenset(pid for pid, p in products.items() if p.is_active and p.is_new)
    return products, active_ids, new_product_ids, max_log_price


def _build_contextual_maps(
    context_rows: Iterable[Mapping[str, Any]],
    active_ids: frozenset[str],
    *,
    prior_strength: float = 15.0,
    top_k: int = 25,
) -> tuple[
    dict[tuple[str, str, str], tuple[tuple[str, float], ...]],
    dict[tuple[str, str], tuple[tuple[str, float], ...]],
    dict[tuple[str], tuple[tuple[str, float], ...]],
    dict[str, float],
]:
    cnt_3: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cnt_2: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cnt_1: dict[tuple[str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    global_buys: dict[str, float] = defaultdict(float)

    tot_3: dict[tuple[str, str, str], float] = defaultdict(float)
    tot_2: dict[tuple[str, str], float] = defaultdict(float)
    tot_1: dict[tuple[str], float] = defaultdict(float)
    tot_global = 0.0

    for row in context_rows:
        pid = str(row.get("product_id") or "").strip()
        if pid not in active_ids:
            continue
        buys = max(_safe_float(row.get("buys"), 0.0), 0.0)
        if buys <= 0.0:
            continue
        region = str(row.get("region") or "").strip()
        device = str(row.get("device") or "").strip()
        page = str(row.get("page") or "").strip()

        cnt_3[(region, device, page)][pid] += buys
        cnt_2[(region, device)][pid] += buys
        cnt_1[(page,)][pid] += buys
        global_buys[pid] += buys

        tot_3[(region, device, page)] += buys
        tot_2[(region, device)] += buys
        tot_1[(page,)] += buys
        tot_global += buys

    global_share: dict[str, float] = {}
    if tot_global > 0.0:
        for pid, b in global_buys.items():
            global_share[pid] = b / tot_global

    def _finalize(
        bucket_map: dict[Any, dict[str, float]],
        totals_map: dict[Any, float],
    ) -> dict[Any, tuple[tuple[str, float], ...]]:
        out: dict[Any, tuple[tuple[str, float], ...]] = {}
        for key, item_counts in bucket_map.items():
            n_bucket = totals_map[key]
            denom = n_bucket + prior_strength
            scored: list[tuple[float, str]] = []
            for pid, cnt in item_counts.items():
                rate = (cnt + prior_strength * global_share.get(pid, 0.0)) / denom
                scored.append((rate, pid))
            scored.sort(reverse=True)
            top = scored[:top_k]
            if not top or top[0][0] <= 0.0:
                continue
            max_r = top[0][0]
            out[key] = tuple((pid, r / max_r) for r, pid in top)
        return out

    return (
        _finalize(cnt_3, tot_3),
        _finalize(cnt_2, tot_2),
        _finalize(cnt_1, tot_1),
        global_share,
    )


def build_product_static_table(
    products: Mapping[str, ProductInfo],
    active_ids: frozenset[str],
    global_pop_norm: Mapping[str, float],
    live_trend_norm: Mapping[str, float],
    max_log_price: float,
    bandit: PositionAwareBandit,
    bandit_state: Any,
    ucb_alpha: float,
) -> dict[str, tuple[str, str, float, float, float, float, float, float]]:
    """Precompute static per-active-product FastLoop lookup tuples."""
    max_lp = max(max_log_price, 1e-6)
    table: dict[str, tuple[str, str, float, float, float, float, float, float]] = {}
    for pid in active_ids:
        info = products.get(pid)
        if info is None:
            continue
        is_new_flag = 1.0 if info.is_new else 0.0
        lp_norm = math.log1p(max(info.price_eur, 0.0)) / max_lp
        exploit_delta, ucb_bonus = bandit.score_components(
            pid,
            info.is_new,
            bandit_state,
            ucb_alpha,
        )
        table[pid] = (
            info.subcategory,
            info.brand,
            global_pop_norm.get(pid, 0.0),
            live_trend_norm.get(pid, 0.0),
            is_new_flag,
            lp_norm,
            exploit_delta,
            ucb_bonus,
        )
    return table


def _prepare_request_signals(
    snapshot: RecommenderSnapshotV4,
    parsed: tuple[str, bool, str, str, str, list[str]],
    *,
    recency_decay: float = 0.65,
    price_std_floor: float = 0.45,
    weights_profile: WeightsProfile | None = None,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    Mapping[tuple[str, str], float],
    bool,
    float,
    float,
    str,
    str,
    dict[str, Any],
]:
    """Compute request-level lookup maps shared identically by training and FastLoop."""
    user_id, is_new_user, region, device, page, recent_views = parsed
    wp = weights_profile if weights_profile is not None else snapshot.weights_profile

    seq_scores: dict[str, float] = {}
    fwd_cold_scores: dict[str, float] = {}
    rev_graft_scores: dict[str, float] = {}
    self_view_scores: dict[str, float] = {}

    decay = wp.recency_decay if wp is not None else recency_decay
    last_prod_factor = wp.view_last_product_page_factor if wp is not None else 0.65
    view_base = wp.view_base_bonus if wp is not None else 0.0
    for rev_idx, seen_pid in enumerate(reversed(recent_views)):
        w_rec = decay**rev_idx
        if seen_pid in snapshot.active_ids and seen_pid not in self_view_scores:
            page_factor = last_prod_factor if (page == "product" and rev_idx == 0) else 1.0
            base_cvr = snapshot.self_cvr.get(seen_pid, 0.02)
            if view_base > 0.0:
                pop_f = 0.45 + 0.55 * snapshot.global_pop_norm.get(seen_pid, 0.3)
                cvr_f = view_base * 0.20 + (1.0 - view_base) * base_cvr
                self_view_scores[seen_pid] = cvr_f * pop_f * w_rec * page_factor
            else:
                self_view_scores[seen_pid] = base_cvr * w_rec * page_factor

        trans_nbs = snapshot.transitions.get(seen_pid)
        if trans_nbs:
            for dst_pid, weight in trans_nbs:
                if dst_pid in snapshot.active_ids:
                    seq_scores[dst_pid] = seq_scores.get(dst_pid, 0.0) + w_rec * weight

        if not trans_nbs or len(trans_nbs) < 3:
            cold_nbs = snapshot.cold_neighbors.get(seen_pid)
            if cold_nbs:
                for warm_nb, sim in cold_nbs:
                    if warm_nb in snapshot.active_ids:
                        cand_s = w_rec * sim * 0.60
                        if cand_s > fwd_cold_scores.get(warm_nb, 0.0):
                            fwd_cold_scores[warm_nb] = cand_s
                    for nb_dst, nb_w in snapshot.transitions.get(warm_nb, ()):
                        if nb_dst in snapshot.active_ids:
                            cand_d = w_rec * sim * nb_w * 0.45
                            if cand_d > fwd_cold_scores.get(nb_dst, 0.0):
                                fwd_cold_scores[nb_dst] = cand_d

        for new_pid, sim in snapshot.reverse_cold_neighbors.get(seen_pid, ()):
            if new_pid in snapshot.active_ids:
                cand_r = w_rec * sim
                if cand_r > rev_graft_scores.get(new_pid, 0.0):
                    rev_graft_scores[new_pid] = cand_r

    ctx_scores: dict[str, float] = {}
    for bucket, weight_scale in (
        (snapshot.ctx_3.get((region, device, page), ()), 1.0),
        (snapshot.ctx_2.get((region, device), ()), 0.70),
        (snapshot.ctx_1.get((page,), ()), 0.45),
    ):
        for pid, score in bucket:
            if pid not in ctx_scores:
                ctx_scores[pid] = score * weight_scale

    profile = None if is_new_user else snapshot.user_profiles.get(user_id)
    intent_scores: dict[str, float] = {}
    affinity_map: Mapping[tuple[str, str], float] = {}
    has_price_hist = False
    mean_lp = 0.0
    std_lp = 0.5
    if profile is not None:
        if profile.intent_items:
            intent_scores = dict(profile.intent_items)
        affinity_map = profile.subcat_brand_affinity
        if profile.log_price_mean > 0.0:
            has_price_hist = True
            mean_lp = profile.log_price_mean
            std_lp = max(profile.log_price_std, price_std_floor)

    anchor_cat = ""
    anchor_subcat = ""
    anchor_brand = ""
    if recent_views:
        last_info = snapshot.products.get(recent_views[-1])
        if last_info is not None:
            anchor_cat = last_info.category
            anchor_subcat = last_info.subcategory
            anchor_brand = last_info.brand

    return (
        seq_scores,
        self_view_scores,
        intent_scores,
        ctx_scores,
        rev_graft_scores,
        fwd_cold_scores,
        affinity_map,
        has_price_hist,
        mean_lp,
        std_lp,
        anchor_subcat,
        anchor_brand,
        {"anchor_cat": anchor_cat},
    )


def _extract_feature_matrix(
    snapshot: RecommenderSnapshotV4,
    request: Mapping[str, Any] | Any,
    candidate_ids: Sequence[str],
    parsed: tuple[str, bool, str, str, str, list[str]] | None = None,
    *,
    weights_profile: WeightsProfile | None = None,
) -> np.ndarray:
    """Vectorized 14-feature builder used cleanly for IPS Re-Ranker training."""
    n = len(candidate_ids)
    if n == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float64)

    wp = weights_profile if weights_profile is not None else snapshot.weights_profile
    std_floor = wp.price_std_floor if wp is not None else 0.45
    z2_clip = wp.price_z2_clip if wp is not None else 6.0
    cheap_asym = wp.price_cheap_asymmetry if wp is not None else 0.0

    p = parsed if parsed is not None else _parse_request_fields(request)
    (
        seq_scores,
        self_view_scores,
        intent_scores,
        ctx_scores,
        rev_graft_scores,
        fwd_cold_scores,
        affinity_map,
        has_price_hist,
        mean_lp,
        std_lp,
        anchor_subcat,
        anchor_brand,
        _,
    ) = _prepare_request_signals(
        snapshot,
        p,
        price_std_floor=std_floor,
        weights_profile=wp,
    )

    max_lp = max(snapshot.max_log_price, 1e-6)
    X = np.zeros((n, NUM_FEATURES), dtype=np.float64)

    for idx, pid in enumerate(candidate_ids):
        st = snapshot.product_static.get(pid)
        seq_v = seq_scores.get(pid, 0.0)
        X[idx, 0] = seq_v if seq_v < 1.8 else 1.8
        sv_v = self_view_scores.get(pid, 0.0) * 5.0
        X[idx, 1] = sv_v if sv_v < 1.2 else 1.2
        X[idx, 2] = intent_scores.get(pid, 0.0)

        if st is not None:
            sc, br, gpop, ltrend, is_new_f, lp_norm, _, _ = st
            aff_val = affinity_map.get((sc, br), 0.0)
            if aff_val == 0.0 and affinity_map:
                aff_val = affinity_map.get((sc, ""), 0.0) * 0.6
            X[idx, 3] = aff_val

            if has_price_hist and lp_norm > 0.0:
                lp = lp_norm * max_lp
                X[idx, 4] = _compute_price_compat(
                    lp,
                    mean_lp,
                    std_lp,
                    z2_clip=z2_clip,
                    cheap_asymmetry=cheap_asym,
                )
            X[idx, 6] = gpop
            X[idx, 7] = ltrend
            X[idx, 8] = is_new_f
            X[idx, 11] = lp_norm
            X[idx, 12] = 1.0 if (anchor_subcat and sc == anchor_subcat) else 0.0
            X[idx, 13] = 1.0 if (anchor_brand and br == anchor_brand) else 0.0
        else:
            X[idx, 6] = snapshot.global_pop_norm.get(pid, 0.0)
            X[idx, 7] = snapshot.live_trend_norm.get(pid, 0.0)

        X[idx, 5] = ctx_scores.get(pid, 0.0)
        X[idx, 9] = rev_graft_scores.get(pid, 0.0)
        X[idx, 10] = fwd_cold_scores.get(pid, 0.0)

    return X


def build_snapshot_v4_from_records(
    catalog_rows: Iterable[Mapping[str, Any]],
    *,
    catalog_updates_rows: Iterable[Mapping[str, Any]] = (),
    popularity_rows: Iterable[Mapping[str, Any]] = (),
    context_rows: Iterable[Mapping[str, Any]] = (),
    transition_rows: Iterable[Mapping[str, Any]] = (),
    user_profile_rows: Iterable[Mapping[str, Any]] = (),
    user_intent_rows: Iterable[Mapping[str, Any]] = (),
    user_affinity_rows: Iterable[Mapping[str, Any]] = (),
    user_bought_rows: Iterable[Mapping[str, Any]] = (),
    out_of_stock_ids: Iterable[str] = (),
    live_trend_rows: Iterable[Mapping[str, Any]] = (),
    live_bought_rows: Iterable[Mapping[str, Any]] = (),
    feedback_rows: Iterable[Mapping[str, Any]] = (),
    training_session_rows: Iterable[Mapping[str, Any]] = (),
    config: ConfigV4 | None = None,
    version: int = 1,
    agent: CalypsoAgent | None = None,
    cart_hist_rows: Iterable[Mapping[str, Any]] = (),
    recurrent_hist_rows: Iterable[Mapping[str, Any]] = (),
    live_cart_event_rows: Iterable[Mapping[str, Any]] = (),
    feedback_purchase_rows: Iterable[Mapping[str, Any]] = (),
    cart_state: tuple[Any, Any, Any] | None = None,
    weights_profile: WeightsProfile | None = None,
) -> tuple[RecommenderSnapshotV4, list[dict[str, Any]]]:
    """Construct complete RecommenderSnapshotV4 with precomputed FastLoop vectors."""
    cfg = config or ConfigV4()
    wp = weights_profile if weights_profile is not None else cfg.load_weights_profile()
    oos_set = frozenset(str(x).strip() for x in out_of_stock_ids if x)
    products, active_ids, new_product_ids, max_log_price = _build_products(
        catalog_rows,
        catalog_updates_rows,
        oos_set,
    )

    pop_rows_list = list(popularity_rows)
    trans_rows_list = list(transition_rows)

    self_cvr: dict[str, float] = {}
    raw_pop: dict[str, float] = {}
    hist_purchases: dict[str, float] = {}
    for row in pop_rows_list:
        pid = str(row.get("product_id") or "").strip()
        if pid not in active_ids:
            continue
        views = _safe_float(row.get("views"), 0.0)
        purchases = max(_safe_float(row.get("purchases"), 0.0), 0.0)
        cvr_val = row.get("self_cvr")
        self_cvr[pid] = (
            _safe_float(cvr_val, 0.02)
            if cvr_val is not None
            else (purchases + 0.5) / (views + 25.0)
        )
        hist_purchases[pid] = purchases
        wpop = max(_safe_float(row.get("weighted_pop"), 0.0), 0.0)
        if wpop > 0.0:
            raw_pop[pid] = math.log1p(wpop)

    max_pop = max(raw_pop.values(), default=1.0) or 1.0
    log_pop_norm = {pid: v / max_pop for pid, v in raw_pop.items()}

    raw_live: dict[str, float] = {}
    live_buys_equiv: dict[str, float] = {}
    for row in live_trend_rows:
        pid = str(row.get("product_id") or "").strip()
        if pid in active_ids:
            sc = max(_safe_float(row.get("live_score"), 0.0), 0.0)
            if sc > 0.0:
                raw_live[pid] = math.log1p(sc)
            if "live_purchases" in row or "live_carts" in row:
                lp_cnt = max(_safe_float(row.get("live_purchases"), 0.0), 0.0)
                lc_cnt = max(_safe_float(row.get("live_carts"), 0.0), 0.0)
                eq = lp_cnt + 0.25 * lc_cnt
            else:
                eq = 0.15 * sc
            if eq > 0.0:
                live_buys_equiv[pid] = eq
    max_live = max(raw_live.values(), default=1.0) or 1.0
    live_trend_norm = {pid: v / max_live for pid, v in raw_live.items()}

    mix_buys: dict[str, float] = {
        pid: hist_purchases.get(pid, 0.0) + wp.live_pop_mix_weight * live_buys_equiv.get(pid, 0.0)
        for pid in active_ids
    }
    if wp.use_sales_rank_base_rate and active_ids:
        ranked_by_sales = sorted(
            active_ids,
            key=lambda pid: (mix_buys[pid], raw_pop.get(pid, 0.0), pid),
            reverse=True,
        )
        denom_rank = math.log(max(len(ranked_by_sales) + 1, 100))
        blend_w = min(max(wp.sales_rank_blend_weight, 0.0), 1.0)
        global_pop_norm: dict[str, float] = {}
        for rank_idx, pid in enumerate(ranked_by_sales, start=1):
            has_signal = mix_buys[pid] > 0.0 or raw_pop.get(pid, 0.0) > 0.0
            sales_rank_norm = max(0.0, 1.0 - math.log(rank_idx) / denom_rank) if has_signal else 0.0
            global_pop_norm[pid] = (
                (1.0 - blend_w) * log_pop_norm.get(pid, 0.0) + blend_w * sales_rank_norm
            )
    else:
        global_pop_norm = dict(log_pop_norm)

    raw_trans: dict[str, list[tuple[str, float]]] = defaultdict(list)
    gamma = min(max(wp.transition_dest_pop_gamma, 0.0), 1.0)
    for row in trans_rows_list:
        src = str(row.get("src_product") or "").strip()
        dst = str(row.get("dst_product") or "").strip()
        if not src or dst not in active_ids or src == dst:
            continue
        raw_score = max(_safe_float(row.get("raw_transition_score", row.get("score")), 0.0), 0.0)
        if raw_score <= 0.0:
            continue
        dst_pop = max(math.expm1(raw_pop.get(dst, 0.0)), 0.0)
        lift_score = raw_score / ((dst_pop + cfg.lift_prior_pop) ** cfg.lift_beta)
        if gamma > 0.0:
            lift_score *= (1.0 - gamma) + gamma * global_pop_norm.get(dst, 0.25)
        raw_trans[src].append((dst, lift_score))

    ease_w = min(max(float(wp.v3_ease_weight if weights_profile is not None else cfg.v3_ease_weight), 0.0), 1.0)
    ease_neighbors = (
        fit_ease_from_transitions(
            pop_rows_list,
            trans_rows_list,
            active_ids,
            lam=250.0,
            top_k=35,
        )
        if ease_w > 0.0
        else {}
    )

    transitions: dict[str, tuple[tuple[str, float], ...]] = {}
    if ease_w <= 0.0:  # comportamiento 2.1 exacto (solo lift)
        for src, edges in raw_trans.items():
            edges.sort(key=lambda kv: kv[1], reverse=True)
            top_edges = edges[:35]
            max_s = top_edges[0][1] if top_edges else 1.0
            if max_s > 0.0:
                transitions[src] = tuple((dst, sc / max_s) for dst, sc in top_edges)
    for src in (set(raw_trans) | set(ease_neighbors)) if ease_w > 0.0 else ():
        lift_edges = sorted(raw_trans.get(src, ()), key=lambda kv: kv[1], reverse=True)[:35]
        max_l = lift_edges[0][1] if lift_edges else 0.0
        lift_norm = {dst: (sc / max_l) for dst, sc in lift_edges} if max_l > 0.0 else {}
        ease_norm = dict(ease_neighbors.get(src, ()))

        combined = [
            (dst, (1.0 - ease_w) * lift_norm.get(dst, 0.0) + ease_w * ease_norm.get(dst, 0.0))
            for dst in (set(lift_norm) | set(ease_norm))
        ]
        combined.sort(key=lambda kv: kv[1], reverse=True)
        top_combined = combined[:35]
        max_c = top_combined[0][1] if top_combined else 0.0
        if max_c > 0.0:
            transitions[src] = tuple((dst, sc / max_c) for dst, sc in top_combined if sc > 0.0)

    ctx_3, ctx_2, ctx_1, _ = _build_contextual_maps(context_rows, active_ids)

    bought_map: dict[str, set[str]] = defaultdict(set)
    price_stats: dict[str, tuple[float, float]] = {}
    for row in user_profile_rows:
        uid = str(row.get("user_id") or "").strip()
        if not uid:
            continue
        for pid in row.get("bought_products") or ():
            s_pid = str(pid).strip()
            if s_pid:
                bought_map[uid].add(s_pid)
        lp_mean = _safe_float(row.get("log_price_mean"), 0.0)
        lp_std = _safe_float(row.get("log_price_std"), 0.0)
        if lp_mean > 0.0:
            price_stats[uid] = (lp_mean, lp_std)

    for row in user_bought_rows:
        uid = str(row.get("user_id") or "").strip()
        if not uid:
            continue
        for pid in row.get("bought_products") or ():
            s_pid = str(pid).strip()
            if s_pid:
                bought_map[uid].add(s_pid)

    intent_map: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in user_intent_rows:
        uid = str(row.get("user_id") or "").strip()
        pid = str(row.get("product_id") or "").strip()
        if uid and pid in active_ids:
            sc = max(_safe_float(row.get("intent_score"), 0.0), 0.0)
            if sc > 0.0:
                intent_map[uid].append((pid, sc))

    affinity_raw: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in user_affinity_rows:
        uid = str(row.get("user_id") or "").strip()
        subcat = str(row.get("subcategory") or "").strip()
        brand = str(row.get("brand") or "").strip()
        aff = max(_safe_float(row.get("affinity"), 0.0), 0.0)
        if uid and subcat and aff > 0.0:
            affinity_raw[uid][(subcat, brand)] = aff
            prev_sub = affinity_raw[uid].get((subcat, ""), 0.0)
            if aff > prev_sub:
                affinity_raw[uid][(subcat, "")] = aff

    all_uids = set(bought_map) | set(price_stats) | set(intent_map) | set(affinity_raw)
    user_profiles: dict[str, UserProfile] = {}
    for uid in all_uids:
        intents = intent_map.get(uid, [])
        intents.sort(key=lambda kv: kv[1], reverse=True)
        max_in = intents[0][1] if intents else 1.0
        norm_intents = tuple((pid, sc / max_in) for pid, sc in intents[:10]) if max_in > 0 else ()

        aff_dict = affinity_raw.get(uid, {})
        max_af = max(aff_dict.values(), default=1.0) or 1.0
        norm_aff = {k: v / max_af for k, v in aff_dict.items()}
        lp_m, lp_s = price_stats.get(uid, (0.0, 0.0))

        user_profiles[uid] = UserProfile(
            bought_products=frozenset(bought_map.get(uid, ())),
            intent_items=norm_intents,
            subcat_brand_affinity=norm_aff,
            log_price_mean=lp_m,
            log_price_std=lp_s,
        )

    live_user_bought: dict[str, frozenset[str]] = {}
    for row in live_bought_rows:
        uid = str(row.get("user_id") or "").strip()
        if not uid:
            continue
        live_user_bought[uid] = frozenset(
            str(pid).strip() for pid in (row.get("live_bought_products") or ()) if pid
        )

    # 4.1: carritos pendientes, recurrentes y compras en vivo (events + feedback)
    if cart_state is not None:
        user_carts, user_recurrent, live_bought_extra = cart_state
    else:
        user_carts, user_recurrent, live_bought_extra = build_cart_and_recurrent_maps(
            cart_hist_rows,
            recurrent_hist_rows,
            live_cart_event_rows,
            feedback_purchase_rows,
            min_recurrent=cfg.v4_recurrent_min_purchases,
        )
    if cfg.v4_live_event_purchases and live_bought_extra:
        for uid, pids in live_bought_extra.items():
            live_user_bought[uid] = live_user_bought.get(uid, frozenset()) | pids

    warm_ids = [
        pid
        for pid, _ in sorted(global_pop_norm.items(), key=lambda kv: kv[1], reverse=True)
        if pid in active_ids and pid not in new_product_ids
    ]
    if not warm_ids:
        warm_ids = [pid for pid in active_ids if pid not in new_product_ids] or list(active_ids)

    sparse_or_cold_targets = [
        info
        for pid, info in products.items()
        if info.is_new or (pid in active_ids and len(transitions.get(pid, ())) < 3)
    ]
    new_infos = [products[pid] for pid in new_product_ids if pid in products]

    cs_index = ColdStartIndexV4(
        sparse_cos_weight=wp.cold_sparse_cos_weight,
        dense_svd_weight=wp.cold_dense_svd_weight,
        subcat_match_bonus=wp.cold_subcat_match_bonus,
        cat_match_bonus=wp.cold_cat_match_bonus,
        brand_match_bonus=wp.cold_brand_match_bonus,
        price_prox_bonus=wp.cold_price_prox_bonus,
    )
    cs_index.fit(products, warm_ids)
    cold_neighbors, reverse_cold_neighbors = cs_index.compute_bidirectional(
        sparse_or_cold_targets,
        new_infos,
        top_k_forward=cfg.max_cold_neighbors,
        top_k_reverse=cfg.max_reverse_cold_per_warm,
        min_similarity=wp.cold_min_similarity,
        norm_floor=wp.cold_norm_floor,
    )

    if wp.cold_neighbor_pop_prior_weight > 0.0 and new_product_ids:
        for pid in new_product_ids:
            if raw_pop.get(pid, 0.0) > 0.0:
                continue
            nbs = cold_neighbors.get(pid)
            if not nbs:
                continue
            sim_sum = 0.0
            pop_sum = 0.0
            for warm_nb, sim in nbs:
                if sim > 0.0:
                    sim_sum += sim
                    pop_sum += sim * global_pop_norm.get(warm_nb, 0.0)
            if sim_sum > 0.0:
                neighbor_pop_avg = pop_sum / sim_sum
                prior_pop = (
                    wp.cold_neighbor_pop_prior_weight * neighbor_pop_avg
                    + 0.30 * live_trend_norm.get(pid, 0.0)
                )
                if prior_pop > global_pop_norm.get(pid, 0.0):
                    global_pop_norm[pid] = prior_pop

    fallback_scores: dict[str, float] = {}
    tie_mode = wp.tie_break_mode
    for pid in active_ids:
        p_obj = products[pid]
        pop_s = global_pop_norm.get(pid, 0.0) + 0.6 * live_trend_norm.get(pid, 0.0)
        lp_s = math.log1p(max(p_obj.price_eur, 0.0)) / max_log_price
        if tie_mode == "cheap":
            tie_term = -abs(wp.revenue_tie_weight or -0.08) * lp_s
        elif tie_mode == "conversion":
            tie_term = 0.0
        else:
            tie_term = wp.revenue_tie_weight * lp_s
        fallback_scores[pid] = pop_s + tie_term + (0.02 if p_obj.is_new else 0.0)

    fallback_ranked = tuple(
        sorted(active_ids, key=lambda pid: (fallback_scores[pid], pid), reverse=True)
    )
    revenue_fallback = tuple(
        sorted(active_ids, key=lambda pid: (products[pid].price_eur, pid), reverse=True)
    )

    fb_list = list(feedback_rows)
    hist_list = list(training_session_rows)
    bandit = PositionAwareBandit(
        exam_power=wp.bandit_exam_power,
        prior_strength=wp.bandit_prior_strength,
        cold_ucb_multiplier=wp.bandit_cold_ucb_multiplier,
        warm_ucb_multiplier=wp.bandit_warm_ucb_multiplier,
        click_reward=wp.bandit_click_reward,
        purchase_reward=wp.bandit_purchase_reward,
        use_empirical_exam_curve=wp.use_empirical_exam_curve,
    )
    bandit_source = fb_list if fb_list else hist_list
    bandit_state = bandit.fit_from_feedback(bandit_source, fallback_cvr=self_cvr)

    ranker = PositionDebiasedRanker.from_priors(
        prior_weights=wp.prior_weights_array(),
        min_prior_ratios=wp.min_prior_ratios_array(),
        max_prior_multiplier=wp.max_prior_multiplier,
    )
    snapshot = RecommenderSnapshotV4(
        version=version,
        built_at=time.time(),
        products=products,
        active_ids=active_ids,
        out_of_stock_ids=oos_set,
        new_product_ids=new_product_ids,
        self_cvr=self_cvr,
        global_pop_norm=global_pop_norm,
        max_log_price=max_log_price,
        transitions=transitions,
        cold_neighbors=cold_neighbors,
        reverse_cold_neighbors=reverse_cold_neighbors,
        ctx_3=ctx_3,
        ctx_2=ctx_2,
        ctx_1=ctx_1,
        user_profiles=user_profiles,
        live_user_bought=live_user_bought,
        live_trend_norm=live_trend_norm,
        fallback_ranked=fallback_ranked,
        revenue_fallback=revenue_fallback,
        bandit_state=bandit_state,
        ranker=ranker,
        user_carts=user_carts,
        user_recurrent=user_recurrent,
        weights_profile=wp,
    )

    # Precompute static lookup table once before IPS dataset extraction
    snapshot.product_static = build_product_static_table(
        products,
        active_ids,
        global_pop_norm,
        live_trend_norm,
        max_log_price,
        bandit,
        bandit_state,
        wp.bandit_ucb_alpha,
    )

    combined_logs = fb_list + hist_list
    if combined_logs:
        X_train, y_train, ips_w = build_ips_training_dataset(
            combined_logs,
            lambda req, cands: _extract_feature_matrix(snapshot, req, cands, weights_profile=wp),
            exam_power=wp.bandit_exam_power,
            max_sessions=3000,
            purchase_weight=wp.ips_purchase_weight,
            click_weight=wp.ips_click_weight,
            use_empirical_exam_curve=wp.use_empirical_exam_curve,
        )
        ranker.fit(
            X_train,
            y_train,
            ips_w,
            max_blend_alpha=wp.ml_max_blend_alpha,
            min_positives=wp.ml_min_positives,
            l2_c=wp.ml_l2_c,
        )

    calypso_agent = agent or CalypsoAgent()
    overrides, _, base_scratch = calypso_agent.diagnose_and_adapt(
        cfg,
        products,
        active_ids,
        new_product_ids,
        fb_list,
        bandit_state,
        ranker,
        weights_profile=wp,
    )
    snapshot.agent_overrides = overrides

    if overrides.bandit_ucb_alpha != wp.bandit_ucb_alpha:
        snapshot.product_static = build_product_static_table(
            products,
            active_ids,
            global_pop_norm,
            live_trend_norm,
            max_log_price,
            bandit,
            bandit_state,
            overrides.bandit_ucb_alpha,
        )

    scratch_payload = calypso_agent.build_full_scratch_payload(
        cfg,
        bandit,
        bandit_state,
        cold_neighbors,
        products,
        overrides,
        base_scratch,
    )
    return snapshot, scratch_payload


class CalypsoRecommenderV4:
    """Sub-millisecond Slot-Gated ML + Bandit recommender protecting WARM top-5 precision."""

    def __init__(self, config: ConfigV4 | None = None) -> None:
        self.config = config or ConfigV4()

    def _gather_candidates(
        self,
        parsed: tuple[str, bool, str, str, str, list[str]],
        snapshot: RecommenderSnapshotV4,
        excluded_ids: frozenset[str],
    ) -> list[str]:
        """Collect bounded multi-channel candidate pool with guaranteed Top-N bestsellers."""
        uid, is_new_user, region, device, page, recent_views = parsed
        wp = snapshot.weights_profile
        pool_limit = wp.candidate_pool_size if wp is not None else self.config.candidate_pool_size
        guaranteed_top = wp.guaranteed_top_sellers_in_pool if wp is not None else 35
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(pid: str) -> None:
            if len(candidates) >= pool_limit:
                return
            if not pid or pid in seen or pid not in snapshot.active_ids or pid in excluded_ids:
                return
            seen.add(pid)
            candidates.append(pid)

        # 1. Recent views first (highest conversion multiplier R1/M3)
        for seen_pid in reversed(recent_views):
            _add(seen_pid)

        # 2. Cart & recurrent items so M6 bonus scoring can always see them
        if uid and not is_new_user:
            for pid, _ in snapshot.user_carts.get(uid, ())[:6]:
                _add(pid)
            for pid, _ in snapshot.user_recurrent.get(uid, ())[:4]:
                _add(pid)

        # 3. Co-views / transitions and bidirectional cold-start neighbors (capped so Top-N always fits)
        trans_cap = max(pool_limit - guaranteed_top - 8, 44)
        for seen_pid in reversed(recent_views):
            if len(candidates) >= trans_cap:
                break
            for dst_pid, _ in snapshot.transitions.get(seen_pid, ()):
                if len(candidates) >= trans_cap:
                    break
                _add(dst_pid)
            for new_pid, _ in snapshot.reverse_cold_neighbors.get(seen_pid, ()):
                if len(candidates) >= trans_cap:
                    break
                _add(new_pid)
            for warm_nb, _ in snapshot.cold_neighbors.get(seen_pid, ()):
                if len(candidates) >= trans_cap:
                    break
                _add(warm_nb)
                for dst_pid, _ in snapshot.transitions.get(warm_nb, ())[:5]:
                    if len(candidates) >= trans_cap:
                        break
                    _add(dst_pid)

        # 4. User intent items & guaranteed Top-N bestsellers (D3/R5)
        profile = None if is_new_user else snapshot.user_profiles.get(uid)
        if profile is not None:
            for pid, _ in profile.intent_items:
                _add(pid)

        for pid in snapshot.fallback_ranked[:guaranteed_top]:
            _add(pid)

        # 5. Contextual buckets & new items
        if len(candidates) < pool_limit:
            for pid, _ in snapshot.ctx_3.get((region, device, page), ())[:18]:
                if len(candidates) >= pool_limit:
                    break
                _add(pid)
        if len(candidates) < pool_limit:
            for pid, _ in snapshot.ctx_2.get((region, device), ())[:12]:
                if len(candidates) >= pool_limit:
                    break
                _add(pid)
        if len(candidates) < pool_limit:
            for pid, _ in snapshot.ctx_1.get((page,), ())[:10]:
                if len(candidates) >= pool_limit:
                    break
                _add(pid)

        if snapshot.new_product_ids:
            anchor_subcat = ""
            if recent_views and recent_views[-1] in snapshot.products:
                anchor_subcat = snapshot.products[recent_views[-1]].subcategory
            added_new = 0
            for new_pid in snapshot.new_product_ids:
                if added_new >= 6 or len(candidates) >= pool_limit:
                    break
                info = snapshot.products.get(new_pid)
                if info is not None and (not anchor_subcat or info.subcategory == anchor_subcat):
                    _add(new_pid)
                    added_new += 1

        for pid in snapshot.fallback_ranked:
            if len(candidates) >= pool_limit:
                break
            _add(pid)

        return candidates

    def recommend(
        self,
        request: Mapping[str, Any] | Any,
        snapshot: RecommenderSnapshotV4,
        k: int = 10,
        now_ms: float | None = None,
    ) -> list[str]:
        """4.1: reglas de negocio (carrito, olvidado, recurrentes, precio) alrededor del modelo 3.1b.

        Lista final = [carrito activo] + [carrito olvidado] + [bloque del modelo con recurrentes
        garantizados]; cada bloque se ordena de más barato a más caro si v4_price_ascending.
        Con todos los interruptores CALYPSO_V4_* a 0 reproduce exactamente 3.1b.
        """
        cfg = snapshot.config_override if snapshot.config_override is not None else self.config
        parsed = _parse_request_fields(request)
        uid, is_new_user = parsed[0], parsed[1]

        profile = None if is_new_user else snapshot.user_profiles.get(uid)
        bought_hist = profile.bought_products if profile is not None else frozenset()
        bought_live = snapshot.live_user_bought.get(uid, frozenset())
        bought_all = bought_hist | bought_live

        # (M1/M2) carrito activo y carrito olvidado
        cart_active: list[str] = []
        cart_forgotten: list[str] = []
        if (cfg.v4_cart_priority or cfg.v4_forgotten_cart) and uid:
            user_cart = snapshot.user_carts.get(uid)
            if user_cart:
                cart_active, cart_forgotten = split_cart_items(
                    user_cart,
                    snapshot.active_ids,
                    time.time() * 1000.0 if now_ms is None else now_ms,
                    use_active=cfg.v4_cart_priority,
                    use_forgotten=cfg.v4_forgotten_cart,
                    active_minutes=cfg.v4_cart_active_minutes,
                    forgotten_max_days=cfg.v4_forgotten_max_days,
                    max_slots=min(cfg.v4_cart_max_slots, k),
                    forgotten_max_slots=cfg.v4_forgotten_max_slots,
                )
        pinned = cart_active + cart_forgotten

        # (M4) recurrentes: comprados >= N veces -> no se excluyen (R3/M6) y se garantizan si v4_recurrent
        recurrent: list[str] = []
        recurrent_rebuy_ids: set[str] = set()
        wp_snap = snapshot.weights_profile
        allow_rebuy = cfg.v4_recurrent or (wp_snap.allow_recurrent_rebuy if wp_snap is not None else True)
        if allow_rebuy and uid:
            pinned_set = set(pinned)
            for pid, n_buys in snapshot.user_recurrent.get(uid, ()):
                if n_buys >= cfg.v4_recurrent_min_purchases and pid in snapshot.active_ids:
                    recurrent_rebuy_ids.add(pid)
                    if (
                        cfg.v4_recurrent
                        and len(recurrent) < cfg.v4_recurrent_max_slots
                        and pid not in pinned_set
                    ):
                        recurrent.append(pid)

        # (M3) comprado una vez -> fuera; recurrente o vuelto a meter en el carrito -> se permite
        excluded_ids = frozenset(bought_all.difference(recurrent_rebuy_ids).difference(pinned))
        k_model = k - len(pinned)
        core_excluded = excluded_ids | frozenset(pinned) if pinned else excluded_ids
        block = (
            self._recommend_core(parsed, snapshot, k_model, core_excluded, _cfg=cfg, now_ms=now_ms)
            if k_model > 0
            else []
        )
        if recurrent and k_model > 0:
            block = guarantee_items(block, recurrent[:k_model])

        if cfg.v4_price_ascending:
            cart_active = sort_by_price(cart_active, snapshot.products)
            cart_forgotten = sort_by_price(cart_forgotten, snapshot.products)
            block = sort_by_price(block, snapshot.products)

        if not pinned and not recurrent and not cfg.v4_price_ascending:
            return block  # camino 3.1b exacto
        return ensure_valid_submission(
            cart_active + cart_forgotten + block,
            snapshot.active_ids,
            snapshot.fallback_ranked,
            exclude_ids=excluded_ids,
            k=k,
        )

    def _recommend_core(
        self,
        parsed: tuple[str, bool, str, str, str, list[str]],
        snapshot: RecommenderSnapshotV4,
        k: int,
        excluded_ids: frozenset[str],
        *,
        _cfg: ConfigV4 | None = None,
        now_ms: float | None = None,
    ) -> list[str]:
        """Modular Slot-Gated scoring driven by WeightsProfile & 5 user segments."""
        cfg = _cfg if _cfg is not None else self.config
        wp = snapshot.weights_profile if snapshot.weights_profile is not None else cfg.load_weights_profile()
        uid, is_new_user, _, _, _, recent_views = parsed
        has_session_intent = bool(recent_views)
        is_cold_session = is_new_user or any(
            pid in snapshot.new_product_ids for pid in recent_views
        )

        candidates = self._gather_candidates(parsed, snapshot, excluded_ids)
        if not candidates:
            return ensure_valid_submission(
                (),
                snapshot.active_ids,
                snapshot.fallback_ranked,
                exclude_ids=excluded_ids,
                k=k,
            )

        req_profile = None if is_new_user else snapshot.user_profiles.get(uid)
        has_profile = req_profile is not None and bool(
            req_profile.bought_products
            or req_profile.intent_items
            or req_profile.subcat_brand_affinity
            or req_profile.log_price_mean > 0.0
        )
        segment = classify_user_segment(is_new_user, has_session_intent, has_profile)
        m_seq, m_int, m_aff, m_ctx, m_pop, m_trd, m_cld = wp.segment_multipliers(segment)

        (
            seq_scores,
            self_view_scores,
            intent_scores,
            ctx_scores,
            rev_graft_scores,
            fwd_cold_scores,
            affinity_map,
            has_price_hist,
            mean_lp,
            std_lp,
            anchor_subcat,
            anchor_brand,
            extra_signals,
        ) = _prepare_request_signals(
            snapshot,
            parsed,
            price_std_floor=wp.price_std_floor,
            weights_profile=wp,
        )
        anchor_cat = str(extra_signals.get("anchor_cat") or "")

        cart_bonus_map: dict[str, float] = {}
        rec_bonus_map: dict[str, float] = {}
        if uid and not is_new_user:
            u_carts = snapshot.user_carts.get(uid)
            if u_carts:
                cur_ms = time.time() * 1000.0 if now_ms is None else now_ms
                active_ms = cfg.v4_cart_active_minutes * 60_000.0
                max_forgot_ms = cfg.v4_forgotten_max_days * 86_400_000.0
                for c_pid, cart_ts in u_carts:
                    age_ms = max(cur_ms - cart_ts, 0.0)
                    if age_ms <= active_ms:
                        if wp.cart_active_score_bonus > 0.0:
                            cart_bonus_map[c_pid] = wp.cart_active_score_bonus
                    elif age_ms <= max_forgot_ms and wp.cart_forgotten_score_bonus > 0.0:
                        age_days = age_ms / 86_400_000.0
                        cart_bonus_map[c_pid] = wp.cart_forgotten_score_bonus * math.exp(-age_days / 7.0)
            u_rec = snapshot.user_recurrent.get(uid)
            if u_rec and wp.recurrent_score_bonus > 0.0:
                for r_pid, n_buys in u_rec:
                    if n_buys >= cfg.v4_recurrent_min_purchases:
                        rec_bonus_map[r_pid] = wp.recurrent_score_bonus

        overrides = snapshot.agent_overrides
        ranker: PositionDebiasedRanker = (
            snapshot.ranker if snapshot.ranker is not None else PositionDebiasedRanker.from_priors(
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

        w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13 = (
            ranker.resolve_weights_tuple(
                blend_alpha_scale=overrides.ml_blend_alpha_scale,
                reverse_graft_override=overrides.weight_reverse_cold_graft,
                forward_cold_override=overrides.weight_forward_cold,
                revenue_tie_override=rev_tie_override,
            )
        )

        max_lp = max(snapshot.max_log_price, 1e-6)
        product_static = snapshot.product_static
        has_aff = bool(affinity_map)

        # Compute split (precision_score, explore_score, cat, subcat, brand, viewed, is_new, confirmed_cold)
        n_cands = len(candidates)
        prec_scores = [0.0] * n_cands
        full_scores = [0.0] * n_cands
        cand_cats = [""] * n_cands
        cand_subcats = [""] * n_cands
        cand_brands = [""] * n_cands
        cand_viewed = [False] * n_cands
        cand_lp = [0.0] * n_cands
        cand_is_new = [False] * n_cands
        cand_confirmed_cold = [False] * n_cands

        if wp.v3_no_subcat_floor or cfg.v3_no_subcat_floor:
            subcat_bonus_w = w12
            brand_bonus_w = w13
        else:
            subcat_bonus_w = max(w12, wp.subcat_floor_min)
            brand_bonus_w = max(w13, wp.brand_floor_min)
        v3_bg_penalty = wp.v3_bg_penalty and cfg.v3_bg_penalty
        same_cat_w = wp.same_cat_bonus_weight
        discount_w = wp.discount_boost_weight
        products_map = snapshot.products

        seq_get = seq_scores.get
        sv_get = self_view_scores.get
        int_get = intent_scores.get
        ctx_get = ctx_scores.get
        rev_get = rev_graft_scores.get
        fwd_get = fwd_cold_scores.get
        has_seq = bool(seq_scores)
        has_sv = bool(self_view_scores)
        has_int = bool(intent_scores)
        has_ctx = bool(ctx_scores)
        has_rev = bool(rev_graft_scores)
        has_fwd = bool(fwd_cold_scores)
        has_cart_b = bool(cart_bonus_map)
        has_rec_b = bool(rec_bonus_map)
        prod_get = products_map.get
        st_get = product_static.get
        aff_get = affinity_map.get
        cart_b_get = cart_bonus_map.get
        rec_b_get = rec_bonus_map.get
        z2_clip = wp.price_z2_clip
        cheap_asym = wp.price_cheap_asymmetry
        bg_same = wp.bg_penalty_same_subcat
        bg_diff = wp.bg_penalty_diff_subcat

        for idx, pid in enumerate(candidates):
            if has_seq:
                raw_seq = seq_get(pid, 0.0)
                s_seq = (1.8 if raw_seq > 1.8 else raw_seq) * m_seq
            else:
                s_seq = 0.0
            if has_sv:
                sv_raw = sv_get(pid, 0.0)
                if sv_raw > 0.0:
                    sv5 = sv_raw * 5.0
                    s_self = 1.2 if sv5 > 1.2 else sv5
                    cand_viewed[idx] = True
                else:
                    s_self = 0.0
            else:
                s_self = 0.0
            s_intent = int_get(pid, 0.0) * m_int if has_int else 0.0
            s_ctx = ctx_get(pid, 0.0) * m_ctx if has_ctx else 0.0
            s_rev = rev_get(pid, 0.0) * m_cld if has_rev else 0.0
            s_fwd = fwd_get(pid, 0.0) * m_cld if has_fwd else 0.0

            p_info = prod_get(pid)
            p_cat = p_info.category if p_info is not None else ""
            discount_r = p_info.discount_ratio if p_info is not None else 0.0
            cand_cats[idx] = p_cat
            rule_bonus = discount_w * discount_r
            if has_cart_b:
                rule_bonus += cart_b_get(pid, 0.0)
            if has_rec_b:
                rule_bonus += rec_b_get(pid, 0.0)

            st = st_get(pid)
            if st is not None:
                sc, br, gpop, ltrend, is_new_f, lp_norm, exploit_d, ucb_b = st
                cand_lp[idx] = lp_norm
                s_aff = 0.0
                if has_aff:
                    s_aff = aff_get((sc, br), 0.0)
                    if s_aff == 0.0:
                        s_aff = aff_get((sc, ""), 0.0) * 0.6
                    s_aff *= m_aff

                s_pcompat = 0.0
                if has_price_hist and lp_norm > 0.0:
                    s_pcompat = _compute_price_compat(
                        lp_norm * max_lp,
                        mean_lp,
                        std_lp,
                        z2_clip=z2_clip,
                        cheap_asymmetry=cheap_asym,
                    )

                same_sc = bool(anchor_subcat and sc == anchor_subcat)
                same_cat = bool(anchor_cat and p_cat == anchor_cat)
                same_br = bool(anchor_brand and br == anchor_brand)
                bg_score = w5 * s_ctx + w6 * (gpop * m_pop) + w7 * (ltrend * m_trd) + exploit_d
                if (
                    v3_bg_penalty
                    and has_session_intent
                    and s_seq == 0.0
                    and s_intent == 0.0
                    and s_fwd == 0.0
                    and s_self == 0.0
                ):
                    bg_score *= bg_same if same_sc else bg_diff

                score_p = (
                    w0 * s_seq
                    + w1 * s_self
                    + w2 * s_intent
                    + w3 * s_aff
                    + w4 * s_pcompat
                    + bg_score
                    + w8 * is_new_f
                    + w9 * s_rev
                    + w10 * s_fwd
                    + w11 * lp_norm
                    + (subcat_bonus_w if same_sc else (same_cat_w if same_cat else 0.0))
                    + (brand_bonus_w if same_br else 0.0)
                    + rule_bonus
                )
                prec_scores[idx] = score_p
                full_scores[idx] = score_p + ucb_b
                cand_subcats[idx] = sc
                cand_brands[idx] = br
                is_n = is_new_f > 0.5
                cand_is_new[idx] = is_n
                if is_n and s_rev > 0.35 and exploit_d > 0.02:
                    cand_confirmed_cold[idx] = True
            else:
                gpop = snapshot.global_pop_norm.get(pid, 0.0) * m_pop
                ltrend = snapshot.live_trend_norm.get(pid, 0.0) * m_trd
                score_p = (
                    w0 * s_seq
                    + w1 * s_self
                    + w2 * s_intent
                    + w5 * s_ctx
                    + w6 * gpop
                    + w7 * ltrend
                    + w9 * s_rev
                    + w10 * s_fwd
                    + rule_bonus
                )
                prec_scores[idx] = score_p
                full_scores[idx] = score_p

        if has_session_intent:
            mmr_start = wp.mmr_start_slot_session
            subcat_lam = wp.mmr_subcat_lambda_session
        else:
            mmr_start = wp.mmr_start_slot_home
            subcat_lam = wp.mmr_subcat_lambda_home
        brand_lam = wp.mmr_brand_lambda if wp.mmr_brand_lambda != cfg.mmr_brand_lambda else overrides.mmr_brand_lambda

        picked = [False] * n_cands
        subcat_counts: dict[str, int] = defaultdict(int)
        brand_counts: dict[str, int] = defaultdict(int)
        selected: list[str] = []

        cold_gate = wp.cold_explore_start_slot_cold if is_cold_session else wp.cold_explore_start_slot_warm
        max_cold_warm = wp.max_cold_explore_in_warm
        slot1_cat_bonus = wp.slot1_category_match_bonus
        tie_eps = wp.tie_break_eps
        new_picked_count = 0
        tie_mode = wp.tie_break_mode
        is_conv_tie = tie_mode == "conversion"
        is_cheap_tie = tie_mode == "cheap"
        use_eps_tie = tie_eps > 0.0 and not is_conv_tie
        use_rel_mmr = wp.mmr_relative_penalty
        confirmed_cold_gate = max(4, cold_gate) if is_cold_session else 4

        target_k = min(k, n_cands)
        while len(selected) < target_k:
            slot_pos = len(selected)
            gate_pass = slot_pos >= cold_gate
            allow_ucb = gate_pass
            allow_new = gate_pass and (is_cold_session or new_picked_count < max_cold_warm)
            allow_confirmed_cold = slot_pos >= confirmed_cold_gate and new_picked_count < max_cold_warm
            apply_mmr = slot_pos >= mmr_start
            apply_slot1_cat = slot_pos == 0 and has_session_intent and slot1_cat_bonus > 0.0

            best_idx = -1
            best_val = -1e18
            best_tie = -1e18
            anchor_val = -1e18
            anchor_sc = ""
            for idx in range(n_cands):
                if picked[idx]:
                    continue
                is_n = cand_is_new[idx]
                confirmed_cold = cand_confirmed_cold[idx]
                if is_n and not allow_new and not (confirmed_cold and allow_confirmed_cold):
                    continue
                val = full_scores[idx] if (allow_ucb or confirmed_cold) else prec_scores[idx]
                if apply_slot1_cat:
                    if (
                        cand_viewed[idx]
                        or (anchor_subcat and cand_subcats[idx] == anchor_subcat)
                        or (anchor_cat and cand_cats[idx] == anchor_cat)
                    ):
                        val += slot1_cat_bonus
                if apply_mmr:
                    sc = cand_subcats[idx]
                    br = cand_brands[idx]
                    if use_rel_mmr:
                        pen = 0.0
                        if sc and subcat_lam > 0.0:
                            pen += subcat_lam * subcat_counts[sc]
                        if br and brand_lam > 0.0:
                            pen += brand_lam * brand_counts[br]
                        if pen > 0.0:
                            mult_p = 1.0 - pen
                            if mult_p < 0.15:
                                mult_p = 0.15
                            val = val * mult_p if val >= 0.0 else val / mult_p
                    else:
                        if sc and subcat_lam > 0.0:
                            val -= subcat_lam * subcat_counts[sc]
                        if br and brand_lam > 0.0:
                            val -= brand_lam * brand_counts[br]
                if is_conv_tie:
                    if val > best_val:
                        best_val = val
                        best_idx = idx
                    continue

                tie_val = -cand_lp[idx] if is_cheap_tie else cand_lp[idx]
                idx_sc = cand_subcats[idx]
                if best_idx < 0:
                    best_idx = idx
                    best_val = val
                    best_tie = tie_val
                    anchor_val = val
                    anchor_sc = idx_sc
                elif val > anchor_val + 1e-9:
                    eps_tol = max(1e-9, tie_eps * abs(val)) if use_eps_tie else 0.0
                    if (
                        use_eps_tie
                        and anchor_sc
                        and idx_sc == anchor_sc
                        and best_val >= val - eps_tol
                        and best_tie > tie_val + 1e-9
                    ):
                        anchor_val = val
                    else:
                        best_idx = idx
                        best_val = val
                        best_tie = tie_val
                        anchor_val = val
                        anchor_sc = idx_sc
                elif use_eps_tie and anchor_sc and idx_sc == anchor_sc:
                    eps_tol = max(1e-9, tie_eps * abs(anchor_val))
                    if val >= anchor_val - eps_tol and (
                        tie_val > best_tie + 1e-9 or (val > best_val and abs(tie_val - best_tie) <= 1e-9)
                    ):
                        best_idx = idx
                        best_val = val
                        best_tie = tie_val
                elif tie_val > best_tie and abs(val - anchor_val) <= 1e-9:
                    best_idx = idx
                    best_val = val
                    best_tie = tie_val
                    anchor_sc = idx_sc

            if best_idx < 0:
                # Fallback if only cold items remain and cap was reached
                for idx in range(n_cands):
                    if not picked[idx] and full_scores[idx] > best_val:
                        best_val = full_scores[idx]
                        best_idx = idx
                if best_idx < 0:
                    break

            picked[best_idx] = True
            selected.append(candidates[best_idx])
            if cand_is_new[best_idx]:
                new_picked_count += 1
            if cand_subcats[best_idx]:
                subcat_counts[cand_subcats[best_idx]] += 1
            if cand_brands[best_idx]:
                brand_counts[cand_brands[best_idx]] += 1

        return ensure_valid_submission(
            selected,
            snapshot.active_ids,
            snapshot.fallback_ranked,
            exclude_ids=excluded_ids,
            k=k,
        )
