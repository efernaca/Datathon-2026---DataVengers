"""Slot-Gated Calypso-2.1 recommender with precomputed static item vectors (<1 ms/req)."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
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
from .collab_v6 import build_collaborative_svd_index, build_query_vector
from .config import ConfigV4
from .ease import fit_ease_from_transitions
from .guardrails import ensure_valid_submission
from .reranker import NUM_FEATURES, PositionDebiasedRanker, build_ips_training_dataset
from .state import ProductInfo, RecommenderSnapshotV4, UserProfile
from .training_data_v6 import (
    build_multisource_v6_training_dataset,
    compute_ranker_holdout_auc,
    load_covisit_edges,
)
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


def _parse_ts_ms(value: Any, default: float = 0.0) -> float:
    """Parse epoch seconds/ms, datetime/Timestamp, or ISO datetime strings into epoch ms."""
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return default
        v = float(value)
        if v <= 0.0:
            return default
        return v * 1000.0 if v < 1e11 else v
    ts_fn = getattr(value, "timestamp", None)
    if callable(ts_fn):
        try:
            return float(ts_fn()) * 1000.0
        except Exception:
            return default
    s = str(value).strip()
    if not s:
        return default
    try:
        # Handle pandas / ClickHouse nanosecond timestamps like '2026-11-11 07:31:49.549000000'
        if "." in s:
            head, frac = s.split(".", 1)
            tz_tail = ""
            for sep in ("+", "Z"):
                if sep in frac:
                    idx_sep = frac.index(sep)
                    tz_tail = frac[idx_sep:]
                    frac = frac[:idx_sep]
                    break
            s_norm = f"{head}.{frac[:6]}{tz_tail}".replace("Z", "+00:00")
        else:
            s_norm = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s_norm).timestamp() * 1000.0
    except Exception:
        return default


def extract_recent_views_with_ts(raw_recent: Any) -> tuple[list[str], list[float]]:
    """Extract ordered (product_id, view_ts_ms) lists from RawTree dicts or string IDs."""
    out_ids: list[str] = []
    out_ts: list[float] = []
    if raw_recent is None:
        return out_ids, out_ts
    try:
        if len(raw_recent) == 0:
            return out_ids, out_ts
    except TypeError:
        pass
    for item in raw_recent:
        if isinstance(item, Mapping):
            pid = item.get("product_id")
            if pid:
                out_ids.append(str(pid).strip())
                out_ts.append(_parse_ts_ms(item.get("ts"), 0.0))
        elif item:
            s_pid = str(item).strip()
            if s_pid:
                out_ids.append(s_pid)
                out_ts.append(0.0)
    return out_ids, out_ts


def extract_recent_product_ids(raw_recent: Any) -> list[str]:
    """Parse recent_views accepting either RawTree dict items or plain string IDs."""
    ids, _ = extract_recent_views_with_ts(raw_recent)
    return ids


def _extract_req_ts_ms(request: Any, raw_recent: Any = None) -> float:
    """Resolve request timestamp in epoch ms if available (0.0 when absent)."""
    ts_raw = request.get("ts") if isinstance(request, Mapping) else getattr(request, "ts", None)
    req_ms = _parse_ts_ms(ts_raw, 0.0)
    if req_ms > 0.0:
        return req_ms
    if raw_recent is not None:
        _, ts_list = extract_recent_views_with_ts(raw_recent)
        valid = [t for t in ts_list if t > 0.0]
        if valid:
            return max(valid)
    return 0.0


def _resolve_now_ms(
    snapshot: RecommenderSnapshotV4,
    now_ms: float | None = None,
    req_ts_ms: float | None = 0.0,
) -> float:
    """Choose coherent clock reference: explicit > request ts > snapshot built_at > wall clock."""
    if now_ms is not None and now_ms > 0.0:
        return float(now_ms)
    if req_ts_ms is not None and req_ts_ms > 0.0:
        return float(req_ts_ms)
    if getattr(snapshot, "built_at", 0.0) > 0.0:
        return float(snapshot.built_at) * 1000.0
    return time.time() * 1000.0


def _parse_request_fields_with_ts(
    request: Any,
) -> tuple[str, bool, str, str, str, list[str], list[float], float]:
    """Extract (user_id, is_new_user, region, device, page, recent_views, view_ts_ms, req_ts_ms)."""
    if isinstance(request, Mapping):
        user_id = str(request.get("user_id") or "").strip()
        is_new_user = bool(request.get("is_new_user", False))
        ctx = request.get("context")
        raw_recent = request.get("recent_views")
        raw_ts = request.get("ts")
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
        raw_ts = getattr(request, "ts", None)
        if isinstance(ctx, Mapping):
            region = str(ctx.get("region") or "").strip()
            device = str(ctx.get("device") or "").strip()
            page = str(ctx.get("page") or "").strip()
        else:
            region = str(getattr(request, "region", "") or "").strip()
            device = str(getattr(request, "device", "") or "").strip()
            page = str(getattr(request, "page", "") or "").strip()

    rv_ids, rv_ts = extract_recent_views_with_ts(raw_recent)
    req_ts_ms = _parse_ts_ms(raw_ts, 0.0)
    if req_ts_ms <= 0.0 and rv_ts:
        valid_ts = [t for t in rv_ts if t > 0.0]
        if valid_ts:
            req_ts_ms = max(valid_ts)
    return user_id, is_new_user, region, device, page, rv_ids, rv_ts, req_ts_ms


def _parse_request_fields(request: Any) -> tuple[str, bool, str, str, str, list[str]]:
    """Extract (user_id, is_new_user, region, device, page, recent_views) from dict or row."""
    return _parse_request_fields_with_ts(request)[:6]


def _build_category_indices(
    products: Mapping[str, ProductInfo],
    active_ids: frozenset[str],
    global_pop_norm: Mapping[str, float],
    live_trend_norm: Mapping[str, float],
    *,
    top_k: int = 24,
) -> tuple[
    dict[str, tuple[tuple[str, float], ...]],
    dict[str, tuple[tuple[str, float], ...]],
    dict[str, tuple[float, float]],
    dict[str, float],
    dict[str, float],
]:
    """Precompute intra-subcategory/category bestsellers, O(1) pop ranks, and subcat price stats."""
    by_sc: dict[str, list[tuple[float, str]]] = defaultdict(list)
    by_cat: dict[str, list[tuple[float, str]]] = defaultdict(list)
    sc_prices: dict[str, list[float]] = defaultdict(list)

    for pid in active_ids:
        info = products.get(pid)
        if info is None:
            continue
        score = global_pop_norm.get(pid, 0.0) + 0.45 * live_trend_norm.get(pid, 0.0)
        if info.subcategory:
            by_sc[info.subcategory].append((score, pid))
            if info.price_eur > 0.0:
                sc_prices[info.subcategory].append(math.log1p(info.price_eur))
        if info.category:
            by_cat[info.category].append((score, pid))

    subcat_top: dict[str, tuple[tuple[str, float], ...]] = {}
    item_subcat_pop: dict[str, float] = {}
    for sc, items in by_sc.items():
        items.sort(reverse=True)
        max_s = items[0][0] if items else 0.0
        denom = max(max_s, 1e-6)
        subcat_top[sc] = tuple((pid, sc_v / denom) for sc_v, pid in items[:top_k] if sc_v > 0.0)
        for sc_v, pid in items:
            if sc_v > 0.0:
                item_subcat_pop[pid] = sc_v / denom

    cat_top: dict[str, tuple[tuple[str, float], ...]] = {}
    item_cat_pop: dict[str, float] = {}
    for cat, items in by_cat.items():
        items.sort(reverse=True)
        max_s = items[0][0] if items else 0.0
        denom = max(max_s, 1e-6)
        cat_top[cat] = tuple((pid, c_v / denom) for c_v, pid in items[:top_k] if c_v > 0.0)
        for c_v, pid in items:
            if c_v > 0.0:
                item_cat_pop[pid] = c_v / denom

    subcat_price_stats: dict[str, tuple[float, float]] = {}
    for sc, lps in sc_prices.items():
        n_p = len(lps)
        mean_lp = sum(lps) / n_p
        var = sum((x - mean_lp) ** 2 for x in lps) / max(n_p, 1)
        subcat_price_stats[sc] = (mean_lp, max(math.sqrt(var), 0.35))

    return subcat_top, cat_top, subcat_price_stats, item_subcat_pop, item_cat_pop


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
    recency_decay: float = 0.70,
    price_std_floor: float = 0.45,
    weights_profile: WeightsProfile | None = None,
    now_ms: float | None = None,
    view_ts_list: Sequence[float] | None = None,
    req_ts_ms: float = 0.0,
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
    page_l = str(page or "").strip().lower()
    cur_ms = _resolve_now_ms(snapshot, now_ms, req_ts_ms)

    seq_scores: dict[str, float] = {}
    ease_scores: dict[str, float] = {}
    hop2_scores: dict[str, float] = {}
    covisit_scores: dict[str, float] = {}
    fwd_cold_scores: dict[str, float] = {}
    rev_graft_scores: dict[str, float] = {}
    self_view_scores: dict[str, float] = {}
    view_recency_map: dict[str, float] = {}
    rv_temporal_map: dict[str, float] = {}
    rv_counts: dict[str, int] = {}

    decay = wp.recency_decay if wp is not None else recency_decay
    svd_decay = wp.svd_recency_decay if wp is not None else 0.68
    last_prod_factor = wp.view_last_product_page_factor if wp is not None else 1.15
    view_base = wp.view_base_bonus if wp is not None else 0.0
    rv_half_life_min = max(wp.rv_temporal_half_life_min if wp is not None else 30.0, 2.0)

    rv_subcat_acc: dict[str, float] = defaultdict(float)
    rv_cat_acc: dict[str, float] = defaultdict(float)
    rv_brand_acc: dict[str, float] = defaultdict(float)
    rv_wt_sum = 0.0
    rv_lp_sum = 0.0
    rv_lp_wt = 0.0
    sess_svd_items: list[tuple[str, float]] = []

    ease_map = snapshot.ease_neighbors
    hop2_map = snapshot.hop2_transitions
    cov_map = snapshot.covisit_neighbors

    n_rv = len(recent_views)
    has_rv_ts = view_ts_list is not None and len(view_ts_list) == n_rv
    for rev_idx, seen_pid in enumerate(reversed(recent_views)):
        w_rec = decay**rev_idx
        rv_counts[seen_pid] = rv_counts.get(seen_pid, 0) + 1
        sess_svd_items.append((seen_pid, svd_decay**rev_idx))
        if seen_pid not in view_recency_map:
            view_recency_map[seen_pid] = w_rec

        if has_rv_ts and cur_ms > 0.0:
            v_ts = float(view_ts_list[n_rv - 1 - rev_idx])
            if v_ts > 0.0:
                dt_min = max(cur_ms - v_ts, 0.0) / 60_000.0
                t_rec = math.exp(-0.693 * dt_min / rv_half_life_min)
                if t_rec > rv_temporal_map.get(seen_pid, 0.0):
                    rv_temporal_map[seen_pid] = t_rec
        elif seen_pid not in rv_temporal_map:
            rv_temporal_map[seen_pid] = w_rec

        s_info = snapshot.products.get(seen_pid)
        if s_info is not None:
            rv_wt_sum += w_rec
            if s_info.subcategory:
                rv_subcat_acc[s_info.subcategory] += w_rec
            if s_info.category:
                rv_cat_acc[s_info.category] += w_rec
            if s_info.brand:
                rv_brand_acc[s_info.brand] += w_rec
            if s_info.price_eur > 0.0:
                rv_lp_sum += w_rec * math.log1p(s_info.price_eur)
                rv_lp_wt += w_rec

        if seen_pid in snapshot.active_ids:
            page_factor = last_prod_factor if (page_l == "product" and rev_idx == 0) else 1.0
            base_cvr = snapshot.self_cvr.get(seen_pid, 0.02)
            if view_base > 0.0:
                pop_f = 0.45 + 0.55 * snapshot.global_pop_norm.get(seen_pid, 0.3)
                cvr_f = view_base * 0.20 + (1.0 - view_base) * base_cvr
                step_sv = cvr_f * pop_f * w_rec * page_factor
            else:
                step_sv = base_cvr * w_rec * page_factor
            if seen_pid not in self_view_scores:
                self_view_scores[seen_pid] = step_sv
            else:
                # Re-viewing the exact same item in one session signals high purchase intent
                self_view_scores[seen_pid] = min(0.24, self_view_scores[seen_pid] + 0.35 * step_sv)

        trans_nbs = snapshot.transitions.get(seen_pid)
        if trans_nbs:
            for dst_pid, weight in trans_nbs:
                if dst_pid in snapshot.active_ids:
                    seq_scores[dst_pid] = seq_scores.get(dst_pid, 0.0) + w_rec * weight

        if ease_map:
            for dst_pid, weight in ease_map.get(seen_pid, ()):
                if dst_pid in snapshot.active_ids:
                    ease_scores[dst_pid] = ease_scores.get(dst_pid, 0.0) + w_rec * weight

        if hop2_map:
            for dst_pid, weight in hop2_map.get(seen_pid, ()):
                if dst_pid in snapshot.active_ids:
                    hop2_scores[dst_pid] = hop2_scores.get(dst_pid, 0.0) + w_rec * weight

        if cov_map:
            for dst_pid, weight in cov_map.get(seen_pid, ()):
                if dst_pid in snapshot.active_ids:
                    covisit_scores[dst_pid] = covisit_scores.get(dst_pid, 0.0) + w_rec * weight

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

    denom_rep = math.log(4.0)
    rv_repeat_norm_map: dict[str, float] = {
        pid: min(1.0, math.log1p(cnt - 1) / denom_rep)
        for pid, cnt in rv_counts.items()
        if cnt > 1
    }

    if rv_wt_sum > 0.0:
        rv_subcat_frac = {k: v / rv_wt_sum for k, v in rv_subcat_acc.items()}
        rv_brand_frac = {k: v / rv_wt_sum for k, v in rv_brand_acc.items()}
    else:
        rv_subcat_frac = {}
        rv_brand_frac = {}
    mean_rv_lp = (rv_lp_sum / rv_lp_wt) if rv_lp_wt > 0.0 else 0.0

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
    user_svd_items: list[tuple[str, float]] = []
    if profile is not None:
        if profile.intent_items:
            intent_scores = dict(profile.intent_items)
            user_svd_items.extend(profile.intent_items[:8])
        affinity_map = profile.subcat_brand_affinity
        if profile.log_price_mean > 0.0:
            has_price_hist = True
            mean_lp = profile.log_price_mean
            std_lp = max(profile.log_price_std, price_std_floor)

    # Continuous cart decay & recurrent log features
    cart_active_decay_map: dict[str, float] = {}
    cart_forgotten_decay_map: dict[str, float] = {}
    rec_log_norm_map: dict[str, float] = {}
    if user_id and not is_new_user:
        u_carts = snapshot.user_carts.get(user_id)
        if u_carts:
            active_ms = (wp.v4_cart_active_minutes if wp is not None else 30.0) * 60_000.0
            max_forgot_ms = (wp.v4_forgotten_max_days if wp is not None else 30.0) * 86_400_000.0
            half_life_min = max(wp.cart_recency_half_life_min if wp is not None else 45.0, 5.0)
            for c_pid, cart_ts in u_carts:
                raw_age_ms = cur_ms - cart_ts
                if raw_age_ms < -60_000.0:
                    continue
                age_ms = max(raw_age_ms, 0.0)
                if age_ms <= active_ms:
                    user_svd_items.append((c_pid, 1.15))
                    age_min = age_ms / 60_000.0
                    cart_active_decay_map[c_pid] = math.exp(-0.693 * age_min / half_life_min)
                elif age_ms <= max_forgot_ms:
                    user_svd_items.append((c_pid, 0.90))
                    age_days = age_ms / 86_400_000.0
                    cart_forgotten_decay_map[c_pid] = math.exp(-age_days / 7.0)
        u_rec = snapshot.user_recurrent.get(user_id)
        if u_rec:
            min_r = wp.v4_recurrent_min_purchases if wp is not None else 2
            denom_log = math.log(8.0)
            for r_pid, n_buys in u_rec:
                if n_buys >= min_r:
                    rec_log_norm_map[r_pid] = min(1.0, math.log1p(n_buys) / denom_log)
                    user_svd_items.append((r_pid, 1.35))

    svd_mat = snapshot.svd_item_vectors
    svd_map = snapshot.svd_pid_to_idx
    svd_sess_vec = build_query_vector(sess_svd_items, svd_mat, svd_map) if (sess_svd_items and svd_map) else None
    svd_user_vec = build_query_vector(user_svd_items, svd_mat, svd_map) if (user_svd_items and svd_map) else None

    anchor_cat = ""
    anchor_subcat = ""
    anchor_brand = ""
    last_viewed_pid = recent_views[-1] if recent_views else ""
    if last_viewed_pid:
        last_info = snapshot.products.get(last_viewed_pid)
        if last_info is not None:
            anchor_cat = last_info.category
            anchor_subcat = last_info.subcategory
            anchor_brand = last_info.brand

    extra_signals = {
        "anchor_cat": anchor_cat,
        "page": page_l,
        "no_views": not bool(recent_views),
        "last_viewed_pid": last_viewed_pid,
        "ease_scores": ease_scores,
        "hop2_scores": hop2_scores,
        "covisit_scores": covisit_scores,
        "view_recency_map": view_recency_map,
        "rv_temporal_map": rv_temporal_map,
        "rv_repeat_norm_map": rv_repeat_norm_map,
        "rv_subcat_frac": rv_subcat_frac,
        "rv_cat_acc": rv_cat_acc,
        "rv_brand_frac": rv_brand_frac,
        "mean_rv_lp": mean_rv_lp,
        "svd_sess_vec": svd_sess_vec,
        "svd_user_vec": svd_user_vec,
        "cart_active_decay_map": cart_active_decay_map,
        "cart_forgotten_decay_map": cart_forgotten_decay_map,
        "rec_log_norm_map": rec_log_norm_map,
    }

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
        extra_signals,
    )


def _extract_feature_matrix(
    snapshot: RecommenderSnapshotV4,
    request: Mapping[str, Any] | Any,
    candidate_ids: Sequence[str],
    parsed: tuple[str, bool, str, str, str, list[str]] | None = None,
    *,
    weights_profile: WeightsProfile | None = None,
    now_ms: float | None = None,
    return_aux: bool = False,
    view_ts_list: Sequence[float] | None = None,
    req_ts_ms: float = 0.0,
) -> Any:
    """Vectorized 50-feature builder shared identically by training and Calypso 6.2 FastLoop."""
    n = len(candidate_ids)
    if n == 0:
        empty_X = np.empty((0, NUM_FEATURES), dtype=np.float64)
        return (empty_X, {}) if return_aux else empty_X

    wp = weights_profile if weights_profile is not None else snapshot.weights_profile
    std_floor = wp.price_std_floor if wp is not None else 0.45
    z2_clip = wp.price_z2_clip if wp is not None else 6.0
    cheap_asym = wp.price_cheap_asymmetry if wp is not None else 0.0

    if parsed is None:
        full_p = _parse_request_fields_with_ts(request)
        p = full_p[:6]
        if view_ts_list is None:
            view_ts_list = full_p[6]
        if req_ts_ms <= 0.0:
            req_ts_ms = full_p[7]
    else:
        p = parsed

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
        p,
        price_std_floor=std_floor,
        weights_profile=wp,
        now_ms=now_ms,
        view_ts_list=view_ts_list,
        req_ts_ms=req_ts_ms,
    )

    page_l: str = extra_signals["page"]
    no_views_flag: bool = bool(extra_signals["no_views"])
    anchor_cat: str = extra_signals["anchor_cat"]
    last_viewed_pid: str = extra_signals["last_viewed_pid"]
    ease_scores: dict[str, float] = extra_signals["ease_scores"]
    hop2_scores: dict[str, float] = extra_signals["hop2_scores"]
    covisit_scores: dict[str, float] = extra_signals["covisit_scores"]
    view_recency_map: dict[str, float] = extra_signals["view_recency_map"]
    rv_temporal_map: dict[str, float] = extra_signals["rv_temporal_map"]
    rv_repeat_norm_map: dict[str, float] = extra_signals["rv_repeat_norm_map"]
    rv_subcat_frac: dict[str, float] = extra_signals["rv_subcat_frac"]
    rv_brand_frac: dict[str, float] = extra_signals["rv_brand_frac"]
    mean_rv_lp: float = extra_signals["mean_rv_lp"]
    svd_sess_vec: np.ndarray | None = extra_signals["svd_sess_vec"]
    svd_user_vec: np.ndarray | None = extra_signals["svd_user_vec"]
    cart_active_decay_map: dict[str, float] = extra_signals["cart_active_decay_map"]
    cart_forgotten_decay_map: dict[str, float] = extra_signals["cart_forgotten_decay_map"]
    rec_log_norm_map: dict[str, float] = extra_signals["rec_log_norm_map"]

    is_pg_prod = 1.0 if page_l == "product" else 0.0
    is_pg_cat = 1.0 if page_l == "category" else 0.0
    is_pg_home = 1.0 if page_l == "home" else 0.0

    max_lp = max(snapshot.max_log_price, 1e-6)
    rows: list[list[float]] = [[0.0] * NUM_FEATURES for _ in range(n)]

    # Batch SVD cosine similarities in a single BLAS vector-matrix multiply
    svd_sess_list: list[float] | None = None
    svd_user_list: list[float] | None = None
    svd_mat = snapshot.svd_item_vectors
    svd_map = snapshot.svd_pid_to_idx
    if svd_map and (svd_sess_vec is not None or svd_user_vec is not None):
        svd_map_get = svd_map.get
        valid_pos: list[int] = []
        valid_svd: list[int] = []
        for idx, pid in enumerate(candidate_ids):
            si = svd_map_get(pid, -1)
            if si >= 0:
                valid_pos.append(idx)
                valid_svd.append(si)
        if valid_svd:
            cand_vecs = svd_mat[valid_svd]
            if svd_sess_vec is not None:
                dots = np.clip((cand_vecs @ svd_sess_vec), 0.0, 1.0).tolist()
                svd_sess_list = [0.0] * n
                for p_i, d_v in zip(valid_pos, dots):
                    svd_sess_list[p_i] = float(d_v)
            if svd_user_vec is not None:
                dots_u = np.clip((cand_vecs @ svd_user_vec), 0.0, 1.0).tolist()
                svd_user_list = [0.0] * n
                for p_i, d_v in zip(valid_pos, dots_u):
                    svd_user_list[p_i] = float(d_v)

    cand_cats = [""] * n if return_aux else []
    cand_subcats = [""] * n if return_aux else []
    cand_brands = [""] * n if return_aux else []
    cand_viewed = [False] * n if return_aux else []
    cand_lp = [0.0] * n if return_aux else []
    cand_is_new = [False] * n if return_aux else []
    cand_confirmed_cold = [False] * n if return_aux else []
    cand_exploit = [0.0] * n if return_aux else []
    cand_ucb = [0.0] * n if return_aux else []
    cand_in_active_cart = [False] * n if return_aux else []

    prod_get = snapshot.products.get
    st_get = snapshot.product_static.get
    has_seq = bool(seq_scores)
    has_sv = bool(self_view_scores)
    has_int = bool(intent_scores)
    has_ctx = bool(ctx_scores)
    has_rev = bool(rev_graft_scores)
    has_fwd = bool(fwd_cold_scores)
    has_ease = bool(ease_scores)
    has_hop2 = bool(hop2_scores)
    has_cov = bool(covisit_scores)
    has_vr = bool(view_recency_map)
    has_rvt = bool(rv_temporal_map)
    has_rvrep = bool(rv_repeat_norm_map)
    has_ca = bool(cart_active_decay_map)
    has_cf = bool(cart_forgotten_decay_map)
    has_rl = bool(rec_log_norm_map)
    has_aff = bool(affinity_map)
    has_rv_sc = bool(rv_subcat_frac)
    has_rv_br = bool(rv_brand_frac)
    has_anchor_cat = bool(anchor_cat)
    has_anchor_sc = bool(anchor_subcat)
    has_anchor_br = bool(anchor_brand)
    inv_std2 = 1.0 / (2.0 * std_lp * std_lp) if has_price_hist else 0.0

    item_sc_pop_get = snapshot.item_subcat_pop.get
    item_cat_pop_get = snapshot.item_cat_pop.get
    sc_price_get = snapshot.subcat_price_stats.get
    subcat_trans_map = (
        snapshot.subcat_transitions.get(anchor_subcat)
        if (has_anchor_sc and snapshot.subcat_transitions)
        else None
    )
    max_seq_pool = 0.0
    max_gpop_pool = 0.0

    seq_get = seq_scores.get
    sv_get = self_view_scores.get
    int_get = intent_scores.get
    ctx_get = ctx_scores.get
    rev_get = rev_graft_scores.get
    fwd_get = fwd_cold_scores.get
    ease_get = ease_scores.get
    hop2_get = hop2_scores.get
    cov_get = covisit_scores.get
    vr_get = view_recency_map.get
    rvt_get = rv_temporal_map.get
    rvrep_get = rv_repeat_norm_map.get
    ca_get = cart_active_decay_map.get
    cf_get = cart_forgotten_decay_map.get
    rl_get = rec_log_norm_map.get
    aff_get = affinity_map.get
    rv_sc_get = rv_subcat_frac.get
    rv_br_get = rv_brand_frac.get
    scvr_get = snapshot.self_cvr.get

    for idx, pid in enumerate(candidate_ids):
        r = rows[idx]
        if has_seq:
            seq_v = seq_get(pid, 0.0)
            seq_c = seq_v if seq_v < 1.8 else 1.8
            r[0] = seq_c
        else:
            seq_c = 0.0
        if has_sv:
            sv_raw = sv_get(pid, 0.0)
            sv_c = sv_raw * 5.0 if sv_raw < 0.24 else 1.2
            r[1] = sv_c
        else:
            sv_raw = 0.0
            sv_c = 0.0
        int_c = int_get(pid, 0.0) if has_int else 0.0
        if int_c:
            r[2] = int_c
        ctx_c = ctx_get(pid, 0.0) if has_ctx else 0.0
        if ctx_c:
            r[5] = ctx_c
        rev_c = rev_get(pid, 0.0) if has_rev else 0.0
        if rev_c:
            r[9] = rev_c
        fwd_c = fwd_get(pid, 0.0) if has_fwd else 0.0
        if fwd_c:
            r[10] = fwd_c
        if has_ease:
            ease_v = ease_get(pid, 0.0)
            ease_c = ease_v if ease_v < 1.8 else 1.8
            r[14] = ease_c
        else:
            ease_c = 0.0
        if has_hop2:
            hv = hop2_get(pid, 0.0)
            hop2_c = hv if hv < 1.5 else 1.5
            r[15] = hop2_c
        else:
            hop2_c = 0.0
        if has_cov:
            cv_ = cov_get(pid, 0.0)
            r[16] = cv_ if cv_ < 1.5 else 1.5
        svd_s = svd_sess_list[idx] if svd_sess_list is not None else 0.0
        svd_u = svd_user_list[idx] if svd_user_list is not None else 0.0
        r[17] = svd_s
        r[18] = svd_u
        if pid == last_viewed_pid:
            r[19] = 1.0
        vr_val = vr_get(pid, 0.0) if has_vr else 0.0
        if vr_val:
            r[20] = vr_val

        act_cart_d = ca_get(pid, 0.0) if has_ca else 0.0
        fgt_cart_d = cf_get(pid, 0.0) if has_cf else 0.0
        r[21] = act_cart_d
        r[22] = fgt_cart_d
        if has_rl:
            r[23] = rl_get(pid, 0.0)

        p_info = prod_get(pid)
        p_cat = p_info.category if p_info is not None else ""
        if p_info is not None and p_info.discount_ratio:
            r[24] = p_info.discount_ratio
        if has_anchor_cat and p_cat == anchor_cat:
            r[28] = 1.0

        r[29] = is_pg_prod
        r[30] = is_pg_cat
        r[31] = is_pg_home

        st = st_get(pid)
        if st is not None:
            sc, br, gpop, ltrend, is_new_f, lp_norm, exploit_d, ucb_b = st
            if has_aff:
                aff_val = aff_get((sc, br), 0.0)
                if aff_val == 0.0:
                    aff_val = aff_get((sc, ""), 0.0) * 0.6
                r[3] = aff_val

            pcompat_val = 0.0
            if has_price_hist and lp_norm > 0.0:
                diff_lp = (lp_norm * max_lp) - mean_lp
                z2 = diff_lp * diff_lp * inv_std2
                if z2 > z2_clip:
                    z2 = z2_clip
                base_pc = math.exp(-z2)
                if cheap_asym > 0.0 and diff_lp < 0.0:
                    pcompat_val = min(1.5, base_pc + cheap_asym * (1.0 - base_pc) * min(1.0, -diff_lp / max(std_lp, 0.15)))
                else:
                    pcompat_val = base_pc
            r[4] = pcompat_val
            r[6] = gpop
            r[7] = ltrend
            r[8] = is_new_f
            r[11] = lp_norm
            same_sc_f = 1.0 if (has_anchor_sc and sc == anchor_subcat) else 0.0
            r[12] = same_sc_f
            if has_anchor_br and br == anchor_brand:
                r[13] = 1.0
            if has_rv_sc:
                r[25] = rv_sc_get(sc, 0.0)
            if has_rv_br:
                r[26] = rv_br_get(br, 0.0)
            if mean_rv_lp > 0.0 and lp_norm > 0.0:
                r[27] = math.exp(-abs(lp_norm * max_lp - mean_rv_lp))

            if sc and lp_norm > 0.0:
                scp = sc_price_get(sc)
                if scp is not None:
                    sc_mu, sc_sd = scp
                    sc_z = ((lp_norm * max_lp) - sc_mu) / sc_sd
                    r[44] = math.exp(-0.5 * min(sc_z * sc_z, z2_clip))

            if return_aux:
                cand_cats[idx] = p_cat
                cand_subcats[idx] = sc
                cand_brands[idx] = br
                if sv_raw > 0.0:
                    cand_viewed[idx] = True
                cand_lp[idx] = lp_norm
                is_n = is_new_f > 0.5
                cand_is_new[idx] = is_n
                if is_n and rev_c > 0.35 and exploit_d > 0.02:
                    cand_confirmed_cold[idx] = True
                cand_exploit[idx] = exploit_d
                cand_ucb[idx] = ucb_b
                if act_cart_d > 0.0:
                    cand_in_active_cart[idx] = True
        else:
            sc = p_info.subcategory if p_info is not None else ""
            gpop = snapshot.global_pop_norm.get(pid, 0.0)
            r[6] = gpop
            r[7] = snapshot.live_trend_norm.get(pid, 0.0)
            same_sc_f = 0.0
            pcompat_val = 0.0
            lp_norm = 0.0
            if return_aux:
                cand_cats[idx] = p_cat
                if sv_raw > 0.0:
                    cand_viewed[idx] = True
                if act_cart_d > 0.0:
                    cand_in_active_cart[idx] = True

        # Non-linear explicit interactions (indices 32..39)
        if is_pg_prod:
            r[32] = same_sc_f
            t33 = seq_c + 0.5 * ease_c
            r[33] = t33 if t33 < 2.0 else 2.0
        elif is_pg_cat:
            r[34] = gpop
        elif is_pg_home:
            r[35] = svd_u if svd_u > int_c else int_c
        r[36] = seq_c * sv_c
        r[37] = ease_c * pcompat_val
        if act_cart_d > 0.0 or fgt_cart_d > 0.0:
            max_cart_d = act_cart_d if act_cart_d > fgt_cart_d else fgt_cart_d
            r[38] = max_cart_d * max(sv_c, scvr_get(pid, 0.02) * 4.0)
        if rev_c > 0.0 or fwd_c > 0.0:
            r[39] = (rev_c if rev_c > fwd_c else fwd_c) * gpop

        # Calypso 6.2 features (indices 40..49)
        if has_rvrep:
            r[40] = rvrep_get(pid, 0.0)
        if has_rvt:
            r[41] = rvt_get(pid, 0.0)
        sc_pop_val = item_sc_pop_get(pid, 0.0)
        r[42] = sc_pop_val
        r[43] = item_cat_pop_get(pid, 0.0)
        if vr_val > 0.0 and act_cart_d > 0.0:
            r[45] = vr_val * act_cart_d
        if same_sc_f > 0.5:
            r[46] = ease_c
            r[47] = hop2_c
            r[48] = svd_s
        if no_views_flag and ctx_c > 0.0:
            r[49] = ctx_c

        # Calypso 6.2a features (indices 50..55)
        if subcat_trans_map is not None and same_sc_f < 0.5 and sc:
            r[50] = subcat_trans_map.get(sc, 0.0)
        if sc_pop_val > 0.0 and r[19] < 0.5:
            r[53] = sc_pop_val
        n_active_ch = (
            (1 if seq_c > 0.10 else 0)
            + (1 if ease_c > 0.10 else 0)
            + (1 if svd_s > 0.18 else 0)
            + (1 if int_c > 0.15 else 0)
            + (1 if (act_cart_d > 0.05 or gpop > 0.72) else 0)
        )
        if n_active_ch > 0:
            r[54] = n_active_ch * 0.2
        if has_price_hist and lp_norm > 0.0:
            signed_p = ((lp_norm * max_lp) - mean_lp) / max(2.0 * std_lp, 0.30)
            r[55] = 1.0 if signed_p > 1.0 else (-1.0 if signed_p < -1.0 else signed_p)

        if seq_c > max_seq_pool:
            max_seq_pool = seq_c
        if gpop > max_gpop_pool:
            max_gpop_pool = gpop

    if max_seq_pool > 1e-6 or max_gpop_pool > 1e-6:
        inv_seq_p = 1.0 / max_seq_pool if max_seq_pool > 1e-6 else 0.0
        inv_gpop_p = 1.0 / max_gpop_pool if max_gpop_pool > 1e-6 else 0.0
        for r in rows:
            if inv_seq_p > 0.0 and r[0] > 0.0:
                r[51] = r[0] * inv_seq_p
            if inv_gpop_p > 0.0 and r[6] > 0.0:
                r[52] = r[6] * inv_gpop_p

    X = np.asarray(rows, dtype=np.float64)
    if not return_aux:
        return X

    aux = {
        "anchor_cat": anchor_cat,
        "anchor_subcat": anchor_subcat,
        "anchor_brand": anchor_brand,
        "page": page_l,
        "svd_sess_vec": svd_sess_vec,
        "svd_user_vec": svd_user_vec,
        "cand_cats": cand_cats,
        "cand_subcats": cand_subcats,
        "cand_brands": cand_brands,
        "cand_viewed": cand_viewed,
        "cand_lp": cand_lp,
        "cand_is_new": cand_is_new,
        "cand_confirmed_cold": cand_confirmed_cold,
        "cand_exploit": cand_exploit,
        "cand_ucb": cand_ucb,
        "cand_in_active_cart": cand_in_active_cart,
    }
    return X, aux


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
    intent_rows_list = list(user_intent_rows)
    cart_hist_list = list(cart_hist_rows)
    rec_hist_list = list(recurrent_hist_rows)

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
    raw_subcat_trans: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
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
        src_p = products.get(src)
        dst_p = products.get(dst)
        if (
            src_p is not None
            and dst_p is not None
            and src_p.subcategory
            and dst_p.subcategory
            and src_p.subcategory != dst_p.subcategory
        ):
            raw_subcat_trans[src_p.subcategory][dst_p.subcategory] += lift_score

    subcat_transitions: dict[str, dict[str, float]] = {}
    for s_sc, dst_map in raw_subcat_trans.items():
        if not dst_map:
            continue
        sorted_dsts = sorted(dst_map.items(), key=lambda kv: kv[1], reverse=True)[:10]
        max_sc_edge = sorted_dsts[0][1] if sorted_dsts else 0.0
        if max_sc_edge > 0.0:
            subcat_transitions[s_sc] = {
                d_sc: (val / max_sc_edge) for d_sc, val in sorted_dsts if val > 0.0
            }

    cov_blend = min(max(float(wp.covisit_blend_weight), 0.0), 1.0)
    covisit_edges, covisit_neighbors = (
        load_covisit_edges(active_ids=active_ids)
        if cov_blend > 0.0 or wp.w_covisit_score > 0.0
        else ([], {})
    )

    ease_w = min(max(float(wp.v3_ease_weight if weights_profile is not None else cfg.v3_ease_weight), 0.0), 1.0)
    use_ease_graph = (
        ease_w > 0.0 or wp.w_ease_score > 0.0 or wp.w_hop2_seq_score > 0.0 or wp.pool_ease_quota > 0
    )
    if use_ease_graph:
        ease_neighbors, hop2_transitions = fit_ease_from_transitions(
            pop_rows_list,
            trans_rows_list,
            active_ids,
            lam=max(float(wp.ease_lambda), 10.0),
            top_k=35,
            covisit_edges=covisit_edges,
            covisit_blend_weight=cov_blend,
            hop2_top_k=24,
            return_hop2=True,
        )
    else:
        ease_neighbors, hop2_transitions = {}, {}

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
    for row in intent_rows_list:
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
            cart_hist_list,
            rec_hist_list,
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

    svd_mat, svd_pid_idx = build_collaborative_svd_index(
        products,
        active_ids,
        transition_rows=trans_rows_list,
        covisit_edges=covisit_edges,
        user_intent_rows=intent_rows_list,
        cart_hist_rows=cart_hist_list,
        recurrent_hist_rows=rec_hist_list,
        svd_dim=max(8, int(wp.svd_dim)),
    )

    ranker = PositionDebiasedRanker.from_priors(
        prior_weights=wp.prior_weights_array(),
        min_prior_ratios=wp.min_prior_ratios_array(),
        max_prior_multiplier=wp.max_prior_multiplier,
    )
    (
        subcat_top_items,
        cat_top_items,
        subcat_price_stats,
        item_subcat_pop,
        item_cat_pop,
    ) = _build_category_indices(
        products,
        active_ids,
        global_pop_norm,
        live_trend_norm,
        top_k=24,
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
        ease_neighbors=ease_neighbors,
        hop2_transitions=hop2_transitions,
        covisit_neighbors=covisit_neighbors,
        svd_item_vectors=svd_mat,
        svd_pid_to_idx=svd_pid_idx,
        subcat_top_items=subcat_top_items,
        cat_top_items=cat_top_items,
        subcat_price_stats=subcat_price_stats,
        item_subcat_pop=item_subcat_pop,
        item_cat_pop=item_cat_pop,
        subcat_transitions=subcat_transitions,
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
    if wp.use_multisource_training and (combined_logs or cart_hist_list or intent_rows_list or trans_rows_list):
        X_train, y_train, ips_w = build_multisource_v6_training_dataset(
            snapshot,
            _extract_feature_matrix,
            combined_logs=combined_logs,
            cart_hist_rows=cart_hist_list,
            recurrent_hist_rows=rec_hist_list,
            user_intent_rows=intent_rows_list,
            transition_rows=trans_rows_list,
            weights_profile=wp,
        )
        if X_train.shape[0] > 0:
            ranker.fit(
                X_train,
                y_train,
                ips_w,
                max_blend_alpha=wp.ml_max_blend_alpha,
                min_positives=wp.ml_min_positives,
                l2_c=wp.ml_l2_c,
                nonlin_blend_alpha=wp.ml_nonlin_blend_alpha,
                nonlin_l2_alpha=wp.ml_nonlin_l2_alpha,
                enable_nonlin=wp.enable_nonlin_mlp,
            )
            ranker.val_auc = compute_ranker_holdout_auc(ranker, X_train, y_train)
    elif combined_logs:
        X_train, y_train, ips_w = build_ips_training_dataset(
            combined_logs,
            lambda req, cands: _extract_feature_matrix(snapshot, req, cands, weights_profile=wp),
            exam_power=wp.bandit_exam_power,
            max_sessions=12000,
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
            nonlin_blend_alpha=wp.ml_nonlin_blend_alpha,
            nonlin_l2_alpha=wp.ml_nonlin_l2_alpha,
            enable_nonlin=wp.enable_nonlin_mlp,
        )
        ranker.val_auc = compute_ranker_holdout_auc(ranker, X_train, y_train)

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
    """Sub-millisecond Calypso 6.1 Hybrid ML + Collaborative Graph + Page-Aware recommender."""

    def __init__(self, config: ConfigV4 | None = None) -> None:
        self.config = config or ConfigV4()
        self._svd_idx_cache_key: int = -1
        self._svd_idx_to_pid: list[str] = []

    def _get_svd_pid_list(self, snapshot: RecommenderSnapshotV4) -> list[str]:
        map_id = id(snapshot.svd_pid_to_idx)
        if map_id != self._svd_idx_cache_key:
            self._svd_idx_to_pid = list(snapshot.svd_pid_to_idx.keys())
            self._svd_idx_cache_key = map_id
        return self._svd_idx_to_pid

    def _gather_candidates(
        self,
        parsed: tuple[str, bool, str, str, str, list[str]],
        snapshot: RecommenderSnapshotV4,
        excluded_ids: frozenset[str],
        *,
        now_ms: float | None = None,
    ) -> list[str]:
        """Collect bounded multi-channel candidate pool with EASE, 2-hop, co-visit, SVD & bestsellers."""
        uid, is_new_user, region, device, page, recent_views = parsed
        wp = snapshot.weights_profile
        pool_limit = wp.candidate_pool_size if wp is not None else self.config.candidate_pool_size
        guaranteed_top = wp.guaranteed_top_sellers_in_pool if wp is not None else 18
        ease_quota = wp.pool_ease_quota if wp is not None else 10
        hop2_quota = wp.pool_hop2_quota if wp is not None else 10
        cov_quota = wp.pool_covisit_quota if wp is not None else 8
        svd_quota = wp.pool_svd_quota if wp is not None else 10
        subcat_quota = wp.pool_subcat_top_quota if wp is not None else 10
        cold_quota = wp.pool_cold_quota if wp is not None else 6
        max_forgot_ms = (wp.v4_forgotten_max_days if wp is not None else 30.0) * 86_400_000.0
        cur_ms = _resolve_now_ms(snapshot, now_ms, None)

        seen: set[str] = set()
        candidates: list[str] = []
        active_ids = snapshot.active_ids
        seen_add = seen.add
        cands_append = candidates.append

        def _add(pid: str) -> bool:
            if len(candidates) >= pool_limit:
                return False
            if not pid or pid in seen or pid not in active_ids or pid in excluded_ids:
                return True
            seen_add(pid)
            cands_append(pid)
            return True

        # 1. Recent views first
        for seen_pid in reversed(recent_views):
            if not _add(seen_pid):
                return candidates

        # 2. Valid non-expired cart & recurrent items
        fresh_cart_pids: list[str] = []
        if uid and not is_new_user:
            for pid, cart_ts in snapshot.user_carts.get(uid, ()):
                if -60_000.0 <= cur_ms - cart_ts <= max_forgot_ms:
                    fresh_cart_pids.append(pid)
            for pid in fresh_cart_pids[:8]:
                _add(pid)
            for pid, _ in snapshot.user_recurrent.get(uid, ())[:6]:
                _add(pid)

        # Reserve tail strictly so ALL guaranteed_top bestsellers fit before slot-fillers
        pre_top_cap = max(pool_limit - guaranteed_top, 44)
        reserve_tail = min(
            pool_limit // 2,
            guaranteed_top + ease_quota // 2 + hop2_quota // 2 + svd_quota // 2 + subcat_quota // 2,
        )
        trans_cap = min(pre_top_cap, max(pool_limit - reserve_tail, 36))
        for v_idx, seen_pid in enumerate(reversed(recent_views)):
            if len(candidates) >= trans_cap:
                break
            per_view_cap = 20 if v_idx == 0 else 10
            added_view = 0
            for dst_pid, _ in snapshot.transitions.get(seen_pid, ()):
                if len(candidates) >= trans_cap or added_view >= per_view_cap:
                    break
                if dst_pid not in seen and dst_pid in active_ids and dst_pid not in excluded_ids:
                    _add(dst_pid)
                    added_view += 1
            for new_pid, _ in snapshot.reverse_cold_neighbors.get(seen_pid, ())[:6]:
                if len(candidates) >= trans_cap:
                    break
                _add(new_pid)
            for warm_nb, _ in snapshot.cold_neighbors.get(seen_pid, ())[:6]:
                if len(candidates) >= trans_cap:
                    break
                _add(warm_nb)
                for dst_pid, _ in snapshot.transitions.get(warm_nb, ())[:3]:
                    if len(candidates) >= trans_cap:
                        break
                    _add(dst_pid)

        # 4. Pure EASE + 2-hop diffusion + pre-CUT co-visitation channels
        if ease_quota > 0 and snapshot.ease_neighbors and recent_views and len(candidates) < pre_top_cap:
            added_ease = 0
            for seen_pid in reversed(recent_views):
                for dst_pid, _ in snapshot.ease_neighbors.get(seen_pid, ()):
                    if len(candidates) >= pre_top_cap or added_ease >= ease_quota:
                        break
                    if dst_pid not in seen and dst_pid in active_ids and dst_pid not in excluded_ids:
                        _add(dst_pid)
                        added_ease += 1
                if len(candidates) >= pre_top_cap or added_ease >= ease_quota:
                    break

        if hop2_quota > 0 and snapshot.hop2_transitions and recent_views and len(candidates) < pre_top_cap:
            added_h2 = 0
            for seen_pid in reversed(recent_views[-4:]):
                for dst_pid, _ in snapshot.hop2_transitions.get(seen_pid, ()):
                    if len(candidates) >= pre_top_cap or added_h2 >= hop2_quota:
                        break
                    if dst_pid not in seen and dst_pid in active_ids and dst_pid not in excluded_ids:
                        _add(dst_pid)
                        added_h2 += 1
                if len(candidates) >= pre_top_cap or added_h2 >= hop2_quota:
                    break

        if cov_quota > 0 and snapshot.covisit_neighbors and recent_views and len(candidates) < pre_top_cap:
            added_cov = 0
            for seen_pid in reversed(recent_views[-3:]):
                for dst_pid, _ in snapshot.covisit_neighbors.get(seen_pid, ()):
                    if len(candidates) >= pre_top_cap or added_cov >= cov_quota:
                        break
                    if dst_pid not in seen and dst_pid in active_ids and dst_pid not in excluded_ids:
                        _add(dst_pid)
                        added_cov += 1
                if len(candidates) >= pre_top_cap or added_cov >= cov_quota:
                    break

        # 5. User intent items
        profile = None if is_new_user else snapshot.user_profiles.get(uid)
        if profile is not None and len(candidates) < pre_top_cap:
            for pid, _ in profile.intent_items:
                if len(candidates) >= pre_top_cap:
                    break
                _add(pid)

        # 6. Fast collaborative SVD retrieval (single BLAS matmul)
        if svd_quota > 0 and snapshot.svd_pid_to_idx and snapshot.svd_item_vectors is not None and len(candidates) < pre_top_cap:
            q_items: list[tuple[str, float]] = [(p, 0.7 ** i) for i, p in enumerate(reversed(recent_views))]
            if profile is not None:
                q_items.extend((p, sc * 0.7) for p, sc in profile.intent_items[:6])
            if fresh_cart_pids:
                q_items.extend((p, 0.9) for p in fresh_cart_pids[:4])
            svd_q = build_query_vector(q_items, snapshot.svd_item_vectors, snapshot.svd_pid_to_idx)
            if svd_q is not None:
                dots = snapshot.svd_item_vectors @ svd_q
                pid_list = self._get_svd_pid_list(snapshot)
                kk = min(svd_quota + 12, len(dots))
                if kk > 0:
                    top_idx = np.argpartition(-dots, kk - 1)[:kk]
                    top_idx = top_idx[np.argsort(-dots[top_idx])]
                    added_svd = 0
                    for i_svd in top_idx:
                        if added_svd >= svd_quota or len(candidates) >= pre_top_cap:
                            break
                        cand_p = pid_list[int(i_svd)]
                        if cand_p not in seen and cand_p in active_ids and cand_p not in excluded_ids:
                            _add(cand_p)
                            added_svd += 1

        # 7. Intra-subcategory & macro-category bestsellers for session/profile exploration
        if subcat_quota > 0 and (snapshot.subcat_top_items or snapshot.cat_top_items) and len(candidates) < pre_top_cap:
            explored_subcats: list[str] = []
            explored_cats: list[str] = []
            for seen_pid in reversed(recent_views[-4:]):
                p_info = snapshot.products.get(seen_pid)
                if p_info is None:
                    continue
                if p_info.subcategory and p_info.subcategory not in explored_subcats:
                    explored_subcats.append(p_info.subcategory)
                if p_info.category and p_info.category not in explored_cats:
                    explored_cats.append(p_info.category)
            if not explored_subcats and profile is not None and profile.subcat_brand_affinity:
                top_sc_pairs = sorted(
                    ((sc, sc_v) for (sc, _br), sc_v in profile.subcat_brand_affinity.items() if sc),
                    key=lambda kv: -kv[1],
                )
                for sc, _ in top_sc_pairs:
                    if sc not in explored_subcats:
                        explored_subcats.append(sc)
                    if len(explored_subcats) >= 3:
                        break
            added_subcat = 0
            for sc in explored_subcats[:3]:
                for pid, _ in snapshot.subcat_top_items.get(sc, ()):
                    if added_subcat >= subcat_quota or len(candidates) >= pre_top_cap:
                        break
                    if pid not in seen and pid in active_ids and pid not in excluded_ids:
                        _add(pid)
                        added_subcat += 1
                if added_subcat >= subcat_quota or len(candidates) >= pre_top_cap:
                    break
            if added_subcat < subcat_quota and explored_cats and len(candidates) < pre_top_cap:
                for mc in explored_cats[:2]:
                    for pid, _ in snapshot.cat_top_items.get(mc, ()):
                        if added_subcat >= subcat_quota or len(candidates) >= pre_top_cap:
                            break
                        if pid not in seen and pid in active_ids and pid not in excluded_ids:
                            _add(pid)
                            added_subcat += 1
                    if added_subcat >= subcat_quota or len(candidates) >= pre_top_cap:
                        break

        # 7b. Cross-subcategory bridge bestsellers (for complementary cross-category purchases)
        cross_sc_quota = wp.pool_cross_subcat_quota if wp is not None else 6
        if cross_sc_quota > 0 and recent_views and snapshot.subcat_transitions and snapshot.subcat_top_items and len(candidates) < pre_top_cap:
            last_p_info = snapshot.products.get(recent_views[-1])
            anchor_sc_g = last_p_info.subcategory if last_p_info is not None else ""
            if anchor_sc_g and anchor_sc_g in snapshot.subcat_transitions:
                added_cross = 0
                cross_targets = sorted(
                    snapshot.subcat_transitions[anchor_sc_g].items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                for cross_sc, _ in cross_targets[:3]:
                    for pid, _ in snapshot.subcat_top_items.get(cross_sc, ())[:3]:
                        if added_cross >= cross_sc_quota or len(candidates) >= pre_top_cap:
                            break
                        if pid not in seen and pid in active_ids and pid not in excluded_ids:
                            _add(pid)
                            added_cross += 1
                    if added_cross >= cross_sc_quota or len(candidates) >= pre_top_cap:
                        break

        # 8. Guaranteed Top-N bestsellers
        for pid in snapshot.fallback_ranked[:guaranteed_top]:
            if not _add(pid):
                break

        # 9. Contextual buckets & ranked new items
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

        if snapshot.new_product_ids and cold_quota > 0 and len(candidates) < pool_limit:
            anchor_subcat = ""
            anchor_cat = ""
            if recent_views and recent_views[-1] in snapshot.products:
                anchor_info = snapshot.products[recent_views[-1]]
                anchor_subcat = anchor_info.subcategory
                anchor_cat = anchor_info.category
            scored_cold: list[tuple[int, float, str]] = []
            for new_pid in snapshot.new_product_ids:
                if new_pid in seen or new_pid not in active_ids or new_pid in excluded_ids:
                    continue
                info = snapshot.products.get(new_pid)
                if info is None:
                    continue
                if anchor_subcat and info.subcategory == anchor_subcat:
                    tier = 2
                elif anchor_cat and info.category == anchor_cat:
                    tier = 1
                elif not anchor_subcat and not anchor_cat:
                    tier = 1
                else:
                    continue
                pop_t = snapshot.global_pop_norm.get(new_pid, 0.0) + snapshot.live_trend_norm.get(new_pid, 0.0)
                scored_cold.append((tier, pop_t, new_pid))
            scored_cold.sort(key=lambda t: (-t[0], -t[1], t[2]))
            added_new = 0
            for _, _, new_pid in scored_cold:
                if added_new >= cold_quota or len(candidates) >= pool_limit:
                    break
                if _add(new_pid):
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
        full_parsed = _parse_request_fields_with_ts(request)
        uid, is_new_user, region, device, page, recent_views, view_ts_list, req_ts_ms = full_parsed
        parsed = (uid, is_new_user, region, device, page, recent_views)
        eff_now_ms = _resolve_now_ms(snapshot, now_ms, req_ts_ms)

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
                    eff_now_ms,
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
            self._recommend_core(
                parsed,
                snapshot,
                k_model,
                core_excluded,
                _cfg=cfg,
                now_ms=eff_now_ms,
                view_ts_list=view_ts_list,
                req_ts_ms=req_ts_ms,
            )
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
        view_ts_list: list[float | None] | None = None,
        req_ts_ms: float | None = None,
    ) -> list[str]:
        """Calypso 6.2 Hybrid Scoring: 50-feature Linear IPS + IPS-Distilled MLP + Page Matrices."""
        cfg = _cfg if _cfg is not None else self.config
        wp = snapshot.weights_profile if snapshot.weights_profile is not None else cfg.load_weights_profile()
        uid, is_new_user, _, _, page, recent_views = parsed
        eff_now_ms = _resolve_now_ms(snapshot, now_ms, req_ts_ms)
        has_session_intent = bool(recent_views)
        is_cold_session = is_new_user or any(
            pid in snapshot.new_product_ids for pid in recent_views
        )

        candidates = self._gather_candidates(parsed, snapshot, excluded_ids, now_ms=eff_now_ms)
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
        m_seq, m_int, m_aff, m_ctx, m_pop, m_trd, m_cld, m_es = wp.segment_multipliers(segment)
        p_seq, p_cvr, p_pop, p_aff, p_es, p_cart, p_rec, p_mmr = wp.page_multipliers(page)

        X, aux = _extract_feature_matrix(
            snapshot,
            None,
            candidates,
            parsed=parsed,
            weights_profile=wp,
            now_ms=eff_now_ms,
            view_ts_list=view_ts_list,
            req_ts_ms=req_ts_ms,
            return_aux=True,
        )

        anchor_cat = str(aux["anchor_cat"])
        anchor_subcat = str(aux["anchor_subcat"])
        page_l = str(aux["page"])
        cand_cats: list[str] = aux["cand_cats"]
        cand_subcats: list[str] = aux["cand_subcats"]
        cand_brands: list[str] = aux["cand_brands"]
        cand_viewed: list[bool] = aux["cand_viewed"]
        cand_lp: list[float] = aux["cand_lp"]
        cand_is_new: list[bool] = aux["cand_is_new"]
        cand_confirmed_cold: list[bool] = aux["cand_confirmed_cold"]
        cand_in_active_cart: list[bool] = aux["cand_in_active_cart"]

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

        w_tup = ranker.resolve_weights_tuple(
            blend_alpha_scale=overrides.ml_blend_alpha_scale,
            reverse_graft_override=overrides.weight_reverse_cold_graft,
            forward_cold_override=overrides.weight_forward_cold,
            revenue_tie_override=rev_tie_override,
        )
        w_req = np.asarray(w_tup, dtype=np.float64).copy()

        # Segment + Page conditional modulation on linear ranker weights
        w_req[0] *= m_seq * p_seq
        w_req[1] *= p_cvr
        w_req[2] *= m_int * p_aff
        w_req[3] *= m_aff * p_aff
        w_req[5] *= m_ctx * p_pop
        w_req[6] *= m_pop * p_pop
        w_req[7] *= m_trd * p_pop

        cold_sess_mult = wp.cold_boost_on_cold_session if is_cold_session else 1.0
        w_req[8] *= cold_sess_mult
        w_req[9] *= m_cld * cold_sess_mult
        w_req[10] *= m_cld * cold_sess_mult

        if not (wp.v3_no_subcat_floor or cfg.v3_no_subcat_floor):
            if w_req[12] < wp.subcat_floor_min:
                w_req[12] = wp.subcat_floor_min
            if w_req[13] < wp.brand_floor_min:
                w_req[13] = wp.brand_floor_min

        for ci in (14, 15, 16, 17, 18):
            w_req[ci] *= m_es * p_es
        w_req[19] *= p_seq
        w_req[20] *= p_cvr
        w_req[21] *= p_cart
        w_req[22] *= p_cart
        w_req[23] *= p_rec
        w_req[25] *= p_seq
        w_req[26] *= p_seq
        if len(w_req) >= 50:
            w_req[40] *= p_seq
            w_req[41] *= p_seq
            w_req[42] *= p_pop
            w_req[43] *= p_pop
            w_req[45] *= p_cart
            w_req[46] *= m_es * p_es
            w_req[47] *= m_es * p_es
            w_req[48] *= m_es * p_es
            w_req[49] *= m_ctx * p_pop
        if len(w_req) >= 56:
            w_req[50] *= m_seq * p_seq
            w_req[51] *= p_seq
            w_req[52] *= m_pop * p_pop
            w_req[53] *= p_pop
            w_req[54] *= p_seq

        lin_vec = X @ w_req
        nl_vec = ranker.score_nonlin_residuals(
            X,
            nonlin_alpha_override=wp.ml_nonlin_blend_alpha,
            nonlin_temp=wp.ml_nonlin_temp,
        )

        # Continuous business, cart decay, recurrent, discount & category rules (only when active)
        exploit_vec = np.asarray(aux["cand_exploit"], dtype=np.float64)
        ucb_vec = np.asarray(aux["cand_ucb"], dtype=np.float64)
        rule_vec = np.zeros(X.shape[0], dtype=np.float64)
        if wp.cart_active_score_bonus and np.any(X[:, 21]):
            rule_vec += wp.cart_active_score_bonus * p_cart * X[:, 21]
        if wp.cart_forgotten_score_bonus and np.any(X[:, 22]):
            rule_vec += wp.cart_forgotten_score_bonus * p_cart * X[:, 22]
        if wp.recurrent_score_bonus and np.any(X[:, 23]):
            r_col = X[:, 23]
            rule_vec += wp.recurrent_score_bonus * p_rec * np.where(r_col > 0.0, 1.0 + wp.recurrent_log_scale * r_col, 0.0)
        if wp.discount_boost_weight and np.any(X[:, 24]):
            d_col = np.clip(X[:, 24], 0.0, 1.0)
            if abs(wp.discount_nonlin_gamma - 1.0) > 1e-6:
                d_col = np.power(d_col, max(wp.discount_nonlin_gamma, 0.2))
            rule_vec += wp.discount_boost_weight * d_col
        if wp.same_cat_bonus_weight and np.any(X[:, 28]):
            rule_vec += wp.same_cat_bonus_weight * np.where(X[:, 12] < 0.5, X[:, 28], 0.0)

        v3_bg_penalty = wp.v3_bg_penalty and cfg.v3_bg_penalty
        if v3_bg_penalty and has_session_intent:
            no_direct = (
                (X[:, 0] == 0.0)
                & (X[:, 1] == 0.0)
                & (X[:, 2] == 0.0)
                & (X[:, 10] == 0.0)
                & (X[:, 14] == 0.0)
            )
            if np.any(no_direct):
                bg_part = w_req[5] * X[:, 5] + w_req[6] * X[:, 6] + w_req[7] * X[:, 7] + exploit_vec
                bg_factor = np.where(X[:, 12] > 0.5, wp.bg_penalty_same_subcat, wp.bg_penalty_diff_subcat)
                lin_vec = lin_vec - np.where(no_direct, bg_part * (1.0 - bg_factor), 0.0)

        prec_arr = lin_vec + nl_vec + rule_vec + exploit_vec
        full_arr = prec_arr + ucb_vec
        prec_scores = prec_arr.tolist()
        full_scores = full_arr.tolist()

        n_cands = len(candidates)
        if has_session_intent:
            mmr_start = wp.mmr_start_slot_session
            subcat_lam = wp.mmr_subcat_lambda_session * p_mmr
        else:
            mmr_start = wp.mmr_start_slot_home
            subcat_lam = wp.mmr_subcat_lambda_home * p_mmr
        brand_lam = wp.mmr_brand_lambda if wp.mmr_brand_lambda != cfg.mmr_brand_lambda else overrides.mmr_brand_lambda

        if page_l == "product":
            cart_top3_pen = wp.cart_top3_penalty_product
        elif page_l == "category":
            cart_top3_pen = wp.cart_top3_penalty_category
        elif page_l == "home":
            cart_top3_pen = wp.cart_top3_penalty_home
        else:
            cart_top3_pen = 0.0

        picked = [False] * n_cands
        subcat_counts: dict[str, int] = defaultdict(int)
        brand_counts: dict[str, int] = defaultdict(int)
        selected: list[str] = []

        cold_gate = wp.cold_explore_start_slot_cold if is_cold_session else wp.cold_explore_start_slot_warm
        max_cold_warm = wp.max_cold_explore_in_warm
        slot1_cat_bonus = wp.slot1_category_match_bonus
        slot1_cat_only_bonus = slot1_cat_bonus * max(0.0, wp.slot1_cat_only_ratio)
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
            apply_cart_top3_pen = slot_pos < 3 and cart_top3_pen > 0.0

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
                if apply_cart_top3_pen and cand_in_active_cart[idx]:
                    val -= cart_top3_pen
                if apply_slot1_cat:
                    if cand_viewed[idx] or (anchor_subcat and cand_subcats[idx] == anchor_subcat):
                        val += slot1_cat_bonus
                    elif anchor_cat and cand_cats[idx] == anchor_cat:
                        val += slot1_cat_only_bonus
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
