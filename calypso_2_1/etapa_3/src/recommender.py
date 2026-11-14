"""Calypso-2.1 closed-loop hybrid ML & position-aware bandit recommender."""
from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .agent_scratch import CalypsoAgent
from .bandits import PositionAwareBandit
from .cold_start import ColdStartIndexV3
from .config import ConfigV3
from .guardrails import ensure_valid_submission
from .reranker import NUM_FEATURES, PositionDebiasedRanker, build_ips_training_dataset
from .state import ProductInfo, RecommenderSnapshotV3, UserProfile


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        res = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(res) else res


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

        products[pid] = ProductInfo(
            product_id=pid,
            category=str(row.get("category") or (existing.category if existing else "")).strip(),
            subcategory=str(row.get("subcategory") or (existing.subcategory if existing else "")).strip(),
            brand=str(row.get("brand") or (existing.brand if existing else "")).strip(),
            price_eur=price,
            description=str(row.get("description") or (existing.description if existing else "")).strip(),
            is_active=is_active,
            is_new=bool(row.get("is_new_product", existing.is_new if existing else True)),
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
    """Build 3-level Bayesian contextual popularity tables and global normalized rates."""
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


def _extract_feature_matrix(
    snapshot: RecommenderSnapshotV3,
    request: Mapping[str, Any] | Any,
    candidate_ids: Sequence[str],
    parsed: tuple[str, bool, str, str, str, list[str]] | None = None,
) -> np.ndarray:
    """Vectorized 14-feature builder shared identically between training and inference."""
    n = len(candidate_ids)
    if n == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float64)

    user_id, _, region, device, page, recent_views = (
        parsed if parsed is not None else _parse_request_fields(request)
    )

    # Pre-aggregate request-level sequential, cold-start, and contextual lookup tables
    seq_scores: dict[str, float] = defaultdict(float)
    fwd_cold_scores: dict[str, float] = defaultdict(float)
    rev_graft_scores: dict[str, float] = defaultdict(float)
    self_view_scores: dict[str, float] = {}

    decay = 0.65
    for rev_idx, seen_pid in enumerate(reversed(recent_views)):
        w_rec = decay**rev_idx
        if seen_pid in snapshot.active_ids and seen_pid not in self_view_scores:
            page_factor = 0.65 if (page == "product" and rev_idx == 0) else 1.0
            self_view_scores[seen_pid] = snapshot.self_cvr.get(seen_pid, 0.02) * w_rec * page_factor

        trans_nbs = snapshot.transitions.get(seen_pid)
        if trans_nbs:
            for dst_pid, weight in trans_nbs:
                if dst_pid in snapshot.active_ids:
                    seq_scores[dst_pid] += w_rec * weight

        if not trans_nbs or len(trans_nbs) < 3:
            cold_nbs = snapshot.cold_neighbors.get(seen_pid)
            if cold_nbs:
                for warm_nb, sim in cold_nbs:
                    if warm_nb in snapshot.active_ids:
                        fwd_cold_scores[warm_nb] = max(
                            fwd_cold_scores[warm_nb],
                            w_rec * sim * 0.60,
                        )
                    for nb_dst, nb_w in snapshot.transitions.get(warm_nb, ()):
                        if nb_dst in snapshot.active_ids:
                            fwd_cold_scores[nb_dst] = max(
                                fwd_cold_scores[nb_dst],
                                w_rec * sim * nb_w * 0.45,
                            )

        for new_pid, sim in snapshot.reverse_cold_neighbors.get(seen_pid, ()):
            if new_pid in snapshot.active_ids:
                rev_graft_scores[new_pid] = max(rev_graft_scores[new_pid], w_rec * sim)

    # Contextual popularity lookup hierarchy: level-3 -> level-2 -> level-1
    ctx_scores: dict[str, float] = {}
    for bucket, weight_scale in (
        (snapshot.ctx_3.get((region, device, page), ()), 1.0),
        (snapshot.ctx_2.get((region, device), ()), 0.70),
        (snapshot.ctx_1.get((page,), ()), 0.45),
    ):
        for pid, score in bucket:
            if pid not in ctx_scores:
                ctx_scores[pid] = score * weight_scale

    # User profile & intent lookups
    profile = snapshot.user_profiles.get(user_id)
    intent_scores: dict[str, float] = {}
    affinity_map: Mapping[tuple[str, str], float] = {}
    has_price_hist = False
    mean_lp = 0.0
    std_lp = 0.5
    if profile is not None:
        intent_scores = dict(profile.intent_items)
        affinity_map = profile.subcat_brand_affinity
        if profile.log_price_mean > 0.0:
            has_price_hist = True
            mean_lp = profile.log_price_mean
            std_lp = max(profile.log_price_std, 0.45)

    anchor_subcat = ""
    anchor_brand = ""
    if recent_views:
        last_info = snapshot.products.get(recent_views[-1])
        if last_info is not None:
            anchor_subcat = last_info.subcategory
            anchor_brand = last_info.brand

    max_lp = max(snapshot.max_log_price, 1e-6)
    X = np.zeros((n, NUM_FEATURES), dtype=np.float64)

    for idx, pid in enumerate(candidate_ids):
        info = snapshot.products.get(pid)
        X[idx, 0] = min(seq_scores.get(pid, 0.0), 1.8)
        X[idx, 1] = min(self_view_scores.get(pid, 0.0) * 5.0, 1.2)
        X[idx, 2] = intent_scores.get(pid, 0.0)

        if info is not None:
            aff_val = affinity_map.get((info.subcategory, info.brand), 0.0)
            if aff_val == 0.0:
                aff_val = affinity_map.get((info.subcategory, ""), 0.0) * 0.6
            X[idx, 3] = aff_val

            lp = math.log1p(max(info.price_eur, 0.0))
            if has_price_hist and info.price_eur > 0.0:
                z = (lp - mean_lp) / std_lp
                X[idx, 4] = math.exp(-0.5 * min(z * z, 6.0))
            X[idx, 8] = 1.0 if info.is_new else 0.0
            X[idx, 11] = lp / max_lp
            X[idx, 12] = 1.0 if (anchor_subcat and info.subcategory == anchor_subcat) else 0.0
            X[idx, 13] = 1.0 if (anchor_brand and info.brand == anchor_brand) else 0.0

        X[idx, 5] = ctx_scores.get(pid, 0.0)
        X[idx, 6] = snapshot.global_pop_norm.get(pid, 0.0)
        X[idx, 7] = snapshot.live_trend_norm.get(pid, 0.0)
        X[idx, 9] = rev_graft_scores.get(pid, 0.0)
        X[idx, 10] = fwd_cold_scores.get(pid, 0.0)

    return X


def build_snapshot_v3_from_records(
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
    config: ConfigV3 | None = None,
    version: int = 1,
    agent: CalypsoAgent | None = None,
) -> tuple[RecommenderSnapshotV3, list[dict[str, Any]]]:
    """Construct complete RecommenderSnapshotV3 and initial scratch telemetry."""
    cfg = config or ConfigV3()
    oos_set = frozenset(str(x).strip() for x in out_of_stock_ids if x)
    products, active_ids, new_product_ids, max_log_price = _build_products(
        catalog_rows,
        catalog_updates_rows,
        oos_set,
    )

    # 1. Popularity & Empirical Self-CVR
    self_cvr: dict[str, float] = {}
    raw_pop: dict[str, float] = {}
    for row in popularity_rows:
        pid = str(row.get("product_id") or "").strip()
        if pid not in active_ids:
            continue
        views = _safe_float(row.get("views"), 0.0)
        purchases = _safe_float(row.get("purchases"), 0.0)
        cvr_val = row.get("self_cvr")
        self_cvr[pid] = (
            _safe_float(cvr_val, 0.02)
            if cvr_val is not None
            else (purchases + 0.5) / (views + 25.0)
        )
        wpop = max(_safe_float(row.get("weighted_pop"), 0.0), 0.0)
        if wpop > 0.0:
            raw_pop[pid] = math.log1p(wpop)

    max_pop = max(raw_pop.values(), default=1.0) or 1.0
    global_pop_norm = {pid: v / max_pop for pid, v in raw_pop.items()}

    # 2. Live trends
    raw_live: dict[str, float] = {}
    for row in live_trend_rows:
        pid = str(row.get("product_id") or "").strip()
        if pid in active_ids:
            sc = max(_safe_float(row.get("live_score"), 0.0), 0.0)
            if sc > 0.0:
                raw_live[pid] = math.log1p(sc)
    max_live = max(raw_live.values(), default=1.0) or 1.0
    live_trend_norm = {pid: v / max_live for pid, v in raw_live.items()}

    # 3. Bayesian Lift-smoothed directed transitions
    raw_trans: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in transition_rows:
        src = str(row.get("src_product") or "").strip()
        dst = str(row.get("dst_product") or "").strip()
        if not src or dst not in active_ids or src == dst:
            continue
        raw_score = max(_safe_float(row.get("raw_transition_score", row.get("score")), 0.0), 0.0)
        if raw_score <= 0.0:
            continue
        dst_pop = max(math.expm1(raw_pop.get(dst, 0.0)), 0.0)
        lift_score = raw_score / ((dst_pop + cfg.lift_prior_pop) ** cfg.lift_beta)
        raw_trans[src].append((dst, lift_score))

    transitions: dict[str, tuple[tuple[str, float], ...]] = {}
    for src, edges in raw_trans.items():
        edges.sort(key=lambda kv: kv[1], reverse=True)
        top_edges = edges[:35]
        max_s = top_edges[0][1] if top_edges else 1.0
        if max_s > 0.0:
            transitions[src] = tuple((dst, sc / max_s) for dst, sc in top_edges)

    # 4. Contextual Bayesian popularity
    ctx_3, ctx_2, ctx_1, _ = _build_contextual_maps(context_rows, active_ids)

    # 5. Deep user profiles
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

    # 6. Bidirectional Cold Start Graph (warm <-> new/sparse)
    warm_ids = [
        pid
        for pid, sc in sorted(global_pop_norm.items(), key=lambda kv: kv[1], reverse=True)
        if pid in active_ids and pid not in new_product_ids
    ]
    if not warm_ids:
        warm_ids = [pid for pid in active_ids if pid not in new_product_ids] or list(active_ids)

    sparse_or_cold_targets = [
        info
        for pid, info in products.items()
        if info.is_new or pid in active_ids and len(transitions.get(pid, ())) < 3
    ]
    new_infos = [products[pid] for pid in new_product_ids if pid in products]

    cs_index = ColdStartIndexV3()
    cs_index.fit(products, warm_ids)
    cold_neighbors, reverse_cold_neighbors = cs_index.compute_bidirectional(
        sparse_or_cold_targets,
        new_infos,
        top_k_forward=cfg.max_cold_neighbors,
        top_k_reverse=cfg.max_reverse_cold_per_warm,
    )

    # 7. Fallbacks (popularity + revenue)
    fallback_scores: dict[str, float] = {}
    for pid in active_ids:
        p_obj = products[pid]
        pop_s = global_pop_norm.get(pid, 0.0) + 0.6 * live_trend_norm.get(pid, 0.0)
        lp_s = math.log1p(max(p_obj.price_eur, 0.0)) / max_log_price
        fallback_scores[pid] = pop_s + cfg.revenue_tie_weight * lp_s + (0.04 if p_obj.is_new else 0.0)

    fallback_ranked = tuple(
        sorted(active_ids, key=lambda pid: (fallback_scores[pid], pid), reverse=True)
    )
    revenue_fallback = tuple(
        sorted(active_ids, key=lambda pid: (products[pid].price_eur, pid), reverse=True)
    )

    # 8. Fit Position-Aware Bandit on feedback (fallback to training sessions)
    fb_list = list(feedback_rows)
    hist_list = list(training_session_rows)
    bandit = PositionAwareBandit(
        exam_power=cfg.bandit_exam_power,
        prior_strength=cfg.bandit_prior_strength,
        cold_ucb_multiplier=cfg.bandit_cold_ucb_multiplier,
    )
    bandit_source = fb_list if fb_list else hist_list
    bandit_state = bandit.fit_from_feedback(bandit_source, fallback_cvr=self_cvr)

    # Build temporary snapshot before ML fit so feature extractor shares identical pipeline
    ranker = PositionDebiasedRanker.from_priors()
    snapshot = RecommenderSnapshotV3(
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
    )

    # 9. Train IPS Position-Debiased ML Re-Ranker on feedback + history sessions (zero train/serve skew)
    combined_logs = fb_list + hist_list
    if combined_logs:
        X_train, y_train, ips_w = build_ips_training_dataset(
            combined_logs,
            lambda req, cands: _extract_feature_matrix(snapshot, req, cands),
            exam_power=cfg.bandit_exam_power,
            max_sessions=3000,
        )
        ranker.fit(
            X_train,
            y_train,
            ips_w,
            max_blend_alpha=cfg.ml_max_blend_alpha,
            min_positives=cfg.ml_min_positives,
            l2_c=cfg.ml_l2_c,
        )

    # 10. CalypsoAgent diagnosis + scratch persistence payload
    calypso_agent = agent or CalypsoAgent()
    overrides, _, base_scratch = calypso_agent.diagnose_and_adapt(
        cfg,
        products,
        active_ids,
        new_product_ids,
        fb_list,
        bandit_state,
        ranker,
    )
    snapshot.agent_overrides = overrides
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


class CalypsoRecommender:
    """Fast-Loop runtime scorer (<1 ms/request) powered by ML Re-Ranker + Bandit + MMR."""

    def __init__(self, config: ConfigV3 | None = None) -> None:
        self.config = config or ConfigV3()
        self.bandit = PositionAwareBandit(
            exam_power=self.config.bandit_exam_power,
            prior_strength=self.config.bandit_prior_strength,
            cold_ucb_multiplier=self.config.bandit_cold_ucb_multiplier,
        )

    def _gather_candidates(
        self,
        parsed: tuple[str, bool, str, str, str, list[str]],
        snapshot: RecommenderSnapshotV3,
        excluded_ids: frozenset[str],
    ) -> list[str]:
        """Collect bounded multi-channel candidate pool preserving active order."""
        uid, is_new_user, region, device, page, recent_views = parsed
        pool_limit = self.config.candidate_pool_size
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(pid: str) -> None:
            if len(candidates) >= pool_limit:
                return
            if not pid or pid in seen or pid not in snapshot.active_ids or pid in excluded_ids:
                return
            seen.add(pid)
            candidates.append(pid)

        for seen_pid in reversed(recent_views):
            _add(seen_pid)
            for dst_pid, _ in snapshot.transitions.get(seen_pid, ()):
                _add(dst_pid)
            for new_pid, _ in snapshot.reverse_cold_neighbors.get(seen_pid, ()):
                _add(new_pid)
            for warm_nb, _ in snapshot.cold_neighbors.get(seen_pid, ()):
                _add(warm_nb)
                for dst_pid, _ in snapshot.transitions.get(warm_nb, ())[:6]:
                    _add(dst_pid)

        profile = None if is_new_user else snapshot.user_profiles.get(uid)
        if profile is not None:
            for pid, _ in profile.intent_items:
                _add(pid)

        for pid, _ in snapshot.ctx_3.get((region, device, page), ()):
            _add(pid)
        for pid, _ in snapshot.ctx_2.get((region, device), ())[:15]:
            _add(pid)
        for pid, _ in snapshot.ctx_1.get((page,), ())[:12]:
            _add(pid)

        # Ensure new Cold Start items from the active subcategory can enter UCB competition
        if snapshot.new_product_ids:
            anchor_subcat = ""
            if recent_views and recent_views[-1] in snapshot.products:
                anchor_subcat = snapshot.products[recent_views[-1]].subcategory
            added_new = 0
            for new_pid in snapshot.new_product_ids:
                if added_new >= 8 or len(candidates) >= pool_limit:
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
        snapshot: RecommenderSnapshotV3,
        k: int = 10,
    ) -> list[str]:
        """Score candidates with IPS ML Re-Ranker + PositionAwareBandit + Session-Safe MMR."""
        parsed = _parse_request_fields(request)
        uid, is_new_user, _, _, _, recent_views = parsed
        has_session_intent = bool(recent_views)

        profile = None if is_new_user else snapshot.user_profiles.get(uid)
        bought_hist = profile.bought_products if profile is not None else frozenset()
        bought_live = snapshot.live_user_bought.get(uid, frozenset())
        excluded_ids = bought_hist | bought_live

        candidates = self._gather_candidates(parsed, snapshot, excluded_ids)
        if not candidates:
            return ensure_valid_submission(
                (),
                snapshot.active_ids,
                snapshot.fallback_ranked,
                exclude_ids=excluded_ids,
                k=k,
            )

        overrides = snapshot.agent_overrides
        X = _extract_feature_matrix(snapshot, request, candidates, parsed=parsed)
        ranker: PositionDebiasedRanker = (
            snapshot.ranker if snapshot.ranker is not None else PositionDebiasedRanker.from_priors()
        )
        base_scores = ranker.score_matrix(
            X,
            blend_alpha_scale=overrides.ml_blend_alpha_scale,
            reverse_graft_override=overrides.weight_reverse_cold_graft,
            forward_cold_override=overrides.weight_forward_cold,
            revenue_tie_override=overrides.revenue_tie_weight,
        )

        ucb_alpha = overrides.bandit_ucb_alpha
        b_state = snapshot.bandit_state
        for idx, pid in enumerate(candidates):
            info = snapshot.products.get(pid)
            is_new = bool(info.is_new) if info is not None else False
            base_scores[idx] += self.bandit.score_adjustment(pid, is_new, b_state, ucb_alpha)

        # Session-Safe MMR selection across k slots
        if has_session_intent:
            mmr_start = self.config.mmr_start_slot_session
            subcat_lam = self.config.mmr_subcat_lambda_session
        else:
            mmr_start = self.config.mmr_start_slot_home
            subcat_lam = self.config.mmr_subcat_lambda_home
        brand_lam = overrides.mmr_brand_lambda

        n_cands = len(candidates)
        picked = np.zeros(n_cands, dtype=bool)
        subcat_counts: dict[str, int] = defaultdict(int)
        brand_counts: dict[str, int] = defaultdict(int)
        selected: list[str] = []

        cand_subcats = [
            snapshot.products[pid].subcategory if pid in snapshot.products else ""
            for pid in candidates
        ]
        cand_brands = [
            snapshot.products[pid].brand if pid in snapshot.products else ""
            for pid in candidates
        ]

        # Pre-sort indices by base score when MMR penalty is zero for O(k) pure slots
        if subcat_lam == 0.0 and brand_lam == 0.0:
            top_indices = heapq.nlargest(k, range(n_cands), key=lambda i: float(base_scores[i]))
            selected = [candidates[i] for i in top_indices]
        else:
            order = np.argsort(base_scores)[::-1]
            # First select pure slots before mmr_start
            pure_limit = min(mmr_start, k, n_cands)
            for slot_pos in range(pure_limit):
                idx = int(order[slot_pos])
                picked[idx] = True
                pid = candidates[idx]
                selected.append(pid)
                if cand_subcats[idx]:
                    subcat_counts[cand_subcats[idx]] += 1
                if cand_brands[idx]:
                    brand_counts[cand_brands[idx]] += 1

            # Greedy MMR for remaining slots
            while len(selected) < min(k, n_cands):
                best_idx = -1
                best_val = -1e18
                for idx in range(n_cands):
                    if picked[idx]:
                        continue
                    pen = 0.0
                    sc = cand_subcats[idx]
                    br = cand_brands[idx]
                    if sc and subcat_lam > 0.0:
                        pen += subcat_lam * subcat_counts[sc]
                    if br and brand_lam > 0.0:
                        pen += brand_lam * brand_counts[br]
                    val = float(base_scores[idx]) - pen
                    if val > best_val:
                        best_val = val
                        best_idx = idx
                if best_idx < 0:
                    break
                picked[best_idx] = True
                selected.append(candidates[best_idx])
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
