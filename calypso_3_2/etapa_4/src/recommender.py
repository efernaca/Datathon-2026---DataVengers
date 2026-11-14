"""Slot-Gated Calypso-3.2 recommender (Stage 2 Especial N6-N13 + EASE + IPS/Bandit/Agent)."""
from __future__ import annotations

from datetime import datetime
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .agent_scratch import CalypsoAgent
from .bandits import PositionAwareBandit
from .cold_start import ColdStartIndexV4
from .config import ConfigV4
from .ease import fit_ease_from_transitions
from .guardrails import ensure_valid_submission
from .reranker import NUM_FEATURES, PositionDebiasedRanker, build_ips_training_dataset
from .state import ProductInfo, RecommenderSnapshotV4, UserProfile


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        res = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(res) else res


def _parse_iso_seconds(ts_raw: Any) -> float | None:
    if ts_raw is None or ts_raw == "":
        return None
    if isinstance(ts_raw, (int, float)):
        return float(ts_raw)
    if not isinstance(ts_raw, str):
        return None
    try:
        return datetime.fromisoformat(ts_raw.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


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


def _extract_recent_with_ts(raw_recent: Any) -> list[tuple[str, float | None]]:
    """Parse recent_views keeping optional event timestamps for N13 half-life decay."""
    out: list[tuple[str, float | None]] = []
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
                out.append((str(pid).strip(), _parse_iso_seconds(item.get("ts"))))
        elif item:
            s_pid = str(item).strip()
            if s_pid:
                out.append((s_pid, None))
    return out


def _parse_request_fields_full(
    request: Any,
) -> tuple[str, bool, str, str, str, list[tuple[str, float | None]], float | None]:
    """Extract request fields plus timestamps for N13 real half-life weighting."""
    if isinstance(request, Mapping):
        user_id = str(request.get("user_id") or "").strip()
        is_new_user = bool(request.get("is_new_user", False))
        ctx = request.get("context")
        raw_recent = request.get("recent_views")
        req_ts = _parse_iso_seconds(request.get("ts"))
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
        req_ts = _parse_iso_seconds(getattr(request, "ts", None))
        if isinstance(ctx, Mapping):
            region = str(ctx.get("region") or "").strip()
            device = str(ctx.get("device") or "").strip()
            page = str(ctx.get("page") or "").strip()
        else:
            region = str(getattr(request, "region", "") or "").strip()
            device = str(getattr(request, "device", "") or "").strip()
            page = str(getattr(request, "page", "") or "").strip()

    return user_id, is_new_user, region, device, page, _extract_recent_with_ts(raw_recent), req_ts


def _parse_request_fields(request: Any) -> tuple[str, bool, str, str, str, list[str]]:
    """Extract (user_id, is_new_user, region, device, page, recent_views) from dict or row."""
    uid, is_new, reg, dev, pg, rv_ts, _ = _parse_request_fields_full(request)
    return uid, is_new, reg, dev, pg, [pid for pid, _ in rv_ts]


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
            initial_price_eur=max(_safe_float(row.get("initial_price_eur"), price), price),
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

        init_price = (
            existing.initial_price_eur
            if (existing is not None and existing.initial_price_eur > 0.0)
            else price
        )
        products[pid] = ProductInfo(
            product_id=pid,
            category=str(row.get("category") or (existing.category if existing else "")).strip(),
            subcategory=str(row.get("subcategory") or (existing.subcategory if existing else "")).strip(),
            brand=str(row.get("brand") or (existing.brand if existing else "")).strip(),
            price_eur=price,
            description=str(row.get("description") or (existing.description if existing else "")).strip(),
            is_active=is_active,
            is_new=bool(row.get("is_new_product", existing.is_new if existing else True)),
            initial_price_eur=init_price,
        )

    active_ids = frozenset(pid for pid, p in products.items() if p.is_active)
    new_product_ids = frozenset(pid for pid, p in products.items() if p.is_active and p.is_new)
    return products, active_ids, new_product_ids, max_log_price


def _build_contextual_maps(
    context_rows: Iterable[Mapping[str, Any]],
    active_ids: frozenset[str],
    *,
    prior_strength: float = 50.0,
    top_k: int = 25,
) -> tuple[
    dict[tuple[str, str, str], tuple[tuple[str, float], ...]],
    dict[tuple[str, str], tuple[tuple[str, float], ...]],
    dict[tuple[str], tuple[tuple[str, float], ...]],
    dict[str, float],
]:
    """N12: 3-level Hierarchical Empirical Bayes context rates: global -> (page,) -> (reg, dev) -> (reg, dev, page)."""
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

    def _compute_eb(
        counts: Mapping[str, float],
        n_bucket: float,
        parent_rates: Mapping[str, float],
    ) -> tuple[dict[str, float], tuple[tuple[str, float], ...]]:
        denom = n_bucket + prior_strength
        if denom <= 0.0:
            return {}, ()
        rates: dict[str, float] = {}
        scored: list[tuple[float, str]] = []
        keys = set(counts) | set(parent_rates)
        for pid in keys:
            r = (counts.get(pid, 0.0) + prior_strength * parent_rates.get(pid, 0.0)) / denom
            if r > 0.0:
                rates[pid] = r
                scored.append((r, pid))
        scored.sort(reverse=True)
        top = scored[:top_k]
        if not top or top[0][0] <= 0.0:
            return rates, ()
        max_r = top[0][0]
        return rates, tuple((pid, r / max_r) for r, pid in top)

    rate_1: dict[tuple[str], dict[str, float]] = {}
    ctx_1: dict[tuple[str], tuple[tuple[str, float], ...]] = {}
    for key, counts in cnt_1.items():
        r1, ranked = _compute_eb(counts, tot_1[key], global_share)
        rate_1[key] = r1
        if ranked:
            ctx_1[key] = ranked

    rate_2: dict[tuple[str, str], dict[str, float]] = {}
    ctx_2: dict[tuple[str, str], tuple[tuple[str, float], ...]] = {}
    for key, counts in cnt_2.items():
        r2, ranked = _compute_eb(counts, tot_2[key], global_share)
        rate_2[key] = r2
        if ranked:
            ctx_2[key] = ranked

    ctx_3: dict[tuple[str, str, str], tuple[tuple[str, float], ...]] = {}
    for key, counts in cnt_3.items():
        parent = rate_2.get((key[0], key[1])) or rate_1.get((key[2],)) or global_share
        _, ranked = _compute_eb(counts, tot_3[key], parent)
        if ranked:
            ctx_3[key] = ranked

    return ctx_3, ctx_2, ctx_1, global_share


def build_product_static_table(
    products: Mapping[str, ProductInfo],
    active_ids: frozenset[str],
    global_pop_norm: Mapping[str, float],
    live_trend_norm: Mapping[str, float],
    max_log_price: float,
    bandit: PositionAwareBandit,
    bandit_state: Any,
    ucb_alpha: float,
) -> dict[str, tuple[str, str, float, float, float, float, float, float, float]]:
    """Precompute static per-active-product FastLoop tuples (including V6 discount_pct)."""
    max_lp = max(max_log_price, 1e-6)
    table: dict[str, tuple[str, str, float, float, float, float, float, float, float]] = {}
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
            info.discount_pct,
        )
    return table


def _prepare_request_signals(
    snapshot: RecommenderSnapshotV4,
    parsed: tuple[str, bool, str, str, str, list[str]],
    *,
    recent_with_ts: Sequence[tuple[str, float | None]] | None = None,
    req_ts: float | None = None,
    recency_decay: float = 0.72,
    half_life_sec: float = 624.0,
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
    dict[str, float],
]:
    """Compute request-level lookup maps shared identically by training and FastLoop (N13)."""
    user_id, is_new_user, region, device, page, recent_views = parsed

    seq_scores: dict[str, float] = {}
    fwd_cold_scores: dict[str, float] = {}
    rev_graft_scores: dict[str, float] = {}
    self_view_scores: dict[str, float] = {}

    rv_items: list[tuple[str, float | None]] = (
        list(recent_with_ts)
        if recent_with_ts is not None
        else [(pid, None) for pid in recent_views]
    )
    ln2 = 0.6931471805599453
    hl = max(half_life_sec, 1.0)
    for rev_idx, (seen_pid, seen_ts) in enumerate(reversed(rv_items)):
        rank_w = recency_decay**rev_idx
        if req_ts is not None and seen_ts is not None and req_ts >= seen_ts:
            dt_sec = req_ts - seen_ts
            w_rec = rank_w * (0.5 + 0.5 * math.exp(-ln2 * dt_sec / hl))
        else:
            w_rec = rank_w
        if seen_pid in snapshot.active_ids and seen_pid not in self_view_scores:
            page_factor = 0.65 if (page == "product" and rev_idx == 0) else 1.0
            self_view_scores[seen_pid] = snapshot.self_cvr.get(seen_pid, 0.02) * w_rec * page_factor

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
            std_lp = max(profile.log_price_std, 0.45)

    anchor_subcat = ""
    anchor_brand = ""
    if recent_views:
        last_info = snapshot.products.get(recent_views[-1])
        if last_info is not None:
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
        {},
    )


def _extract_feature_matrix(
    snapshot: RecommenderSnapshotV4,
    request: Mapping[str, Any] | Any,
    candidate_ids: Sequence[str],
    parsed: tuple[str, bool, str, str, str, list[str]] | None = None,
) -> np.ndarray:
    """Vectorized 14-feature builder used cleanly for IPS Re-Ranker training."""
    n = len(candidate_ids)
    if n == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float64)

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
    ) = _prepare_request_signals(snapshot, p)

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
            sc, br, gpop, ltrend, is_new_f, lp_norm, _, _, _ = st
            aff_val = affinity_map.get((sc, br), 0.0)
            if aff_val == 0.0 and affinity_map:
                aff_val = affinity_map.get((sc, ""), 0.0) * 0.6
            X[idx, 3] = aff_val

            if has_price_hist and lp_norm > 0.0:
                lp = lp_norm * max_lp
                z = (lp - mean_lp) / std_lp
                z2 = z * z
                X[idx, 4] = math.exp(-0.5 * (z2 if z2 < 6.0 else 6.0))
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
) -> tuple[RecommenderSnapshotV4, list[dict[str, Any]]]:
    """Construct complete RecommenderSnapshotV4 with precomputed FastLoop vectors."""
    cfg = config or ConfigV4()
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
    for row in pop_rows_list:
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

    raw_live: dict[str, float] = {}
    for row in live_trend_rows:
        pid = str(row.get("product_id") or "").strip()
        if pid in active_ids:
            sc = max(_safe_float(row.get("live_score"), 0.0), 0.0)
            if sc > 0.0:
                raw_live[pid] = math.log1p(sc)
    max_live = max(raw_live.values(), default=1.0) or 1.0
    live_trend_norm = {pid: v / max_live for pid, v in raw_live.items()}

    raw_trans: dict[str, list[tuple[str, float]]] = defaultdict(list)
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
        raw_trans[src].append((dst, lift_score))

    ease_neighbors = fit_ease_from_transitions(
        pop_rows_list,
        trans_rows_list,
        active_ids,
        lam=250.0,
        top_k=35,
    )

    transitions: dict[str, tuple[tuple[str, float], ...]] = {}
    for src in set(raw_trans) | set(ease_neighbors):
        lift_edges = sorted(raw_trans.get(src, ()), key=lambda kv: kv[1], reverse=True)[:35]
        max_l = lift_edges[0][1] if lift_edges else 0.0
        lift_norm = {dst: (sc / max_l) for dst, sc in lift_edges} if max_l > 0.0 else {}
        ease_norm = dict(ease_neighbors.get(src, ()))

        combined = [
            (dst, 0.75 * lift_norm.get(dst, 0.0) + 0.25 * ease_norm.get(dst, 0.0))
            for dst in (set(lift_norm) | set(ease_norm))
        ]
        combined.sort(key=lambda kv: kv[1], reverse=True)
        top_combined = combined[:35]
        max_c = top_combined[0][1] if top_combined else 0.0
        if max_c > 0.0:
            transitions[src] = tuple((dst, sc / max_c) for dst, sc in top_combined if sc > 0.0)

    ctx_3, ctx_2, ctx_1, _ = _build_contextual_maps(
        context_rows,
        active_ids,
        prior_strength=cfg.ctx_kappa,
    )

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

    cs_index = ColdStartIndexV4()
    cs_index.fit(products, warm_ids)
    cold_neighbors, reverse_cold_neighbors = cs_index.compute_bidirectional(
        sparse_or_cold_targets,
        new_infos,
        top_k_forward=cfg.max_cold_neighbors,
        top_k_reverse=cfg.max_reverse_cold_per_warm,
        min_reverse_sim=cfg.reverse_min_sim,
        reverse_same_subcat=cfg.reverse_same_subcat,
    )

    fallback_scores: dict[str, float] = {}
    for pid in active_ids:
        p_obj = products[pid]
        pop_s = global_pop_norm.get(pid, 0.0) + 0.6 * live_trend_norm.get(pid, 0.0)
        lp_s = math.log1p(max(p_obj.price_eur, 0.0)) / max_log_price
        fallback_scores[pid] = pop_s + cfg.revenue_tie_weight * lp_s + (0.02 if p_obj.is_new else 0.0)

    fallback_ranked = tuple(
        sorted(active_ids, key=lambda pid: (fallback_scores[pid], pid), reverse=True)
    )
    revenue_fallback = tuple(
        sorted(active_ids, key=lambda pid: (products[pid].price_eur, pid), reverse=True)
    )

    fb_list = list(feedback_rows)
    hist_list = list(training_session_rows)
    bandit = PositionAwareBandit(
        exam_power=cfg.bandit_exam_power,
        prior_strength=cfg.bandit_prior_strength,
        cold_ucb_multiplier=cfg.bandit_cold_ucb_multiplier,
    )
    bandit_source = fb_list if fb_list else hist_list
    bandit_state = bandit.fit_from_feedback(bandit_source, fallback_cvr=self_cvr)

    ranker = PositionDebiasedRanker.from_priors()
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
        cfg.bandit_ucb_alpha,
    )

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

    if overrides.bandit_ucb_alpha != cfg.bandit_ucb_alpha:
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
        """Collect bounded multi-channel candidate pool of <= candidate_pool_size items."""
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
                for dst_pid, _ in snapshot.transitions.get(warm_nb, ())[:5]:
                    _add(dst_pid)

        profile = None if is_new_user else snapshot.user_profiles.get(uid)
        if profile is not None:
            for pid, _ in profile.intent_items:
                _add(pid)

        for pid, _ in snapshot.ctx_3.get((region, device, page), ())[:18]:
            _add(pid)
        for pid, _ in snapshot.ctx_2.get((region, device), ())[:12]:
            _add(pid)
        for pid, _ in snapshot.ctx_1.get((page,), ())[:10]:
            _add(pid)

        if snapshot.new_product_ids:
            anchor_subcat = ""
            if recent_views and recent_views[-1] in snapshot.products:
                anchor_subcat = snapshot.products[recent_views[-1]].subcategory
            added_new = 0
            for new_pid in snapshot.new_product_ids:
                if added_new >= 4 or len(candidates) >= pool_limit:
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
    ) -> list[str]:
        """Slot-Gated scoring: Stage 2 Especial N6-N13 + segment weights + relative MMR + N9 tie-break."""
        uid, is_new_user, region, device, page, recent_with_ts, req_ts = _parse_request_fields_full(
            request
        )
        recent_views = [pid for pid, _ in recent_with_ts]
        parsed = (uid, is_new_user, region, device, page, recent_views)
        has_session_intent = bool(recent_views)
        has_new_in_session = any(pid in snapshot.new_product_ids for pid in recent_views)
        is_cold_session = is_new_user or has_new_in_session

        # 4.3 Segment-aware multipliers from Stage 2 Especial
        if has_new_in_session:
            seg = "producto_nuevo"
        elif is_new_user:
            seg = "nuevo_sesion" if has_session_intent else "nuevo_sin_sesion"
        else:
            seg = "conocido_sesion" if has_session_intent else "conocido_sin_sesion"

        seq_scale = 1.06 if seg == "nuevo_sesion" else 1.0
        intent_scale = 1.15 if seg == "conocido_sin_sesion" else 1.0
        ctx_scale = 1.25 if seg == "nuevo_sin_sesion" else (1.10 if seg == "nuevo_sesion" else 1.0)
        pop_scale = 1.15 if seg == "nuevo_sin_sesion" else 1.0
        cold_scale = 1.25 if seg == "producto_nuevo" else 1.0

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
            parsed,
            recent_with_ts=recent_with_ts,
            req_ts=req_ts,
            recency_decay=self.config.recency_decay,
            half_life_sec=self.config.recent_halflife_sec,
        )

        overrides = snapshot.agent_overrides
        ranker: PositionDebiasedRanker = (
            snapshot.ranker if snapshot.ranker is not None else PositionDebiasedRanker.from_priors()
        )
        w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13 = (
            ranker.resolve_weights_tuple(
                blend_alpha_scale=overrides.ml_blend_alpha_scale,
                reverse_graft_override=overrides.weight_reverse_cold_graft,
                forward_cold_override=overrides.weight_forward_cold,
                revenue_tie_override=overrides.revenue_tie_weight,
            )
        )

        max_lp = max(snapshot.max_log_price, 1e-6)
        product_static = snapshot.product_static
        has_aff = bool(affinity_map)
        w_price_drop = self.config.weight_price_drop

        # Compute split (precision_score, explore_score, subcat, brand, is_new, confirmed_cold)
        n_cands = len(candidates)
        prec_scores = [0.0] * n_cands
        full_scores = [0.0] * n_cands
        cand_subcats = [""] * n_cands
        cand_brands = [""] * n_cands
        cand_is_new = [False] * n_cands
        cand_confirmed_cold = [False] * n_cands

        subcat_bonus_w = w12
        brand_bonus_w = w13

        for idx, pid in enumerate(candidates):
            s_seq = seq_scores.get(pid, 0.0)
            if s_seq > 1.8:
                s_seq = 1.8
            s_self = self_view_scores.get(pid, 0.0) * 5.0
            if s_self > 1.2:
                s_self = 1.2
            s_intent = intent_scores.get(pid, 0.0)
            s_ctx = ctx_scores.get(pid, 0.0)
            s_rev = rev_graft_scores.get(pid, 0.0)
            s_fwd = fwd_cold_scores.get(pid, 0.0)

            st = product_static.get(pid)
            if st is not None:
                sc, br, gpop, ltrend, is_new_f, lp_norm, exploit_d, ucb_b, disc_pct = st
                s_aff = 0.0
                if has_aff:
                    s_aff = affinity_map.get((sc, br), 0.0)
                    if s_aff == 0.0:
                        s_aff = affinity_map.get((sc, ""), 0.0) * 0.6

                s_pcompat = 0.0
                if has_price_hist and lp_norm > 0.0:
                    z = (lp_norm * max_lp - mean_lp) / std_lp
                    z2 = z * z
                    s_pcompat = math.exp(-0.5 * (z2 if z2 < 6.0 else 6.0))

                same_sc = bool(anchor_subcat and sc == anchor_subcat)
                same_br = bool(anchor_brand and br == anchor_brand)
                bg_score = (
                    w5 * ctx_scale * s_ctx
                    + w6 * pop_scale * gpop
                    + w7 * ltrend
                    + exploit_d
                    + w_price_drop * disc_pct
                )
                if (
                    has_session_intent
                    and s_seq == 0.0
                    and s_intent == 0.0
                    and s_fwd == 0.0
                    and s_self == 0.0
                ):
                    bg_score *= 0.18 if not same_sc else 0.45

                score_p = (
                    w0 * seq_scale * s_seq
                    + w1 * s_self
                    + w2 * intent_scale * s_intent
                    + w3 * intent_scale * s_aff
                    + w4 * s_pcompat
                    + bg_score
                    + w8 * is_new_f
                    + w9 * cold_scale * s_rev
                    + w10 * cold_scale * s_fwd
                    + w11 * lp_norm
                    + (subcat_bonus_w if same_sc else 0.0)
                    + (brand_bonus_w if same_br else 0.0)
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
                gpop = snapshot.global_pop_norm.get(pid, 0.0)
                ltrend = snapshot.live_trend_norm.get(pid, 0.0)
                score_p = (
                    w0 * seq_scale * s_seq
                    + w1 * s_self
                    + w2 * intent_scale * s_intent
                    + w5 * ctx_scale * s_ctx
                    + w6 * pop_scale * gpop
                    + w7 * ltrend
                    + w9 * cold_scale * s_rev
                    + w10 * cold_scale * s_fwd
                )
                prec_scores[idx] = score_p
                full_scores[idx] = score_p

        if has_session_intent:
            mmr_start = self.config.mmr_start_slot_session
            subcat_lam = self.config.mmr_subcat_lambda_session
        else:
            mmr_start = self.config.mmr_start_slot_home
            subcat_lam = self.config.mmr_subcat_lambda_home
        brand_lam = overrides.mmr_brand_lambda

        picked = [False] * n_cands
        subcat_counts: dict[str, int] = defaultdict(int)
        brand_counts: dict[str, int] = defaultdict(int)
        selected_indices: list[int] = []

        cold_gate = 0 if is_cold_session else self.config.cold_explore_start_slot_warm
        max_cold_warm = self.config.max_cold_explore_in_warm
        new_picked_count = 0

        target_k = min(k, n_cands)
        while len(selected_indices) < target_k:
            slot_pos = len(selected_indices)
            allow_ucb = is_cold_session or slot_pos >= cold_gate
            allow_new = is_cold_session or (allow_ucb and new_picked_count < max_cold_warm)
            apply_mmr = slot_pos >= mmr_start

            best_idx = -1
            best_val = -1e18
            for idx in range(n_cands):
                if picked[idx]:
                    continue
                is_n = cand_is_new[idx]
                confirmed_cold = cand_confirmed_cold[idx]
                if (
                    is_n
                    and not allow_new
                    and not (slot_pos >= 4 and confirmed_cold and new_picked_count < max_cold_warm)
                ):
                    continue
                val = full_scores[idx] if (allow_ucb or confirmed_cold) else prec_scores[idx]
                if apply_mmr:
                    sc = cand_subcats[idx]
                    br = cand_brands[idx]
                    if val > 0.0:
                        mult = 1.0
                        if sc and subcat_lam > 0.0:
                            mult *= max(0.0, 1.0 - subcat_lam * subcat_counts[sc])
                        if br and brand_lam > 0.0:
                            mult *= max(0.0, 1.0 - brand_lam * brand_counts[br])
                        val *= mult
                    else:
                        if sc and subcat_lam > 0.0:
                            val -= subcat_lam * subcat_counts[sc]
                        if br and brand_lam > 0.0:
                            val -= brand_lam * brand_counts[br]
                if val > best_val:
                    best_val = val
                    best_idx = idx

            if best_idx < 0:
                for idx in range(n_cands):
                    if not picked[idx] and full_scores[idx] > best_val:
                        best_val = full_scores[idx]
                        best_idx = idx
                if best_idx < 0:
                    break

            picked[best_idx] = True
            selected_indices.append(best_idx)
            if cand_is_new[best_idx]:
                new_picked_count += 1
            if cand_subcats[best_idx]:
                subcat_counts[cand_subcats[best_idx]] += 1
            if cand_brands[best_idx]:
                brand_counts[cand_brands[best_idx]] += 1

        # N9: Lexicographic revenue tie-break within revenue_eps without breaking slot gating
        eps = self.config.revenue_eps
        if eps > 0.0 and len(selected_indices) >= 2:
            for _ in range(2):
                for a in range(len(selected_indices) - 1):
                    ia, ib = selected_indices[a], selected_indices[a + 1]
                    if cand_is_new[ib] and not is_cold_session and a < cold_gate:
                        if not (a >= 4 and cand_confirmed_cold[ib]):
                            continue
                    sa, sb = prec_scores[ia], prec_scores[ib]
                    if abs(sa - sb) < eps * max(abs(sa), 1e-9):
                        pa_info = snapshot.products.get(candidates[ia])
                        pb_info = snapshot.products.get(candidates[ib])
                        pa_p = pa_info.price_eur if pa_info is not None else 0.0
                        pb_p = pb_info.price_eur if pb_info is not None else 0.0
                        if pb_p > pa_p:
                            selected_indices[a], selected_indices[a + 1] = ib, ia

        selected = [candidates[i] for i in selected_indices]
        return ensure_valid_submission(
            selected,
            snapshot.active_ids,
            snapshot.fallback_ranked,
            exclude_ids=excluded_ids,
            k=k,
        )
