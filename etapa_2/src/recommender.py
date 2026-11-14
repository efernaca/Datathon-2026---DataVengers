"""Etapa 2 hybrid recommender: Lift transitions, deep user profiles, bidirectional cold start & MMR."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
import math
import time
from typing import Any

from .cold_start import ColdStartIndexV2
from .config import ConfigV2
from .guardrails import NUM_SLOTS, ensure_valid_submission
from .state import ProductInfo, RecommenderSnapshotV2, UserProfile


def _normalize_pairs(items: Iterable[tuple[str, float]], limit: int) -> tuple[tuple[str, float], ...]:
    sorted_items = sorted(items, key=lambda x: (-x[1], x[0]))[:limit]
    if not sorted_items:
        return ()
    top = max(sorted_items[0][1], 1e-9)
    return tuple((pid, float(score / top)) for pid, score in sorted_items if score > 0.0)


def _parse_iso_seconds(ts_raw: Any) -> float | None:
    if not ts_raw or not isinstance(ts_raw, str):
        return None
    try:
        return datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _extract_recent_views(recent_views: Any) -> list[tuple[str, float | None]]:
    """Extract (product_id, timestamp_sec_or_None) in chronological order."""
    if not recent_views:
        return []
    out: list[tuple[str, float | None]] = []
    for item in recent_views:
        if isinstance(item, Mapping):
            pid = item.get("product_id")
            if pid:
                out.append((str(pid), _parse_iso_seconds(item.get("ts"))))
        elif item:
            out.append((str(item), None))
    return out


def build_snapshot_v2_from_records(
    catalog_rows: Iterable[Mapping[str, Any]],
    *,
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
    lift_beta: float = 0.45,
    lift_prior_pop: float = 30.0,
    version: int = 1,
    max_cold_neighbors: int = 8,
    max_reverse_cold_per_warm: int = 3,
) -> RecommenderSnapshotV2:
    """Assemble RecommenderSnapshotV2 from RawTree queries or offline tables."""
    products: dict[str, ProductInfo] = {}
    subcat_prices: dict[str, list[float]] = defaultdict(list)

    for r in catalog_rows:
        pid = str(r["product_id"])
        price = float(r.get("price_eur", 0.0) or 0.0)
        init_price = float(r.get("initial_price_eur", price) or price)
        is_active = bool(r.get("is_active", True))
        subcat = str(r.get("subcategory", ""))
        products[pid] = ProductInfo(
            product_id=pid,
            category=str(r.get("category", "")),
            subcategory=subcat,
            brand=str(r.get("brand", "")),
            price_eur=price,
            initial_price_eur=init_price,
            description=str(r.get("description", "")),
            is_active=is_active,
            is_new=bool(r.get("is_new_product", r.get("is_new", False))),
        )
        if is_active and price > 0.0:
            subcat_prices[subcat].append(price)

    subcat_avg_price = {
        sc: float(sum(vals) / len(vals)) for sc, vals in subcat_prices.items() if vals
    }

    oos_set = frozenset(str(x) for x in out_of_stock_ids)
    active_ids = frozenset(
        pid for pid, info in products.items() if info.is_active and pid not in oos_set
    )

    raw_pop: dict[str, float] = {}
    raw_self_cvr: dict[str, float] = {}
    for r in popularity_rows:
        pid = str(r["product_id"])
        wpop = float(r.get("weighted_pop", r.get("score", r.get("n", 0.0))) or 0.0)
        raw_pop[pid] = wpop
        if "self_cvr" in r:
            raw_self_cvr[pid] = float(r["self_cvr"] or 0.0)
        else:
            views = float(r.get("views", 0.0) or 0.0)
            buys = float(r.get("purchases", 0.0) or 0.0)
            raw_self_cvr[pid] = (buys + 0.5) / (views + 25.0)

    max_pop = max(raw_pop.values(), default=1.0) or 1.0
    product_pop_score = {pid: s / max_pop for pid, s in raw_pop.items() if s > 0.0}

    max_scvr = max(raw_self_cvr.values(), default=1.0) or 1.0
    self_conversion = {pid: s / max_scvr for pid, s in raw_self_cvr.items() if s > 0.0}

    global_ranked = tuple(sorted(active_ids, key=lambda pid: (-raw_pop.get(pid, 0.0), pid)))

    subcat_buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for pid in active_ids:
        info = products[pid]
        subcat_buckets[info.subcategory].append((pid, product_pop_score.get(pid, 1e-4)))
    subcategory_ranked = {
        sc: _normalize_pairs(pairs, limit=25) for sc, pairs in subcat_buckets.items()
    }

    # 3-level Hierarchical Empirical Bayes Context Rankings
    lvl3_buckets: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    lvl2_buckets: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    lvl1_buckets: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for r in context_rows:
        reg = str(r.get("region", ""))
        dev = str(r.get("device", ""))
        page = str(r.get("page", ""))
        pid = str(r["product_id"])
        buys = float(r.get("buys", r.get("score", 0.0)) or 0.0)
        if buys > 0.0:
            lvl3_buckets[(reg, dev, page)][pid] += buys
            lvl2_buckets[(reg, "", page)][pid] += buys
            lvl1_buckets[("", "", page)][pid] += buys

    context_ranked: dict[tuple[str, str, str], tuple[tuple[str, float], ...]] = {}
    for ctx_key, pid_scores in lvl1_buckets.items():
        context_ranked[ctx_key] = _normalize_pairs(pid_scores.items(), limit=25)
    for ctx_key, pid_scores in lvl2_buckets.items():
        page_prior = dict(context_ranked.get(("", "", ctx_key[2]), ()))
        blended = dict(pid_scores)
        for pid, p_s in page_prior.items():
            blended[pid] = blended.get(pid, 0.0) + 3.0 * p_s
        context_ranked[ctx_key] = _normalize_pairs(blended.items(), limit=25)
    for ctx_key, pid_scores in lvl3_buckets.items():
        reg_prior = dict(context_ranked.get((ctx_key[0], "", ctx_key[2]), ()))
        blended = dict(pid_scores)
        for pid, r_s in reg_prior.items():
            blended[pid] = blended.get(pid, 0.0) + 4.0 * r_s
        context_ranked[ctx_key] = _normalize_pairs(blended.items(), limit=25)

    # Directed session transitions debiased via Bayesian-smoothed Lift: raw / (pop(dst) + prior)^beta
    trans_buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in transition_rows:
        src = str(r["src_product"])
        dst = str(r["dst_product"])
        raw_t = float(r.get("raw_transition_score", r.get("transition_score", 0.0)) or 0.0)
        if src != dst and raw_t > 0.0:
            dst_pop = max(raw_pop.get(dst, 0.0), 0.0)
            lift_score = raw_t / ((dst_pop + lift_prior_pop) ** lift_beta)
            trans_buckets[src].append((dst, lift_score))
    transitions = {
        src: _normalize_pairs(pairs, limit=35) for src, pairs in trans_buckets.items()
    }

    # Deep User Profiles
    bought_map: dict[str, frozenset[str]] = {}
    price_stats: dict[str, tuple[float, float]] = {}
    for r in user_profile_rows:
        uid = str(r["user_id"])
        bought_map[uid] = frozenset(str(p) for p in (r.get("bought_products") or ()))
        price_stats[uid] = (
            float(r.get("log_price_mean", 0.0) or 0.0),
            float(r.get("log_price_std", 0.0) or 0.0),
        )
    for r in user_bought_rows:
        uid = str(r["user_id"])
        if uid not in bought_map:
            bought_map[uid] = frozenset(str(p) for p in (r.get("bought_products") or ()))

    intent_buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in user_intent_rows:
        uid = str(r["user_id"])
        pid = str(r["product_id"])
        sc = float(r.get("intent_score", 0.0) or 0.0)
        if sc > 0.0 and pid not in bought_map.get(uid, frozenset()):
            intent_buckets[uid].append((pid, sc))

    subcat_aff_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    brand_aff_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in user_affinity_rows:
        uid = str(r["user_id"])
        aff = float(r.get("affinity", 0.0) or 0.0)
        if aff <= 0.0:
            continue
        sc = str(r.get("subcategory", ""))
        br = str(r.get("brand", ""))
        if sc:
            subcat_aff_buckets[uid][sc] += aff
        if br:
            brand_aff_buckets[uid][br] += aff

    all_uids = (
        set(bought_map) | set(intent_buckets) | set(subcat_aff_buckets) | set(brand_aff_buckets)
    )
    user_profiles: dict[str, UserProfile] = {}
    for uid in all_uids:
        lp_mean, lp_std = price_stats.get(uid, (0.0, 0.0))
        user_profiles[uid] = UserProfile(
            user_id=uid,
            bought=bought_map.get(uid, frozenset()),
            intent_items=_normalize_pairs(intent_buckets.get(uid, ()), limit=10),
            subcat_affinity=_normalize_pairs(subcat_aff_buckets.get(uid, {}).items(), limit=5),
            brand_affinity=_normalize_pairs(brand_aff_buckets.get(uid, {}).items(), limit=5),
            log_price_mean=lp_mean,
            log_price_std=lp_std,
        )

    raw_trends = {
        str(r["product_id"]): float(r.get("live_momentum", r.get("score", 0.0)) or 0.0)
        for r in live_trend_rows
    }
    max_trend = max(raw_trends.values(), default=1.0) or 1.0
    live_trends = {pid: v / max_trend for pid, v in raw_trends.items() if v > 0.0}

    live_user_bought = {
        str(r["user_id"]): frozenset(str(p) for p in (r.get("live_bought_products") or ()))
        for r in live_bought_rows
    }

    # Bidirectional Cold Start Index (TF-IDF + SVD)
    warm_ids = [
        pid
        for pid, info in products.items()
        if pid in active_ids
        and not info.is_new
        and (len(transitions.get(pid, ())) >= 2 or raw_pop.get(pid, 0.0) > 0.0)
    ]
    forward_targets = [
        info
        for pid, info in products.items()
        if info.is_new or len(transitions.get(pid, ())) < 3
    ]
    new_products = [info for info in products.values() if info.is_new]

    cold_neighbors: dict[str, tuple[tuple[str, float], ...]] = {}
    warm_to_cold: dict[str, tuple[tuple[str, float], ...]] = {}
    if products and (forward_targets or new_products):
        idx_v2 = ColdStartIndexV2().fit(products, warm_ids=warm_ids)
        cold_neighbors, warm_to_cold = idx_v2.compute_bidirectional(
            forward_targets=forward_targets,
            new_products=new_products,
            top_k_forward=max_cold_neighbors,
            top_k_reverse=max_reverse_cold_per_warm,
        )

    return RecommenderSnapshotV2(
        version=version,
        created_at=time.time(),
        products=products,
        active_ids=active_ids,
        out_of_stock=oos_set,
        global_ranked=global_ranked,
        product_pop_score=product_pop_score,
        self_conversion=self_conversion,
        subcat_avg_price=subcat_avg_price,
        context_ranked=context_ranked,
        subcategory_ranked=subcategory_ranked,
        transitions=transitions,
        cold_neighbors=cold_neighbors,
        warm_to_cold=warm_to_cold,
        user_profiles=user_profiles,
        live_trends=live_trends,
        live_user_bought=live_user_bought,
    )


class Stage2Recommender:
    """Sequential Lift + Deep User Profile + Bidirectional Cold Start + MMR Re-Ranker."""

    def __init__(self, config: ConfigV2 | None = None) -> None:
        self.config = config or ConfigV2()

    def _select_with_mmr(
        self,
        ranked_scores: list[tuple[str, float]],
        snapshot: RecommenderSnapshotV2,
        k: int,
        has_session_intent: bool,
    ) -> list[str]:
        """Keep top slots pure; diversify subcategories on cold/home entry and brands lightly."""
        cfg = self.config
        start_slot = 5 if has_session_intent else cfg.mmr_start_slot
        subcat_lambda = 0.0 if has_session_intent else cfg.mmr_lambda_subcat
        brand_lambda = cfg.mmr_lambda_brand

        if len(ranked_scores) <= start_slot:
            return [pid for pid, _ in ranked_scores[:k]]

        selected: list[str] = []
        subcat_counts: dict[str, int] = defaultdict(int)
        brand_counts: dict[str, int] = defaultdict(int)

        for pid, _ in ranked_scores[:start_slot]:
            selected.append(pid)
            info = snapshot.products.get(pid)
            if info is not None:
                subcat_counts[info.subcategory] += 1
                brand_counts[info.brand] += 1

        pool = list(ranked_scores[start_slot : start_slot + 40])
        while len(selected) < k and pool:
            best_idx = 0
            best_mmr = -1e18
            for idx, (pid, base_score) in enumerate(pool):
                info = snapshot.products.get(pid)
                penalty = 0.0
                if info is not None:
                    penalty = (
                        subcat_lambda * subcat_counts[info.subcategory]
                        + brand_lambda * brand_counts[info.brand]
                    )
                mmr_val = base_score - penalty
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = idx
            chosen_pid, _ = pool.pop(best_idx)
            selected.append(chosen_pid)
            info = snapshot.products.get(chosen_pid)
            if info is not None:
                subcat_counts[info.subcategory] += 1
                brand_counts[info.brand] += 1

        return selected

    def recommend(
        self,
        request: Mapping[str, Any] | Any,
        snapshot: RecommenderSnapshotV2,
        k: int = NUM_SLOTS,
    ) -> list[str]:
        """Score candidates in memory and return exactly k valid, diversified product IDs."""
        if isinstance(request, Mapping):
            user_id = str(request.get("user_id", "") or "")
            is_new_user = bool(request.get("is_new_user", False))
            context = request.get("context") or {}
            recent_views_raw = request.get("recent_views") or ()
            req_ts = _parse_iso_seconds(request.get("ts"))
        else:
            user_id = str(getattr(request, "user_id", "") or "")
            is_new_user = bool(getattr(request, "is_new_user", False))
            context = getattr(request, "context", None) or {}
            recent_views_raw = getattr(request, "recent_views", None) or ()
            req_ts = _parse_iso_seconds(getattr(request, "ts", None))

        if isinstance(context, Mapping):
            region = str(context.get("region", "") or "")
            device = str(context.get("device", "") or "")
            page = str(context.get("page", "") or "")
        else:
            region, device, page = "", "", ""

        profile = None if is_new_user else snapshot.user_profiles.get(user_id)
        already_bought = (
            (profile.bought if profile is not None else frozenset())
            | snapshot.live_user_bought.get(user_id, frozenset())
        )
        recent_views = _extract_recent_views(recent_views_raw)
        cfg = self.config

        scores: dict[str, float] = {}
        subcat_weights: dict[str, float] = {}

        # 1. Dual-decay session signals (most recent view first)
        for idx, (v_pid, v_ts) in enumerate(reversed(recent_views)):
            rank_w = cfg.recent_view_decay ** idx
            if req_ts is not None and v_ts is not None and req_ts >= v_ts:
                dt_sec = req_ts - v_ts
                time_w = math.exp(-dt_sec / cfg.time_decay_half_life_sec)
                w = rank_w * (0.5 + 0.5 * time_w)
            else:
                w = rank_w

            v_info = snapshot.products.get(v_pid)

            # Self-conversion re-engagement
            if v_pid in snapshot.active_ids and v_pid not in already_bought:
                scvr = snapshot.self_conversion.get(v_pid, 0.35)
                page_factor = 0.65 if (page == "product" and idx == 0) else 1.0
                scores[v_pid] = (
                    scores.get(v_pid, 0.0)
                    + cfg.weight_self_conversion * (0.4 + 0.6 * scvr) * page_factor * w
                )

            if v_info is not None and v_info.subcategory:
                subcat_weights[v_info.subcategory] = subcat_weights.get(v_info.subcategory, 0.0) + w

            # Lift-debiased directed transitions
            trans = snapshot.transitions.get(v_pid, ())
            for dst_pid, t_score in trans:
                scores[dst_pid] = scores.get(dst_pid, 0.0) + cfg.weight_coview_lift * w * t_score

            # Forward Cold-Start bridge (when viewing a new or sparse product)
            is_cold_item = (v_info is not None and v_info.is_new) or len(trans) < 3
            if is_cold_item:
                for nb_pid, sim in snapshot.cold_neighbors.get(v_pid, ()):
                    scores[nb_pid] = scores.get(nb_pid, 0.0) + cfg.weight_cold_neighbor * w * sim
                    for dst_pid, t_score in snapshot.transitions.get(nb_pid, ())[:6]:
                        if dst_pid != v_pid:
                            scores[dst_pid] = (
                                scores.get(dst_pid, 0.0) + cfg.weight_cold_graft * w * sim * t_score
                            )

            # Reverse Cold-Start grafting (promote matching new_products when viewing warm items)
            for new_pid, sim in snapshot.warm_to_cold.get(v_pid, ()):
                scores[new_pid] = scores.get(new_pid, 0.0) + cfg.weight_reverse_cold_graft * w * sim

        # 2. Session subcategory affinity
        if subcat_weights:
            top_subcats = sorted(subcat_weights.items(), key=lambda x: -x[1])[:2]
            max_sw = top_subcats[0][1] or 1.0
            for subcat, sw in top_subcats:
                norm_sw = sw / max_sw
                for pid, item_score in snapshot.subcategory_ranked.get(subcat, ())[:15]:
                    scores[pid] = (
                        scores.get(pid, 0.0) + cfg.weight_subcategory_pop * norm_sw * item_score
                    )

        # 3. Deep User Profile (cart/view intent recovery + subcategory/brand affinity)
        user_preferred_brands: dict[str, float] = {}
        if profile is not None:
            for pid, intent_w in profile.intent_items:
                scores[pid] = scores.get(pid, 0.0) + cfg.weight_intent_recovery * intent_w

            for subcat, u_aff in profile.subcat_affinity[:2]:
                for pid, item_score in snapshot.subcategory_ranked.get(subcat, ())[:12]:
                    scores[pid] = scores.get(pid, 0.0) + cfg.weight_user_subcat * u_aff * item_score

            user_preferred_brands = dict(profile.brand_affinity)

        # 4. 3-level Hierarchical Context Popularity
        ctx_list = (
            snapshot.context_ranked.get((region, device, page))
            or snapshot.context_ranked.get((region, "", page))
            or snapshot.context_ranked.get(("", "", page))
        )
        if ctx_list:
            for pid, ctx_score in ctx_list[:20]:
                scores[pid] = scores.get(pid, 0.0) + cfg.weight_context_pop * ctx_score

        # 5. Ensure fallback & live trending items enter the candidate pool
        for pid in snapshot.global_ranked[:20]:
            scores.setdefault(pid, 0.0)
        for pid in list(snapshot.live_trends)[:15]:
            if pid in snapshot.active_ids:
                scores.setdefault(pid, 0.0)

        # 6. Final candidate scoring: global pop, live trend, price drops, brand/budget fit & revenue tie
        has_price_profile = profile is not None and profile.log_price_mean > 0.0
        for pid in list(scores):
            if pid not in snapshot.active_ids or pid in already_bought:
                del scores[pid]
                continue
            val = scores[pid]
            val += cfg.weight_global_pop * snapshot.product_pop_score.get(pid, 0.0)
            val += cfg.weight_live_trend * snapshot.live_trends.get(pid, 0.0)

            info = snapshot.products.get(pid)
            if info is not None:
                if info.discount_pct > 0.0:
                    val += cfg.weight_price_drop * info.discount_pct
                if user_preferred_brands:
                    val += cfg.weight_user_brand * user_preferred_brands.get(info.brand, 0.0)
                if has_price_profile and info.price_eur > 0.0:
                    assert profile is not None
                    z = (math.log1p(info.price_eur) - profile.log_price_mean) / max(
                        profile.log_price_std, 0.35
                    )
                    val += cfg.weight_user_price_fit * math.exp(-0.5 * z * z)
                avg_sub_p = snapshot.subcat_avg_price.get(info.subcategory, 0.0)
                if avg_sub_p > 0.0 and cfg.revenue_tie_alpha > 0.0:
                    rel_price = min(info.price_eur / avg_sub_p, 2.0)
                    val *= 1.0 + cfg.revenue_tie_alpha * rel_price

            scores[pid] = val

        ranked_pairs = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], -snapshot.product_pop_score.get(kv[0], 0.0), kv[0]),
        )
        diversified = self._select_with_mmr(
            ranked_pairs,
            snapshot,
            k=k,
            has_session_intent=bool(recent_views),
        )

        return ensure_valid_submission(
            candidates=diversified,
            active_ids=snapshot.active_ids,
            fallback_ranked=snapshot.global_ranked,
            exclude_ids=already_bought,
            k=k,
        )
