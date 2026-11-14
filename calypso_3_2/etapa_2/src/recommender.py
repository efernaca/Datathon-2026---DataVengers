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
from .ease import fit_ease_from_transitions
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
    ctx_kappa: float = 50.0,
    version: int = 1,
    max_cold_neighbors: int = 8,
    max_reverse_cold_per_warm: int = 3,
    reverse_min_sim: float = 0.45,
    reverse_same_subcat: bool = True,
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

    pop_rows_list = list(popularity_rows)
    trans_rows_list = list(transition_rows)

    raw_pop: dict[str, float] = {}
    raw_self_cvr: dict[str, float] = {}
    for r in pop_rows_list:
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
        sc: _normalize_pairs(pairs, limit=40) for sc, pairs in subcat_buckets.items()
    }

    # N12: 3-level Hierarchical Empirical Bayes Context Rankings
    # rate_1 = (n_1 + kappa * global_pop) / (N_1 + kappa)
    # rate_2 = (n_2 + kappa * rate_1) / (N_2 + kappa)
    # rate_3 = (n_3 + kappa * rate_2) / (N_3 + kappa)
    total_pop_mass = sum(max(raw_pop.get(pid, 0.0), 0.0) for pid in active_ids)
    n_act = max(len(active_ids), 1)
    pop_share = {
        pid: (max(raw_pop.get(pid, 0.0), 0.0) / total_pop_mass if total_pop_mass > 0.0 else 1.0 / n_act)
        for pid in active_ids
    }

    lvl3_buckets: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    lvl2_buckets: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    lvl1_buckets: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for r in context_rows:
        reg = str(r.get("region", ""))
        dev = str(r.get("device", ""))
        page = str(r.get("page", ""))
        pid = str(r["product_id"])
        buys = float(r.get("buys", r.get("score", 0.0)) or 0.0)
        if buys > 0.0 and pid in active_ids:
            lvl3_buckets[(reg, dev, page)][pid] += buys
            lvl2_buckets[(reg, "", page)][pid] += buys
            lvl1_buckets[("", "", page)][pid] += buys

    def _eb_rates(
        counts: Mapping[str, float],
        prior_map: Mapping[str, float],
        kappa: float,
    ) -> dict[str, float]:
        tot_n = sum(counts.values())
        denom = tot_n + kappa
        keys = set(counts) | set(prior_map)
        return {
            pid: (counts.get(pid, 0.0) + kappa * prior_map.get(pid, 0.0)) / max(denom, 1e-9)
            for pid in keys
        }

    rate1_maps: dict[tuple[str, str, str], dict[str, float]] = {}
    rate2_maps: dict[tuple[str, str, str], dict[str, float]] = {}
    context_ranked: dict[tuple[str, str, str], tuple[tuple[str, float], ...]] = {}

    for ctx_key, pid_counts in lvl1_buckets.items():
        r1 = _eb_rates(pid_counts, pop_share, ctx_kappa)
        rate1_maps[ctx_key] = r1
        context_ranked[ctx_key] = _normalize_pairs(r1.items(), limit=25)
    for ctx_key, pid_counts in lvl2_buckets.items():
        parent = rate1_maps.get(("", "", ctx_key[2]), pop_share)
        r2 = _eb_rates(pid_counts, parent, ctx_kappa)
        rate2_maps[ctx_key] = r2
        context_ranked[ctx_key] = _normalize_pairs(r2.items(), limit=25)
    for ctx_key, pid_counts in lvl3_buckets.items():
        parent = rate2_maps.get((ctx_key[0], "", ctx_key[2]), pop_share)
        r3 = _eb_rates(pid_counts, parent, ctx_kappa)
        context_ranked[ctx_key] = _normalize_pairs(r3.items(), limit=25)

    # Directed session transitions debiased via Bayesian-smoothed Lift + closed-form EASE
    trans_buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in trans_rows_list:
        src = str(r["src_product"])
        dst = str(r["dst_product"])
        raw_t = float(r.get("raw_transition_score", r.get("transition_score", 0.0)) or 0.0)
        if src != dst and raw_t > 0.0:
            dst_pop = max(raw_pop.get(dst, 0.0), 0.0)
            lift_score = raw_t / ((dst_pop + lift_prior_pop) ** lift_beta)
            trans_buckets[src].append((dst, lift_score))

    ease_neighbors = fit_ease_from_transitions(
        pop_rows_list,
        trans_rows_list,
        active_ids,
        lam=120.0,
        top_k=35,
    )
    transitions: dict[str, tuple[tuple[str, float], ...]] = {}
    for src in set(trans_buckets) | set(ease_neighbors):
        lift_norm = dict(_normalize_pairs(trans_buckets.get(src, ()), limit=35))
        ease_norm = dict(ease_neighbors.get(src, ()))
        blended = [
            (dst, 0.75 * lift_norm.get(dst, 0.0) + 0.25 * ease_norm.get(dst, 0.0))
            for dst in (set(lift_norm) | set(ease_norm))
        ]
        transitions[src] = _normalize_pairs(blended, limit=35)

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
            min_reverse_sim=reverse_min_sim,
            reverse_same_subcat=reverse_same_subcat,
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
        """N10: Keep top slots pure; apply relative multiplicative MMR diversification."""
        cfg = self.config
        start_slot = getattr(cfg, "mmr_start_slot_session", 5) if has_session_intent else cfg.mmr_start_slot
        subcat_lambda = 0.0 if has_session_intent else cfg.mmr_lambda_subcat
        brand_lambda = min(cfg.mmr_lambda_brand, 0.02) if has_session_intent else cfg.mmr_lambda_brand

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

        pool = list(ranked_scores[start_slot : start_slot + 50])
        while len(selected) < k and pool:
            best_idx = 0
            best_mmr = -1e18
            for idx, (pid, base_score) in enumerate(pool):
                info = snapshot.products.get(pid)
                if info is not None and base_score > 0.0:
                    mult = (
                        max(0.0, 1.0 - subcat_lambda * subcat_counts[info.subcategory])
                        * max(0.0, 1.0 - brand_lambda * brand_counts[info.brand])
                    )
                    mmr_val = base_score * mult
                else:
                    penalty = (
                        (subcat_lambda * subcat_counts[info.subcategory] + brand_lambda * brand_counts[info.brand])
                        if info is not None
                        else 0.0
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

    @staticmethod
    def _revenue_tiebreak(
        order: list[str],
        scores: Mapping[str, float],
        products: Mapping[str, ProductInfo],
        eps: float,
    ) -> list[str]:
        """N9: Lexicographic revenue tie-break only within |Δscore| < eps * |score|."""
        if eps <= 0.0 or len(order) < 2:
            return order
        out = list(order)
        for _ in range(2):
            for a in range(len(out) - 1):
                pa, pb = out[a], out[a + 1]
                sa = scores.get(pa, 0.0)
                sb = scores.get(pb, 0.0)
                info_a = products.get(pa)
                info_b = products.get(pb)
                price_a = info_a.price_eur if info_a is not None else 0.0
                price_b = info_b.price_eur if info_b is not None else 0.0
                if abs(sa - sb) < eps * max(abs(sa), 1e-9) and price_b > price_a:
                    out[a], out[a + 1] = pb, pa
        return out

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
        session_brands: dict[str, float] = {}

        # 1. Dual-decay session signals (most recent view first)
        for idx, (v_pid, v_ts) in enumerate(reversed(recent_views)):
            rank_w = cfg.recent_view_decay ** idx
            if req_ts is not None and v_ts is not None and req_ts >= v_ts:
                dt_sec = req_ts - v_ts
                time_w = math.exp(-math.log(2.0) * dt_sec / cfg.time_decay_half_life_sec)
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
            if v_info is not None and v_info.brand:
                session_brands[v_info.brand] = session_brands.get(v_info.brand, 0.0) + w

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

        # 2. Session subcategory affinity (+ same-brand boost)
        if subcat_weights:
            top_subcats = sorted(subcat_weights.items(), key=lambda x: -x[1])[:2]
            max_sw = top_subcats[0][1] or 1.0
            for subcat, sw in top_subcats:
                norm_sw = sw / max_sw
                for pid, item_score in snapshot.subcategory_ranked.get(subcat, ())[:30]:
                    s_info = snapshot.products.get(pid)
                    brand_boost = 0.45 * norm_sw if (s_info is not None and s_info.brand in session_brands) else 0.0
                    scores[pid] = (
                        scores.get(pid, 0.0) + cfg.weight_subcategory_pop * norm_sw * item_score + brand_boost
                    )

        # 3. Deep User Profile (cart/view intent recovery + subcategory/brand affinity)
        user_preferred_brands: dict[str, float] = {}
        if profile is not None:
            for pid, intent_w in profile.intent_items:
                scores[pid] = scores.get(pid, 0.0) + cfg.weight_intent_recovery * intent_w

            for subcat, u_aff in profile.subcat_affinity[:2]:
                for pid, item_score in snapshot.subcategory_ranked.get(subcat, ())[:30]:
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

            scores[pid] = val

        ranked_pairs = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], -snapshot.product_pop_score.get(kv[0], 0.0), kv[0]),
        )
        diversified = self._select_with_mmr(
            ranked_pairs,
            snapshot,
            k=k,
            has_session_intent=(
                bool(recent_views)
                or (profile is not None and bool(profile.subcat_affinity or profile.intent_items))
            ),
        )
        diversified = self._revenue_tiebreak(
            diversified,
            scores,
            snapshot.products,
            getattr(cfg, "revenue_eps", 0.015),
        )

        return ensure_valid_submission(
            candidates=diversified,
            active_ids=snapshot.active_ids,
            fallback_ranked=snapshot.global_ranked,
            exclude_ids=already_bought,
            k=k,
        )
