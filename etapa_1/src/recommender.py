"""Stage 1 hybrid recommender policy and snapshot builder."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
import time

from .cold_start import ColdStartIndex
from .config import Config
from .guardrails import NUM_SLOTS, ensure_valid_submission
from .state import ProductInfo, RecommenderSnapshot


def _normalize_pairs(items: Iterable[tuple[str, float]], limit: int) -> tuple[tuple[str, float], ...]:
    """Sort descending by score, keep top `limit`, and scale scores into (0, 1]."""
    sorted_items = sorted(items, key=lambda x: (-x[1], x[0]))[:limit]
    if not sorted_items:
        return ()
    top = max(sorted_items[0][1], 1e-9)
    return tuple((pid, float(score / top)) for pid, score in sorted_items if score > 0.0)


def build_snapshot_from_records(
    catalog_rows: Iterable[Mapping[str, Any]],
    *,
    popularity_rows: Iterable[Mapping[str, Any]] = (),
    context_rows: Iterable[Mapping[str, Any]] = (),
    transition_rows: Iterable[Mapping[str, Any]] = (),
    user_bought_rows: Iterable[Mapping[str, Any]] = (),
    user_subcat_rows: Iterable[Mapping[str, Any]] = (),
    out_of_stock_ids: Iterable[str] = (),
    live_trend_rows: Iterable[Mapping[str, Any]] = (),
    version: int = 1,
    max_cold_neighbors: int = 8,
) -> RecommenderSnapshot:
    """Assemble a RecommenderSnapshot from RawTree SQL query outputs or offline tables."""
    products: dict[str, ProductInfo] = {}
    for r in catalog_rows:
        pid = str(r["product_id"])
        price = float(r.get("price_eur", 0.0) or 0.0)
        init_price = float(r.get("initial_price_eur", price) or price)
        products[pid] = ProductInfo(
            product_id=pid,
            category=str(r.get("category", "")),
            subcategory=str(r.get("subcategory", "")),
            brand=str(r.get("brand", "")),
            price_eur=price,
            initial_price_eur=init_price,
            description=str(r.get("description", "")),
            is_active=bool(r.get("is_active", True)),
            is_new=bool(r.get("is_new_product", r.get("is_new", False))),
        )

    oos_set = frozenset(str(x) for x in out_of_stock_ids)
    active_ids = frozenset(
        pid for pid, info in products.items() if info.is_active and pid not in oos_set
    )

    raw_pop: dict[str, float] = {}
    for r in popularity_rows:
        pid = str(r["product_id"])
        raw_pop[pid] = float(r.get("weighted_pop", r.get("score", r.get("n", 0.0))) or 0.0)

    max_pop = max(raw_pop.values(), default=1.0) or 1.0
    product_pop_score = {pid: score / max_pop for pid, score in raw_pop.items() if score > 0.0}

    # Global active ranking: highest weighted popularity first, deterministic tie-break
    global_ranked = tuple(
        sorted(
            active_ids,
            key=lambda pid: (-raw_pop.get(pid, 0.0), pid),
        )
    )

    # Subcategory rankings
    subcat_buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for pid in active_ids:
        info = products[pid]
        pop = product_pop_score.get(pid, 1e-4)
        subcat_buckets[info.subcategory].append((pid, pop))
    subcategory_ranked = {
        subcat: _normalize_pairs(pairs, limit=25)
        for subcat, pairs in subcat_buckets.items()
    }

    # Contextual rankings: (region, device, page) -> top products
    ctx_buckets: dict[tuple[str, str, str], list[tuple[str, float]]] = defaultdict(list)
    page_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in context_rows:
        region = str(r.get("region", ""))
        device = str(r.get("device", ""))
        page = str(r.get("page", ""))
        pid = str(r["product_id"])
        score = float(r.get("buys", r.get("score", 0.0)) or 0.0)
        if score > 0.0:
            ctx_buckets[(region, device, page)].append((pid, score))
            page_buckets[page][pid] += score

    context_ranked: dict[tuple[str, str, str], tuple[tuple[str, float], ...]] = {
        ctx: _normalize_pairs(pairs, limit=25) for ctx, pairs in ctx_buckets.items()
    }
    for page, pid_scores in page_buckets.items():
        context_ranked[("", "", page)] = _normalize_pairs(pid_scores.items(), limit=25)

    # Directed session transition adjacency list
    trans_buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in transition_rows:
        src = str(r["src_product"])
        dst = str(r["dst_product"])
        score = float(r.get("transition_score", r.get("score", 0.0)) or 0.0)
        if src != dst and score > 0.0:
            trans_buckets[src].append((dst, score))
    transitions = {
        src: _normalize_pairs(pairs, limit=30) for src, pairs in trans_buckets.items()
    }

    # User purchase history (for repeat-purchase suppression)
    user_bought: dict[str, frozenset[str]] = {}
    for r in user_bought_rows:
        uid = str(r["user_id"])
        bought = r.get("bought_products") or ()
        user_bought[uid] = frozenset(str(p) for p in bought)

    # Optional user subcategory affinity
    user_subcat_affinity: dict[str, tuple[tuple[str, float], ...]] = {}
    for r in user_subcat_rows:
        uid = str(r["user_id"])
        pairs = r.get("subcat_scores") or ()
        user_subcat_affinity[uid] = _normalize_pairs(pairs, limit=5)

    # Live trends
    raw_trends = {
        str(r["product_id"]): float(r.get("live_momentum", r.get("score", 0.0)) or 0.0)
        for r in live_trend_rows
    }
    max_trend = max(raw_trends.values(), default=1.0) or 1.0
    live_trends = {pid: val / max_trend for pid, val in raw_trends.items() if val > 0.0}

    # Cold-start index: warm items are those with transition rules or historical popularity
    warm_ids = [
        pid
        for pid in products
        if pid in active_ids and (len(transitions.get(pid, ())) >= 2 or raw_pop.get(pid, 0.0) > 0.0)
    ]
    cold_targets = [
        info
        for pid, info in products.items()
        if info.is_new or len(transitions.get(pid, ())) < 3
    ]
    cold_neighbors: dict[str, tuple[tuple[str, float], ...]] = {}
    if cold_targets and products:
        cold_index = ColdStartIndex().fit(products, warm_ids=warm_ids)
        cold_neighbors = cold_index.neighbors_for_batch(cold_targets, top_k=max_cold_neighbors)

    return RecommenderSnapshot(
        version=version,
        created_at=time.time(),
        products=products,
        active_ids=active_ids,
        out_of_stock=oos_set,
        global_ranked=global_ranked,
        product_pop_score=product_pop_score,
        context_ranked=context_ranked,
        subcategory_ranked=subcategory_ranked,
        transitions=transitions,
        cold_neighbors=cold_neighbors,
        live_trends=live_trends,
        user_bought=user_bought,
        user_subcat_affinity=user_subcat_affinity,
    )


def _extract_recent_product_ids(recent_views: Any) -> list[str]:
    """Normalize recent_views from dicts or bare strings in chronological order."""
    if not recent_views:
        return []
    out: list[str] = []
    for item in recent_views:
        if isinstance(item, Mapping):
            pid = item.get("product_id")
            if pid:
                out.append(str(pid))
        elif item:
            out.append(str(item))
    return out


class Stage1Recommender:
    """Fast Stage 1 hybrid policy operating purely in memory over RecommenderSnapshot."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def recommend(
        self,
        request: Mapping[str, Any] | Any,
        snapshot: RecommenderSnapshot,
        k: int = NUM_SLOTS,
    ) -> list[str]:
        """Score candidates and return exactly k distinct, active product IDs."""
        if isinstance(request, Mapping):
            user_id = str(request.get("user_id", "") or "")
            is_new_user = bool(request.get("is_new_user", False))
            context = request.get("context") or {}
            recent_views_raw = request.get("recent_views") or ()
        else:
            user_id = str(getattr(request, "user_id", "") or "")
            is_new_user = bool(getattr(request, "is_new_user", False))
            context = getattr(request, "context", None) or {}
            recent_views_raw = getattr(request, "recent_views", None) or ()

        if isinstance(context, Mapping):
            region = str(context.get("region", "") or "")
            device = str(context.get("device", "") or "")
            page = str(context.get("page", "") or "")
        else:
            region, device, page = "", "", ""

        already_bought = (
            frozenset() if is_new_user else snapshot.user_bought.get(user_id, frozenset())
        )
        recent_pids = _extract_recent_product_ids(recent_views_raw)
        cfg = self.config

        scores: dict[str, float] = {}
        subcat_weights: dict[str, float] = {}

        # 1. Session signals from recent_views (most recent first)
        for idx, v_pid in enumerate(reversed(recent_pids)):
            w = cfg.recent_view_decay ** idx
            v_info = snapshot.products.get(v_pid)

            if v_pid in snapshot.active_ids and v_pid not in already_bought:
                # On a product page, the shopper is already looking at the last viewed product,
                # so slightly dampen its direct repeat weight in favor of complements.
                exact_w = cfg.weight_exact_recent_view * (0.6 if (page == "product" and idx == 0) else 1.0)
                scores[v_pid] = scores.get(v_pid, 0.0) + exact_w * w

            if v_info is not None and v_info.subcategory:
                subcat_weights[v_info.subcategory] = subcat_weights.get(v_info.subcategory, 0.0) + w

            trans = snapshot.transitions.get(v_pid, ())
            for dst_pid, t_score in trans:
                scores[dst_pid] = scores.get(dst_pid, 0.0) + cfg.weight_coview * w * t_score

            # Cold-start bridge + Graph Grafting for new or sparse items
            is_cold_item = (v_info is not None and v_info.is_new) or len(trans) < 3
            if is_cold_item:
                for nb_pid, sim in snapshot.cold_neighbors.get(v_pid, ()):
                    scores[nb_pid] = scores.get(nb_pid, 0.0) + cfg.weight_cold_neighbor * w * sim
                    for dst_pid, t_score in snapshot.transitions.get(nb_pid, ())[:6]:
                        if dst_pid != v_pid:
                            scores[dst_pid] = (
                                scores.get(dst_pid, 0.0) + cfg.weight_cold_graft * w * sim * t_score
                            )

        # 2. Subcategory affinity (session + known user profile)
        if subcat_weights:
            top_subcats = sorted(subcat_weights.items(), key=lambda x: -x[1])[:2]
            max_sw = top_subcats[0][1] or 1.0
            for subcat, sw in top_subcats:
                norm_sw = sw / max_sw
                for pid, item_score in snapshot.subcategory_ranked.get(subcat, ())[:15]:
                    scores[pid] = scores.get(pid, 0.0) + cfg.weight_subcategory_pop * norm_sw * item_score

        if not is_new_user and user_id:
            for subcat, u_aff in snapshot.user_subcat_affinity.get(user_id, ())[:2]:
                for pid, item_score in snapshot.subcategory_ranked.get(subcat, ())[:12]:
                    scores[pid] = scores.get(pid, 0.0) + cfg.weight_user_affinity * u_aff * item_score

        # 3. Context popularity (region, device, page)
        ctx_list = snapshot.context_ranked.get((region, device, page))
        if not ctx_list and page:
            ctx_list = snapshot.context_ranked.get(("", "", page))
        if ctx_list:
            for pid, ctx_score in ctx_list[:20]:
                scores[pid] = scores.get(pid, 0.0) + cfg.weight_context_pop * ctx_score

        # 4. Seed global top-20 & live trending items so empty/cold sessions always score smoothly
        for pid in snapshot.global_ranked[:20]:
            scores.setdefault(pid, 0.0)
        if snapshot.live_trends:
            for pid, trend in list(snapshot.live_trends.items())[:15]:
                if pid in snapshot.active_ids:
                    scores.setdefault(pid, 0.0)

        # 5. Add global popularity, live momentum, and price-drop elasticity
        for pid in list(scores):
            if pid not in snapshot.active_ids or pid in already_bought:
                del scores[pid]
                continue
            val = scores[pid]
            val += cfg.weight_global_pop * snapshot.product_pop_score.get(pid, 0.0)
            val += cfg.weight_live_trend * snapshot.live_trends.get(pid, 0.0)
            info = snapshot.products.get(pid)
            if info is not None and info.discount_pct > 0.0:
                val += cfg.weight_price_drop * info.discount_pct
            scores[pid] = val

        ranked_candidates = [
            pid
            for pid, _ in sorted(
                scores.items(),
                key=lambda kv: (-kv[1], -snapshot.product_pop_score.get(kv[0], 0.0), kv[0]),
            )
        ]

        return ensure_valid_submission(
            candidates=ranked_candidates,
            active_ids=snapshot.active_ids,
            fallback_ranked=snapshot.global_ranked,
            exclude_ids=already_bought,
            k=k,
        )
