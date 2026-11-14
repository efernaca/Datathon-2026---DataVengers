"""Compatibility bridge exposing Stage1Recommender & build_snapshot_from_records
backed by Etapa 1 (versión especial) EASE + 30-feature vectorized policy."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .catalog import CatalogManager
from .config import Config
from .ease import fit_ease
from .guardrails import NUM_SLOTS
from .live import LiveTracker
from .model import LinearModel
from .policy import Recommender
from .state import HistoryBundle, LiveView, Snapshot, UserProfile


class CompatSnapshot(Snapshot):
    """Snapshot wrapper exposing `.active_ids` and `.global_ranked` for cross-stage evals."""

    @property
    def active_ids(self) -> frozenset[str]:
        return self.catalog.active_ids

    @property
    def global_ranked(self) -> tuple[str, ...]:
        return self.catalog.global_ranked


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
    config: Config | None = None,
) -> CompatSnapshot:
    """Assemble an EASE-powered Snapshot compatible with Etapa 2/3/4 offline benchmarks."""
    cfg = config or Config(cold_top_k=max_cold_neighbors)
    cat_list = [dict(r) for r in catalog_rows]

    pop_map: dict[str, float] = {}
    item_stats: dict[str, tuple[float, float, float, float]] = {}
    item_counts_ease: dict[str, float] = {}
    for r in popularity_rows:
        pid = str(r.get("product_id") or "")
        if not pid:
            continue
        wpop = float(r.get("weighted_pop", r.get("score", r.get("n", 0.0))) or 0.0)
        views = float(r.get("views", wpop) or 0.0)
        purchases = float(r.get("purchases", 0.0) or 0.0)
        pop_map[pid] = wpop
        if views > 0.0 or purchases > 0.0:
            item_stats[pid] = (views, views, 0.0, purchases)
        item_counts_ease[pid] = max(views + purchases * 3.0, wpop, 1.0)

    ctx_prior: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in context_rows:
        key = (str(r.get("region", "") or ""), str(r.get("device", "") or ""), str(r.get("page", "") or ""))
        pid = str(r.get("product_id") or "")
        val = float(r.get("buys", r.get("score", 0.0)) or 0.0)
        if pid and val > 0.0:
            ctx_prior[key][pid] = ctx_prior[key].get(pid, 0.0) + val

    trans_buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    undirected_pairs: dict[tuple[str, str], float] = defaultdict(float)
    for r in transition_rows:
        src = str(r.get("src_product") or "")
        dst = str(r.get("dst_product") or "")
        sc = float(r.get("transition_score", r.get("raw_transition_score", r.get("score", 0.0))) or 0.0)
        if not src or not dst or src == dst or sc <= 0.0:
            continue
        trans_buckets[src].append((dst, sc))
        a, b = (src, dst) if src < dst else (dst, src)
        undirected_pairs[(a, b)] += sc * 0.5
        item_counts_ease.setdefault(src, 1.0)
        item_counts_ease.setdefault(dst, 1.0)

    transitions: dict[str, list[tuple[str, float]]] = {}
    for src, pairs in trans_buckets.items():
        pairs.sort(key=lambda kv: (-kv[1], kv[0]))
        top_pairs = pairs[: cfg.transition_topk]
        max_sc = top_pairs[0][1] if top_pairs else 1.0
        transitions[src] = [(dst, sc / max(max_sc, 1e-9)) for dst, sc in top_pairs]

    pair_rows = [{"i": a, "j": b, "c": c} for (a, b), c in undirected_pairs.items()]
    ease_model = (
        fit_ease(item_counts_ease, pair_rows, lam=min(cfg.ease_lambda, 150.0), min_count=1.0)
        if len(item_counts_ease) >= 2 and pair_rows
        else None
    )

    users: dict[str, UserProfile] = {}
    for r in user_bought_rows:
        uid = str(r.get("user_id") or "")
        if not uid:
            continue
        bought = frozenset(str(p) for p in (r.get("bought_products") or ()) if p)
        users[uid] = UserProfile(bought=bought)

    for r in user_subcat_rows:
        uid = str(r.get("user_id") or "")
        if not uid:
            continue
        prof = users.get(uid) or UserProfile()
        subcats = {str(sc): float(val) for sc, val in (r.get("subcat_scores") or ()) if sc}
        if subcats:
            mx = max(subcats.values()) or 1.0
            prof.subcat = {k: v / mx for k, v in subcats.items()}
        users[uid] = prof

    history = HistoryBundle(
        catalog_rows=cat_list,
        pop=pop_map,
        item_stats=item_stats,
        ctx_prior=dict(ctx_prior),
        transitions=transitions,
        ease=ease_model,
        users=users,
    )

    cm = CatalogManager(history, cfg)
    cm.version = max(0, int(version) - 1)
    for r in cat_list:
        pid = str(r.get("product_id") or "")
        if pid in cm.rows:
            is_new_flag = bool(r.get("is_new_product", r.get("is_new", False)))
            cm.rows[pid]["is_new"] = is_new_flag
            if is_new_flag and pid in cm.h.pop:
                cm.rows[pid]["is_new"] = True
    cm.oos.update(str(p) for p in out_of_stock_ids if p)
    cat = cm.build()

    tracker = LiveTracker(cfg)
    base_view = tracker.view(cat, 1.0)
    raw_trends = {
        str(r.get("product_id") or ""): float(
            r.get("live_momentum", r.get("live_score", r.get("score", 0.0))) or 0.0
        )
        for r in live_trend_rows
    }
    if raw_trends:
        max_tr = max(raw_trends.values()) or 1.0
        trend_arr = base_view.trend.copy()
        for pid, val in raw_trends.items():
            j = cat.idx.get(pid)
            if j is not None and val > 0.0:
                trend_arr[j] = min(3.0, 2.0 * (val / max_tr))
        base_view = LiveView(
            n_items=base_view.n_items,
            log_pop=base_view.log_pop,
            trend=trend_arr,
            total_weight=base_view.total_weight,
            now_ts=base_view.now_ts,
            kappa_mult=base_view.kappa_mult,
        )

    live_map = {arm[0]: base_view for arm in cfg.bandit_arms}
    return CompatSnapshot(history=history, catalog=cat, live=live_map)


class Stage1Recommender:
    """Compatibility wrapper exposing Stage1Recommender.recommend(req, snap, k=10)."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config(bandit_enabled=False, diversify="no_session")
        self._inner = Recommender(
            self.config,
            model=LinearModel.default(),
            bandit=None,
            seed=0,
        )

    def recommend(
        self,
        request: Mapping[str, Any] | Any,
        snapshot: Snapshot,
        k: int = NUM_SLOTS,
    ) -> list[str]:
        decision = self._inner.recommend(request, snapshot, k=k, arm="hist", variant="model")
        return decision.items
