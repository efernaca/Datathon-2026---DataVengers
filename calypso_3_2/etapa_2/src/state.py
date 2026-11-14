"""Immutable state snapshots and user profiles for Calypso 3.2 Etapa 2."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True, slots=True)
class ProductInfo:
    """Consolidated metadata for one catalog product."""

    product_id: str
    category: str
    subcategory: str
    brand: str
    price_eur: float
    initial_price_eur: float
    description: str
    is_active: bool
    is_new: bool = False

    @property
    def discount_pct(self) -> float:
        """Relative price drop from initial catalog price, clipped to [0, 0.9]."""
        if self.initial_price_eur <= 0.0 or self.price_eur >= self.initial_price_eur:
            return 0.0
        drop = (self.initial_price_eur - self.price_eur) / self.initial_price_eur
        return min(max(drop, 0.0), 0.9)


@dataclass(slots=True)
class UserProfile:
    """Historical user profile supporting both Stage2Recommender and Stage2Special."""

    user_id: str = ""
    bought: frozenset[str] = frozenset()
    intent_items: tuple[tuple[str, float], ...] = ()
    subcat_affinity: tuple[tuple[str, float], ...] = ()
    brand_affinity: tuple[tuple[str, float], ...] = ()
    log_price_mean: float = 0.0
    log_price_std: float = 0.0
    # Etapa 2 Especial fields
    subcat: dict[str, float] = field(default_factory=dict)
    brand: dict[str, float] = field(default_factory=dict)
    price_mu: float = float("nan")
    price_sigma: float = float("nan")
    n_buys: int = 0


@dataclass(slots=True)
class EaseModel:
    """Closed-form EASE item-item matrix B (n x n, float32) with zero diagonal."""

    item_ids: list[str]
    B: np.ndarray
    lam: float = 0.0

    @property
    def index(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.item_ids)}


@dataclass(slots=True)
class HistoryBundle:
    """Precomputed 60-day history bundle (persisted via pickle)."""

    catalog_rows: list[dict[str, Any]]
    pop: dict[str, float]
    item_stats: dict[str, tuple[float, float, float, float]]
    ctx_prior: dict[tuple[str, str, str], dict[str, float]]
    transitions: dict[str, list[tuple[str, float]]]
    ease: EaseModel | None
    users: dict[str, UserProfile]
    visto_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    intent: dict[str, tuple[tuple[str, float, float], ...]] = field(default_factory=dict)
    version: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    ok_components: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogState:
    """Aligned catalog arrays built by CatalogManager (append-only for new products)."""

    version: int
    ids: tuple[str, ...]
    idx: dict[str, int]
    category: np.ndarray
    subcategory: np.ndarray
    brand: np.ndarray
    cat_names: tuple[str, ...]
    subcat_names: tuple[str, ...]
    brand_codes: dict[str, int]
    subcat_codes: dict[str, int]
    price: np.ndarray
    log_price: np.ndarray
    price_rel_z: np.ndarray
    active: np.ndarray
    is_new: np.ndarray
    active_ids: frozenset[str]
    global_ranked: tuple[str, ...]
    log_pop_prior: np.ndarray
    pop_share_prior: np.ndarray
    subcat_log_pop: np.ndarray
    cvr_logit: np.ndarray
    ctx_arrays: dict[tuple[str, str, str], np.ndarray]
    ease_proj: sp.csr_matrix | None
    content: sp.csr_matrix
    cold_neighbors: dict[str, tuple[tuple[str, float], ...]]
    price_change_rel: np.ndarray
    price_change_ts: np.ndarray
    visto_conv: np.ndarray | None = None
    reverse: dict[str, tuple[tuple[str, float], ...]] = field(default_factory=dict)
    created_at: float = 0.0

    @property
    def n(self) -> int:
        return len(self.ids)


@dataclass(frozen=True, slots=True)
class LiveView:
    """D1/V5 aligned live popularity & trend arrays."""

    n_items: int
    log_pop: np.ndarray
    trend: np.ndarray
    total_weight: float = 0.0
    now_ts: float = 0.0
    kappa_mult: float = 1.0


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable snapshot consumed by Stage2Special."""

    history: HistoryBundle
    catalog: CatalogState
    live: dict[str, LiveView] = field(default_factory=dict)
    live_bought: dict[str, frozenset[str]] = field(default_factory=dict)
    new_impressions: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecommenderSnapshotV2:
    """Immutable read-only snapshot consumed by Stage2Recommender."""

    version: int
    created_at: float
    products: dict[str, ProductInfo]
    active_ids: frozenset[str]
    out_of_stock: frozenset[str]
    global_ranked: tuple[str, ...]
    product_pop_score: dict[str, float]
    self_conversion: dict[str, float]
    subcat_avg_price: dict[str, float]
    context_ranked: dict[tuple[str, str, str], tuple[tuple[str, float], ...]]
    subcategory_ranked: dict[str, tuple[tuple[str, float], ...]]
    transitions: dict[str, tuple[tuple[str, float], ...]]
    cold_neighbors: dict[str, tuple[tuple[str, float], ...]]
    warm_to_cold: dict[str, tuple[tuple[str, float], ...]]
    user_profiles: dict[str, UserProfile] = field(default_factory=dict)
    live_trends: dict[str, float] = field(default_factory=dict)
    live_user_bought: dict[str, frozenset[str]] = field(default_factory=dict)

    def with_fast_updates(
        self,
        *,
        out_of_stock: frozenset[str] | None = None,
        live_trends: dict[str, float] | None = None,
        live_user_bought: dict[str, frozenset[str]] | None = None,
        products: dict[str, ProductInfo] | None = None,
        cold_neighbors: dict[str, tuple[tuple[str, float], ...]] | None = None,
        warm_to_cold: dict[str, tuple[tuple[str, float], ...]] | None = None,
    ) -> RecommenderSnapshotV2:
        """Produce an updated snapshot while sharing unchanged heavy matrices."""
        new_products = products if products is not None else self.products
        new_oos = out_of_stock if out_of_stock is not None else self.out_of_stock
        new_active = frozenset(
            pid for pid, info in new_products.items() if info.is_active and pid not in new_oos
        )
        ranked_active = [pid for pid in self.global_ranked if pid in new_active]
        ranked_set = set(ranked_active)
        for pid in new_active:
            if pid not in ranked_set:
                ranked_active.append(pid)

        return RecommenderSnapshotV2(
            version=self.version + 1,
            created_at=time.time(),
            products=new_products,
            active_ids=new_active,
            out_of_stock=new_oos,
            global_ranked=tuple(ranked_active),
            product_pop_score=self.product_pop_score,
            self_conversion=self.self_conversion,
            subcat_avg_price=self.subcat_avg_price,
            context_ranked=self.context_ranked,
            subcategory_ranked=self.subcategory_ranked,
            transitions=self.transitions,
            cold_neighbors=cold_neighbors if cold_neighbors is not None else self.cold_neighbors,
            warm_to_cold=warm_to_cold if warm_to_cold is not None else self.warm_to_cold,
            user_profiles=self.user_profiles,
            live_trends=live_trends if live_trends is not None else self.live_trends,
            live_user_bought=(
                live_user_bought if live_user_bought is not None else self.live_user_bought
            ),
        )


class StateStore:
    """Thread-safe container enabling atomic swaps & partial updates."""

    def __init__(self, initial: Any = None) -> None:
        self._lock = threading.Lock()
        self._snapshot = initial

    def get(self) -> Any:
        return self._snapshot

    def set(self, snapshot: Any) -> None:
        with self._lock:
            self._snapshot = snapshot

    def update(self, **changes: Any) -> Any:
        with self._lock:
            if self._snapshot is None:
                return None
            self._snapshot = replace(self._snapshot, **changes)
            return self._snapshot

    def mutate(self, fn: Any) -> Any:
        with self._lock:
            if self._snapshot is None:
                return None
            changes = fn(self._snapshot)
            if changes:
                self._snapshot = replace(self._snapshot, **changes)
            return self._snapshot


def now_s() -> float:
    return time.time()
