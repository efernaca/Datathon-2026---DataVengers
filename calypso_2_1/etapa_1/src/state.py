"""Thread-safe immutable state snapshot and atomic pointer swap for serving."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


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


@dataclass(frozen=True, slots=True)
class RecommenderSnapshot:
    """Immutable read-only snapshot consumed by the Fast Serving Loop."""

    version: int
    created_at: float
    products: dict[str, ProductInfo]
    active_ids: frozenset[str]
    out_of_stock: frozenset[str]
    global_ranked: tuple[str, ...]
    product_pop_score: dict[str, float]
    context_ranked: dict[tuple[str, str, str], tuple[tuple[str, float], ...]]
    subcategory_ranked: dict[str, tuple[tuple[str, float], ...]]
    transitions: dict[str, tuple[tuple[str, float], ...]]
    cold_neighbors: dict[str, tuple[tuple[str, float], ...]]
    live_trends: dict[str, float] = field(default_factory=dict)
    user_bought: dict[str, frozenset[str]] = field(default_factory=dict)
    user_subcat_affinity: dict[str, tuple[tuple[str, float], ...]] = field(default_factory=dict)

    def with_fast_updates(
        self,
        *,
        out_of_stock: frozenset[str] | None = None,
        live_trends: dict[str, float] | None = None,
        products: dict[str, ProductInfo] | None = None,
        cold_neighbors: dict[str, tuple[tuple[str, float], ...]] | None = None,
    ) -> RecommenderSnapshot:
        """Create a new snapshot sharing unchanged heavy historical structures."""
        new_products = products if products is not None else self.products
        new_oos = out_of_stock if out_of_stock is not None else self.out_of_stock
        new_active = frozenset(
            pid for pid, info in new_products.items() if info.is_active and pid not in new_oos
        )
        # Keep pre-filtered active global fallback ordered; append newly active items at the end
        ranked_active = [pid for pid in self.global_ranked if pid in new_active]
        ranked_set = set(ranked_active)
        for pid in new_active:
            if pid not in ranked_set:
                ranked_active.append(pid)

        return RecommenderSnapshot(
            version=self.version + 1,
            created_at=time.time(),
            products=new_products,
            active_ids=new_active,
            out_of_stock=new_oos,
            global_ranked=tuple(ranked_active),
            product_pop_score=self.product_pop_score,
            context_ranked=self.context_ranked,
            subcategory_ranked=self.subcategory_ranked,
            transitions=self.transitions,
            cold_neighbors=cold_neighbors if cold_neighbors is not None else self.cold_neighbors,
            live_trends=live_trends if live_trends is not None else self.live_trends,
            user_bought=self.user_bought,
            user_subcat_affinity=self.user_subcat_affinity,
        )


class StateStore:
    """Thread-safe container enabling lock-free-like atomic snapshot swaps."""

    def __init__(self, initial: RecommenderSnapshot | None = None) -> None:
        self._lock = threading.Lock()
        self._snapshot = initial

    def get(self) -> RecommenderSnapshot | None:
        with self._lock:
            return self._snapshot

    def set(self, snapshot: RecommenderSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot
