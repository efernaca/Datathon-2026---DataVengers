"""Immutable snapshot containers and atomic StateStore for calypso-2.1."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProductInfo:
    """Active or inactive catalog item metadata."""

    product_id: str
    category: str
    subcategory: str
    brand: str
    price_eur: float
    description: str
    is_active: bool
    is_new: bool = False


@dataclass(frozen=True, slots=True)
class UserProfile:
    """Historical user profile: purchased items, active intents, and price sensitivity."""

    bought_products: frozenset[str] = frozenset()
    intent_items: tuple[tuple[str, float], ...] = ()
    subcat_brand_affinity: dict[tuple[str, str], float] = field(default_factory=dict)
    log_price_mean: float = 0.0
    log_price_std: float = 0.0


@dataclass(frozen=True, slots=True)
class BanditState:
    """Position-adjusted empirical conversion & exposure state for UCB scoring."""

    # product_id -> (effective_exposures, weighted_rewards, raw_impressions, raw_purchases, raw_clicks)
    arms: dict[str, tuple[float, float, int, int, int]] = field(default_factory=dict)
    total_effective_exposures: float = 0.0
    global_reward_rate: float = 0.05
    exam_weights: tuple[float, ...] = ()
    feedback_rows_processed: int = 0


@dataclass(frozen=True, slots=True)
class AgentOverrides:
    """Dynamic runtime adjustments selected by CalypsoAgent during Slow Loop cycles."""

    ml_blend_alpha_scale: float = 1.0
    bandit_ucb_alpha: float = 0.14
    weight_reverse_cold_graft: float = 0.30
    weight_forward_cold: float = 0.45
    revenue_tie_weight: float = 0.03
    mmr_brand_lambda: float = 0.04
    regime: str = "nominal"
    reason: str = "Initial Stage 3 defaults"
    diagnosed_at: str = ""


@dataclass(slots=True)
class RecommenderSnapshotV3:
    """Complete read-only state swapped atomically between Slow Loop and Fast Loop."""

    version: int
    built_at: float
    products: dict[str, ProductInfo]
    active_ids: frozenset[str]
    out_of_stock_ids: frozenset[str]
    new_product_ids: frozenset[str]

    self_cvr: dict[str, float]
    global_pop_norm: dict[str, float]
    max_log_price: float

    transitions: dict[str, tuple[tuple[str, float], ...]]
    cold_neighbors: dict[str, tuple[tuple[str, float], ...]]
    reverse_cold_neighbors: dict[str, tuple[tuple[str, float], ...]]

    ctx_3: dict[tuple[str, str, str], tuple[tuple[str, float], ...]]
    ctx_2: dict[tuple[str, str], tuple[tuple[str, float], ...]]
    ctx_1: dict[tuple[str], tuple[tuple[str, float], ...]]

    user_profiles: dict[str, UserProfile]
    live_user_bought: dict[str, frozenset[str]]
    live_trend_norm: dict[str, float]

    fallback_ranked: tuple[str, ...]
    revenue_fallback: tuple[str, ...]

    bandit_state: BanditState = field(default_factory=BanditState)
    ranker: Any = None
    agent_overrides: AgentOverrides = field(default_factory=AgentOverrides)


class StateStore:
    """Thread-safe holder enabling zero-lock fast reads and atomic snapshot swaps."""

    def __init__(self, initial: RecommenderSnapshotV3 | None = None) -> None:
        self._lock = threading.Lock()
        self._snapshot = initial

    def get(self) -> RecommenderSnapshotV3 | None:
        return self._snapshot

    def update(self, new_snapshot: RecommenderSnapshotV3) -> None:
        with self._lock:
            self._snapshot = new_snapshot
