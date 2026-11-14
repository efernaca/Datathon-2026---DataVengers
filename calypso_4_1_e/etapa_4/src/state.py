"""Immutable snapshot containers, StaticHistoryBundle, and atomic StateStore."""
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

    arms: dict[str, tuple[float, float, int, int, int]] = field(default_factory=dict)
    total_effective_exposures: float = 0.0
    global_reward_rate: float = 0.05
    exam_weights: tuple[float, ...] = ()
    feedback_rows_processed: int = 0


@dataclass(frozen=True, slots=True)
class AgentOverrides:
    """Dynamic runtime adjustments selected by CalypsoAgent during live cycles."""

    ml_blend_alpha_scale: float = 1.0
    bandit_ucb_alpha: float = 0.14
    weight_reverse_cold_graft: float = 0.30
    weight_forward_cold: float = 0.48
    revenue_tie_weight: float = 0.03
    mmr_brand_lambda: float = 0.04
    regime: str = "nominal"
    reason: str = "Initial Stage 4 defaults"
    diagnosed_at: str = ""


@dataclass(slots=True)
class StaticHistoryBundle:
    """Pre-aggregated static 60-day SQL tables persisted to disk once."""

    fetched_at: float
    catalog_rows: list[dict[str, Any]]
    popularity_rows: list[dict[str, Any]]
    transition_rows: list[dict[str, Any]]
    user_profile_rows: list[dict[str, Any]]
    user_intent_rows: list[dict[str, Any]]
    user_affinity_rows: list[dict[str, Any]]
    context_rows: list[dict[str, Any]]
    training_session_rows: list[dict[str, Any]]
    # 4.1: carritos pendientes y compras repetidas del histórico (shop.history_events)
    cart_hist_rows: list[dict[str, Any]] = field(default_factory=list)
    recurrent_hist_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RecommenderSnapshotV4:
    """Read-only state swapped atomically between updater and FastLoop."""

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

    # Precomputed per-active-item FastLoop tuples:
    # (subcat, brand, global_pop, live_trend, is_new_f, log_price_norm, exploit_delta, ucb_bonus)
    product_static: dict[
        str, tuple[str, str, float, float, float, float, float, float]
    ] = field(default_factory=dict)

    bandit_state: BanditState = field(default_factory=BanditState)
    ranker: Any = None
    agent_overrides: AgentOverrides = field(default_factory=AgentOverrides)

    # 4.1 · reglas de negocio
    # user_id -> ((product_id, last_add_to_cart_epoch_ms), ...) sin compra posterior, más reciente primero
    user_carts: dict[str, tuple[tuple[str, float], ...]] = field(default_factory=dict)
    # user_id -> ((product_id, n_compras), ...) con n_compras >= 2, más comprado primero
    user_recurrent: dict[str, tuple[tuple[str, int], ...]] = field(default_factory=dict)

    # 4.1.e · perfil de pesos modular actiu
    weights_profile: Any = None
    config_override: Any = None


class StateStore:
    """Thread-safe holder enabling zero-lock fast reads and atomic snapshot swaps."""

    def __init__(self, initial: RecommenderSnapshotV4 | None = None) -> None:
        self._lock = threading.Lock()
        self._snapshot = initial

    def get(self) -> RecommenderSnapshotV4 | None:
        return self._snapshot

    def update(self, new_snapshot: RecommenderSnapshotV4) -> None:
        with self._lock:
            self._snapshot = new_snapshot
