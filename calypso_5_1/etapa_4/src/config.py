"""Runtime configuration, .env loader, and cache path management for Stage 4."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .weights_profile import WeightsProfile, load_active_profile


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw is not None and raw.strip() else default
    except ValueError:
        return default


def _parse_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from an env file without overriding existing env vars."""
    if not path.is_file():
        return
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in raw_lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_dotenv_files() -> None:
    """Discover and load .env / group_API.env from workspace and parent folders."""
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    seen_dirs: set[Path] = set()
    for base in [Path.cwd(), here.parent, here.parents[1], here.parents[2], here.parents[3]]:
        resolved = base.resolve()
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        candidates.append(resolved / ".env")
        candidates.append(resolved / "group_API.env")
    for env_path in candidates:
        _parse_env_file(env_path)


@dataclass(frozen=True, slots=True)
class ConfigV4:
    """Immutable configuration for Stage 4 (calypso-5.1: alta conversió + modular hot-reload)."""

    api_url: str = "https://api.rawtree.com"
    team_id: str = ""
    api_key: str = ""
    model_version: str = "calypso-5.1"
    weights_profile_path: str = "weights_profile.json"

    # SLA-hardened Fast / Live / Stock cadence timings
    poll_interval_s: float = 0.12
    stock_refresh_interval_s: float = 1.5
    live_refresh_interval_s: float = 15.0
    static_refresh_interval_s: float = 900.0
    max_requests_per_batch: int = 25
    max_request_age_ms: int = 2400
    cache_dir: str = "cache"

    # Fast candidate retrieval & sequential scoring
    candidate_pool_size: int = 84
    recency_decay: float = 0.65
    lift_beta: float = 0.45
    lift_prior_pop: float = 30.0
    self_view_weight: float = 1.80
    intent_weight: float = 0.50
    affinity_weight: float = 0.25
    price_compat_weight: float = 0.20
    ctx_bayes_weight: float = 0.30
    global_pop_weight: float = 0.22
    live_trend_weight: float = 0.25
    weight_forward_cold: float = 0.48
    weight_reverse_cold_graft: float = 0.30

    # Slot-Gated ML Re-Ranker & Position-Aware Bandit
    ml_max_blend_alpha: float = 0.45
    ml_min_positives: int = 25
    ml_l2_c: float = 1.2
    bandit_prior_strength: float = 20.0
    bandit_ucb_alpha: float = 0.14
    bandit_cold_ucb_multiplier: float = 1.8
    bandit_exam_power: float = 0.75
    cold_explore_start_slot_warm: int = 7
    max_cold_explore_in_warm: int = 2

    # Diversity / MMR guardrails
    mmr_start_slot_session: int = 5
    mmr_start_slot_home: int = 3
    mmr_subcat_lambda_session: float = 0.0
    mmr_subcat_lambda_home: float = 0.15
    mmr_brand_lambda: float = 0.04
    revenue_tie_weight: float = 0.03

    # Cold-start graph limits
    max_cold_neighbors: int = 8
    max_reverse_cold_per_warm: int = 3

    # ── Interruptores de los 4 cambios de 3.1 respecto a 2.1 ──────────────
    # Cada uno se puede forzar con su variable de entorno (1/0) sin tocar código.
    # (1) Quitar los suelos max(w12, 0.28) / max(w13, 0.12) del bonus subcat/marca
    v3_no_subcat_floor: bool = field(default_factory=lambda: _env_bool("CALYPSO_V3_NO_SUBCAT_FLOOR", False))
    # (1b) Penalizar también el fondo de productos de la misma subcategoría (x0.45 / x0.18)
    v3_bg_penalty: bool = field(default_factory=lambda: _env_bool("CALYPSO_V3_BG_PENALTY", True))
    # (2) Peso del "EASE" mezclado con las transiciones lift (0.0 = como 2.1, 0.25 = como 3.1)
    v3_ease_weight: float = field(default_factory=lambda: _env_float("CALYPSO_V3_EASE_WEIGHT", 0.0))
    # (3) SQL de transiciones solo hacia delante (b.t >= a.t, LIMIT 60). Solo afecta a datos reales.
    v3_directional_transitions: bool = field(default_factory=lambda: _env_bool("CALYPSO_V3_DIRECTIONAL", False))
    # (4) Umbrales de cold start (min_similarity 0.08 + normalizar por max(max,0.25))
    v3_cold_thresholds: bool = field(default_factory=lambda: _env_bool("CALYPSO_V3_COLD_THRESHOLDS", False))

    # ── Reglas de negocio de 4.1 (cada una conmutable por variable de entorno) ──
    # (M1) Producto en el carrito (añadido hace <= v4_cart_active_minutes y sin comprar) -> primeros slots
    v4_cart_priority: bool = field(default_factory=lambda: _env_bool("CALYPSO_V4_CART_PRIORITY", True))
    v4_cart_active_minutes: float = field(default_factory=lambda: _env_float("CALYPSO_V4_CART_ACTIVE_MIN", 30.0))
    # Máximo de slots que pueden ocupar los productos de carrito (activo + olvidado) entre los 10
    v4_cart_max_slots: int = field(default_factory=lambda: int(_env_float("CALYPSO_V4_CART_MAX_SLOTS", 3)))
    # (M2) Carrito olvidado (añadido hace más de v4_cart_active_minutes y sin comprar) -> justo detrás
    v4_forgotten_cart: bool = field(default_factory=lambda: _env_bool("CALYPSO_V4_FORGOTTEN_CART", True))
    v4_forgotten_max_slots: int = field(default_factory=lambda: int(_env_float("CALYPSO_V4_FORGOTTEN_MAX_SLOTS", 2)))
    v4_forgotten_max_days: float = field(default_factory=lambda: _env_float("CALYPSO_V4_FORGOTTEN_MAX_DAYS", 30.0))
    # (M3) Orden final de cada bloque de más barato a más caro
    v4_price_ascending: bool = field(default_factory=lambda: _env_bool("CALYPSO_V4_PRICE_ASC", True))
    # (M4) Comprado 1 vez -> excluido (ya en 3.1b); comprado >= N veces -> recurrente, se recomienda
    v4_recurrent: bool = field(default_factory=lambda: _env_bool("CALYPSO_V4_RECURRENT", True))
    v4_recurrent_min_purchases: int = field(default_factory=lambda: int(_env_float("CALYPSO_V4_RECURRENT_MIN", 2)))
    v4_recurrent_max_slots: int = field(default_factory=lambda: int(_env_float("CALYPSO_V4_RECURRENT_MAX_SLOTS", 2)))
    # (M5) Compras orgánicas en vivo (shop.events) también cuentan como "ya comprado" (antes solo feedback)
    v4_live_event_purchases: bool = field(default_factory=lambda: _env_bool("CALYPSO_V4_LIVE_EVENT_PURCHASES", True))

    def resolve_cache_dir(self) -> Path:
        """Return absolute path to the encapsulated model cache directory."""
        p = Path(self.cache_dir)
        if p.is_absolute():
            return p
        return Path(__file__).resolve().parents[2] / p

    def resolve_weights_profile_path(self) -> Path:
        """Return absolute path to the modular weights profile JSON."""
        p = Path(self.weights_profile_path)
        if p.is_absolute():
            return p
        return Path(__file__).resolve().parents[2] / p

    def load_weights_profile(self) -> WeightsProfile:
        """Load active WeightsProfile from JSON file or CALYPSO_PRESET env."""
        return load_active_profile(self.resolve_weights_profile_path())

    @classmethod
    def from_profile(cls, wp: WeightsProfile, **overrides: Any) -> ConfigV4:
        """Construct ConfigV4 synchronized with a modular WeightsProfile."""
        kwargs: dict[str, Any] = {
            "candidate_pool_size": wp.candidate_pool_size,
            "recency_decay": wp.recency_decay,
            "self_view_weight": wp.w_self_cvr_view,
            "intent_weight": wp.w_intent_score,
            "affinity_weight": wp.w_user_affinity,
            "price_compat_weight": wp.w_price_compat,
            "ctx_bayes_weight": wp.w_ctx_bayes,
            "global_pop_weight": wp.w_global_pop,
            "live_trend_weight": wp.w_live_trend,
            "weight_forward_cold": wp.w_forward_cold,
            "weight_reverse_cold_graft": wp.w_reverse_graft,
            "ml_max_blend_alpha": wp.ml_max_blend_alpha,
            "ml_min_positives": wp.ml_min_positives,
            "ml_l2_c": wp.ml_l2_c,
            "bandit_prior_strength": wp.bandit_prior_strength,
            "bandit_ucb_alpha": wp.bandit_ucb_alpha,
            "bandit_cold_ucb_multiplier": wp.bandit_cold_ucb_multiplier,
            "bandit_exam_power": wp.bandit_exam_power,
            "cold_explore_start_slot_warm": wp.cold_explore_start_slot_warm,
            "max_cold_explore_in_warm": wp.max_cold_explore_in_warm,
            "mmr_start_slot_session": wp.mmr_start_slot_session,
            "mmr_start_slot_home": wp.mmr_start_slot_home,
            "mmr_subcat_lambda_session": wp.mmr_subcat_lambda_session,
            "mmr_subcat_lambda_home": wp.mmr_subcat_lambda_home,
            "mmr_brand_lambda": wp.mmr_brand_lambda,
            "revenue_tie_weight": wp.revenue_tie_weight,
            "v3_no_subcat_floor": wp.v3_no_subcat_floor,
            "v3_bg_penalty": wp.v3_bg_penalty,
            "v3_ease_weight": wp.v3_ease_weight,
            "v4_cart_priority": wp.v4_cart_priority,
            "v4_cart_active_minutes": wp.v4_cart_active_minutes,
            "v4_cart_max_slots": wp.v4_cart_max_slots,
            "v4_forgotten_cart": wp.v4_forgotten_cart,
            "v4_forgotten_max_slots": wp.v4_forgotten_max_slots,
            "v4_forgotten_max_days": wp.v4_forgotten_max_days,
            "v4_price_ascending": wp.v4_price_ascending,
            "v4_recurrent": wp.v4_recurrent,
            "v4_recurrent_min_purchases": wp.v4_recurrent_min_purchases,
            "v4_recurrent_max_slots": wp.v4_recurrent_max_slots,
            "v4_live_event_purchases": wp.v4_live_event_purchases,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_env(cls, weights_profile_path: str | Path | None = None) -> ConfigV4:
        """Build ConfigV4 from WeightsProfile, environment variables, and env files."""
        load_dotenv_files()
        defaults = cls()
        raw_wp_path = (
            str(weights_profile_path)
            if weights_profile_path is not None
            else (
                os.environ.get("CALYPSO_WEIGHTS_PROFILE", defaults.weights_profile_path).strip()
                or defaults.weights_profile_path
            )
        )
        p_obj = Path(raw_wp_path)
        resolved_wp = p_obj if p_obj.is_absolute() else (Path(__file__).resolve().parents[2] / p_obj)
        wp = load_active_profile(resolved_wp)
        return cls.from_profile(
            wp,
            api_url=os.environ.get("RAWTREE_API_URL", defaults.api_url).strip() or defaults.api_url,
            team_id=os.environ.get("TEAM_ID", defaults.team_id).strip(),
            api_key=os.environ.get("RAWTREE_API_KEY", defaults.api_key).strip(),
            model_version=os.environ.get("MODEL_VERSION", defaults.model_version).strip()
            or defaults.model_version,
            weights_profile_path=raw_wp_path,
            v3_no_subcat_floor=_env_bool("CALYPSO_V3_NO_SUBCAT_FLOOR", wp.v3_no_subcat_floor),
            v3_bg_penalty=_env_bool("CALYPSO_V3_BG_PENALTY", wp.v3_bg_penalty),
            v3_ease_weight=_env_float("CALYPSO_V3_EASE_WEIGHT", wp.v3_ease_weight),
            v4_cart_priority=_env_bool("CALYPSO_V4_CART_PRIORITY", wp.v4_cart_priority),
            v4_cart_active_minutes=_env_float("CALYPSO_V4_CART_ACTIVE_MIN", wp.v4_cart_active_minutes),
            v4_cart_max_slots=int(_env_float("CALYPSO_V4_CART_MAX_SLOTS", float(wp.v4_cart_max_slots))),
            v4_forgotten_cart=_env_bool("CALYPSO_V4_FORGOTTEN_CART", wp.v4_forgotten_cart),
            v4_forgotten_max_slots=int(_env_float("CALYPSO_V4_FORGOTTEN_MAX_SLOTS", float(wp.v4_forgotten_max_slots))),
            v4_forgotten_max_days=_env_float("CALYPSO_V4_FORGOTTEN_MAX_DAYS", wp.v4_forgotten_max_days),
            v4_price_ascending=_env_bool("CALYPSO_V4_PRICE_ASC", wp.v4_price_ascending),
            v4_recurrent=_env_bool("CALYPSO_V4_RECURRENT", wp.v4_recurrent),
            v4_recurrent_min_purchases=int(_env_float("CALYPSO_V4_RECURRENT_MIN", float(wp.v4_recurrent_min_purchases))),
            v4_recurrent_max_slots=int(_env_float("CALYPSO_V4_RECURRENT_MAX_SLOTS", float(wp.v4_recurrent_max_slots))),
            v4_live_event_purchases=_env_bool("CALYPSO_V4_LIVE_EVENT_PURCHASES", wp.v4_live_event_purchases),
        )
