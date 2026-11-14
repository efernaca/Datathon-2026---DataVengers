"""Unified configuration for Calypso 3.2 Etapa 2 (Stage2Recommender + Etapa 2 Especial).

Combines Stage 2's ``ConfigV2`` (with N6–N13 improvements) and Etapa 2 Especial's
``Config`` / signal tables / segment nomenclature under ``model_version = "calypso-3.2"``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

ETAPA_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ETAPA_DIR.parent


def load_dotenv_files() -> list[Path]:
    """Load key=value pairs from `.env` and `group_API.env` (handles CRLF) without overwriting."""
    here = Path(__file__).resolve()
    search_dirs = [
        Path.cwd(),
        ETAPA_DIR,
        REPO_DIR,
        REPO_DIR.parent,
        here.parents[2] if len(here.parents) > 2 else REPO_DIR,
        here.parents[3] if len(here.parents) > 3 else REPO_DIR,
        REPO_DIR / "datathon-fme-2026-main",
    ]
    loaded: list[Path] = []
    seen_paths: set[Path] = set()
    for directory in search_dirs:
        for fname in (".env", "group_API.env"):
            path = (directory / fname).resolve()
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)
            for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
                line = raw_line.strip().replace("\r", "")
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                clean_key = key.strip().removeprefix("export ").strip()
                clean_val = value.strip().strip("'\"")
                if clean_key:
                    os.environ.setdefault(clean_key, clean_val)
            loaded.append(path)
    return loaded


DEFAULT_ATTENTION: tuple[float, ...] = (
    1.0, 0.82, 0.70, 0.61, 0.55, 0.50, 0.46, 0.43, 0.40, 0.38,
)

# Señales de la Etapa 2 (nombres corregidos, N13). El orden fija el vector de señales.
SIGNALS: tuple[str, ...] = (
    "visto_conv",       # P(compra | producto ya en recent_views), por producto
    "lift",             # transiciones dirigidas con lift (S4 / N6)
    "ease",             # EASE (señal extra para 4.2)
    "cold_vecino",      # vecinos de contenido de un producto visto frío (sentido directo)
    "cold_injerto",     # transiciones heredadas de esos vecinos
    "injerto_inverso",  # productos nuevos parecidos a lo visto (warm -> new)
    "subcat_sesion",    # popularidad en la subcategoría de la sesión
    "intencion",        # carritos/vistas sin comprar, con decaimiento por antigüedad (N11)
    "user_subcat",
    "user_marca",
    "precio_fit",       # encaje con el precio habitual del usuario
    "contexto",         # prior por contexto con suavizado bayesiano jerárquico (N12)
    "pop",              # popularidad (viva suavizada hacia el histórico)
    "tendencia",        # contadores con decaimiento (V5)
    "bajada_precio",    # cambio respecto al precio anterior, con decaimiento (V6)
)

# Pesos de la Etapa 2 original, como punto de partida (se sustituyen por los aprendidos, 4.2).
DEFAULT_WEIGHTS: dict[str, float] = {
    "visto_conv": 0.65,
    "lift": 1.35,
    "ease": 0.8,
    "cold_vecino": 1.0,
    "cold_injerto": 0.8,
    "injerto_inverso": 0.28,
    "subcat_sesion": 0.45,
    "intencion": 0.5,
    "user_subcat": 0.35,
    "user_marca": 0.25,
    "precio_fit": 0.20,
    "contexto": 0.45,
    "pop": 0.12,
    "tendencia": 0.30,
    "bajada_precio": 0.25,
}

SEGMENTS: tuple[str, ...] = (
    "conocido_sesion",
    "conocido_sin_sesion",
    "nuevo_sesion",
    "nuevo_sin_sesion",
    "producto_nuevo",
)

# Interruptores del experimento factorial (4.1): señal o componente que apagan
SWITCHES: tuple[str, ...] = ("mmr", "rev", "rg", "int", "pf", "self")


@dataclass(frozen=True, slots=True)
class ConfigV2:
    """Runtime settings and hyperparameters for Stage 2 (Calypso 3.2)."""

    api_url: str = "https://api.rawtree.com"
    team_id: str = ""
    api_key: str = ""
    model_version: str = "calypso-3.2"

    # Fast Serving Loop budget (< 3,000 ms SLA)
    poll_interval_sec: float = 0.20
    request_batch_limit: int = 50
    seen_window_size: int = 20_000
    query_timeout_fast_sec: float = 1.2
    push_timeout_sec: float = 1.4

    # Slow Background Updater intervals
    stock_refresh_sec: float = 2.0
    trends_refresh_sec: float = 20.0
    feedback_refresh_sec: float = 10.0
    history_refresh_sec: float = 180.0
    query_timeout_slow_sec: float = 35.0

    # Stage 2 Especial N6-N13 algorithmic parameters
    recent_view_decay: float = 0.72
    time_decay_half_life_sec: float = 624.0   # N13: semivida real (exp(-ln2 * dt / 624))
    lift_beta: float = 0.45
    lift_prior_pop: float = 30.0
    ctx_kappa: float = 50.0                   # N12: Empirical Bayes jeràrquic en 3 nivells
    max_cold_neighbors: int = 8
    max_reverse_cold_per_warm: int = 3
    reverse_min_sim: float = 0.45             # N7: llindar absolut
    reverse_same_subcat: bool = True          # N7: mateixa subcategoria

    weight_self_conversion: float = 0.65
    weight_coview_lift: float = 1.35
    weight_cold_neighbor: float = 1.00
    weight_cold_graft: float = 0.80
    weight_reverse_cold_graft: float = 0.28
    weight_intent_recovery: float = 0.50
    weight_subcategory_pop: float = 0.45
    weight_user_subcat: float = 0.35
    weight_user_brand: float = 0.25
    weight_user_price_fit: float = 0.20
    weight_context_pop: float = 0.45
    weight_live_trend: float = 0.30
    weight_price_drop: float = 0.25
    weight_global_pop: float = 0.12

    # N9 lexicographic revenue tie-breaker & N10 relative MMR diversification
    revenue_eps: float = 0.015
    revenue_tie_alpha: float = 0.03
    mmr_start_slot: int = 3
    mmr_start_slot_session: int = 5
    mmr_lambda_subcat: float = 0.15
    mmr_lambda_brand: float = 0.05

    @classmethod
    def from_env(cls) -> ConfigV2:
        """Build ConfigV2 after loading local .env / group_API.env files."""
        load_dotenv_files()
        defaults = cls()
        return cls(
            api_url=os.environ.get("RAWTREE_API_URL", defaults.api_url).rstrip("/"),
            team_id=os.environ.get("TEAM_ID", defaults.team_id),
            api_key=os.environ.get("RAWTREE_API_KEY", defaults.api_key),
            model_version=os.environ.get("MODEL_VERSION", defaults.model_version),
        )


@dataclass(frozen=True, slots=True)
class Config:
    """Configuración de la Etapa 2 · versión especial (Calypso 3.2)."""

    api_url: str = "https://api.rawtree.com"
    team_id: str = ""
    api_key: str = ""
    model_version: str = "calypso-3.2"

    # --- Bucle rápido ---
    poll_interval_sec: float = 0.15
    read_window_sec: int = 4
    query_timeout_fast_sec: float = 1.0
    push_timeout_sec: float = 1.2
    push_workers: int = 3
    retry_max_age_ms: float = 2200.0
    drop_age_ms: float = 2750.0
    seen_window_size: int = 50_000

    # --- Hilos ligeros del servidor ---
    catalog_refresh_sec: float = 0.5
    events_refresh_sec: float = 2.0
    feedback_refresh_sec: float = 5.0
    state_reload_check_sec: float = 10.0
    query_timeout_slow_sec: float = 300.0
    query_timeout_bg_sec: float = 3.0

    # --- Rutas ---
    cache_dir: str = str(ETAPA_DIR / "cache")
    state_dir: str = str(ETAPA_DIR / "state")
    log_dir: str = str(ETAPA_DIR / "logs")
    heartbeat_file: str = str(ETAPA_DIR / "cache" / "heartbeat")
    standby_takeover_sec: float = 2.0

    # --- Histórico ---
    ease_lambda: float = 400.0
    ease_min_pair_count: int = 2
    ease_max_items_per_session: int = 40
    transition_beta: float = 0.45
    transition_prior: float = 30.0
    transition_min_support: int = 3
    transition_topk: int = 40
    cvr_prior_strength: float = 50.0
    visto_prior_strength: float = 30.0
    intent_halflife_days: float = 7.0
    intent_limit_per_user: int = 10
    ctx_kappa: float = 50.0

    # --- Sesión ---
    recent_view_decay: float = 0.72
    recent_halflife_sec: float = 624.0

    # --- Cold start ---
    cold_top_k: int = 8
    cold_min_sim: float = 0.15
    cold_same_category: bool = True
    cold_inherit_shrink: float = 0.6
    reverse_min_sim: float = 0.45
    reverse_same_subcat: bool = True
    reverse_top_k: int = 3

    # --- Señales vivas ---
    live_halflife_short_min: float = 15.0
    live_halflife_long_min: float = 180.0
    live_pop_halflife_min: float = 45.0
    live_pop_kappa: float = 2000.0
    trend_kappa: float = 3.0
    price_change_tau_min: float = 120.0
    event_weights: tuple[tuple[str, float], ...] = (
        ("view", 1.0),
        ("add_to_cart", 3.0),
        ("purchase", 8.0),
    )

    # --- Política ---
    attention: tuple[float, ...] = DEFAULT_ATTENTION
    mmr_lambda_subcat: float = 0.15
    mmr_lambda_brand: float = 0.05
    mmr_keep_top_session: int = 5
    mmr_keep_top_empty: int = 3
    revenue_eps: float = 0.0

    # --- Experimentos ---
    experiments: bool = True
    shuffle_rate: float = 0.10
    final_mode_rate: float = 0.20
    switch_prob: float = 0.5

    # --- Aprendizaje de pesos ---
    learn_min_rows_segment: int = 20_000
    learn_val_share: float = 0.2

    @classmethod
    def from_env(cls, **overrides) -> Config:
        load_dotenv_files()
        d = cls()
        cfg = replace(
            d,
            api_url=os.environ.get("RAWTREE_API_URL", d.api_url).rstrip("/"),
            team_id=os.environ.get("TEAM_ID", d.team_id),
            api_key=os.environ.get("RAWTREE_API_KEY", d.api_key),
            model_version=os.environ.get("MODEL_VERSION", d.model_version),
        )
        att = ETAPA_DIR / "eda" / "attention.json"
        if att.is_file() and "attention" not in overrides:
            try:
                vals = tuple(float(x) for x in json.loads(att.read_text())["attention"])
                if len(vals) == 10 and vals[0] > 0:
                    cfg = replace(cfg, attention=vals)
            except (ValueError, KeyError):
                pass
        valid = {f.name for f in fields(cls)}
        return replace(
            cfg, **{k: v for k, v in overrides.items() if k in valid and v is not None}
        )

    def event_weight(self, event_type: str) -> float:
        for name, w in self.event_weights:
            if name == event_type:
                return w
        return 0.0
