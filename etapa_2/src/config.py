"""Centralized configuration and .env / group_API.env loader for Etapa 2."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_files() -> list[Path]:
    """Load key=value pairs from .env and group_API.env without overwriting existing env vars."""
    here = Path(__file__).resolve()
    search_dirs = [
        Path.cwd(),
        here.parents[1],
        here.parents[2],
        here.parents[2] / "datathon-fme-2026-main",
        here.parents[2] / "datathon-fme-2026-main" / "datathon-fme-2026-main",
    ]
    filenames = (".env", "group_API.env")
    loaded: list[Path] = []
    seen_paths: set[Path] = set()
    for directory in search_dirs:
        for fname in filenames:
            path = (directory / fname).resolve()
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                clean_key = key.strip()
                clean_val = value.strip().strip("'\"")
                if clean_key:
                    os.environ.setdefault(clean_key, clean_val)
            loaded.append(path)
    return loaded


@dataclass(frozen=True, slots=True)
class ConfigV2:
    """Runtime settings and hyperparameters for Stage 2."""

    api_url: str = "https://api.rawtree.com"
    team_id: str = ""
    api_key: str = ""
    model_version: str = "etapa2-seq-mmr-v1"

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

    # Stage 2 algorithmic parameters
    recent_view_decay: float = 0.72
    time_decay_half_life_sec: float = 900.0
    lift_beta: float = 0.45
    lift_prior_pop: float = 30.0
    max_cold_neighbors: int = 8
    max_reverse_cold_per_warm: int = 2

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

    # Revenue tie-breaker & MMR diversification
    revenue_tie_alpha: float = 0.03
    mmr_start_slot: int = 3
    mmr_lambda_subcat: float = 0.15
    mmr_lambda_brand: float = 0.04

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
