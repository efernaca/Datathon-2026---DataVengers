"""Centralized configuration and .env loader for Stage 1."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_files() -> list[Path]:
    """Load key=value pairs from .env files into os.environ without overwriting existing vars."""
    here = Path(__file__).resolve()
    search_dirs = [
        Path.cwd(),
        here.parents[1],
        here.parents[2],
        here.parents[3],
        here.parents[2] / "datathon-fme-2026-main",
        here.parents[3] / "datathon-fme-2026-main",
        here.parents[3] / "datathon-fme-2026-main" / "datathon-fme-2026-main",
    ]
    filenames = (".env", "group_API.env")
    candidates = [d / fname for d in search_dirs for fname in filenames]
    loaded: list[Path] = []
    seen_paths: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen_paths or not resolved.is_file():
            continue
        seen_paths.add(resolved)
        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            clean_key = key.strip()
            clean_val = value.strip().strip("'\"")
            if clean_key:
                os.environ.setdefault(clean_key, clean_val)
        loaded.append(resolved)
    return loaded


@dataclass(frozen=True, slots=True)
class Config:
    """Runtime hyperparameters, timeouts, and polling intervals."""

    api_url: str = "https://api.rawtree.com"
    team_id: str = ""
    api_key: str = ""
    model_version: str = "etapa1-hybrid-v1"

    # Fast loop timings (strictly budgeted below 3,000 ms SLA)
    poll_interval_sec: float = 0.20
    request_batch_limit: int = 50
    seen_window_size: int = 20_000
    query_timeout_fast_sec: float = 1.2
    push_timeout_sec: float = 1.4

    # Slow loop intervals (background daemon thread)
    stock_refresh_sec: float = 2.0
    trends_refresh_sec: float = 20.0
    catalog_refresh_sec: float = 30.0
    history_refresh_sec: float = 180.0
    query_timeout_slow_sec: float = 30.0

    # Stage 1 recommendation hyperparameters
    recent_view_decay: float = 0.70
    max_cold_neighbors: int = 8
    weight_exact_recent_view: float = 0.75
    weight_coview: float = 1.15
    weight_cold_neighbor: float = 0.95
    weight_cold_graft: float = 0.70
    weight_subcategory_pop: float = 0.40
    weight_context_pop: float = 0.45
    weight_user_affinity: float = 0.35
    weight_live_trend: float = 0.25
    weight_price_drop: float = 0.20
    weight_global_pop: float = 0.15

    @classmethod
    def from_env(cls) -> Config:
        """Build Config after loading local .env files."""
        load_dotenv_files()
        defaults = cls()
        return cls(
            api_url=os.environ.get("RAWTREE_API_URL", defaults.api_url).rstrip("/"),
            team_id=os.environ.get("TEAM_ID", defaults.team_id),
            api_key=os.environ.get("RAWTREE_API_KEY", defaults.api_key),
            model_version=os.environ.get("MODEL_VERSION", defaults.model_version),
        )
