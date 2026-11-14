"""Runtime configuration and .env / group_API.env loader for Stage 3 (calypso-2.1)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
class ConfigV3:
    """Immutable configuration and hyperparameters for calypso-2.1."""

    api_url: str = "https://api.rawtree.com"
    team_id: str = ""
    api_key: str = ""
    model_version: str = "calypso-3.1"

    # Fast / Slow serving timings
    poll_interval_s: float = 0.20
    slow_interval_s: float = 18.0
    max_requests_per_batch: int = 500

    # Candidate retrieval & sequential scoring
    candidate_pool_size: int = 90
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
    weight_forward_cold: float = 0.45
    weight_reverse_cold_graft: float = 0.30

    # Position-debiased ML Re-Ranker & Position-Aware Bandit
    ml_max_blend_alpha: float = 0.65
    ml_min_positives: int = 25
    ml_l2_c: float = 1.2
    bandit_prior_strength: float = 20.0
    bandit_ucb_alpha: float = 0.22
    bandit_cold_ucb_multiplier: float = 2.0
    bandit_exam_power: float = 0.75

    # Diversity / MMR guardrails (active on slots >= mmr_start_slot)
    mmr_start_slot_session: int = 5
    mmr_start_slot_home: int = 3
    mmr_subcat_lambda_session: float = 0.0
    mmr_subcat_lambda_home: float = 0.15
    mmr_brand_lambda: float = 0.04
    revenue_tie_weight: float = 0.03

    # Cold-start graph limits
    max_cold_neighbors: int = 8
    max_reverse_cold_per_warm: int = 3

    @classmethod
    def from_env(cls) -> ConfigV3:
        """Build ConfigV3 from environment variables and project env files."""
        load_dotenv_files()
        defaults = cls()
        return cls(
            api_url=os.environ.get("RAWTREE_API_URL", defaults.api_url).strip() or defaults.api_url,
            team_id=os.environ.get("TEAM_ID", defaults.team_id).strip(),
            api_key=os.environ.get("RAWTREE_API_KEY", defaults.api_key).strip(),
            model_version=os.environ.get("MODEL_VERSION", defaults.model_version).strip()
            or defaults.model_version,
        )
