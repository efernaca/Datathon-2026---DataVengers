"""Configuración centralizada y carga de credenciales (Etapa 1 · versión especial).

U2: se buscan `.env` y `group_API.env` (con o sin finales de línea CRLF de Windows).
Todos los hiperparámetros están aquí para que el README pueda referenciarlos.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

ETAPA_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ETAPA_DIR.parent


def load_dotenv_files() -> list[Path]:
    """Carga pares clave=valor sin sobrescribir variables ya definidas."""
    search_dirs = [Path.cwd(), ETAPA_DIR, REPO_DIR, REPO_DIR.parent, REPO_DIR / "datathon-fme-2026-main"]
    loaded: list[Path] = []
    seen: set[Path] = set()
    for d in search_dirs:
        for fname in (".env", "group_API.env"):
            path = (d / fname).resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            for raw in path.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip().replace("\r", "")
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip().removeprefix("export ").strip()
                if key:
                    os.environ.setdefault(key, val.strip().strip("'\""))
            loaded.append(path)
    return loaded


DEFAULT_ATTENTION = (1.0, 0.82, 0.70, 0.61, 0.55, 0.50, 0.46, 0.43, 0.40, 0.38)


@dataclass(frozen=True, slots=True)
class Config:
    api_url: str = "https://api.rawtree.com"
    team_id: str = ""
    api_key: str = ""
    model_version: str = "calypso-3.1"

    # --- Bucle rápido (F1–F3) ---
    poll_interval_sec: float = 0.15
    read_window_sec: int = 4            # S1: solo peticiones de los últimos N segundos
    query_timeout_fast_sec: float = 1.0
    push_timeout_sec: float = 1.2
    push_workers: int = 3
    retry_max_age_ms: float = 2200.0    # F1: reintentar solo si la petición más antigua es más joven
    drop_age_ms: float = 2750.0         # no tiene sentido responder después de esto
    seen_window_size: int = 50_000

    # --- Hilos de fondo (F4, C1, V2) ---
    catalog_refresh_sec: float = 0.5    # stock + catálogo incremental (S2)
    events_refresh_sec: float = 2.0     # shop.events incremental
    feedback_refresh_sec: float = 5.0   # team.feedback incremental
    history_refresh_sec: float = 0.0    # 0 = el histórico no se recalcula en vivo (F5)
    query_timeout_slow_sec: float = 300.0
    query_timeout_bg_sec: float = 3.0

    # --- Caché de histórico (F5) ---
    cache_dir: str = str(ETAPA_DIR / "cache")
    artifacts_dir: str = str(ETAPA_DIR / "artifacts")
    heartbeat_file: str = str(ETAPA_DIR / "cache" / "heartbeat")
    standby_takeover_sec: float = 2.0

    # --- Histórico / EASE (C, V3, V4) ---
    ease_lambda: float = 400.0
    ease_min_pair_count: int = 2
    ease_max_items_per_session: int = 40
    transition_beta: float = 0.5
    transition_min_support: int = 3
    transition_topk: int = 40
    cvr_prior_strength: float = 50.0

    # --- Cold start (C1, C3, C4) ---
    cold_top_k: int = 8
    cold_min_sim: float = 0.15
    cold_same_category: bool = True
    cold_inherit_shrink: float = 0.6    # los productos nuevos heredan el 60 % de la señal de sus vecinos

    # --- Señales en vivo (D1, V5, V6) ---
    live_halflife_short_min: float = 15.0
    live_halflife_long_min: float = 180.0
    live_pop_halflife_min: float = 45.0
    live_pop_kappa: float = 2000.0      # fuerza del prior histórico (en unidades de eventos ponderados)
    trend_kappa: float = 3.0
    price_change_tau_min: float = 120.0
    event_weights: tuple[tuple[str, float], ...] = (("view", 1.0), ("add_to_cart", 3.0), ("purchase", 8.0))

    # --- Política (A4, D2, D3, E) ---
    attention: tuple[float, ...] = DEFAULT_ATTENTION
    weight_trend: float = 0.15          # ajuste en logit, fuera del modelo (no hay tendencia en el histórico)
    weight_price_drop: float = 0.5
    revenue_tiebreak: float = 0.01
    diversify: str = "no_session"       # "never" | "no_session" | "always"   (E, depende de B6)
    explore_slots: tuple[int, ...] = (8, 10)  # 1-indexado, dentro de 7–10 (D3)
    explore_max_impressions: int = 300
    bandit_enabled: bool = True
    ab_treatment_share: float = 1.0     # hash(request_id): fracción con el modelo; el resto, heurística

    # Variantes de la mezcla vivo/histórico para Thompson (D2)
    bandit_arms: tuple[tuple[str, float, float, float], ...] = (
        # nombre, multiplicador de kappa, peso de tendencia, tasa de exploración
        ("hist", 4.0, 0.00, 0.00),
        ("mix", 1.0, 0.15, 0.10),
        ("live", 0.25, 0.30, 0.15),
        ("explore", 1.0, 0.15, 0.40),
    )

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        load_dotenv_files()
        d = cls()
        cfg = replace(
            d,
            api_url=os.environ.get("RAWTREE_API_URL", d.api_url).rstrip("/"),
            team_id=os.environ.get("TEAM_ID", d.team_id),
            api_key=os.environ.get("RAWTREE_API_KEY", d.api_key),
            model_version=os.environ.get("MODEL_VERSION", d.model_version),
        )
        att = ETAPA_DIR / "eda" / "attention.json"      # B1 → curva de atención medida
        if att.is_file() and "attention" not in overrides:
            try:
                import json

                vals = tuple(float(x) for x in json.loads(att.read_text())["attention"])
                if len(vals) == 10 and vals[0] > 0:
                    cfg = replace(cfg, attention=vals)
            except (ValueError, KeyError):
                pass
        valid = {f.name for f in fields(cls)}
        return replace(cfg, **{k: v for k, v in overrides.items() if k in valid and v is not None})

    def event_weight(self, event_type: str) -> float:
        for name, w in self.event_weights:
            if name == event_type:
                return w
        return 0.0
