"""Estructuras de estado inmutables y cambio atómico de snapshot.

El snapshot se compone de tres piezas que actualizan hilos distintos:

* `HistoryBundle`  → histórico de 60 días. Se calcula una vez y se guarda en disco (F5).
* `CatalogState`   → catálogo, stock y precios alineados en arrays NumPy. Lo reconstruye
                     el hilo de catálogo cada vez que llega un cambio (F4, C1).
* `LiveView`       → popularidad y tendencia en vivo con decaimiento (D1, V5, V6). Lo
                     reconstruye el hilo de eventos.

El bucle rápido solo lee `store.get()`; nunca bloquea a los hilos de fondo.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass(slots=True)
class UserProfile:
    """V1: afinidad por subcategoría y marca, rango de precio y compras previas."""

    subcat: dict[str, float] = field(default_factory=dict)   # normalizado a [0, 1]
    brand: dict[str, float] = field(default_factory=dict)
    price_mu: float = float("nan")      # media de log1p(precio) de sus compras
    price_sigma: float = float("nan")
    n_buys: int = 0
    bought: frozenset[str] = frozenset()


@dataclass(slots=True)
class EaseModel:
    """C: matriz ítem-ítem B (n×n, float32) con diagonal 0 sobre los productos con historial."""

    item_ids: list[str]
    B: np.ndarray
    lam: float = 0.0

    @property
    def index(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.item_ids)}


@dataclass(slots=True)
class HistoryBundle:
    """Todo lo que sale del histórico. Se serializa entero con pickle."""

    catalog_rows: list[dict[str, Any]]
    pop: dict[str, float]                                   # popularidad orgánica ponderada
    item_stats: dict[str, tuple[float, float, float, float]]  # pid -> (imps, imps_w, clicks_w, buys_w) IPS
    ctx_prior: dict[tuple[str, str, str], dict[str, float]] # V4: compras IPS por contexto
    transitions: dict[str, list[tuple[str, float]]]         # V3: lift dirigido y ordenado en el tiempo
    ease: EaseModel | None
    users: dict[str, UserProfile]
    meta: dict[str, Any] = field(default_factory=dict)
    ok_components: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogState:
    """Catálogo actual alineado por índice. Los índices de un producto no cambian nunca:
    los productos nuevos se añaden al final (append-only)."""

    version: int
    ids: tuple[str, ...]
    idx: dict[str, int]
    category: np.ndarray        # int32 códigos
    subcategory: np.ndarray
    brand: np.ndarray
    cat_names: tuple[str, ...]
    subcat_names: tuple[str, ...]
    brand_codes: dict[str, int]
    subcat_codes: dict[str, int]
    price: np.ndarray           # float64
    log_price: np.ndarray
    price_rel_z: np.ndarray     # log-precio relativo dentro de la subcategoría
    active: np.ndarray          # bool: is_active y con stock
    is_new: np.ndarray          # bool: dado de alta durante el evento
    active_ids: frozenset[str]
    global_ranked: tuple[str, ...]   # fallback: populares activos
    # Señales históricas alineadas
    log_pop_prior: np.ndarray   # log(share * N) del histórico (heredado para nuevos)
    pop_share_prior: np.ndarray # share que suma 1 sobre el catálogo
    subcat_log_pop: np.ndarray
    cvr_logit: np.ndarray       # CVR IPS suavizada hacia la subcategoría (identidad)
    ctx_arrays: dict[tuple[str, str, str], np.ndarray]
    ease_proj: sp.csr_matrix | None  # n_items × n_ease: identidad (con historial) o vecinos (nuevos)
    content: sp.csr_matrix      # TF-IDF L2-normalizado
    cold_neighbors: dict[str, tuple[tuple[str, float], ...]]
    price_change_rel: np.ndarray   # V6: (precio_anterior - precio_nuevo)/precio_anterior del último cambio
    price_change_ts: np.ndarray    # epoch s del último cambio (0 si nunca)
    created_at: float = 0.0

    @property
    def n(self) -> int:
        return len(self.ids)


@dataclass(frozen=True, slots=True)
class LiveView:
    """D1/V5: arrays en vivo alineados con un `CatalogState` concreto (por longitud)."""

    n_items: int
    log_pop: np.ndarray       # log(share_vivo_suavizado * N)
    trend: np.ndarray         # log((corto + κ) / (esperado + κ))
    total_weight: float = 0.0
    now_ts: float = 0.0
    kappa_mult: float = 1.0


@dataclass(frozen=True, slots=True)
class Snapshot:
    history: HistoryBundle
    catalog: CatalogState
    live: dict[str, LiveView] = field(default_factory=dict)   # una vista por brazo del bandit
    live_bought: dict[str, frozenset[str]] = field(default_factory=dict)
    new_impressions: dict[str, int] = field(default_factory=dict)


class StateStore:
    def __init__(self, initial: Snapshot | None = None) -> None:
        self._lock = threading.Lock()
        self._snap = initial

    def get(self) -> Snapshot | None:
        return self._snap   # lectura atómica de una referencia

    def set(self, snap: Snapshot) -> None:
        with self._lock:
            self._snap = snap

    def update(self, **changes: Any) -> Snapshot | None:
        """Sustituye solo los componentes indicados (p. ej. catalog=..., live=...)."""
        with self._lock:
            if self._snap is None:
                return None
            self._snap = replace(self._snap, **changes)
            return self._snap

    def mutate(self, fn) -> Snapshot | None:
        """Lectura-modificación-escritura atómica: `fn(snap) -> dict de cambios`."""
        with self._lock:
            if self._snap is None:
                return None
            changes = fn(self._snap)
            if changes:
                self._snap = replace(self._snap, **changes)
            return self._snap


def now_s() -> float:
    return time.time()
