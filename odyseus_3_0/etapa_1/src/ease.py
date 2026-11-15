"""EASE (Steck, 2019): modelo ítem-ítem con solución cerrada (enfoque C).

G = XᵀX (co-ocurrencias sesión × producto, agregadas en RawTree),
P = (G + λI)⁻¹,  B = −P / diag(P),  diag(B) = 0.
En vivo: scores = Σ_v w_v · B[v, :].
Un solo hiperparámetro (λ) y, por construcción, descuenta a los productos populares (V3).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .state import EaseModel

logger = logging.getLogger(__name__)


def fit_ease(
    item_counts: Mapping[str, float],
    pair_rows: Iterable[Mapping[str, Any]],
    lam: float = 400.0,
    min_count: float = 1.0,
) -> EaseModel | None:
    """`item_counts`: nº de sesiones por producto (diagonal de G).
    `pair_rows`: filas {i, j, c} con i < j (triángulo superior de G)."""
    items = sorted(p for p, c in item_counts.items() if c >= min_count)
    if len(items) < 2:
        return None
    idx = {p: k for k, p in enumerate(items)}
    n = len(items)
    t0 = time.perf_counter()
    G = np.zeros((n, n), dtype=np.float64)
    for p, c in item_counts.items():
        k = idx.get(p)
        if k is not None:
            G[k, k] = float(c)
    for r in pair_rows:
        a = idx.get(str(r["i"]))
        b = idx.get(str(r["j"]))
        if a is None or b is None or a == b:
            continue
        c = float(r["c"])
        G[a, b] += c
        G[b, a] += c
    G[np.diag_indices(n)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)[None, :]
    B[np.diag_indices(n)] = 0.0
    logger.info("EASE ajustado: %d productos, λ=%.0f, %.1fs", n, lam, time.perf_counter() - t0)
    return EaseModel(item_ids=items, B=B.astype(np.float32), lam=lam)


def ease_top_neighbors(model: EaseModel, k: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Top-k destinos por producto (para publicar en `team.scratch` y para la presentación)."""
    out: dict[str, list[tuple[str, float]]] = {}
    B = model.B
    kk = min(k, B.shape[1] - 1)
    for i, pid in enumerate(model.item_ids):
        row = B[i]
        top = np.argpartition(-row, kk)[:kk]
        top = top[np.argsort(-row[top])]
        out[pid] = [(model.item_ids[j], float(row[j])) for j in top if row[j] > 0]
    return out


def ease_from_neighbors(nbs: Mapping[str, list[tuple[str, float]]]) -> EaseModel:
    """Reconstruye una B dispersa (top-k) a partir de vecinos leídos de `team.scratch`."""
    items = sorted(set(nbs) | {d for v in nbs.values() for d, _ in v})
    idx = {p: k for k, p in enumerate(items)}
    B = np.zeros((len(items), len(items)), dtype=np.float32)
    for src, v in nbs.items():
        for dst, w in v:
            B[idx[src], idx[dst]] = w
    return EaseModel(item_ids=items, B=B, lam=float("nan"))
