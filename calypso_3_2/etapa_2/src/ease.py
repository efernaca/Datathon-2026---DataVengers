"""Closed-form EASE (Embarrassingly Shallow Autoencoders) item-item matrix for Etapa 2."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Set as AbstractSet
import logging
import math
import time
from typing import Any

import numpy as np

from .state import EaseModel

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        res = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(res) else res


def fit_ease_from_transitions(
    popularity_rows: Iterable[Mapping[str, Any]],
    transition_rows: Iterable[Mapping[str, Any]],
    active_ids: AbstractSet[str],
    *,
    lam: float = 250.0,
    top_k: int = 35,
    max_items: int = 6000,
) -> dict[str, tuple[tuple[str, float], ...]]:
    """Compute closed-form EASE top-k neighbors B = -P / diag(P) with diag(B) = 0."""
    item_diag: dict[str, float] = {}
    for row in popularity_rows:
        pid = str(row.get("product_id") or "").strip()
        if not pid or pid not in active_ids:
            continue
        views = max(_safe_float(row.get("views"), 0.0), 0.0)
        carts = max(_safe_float(row.get("carts"), 0.0), 0.0)
        purchases = max(_safe_float(row.get("purchases"), 0.0), 0.0)
        wpop = max(_safe_float(row.get("weighted_pop"), 0.0), 0.0)
        diag_v = max(views + 2.0 * carts + 4.0 * purchases, wpop, 1.0)
        item_diag[pid] = diag_v

    pair_weights: dict[tuple[str, str], float] = {}
    connected_items: set[str] = set()
    for row in transition_rows:
        src = str(row.get("src_product") or "").strip()
        dst = str(row.get("dst_product") or "").strip()
        if not src or not dst or src == dst:
            continue
        if src not in active_ids or dst not in active_ids:
            continue
        sc = max(_safe_float(row.get("raw_transition_score", row.get("score")), 0.0), 0.0)
        if sc <= 0.0:
            continue
        a, b = (src, dst) if src < dst else (dst, src)
        pair_weights[(a, b)] = pair_weights.get((a, b), 0.0) + 0.5 * sc
        connected_items.add(src)
        connected_items.add(dst)
        item_diag.setdefault(src, 1.0)
        item_diag.setdefault(dst, 1.0)

    if len(connected_items) < 2 or not pair_weights:
        return {}

    ranked_items = sorted(connected_items, key=lambda pid: (-item_diag.get(pid, 1.0), pid))[:max_items]
    n = len(ranked_items)
    if n < 2:
        return {}

    idx = {pid: k for k, pid in enumerate(ranked_items)}
    G = np.zeros((n, n), dtype=np.float64)
    row_sums = np.zeros(n, dtype=np.float64)

    for (a, b), w in pair_weights.items():
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        G[ia, ib] = w
        G[ib, ia] = w
        row_sums[ia] += w
        row_sums[ib] += w

    for k, pid in enumerate(ranked_items):
        # Ensure diagonal dominates off-diagonal energy + ridge lam so G is SPD
        G[k, k] = max(item_diag.get(pid, 1.0), 0.25 * row_sums[k] + 1.0) + lam

    try:
        P = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return {}

    diag_p = np.diag(P)
    safe_diag = np.where(np.abs(diag_p) > 1e-12, diag_p, 1.0)
    B = -P / safe_diag[None, :]
    np.fill_diagonal(B, 0.0)
    B_pos = np.maximum(B, 0.0)

    out: dict[str, tuple[tuple[str, float], ...]] = {}
    kk = min(top_k, n - 1)
    if kk <= 0:
        return out

    for i, src_pid in enumerate(ranked_items):
        row = B_pos[i]
        top_idx = np.argpartition(-row, kk)[:kk]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        max_v = float(row[top_idx[0]]) if len(top_idx) else 0.0
        if max_v <= 0.0:
            continue
        edges = tuple(
            (ranked_items[int(j)], float(row[int(j)] / max_v))
            for j in top_idx
            if row[int(j)] > 0.0
        )
        if edges:
            out[src_pid] = edges
    return out


def fit_ease(
    item_counts: Mapping[str, float],
    pair_rows: Iterable[Mapping[str, Any]],
    lam: float = 400.0,
    min_count: float = 1.0,
) -> EaseModel | None:
    """Closed-form EASE returning the dense EaseModel for Etapa 2 Especial."""
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
    logger.info("EASE adjusted: %d items, lam=%.0f, %.1fs", n, lam, time.perf_counter() - t0)
    return EaseModel(item_ids=items, B=B.astype(np.float32), lam=lam)


def ease_top_neighbors(model: EaseModel, k: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Top-k destinations per item from an EaseModel."""
    out: dict[str, list[tuple[str, float]]] = {}
    B = model.B
    kk = min(k, B.shape[1] - 1)
    if kk <= 0:
        return out
    for i, pid in enumerate(model.item_ids):
        row = B[i]
        top = np.argpartition(-row, kk)[:kk]
        top = top[np.argsort(-row[top])]
        out[pid] = [(model.item_ids[j], float(row[j])) for j in top if row[j] > 0]
    return out


def ease_from_neighbors(nbs: Mapping[str, list[tuple[str, float]]]) -> EaseModel:
    """Reconstruct sparse-top-k EaseModel B matrix from scratch neighbor records."""
    items = sorted(set(nbs) | {d for v in nbs.values() for d, _ in v})
    idx = {p: k for k, p in enumerate(items)}
    B = np.zeros((len(items), len(items)), dtype=np.float32)
    for src, v in nbs.items():
        for dst, w in v:
            B[idx[src], idx[dst]] = w
    return EaseModel(item_ids=items, B=B, lam=float("nan"))
