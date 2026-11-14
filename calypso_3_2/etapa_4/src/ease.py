"""Closed-form EASE (Embarrassingly Shallow Autoencoders) item-item matrix for Calypso 3.2."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Set as AbstractSet
import math
from typing import Any

import numpy as np


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
