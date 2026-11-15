"""Closed-form EASE, co-visit fusion, hub-regularized normalization, and 2-hop graph diffusion for Calypso 6.2."""
from __future__ import annotations

from collections import defaultdict
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
    lam: float = 240.0,
    top_k: int = 35,
    max_items: int = 6000,
    covisit_edges: Iterable[tuple[str, str, float]] | None = None,
    covisit_blend_weight: float = 0.30,
    hop2_top_k: int = 24,
    return_hop2: bool = False,
) -> dict[str, tuple[tuple[str, float], ...]] | tuple[
    dict[str, tuple[tuple[str, float], ...]],
    dict[str, tuple[tuple[str, float], ...]],
]:
    """Compute closed-form EASE B = -P / diag(P) fused with co-visits, plus optional 2-hop diffusion."""
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

    if covisit_edges is not None and covisit_blend_weight > 0.0:
        max_cov = 0.0
        cov_buf: list[tuple[str, str, float]] = []
        for src, dst, raw_n in covisit_edges:
            if not src or not dst or src == dst:
                continue
            if src not in active_ids or dst not in active_ids:
                continue
            val = max(float(raw_n), 0.0)
            if val > 0.0:
                if val > max_cov:
                    max_cov = val
                cov_buf.append((src, dst, val))
        scale = float(covisit_blend_weight) * (12.0 / max(math.log1p(max_cov), 1.0))
        for src, dst, val in cov_buf:
            a, b = (src, dst) if src < dst else (dst, src)
            pair_weights[(a, b)] = pair_weights.get((a, b), 0.0) + 0.5 * scale * math.log1p(val)
            connected_items.add(src)
            connected_items.add(dst)
            item_diag.setdefault(src, 1.0)
            item_diag.setdefault(dst, 1.0)

    if len(connected_items) < 2 or not pair_weights:
        return ({}, {}) if return_hop2 else {}

    ranked_items = sorted(connected_items, key=lambda pid: (-item_diag.get(pid, 1.0), pid))[:max_items]
    n = len(ranked_items)
    if n < 2:
        return ({}, {}) if return_hop2 else {}

    idx = {pid: k for k, pid in enumerate(ranked_items)}
    G = np.zeros((n, n), dtype=np.float64)
    row_sums = np.zeros(n, dtype=np.float64)

    for (a, b), w in pair_weights.items():
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        # Soft hub-degree dampening so super-popular items do not monopolize neighbor lists
        damp = 1.0 / ((max(item_diag.get(a, 1.0), 1.0) * max(item_diag.get(b, 1.0), 1.0)) ** 0.04)
        w_adj = w * damp
        G[ia, ib] = w_adj
        G[ib, ia] = w_adj
        row_sums[ia] += w_adj
        row_sums[ib] += w_adj

    for k, pid in enumerate(ranked_items):
        # Ensure diagonal dominates off-diagonal energy + ridge lam so G is SPD
        G[k, k] = max(item_diag.get(pid, 1.0), 0.25 * row_sums[k] + 1.0) + lam

    try:
        P = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return ({}, {}) if return_hop2 else {}

    diag_p = np.diag(P)
    safe_diag = np.where(np.abs(diag_p) > 1e-12, diag_p, 1.0)
    B = -P / safe_diag[None, :]
    np.fill_diagonal(B, 0.0)
    B_pos = np.maximum(B, 0.0)

    out: dict[str, tuple[tuple[str, float], ...]] = {}
    kk = min(top_k, n - 1)
    if kk <= 0:
        return (out, {}) if return_hop2 else out

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

    if not return_hop2:
        return out

    # Fast sparse 2-hop diffusion over normalized top-k EASE edges (excluding direct top-16)
    hop2_out: dict[str, tuple[tuple[str, float], ...]] = {}
    hk = max(1, int(hop2_top_k))
    for src_pid, nbs in out.items():
        acc: dict[str, float] = defaultdict(float)
        direct_set = {dst for dst, _ in nbs[:16]}
        for mid_pid, w1 in nbs[:20]:
            for dst_pid, w2 in out.get(mid_pid, ())[:16]:
                if dst_pid == src_pid or dst_pid in direct_set:
                    continue
                acc[dst_pid] += w1 * w2
        if not acc:
            continue
        sorted_h2 = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:hk]
        max_h2 = sorted_h2[0][1] if sorted_h2 else 0.0
        if max_h2 > 0.0:
            hop2_out[src_pid] = tuple((dst, float(sc / max_h2)) for dst, sc in sorted_h2)

    return out, hop2_out
