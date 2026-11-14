"""Collaborative TruncatedSVD item & request latent embeddings for Calypso 6.1."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Set as AbstractSet
import math
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .state import ProductInfo


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        res = float(val)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(res) else res


def build_collaborative_svd_index(
    products: Mapping[str, ProductInfo],
    active_ids: AbstractSet[str],
    *,
    transition_rows: Iterable[Mapping[str, Any]] = (),
    covisit_edges: Iterable[tuple[str, str, float]] = (),
    user_intent_rows: Iterable[Mapping[str, Any]] = (),
    cart_hist_rows: Iterable[Mapping[str, Any]] = (),
    recurrent_hist_rows: Iterable[Mapping[str, Any]] = (),
    svd_dim: int = 32,
    max_interaction_cols: int = 24000,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build L2-normalized float32 collaborative + attribute SVD item embeddings."""
    ordered_ids = sorted(active_ids)
    n_items = len(ordered_ids)
    if n_items < 8:
        return np.empty((0, 0), dtype=np.float32), {}

    pid_to_row = {pid: idx for idx, pid in enumerate(ordered_ids)}

    rows_idx: list[int] = []
    cols_idx: list[int] = []
    data_vals: list[float] = []
    col_map: dict[tuple[str, str], int] = {}

    def _get_col(kind: str, key: str) -> int | None:
        tk = (kind, key)
        c = col_map.get(tk)
        if c is not None:
            return c
        if len(col_map) >= max_interaction_cols:
            return None
        c = len(col_map)
        col_map[tk] = c
        return c

    # 1. Catalog taxonomy smoothing columns so every active item has non-zero support
    for pid in ordered_ids:
        p = products.get(pid)
        if p is None:
            continue
        r = pid_to_row[pid]
        for kind, val, weight in (
            ("cat", p.category, 0.35),
            ("sub", p.subcategory, 0.70),
            ("brd", p.brand, 0.45),
        ):
            if val:
                c = _get_col(kind, val)
                if c is not None:
                    rows_idx.append(r)
                    cols_idx.append(c)
                    data_vals.append(weight)

    # 2. Item-item transition & covisit columns
    for row in transition_rows:
        src = str(row.get("src_product") or "").strip()
        dst = str(row.get("dst_product") or "").strip()
        r_src = pid_to_row.get(src)
        if r_src is None or dst not in pid_to_row or src == dst:
            continue
        sc = max(_safe_float(row.get("raw_transition_score", row.get("score")), 0.0), 0.0)
        if sc <= 0.0:
            continue
        c = _get_col("tr", dst)
        if c is not None:
            rows_idx.append(r_src)
            cols_idx.append(c)
            data_vals.append(math.log1p(sc))

    for src, dst, n_cov in covisit_edges:
        r_src = pid_to_row.get(src)
        if r_src is None or dst not in pid_to_row or src == dst:
            continue
        if n_cov > 0.0:
            c = _get_col("cv", dst)
            if c is not None:
                rows_idx.append(r_src)
                cols_idx.append(c)
                data_vals.append(0.8 * math.log1p(float(n_cov)))

    # 3. User-item interactions (carts, recurrents, intents) — strong collaborative signal
    for row in recurrent_hist_rows:
        uid = str(row.get("user_id") or "").strip()
        pid = str(row.get("product_id") or "").strip()
        r = pid_to_row.get(pid)
        if not uid or r is None:
            continue
        nb = max(_safe_float(row.get("n_purchases"), 1.0), 1.0)
        c = _get_col("u", uid)
        if c is not None:
            rows_idx.append(r)
            cols_idx.append(c)
            data_vals.append(1.8 + 0.6 * math.log1p(nb))

    for row in cart_hist_rows:
        uid = str(row.get("user_id") or "").strip()
        pid = str(row.get("product_id") or "").strip()
        r = pid_to_row.get(pid)
        if not uid or r is None:
            continue
        nb = max(_safe_float(row.get("n_purchases"), 0.0), 0.0)
        last_c = max(_safe_float(row.get("last_cart_ms"), 0.0), 0.0)
        last_b = max(_safe_float(row.get("last_purchase_ms"), 0.0), 0.0)
        w = 1.6 if (nb > 0.0 or last_b >= last_c > 0.0) else 0.75
        c = _get_col("u", uid)
        if c is not None:
            rows_idx.append(r)
            cols_idx.append(c)
            data_vals.append(w)

    for row in user_intent_rows:
        uid = str(row.get("user_id") or "").strip()
        pid = str(row.get("product_id") or "").strip()
        r = pid_to_row.get(pid)
        if not uid or r is None:
            continue
        sc = max(_safe_float(row.get("intent_score"), 0.0), 0.0)
        if sc <= 0.0:
            continue
        c = _get_col("u", uid)
        if c is not None:
            rows_idx.append(r)
            cols_idx.append(c)
            data_vals.append(0.55 * min(math.log1p(sc), 3.0))

    n_cols = len(col_map)
    if n_cols < 8 or not data_vals:
        return np.empty((0, 0), dtype=np.float32), {}

    mat = csr_matrix(
        (np.asarray(data_vals, dtype=np.float32), (rows_idx, cols_idx)),
        shape=(n_items, n_cols),
    )
    n_comp = min(int(svd_dim), n_items - 1, n_cols - 1)
    if n_comp < 4:
        return np.empty((0, 0), dtype=np.float32), {}

    svd = TruncatedSVD(n_components=n_comp, n_iter=5, random_state=42)
    dense = svd.fit_transform(mat).astype(np.float32)
    normed = normalize(dense, norm="l2", copy=False).astype(np.float32)
    return normed, pid_to_row


def build_query_vector(
    weighted_items: Iterable[tuple[str, float]],
    svd_mat: np.ndarray,
    pid_to_idx: Mapping[str, int],
) -> np.ndarray | None:
    """Aggregate normalized float32 SVD embedding for a session or user profile."""
    if svd_mat.size == 0 or not pid_to_idx:
        return None
    acc = np.zeros(svd_mat.shape[1], dtype=np.float32)
    has_any = False
    for pid, w in weighted_items:
        if w <= 0.0:
            continue
        idx = pid_to_idx.get(pid)
        if idx is not None:
            acc += float(w) * svd_mat[idx]
            has_any = True
    if not has_any:
        return None
    norm = float(np.linalg.norm(acc))
    if norm <= 1e-8:
        return None
    acc /= norm
    return acc
