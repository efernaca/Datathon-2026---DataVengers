"""Bidirectional Cold Start semantic index (TF-IDF + TruncatedSVD + attributes)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

if TYPE_CHECKING:
    from .state import ProductInfo


class ColdStartIndexV4:
    """Computes forward (cold/sparse -> warm) and reverse (warm -> new_product) maps."""

    def __init__(self, max_features: int = 4096, svd_dim: int = 64) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.svd_dim = svd_dim
        self._svd: TruncatedSVD | None = None
        self._warm_ids: list[str] = []
        self._warm_sparse: Any = None
        self._warm_dense: np.ndarray | None = None
        self._warm_cats: np.ndarray = np.empty(0, dtype=object)
        self._warm_subcats: np.ndarray = np.empty(0, dtype=object)
        self._warm_brands: np.ndarray = np.empty(0, dtype=object)
        self._warm_log_prices: np.ndarray = np.empty(0, dtype=np.float32)

    @staticmethod
    def _product_text(p: ProductInfo) -> str:
        return f"{p.category} {p.subcategory} {p.brand} {p.description}".strip()

    def _encode_dense(self, sparse_mat: Any) -> np.ndarray | None:
        if self._svd is None:
            return None
        dense = self._svd.transform(sparse_mat).astype(np.float32)
        return normalize(dense, norm="l2", copy=False)

    def fit(self, products: Mapping[str, ProductInfo], warm_ids: Iterable[str]) -> ColdStartIndexV4:
        """Fit TF-IDF + TruncatedSVD on the catalog and index active warm products."""
        all_products = list(products.values())
        if not all_products:
            self._warm_ids = []
            self._warm_sparse = None
            self._warm_dense = None
            return self

        full_sparse = self.vectorizer.fit_transform(
            [self._product_text(p) for p in all_products]
        ).astype(np.float32)

        n_items, n_feats = full_sparse.shape
        n_comp = min(self.svd_dim, n_items - 1, n_feats - 1)
        if n_comp >= 8:
            self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self._svd.fit(full_sparse)
        else:
            self._svd = None

        warm_list = [pid for pid in warm_ids if pid in products and products[pid].is_active]
        if not warm_list:
            warm_list = [pid for pid, info in products.items() if info.is_active]

        self._warm_ids = warm_list
        if not warm_list:
            self._warm_sparse = None
            self._warm_dense = None
            return self

        warm_prods = [products[pid] for pid in warm_list]
        self._warm_sparse = self.vectorizer.transform(
            [self._product_text(p) for p in warm_prods]
        ).astype(np.float32)
        self._warm_dense = self._encode_dense(self._warm_sparse)
        self._warm_cats = np.array([p.category for p in warm_prods], dtype=object)
        self._warm_subcats = np.array([p.subcategory for p in warm_prods], dtype=object)
        self._warm_brands = np.array([p.brand for p in warm_prods], dtype=object)
        self._warm_log_prices = np.log1p(
            np.array([max(p.price_eur, 0.0) for p in warm_prods], dtype=np.float32)
        )
        return self

    def _similarity_matrix(self, targets: list[ProductInfo]) -> np.ndarray:
        q_sparse = self.vectorizer.transform(
            [self._product_text(p) for p in targets]
        ).astype(np.float32)
        sparse_cos = (q_sparse @ self._warm_sparse.T).toarray()

        if self._warm_dense is not None:
            q_dense = self._encode_dense(q_sparse)
            dense_cos = np.clip(q_dense @ self._warm_dense.T, 0.0, 1.0)
            sim = 0.65 * sparse_cos + 0.35 * dense_cos
        else:
            sim = sparse_cos

        for idx, p in enumerate(targets):
            row = sim[idx]
            row += 0.35 * (self._warm_subcats == p.subcategory)
            row += 0.15 * (self._warm_cats == p.category)
            if p.brand:
                row += 0.12 * (self._warm_brands == p.brand)
            q_log_p = float(np.log1p(max(p.price_eur, 0.0)))
            row += 0.15 * np.exp(-np.abs(self._warm_log_prices - q_log_p))

        return sim

    def compute_bidirectional(
        self,
        forward_targets: Iterable[ProductInfo],
        new_products: Iterable[ProductInfo],
        *,
        top_k_forward: int = 8,
        top_k_reverse: int = 3,
        min_reverse_ratio: float = 0.65,
        min_similarity: float = 0.08,
    ) -> tuple[
        dict[str, tuple[tuple[str, float], ...]],
        dict[str, tuple[tuple[str, float], ...]],
    ]:
        """Compute forward (cold/sparse -> warm) and reverse (warm -> new_product) maps."""
        if self._warm_sparse is None or not self._warm_ids:
            return {}, {}

        fwd_list = list(forward_targets)
        new_list = [p for p in new_products if p.is_active]

        seen_pids: set[str] = set()
        union_targets: list[ProductInfo] = []
        for p in fwd_list + new_list:
            if p.product_id not in seen_pids:
                seen_pids.add(p.product_id)
                union_targets.append(p)

        if not union_targets:
            return {}, {}

        sim_mat = self._similarity_matrix(union_targets)
        pid_to_row = {p.product_id: idx for idx, p in enumerate(union_targets)}
        n_warm = len(self._warm_ids)

        forward_map: dict[str, tuple[tuple[str, float], ...]] = {}
        k_f = min(top_k_forward + 1, n_warm)
        for p in fwd_list:
            row = sim_mat[pid_to_row[p.product_id]]
            if k_f < n_warm:
                top_idx = np.argpartition(row, -k_f)[-k_f:]
                top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            else:
                top_idx = np.argsort(row)[::-1]

            neighbors: list[tuple[str, float]] = []
            max_val = 0.0
            for j in top_idx:
                nb_id = self._warm_ids[int(j)]
                if nb_id == p.product_id:
                    continue
                val = float(row[int(j)])
                if val < min_similarity:
                    continue
                if not neighbors:
                    max_val = val
                neighbors.append((nb_id, val / max(max_val, 0.25)))
                if len(neighbors) >= top_k_forward:
                    break
            forward_map[p.product_id] = tuple(neighbors)

        reverse_map: dict[str, tuple[tuple[str, float], ...]] = {}
        if new_list:
            new_row_indices = [pid_to_row[p.product_id] for p in new_list]
            warm_to_new_mat = sim_mat[new_row_indices].T
            n_new = len(new_list)
            k_r = min(top_k_reverse, n_new)
            for w_idx, warm_pid in enumerate(self._warm_ids):
                scores = warm_to_new_mat[w_idx]
                top_n = (
                    np.argpartition(scores, -k_r)[-k_r:]
                    if k_r < n_new
                    else np.arange(n_new)
                )
                top_n = top_n[np.argsort(scores[top_n])[::-1]]
                max_s = float(scores[top_n[0]]) if len(top_n) else 0.0
                if max_s < min_similarity:
                    continue
                attached: list[tuple[str, float]] = []
                for n_idx in top_n:
                    val = float(scores[int(n_idx)])
                    if val < min_similarity:
                        continue
                    new_p = new_list[int(n_idx)]
                    if new_p.subcategory != self._warm_subcats[w_idx] and val < max_s * 0.9:
                        continue
                    norm_s = val / max(max_s, 0.25)
                    if norm_s >= min_reverse_ratio:
                        attached.append((new_p.product_id, norm_s))
                if attached:
                    reverse_map[warm_pid] = tuple(attached)

        return forward_map, reverse_map


def build_cold_start_scratch_records(
    cold_neighbors: Mapping[str, tuple[tuple[str, float], ...]],
    products: Mapping[str, ProductInfo],
    model_version: str,
    ts_iso: str,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """Format new-product neighbor mappings for persistence into team_NN.scratch."""
    records: list[dict[str, Any]] = []
    for pid, nbs in cold_neighbors.items():
        info = products.get(pid)
        if info is None or not info.is_new:
            continue
        records.append(
            {
                "kind": "cold_start_neighbors",
                "model_version": model_version,
                "ts": ts_iso,
                "product_id": pid,
                "subcategory": info.subcategory,
                "neighbors": [{"product_id": q, "score": round(s, 4)} for q, s in nbs[:5]],
            }
        )
        if len(records) >= limit:
            break
    return records
