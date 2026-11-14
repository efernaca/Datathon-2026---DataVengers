"""Content-based Cold Start index for new and low-interaction products."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from .state import ProductInfo


class ColdStartIndex:
    """Maps new or sparse products to their most similar warm catalog neighbors.

    Combines TF-IDF text similarity (category, subcategory, brand, description)
    with structured boosts for exact subcategory/category/brand match and price proximity.
    """

    def __init__(self, max_features: int = 4096) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._warm_ids: list[str] = []
        self._warm_matrix: np.ndarray | None = None
        self._warm_cats: list[str] = []
        self._warm_subcats: list[str] = []
        self._warm_brands: list[str] = []
        self._warm_log_prices: np.ndarray = np.empty(0, dtype=np.float32)

    @staticmethod
    def _product_text(p: ProductInfo) -> str:
        return f"{p.category} {p.subcategory} {p.brand} {p.description}".strip()

    def fit(self, products: Mapping[str, ProductInfo], warm_ids: Iterable[str]) -> ColdStartIndex:
        """Fit vectorizer on the full catalog and store normalized vectors for warm items."""
        all_products = list(products.values())
        if not all_products:
            self._warm_ids = []
            self._warm_matrix = None
            return self

        self.vectorizer.fit([self._product_text(p) for p in all_products])

        warm_list = [pid for pid in warm_ids if pid in products and products[pid].is_active]
        if not warm_list:
            warm_list = [pid for pid, info in products.items() if info.is_active]

        self._warm_ids = warm_list
        if not warm_list:
            self._warm_matrix = None
            return self

        warm_prods = [products[pid] for pid in warm_list]
        self._warm_matrix = self.vectorizer.transform(
            [self._product_text(p) for p in warm_prods]
        ).astype(np.float32)
        self._warm_cats = [p.category for p in warm_prods]
        self._warm_subcats = [p.subcategory for p in warm_prods]
        self._warm_brands = [p.brand for p in warm_prods]
        self._warm_log_prices = np.log1p(
            np.array([max(p.price_eur, 0.0) for p in warm_prods], dtype=np.float32)
        )
        return self

    def neighbors_for_batch(
        self,
        targets: Iterable[ProductInfo],
        top_k: int = 8,
    ) -> dict[str, tuple[tuple[str, float], ...]]:
        """Compute top-K warm neighbors and normalized similarity scores for target products."""
        target_list = list(targets)
        if not target_list or self._warm_matrix is None or not self._warm_ids:
            return {}

        q_mat = self.vectorizer.transform([self._product_text(p) for p in target_list]).astype(np.float32)
        # TF-IDF rows are L2-normalized by sklearn, so dot product == cosine similarity
        cos_sim = (q_mat @ self._warm_matrix.T).toarray()

        warm_cats = np.array(self._warm_cats, dtype=object)
        warm_subcats = np.array(self._warm_subcats, dtype=object)
        warm_brands = np.array(self._warm_brands, dtype=object)

        result: dict[str, tuple[tuple[str, float], ...]] = {}
        n_warm = len(self._warm_ids)
        k = min(top_k + 1, n_warm)

        for idx, p in enumerate(target_list):
            scores = cos_sim[idx].copy()
            scores += 0.35 * (warm_subcats == p.subcategory)
            scores += 0.15 * (warm_cats == p.category)
            if p.brand:
                scores += 0.10 * (warm_brands == p.brand)

            q_log_price = float(np.log1p(max(p.price_eur, 0.0)))
            price_sim = np.exp(-np.abs(self._warm_log_prices - q_log_price))
            scores += 0.15 * price_sim

            # Fast partial sort for top-K
            if k < n_warm:
                top_idx = np.argpartition(scores, -k)[-k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            else:
                top_idx = np.argsort(scores)[::-1]

            neighbors: list[tuple[str, float]] = []
            max_score = 0.0
            for j in top_idx:
                nb_id = self._warm_ids[int(j)]
                if nb_id == p.product_id:
                    continue
                val = float(scores[int(j)])
                if val <= 0.0:
                    continue
                if not neighbors:
                    max_score = val
                neighbors.append((nb_id, val / max(max_score, 1e-6)))
                if len(neighbors) >= top_k:
                    break
            result[p.product_id] = tuple(neighbors)

        return result
