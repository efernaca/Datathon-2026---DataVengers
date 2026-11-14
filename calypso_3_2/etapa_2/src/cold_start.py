"""Bidirectional Cold Start index (TF-IDF + TruncatedSVD + structured attributes) with N7 fixes."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

if TYPE_CHECKING:
    from .state import ProductInfo

_SIM_WEIGHT_TOTAL = 1.0 + 0.35 + 0.15 + 0.12 + 0.15


@dataclass(frozen=True)
class SimWeights:
    """Component weights for Etapa 2 Especial absolute content similarity."""

    text: float = 1.0
    subcat: float = 0.35
    cat: float = 0.15
    brand: float = 0.10
    price: float = 0.15

    @property
    def total(self) -> float:
        return self.text + self.subcat + self.cat + self.brand + self.price


def product_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(k, "") or "") for k in ("category", "subcategory", "brand", "description")
    ).strip()


class ContentIndex:
    """Fast incremental TF-IDF encoder used by Etapa 2 Especial CatalogManager."""

    def __init__(self, max_features: int = 8192) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.fitted = False

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> ContentIndex:
        texts = [product_text(r) for r in rows] or ["empty"]
        self.vectorizer.fit(texts)
        self.fitted = True
        return self

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> sp.csr_matrix:
        if not rows:
            return sp.csr_matrix((0, len(self.vectorizer.vocabulary_)), dtype=np.float32)
        return (
            self.vectorizer.transform([product_text(r) for r in rows])
            .astype(np.float32)
            .tocsr()
        )


def content_neighbors(
    q_vecs: sp.csr_matrix,
    q_cat: np.ndarray,
    q_subcat: np.ndarray,
    q_brand: np.ndarray,
    q_logp: np.ndarray,
    pool_vecs: sp.csr_matrix,
    pool_cat: np.ndarray,
    pool_subcat: np.ndarray,
    pool_brand: np.ndarray,
    pool_logp: np.ndarray,
    *,
    top_k: int = 8,
    min_sim: float = 0.15,
    same_category: bool = True,
    weights: SimWeights = SimWeights(),
    exclude_pool_idx: Sequence[int] | None = None,
) -> list[list[tuple[int, float]]]:
    """Absolute similarity in [0, 1] for query items against warm catalog pool (N7)."""
    if q_vecs.shape[0] == 0 or pool_vecs.shape[0] == 0:
        return [[] for _ in range(q_vecs.shape[0])]
    cos = (
        (q_vecs @ pool_vecs.T).toarray()
        if weights.text
        else np.zeros((q_vecs.shape[0], pool_vecs.shape[0]))
    )
    out: list[list[tuple[int, float]]] = []
    for r in range(q_vecs.shape[0]):
        s = weights.text * cos[r]
        s = s + weights.subcat * (pool_subcat == q_subcat[r])
        s = s + weights.cat * (pool_cat == q_cat[r])
        s = s + weights.brand * (pool_brand == q_brand[r])
        s = s + weights.price * np.exp(-np.abs(pool_logp - q_logp[r]))
        s = s / weights.total
        if same_category:
            s = np.where(pool_cat == q_cat[r], s, -1.0)
        if (
            exclude_pool_idx is not None
            and exclude_pool_idx[r] is not None
            and exclude_pool_idx[r] >= 0
        ):
            s[exclude_pool_idx[r]] = -1.0
        k = min(top_k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        out.append([(int(j), float(s[j])) for j in top if s[j] >= min_sim])
    return out


class ColdStartIndexV2:
    """Computes forward (new -> warm) and reverse (warm -> new) semantic neighbors."""

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

    def fit(self, products: Mapping[str, ProductInfo], warm_ids: Iterable[str]) -> ColdStartIndexV2:
        """Fit TF-IDF + SVD on the full catalog and index active warm products."""
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
        """Return (len(targets), len(warm_ids)) blended semantic + structured similarity matrix."""
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
            row /= _SIM_WEIGHT_TOTAL
            if p.category:
                row = np.where(self._warm_cats == p.category, row, 0.0)
                sim[idx] = row

        return sim

    def compute_bidirectional(
        self,
        forward_targets: Iterable[ProductInfo],
        new_products: Iterable[ProductInfo],
        *,
        top_k_forward: int = 8,
        top_k_reverse: int = 3,
        min_reverse_sim: float = 0.45,
        reverse_same_subcat: bool = True,
        min_similarity: float = 0.15,
    ) -> tuple[
        dict[str, tuple[tuple[str, float], ...]],
        dict[str, tuple[tuple[str, float], ...]],
    ]:
        """Compute forward (cold/sparse -> warm) and reverse (warm -> new_product) maps (N7)."""
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
            for j in top_idx:
                nb_id = self._warm_ids[int(j)]
                if nb_id == p.product_id:
                    continue
                val = float(row[int(j)])
                if val < min_similarity:
                    continue
                # N7: keep absolute similarity in [0, 1] rather than max-normalizing weak items to 1.0
                neighbors.append((nb_id, min(max(val, 0.0), 1.0)))
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
                attached: list[tuple[str, float]] = []
                for n_idx in top_n:
                    val = float(scores[int(n_idx)])
                    if val < min_reverse_sim:
                        continue
                    new_p = new_list[int(n_idx)]
                    if reverse_same_subcat and new_p.subcategory != self._warm_subcats[w_idx]:
                        continue
                    attached.append((new_p.product_id, min(max(val, 0.0), 1.0)))
                if attached:
                    reverse_map[warm_pid] = tuple(attached)

        return forward_map, reverse_map


def build_cold_start_scratch_records(
    cold_neighbors: Mapping[str, tuple[tuple[str, float], ...]],
    products: Mapping[str, ProductInfo],
    model_version: str,
    ts_iso: str,
    limit: int = 200,
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
