"""Índice de contenido para productos nuevos (C1, C3, C4).

Diferencias con la Etapa 1:
* La similitud es **absoluta** (en [0, 1]), no normalizada por el máximo de cada producto:
  así un producto poco parecido no recibe nunca peso 1,0.
* Se exige una similitud mínima y, por defecto, la misma categoría.
* `transform` de un producto nuevo cuesta milisegundos, así que los vecinos se calculan
  en el momento en que llega el `new_product` (C1), sin esperar a una recarga.
* Los pesos de los bonus son parámetros para poder compararlos en el test C4.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class SimWeights:
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
    def __init__(self, max_features: int = 8192) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                          min_df=1, sublinear_tf=True)
        self.fitted = False

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "ContentIndex":
        texts = [product_text(r) for r in rows] or ["empty"]
        self.vectorizer.fit(texts)
        self.fitted = True
        return self

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> sp.csr_matrix:
        if not rows:
            return sp.csr_matrix((0, len(self.vectorizer.vocabulary_)), dtype=np.float32)
        return self.vectorizer.transform([product_text(r) for r in rows]).astype(np.float32).tocsr()


def content_neighbors(
    q_vecs: sp.csr_matrix,
    q_cat: np.ndarray, q_subcat: np.ndarray, q_brand: np.ndarray, q_logp: np.ndarray,
    pool_vecs: sp.csr_matrix,
    pool_cat: np.ndarray, pool_subcat: np.ndarray, pool_brand: np.ndarray, pool_logp: np.ndarray,
    *,
    top_k: int = 8,
    min_sim: float = 0.15,
    same_category: bool = True,
    weights: SimWeights = SimWeights(),
    exclude_pool_idx: Sequence[int] | None = None,
) -> list[list[tuple[int, float]]]:
    """Para cada consulta, devuelve [(índice_en_pool, similitud_absoluta)] ordenado."""
    if q_vecs.shape[0] == 0 or pool_vecs.shape[0] == 0:
        return [[] for _ in range(q_vecs.shape[0])]
    cos = (q_vecs @ pool_vecs.T).toarray() if weights.text else np.zeros((q_vecs.shape[0], pool_vecs.shape[0]))
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
        if exclude_pool_idx is not None and exclude_pool_idx[r] is not None and exclude_pool_idx[r] >= 0:
            s[exclude_pool_idx[r]] = -1.0
        k = min(top_k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        out.append([(int(j), float(s[j])) for j in top if s[j] >= min_sim])
    return out
