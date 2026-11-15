"""Odyseus 2.0 · modelo residual: inferencia en NumPy puro (sin dependencias nuevas en serve.py).

`ml_model.json` lo escribe `ml_train.py`. Contiene la normalización de las variables, el recorte al
rango visto en entrenamiento y una o varias redes pequeñas (una por semilla; se promedian).

Al entrenar:  logit(compra) = b + s·prior + f(x)
Al servir:    score = prior + alpha · f(x) / s        (mismo orden que el logit cuando alpha = 1)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
MODEL_FILENAME = "ml_model.json"


class ResidualModel:
    """Conjunto de MLPs (o lineal) que devuelve la corrección aditiva sobre el prior."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.feature_names: list[str] = list(payload["feature_names"])
        self.mean = np.asarray(payload["mean"], dtype=np.float64)
        self.std = np.asarray(payload["std"], dtype=np.float64)
        self.lo = np.asarray(payload["clip_lo"], dtype=np.float64)
        self.hi = np.asarray(payload["clip_hi"], dtype=np.float64)
        self.prior_scale = float(payload["prior_scale"])
        self.alpha = float(payload.get("alpha", 1.0))
        self.name = str(payload.get("name", "ml"))
        self.nets: list[list[tuple[np.ndarray, np.ndarray]]] = [
            [(np.asarray(layer["W"], dtype=np.float64), np.asarray(layer["b"], dtype=np.float64)) for layer in net]
            for net in payload["nets"]
        ]
        if not self.nets or self.prior_scale <= 1e-6:
            raise ValueError("ml_model.json sin redes o con prior_scale no positivo")
        self.n_features = len(self.feature_names)

    @classmethod
    def load(cls, path: Path) -> "ResidualModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def raw_logit(self, X: np.ndarray) -> np.ndarray:
        """f(x) en unidades de logit de compra (salida 0 de la red), promediada entre semillas."""
        Z = (np.asarray(X, dtype=np.float64) - self.mean) / self.std
        Z = np.clip(Z, self.lo, self.hi)
        acc = np.zeros(Z.shape[0], dtype=np.float64)
        for net in self.nets:
            h = Z
            last = len(net) - 1
            for li, (W, b) in enumerate(net):
                h = h @ W + b
                if li < last:
                    np.maximum(h, 0.0, out=h)
            acc += h[:, 0]
        return acc / len(self.nets)

    def residual(self, X: np.ndarray) -> np.ndarray:
        """Corrección en unidades del prior de Calypso: alpha · f(x) / s. Nunca devuelve no finitos."""
        r = self.alpha * self.raw_logit(X) / self.prior_scale
        if not np.all(np.isfinite(r)):
            return np.zeros(len(r), dtype=np.float64)
        return r


class ModelHolder:
    """Recarga en caliente de ml_model.json por mtime. Si falta o falla, no hay modelo (se sirve el prior)."""

    def __init__(self, path: Path, expected_features: tuple[str, ...]) -> None:
        self.path = Path(path)
        self.expected = list(expected_features)
        self.model: ResidualModel | None = None
        self._mtime: float | None = None

    def refresh(self) -> ResidualModel | None:
        try:
            mtime = self.path.stat().st_mtime if self.path.is_file() else None
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return self.model
        self._mtime = mtime
        if mtime is None:
            if self.model is not None:
                logger.warning("[ML] %s ha desaparecido: se sirve solo el prior", self.path.name)
            self.model = None
            return None
        try:
            m = ResidualModel.load(self.path)
            if m.feature_names != self.expected:
                raise ValueError("las variables del modelo no coinciden con ml_features.ML_FEATURE_NAMES")
            self.model = m
            logger.info("[ML] Modelo residual cargado: %s | %d redes | alpha=%.2f | prior_scale=%.3f",
                        m.name, len(m.nets), m.alpha, m.prior_scale)
        except Exception as exc:
            logger.warning("[ML] No se pudo cargar %s (%s): se sirve solo el prior", self.path.name, exc)
            self.model = None
        return self.model
