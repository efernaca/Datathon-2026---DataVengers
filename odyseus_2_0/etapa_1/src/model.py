"""Modelos de respuesta: lineal (logística) y GBM (LightGBM / sklearn).

Sin modelo entrenado se usa `LinearModel.default()`: pesos a mano en escala logit que
reproducen la lógica de la Etapa 1 sobre las variables nuevas. Es el "control" del A/B.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURES, FIDX

DEFAULT_WEIGHTS: dict[str, float] = {
    "slot_log": -0.5, "in_recent": 1.0, "recent_recency": 0.5, "is_last_viewed": -0.3,
    "same_subcat_last": 0.8, "same_cat_last": 0.2, "same_brand_last": 0.2,
    "session_subcat_share": 0.6, "ease": 1.5, "trans_lift": 1.0, "content_sim_last": 0.5,
    "price_dist_last": -0.2, "log_pop": 0.35, "subcat_log_pop": 0.1, "item_cvr": 0.3,
    "price_rel_z": -0.05, "ctx_pop": 0.6, "user_subcat_aff": 0.5, "user_brand_aff": 0.3,
    "user_price_fit": 0.3, "already_bought": -2.0, "x_product_same_subcat": 0.2,
    "x_product_in_recent": -0.2, "x_home_log_pop": 0.1, "x_nosession_ctx_pop": 0.4,
}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LinearModel:
    kind = "linear"

    def __init__(self, coef: dict[str, float], intercept: float, name: str = "linear",
                 mean: dict[str, float] | None = None, scale: dict[str, float] | None = None,
                 meta: dict[str, Any] | None = None, clip_lo: dict[str, float] | None = None,
                 clip_hi: dict[str, float] | None = None) -> None:
        self.name = name
        self.intercept = float(intercept)
        self.meta = meta or {}
        F = len(FEATURES)
        w = np.zeros(F)
        mu = np.zeros(F)
        sc = np.ones(F)
        for k, v in coef.items():
            if k in FIDX:
                w[FIDX[k]] = v
        for k, v in (mean or {}).items():
            if k in FIDX:
                mu[FIDX[k]] = v
        for k, v in (scale or {}).items():
            if k in FIDX and v > 0:
                sc[FIDX[k]] = v
        # Folding de la estandarización: logit = b + Σ w_j (x_j - μ_j)/σ_j
        self.w_eff = (w / sc).astype(np.float32)
        self.b_eff = float(self.intercept - np.sum(w * mu / sc))
        self.coef, self.mean, self.scale = coef, mean or {}, scale or {}
        # Riesgo 1 de la revisión: el log solo cubre productos populares. Recortamos cada
        # variable al rango visto en entrenamiento (p1–p99) para no extrapolar linealmente.
        self.clip_lo_d, self.clip_hi_d = clip_lo or {}, clip_hi or {}
        lo = np.full(F, -np.inf, dtype=np.float32)
        hi = np.full(F, np.inf, dtype=np.float32)
        for k, v in self.clip_lo_d.items():
            if k in FIDX:
                lo[FIDX[k]] = v
        for k, v in self.clip_hi_d.items():
            if k in FIDX:
                hi[FIDX[k]] = v
        self.lo, self.hi = lo, hi
        self.has_clip = bool(self.clip_lo_d or self.clip_hi_d)

    @classmethod
    def default(cls) -> "LinearModel":
        return cls(DEFAULT_WEIGHTS, intercept=-5.5, name="heuristic")

    def logit(self, X: np.ndarray) -> np.ndarray:
        if self.has_clip:
            X = np.clip(X, self.lo, self.hi)
        return X @ self.w_eff + self.b_eff

    def predict(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.logit(X))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"type": "linear", "name": self.name, "coef": self.coef,
                                    "intercept": self.intercept, "mean": self.mean,
                                    "scale": self.scale, "meta": self.meta,
                                    "clip_lo": self.clip_lo_d, "clip_hi": self.clip_hi_d}, indent=1))

    def importance(self) -> list[tuple[str, float]]:
        """|coef estandarizado|: cuánto mueve el logit una desviación típica."""
        return sorted(((k, abs(v)) for k, v in self.coef.items()), key=lambda x: -x[1])


class GBMModel:
    kind = "gbm"

    def __init__(self, model: Any, features: list[str], name: str = "gbm",
                 meta: dict[str, Any] | None = None) -> None:
        self.model = model
        self.features = list(features)
        self.cols = np.array([FIDX[f] for f in features])
        self.name = name
        self.meta = meta or {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = X[:, self.cols]
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(Xs)[:, 1]
        return np.asarray(self.model.predict(Xs))

    def logit(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.predict(X), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def save(self, path: Path) -> None:
        with path.open("wb") as fh:
            pickle.dump({"type": "gbm", "model": self.model, "features": self.features,
                         "name": self.name, "meta": self.meta}, fh)

    def importance(self) -> list[tuple[str, float]]:
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return []
        return sorted(zip(self.features, map(float, imp)), key=lambda x: -x[1])


def load_model(path: Path | str | None):
    if path is None:
        return LinearModel.default()
    p = Path(path)
    if not p.is_file():
        return LinearModel.default()
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        return LinearModel(d["coef"], d["intercept"], name=d.get("name", "linear"),
                           mean=d.get("mean"), scale=d.get("scale"), meta=d.get("meta"),
                           clip_lo=d.get("clip_lo"), clip_hi=d.get("clip_hi"))
    with p.open("rb") as fh:
        d = pickle.load(fh)
    return GBMModel(d["model"], d["features"], name=d.get("name", "gbm"), meta=d.get("meta"))
