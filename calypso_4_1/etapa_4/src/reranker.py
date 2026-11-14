"""Vectorized 14-feature Position-Debiased (IPS) ML Re-Ranker for Stage 4."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES: tuple[str, ...] = (
    "seq_lift_score",
    "self_cvr_view",
    "intent_score",
    "user_affinity",
    "price_compat",
    "ctx_bayes_score",
    "global_pop_norm",
    "live_trend_norm",
    "is_new_product",
    "reverse_graft_score",
    "forward_cold_score",
    "log_price_norm",
    "same_top_subcat",
    "same_top_brand",
)

NUM_FEATURES = len(FEATURE_NAMES)

DEFAULT_PRIOR_WEIGHTS = np.array(
    [
        1.05,  # seq_lift_score
        1.85,  # self_cvr_view
        0.52,  # intent_score
        0.25,  # user_affinity
        0.20,  # price_compat
        0.30,  # ctx_bayes_score
        0.22,  # global_pop_norm
        0.25,  # live_trend_norm
        0.02,  # is_new_product
        0.32,  # reverse_graft_score
        0.48,  # forward_cold_score
        0.03,  # log_price_norm
        0.10,  # same_top_subcat
        0.04,  # same_top_brand
    ],
    dtype=np.float64,
)

_MIN_PRIOR_RATIO = np.array(
    [
        0.72,  # seq_lift_score
        0.75,  # self_cvr_view
        0.65,  # intent_score
        0.55,  # user_affinity
        0.45,  # price_compat
        0.50,  # ctx_bayes_score
        0.25,  # global_pop_norm
        0.30,  # live_trend_norm
        0.00,  # is_new_product
        0.55,  # reverse_graft_score
        0.55,  # forward_cold_score
        0.00,  # log_price_norm
        0.45,  # same_top_subcat
        0.00,  # same_top_brand
    ],
    dtype=np.float64,
)


@dataclass(slots=True)
class PositionDebiasedRanker:
    """Blend of IPS-weighted LogisticRegression and monotonic domain priors."""

    prior_weights: np.ndarray
    effective_weights: np.ndarray
    ml_weights_raw: np.ndarray
    ml_blend_alpha: float = 0.0
    train_samples: int = 0
    positive_samples: int = 0

    @classmethod
    def from_priors(cls, prior_weights: np.ndarray | None = None) -> PositionDebiasedRanker:
        pw = (DEFAULT_PRIOR_WEIGHTS if prior_weights is None else prior_weights).astype(
            np.float64, copy=True
        )
        return cls(
            prior_weights=pw,
            effective_weights=pw.copy(),
            ml_weights_raw=pw.copy(),
            ml_blend_alpha=0.0,
            train_samples=0,
            positive_samples=0,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        *,
        max_blend_alpha: float = 0.45,
        min_positives: int = 25,
        l2_c: float = 1.2,
    ) -> PositionDebiasedRanker:
        """Fit standardized IPS LogisticRegression anchored to monotonic domain priors."""
        n_samples = int(X.shape[0]) if X.ndim == 2 else 0
        n_pos = int(np.sum(y > 0.5)) if n_samples > 0 else 0
        self.train_samples = n_samples
        self.positive_samples = n_pos

        if n_samples < 40 or n_pos < min_positives or len(np.unique(y)) < 2:
            self.ml_blend_alpha = 0.0
            self.effective_weights = self.prior_weights.copy()
            return self

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X.astype(np.float64, copy=False))
        clf = LogisticRegression(
            C=l2_c,
            solver="lbfgs",
            max_iter=400,
            random_state=42,
        )
        clf.fit(X_std, y.astype(np.int64, copy=False), sample_weight=sample_weights)

        scales = np.maximum(scaler.scale_, 1e-3)
        raw_ml = clf.coef_[0].astype(np.float64, copy=True) / scales

        pos_norm = float(np.sum(np.abs(self.prior_weights)))
        ml_norm = float(np.sum(np.maximum(raw_ml, 0.0)))
        if ml_norm <= 1e-6:
            self.ml_blend_alpha = 0.0
            self.effective_weights = self.prior_weights.copy()
            return self

        rescaled_ml = raw_ml * (pos_norm / ml_norm)
        floors = _MIN_PRIOR_RATIO * self.prior_weights
        ceilings = np.maximum(self.prior_weights * 2.2, 0.30)
        stabilized_ml = np.clip(rescaled_ml, floors, ceilings)

        ramp = min(1.0, max(0.0, (n_pos - min_positives) / 350.0))
        alpha = max_blend_alpha * ramp
        self.ml_weights_raw = stabilized_ml
        self.ml_blend_alpha = float(alpha)
        self.effective_weights = (1.0 - alpha) * self.prior_weights + alpha * stabilized_ml
        return self

    def resolve_weights_tuple(
        self,
        *,
        blend_alpha_scale: float = 1.0,
        reverse_graft_override: float | None = None,
        forward_cold_override: float | None = None,
        revenue_tie_override: float | None = None,
    ) -> tuple[float, ...]:
        """Return effective 14 weights as a compact Python tuple for zero-alloc scoring."""
        alpha = min(1.0, max(0.0, self.ml_blend_alpha * blend_alpha_scale))
        if alpha <= 0.0:
            w = self.prior_weights.tolist()
        elif alpha == self.ml_blend_alpha:
            w = self.effective_weights.tolist()
        else:
            w = ((1.0 - alpha) * self.prior_weights + alpha * self.ml_weights_raw).tolist()

        if reverse_graft_override is not None:
            w[9] = float(reverse_graft_override)
        if forward_cold_override is not None:
            w[10] = float(forward_cold_override)
        if revenue_tie_override is not None:
            w[11] = float(revenue_tie_override)
        return tuple(float(v) for v in w)

    def score_matrix(
        self,
        X: np.ndarray,
        *,
        blend_alpha_scale: float = 1.0,
        reverse_graft_override: float | None = None,
        forward_cold_override: float | None = None,
        revenue_tie_override: float | None = None,
    ) -> np.ndarray:
        """Score a candidate feature matrix (N, 14) in a single BLAS call."""
        if X.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        w_tup = self.resolve_weights_tuple(
            blend_alpha_scale=blend_alpha_scale,
            reverse_graft_override=reverse_graft_override,
            forward_cold_override=forward_cold_override,
            revenue_tie_override=revenue_tie_override,
        )
        return X @ np.asarray(w_tup, dtype=np.float64)

    def export_scratch_record(self, model_version: str, ts_iso: str) -> dict[str, Any]:
        """Serialize ranker weights and diagnostics for team_NN.scratch."""
        weights_map = {
            name: round(float(val), 4)
            for name, val in zip(FEATURE_NAMES, self.effective_weights, strict=True)
        }
        return {
            "kind": "reranker_weights",
            "model_version": model_version,
            "ts": ts_iso,
            "train_samples": self.train_samples,
            "positive_samples": self.positive_samples,
            "ml_blend_alpha": round(self.ml_blend_alpha, 4),
            "weights": weights_map,
        }


def build_ips_training_dataset(
    session_rows: Iterable[Mapping[str, Any]],
    feature_extractor: Any,
    *,
    exam_power: float = 0.75,
    max_ips_weight: float = 3.5,
    max_sessions: int = 3000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert feedback/session impression logs into IPS-weighted (X, y, weights)."""
    feat_rows: list[np.ndarray] = []
    labels: list[int] = []
    sample_weights: list[float] = []

    processed = 0
    for row in session_rows:
        if processed >= max_sessions:
            break
        if isinstance(row, Mapping):
            raw_items = row.get("items")
            if raw_items is None or len(raw_items) == 0:
                raw_items = row.get("shown")
            if raw_items is None or len(raw_items) == 0:
                raw_items = row.get("recommendations")
            purchased = str(row.get("purchased") or "").strip()
            clicked = str(row.get("clicked") or "").strip()
        else:
            raw_items = getattr(row, "items", None)
            if raw_items is None or len(raw_items) == 0:
                raw_items = getattr(row, "shown", None)
            if raw_items is None or len(raw_items) == 0:
                raw_items = getattr(row, "recommendations", None)
            purchased = str(getattr(row, "purchased", "") or "").strip()
            clicked = str(getattr(row, "clicked", "") or "").strip()

        if raw_items is None or len(raw_items) == 0:
            continue

        shown_list = [str(pid) for pid in raw_items[:10]]
        feat_mat = feature_extractor(row, shown_list)
        if feat_mat.shape[0] != len(shown_list):
            continue

        processed += 1
        for slot_idx, pid in enumerate(shown_list):
            exam_p = (slot_idx + 1.0) ** (-exam_power)
            ips_w = min(1.0 / max(exam_p, 1e-4), max_ips_weight)

            if pid == purchased:
                feat_rows.append(feat_mat[slot_idx])
                labels.append(1)
                sample_weights.append(ips_w * 2.0)
            elif pid == clicked:
                feat_rows.append(feat_mat[slot_idx])
                labels.append(1)
                sample_weights.append(ips_w * 0.45)
            else:
                feat_rows.append(feat_mat[slot_idx])
                labels.append(0)
                sample_weights.append(exam_p)

    if not feat_rows:
        return (
            np.empty((0, NUM_FEATURES), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    return (
        np.vstack(feat_rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sample_weights, dtype=np.float64),
    )
