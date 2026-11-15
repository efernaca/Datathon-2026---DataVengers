"""Vectorized 50-feature Hybrid Position-Debiased (IPS + Weighted-Distilled MLP) Ranker for Calypso 6.2."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .bandits import EMPIRICAL_SLOT_WEIGHTS

FEATURE_NAMES: tuple[str, ...] = (
    # 0..13: Canonical 14 Stage-4 features (exact index alignment preserved)
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
    # 14..20: Collaborative graph & SVD v6 features
    "ease_score",
    "hop2_seq_score",
    "covisit_score",
    "svd_session_sim",
    "svd_user_sim",
    "is_last_viewed",
    "view_recency_max",
    # 21..28: Continuous business, cart decay, recurrent, discount & session coherence
    "in_active_cart_decay",
    "in_forgotten_cart_decay",
    "recurrent_log_norm",
    "discount_ratio",
    "rv_subcat_match_frac",
    "rv_brand_match_frac",
    "price_ratio_to_rv_mean",
    "same_top_cat",
    # 29..39: Page context & explicit second-order non-linear interactions
    "is_page_product",
    "is_page_category",
    "is_page_home",
    "page_prod_x_subcat",
    "page_prod_x_seq",
    "page_cat_x_pop",
    "page_home_x_svd",
    "seq_x_self_cvr",
    "ease_x_price_compat",
    "cart_x_self_cvr",
    "cold_sim_x_pop",
    # 40..49: Calypso 6.2 temporal views, within-subcat pop, subcat price & graph interactions
    "rv_repeat_count_norm",
    "rv_temporal_recency",
    "subcat_pop_rank_score",
    "cat_pop_rank_score",
    "price_rel_subcat_compat",
    "is_viewed_cart_combo",
    "ease_x_same_subcat",
    "hop2_x_same_subcat",
    "svd_sess_x_subcat",
    "ctx_x_no_views",
    # 50..55: Calypso 6.2a cross-subcategory lift, pool-relative norms & consensus precision
    "cross_subcat_lift",
    "pool_rel_seq_norm",
    "pool_rel_pop_norm",
    "subcat_leader_nonlast",
    "multi_channel_consensus",
    "price_vs_user_pref_signed",
)

NUM_FEATURES = len(FEATURE_NAMES)

DEFAULT_PRIOR_WEIGHTS = np.array(
    [
        # 0..13
        1.15,   # seq_lift_score
        2.2702, # self_cvr_view
        0.62,   # intent_score
        0.42,   # user_affinity
        0.22,   # price_compat
        0.36,   # ctx_bayes_score
        0.80,   # global_pop_norm
        0.3585, # live_trend_norm
        0.03,   # is_new_product
        0.38,   # reverse_graft_score
        0.52,   # forward_cold_score
        0.00,   # log_price_norm
        0.1856, # same_top_subcat
        0.1101, # same_top_brand
        # 14..20
        0.12,   # ease_score
        0.00,   # hop2_seq_score
        0.00,   # covisit_score
        0.12,   # svd_session_sim
        0.10,   # svd_user_sim
        0.22,   # is_last_viewed
        0.30,   # view_recency_max
        # 21..28
        1.30,   # in_active_cart_decay
        0.42,   # in_forgotten_cart_decay
        0.35,   # recurrent_log_norm
        0.16,   # discount_ratio
        0.00,   # rv_subcat_match_frac
        0.00,   # rv_brand_match_frac
        0.00,   # price_ratio_to_rv_mean
        0.00,   # same_top_cat
        # 29..39
        0.00,   # is_page_product
        0.00,   # is_page_category
        0.00,   # is_page_home
        0.00,   # page_prod_x_subcat
        0.00,   # page_prod_x_seq
        0.00,   # page_cat_x_pop
        0.00,   # page_home_x_svd
        0.16,   # seq_x_self_cvr
        0.00,   # ease_x_price_compat
        0.20,   # cart_x_self_cvr
        0.00,   # cold_sim_x_pop
        # 40..49
        0.22,   # rv_repeat_count_norm
        0.18,   # rv_temporal_recency
        0.10,   # subcat_pop_rank_score
        0.12,   # cat_pop_rank_score
        0.10,   # price_rel_subcat_compat
        0.22,   # is_viewed_cart_combo
        0.05,   # ease_x_same_subcat
        0.00,   # hop2_x_same_subcat
        0.05,   # svd_sess_x_subcat
        0.18,   # ctx_x_no_views
        # 50..55
        0.00,   # cross_subcat_lift
        0.00,   # pool_rel_seq_norm
        0.00,   # pool_rel_pop_norm
        0.00,   # subcat_leader_nonlast
        0.00,   # multi_channel_consensus
        0.00,   # price_vs_user_pref_signed
    ],
    dtype=np.float64,
)

_MIN_PRIOR_RATIO = np.array(
    [
        0.72, 0.75, 0.65, 0.55, 0.45, 0.50, 0.30, 0.30, 0.00, 0.55, 0.55, 0.00, 0.40, 0.00,
        0.35, 0.25, 0.25, 0.30, 0.25, 0.00, 0.20,
        0.40, 0.25, 0.25, 0.10, 0.20, 0.00, 0.20, 0.10,
        0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.20, 0.15, 0.20, 0.15,
        0.25, 0.20, 0.25, 0.10, 0.15, 0.20, 0.20, 0.15, 0.15, 0.20,
        0.25, 0.20, 0.20, 0.20, 0.25, 0.00,
    ],
    dtype=np.float64,
)


def _pad_or_trim_1d(arr: np.ndarray, target_len: int, fallback: np.ndarray) -> np.ndarray:
    out = fallback.astype(np.float64, copy=True)
    n = min(len(arr), target_len)
    if n > 0:
        out[:n] = arr[:n]
    return out


@dataclass(slots=True)
class PositionDebiasedRanker:
    """Hybrid 2-layer IPS Ranker: anchored Linear LogisticRegression + Distilled NumPy MLP residual."""

    prior_weights: np.ndarray
    effective_weights: np.ndarray
    ml_weights_raw: np.ndarray
    ml_blend_alpha: float = 0.0
    train_samples: int = 0
    positive_samples: int = 0
    min_prior_ratios: np.ndarray = field(default_factory=lambda: _MIN_PRIOR_RATIO.copy())
    max_prior_multiplier: float = 2.4

    # Layer 2: Compact pure-NumPy float32 residual MLP (distilled from HistGradientBoosting + data)
    nl_mean: np.ndarray | None = None
    nl_inv_std: np.ndarray | None = None
    nl_W1: np.ndarray | None = None
    nl_b1: np.ndarray | None = None
    nl_W2: np.ndarray | None = None
    nl_b2: np.ndarray | None = None
    nl_W3: np.ndarray | None = None
    nl_b3: np.ndarray | None = None
    nl_scale: float = 0.25
    nl_offset: float = 0.0
    nl_blend_alpha: float = 0.0
    val_auc: float = 0.0

    @classmethod
    def from_priors(
        cls,
        prior_weights: np.ndarray | None = None,
        *,
        min_prior_ratios: np.ndarray | None = None,
        max_prior_multiplier: float = 2.4,
    ) -> PositionDebiasedRanker:
        pw = (
            DEFAULT_PRIOR_WEIGHTS.copy()
            if prior_weights is None
            else _pad_or_trim_1d(np.asarray(prior_weights, dtype=np.float64), NUM_FEATURES, DEFAULT_PRIOR_WEIGHTS)
        )
        ratios = (
            _MIN_PRIOR_RATIO.copy()
            if min_prior_ratios is None
            else _pad_or_trim_1d(np.asarray(min_prior_ratios, dtype=np.float64), NUM_FEATURES, _MIN_PRIOR_RATIO)
        )
        return cls(
            prior_weights=pw,
            effective_weights=pw.copy(),
            ml_weights_raw=pw.copy(),
            ml_blend_alpha=0.0,
            train_samples=0,
            positive_samples=0,
            min_prior_ratios=ratios,
            max_prior_multiplier=float(max_prior_multiplier),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        *,
        max_blend_alpha: float = 0.72,
        min_positives: int = 25,
        l2_c: float = 1.2,
        nonlin_blend_alpha: float = 0.45,
        nonlin_l2_alpha: float = 2.5,
        enable_nonlin: bool = True,
    ) -> PositionDebiasedRanker:
        """Fit standardized IPS Linear + Distilled Non-Linear Residual MLP."""
        if X.ndim == 2 and X.shape[1] < NUM_FEATURES:
            padded = np.zeros((X.shape[0], NUM_FEATURES), dtype=np.float64)
            padded[:, : X.shape[1]] = X
            X = padded

        n_samples = int(X.shape[0]) if X.ndim == 2 else 0
        n_pos = int(np.sum(y > 0.5)) if n_samples > 0 else 0
        self.train_samples = n_samples
        self.positive_samples = n_pos

        if n_samples < 40 or n_pos < min_positives or len(np.unique(y)) < 2:
            self.ml_blend_alpha = 0.0
            self.nl_blend_alpha = 0.0
            self.effective_weights = self.prior_weights.copy()
            return self

        X64 = X.astype(np.float64, copy=False)
        y_int = y.astype(np.int64, copy=False)
        sw = np.maximum(sample_weights.astype(np.float64, copy=False), 1e-4)

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X64)
        clf = LogisticRegression(
            C=l2_c,
            solver="lbfgs",
            max_iter=500,
            random_state=42,
        )
        clf.fit(X_std, y_int, sample_weight=sw)

        scales = np.maximum(scaler.scale_, 1e-3)
        raw_ml = clf.coef_[0].astype(np.float64, copy=True) / scales

        pos_norm = float(np.sum(np.abs(self.prior_weights)))
        ml_norm = float(np.sum(np.maximum(raw_ml, 0.0)))
        if ml_norm <= 1e-6:
            self.ml_blend_alpha = 0.0
            self.nl_blend_alpha = 0.0
            self.effective_weights = self.prior_weights.copy()
            return self

        rescaled_ml = raw_ml * (pos_norm / ml_norm)
        ratios = (
            self.min_prior_ratios
            if len(self.min_prior_ratios) == NUM_FEATURES
            else _MIN_PRIOR_RATIO
        )
        mult = max(float(self.max_prior_multiplier), 1.0)
        pos_mask = self.prior_weights >= 0.0
        ext_disabled = (np.arange(NUM_FEATURES) >= 50) & (np.abs(self.prior_weights) < 1e-6)
        floors = np.where(
            pos_mask,
            ratios * self.prior_weights,
            np.minimum(self.prior_weights * mult, -0.02),
        )
        ceilings = np.where(
            pos_mask,
            np.where(ext_disabled, 0.0, np.maximum(self.prior_weights * mult, 0.30)),
            ratios * self.prior_weights,
        )
        low = np.minimum(floors, ceilings)
        high = np.maximum(floors, ceilings)
        stabilized_ml = np.clip(rescaled_ml, low, high)

        ramp = min(1.0, max(0.0, (n_pos - min_positives) / 350.0))
        alpha = max_blend_alpha * ramp
        self.ml_weights_raw = stabilized_ml
        self.ml_blend_alpha = float(alpha)
        self.effective_weights = (1.0 - alpha) * self.prior_weights + alpha * stabilized_ml

        # Layer 2: Distilled Non-Linear Residual Learner (HistGBDT teacher -> Pure NumPy float32 MLP)
        self.nl_blend_alpha = 0.0
        if (
            enable_nonlin
            and n_samples >= 200
            and n_pos >= max(int(min_positives), 20)
            and nonlin_blend_alpha > 0.0
        ):
            try:
                self._fit_nonlin_residual(
                    X_std,
                    X64,
                    y_int,
                    sw,
                    scaler.mean_,
                    scales,
                    nonlin_blend_alpha=nonlin_blend_alpha,
                    nonlin_l2_alpha=nonlin_l2_alpha,
                )
            except Exception:
                self.nl_blend_alpha = 0.0
        return self

    def _fit_nonlin_residual(
        self,
        X_std: np.ndarray,
        X64: np.ndarray,
        y_int: np.ndarray,
        sw: np.ndarray,
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        *,
        nonlin_blend_alpha: float,
        nonlin_l2_alpha: float,
    ) -> None:
        lin_raw = X64 @ self.effective_weights
        lin_mu = float(np.mean(lin_raw))
        lin_sd = max(float(np.std(lin_raw)), 1e-4)

        # Calibrate 1D linear log-odds against IPS-weighted target so teacher learns true delta
        try:
            cal = LogisticRegression(C=2.0, solver="lbfgs", max_iter=200, random_state=42)
            cal.fit(((lin_raw - lin_mu) / lin_sd)[:, None], y_int, sample_weight=sw)
            lin_logit = (cal.coef_[0, 0] * ((lin_raw - lin_mu) / lin_sd) + cal.intercept_[0]).astype(np.float64)
        except Exception:
            lin_logit = np.clip((lin_raw - lin_mu) / lin_sd, -6.0, 6.0)
        lin_p = 1.0 / (1.0 + np.exp(-np.clip(lin_logit, -8.0, 8.0)))

        # Teacher HistGradientBoosting sized adaptively to real positive count
        n_pos = int(np.sum(y_int > 0))
        min_leaf = max(8, min(28, n_pos // 4)) if n_pos > 0 else 20
        use_early = n_pos >= 60 and X_std.shape[0] >= 500
        teacher = HistGradientBoostingClassifier(
            max_depth=5,
            max_iter=180 if use_early else 95,
            learning_rate=0.045,
            min_samples_leaf=min_leaf,
            l2_regularization=2.0,
            max_bins=128,
            early_stopping=use_early,
            validation_fraction=0.12 if use_early else 0.1,
            n_iter_no_change=14,
            random_state=42,
        )
        teacher.fit(X_std, y_int, sample_weight=sw)
        t_prob = np.clip(teacher.predict_proba(X_std)[:, 1], 1e-4, 1.0 - 1e-4)
        t_logit = np.log(t_prob / (1.0 - t_prob))

        # True centered residual over linear log-odds + direct IPS margin
        raw_res = (t_logit - lin_logit) * 0.80 + 0.20 * ((y_int.astype(np.float64) - lin_p) * 3.0)
        res_target = np.clip(raw_res - float(np.mean(raw_res)), -4.5, 4.5)

        # IPS-weighted balanced sampling so MLPRegressor respects purchase weights despite no sample_weight arg
        pos_idx = np.flatnonzero(y_int > 0)
        neg_idx = np.flatnonzero(y_int == 0)
        if len(pos_idx) >= 12 and len(neg_idx) >= 24:
            rng = np.random.default_rng(42)
            # Replicate positives according to normalized IPS weights up to ~2400 positive draws
            pos_w = sw[pos_idx]
            pos_probs = pos_w / max(float(pos_w.sum()), 1e-9)
            n_pos_draw = min(max(len(pos_idx) * 3, 256), 2400)
            draw_pos = rng.choice(pos_idx, size=n_pos_draw, replace=True, p=pos_probs)

            # Negatives weighted by sw plus modest uniform floor for coverage
            neg_w = sw[neg_idx] + 0.05 * float(np.mean(sw[neg_idx]))
            neg_probs = neg_w / max(float(neg_w.sum()), 1e-9)
            n_neg_draw = min(len(neg_idx), max(n_pos_draw * 5, 3200))
            draw_neg = rng.choice(neg_idx, size=n_neg_draw, replace=(n_neg_draw > len(neg_idx)), p=neg_probs)

            distill_idx = np.concatenate([draw_pos, draw_neg])
            rng.shuffle(distill_idx)
            X_mlp = X_std[distill_idx]
            y_mlp = res_target[distill_idx]
        else:
            X_mlp = X_std
            y_mlp = res_target

        student = MLPRegressor(
            hidden_layer_sizes=(80, 40),
            activation="tanh",
            alpha=max(float(nonlin_l2_alpha) * 0.85, 0.5),
            batch_size=min(512, max(64, len(y_mlp) // 16)),
            learning_rate_init=0.003,
            max_iter=240,
            early_stopping=len(y_mlp) >= 250,
            validation_fraction=0.10,
            n_iter_no_change=14,
            random_state=42,
        )
        student.fit(X_mlp, y_mlp)

        self.nl_mean = x_mean.astype(np.float32)
        self.nl_inv_std = (1.0 / x_scale).astype(np.float32)
        self.nl_W1 = student.coefs_[0].astype(np.float32)
        self.nl_b1 = student.intercepts_[0].astype(np.float32)
        self.nl_W2 = student.coefs_[1].astype(np.float32)
        self.nl_b2 = student.intercepts_[1].astype(np.float32)
        self.nl_W3 = student.coefs_[2].astype(np.float32)
        self.nl_b3 = student.intercepts_[2].astype(np.float32)

        nl_pred = self._raw_nonlin_scores(X64)
        nl_mu = float(np.mean(nl_pred))
        nl_sd = max(float(np.std(nl_pred)), 1e-4)
        self.nl_offset = nl_mu
        # Calibrate non-linear delta amplitude cleanly relative to linear score dispersion
        self.nl_scale = (0.72 * lin_sd) / nl_sd
        nl_ramp = min(1.0, max(0.0, (len(y_int) - 300) / 4000.0))
        self.nl_blend_alpha = float(nonlin_blend_alpha * max(0.35, nl_ramp))

    def _raw_nonlin_scores(self, X: np.ndarray) -> np.ndarray:
        if (
            self.nl_W1 is None
            or self.nl_mean is None
            or self.nl_inv_std is None
            or X.shape[0] == 0
        ):
            return np.zeros(X.shape[0], dtype=np.float64)
        d = min(int(X.shape[1]), int(self.nl_mean.shape[0]))
        Z = (X[:, :d].astype(np.float32, copy=False) - self.nl_mean[:d]) * self.nl_inv_std[:d]
        H1 = np.tanh(Z @ self.nl_W1[:d] + self.nl_b1)
        H2 = np.tanh(H1 @ self.nl_W2 + self.nl_b2)
        out = (H2 @ self.nl_W3 + self.nl_b3).ravel()
        return out.astype(np.float64)

    def resolve_weights_tuple(
        self,
        *,
        blend_alpha_scale: float = 1.0,
        reverse_graft_override: float | None = None,
        forward_cold_override: float | None = None,
        revenue_tie_override: float | None = None,
    ) -> tuple[float, ...]:
        """Return effective NUM_FEATURES weights as a compact Python tuple."""
        alpha = min(1.0, max(0.0, self.ml_blend_alpha * blend_alpha_scale))
        if alpha <= 0.0:
            w = self.prior_weights.tolist()
        elif alpha == self.ml_blend_alpha:
            w = self.effective_weights.tolist()
        else:
            w = ((1.0 - alpha) * self.prior_weights + alpha * self.ml_weights_raw).tolist()

        if len(w) < NUM_FEATURES:
            w.extend(DEFAULT_PRIOR_WEIGHTS[len(w) :].tolist())

        if reverse_graft_override is not None:
            w[9] = float(reverse_graft_override)
        if forward_cold_override is not None:
            w[10] = float(forward_cold_override)
        if revenue_tie_override is not None:
            w[11] = float(revenue_tie_override)
        return tuple(float(v) for v in w[:NUM_FEATURES])

    def score_nonlin_residuals(
        self,
        X: np.ndarray,
        *,
        nonlin_alpha_override: float | None = None,
        nonlin_temp: float = 1.0,
    ) -> np.ndarray:
        """Compute pure-NumPy calibrated non-linear residual deltas for candidate matrix X."""
        if X.shape[0] == 0 or self.nl_W1 is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        eff_nl = self.nl_blend_alpha if nonlin_alpha_override is None else float(nonlin_alpha_override)
        if eff_nl <= 0.0:
            return np.zeros(X.shape[0], dtype=np.float64)
        raw = self._raw_nonlin_scores(X)
        temp = max(float(nonlin_temp), 0.1)
        return eff_nl * np.tanh(((raw - self.nl_offset) * self.nl_scale) / temp)

    def score_matrix(
        self,
        X: np.ndarray,
        *,
        blend_alpha_scale: float = 1.0,
        reverse_graft_override: float | None = None,
        forward_cold_override: float | None = None,
        revenue_tie_override: float | None = None,
        nonlin_alpha_override: float | None = None,
        nonlin_temp: float = 1.0,
    ) -> np.ndarray:
        """Score a candidate feature matrix (N, NUM_FEATURES) with Linear + Distilled MLP."""
        if X.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        if X.shape[1] < NUM_FEATURES:
            padded = np.zeros((X.shape[0], NUM_FEATURES), dtype=np.float64)
            padded[:, : X.shape[1]] = X
            X = padded
        w_tup = self.resolve_weights_tuple(
            blend_alpha_scale=blend_alpha_scale,
            reverse_graft_override=reverse_graft_override,
            forward_cold_override=forward_cold_override,
            revenue_tie_override=revenue_tie_override,
        )
        lin = X @ np.asarray(w_tup, dtype=np.float64)
        if self.nl_W1 is not None:
            lin = lin + self.score_nonlin_residuals(
                X,
                nonlin_alpha_override=nonlin_alpha_override,
                nonlin_temp=nonlin_temp,
            )
        return lin

    def with_updated_priors(
        self,
        prior_weights: np.ndarray,
        min_prior_ratios: np.ndarray,
        max_prior_multiplier: float,
    ) -> PositionDebiasedRanker:
        """Clone ranker updating prior weights while preserving fitted ML & non-linear MLP layers."""
        pw = _pad_or_trim_1d(np.asarray(prior_weights, dtype=np.float64), NUM_FEATURES, DEFAULT_PRIOR_WEIGHTS)
        ratios = _pad_or_trim_1d(np.asarray(min_prior_ratios, dtype=np.float64), NUM_FEATURES, _MIN_PRIOR_RATIO)
        raw_ml = _pad_or_trim_1d(self.ml_weights_raw, NUM_FEATURES, pw)
        alpha = self.ml_blend_alpha
        eff = (1.0 - alpha) * pw + alpha * raw_ml
        return PositionDebiasedRanker(
            prior_weights=pw,
            effective_weights=eff,
            ml_weights_raw=raw_ml.copy(),
            ml_blend_alpha=alpha,
            train_samples=self.train_samples,
            positive_samples=self.positive_samples,
            min_prior_ratios=ratios,
            max_prior_multiplier=float(max_prior_multiplier),
            nl_mean=self.nl_mean,
            nl_inv_std=self.nl_inv_std,
            nl_W1=self.nl_W1,
            nl_b1=self.nl_b1,
            nl_W2=self.nl_W2,
            nl_b2=self.nl_b2,
            nl_W3=self.nl_W3,
            nl_b3=self.nl_b3,
            nl_scale=self.nl_scale,
            nl_offset=self.nl_offset,
            nl_blend_alpha=self.nl_blend_alpha,
            val_auc=self.val_auc,
        )

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
            "nl_blend_alpha": round(self.nl_blend_alpha, 4),
            "has_nonlin_mlp": self.nl_W1 is not None,
            "weights": weights_map,
        }


def build_ips_training_dataset(
    session_rows: Iterable[Mapping[str, Any]],
    feature_extractor: Any,
    *,
    exam_power: float = 0.75,
    max_ips_weight: float = 3.5,
    max_sessions: int = 3000,
    purchase_weight: float = 2.0,
    click_weight: float = 0.45,
    use_empirical_exam_curve: bool = False,
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
            if use_empirical_exam_curve and slot_idx < len(EMPIRICAL_SLOT_WEIGHTS):
                exam_p = EMPIRICAL_SLOT_WEIGHTS[slot_idx]
            else:
                exam_p = (slot_idx + 1.0) ** (-exam_power)
            ips_w = min(1.0 / max(exam_p, 1e-4), max_ips_weight)

            if pid == purchased:
                feat_rows.append(feat_mat[slot_idx])
                labels.append(1)
                sample_weights.append(ips_w * purchase_weight)
            elif pid == clicked and click_weight > 0.0:
                feat_rows.append(feat_mat[slot_idx])
                labels.append(1)
                sample_weights.append(ips_w * click_weight)
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
