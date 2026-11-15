"""Vectorized 64-feature Hybrid Position-Debiased (IPS + Weighted-Distilled MLP + Slot-1 Precision) Ranker for Calypso 6.2b."""
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
    # 56..63: Calypso 6.2b Slot-1 conversion precision, live CVR velocity & over-hype risk guard
    "ctx_x_pop_consensus",
    "affinity_x_pop",
    "seq_x_pop_dampened",
    "cross_subcat_x_pop",
    "affordable_pop_fit",
    "slot1_ctx_leader_ind",
    "live_cvr_velocity",
    "seq_overhype_risk",
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
        # 50..55 (0.00 synthetic-safe defaults; overridden by tuned WeightsProfile)
        0.00,   # cross_subcat_lift
        0.00,   # pool_rel_seq_norm
        0.00,   # pool_rel_pop_norm
        0.00,   # subcat_leader_nonlast
        0.00,   # multi_channel_consensus
        0.00,   # price_vs_user_pref_signed
        # 56..63 (0.00 synthetic-safe defaults; overridden by tuned WeightsProfile)
        0.00,   # ctx_x_pop_consensus
        0.00,   # affinity_x_pop
        0.00,   # seq_x_pop_dampened
        0.00,   # cross_subcat_x_pop
        0.00,   # affordable_pop_fit
        0.00,   # slot1_ctx_leader_ind
        0.00,   # live_cvr_velocity
        0.00,   # seq_overhype_risk
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
        0.25, 0.25, 0.20, 0.25, 0.20, 0.25, 0.20, 0.25,
    ],
    dtype=np.float64,
)


def _pad_or_trim_1d(arr: np.ndarray, target_len: int, fallback: np.ndarray) -> np.ndarray:
    out = fallback.astype(np.float64, copy=True)
    n = min(len(arr), target_len)
    if n > 0:
        out[:n] = arr[:n]
    return out


_NL_FOLD_CACHE: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}


def _raw_nonlin_get_folded(
    nl_mean: np.ndarray,
    nl_inv_std: np.ndarray,
    nl_W1: np.ndarray,
    nl_b1: np.ndarray,
    nl_W3: np.ndarray,
    nl_b3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    key = (id(nl_W1), id(nl_mean), id(nl_inv_std), id(nl_W3))
    cached = _NL_FOLD_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_NL_FOLD_CACHE) >= 16:
        _NL_FOLD_CACHE.clear()
    w1_fold = (nl_inv_std[:, None] * nl_W1).astype(np.float32, copy=False)
    b1_fold = (nl_b1 - (nl_mean * nl_inv_std) @ nl_W1).astype(np.float32, copy=False)
    w3_1d = nl_W3.ravel().astype(np.float32, copy=False)
    b3_scalar = float(nl_b3.ravel()[0]) if nl_b3.size else 0.0
    val = (w1_fold, b1_fold, w3_1d, b3_scalar)
    _NL_FOLD_CACHE[key] = val
    return val


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

    # Layer 3: Two-Head Slot-1 Conversion Precision Head & Synthetic->Real Transfer State
    slot1_prior_weights: np.ndarray | None = None
    slot1_weights: np.ndarray | None = None
    slot1_ml_blend_alpha: float = 0.0
    slot1_head_alpha: float = 0.0
    slot1_pos_samples: int = 0
    pretrain_lin_weights: np.ndarray | None = None
    syn_pretrained: bool = False
    transfer_fine_tuned: bool = False

    def _anchor_weights(
        self,
        prior: np.ndarray,
        raw_ml: np.ndarray,
        *,
        clamp_ext_disabled: bool = True,
    ) -> np.ndarray | None:
        pos_norm = float(np.sum(np.abs(prior)))
        has_neg_prior = bool(np.any(prior < -1e-6))
        ml_norm = (
            float(np.sum(np.abs(raw_ml)))
            if has_neg_prior
            else float(np.sum(np.maximum(raw_ml, 0.0)))
        )
        if ml_norm <= 1e-6 or pos_norm <= 1e-6:
            return None

        rescaled_ml = raw_ml * (pos_norm / ml_norm)
        ratios = (
            self.min_prior_ratios
            if len(self.min_prior_ratios) == NUM_FEATURES
            else _MIN_PRIOR_RATIO
        )
        mult = max(float(self.max_prior_multiplier), 1.0)
        pos_mask = prior >= 0.0
        floors = np.where(
            pos_mask,
            ratios * prior,
            np.minimum(prior * mult, -0.02),
        )
        ceilings = np.where(
            pos_mask,
            np.maximum(prior * mult, 0.30),
            ratios * prior,
        )
        if clamp_ext_disabled:
            ext_disabled = (np.arange(NUM_FEATURES) >= 50) & (np.abs(prior) < 1e-6)
            floors = np.where(ext_disabled, 0.0, floors)
            ceilings = np.where(ext_disabled, 0.0, ceilings)
        low = np.minimum(floors, ceilings)
        high = np.maximum(floors, ceilings)
        return np.clip(rescaled_ml, low, high)

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

        stabilized_ml = self._anchor_weights(self.prior_weights, raw_ml, clamp_ext_disabled=True)
        if stabilized_ml is None:
            self.ml_blend_alpha = 0.0
            self.nl_blend_alpha = 0.0
            self.effective_weights = self.prior_weights.copy()
            return self

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
        n_feats = int(self.nl_mean.shape[0])
        if int(X.shape[1]) == n_feats:
            w1_fold, b1_fold, w3_1d, b3_s = _raw_nonlin_get_folded(
                self.nl_mean,
                self.nl_inv_std,
                self.nl_W1,
                self.nl_b1,
                self.nl_W3,
                self.nl_b3,
            )
            H1 = X.astype(np.float32, copy=False) @ w1_fold
            H1 += b1_fold
            np.tanh(H1, out=H1)
            H2 = H1 @ self.nl_W2
            H2 += self.nl_b2
            np.tanh(H2, out=H2)
            out = H2 @ w3_1d
            out += b3_s
            return out.astype(np.float64)
        d = min(int(X.shape[1]), n_feats)
        Z = (X[:, :d].astype(np.float32, copy=False) - self.nl_mean[:d]) * self.nl_inv_std[:d]
        H1 = Z @ self.nl_W1[:d]
        H1 += self.nl_b1
        np.tanh(H1, out=H1)
        H2 = H1 @ self.nl_W2
        H2 += self.nl_b2
        np.tanh(H2, out=H2)
        out = (H2 @ self.nl_W3 + self.nl_b3).ravel()
        return out.astype(np.float64)

    def resolve_weights_array(
        self,
        *,
        blend_alpha_scale: float = 1.0,
        reverse_graft_override: float | None = None,
        forward_cold_override: float | None = None,
        revenue_tie_override: float | None = None,
    ) -> np.ndarray:
        """Return effective NUM_FEATURES weights as a contiguous float64 NumPy array."""
        alpha = min(1.0, max(0.0, self.ml_blend_alpha * blend_alpha_scale))
        if alpha <= 0.0:
            w_src = self.prior_weights
        elif alpha == self.ml_blend_alpha:
            w_src = self.effective_weights
        else:
            w_src = (1.0 - alpha) * self.prior_weights + alpha * self.ml_weights_raw

        if len(w_src) == NUM_FEATURES:
            w_out = np.array(w_src, dtype=np.float64, copy=True)
        else:
            w_out = _pad_weights_vec(w_src, DEFAULT_PRIOR_WEIGHTS, NUM_FEATURES)

        if reverse_graft_override is not None:
            w_out[9] = float(reverse_graft_override)
        if forward_cold_override is not None:
            w_out[10] = float(forward_cold_override)
        if revenue_tie_override is not None:
            w_out[11] = float(revenue_tie_override)
        return w_out

    def resolve_weights_tuple(
        self,
        *,
        blend_alpha_scale: float = 1.0,
        reverse_graft_override: float | None = None,
        forward_cold_override: float | None = None,
        revenue_tie_override: float | None = None,
    ) -> tuple[float, ...]:
        """Return effective NUM_FEATURES weights as a compact Python tuple."""
        return tuple(
            self.resolve_weights_array(
                blend_alpha_scale=blend_alpha_scale,
                reverse_graft_override=reverse_graft_override,
                forward_cold_override=forward_cold_override,
                revenue_tie_override=revenue_tie_override,
            ).tolist()
        )

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

    def pretrain_synthetic(
        self,
        X_syn: np.ndarray,
        y_syn: np.ndarray,
        sw_syn: np.ndarray,
        *,
        transfer_prior_pull: float = 0.25,
        l2_c: float = 1.0,
        nonlin_blend_alpha: float = 0.45,
        nonlin_l2_alpha: float = 2.5,
        enable_nonlin: bool = True,
    ) -> PositionDebiasedRanker:
        """Stage-1 Synthetic Pretraining: learn transferable linear prior & MLP representation."""
        if X_syn.ndim == 2 and X_syn.shape[1] < NUM_FEATURES:
            padded = np.zeros((X_syn.shape[0], NUM_FEATURES), dtype=np.float64)
            padded[:, : X_syn.shape[1]] = X_syn
            X_syn = padded
        n_samples = int(X_syn.shape[0]) if X_syn.ndim == 2 else 0
        n_pos = int(np.sum(y_syn > 0.5)) if n_samples > 0 else 0
        if n_samples < 80 or n_pos < 20 or len(np.unique(y_syn)) < 2:
            return self

        X64 = X_syn.astype(np.float64, copy=False)
        y_int = y_syn.astype(np.int64, copy=False)
        sw = np.maximum(sw_syn.astype(np.float64, copy=False), 1e-4)

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X64)
        clf = LogisticRegression(C=l2_c, solver="lbfgs", max_iter=400, random_state=42)
        clf.fit(X_std, y_int, sample_weight=sw)
        scales = np.maximum(scaler.scale_, 1e-3)
        raw_ml = clf.coef_[0].astype(np.float64, copy=True) / scales
        stabilized_syn = self._anchor_weights(self.prior_weights, raw_ml, clamp_ext_disabled=True)
        if stabilized_syn is not None:
            pull = min(max(float(transfer_prior_pull), 0.0), 0.85)
            self.pretrain_lin_weights = (1.0 - pull) * self.prior_weights + pull * stabilized_syn
            self.syn_pretrained = True

        if enable_nonlin and n_samples >= 200 and nonlin_blend_alpha > 0.0:
            saved_eff = self.effective_weights.copy()
            if self.pretrain_lin_weights is not None:
                self.effective_weights = self.pretrain_lin_weights.copy()
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
                pass
            finally:
                self.effective_weights = saved_eff
        return self

    def transfer_fine_tune(
        self,
        X_real: np.ndarray,
        y_real: np.ndarray,
        sw_real: np.ndarray,
        *,
        transfer_prior_pull: float = 0.35,
        max_blend_alpha: float = 0.72,
        min_positives: int = 12,
        l2_c: float = 1.2,
        nonlin_blend_alpha: float = 0.45,
        nonlin_l2_alpha: float = 4.0,
        enable_nonlin: bool = True,
        ridge_only: bool = False,
    ) -> PositionDebiasedRanker:
        """Stage-2 Synthetic->Real Transfer: anchor linear weights & Ridge-adapt MLP last layer (< 3ms)."""
        if X_real.ndim == 2 and X_real.shape[1] < NUM_FEATURES:
            padded = np.zeros((X_real.shape[0], NUM_FEATURES), dtype=np.float64)
            padded[:, : X_real.shape[1]] = X_real
            X_real = padded
        n_samples = int(X_real.shape[0]) if X_real.ndim == 2 else 0
        n_pos = int(np.sum(y_real > 0.5)) if n_samples > 0 else 0
        if n_samples < 30 or n_pos < min_positives or len(np.unique(y_real)) < 2:
            return self

        X64 = X_real.astype(np.float64, copy=False)
        y_int = y_real.astype(np.int64, copy=False)
        sw = np.maximum(sw_real.astype(np.float64, copy=False), 1e-4)

        pull = min(max(float(transfer_prior_pull), 0.0), 0.75) if self.syn_pretrained else 0.0
        anchor_base = (
            (1.0 - pull) * self.prior_weights + pull * self.pretrain_lin_weights
            if (self.syn_pretrained and self.pretrain_lin_weights is not None)
            else self.prior_weights
        )

        # Fast path or full real adaptation for linear head
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X64)
        scales = np.maximum(scaler.scale_, 1e-3)
        if not ridge_only:
            clf = LogisticRegression(C=l2_c, solver="lbfgs", max_iter=500, random_state=42)
            clf.fit(X_std, y_int, sample_weight=sw)
            raw_ml = clf.coef_[0].astype(np.float64, copy=True) / scales
            stabilized_ml = self._anchor_weights(anchor_base, raw_ml, clamp_ext_disabled=True)
            if stabilized_ml is not None:
                ramp = min(1.0, max(0.0, (n_pos - min_positives) / 350.0))
                alpha = max_blend_alpha * ramp
                self.train_samples = n_samples
                self.positive_samples = n_pos
                self.ml_weights_raw = stabilized_ml
                self.ml_blend_alpha = float(alpha)
                self.effective_weights = (1.0 - alpha) * self.prior_weights + alpha * stabilized_ml

        # Frozen-trunk closed-form Ridge transfer on MLP last layer (nl_W3, nl_b3)
        if (
            enable_nonlin
            and nonlin_blend_alpha > 0.0
            and self.nl_W1 is not None
            and self.nl_W2 is not None
            and self.nl_W3 is not None
            and self.nl_mean is not None
            and self.nl_inv_std is not None
        ):
            lin_raw = X64 @ self.effective_weights
            lin_mu = float(np.mean(lin_raw))
            lin_sd = max(float(np.std(lin_raw)), 1e-4)
            z_lin = (lin_raw - lin_mu) / lin_sd
            try:
                cal = LogisticRegression(C=2.0, solver="lbfgs", max_iter=150, random_state=42)
                cal.fit(z_lin[:, None], y_int, sample_weight=sw)
                lin_logit = cal.coef_[0, 0] * z_lin + cal.intercept_[0]
            except Exception:
                lin_logit = np.clip(z_lin, -6.0, 6.0)
            lin_p = 1.0 / (1.0 + np.exp(-np.clip(lin_logit, -8.0, 8.0)))

            res_direct = (y_int.astype(np.float64) - lin_p) * 3.0
            if not ridge_only and n_samples >= 450 and n_pos >= 50:
                try:
                    min_leaf = max(8, min(28, n_pos // 4))
                    teacher = HistGradientBoostingClassifier(
                        max_depth=5,
                        max_iter=160,
                        learning_rate=0.045,
                        min_samples_leaf=min_leaf,
                        l2_regularization=2.5,
                        max_bins=128,
                        early_stopping=True,
                        validation_fraction=0.12,
                        n_iter_no_change=12,
                        random_state=42,
                    )
                    teacher.fit(X_std, y_int, sample_weight=sw)
                    t_prob = np.clip(teacher.predict_proba(X_std)[:, 1], 1e-4, 1.0 - 1e-4)
                    t_logit = np.log(t_prob / (1.0 - t_prob))
                    raw_res = (t_logit - lin_logit) * 0.80 + 0.20 * res_direct
                except Exception:
                    raw_res = res_direct
            else:
                raw_res = res_direct
            res_target = np.clip(raw_res - float(np.mean(raw_res)), -4.5, 4.5)

            d = min(int(X64.shape[1]), int(self.nl_mean.shape[0]))
            Z = (X64[:, :d].astype(np.float32, copy=False) - self.nl_mean[:d]) * self.nl_inv_std[:d]
            H1 = np.tanh(Z @ self.nl_W1[:d] + self.nl_b1)
            H2 = np.tanh(H1 @ self.nl_W2 + self.nl_b2).astype(np.float64, copy=False)

            w_norm = sw / max(float(np.mean(sw)), 1e-6)
            h_dim = int(H2.shape[1])
            H_aug = np.empty((n_samples, h_dim + 1), dtype=np.float64)
            H_aug[:, :h_dim] = H2
            H_aug[:, h_dim] = 1.0

            w3_prior = self.nl_W3.ravel().astype(np.float64)
            b3_prior = float(self.nl_b3.ravel()[0]) if self.nl_b3 is not None else 0.0
            theta_prior = np.empty(h_dim + 1, dtype=np.float64)
            theta_prior[:h_dim] = w3_prior
            theta_prior[h_dim] = b3_prior

            pred_prior = H_aug @ theta_prior
            resid_diff = res_target - pred_prior
            Hw = H_aug * w_norm[:, None]
            lam = max(float(nonlin_l2_alpha) * 3.0, 2.0)
            reg_diag = np.full(h_dim + 1, lam, dtype=np.float64)
            reg_diag[h_dim] = lam * 0.25
            A = Hw.T @ H_aug + np.diag(reg_diag)
            b_vec = Hw.T @ resid_diff
            try:
                delta_theta = np.linalg.solve(A, b_vec)
                theta_new = theta_prior + np.clip(delta_theta, -2.5, 2.5)
                self.nl_W3 = theta_new[:h_dim, None].astype(np.float32)
                self.nl_b3 = np.array([theta_new[h_dim]], dtype=np.float32)
            except np.linalg.LinAlgError:
                pass

            nl_pred = self._raw_nonlin_scores(X64)
            nl_mu = float(np.mean(nl_pred))
            nl_sd = max(float(np.std(nl_pred)), 1e-4)
            self.nl_offset = nl_mu
            self.nl_scale = (0.72 * lin_sd) / nl_sd
            nl_ramp = min(1.0, max(0.0, (len(y_int) - 200) / 3500.0))
            self.nl_blend_alpha = float(nonlin_blend_alpha * max(0.35, nl_ramp))
            self.transfer_fine_tuned = True
        elif enable_nonlin and n_samples >= 200 and n_pos >= max(int(min_positives), 20) and not ridge_only:
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

    def fit_slot1_head(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        slot1_mask: np.ndarray | None = None,
        *,
        slot1_prior_weights: np.ndarray | None = None,
        slot1_head_alpha: float = 0.75,
        max_blend_alpha: float = 0.75,
        min_positives: int = 15,
        l2_c: float = 0.9,
    ) -> PositionDebiasedRanker:
        """Fit dedicated Slot-1 conversion precision head separating Slot-1 buyers from hype traps."""
        s1_prior = (
            self.prior_weights.copy()
            if slot1_prior_weights is None
            else _pad_or_trim_1d(
                np.asarray(slot1_prior_weights, dtype=np.float64),
                NUM_FEATURES,
                DEFAULT_PRIOR_WEIGHTS,
            )
        )
        self.slot1_prior_weights = s1_prior
        self.slot1_head_alpha = float(max(0.0, min(1.0, slot1_head_alpha)))

        if X.ndim == 2 and X.shape[1] < NUM_FEATURES:
            padded = np.zeros((X.shape[0], NUM_FEATURES), dtype=np.float64)
            padded[:, : X.shape[1]] = X
            X = padded

        n_samples = int(X.shape[0]) if X.ndim == 2 else 0
        n_pos = int(np.sum(y > 0.5)) if n_samples > 0 else 0
        self.slot1_pos_samples = (
            int(np.sum((y > 0.5) & slot1_mask))
            if (slot1_mask is not None and len(slot1_mask) == n_samples)
            else n_pos
        )
        if n_samples < 40 or n_pos < min_positives or len(np.unique(y)) < 2:
            self.slot1_weights = s1_prior.copy()
            self.slot1_ml_blend_alpha = 0.0
            return self

        X64 = X.astype(np.float64, copy=False)
        y_int = y.astype(np.int64, copy=False)
        sw = np.maximum(sample_weights.astype(np.float64, copy=False), 1e-4)
        if slot1_mask is not None and len(slot1_mask) == n_samples:
            sw = sw * np.where(slot1_mask, 3.0, 0.65)

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X64)
        clf = LogisticRegression(C=l2_c, solver="lbfgs", max_iter=500, random_state=43)
        clf.fit(X_std, y_int, sample_weight=sw)
        scales = np.maximum(scaler.scale_, 1e-3)
        raw_ml = clf.coef_[0].astype(np.float64, copy=True) / scales
        stabilized_s1 = self._anchor_weights(s1_prior, raw_ml, clamp_ext_disabled=True)
        if stabilized_s1 is None:
            self.slot1_weights = s1_prior.copy()
            self.slot1_ml_blend_alpha = 0.0
            return self

        ramp = min(1.0, max(0.0, (n_pos - min_positives) / 280.0))
        alpha = max_blend_alpha * ramp
        self.slot1_ml_blend_alpha = float(alpha)
        self.slot1_weights = (1.0 - alpha) * s1_prior + alpha * stabilized_s1
        return self

    def score_slot1(
        self,
        X: np.ndarray,
        *,
        base_lin_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """Score candidates specifically for Slot-1 precision (matched to linear dispersion)."""
        if X.shape[0] == 0 or self.slot1_weights is None or self.slot1_head_alpha <= 0.0:
            return (
                base_lin_scores
                if base_lin_scores is not None
                else np.zeros(X.shape[0], dtype=np.float64)
            )
        d = min(int(X.shape[1]), int(self.slot1_weights.shape[0]))
        s1_raw = X[:, :d] @ self.slot1_weights[:d]
        base = (
            base_lin_scores
            if base_lin_scores is not None
            else (X[:, :d] @ self.effective_weights[:d])
        )
        n_c = len(s1_raw)
        if n_c > 1:
            bm = float(base.sum()) / n_c
            sm = float(s1_raw.sum()) / n_c
            db = base - bm
            ds = s1_raw - sm
            base_sd = math.sqrt(float(np.dot(db, db)) / n_c)
            s1_sd = math.sqrt(float(np.dot(ds, ds)) / n_c)
            if base_sd < 1e-4:
                base_sd = 1e-4
            if s1_sd < 1e-4:
                s1_sd = 1e-4
            s1_aligned = bm + ds * (base_sd / s1_sd)
        else:
            s1_aligned = s1_raw
        a = self.slot1_head_alpha
        return (1.0 - a) * base + a * s1_aligned

    def with_updated_priors(
        self,
        prior_weights: np.ndarray,
        min_prior_ratios: np.ndarray,
        max_prior_multiplier: float,
    ) -> PositionDebiasedRanker:
        """Clone ranker updating prior weights while preserving fitted ML, MLP, transfer & Slot-1 layers."""
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
            slot1_prior_weights=self.slot1_prior_weights,
            slot1_weights=self.slot1_weights,
            slot1_ml_blend_alpha=self.slot1_ml_blend_alpha,
            slot1_head_alpha=self.slot1_head_alpha,
            slot1_pos_samples=self.slot1_pos_samples,
            pretrain_lin_weights=self.pretrain_lin_weights,
            syn_pretrained=self.syn_pretrained,
            transfer_fine_tuned=self.transfer_fine_tuned,
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
            "syn_pretrained": self.syn_pretrained,
            "transfer_fine_tuned": self.transfer_fine_tuned,
            "slot1_head_alpha": round(self.slot1_head_alpha, 4),
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
    return_slot_idx: bool = False,
) -> tuple[Any, ...]:
    """Convert feedback/session impression logs into IPS-weighted (X, y, weights[, slot_idx])."""
    feat_rows: list[np.ndarray] = []
    labels: list[int] = []
    sample_weights: list[float] = []
    slot_indices: list[int] = []

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
                slot_indices.append(slot_idx)
            elif pid == clicked and click_weight > 0.0:
                feat_rows.append(feat_mat[slot_idx])
                labels.append(1)
                sample_weights.append(ips_w * click_weight)
                slot_indices.append(slot_idx)
            else:
                feat_rows.append(feat_mat[slot_idx])
                labels.append(0)
                sample_weights.append(exam_p)
                slot_indices.append(slot_idx)

    if not feat_rows:
        empty = (
            np.empty((0, NUM_FEATURES), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
        return (*empty, np.empty(0, dtype=np.int64)) if return_slot_idx else empty

    out = (
        np.vstack(feat_rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sample_weights, dtype=np.float64),
    )
    return (*out, np.asarray(slot_indices, dtype=np.int64)) if return_slot_idx else out
