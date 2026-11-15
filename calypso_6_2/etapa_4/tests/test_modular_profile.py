"""Unit tests for Calypso 6.2 modular WeightsProfile, presets, 6.2 hybrid architecture, hot-reload, and tuner."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
import unittest

import numpy as np

ETAPA4_DIR = Path(__file__).resolve().parents[1]
CALYPSO_ROOT = ETAPA4_DIR.parent
for _p in (CALYPSO_ROOT, ETAPA4_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.bandits import EMPIRICAL_SLOT_WEIGHTS, PositionAwareBandit  # noqa: E402
from src.config import ConfigV4  # noqa: E402
from src.disk_cache import DiskCache  # noqa: E402
from src.eval import generate_synthetic_dataset_v4  # noqa: E402
from src.recommender import (  # noqa: E402
    CalypsoRecommenderV4,
    _prepare_request_signals,
    build_snapshot_v4_from_records,
)
from src.reranker import NUM_FEATURES, PositionDebiasedRanker  # noqa: E402
from src.serve import Stage4BackgroundUpdater  # noqa: E402
from src.state import StateStore  # noqa: E402
from src.tuner import evaluate_profile_on_dataset, optimize_profile, parse_set_assignments  # noqa: E402
from src.weights_profile import PRESETS, WeightsProfile, get_preset, list_presets  # noqa: E402


class _DummyClient:
    """Minimal RawTreeClient stub for updater hot-reload test."""

    def query_safe(self, sql: str, timeout: float = 1.0) -> list[dict]:
        return []


class TestModularWeightsProfile(unittest.TestCase):
    """Verify modular profile persistence, signed weights, ranking impact, hot-reload, and tuner."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.syn = generate_synthetic_dataset_v4(
            seed=17,
            n_warm_products=140,
            n_new_products=14,
            n_warm_users=160,
            n_history_sessions=500,
            n_feedback_sessions=160,
            n_test_sessions=120,
        )

    def test_presets_and_json_roundtrip(self) -> None:
        expected = {
            "conversion_first",
            "cheap_boost",
            "cheap_hard_sort",
            "user_trend_focus",
            "product_profile_focus",
            "calibrated_m3",
        }
        self.assertEqual(set(list_presets()), expected)
        with tempfile.TemporaryDirectory() as tmp:
            for name in expected:
                wp = get_preset(name)
                out_path = Path(tmp) / f"{name}.json"
                wp.save_json(out_path)
                reloaded = WeightsProfile.load_json(out_path)
                self.assertEqual(reloaded, wp)
                self.assertEqual(reloaded.prior_weights_array().shape, (NUM_FEATURES,))
                self.assertEqual(reloaded.min_prior_ratios_array().shape, (NUM_FEATURES,))

    def test_signed_negative_weight_survives_ml_fit(self) -> None:
        cheap_wp = get_preset("cheap_boost")
        self.assertLess(cheap_wp.w_log_price_norm, 0.0)
        ranker = PositionDebiasedRanker.from_priors(
            prior_weights=cheap_wp.prior_weights_array(),
            min_prior_ratios=cheap_wp.min_prior_ratios_array(),
            max_prior_multiplier=cheap_wp.max_prior_multiplier,
        )
        rng = np.random.default_rng(123)
        X = rng.normal(size=(240, NUM_FEATURES))
        # Positive outcome prefers feature 0 and lower feature 11 (log_price_norm)
        logits = 1.5 * X[:, 0] - 0.8 * X[:, 11]
        y = (logits + rng.normal(scale=0.3, size=240) > 0.0).astype(np.int64)
        sw = np.ones(240, dtype=np.float64)
        ranker.fit(X, y, sw, max_blend_alpha=0.45, min_positives=20, l2_c=1.2)
        self.assertGreater(ranker.ml_blend_alpha, 0.0)
        # Negative prior weight must stay negative (not clipped to 0.0)
        self.assertLess(float(ranker.effective_weights[11]), -0.02)

    def test_cheap_boost_lowers_top3_price_vs_conversion_first(self) -> None:
        wp_conv = get_preset("conversion_first")
        wp_cheap = parse_set_assignments(
            ["w_log_price_norm=-0.35", "price_cheap_asymmetry=0.75", "tie_break_mode=cheap"],
            get_preset("cheap_boost"),
        )
        records = self.syn["v4_records"]
        snap_conv, _ = build_snapshot_v4_from_records(
            **records, config=ConfigV4.from_profile(wp_conv), weights_profile=wp_conv
        )
        snap_cheap, _ = build_snapshot_v4_from_records(
            **records, config=ConfigV4.from_profile(wp_cheap), weights_profile=wp_cheap
        )
        rec_conv = CalypsoRecommenderV4(ConfigV4.from_profile(wp_conv))
        rec_cheap = CalypsoRecommenderV4(ConfigV4.from_profile(wp_cheap))

        test_rows = self.syn["df_test"].to_dict(orient="records")[:60]
        prices_conv: list[float] = []
        prices_cheap: list[float] = []
        for req in test_rows:
            out_conv = rec_conv.recommend(req, snap_conv)
            out_cheap = rec_cheap.recommend(req, snap_cheap)
            prices_conv.append(
                sum(snap_conv.products[p].price_eur for p in out_conv[:3]) / 3.0
            )
            prices_cheap.append(
                sum(snap_cheap.products[p].price_eur for p in out_cheap[:3]) / 3.0
            )
        self.assertLess(sum(prices_cheap) / len(prices_cheap), sum(prices_conv) / len(prices_conv))

    def test_hot_reload_detects_profile_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wp_path = tmp_path / "weights_profile.json"
            get_preset("conversion_first").save_json(wp_path)
            cfg = ConfigV4.from_profile(get_preset("conversion_first"), weights_profile_path=str(wp_path))
            updater = Stage4BackgroundUpdater(
                client=_DummyClient(),  # type: ignore[arg-type]
                store=StateStore(),
                config=cfg,
                cache=DiskCache(tmp_path / "cache"),
                stop_event=threading.Event(),
            )
            self.assertEqual(updater._active_profile.profile_name, "conversion_first")
            self.assertFalse(updater._check_profile_reload())

            # Bump mtime explicitly after rewriting file to simulate live edit
            time.sleep(0.05)
            get_preset("cheap_hard_sort").save_json(wp_path)
            stat = wp_path.stat()
            os.utime(wp_path, (stat.st_atime + 2.0, stat.st_mtime + 2.0))

            self.assertTrue(updater._check_profile_reload())
            self.assertEqual(updater._active_profile.profile_name, "cheap_hard_sort")
            self.assertTrue(updater.config.v4_price_ascending)

    def test_tuner_optimize_and_eval(self) -> None:
        base_wp = get_preset("conversion_first")
        eval_res = evaluate_profile_on_dataset(self.syn, base_wp, rebuild_snapshot=False)
        self.assertGreater(float(eval_res["objective"]), 0.0)
        best_wp, best_eval, history = optimize_profile(
            base_wp, self.syn, iters=4, seed=42
        )
        self.assertTrue(best_wp.profile_name.endswith("_opt"))
        self.assertEqual(len(history), 5)
        self.assertGreater(float(best_eval["ALL"]["hit_rate_buy@10"]), 0.5)
        self.assertIn("cheap_boost", PRESETS)
        self.assertIn("calibrated_m3", PRESETS)

    def test_v5_1_conversion_measures(self) -> None:
        wp = get_preset("conversion_first")
        cfg = ConfigV4.from_profile(wp)
        records = self.syn["v4_records"]
        snap, _ = build_snapshot_v4_from_records(**records, config=cfg, weights_profile=wp)
        rec = CalypsoRecommenderV4(cfg)

        # (D3) Pool >= 75 and Top-N bestsellers preserved even with multiple recent_views
        warm_pool = [pid for pid in snap.fallback_ranked if pid not in snap.new_product_ids]
        rv = warm_pool[40:44] if len(warm_pool) > 44 else warm_pool[:3]
        cands = rec._gather_candidates(("u_x", False, "Madrid", "web", "product", rv), snap, frozenset())
        self.assertGreaterEqual(cfg.candidate_pool_size, 75)
        self.assertGreaterEqual(wp.guaranteed_top_sellers_in_pool, 12)
        self.assertGreaterEqual(len(cands), 75)
        cands_set = set(cands)
        expected_top = [
            pid for pid in snap.fallback_ranked[: wp.guaranteed_top_sellers_in_pool] if pid in snap.active_ids
        ]
        for pid in expected_top:
            self.assertIn(pid, cands_set)

        # (D6 / 6.1) Last viewed item receives page-contextual multiplier (product >= category)
        last_pid = warm_pool[0]
        prev_pid = warm_pool[1]
        _, sv_last_on_product, *_ = _prepare_request_signals(
            snap, ("u_x", False, "Madrid", "web", "product", [prev_pid, last_pid]), weights_profile=wp
        )
        _, sv_last_on_category, *_ = _prepare_request_signals(
            snap, ("u_x", False, "Madrid", "web", "category", [prev_pid, last_pid]), weights_profile=wp
        )
        self.assertAlmostEqual(
            sv_last_on_product[last_pid],
            sv_last_on_category[last_pid] * wp.view_last_product_page_factor,
            places=6,
        )

        # (X10) New products without historical sales inherit positive global_pop_norm from neighbors
        cold_with_nbs = [pid for pid in snap.new_product_ids if snap.cold_neighbors.get(pid)]
        self.assertTrue(cold_with_nbs)
        self.assertTrue(any(snap.global_pop_norm.get(pid, 0.0) > 0.0 for pid in cold_with_nbs))

        # (M5) Empirical attention curve applied on PositionAwareBandit
        bandit_emp = PositionAwareBandit(use_empirical_exam_curve=wp.use_empirical_exam_curve)
        self.assertEqual(bandit_emp.exam_weights, EMPIRICAL_SLOT_WEIGHTS)

        # (M7) Softened subcategory MMR diversification on home page (<= 0.08)
        self.assertLessEqual(wp.mmr_subcat_lambda_home, 0.08)

    def test_calypso_6_2_hybrid_architecture(self) -> None:
        from src.recommender import _extract_feature_matrix
        from src.tuner import OPTIMIZABLE_PARAMS

        wp = PRESETS["conversion_first"]
        self.assertGreaterEqual(wp.scalar_count(), 250)
        self.assertEqual(NUM_FEATURES, 50)
        self.assertEqual(ConfigV4.from_profile(wp).candidate_pool_size, 92)
        for key in OPTIMIZABLE_PARAMS:
            self.assertTrue(hasattr(wp, key), f"Missing OPTIMIZABLE_PARAMS field {key}")

        snap = self.syn["snap_v4"]
        self.assertEqual(snap.ranker.prior_weights.shape, (50,))
        self.assertTrue(snap.ease_neighbors)
        self.assertTrue(snap.hop2_transitions)
        self.assertTrue(snap.subcat_top_items)
        self.assertTrue(snap.cat_top_items)
        self.assertGreater(snap.svd_item_vectors.shape[0], 0)
        self.assertEqual(snap.svd_item_vectors.shape[1], wp.svd_dim)

        # Verify Hybrid Layer 1 Linear IPS + Layer 2 Balanced IPS-Distilled NumPy MLP
        rng = np.random.default_rng(123)
        X_syn = rng.uniform(0.0, 1.0, size=(600, NUM_FEATURES))
        logits = 2.0 * X_syn[:, 0] + 2.5 * X_syn[:, 1] + 1.5 * X_syn[:, 1] * X_syn[:, 21] - 2.6
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_syn = (rng.random(600) < probs).astype(np.int64)
        sw_syn = np.ones(600, dtype=np.float64)
        rk = PositionDebiasedRanker.from_priors(
            prior_weights=wp.prior_weights_array(),
            min_prior_ratios=wp.min_prior_ratios_array(),
            max_prior_multiplier=wp.max_prior_multiplier,
        )
        rk.fit(
            X_syn,
            y_syn,
            sw_syn,
            max_blend_alpha=wp.ml_max_blend_alpha,
            min_positives=20,
            l2_c=1.2,
            nonlin_blend_alpha=wp.ml_nonlin_blend_alpha,
            nonlin_l2_alpha=wp.ml_nonlin_l2_alpha,
            enable_nonlin=True,
        )
        self.assertIsNotNone(rk.nl_W1)
        deltas = rk.score_nonlin_residuals(X_syn[:16])
        self.assertEqual(deltas.shape, (16,))

        # Verify differentiated page context matrices (3x8)
        p_prod = wp.page_multipliers("product")
        p_cat = wp.page_multipliers("category")
        p_home = wp.page_multipliers("home")
        self.assertEqual(len(p_prod), 8)
        self.assertEqual(len(p_cat), 8)
        self.assertEqual(len(p_home), 8)
        self.assertNotEqual(p_prod, p_cat)
        self.assertNotEqual(p_prod, p_home)

        # Verify repeat-view & temporal decay feature extraction from timestamped requests
        sample_pid = snap.fallback_ranked[0]
        req_ts_feat = {
            "user_id": "u_ts_check",
            "ts": "2026-11-11T12:00:00Z",
            "context": {"region": "Madrid", "device": "web", "page": "product"},
            "recent_views": [
                {"product_id": sample_pid, "ts": "2026-11-11T11:55:00Z"},
                {"product_id": sample_pid, "ts": "2026-11-11T11:58:00Z"},
            ],
        }
        mat_rep = _extract_feature_matrix(snap, req_ts_feat, [sample_pid], weights_profile=wp)
        self.assertEqual(mat_rep.shape, (1, 50))
        self.assertGreater(mat_rep[0, 40], 0.4)  # rv_repeat_count_norm
        self.assertGreater(mat_rep[0, 41], 0.8)  # rv_temporal_recency

        # Verify with_updated_priors preserves distilled MLP matrices
        cloned = rk.with_updated_priors(
            wp.prior_weights_array(),
            wp.min_prior_ratios_array(),
            wp.max_prior_multiplier,
        )
        self.assertIs(cloned.nl_W1, rk.nl_W1)
        self.assertEqual(cloned.prior_weights.shape, (50,))


if __name__ == "__main__":
    unittest.main()
