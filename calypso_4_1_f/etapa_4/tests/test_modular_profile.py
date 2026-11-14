"""Unit tests for Calypso 4.1.e modular WeightsProfile, presets, signed ML, hot-reload, and tuner."""
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

from src.config import ConfigV4  # noqa: E402
from src.disk_cache import DiskCache  # noqa: E402
from src.eval import generate_synthetic_dataset_v4  # noqa: E402
from src.recommender import CalypsoRecommenderV4, build_snapshot_v4_from_records  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
