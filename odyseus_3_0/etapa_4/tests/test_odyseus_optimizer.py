"""Unit tests for the Odyseus offline iterative conversion optimizer."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

import pandas as pd

ETAPA4_DIR = Path(__file__).resolve().parents[1]
MODEL_ROOT = ETAPA4_DIR.parent
for _p in (MODEL_ROOT, ETAPA4_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.eval import generate_synthetic_dataset_v4  # noqa: E402
from src.odyseus_optimizer import (  # noqa: E402
    PARAM_SPECS,
    _strip_future_test_purchases,
    _update_snapshot_for_profile,
    apply_param_updates,
    get_param_value,
    run_odyseus_optimization,
)
from src.weights_profile import WeightsProfile, get_preset  # noqa: E402


class TestOdyseusOptimizer(unittest.TestCase):
    """Verify full-space parameter updates, leakage stripping, and single-file iteration archive."""

    def test_segment_and_direct_param_roundtrip(self) -> None:
        wp = get_preset("conversion_first")
        updated = apply_param_updates(
            wp,
            {
                "w_global_pop": 0.75,
                "v4_price_ascending": False,
                "tie_break_mode": "cheap",
                "seg_known_with_views__pop": 1.45,
                "seg_new_no_views__ctx": 1.70,
            },
        )
        self.assertAlmostEqual(get_param_value(updated, "w_global_pop"), 0.75)
        self.assertFalse(get_param_value(updated, "v4_price_ascending"))
        self.assertEqual(get_param_value(updated, "tie_break_mode"), "cheap")
        self.assertAlmostEqual(get_param_value(updated, "seg_known_with_views__pop"), 1.45)
        self.assertAlmostEqual(get_param_value(updated, "seg_new_no_views__ctx"), 1.70)
        self.assertGreater(len(PARAM_SPECS), 60)

    def test_ucb_alpha_updates_product_static(self) -> None:
        syn = generate_synthetic_dataset_v4(
            seed=9,
            n_warm_products=80,
            n_new_products=8,
            n_warm_users=80,
            n_history_sessions=200,
            n_feedback_sessions=60,
            n_test_sessions=40,
        )
        snap = syn["snap_v4"]
        base_alpha = float(snap.agent_overrides.bandit_ucb_alpha)
        wp = apply_param_updates(get_preset("conversion_first"), {"bandit_ucb_alpha": 0.32})
        new_snap, _ = _update_snapshot_for_profile(snap, wp, base_ucb_alpha=base_alpha)
        sample_pid = next(iter(snap.active_ids))
        self.assertNotEqual(new_snap.product_static[sample_pid][7], snap.product_static[sample_pid][7])

    def test_strip_future_test_purchases(self) -> None:
        syn = generate_synthetic_dataset_v4(
            seed=5,
            n_warm_products=60,
            n_new_products=6,
            n_warm_users=40,
            n_history_sessions=150,
            n_feedback_sessions=40,
            n_test_sessions=30,
        )
        snap = syn["snap_v4"]
        uid = next(u for u, prof in snap.user_profiles.items() if prof.bought_products)
        bought_pid = next(iter(snap.user_profiles[uid].bought_products))
        df_fake = pd.DataFrame([{"user_id": uid, "purchased": bought_pid}])
        clean_snap = _strip_future_test_purchases(snap, df_fake)
        self.assertNotIn(bought_pid, clean_snap.user_profiles[uid].bought_products)

    def test_single_file_history_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            history_path = tmp_dir / "odyseus_history.json"
            profile_path = tmp_dir / "weights_profile.json"
            base_wp = get_preset("conversion_first")

            res1 = run_odyseus_optimization(
                MODEL_ROOT,
                base_profile=base_wp,
                history_path=history_path,
                eval_mode="synthetic",
                iters=3,
                seed=11,
                n_syn=120,
                eval_presets_first=False,
                save_best=True,
                target_profile_path=profile_path,
            )
            self.assertTrue(history_path.is_file())
            self.assertTrue(profile_path.is_file())
            archive1 = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual(archive1["total_iterations"], 4)  # iter 0 + 3 iters
            self.assertEqual(len(archive1["iterations"]), 4)
            for it in archive1["iterations"]:
                self.assertIn("params", it)
                self.assertIn("metrics", it)
                self.assertIn("objective", it)
                WeightsProfile.from_dict(it["params"])

            res2 = run_odyseus_optimization(
                MODEL_ROOT,
                base_profile=base_wp,
                history_path=history_path,
                eval_mode="synthetic",
                iters=2,
                seed=12,
                n_syn=120,
                eval_presets_first=False,
                resume=True,
            )
            archive2 = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual(res2["total_iterations"], 6)
            self.assertEqual(len(archive2["iterations"]), 6)
            self.assertGreaterEqual(res2["best_objective"], res1["best_objective"])


if __name__ == "__main__":
    unittest.main()
