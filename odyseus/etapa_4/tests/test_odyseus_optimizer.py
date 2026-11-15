"""Unit tests for the Odyseus Calypso 5.1f adaptive iterative conversion optimizer."""
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
    MODEL_VERSION_LABEL,
    PARAM_GROUPS,
    PARAM_SPECS,
    _is_param_relevant,
    _prepare_structural_tables,
    _strip_future_test_purchases,
    _update_snapshot_for_profile,
    apply_param_updates,
    compute_adaptive_intelligence,
    get_param_value,
    run_odyseus_optimization,
)
from src.recommender import CalypsoRecommenderV4  # noqa: E402
from src.weights_profile import WeightsProfile, get_preset  # noqa: E402


class TestOdyseusOptimizer51(unittest.TestCase):
    """Verify full 5.1 parameter space, structural updates, adaptive selector, and single-file archive."""

    def test_full_5_1_and_segment_roundtrip(self) -> None:
        wp = get_preset("conversion_first")
        updated = apply_param_updates(
            wp,
            {
                "w_global_pop": 0.68,
                "sales_rank_blend_weight": 0.72,
                "transition_dest_pop_gamma": 0.45,
                "cart_active_score_bonus": 1.35,
                "cold_explore_start_slot_cold": 2,
                "cold_sparse_cos_weight": 0.70,
                "tie_break_mode": "cheap",
                "seg_known_with_views__pop": 1.45,
                "seg_new_no_views__ctx": 1.70,
            },
        )
        self.assertAlmostEqual(get_param_value(updated, "sales_rank_blend_weight"), 0.72)
        self.assertAlmostEqual(get_param_value(updated, "transition_dest_pop_gamma"), 0.45)
        self.assertAlmostEqual(get_param_value(updated, "cart_active_score_bonus"), 1.35)
        self.assertEqual(get_param_value(updated, "cold_explore_start_slot_cold"), 2)
        self.assertAlmostEqual(updated.cold_dense_svd_weight, 0.30)
        self.assertAlmostEqual(get_param_value(updated, "seg_known_with_views__pop"), 1.45)
        self.assertGreater(len(PARAM_SPECS), 85)
        self.assertGreaterEqual(len(PARAM_GROUPS), 8)

    def test_structural_5_1_knobs_update_snapshot(self) -> None:
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
        v4_rec = syn["v4_records"]
        struct_prep = _prepare_structural_tables(
            snap,
            v4_rec["popularity_rows"],
            v4_rec["transition_rows"],
        )
        base_alpha = float(snap.agent_overrides.bandit_ucb_alpha)
        wp = apply_param_updates(
            get_preset("conversion_first"),
            {
                "bandit_ucb_alpha": 0.28,
                "sales_rank_blend_weight": 0.90,
                "transition_dest_pop_gamma": 0.75,
            },
        )
        new_snap, new_cfg = _update_snapshot_for_profile(
            snap,
            wp,
            base_ucb_alpha=base_alpha,
            struct_prep=struct_prep,
        )
        self.assertEqual(new_cfg.model_version, MODEL_VERSION_LABEL)
        sample_pid = next(iter(snap.active_ids))
        self.assertNotEqual(new_snap.product_static[sample_pid][7], snap.product_static[sample_pid][7])
        pop_diffs = sum(
            1
            for pid in snap.active_ids
            if abs(new_snap.global_pop_norm.get(pid, 0.0) - snap.global_pop_norm.get(pid, 0.0)) > 1e-6
        )
        self.assertGreater(pop_diffs, 10)

    def test_adaptive_intelligence_rewards_winning_parameters(self) -> None:
        fake_iters = [
            {"iter": 0, "objective": 0.400, "accepted": True, "changed_params": {}},
            {
                "iter": 1,
                "objective": 0.435,
                "accepted": True,
                "changed_params": {"sales_rank_blend_weight": {"old": 0.32, "new": 0.65}},
            },
            {
                "iter": 2,
                "objective": 0.390,
                "accepted": False,
                "changed_params": {"mmr_brand_lambda": {"old": 0.03, "new": 0.12}},
            },
        ]
        intel = compute_adaptive_intelligence(fake_iters)
        self.assertGreater(
            intel["param_scores"]["sales_rank_blend_weight"],
            intel["param_scores"]["mmr_brand_lambda"],
        )
        self.assertGreater(
            intel["group_scores"]["base_popularity_5_1"],
            intel["group_scores"]["diversity_mmr"],
        )

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
            self.assertEqual(archive1["model"], "calypso-5.1f")
            self.assertEqual(archive1["selector_mode"], "adaptive_ucb_bandit")
            self.assertEqual(archive1["total_iterations"], 4)
            self.assertIn("selector_summary", archive1)
            for it in archive1["iterations"]:
                self.assertIn("params", it)
                self.assertIn("metrics", it)
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
            self.assertEqual(archive2["total_iterations"], 6)
            self.assertEqual([it["iter"] for it in archive2["iterations"]], [0, 1, 2, 3, 4, 5])
            self.assertGreaterEqual(res2["best_objective"], res1["best_objective"])

    def test_calibrated_and_hybrid_scoring_modes_and_relevance_filter(self) -> None:
        syn = generate_synthetic_dataset_v4(
            seed=13,
            n_warm_products=80,
            n_new_products=8,
            n_warm_users=60,
            n_history_sessions=160,
            n_feedback_sessions=40,
            n_test_sessions=20,
        )
        snap = syn["snap_v4"]
        v4_rec = syn["v4_records"]
        struct_prep = _prepare_structural_tables(
            snap,
            v4_rec["popularity_rows"],
            v4_rec["transition_rows"],
        )
        base_alpha = float(snap.agent_overrides.bandit_ucb_alpha)
        row = syn["df_test"].iloc[0]

        for preset_name in ("calibrated_logit", "calibrated_hybrid"):
            wp = get_preset(preset_name)
            new_snap, new_cfg = _update_snapshot_for_profile(
                snap,
                wp,
                base_ucb_alpha=base_alpha,
                struct_prep=struct_prep,
            )
            self.assertTrue(len(new_snap.sales_rank) > 0)
            rec = CalypsoRecommenderV4(new_cfg)
            out = rec.recommend(row, new_snap)
            self.assertEqual(len(out), 10)
            self.assertEqual(len(set(out)), 10)

        self.assertFalse(_is_param_relevant("cal_w_rank", get_preset("conversion_first")))
        self.assertTrue(_is_param_relevant("w_global_pop", get_preset("conversion_first")))
        self.assertFalse(_is_param_relevant("w_global_pop", get_preset("calibrated_logit")))
        self.assertTrue(_is_param_relevant("cal_w_rank", get_preset("calibrated_logit")))

    def test_duration_budget_and_report_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            history_path = tmp_dir / "odyseus_history.json"
            base_wp = get_preset("conversion_first")
            res = run_odyseus_optimization(
                MODEL_ROOT,
                base_profile=base_wp,
                history_path=history_path,
                eval_mode="synthetic",
                iters=100,
                duration_seconds=0.35,
                seed=7,
                n_syn=80,
                eval_presets_first=False,
                export_report=True,
            )
            self.assertGreaterEqual(res["total_iterations"], 2)
            self.assertLess(res["total_iterations"], 100)
            self.assertTrue((tmp_dir / "odyseus_iterations.csv").is_file())
            self.assertTrue((tmp_dir / "INFORME_ODYSEUS_5_1F.md").is_file())
            csv_lines = (tmp_dir / "odyseus_iterations.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(csv_lines), res["total_iterations"] + 1)


if __name__ == "__main__":
    unittest.main()
