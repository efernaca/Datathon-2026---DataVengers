"""Automated test suite for Stage 3 (calypso-2.1)."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ETAPA3_DIR = Path(__file__).resolve().parents[1]
if str(ETAPA3_DIR) not in sys.path:
    sys.path.insert(0, str(ETAPA3_DIR))

import numpy as np

from src.agent_scratch import CalypsoAgent
from src.bandits import PositionAwareBandit, slot_examination_weights
from src.config import ConfigV3
from src.eval import evaluate_policy, generate_synthetic_dataset_v3
from src.guardrails import validate_submission_record
from src.recommender import CalypsoRecommender, extract_recent_product_ids
from src.reranker import FEATURE_NAMES, PositionDebiasedRanker
from src.serve import CalypsoFastLoop
from src.state import ProductInfo, StateStore


class TestStage3Calypso(unittest.TestCase):
    """Core unit, ML debiasing, bandit exploration, agent, and benchmark tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.syn = generate_synthetic_dataset_v3(
            seed=42,
            n_warm_products=300,
            n_new_products=30,
            n_warm_users=600,
            n_history_sessions=2400,
            n_feedback_sessions=800,
            n_test_sessions=900,
        )

    def test_config_and_model_version_calypso(self) -> None:
        cfg = ConfigV3()
        self.assertEqual(cfg.model_version, "calypso-3.1")
        cfg_env = ConfigV3.from_env()
        self.assertEqual(cfg_env.model_version, "calypso-3.1")
        self.assertTrue(bool(cfg_env.api_url))

    def test_recent_views_parsing_both_formats(self) -> None:
        raw_dicts = [{"product_id": "p_0001", "ts": "2026-11-14"}, {"product_id": "p_0002"}]
        raw_strs = ["p_0001", "p_0002"]
        self.assertEqual(extract_recent_product_ids(raw_dicts), ["p_0001", "p_0002"])
        self.assertEqual(extract_recent_product_ids(raw_strs), ["p_0001", "p_0002"])
        self.assertEqual(extract_recent_product_ids(None), [])

    def test_position_aware_bandit_debiasing_and_cold_ucb(self) -> None:
        weights = slot_examination_weights(k_slots=10, exam_power=0.75)
        self.assertAlmostEqual(weights[0], 1.0, places=6)
        self.assertGreater(weights[0], weights[4])
        self.assertGreater(weights[4], weights[9])

        bandit = PositionAwareBandit()
        # 'p_slot8' converts at slot 8 (low examination); 'p_slot0' only converts once in slot 0
        fb_rows = []
        for _ in range(40):
            fb_rows.append(
                {
                    "items": ["p_slot0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "p_slot8", "f9"],
                    "clicked": "p_slot8",
                    "purchased": "p_slot8",
                }
            )
        state = bandit.fit_from_feedback(fb_rows)
        adj_slot8 = bandit.score_adjustment("p_slot8", is_new=False, state=state, ucb_alpha=0.14)
        adj_slot0 = bandit.score_adjustment("p_slot0", is_new=False, state=state, ucb_alpha=0.14)
        adj_cold_unseen = bandit.score_adjustment("p_cold", is_new=True, state=state, ucb_alpha=0.14)

        self.assertGreater(adj_slot8, adj_slot0)
        self.assertGreater(adj_cold_unseen, adj_slot0)

    def test_position_debiased_ranker_trains_cleanly(self) -> None:
        snap = self.syn["snap_v3"]
        ranker: PositionDebiasedRanker = snap.ranker
        self.assertEqual(len(FEATURE_NAMES), 14)
        self.assertGreater(ranker.train_samples, 100)
        self.assertGreater(ranker.ml_blend_alpha, 0.05)
        # Personalization weights remain strictly positive under monotonic floor guards
        self.assertGreater(ranker.effective_weights[0], 0.5)
        self.assertGreater(ranker.effective_weights[1], 1.0)

        rng = np.random.default_rng(7)
        X = rng.uniform(0.0, 1.0, size=(12, len(FEATURE_NAMES)))
        scores = ranker.score_matrix(X)
        self.assertEqual(scores.shape, (12,))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_calypso_agent_regime_and_scratch_records(self) -> None:
        agent = CalypsoAgent()
        cfg = ConfigV3()
        products = {
            f"p_{i}": ProductInfo(
                product_id=f"p_{i}",
                category="Electronics",
                subcategory="Audio",
                brand="Sonic",
                price_eur=50.0,
                description="wireless anc",
                is_active=True,
                is_new=(i >= 70),
            )
            for i in range(100)
        }
        active_ids = frozenset(products.keys())
        new_ids = frozenset(f"p_{i}" for i in range(70, 100))
        ranker = PositionDebiasedRanker.from_priors()
        b_state = self.syn["snap_v3"].bandit_state

        overrides, telemetry, base_records = agent.diagnose_and_adapt(
            cfg,
            products,
            active_ids,
            new_ids,
            feedback_rows=(),
            bandit_state=b_state,
            ranker=ranker,
        )
        self.assertEqual(overrides.regime, "cold_surge")
        self.assertGreater(overrides.bandit_ucb_alpha, cfg.bandit_ucb_alpha)
        self.assertGreater(telemetry.new_product_share, 0.25)

        full_scratch = agent.build_full_scratch_payload(
            cfg,
            PositionAwareBandit(),
            b_state,
            {"p_70": (("p_0", 0.9),)},
            products,
            overrides,
            base_records,
        )
        kinds = {r["kind"] for r in full_scratch}
        self.assertIn("agent_decision", kinds)
        self.assertIn("reranker_weights", kinds)
        self.assertIn("bandit_arm", kinds)
        self.assertIn("cold_start_neighbors", kinds)

    def test_recommender_guardrails_and_benchmark_lift(self) -> None:
        snap_v3 = self.syn["snap_v3"]
        rec_v3 = CalypsoRecommender(ConfigV3())
        store = StateStore(snap_v3)
        fast_loop = CalypsoFastLoop(client=None, store=store, recommender=rec_v3, config=ConfigV3())

        test_rows = self.syn["df_test"].to_dict(orient="records")[:40]
        answers = fast_loop.answer_batch(test_rows)
        self.assertEqual(len(answers), 40)
        for ans in answers:
            self.assertEqual(ans["model_version"], "calypso-3.1")
            self.assertTrue(validate_submission_record(ans, snap_v3.active_ids))

        # Benchmark calypso-2.1 vs baseline on synthetic holdout
        df_test = self.syn["df_test"]
        baseline_top10 = self.syn["baseline_top10"]
        res_base = evaluate_policy(
            lambda _: list(baseline_top10),
            df_test,
            cold_product_ids=self.syn["cold_product_ids"],
        )
        res_calypso = evaluate_policy(
            lambda row: rec_v3.recommend(row, snap_v3),
            df_test,
            cold_product_ids=self.syn["cold_product_ids"],
        )
        self.assertGreater(res_calypso["hit_rate_buy@10"], res_base["hit_rate_buy@10"] * 1.4)
        self.assertGreater(res_calypso["cold_item_hit@10"], res_base["cold_item_hit@10"])
        self.assertLess(res_calypso["latency_ms_mean"], 5.0)


if __name__ == "__main__":
    unittest.main()
