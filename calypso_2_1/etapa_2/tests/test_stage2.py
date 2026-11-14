"""Unit and integration verification tests for Etapa 2."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ETAPA2_DIR = Path(__file__).resolve().parents[1]
if str(ETAPA2_DIR) not in sys.path:
    sys.path.insert(0, str(ETAPA2_DIR))

from src.cold_start import ColdStartIndexV2, build_cold_start_scratch_records
from src.config import ConfigV2
from src.eval import generate_synthetic_dataset_v2, run_benchmark_v2
from src.guardrails import validate_submission_record
from src.recommender import Stage2Recommender, build_snapshot_v2_from_records
from src.serve import FastServingLoopV2
from src.state import ProductInfo, StateStore


class TestColdStartV2(unittest.TestCase):
    def test_bidirectional_cold_start_forward_and_reverse_grafting(self) -> None:
        products = {
            f"p_w_{i}": ProductInfo(
                f"p_w_{i}",
                "Electronics",
                "Headphones" if i < 6 else "Coffee Makers",
                "Sonic" if i < 3 else "Barista",
                100.0 + i,
                100.0 + i,
                (
                    "Wireless bluetooth noise cancelling studio headphones"
                    if i < 6
                    else "Espresso coffee maker stainless grinder"
                ),
                True,
                False,
            )
            for i in range(12)
        }
        products["p_new_hp"] = ProductInfo(
            "p_new_hp",
            "Electronics",
            "Headphones",
            "Sonic",
            102.0,
            102.0,
            "Brand new wireless bluetooth ANC studio headphones",
            True,
            True,
        )

        warm_ids = [f"p_w_{i}" for i in range(12)]
        idx = ColdStartIndexV2(svd_dim=8).fit(products, warm_ids=warm_ids)
        fwd, rev = idx.compute_bidirectional(
            forward_targets=[products["p_new_hp"]],
            new_products=[products["p_new_hp"]],
            top_k_forward=3,
            top_k_reverse=2,
        )
        # Forward: p_new_hp maps to headphones (p_w_0..p_w_2), not coffee makers
        fwd_ids = [pid for pid, _ in fwd["p_new_hp"]]
        self.assertTrue(all(pid in {f"p_w_{i}" for i in range(6)} for pid in fwd_ids))

        # Reverse: viewing warm headphone p_w_0 grafts p_new_hp
        rev_ids = [pid for pid, _ in rev.get("p_w_0", ())]
        self.assertIn("p_new_hp", rev_ids)

        scratch = build_cold_start_scratch_records(
            fwd, products, model_version="etapa2-test", ts_iso="2026-11-14T13:00:00Z"
        )
        self.assertEqual(len(scratch), 1)
        self.assertEqual(scratch[0]["product_id"], "p_new_hp")


class TestPersonalizationAndMMR(unittest.TestCase):
    def test_intent_recovery_live_bought_suppression_and_mmr_diversity(self) -> None:
        catalog = []
        # 6 items in subcat A (brand X), 6 items in subcat B (brand Y), 6 items in subcat C (brand Z)
        for i in range(18):
            sc = ["Audio", "Shoes", "Coffee"][i // 6]
            br = ["Sonic", "Stride", "Crema"][i // 6]
            catalog.append(
                {
                    "product_id": f"p_{i:02d}",
                    "category": "General",
                    "subcategory": sc,
                    "brand": br,
                    "price_eur": 50.0 + i,
                    "description": f"{br} {sc} item {i}",
                    "is_active": True,
                }
            )
        pop = [{"product_id": f"p_{i:02d}", "views": 100 - i, "purchases": 20 - i, "weighted_pop": 200 - i * 5} for i in range(18)]
        # User previously added p_14 (Coffee) to cart without buying, and bought p_00 already
        profiles = [{"user_id": "u_1", "bought_products": ["p_00"], "log_price_mean": 4.1, "log_price_std": 0.3}]
        intent = [{"user_id": "u_1", "product_id": "p_14", "intent_score": 5.0}]

        snap = build_snapshot_v2_from_records(
            catalog_rows=catalog,
            popularity_rows=pop,
            user_profile_rows=profiles,
            user_intent_rows=intent,
        )
        # Simulate live purchase of p_01 from feedback -> must NOT appear in recommendations
        snap_live = snap.with_fast_updates(live_user_bought={"u_1": frozenset({"p_01"})})

        rec = Stage2Recommender(ConfigV2())
        items = rec.recommend(
            {"request_id": "r_1", "user_id": "u_1", "is_new_user": False, "context": {"page": "home"}, "recent_views": []},
            snap_live,
        )

        self.assertEqual(len(items), 10)
        self.assertEqual(len(set(items)), 10)
        self.assertNotIn("p_00", items)
        self.assertNotIn("p_01", items)
        # Unbought intent item p_14 should rank right at the top
        self.assertIn("p_14", items[:3])
        # MMR should ensure more than 1 subcategory appears across the 10 slots
        subcats_used = {snap_live.products[p].subcategory for p in items}
        self.assertGreaterEqual(len(subcats_used), 2)


class TestStage2BenchmarkAndServing(unittest.TestCase):
    def test_etapa2_beats_baseline_and_etapa1_within_latency_budget(self) -> None:
        results = run_benchmark_v2(seed=42)

        all_base_hr = float(results.loc[("ALL", "baseline"), "hit_rate_buy@10"])
        all_e1_hr = float(results.loc[("ALL", "etapa1_hybrid"), "hit_rate_buy@10"])
        all_e2_hr = float(results.loc[("ALL", "etapa2_seq_mmr"), "hit_rate_buy@10"])
        self.assertGreater(all_e2_hr, all_base_hr)
        self.assertGreaterEqual(all_e2_hr, all_e1_hr)

        all_base_snips = float(results.loc[("ALL", "baseline"), "SNIPS_cvr_slot1"])
        all_e2_snips = float(results.loc[("ALL", "etapa2_seq_mmr"), "SNIPS_cvr_slot1"])
        self.assertGreater(all_e2_snips, all_base_snips)

        cold_base_hr = float(results.loc[("COLD_START", "baseline"), "hit_rate_buy@10"])
        cold_e2_hr = float(results.loc[("COLD_START", "etapa2_seq_mmr"), "hit_rate_buy@10"])
        self.assertGreater(cold_e2_hr, cold_base_hr)

        mean_latency_ms = float(results.loc[("ALL", "etapa2_seq_mmr"), "latency_ms_mean"])
        self.assertLess(mean_latency_ms, 1.0)

    def test_fast_serving_loop_v2_and_config_env(self) -> None:
        cfg = ConfigV2.from_env()
        self.assertTrue(bool(cfg.team_id))
        self.assertTrue(bool(cfg.api_key))

        syn = generate_synthetic_dataset_v2(seed=9, n_test_sessions=80)
        snap = syn["snapshot_v2"]
        top_item = snap.global_ranked[0]
        snap = snap.with_fast_updates(out_of_stock=frozenset({top_item}))

        loop = FastServingLoopV2(
            client=None,
            store=StateStore(snap),
            recommender=Stage2Recommender(cfg),
            config=cfg,
        )
        batch = syn["df_test"].head(50).to_dict(orient="records")
        t0 = time.perf_counter()
        answers = loop.answer_batch(batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(len(answers), 50)
        self.assertLess(elapsed_ms, 150.0)
        for ans in answers:
            self.assertTrue(validate_submission_record(ans, snap.active_ids))
            self.assertNotIn(top_item, ans["items"])
        self.assertEqual(loop.answer_batch(batch), [])


if __name__ == "__main__":
    unittest.main()
