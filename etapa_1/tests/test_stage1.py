"""Unit and integration verification tests for Stage 1."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ETAPA1_DIR = Path(__file__).resolve().parents[1]
if str(ETAPA1_DIR) not in sys.path:
    sys.path.insert(0, str(ETAPA1_DIR))

from src.cold_start import ColdStartIndex
from src.config import Config
from src.eval import generate_synthetic_dataset, run_synthetic_benchmark
from src.guardrails import ensure_valid_submission, validate_submission_record
from src.recommender import Stage1Recommender, build_snapshot_from_records
from src.serve import FastServingLoop
from src.state import ProductInfo, StateStore


class TestGuardrails(unittest.TestCase):
    def test_always_returns_10_distinct_active_items(self) -> None:
        active = {f"p_{i:02d}" for i in range(15)}
        fallback = [f"p_{i:02d}" for i in range(15)]
        # Candidates contain duplicates, out-of-stock items, and already bought items
        candidates = ["p_00", "p_00", "p_oos", "p_01", "p_02"]
        exclude = {"p_01", "p_03", "p_04"}

        items = ensure_valid_submission(
            candidates=candidates,
            active_ids=active,
            fallback_ranked=fallback,
            exclude_ids=exclude,
            k=10,
        )
        self.assertEqual(len(items), 10)
        self.assertEqual(len(set(items)), 10)
        self.assertTrue(all(pid in active for pid in items))
        self.assertNotIn("p_oos", items)
        self.assertTrue(all(pid not in exclude for pid in items))
        self.assertTrue(
            validate_submission_record({"request_id": "r_1", "items": items}, active_ids=active)
        )

    def test_relaxes_excludes_when_catalog_tight(self) -> None:
        active = {f"p_{i:02d}" for i in range(10)}
        exclude = {f"p_{i:02d}" for i in range(5)}
        items = ensure_valid_submission(
            candidates=["p_05", "p_06"],
            active_ids=active,
            fallback_ranked=sorted(active),
            exclude_ids=exclude,
            k=10,
        )
        self.assertEqual(len(items), 10)
        self.assertEqual(len(set(items)), 10)
        self.assertTrue(all(pid in active for pid in items))


class TestColdStart(unittest.TestCase):
    def test_maps_new_product_to_matching_subcategory_and_description(self) -> None:
        products = {
            "p_audio_1": ProductInfo(
                "p_audio_1", "Electronics", "Headphones", "Sonic", 99.0, 99.0,
                "Wireless noise cancelling bluetooth headphones", True, False
            ),
            "p_audio_2": ProductInfo(
                "p_audio_2", "Electronics", "Headphones", "Aura", 109.0, 109.0,
                "Bluetooth over-ear ANC wireless headphones", True, False
            ),
            "p_shoe_1": ProductInfo(
                "p_shoe_1", "Sports", "Running Shoes", "Stride", 85.0, 85.0,
                "Waterproof trail running shoes", True, False
            ),
            "p_new_audio": ProductInfo(
                "p_new_audio", "Electronics", "Headphones", "Sonic", 105.0, 105.0,
                "New wireless bluetooth ANC headphones studio sound", True, True
            ),
        }
        idx = ColdStartIndex().fit(products, warm_ids=["p_audio_1", "p_audio_2", "p_shoe_1"])
        neighbors = idx.neighbors_for_batch([products["p_new_audio"]], top_k=2)["p_new_audio"]
        top_ids = [pid for pid, _ in neighbors]
        self.assertEqual(len(top_ids), 2)
        self.assertIn("p_audio_1", top_ids)
        self.assertIn("p_audio_2", top_ids)
        self.assertNotIn("p_shoe_1", top_ids)


class TestStage1EndToEnd(unittest.TestCase):
    def test_stage1_beats_baseline_and_meets_latency_budget(self) -> None:
        results = run_synthetic_benchmark(seed=42)

        all_base_hr = float(results.loc[("ALL", "baseline"), "hit_rate_buy@10"])
        all_s1_hr = float(results.loc[("ALL", "stage1_hybrid"), "hit_rate_buy@10"])
        self.assertGreater(all_s1_hr, all_base_hr)

        all_base_snips = float(results.loc[("ALL", "baseline"), "SNIPS_cvr_slot1"])
        all_s1_snips = float(results.loc[("ALL", "stage1_hybrid"), "SNIPS_cvr_slot1"])
        self.assertGreater(all_s1_snips, all_base_snips)

        cold_base_hr = float(results.loc[("COLD_START", "baseline"), "hit_rate_buy@10"])
        cold_s1_hr = float(results.loc[("COLD_START", "stage1_hybrid"), "hit_rate_buy@10"])
        self.assertGreater(cold_s1_hr, cold_base_hr)

        cold_base_snips = float(results.loc[("COLD_START", "baseline"), "SNIPS_cvr_slot1"])
        cold_s1_snips = float(results.loc[("COLD_START", "stage1_hybrid"), "SNIPS_cvr_slot1"])
        self.assertGreater(cold_s1_snips, cold_base_snips)

        mean_latency_ms = float(results.loc[("ALL", "stage1_hybrid"), "latency_ms_mean"])
        self.assertLess(mean_latency_ms, 0.5)

    def test_snapshot_fast_stock_update_and_serving_loop(self) -> None:
        syn = generate_synthetic_dataset(seed=11, n_test_sessions=60)
        snap = syn["snapshot"]
        # Put the #1 global item out of stock and verify fast swap excludes it immediately
        top_item = snap.global_ranked[0]
        updated = snap.with_fast_updates(out_of_stock=frozenset({top_item}))
        self.assertNotIn(top_item, updated.active_ids)

        store = StateStore(updated)
        cfg = Config()
        loop = FastServingLoop(client=None, store=store, recommender=Stage1Recommender(cfg), config=cfg)

        batch = syn["df_test"].head(50).to_dict(orient="records")
        t0 = time.perf_counter()
        answers = loop.answer_batch(batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(len(answers), 50)
        for ans in answers:
            self.assertEqual(len(ans["items"]), 10)
            self.assertEqual(len(set(ans["items"])), 10)
            self.assertNotIn(top_item, ans["items"])
        self.assertLess(elapsed_ms, 100.0)

        # Duplicate batch should be skipped by seen window
        self.assertEqual(loop.answer_batch(batch), [])

    def test_config_from_env_loads_credentials(self) -> None:
        cfg = Config.from_env()
        self.assertTrue(cfg.api_url.startswith("https://"))
        self.assertTrue(bool(cfg.team_id))
        self.assertTrue(bool(cfg.api_key))


if __name__ == "__main__":
    unittest.main()
