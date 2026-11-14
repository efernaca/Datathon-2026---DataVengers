"""Automated test suite for Stage 4 (DiskCache recovery, SLA retries, Slot-Gated calypso-2.1)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

ETAPA4_DIR = Path(__file__).resolve().parents[1]
CALYPSO_ROOT = ETAPA4_DIR.parent
for _p in (CALYPSO_ROOT, ETAPA4_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config import ConfigV4
from src.disk_cache import DiskCache
from src.eval import evaluate_stratified, generate_synthetic_dataset_v4
from src.guardrails import ensure_valid_submission, validate_submission_record
from src.monitor import summarize_feedback_rows
from src.recommender import CalypsoRecommenderV4
from src.serve import CalypsoFastLoopV4
from src.state import StateStore


class _FlakyClient:
    """Stub RawTree client whose first push fails, verifying post-push _seen semantics."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.fail_next_push = True
        self.pushed_batches: list[list[dict]] = []

    def query_safe(self, sql: str, timeout: float = 0.9) -> list[dict]:
        return list(self._rows)

    def push_submissions(self, answers: list[dict], timeout: float = 0.9) -> None:
        if self.fail_next_push:
            self.fail_next_push = False
            raise RuntimeError("Transient 503 push failure")
        self.pushed_batches.append(answers)


class TestStage4Hardening(unittest.TestCase):
    """Verify crash recovery disk cache, post-push retry, monitor, and Slot-Gated lift."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.syn = generate_synthetic_dataset_v4(
            seed=42,
            n_warm_products=300,
            n_new_products=30,
            n_warm_users=600,
            n_history_sessions=2400,
            n_feedback_sessions=800,
            n_test_sessions=1200,
        )

    def test_disk_cache_atomic_roundtrip_and_corruption_safety(self) -> None:
        snap = self.syn["snap_v4"]
        bundle = self.syn["static_bundle"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = DiskCache(Path(tmp_dir))
            cache.save_snapshot(snap)
            cache.save_static_history(bundle)

            reloaded_snap = cache.load_snapshot()
            reloaded_bundle = cache.load_static_history()
            self.assertIsNotNone(reloaded_snap)
            self.assertIsNotNone(reloaded_bundle)
            assert reloaded_snap is not None and reloaded_bundle is not None
            self.assertEqual(reloaded_snap.active_ids, snap.active_ids)
            self.assertEqual(len(reloaded_snap.product_static), len(snap.product_static))
            self.assertEqual(len(reloaded_bundle.catalog_rows), len(bundle.catalog_rows))

            # Verify corrupted cache returns None cleanly instead of raising
            cache.snapshot_path.write_bytes(b"corrupt_pickle_bytes")
            self.assertIsNone(cache.load_snapshot())

    def test_post_push_seen_commit_retries_after_failure(self) -> None:
        snap = self.syn["snap_v4"]
        store = StateStore(snap)
        req_rows = self.syn["df_test"].to_dict(orient="records")[:3]
        flaky_client = _FlakyClient(req_rows)
        loop = CalypsoFastLoopV4(
            client=flaky_client,  # type: ignore[arg-type]
            store=store,
            recommender=CalypsoRecommenderV4(ConfigV4()),
            config=ConfigV4(),
        )

        # Cycle 1: push fails -> exception raised, _seen stays empty so requests aren't lost
        with self.assertRaises(RuntimeError):
            loop.step()
        self.assertEqual(len(flaky_client.pushed_batches), 0)

        # Cycle 2: retry pushes all 3 answers cleanly and commits _seen
        sent = loop.step()
        self.assertEqual(sent, 3)
        self.assertEqual(len(flaky_client.pushed_batches), 1)
        for rec in flaky_client.pushed_batches[0]:
            self.assertEqual(rec["model_version"], "calypso-4.1")
            self.assertTrue(validate_submission_record(rec, snap.active_ids))

        # Cycle 3: same requests are now skipped because push was confirmed
        self.assertEqual(loop.step(), 0)

    def test_monitor_summary_metrics(self) -> None:
        rows = [
            {
                "model_version": "calypso-4.1",
                "served_by": "team",
                "latency_ms": 310.0,
                "clicked": "p_1",
                "purchased": "p_1",
                "revenue_eur": 50.0,
            },
            {
                "model_version": "calypso-4.1",
                "served_by": "baseline",
                "latency_ms": 3400.0,
                "clicked": "",
                "purchased": "",
                "revenue_eur": 0.0,
            },
        ]
        summary = summarize_feedback_rows(rows)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["model_version"], "calypso-4.1")
        self.assertEqual(row["on_time_pct"], 50.0)
        self.assertEqual(row["cvr_pct"], 50.0)
        self.assertEqual(row["revenue_eur"], 50.0)

    def test_slot_gated_calypso_protects_warm_and_beats_previous_stages(self) -> None:
        snap_v4 = self.syn["snap_v4"]
        snap_v2 = self.syn["snap_v2"]
        rec_v4 = CalypsoRecommenderV4(ConfigV4())
        from etapa_2.src.config import ConfigV2
        from etapa_2.src.recommender import Stage2Recommender

        rec_v2 = Stage2Recommender(ConfigV2())
        base10 = self.syn["baseline_top10"]
        policies = {
            "baseline": lambda _: ensure_valid_submission(base10, snap_v4.active_ids, snap_v4.fallback_ranked),
            "etapa2_seq_mmr": lambda r: rec_v2.recommend(r, snap_v2),
            "etapa4_calypso_v2.1": lambda r: rec_v4.recommend(r, snap_v4),
        }
        report = evaluate_stratified(
            policies,
            self.syn["df_test"],
            cold_product_ids=self.syn["cold_product_ids"],
            warm_user_ids=self.syn["warm_user_ids"],
        )

        warm_v4 = report.loc[("WARM", "etapa4_calypso_v2.1")]
        all_v4 = report.loc[("ALL", "etapa4_calypso_v2.1")]
        all_v2 = report.loc[("ALL", "etapa2_seq_mmr")]

        self.assertGreaterEqual(float(warm_v4["hit_rate_buy@10"]), 0.83)
        self.assertGreater(float(all_v4["hit_rate_buy@10"]), float(all_v2["hit_rate_buy@10"]))
        self.assertLess(float(all_v4["latency_ms_mean"]), 1.5)


if __name__ == "__main__":
    unittest.main()
