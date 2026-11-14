"""Tests de las medidas de calypso_4_1/MEDIDAS_CONVERSION.md implementadas en 4.2 (sobre la base de 4.1e)."""
from __future__ import annotations

import math
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path

ETAPA4_DIR = Path(__file__).resolve().parents[1]
CALYPSO_ROOT = ETAPA4_DIR.parent
for _p in (CALYPSO_ROOT, ETAPA4_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.bandits import slot_examination_weights  # noqa: E402
from src.config import ConfigV4  # noqa: E402
from src.eval import generate_synthetic_dataset_v4  # noqa: E402
from src.guardrails import validate_submission_record  # noqa: E402
from src.recommender import CalypsoRecommenderV4, _build_sales_ranking  # noqa: E402
from src.serve import SQL_LIVE_TRENDS, _ab_bucket, sql_live_trends  # noqa: E402
from src.weights_profile import WeightsProfile, get_preset  # noqa: E402

NOW = time.time() * 1000.0
WP = get_preset("medidas_conversion")


class TestDefaultsUnchanged(unittest.TestCase):
    def test_default_profile_is_legacy(self) -> None:
        base = WeightsProfile()
        self.assertEqual(base.scoring_mode, "legacy")
        self.assertEqual(base.explore_min_slot, 0)
        self.assertEqual(base.live_window_minutes, 0)
        self.assertEqual(base.candidate_top_sellers, 0)
        self.assertEqual(base.attention_curve, ())
        self.assertFalse(base.v4_cart_as_signal)
        self.assertEqual(base.ab_split_ratio, 0.0)

    def test_preset_roundtrip(self) -> None:
        back = WeightsProfile.from_dict(WP.to_dict())
        self.assertEqual(back, WP)
        self.assertEqual(len(back.attention_curve), 10)


class TestSalesRanking(unittest.TestCase):
    def test_live_purchases_take_over_with_mass(self) -> None:
        active = frozenset({"a", "b", "c", "n"})
        pop = [{"product_id": "a", "purchases": 100}, {"product_id": "b", "purchases": 50},
               {"product_id": "c", "purchases": 10}]
        rank, ordered, _ = _build_sales_ranking(active, frozenset(), pop, [], {}, {}, WP)
        self.assertEqual(ordered[:3], ("a", "b", "c"))
        self.assertEqual(rank["n"], WP.cal_max_rank)  # sin compras: puesto máximo
        # Pocas compras vivas: manda el histórico
        live_few = [{"product_id": "c", "live_purchases": 3, "live_carts": 0}]
        _, ordered, _ = _build_sales_ranking(active, frozenset(), pop, live_few, {}, {}, WP)
        self.assertEqual(ordered[0], "a")
        # Mucha señal viva: manda lo vivo
        live_many = [{"product_id": "c", "live_purchases": 400, "live_carts": 0}]
        rank, ordered, live = _build_sales_ranking(active, frozenset(), pop, live_many, {}, {}, WP)
        self.assertEqual(ordered[0], "c")
        self.assertEqual(live["c"], 400.0)

    def test_new_product_prior_from_neighbors(self) -> None:
        active = frozenset({"a", "b", "c", "new"})
        pop = [{"product_id": "a", "purchases": 100}, {"product_id": "b", "purchases": 50},
               {"product_id": "c", "purchases": 1}]
        nbs = {"new": (("a", 0.9), ("b", 0.8), ("c", 0.1))}
        rank, ordered, _ = _build_sales_ranking(active, frozenset({"new"}), pop, [], nbs, {}, WP)
        self.assertLess(rank["new"], WP.cal_max_rank)
        self.assertLess(rank["new"], rank["c"])  # mediana = share de "b"

    def test_old_live_query_ignored(self) -> None:
        active = frozenset({"a"})
        _, _, live = _build_sales_ranking(active, frozenset(), [], [{"product_id": "a", "live_score": 9}], {}, {}, WP)
        self.assertEqual(live, {})


class TestCalibratedRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = ConfigV4.from_profile(WP)
        cls.cfg = cfg
        cls.data = generate_synthetic_dataset_v4(
            seed=5, n_warm_products=150, n_new_products=12, n_warm_users=80,
            n_history_sessions=500, n_feedback_sessions=80, n_test_sessions=60,
            config_v4=cfg, weights_profile=WP,
        )
        cls.snap = cls.data["snap_v4"]
        cls.snap.config_override = cfg
        cls.rec = CalypsoRecommenderV4(cfg)

    def _req(self, views, page="product", new_user=False, uid="u_test"):
        return {"request_id": "r", "user_id": uid, "is_new_user": new_user,
                "context": {"region": "Madrid", "device": "web", "page": page}, "recent_views": views}

    def test_valid_and_ordered_by_rank_on_home(self) -> None:
        out = self.rec.recommend(self._req([], page="home"), self.snap, now_ms=NOW)
        self.assertTrue(validate_submission_record({"request_id": "r", "items": out}, self.snap.active_ids))
        ranks = [self.snap.sales_rank[p] for p in out[:7]]
        # sin señales de sesión manda el ranking de ventas (el contexto solo reordena vecinos)
        self.assertTrue(all(r < 12 for r in ranks), ranks)

    def test_seen_top_seller_beats_everything(self) -> None:
        top = [p for p in self.snap.top_sellers if p not in self.snap.new_product_ids]
        seen = top[3]
        out = self.rec.recommend(self._req([seen]), self.snap, now_ms=NOW)
        self.assertEqual(out[0], seen)

    def test_seen_tail_product_does_not_displace_best_sellers(self) -> None:
        tail = [p for p in self.snap.top_sellers if self.snap.sales_rank[p] >= WP.cal_max_rank]
        if not tail:
            self.skipTest("sin productos de cola en el sintético")
        out = self.rec.recommend(self._req([tail[0]]), self.snap, now_ms=NOW)
        self.assertNotEqual(out[0], tail[0])

    def test_no_unconfirmed_new_products_before_slot_8_even_cold(self) -> None:
        for req in (self._req([], page="home", new_user=True), self._req([], page="category", new_user=True)):
            out = self.rec.recommend(req, self.snap, now_ms=NOW)
            for pid in out[: WP.explore_min_slot]:
                is_new_unconfirmed = pid in self.snap.new_product_ids and self.snap.live_buys.get(pid, 0) == 0
                self.assertFalse(is_new_unconfirmed, f"{pid} nuevo sin confirmar antes del slot 8")

    def test_cart_item_competes_but_is_present(self) -> None:
        pid = self.snap.top_sellers[60]
        snap = replace(self.snap, user_carts={"u_test": ((pid, NOW - 60_000.0),)})
        out = self.rec.recommend(self._req([]), snap, now_ms=NOW)
        self.assertIn(pid, out)
        self.assertLess(out.index(pid), 3)

    def test_ab_branch_uses_alt_profile(self) -> None:
        legacy = get_preset("conversion_first")
        out = self.rec.recommend(self._req([], page="home"), self.snap, now_ms=NOW, weights_profile=legacy)
        self.assertEqual(len(out), 10)


class TestServeHelpers(unittest.TestCase):
    def test_live_sql(self) -> None:
        self.assertEqual(sql_live_trends(0), SQL_LIVE_TRENDS)
        q = sql_live_trends(60)
        self.assertIn("INTERVAL 60 MINUTE", q)
        self.assertIn("live_purchases", q)

    def test_ab_bucket_stable_and_balanced(self) -> None:
        self.assertEqual(_ab_bucket("abc"), _ab_bucket("abc"))
        vals = [_ab_bucket(f"req-{i}") for i in range(4000)]
        frac = sum(v < 0.5 for v in vals) / len(vals)
        self.assertTrue(0.45 < frac < 0.55)

    def test_attention_curve(self) -> None:
        w = slot_examination_weights(10, 0.75, WP.attention_curve)
        self.assertEqual(w[0], 1.0)
        self.assertAlmostEqual(w[9], 0.5)
        self.assertTrue(math.isclose(slot_examination_weights(3, 0.75)[1], 2 ** -0.75))


if __name__ == "__main__":
    unittest.main()
