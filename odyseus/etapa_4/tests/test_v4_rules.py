"""Tests de las reglas de negocio de calypso 4.1 (carrito, olvidado, recurrentes, precio)."""
from __future__ import annotations

from dataclasses import replace
import sys
import time
from pathlib import Path
import unittest

ETAPA4_DIR = Path(__file__).resolve().parents[1]
CALYPSO_ROOT = ETAPA4_DIR.parent
for _p in (CALYPSO_ROOT, ETAPA4_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.business_rules import build_cart_and_recurrent_maps, guarantee_items  # noqa: E402
from src.config import ConfigV4  # noqa: E402
from src.eval import generate_synthetic_dataset_v4  # noqa: E402
from src.guardrails import validate_submission_record  # noqa: E402
from src.recommender import CalypsoRecommenderV4  # noqa: E402
from src.serve import _merge_live_cart_rows  # noqa: E402

OFF = dict(v4_cart_priority=False, v4_forgotten_cart=False, v4_price_ascending=False,
           v4_recurrent=False, v4_live_event_purchases=False)
NOW = time.time() * 1000.0
MIN = 60_000.0
DAY = 86_400_000.0


class TestCartIndex(unittest.TestCase):
    def test_cart_cleared_by_later_purchase_and_recurrence(self) -> None:
        carts, rec, live_bought = build_cart_and_recurrent_maps(
            cart_hist_rows=[
                {"user_id": "u1", "product_id": "a", "n_purchases": 0, "last_cart_ms": NOW - 2 * DAY, "last_purchase_ms": 0},
                {"user_id": "u1", "product_id": "b", "n_purchases": 0, "last_cart_ms": NOW - 3 * DAY, "last_purchase_ms": 0},
            ],
            recurrent_hist_rows=[
                {"user_id": "u1", "product_id": "r", "n_purchases": 3, "last_purchase_ms": NOW - 5 * DAY},
            ],
            live_event_rows=[  # compra en vivo de "b" -> sale del carrito
                {"user_id": "u1", "product_id": "b", "n_purchases": 1, "last_cart_ms": 0, "last_purchase_ms": NOW - MIN},
                {"user_id": "u2", "product_id": "x", "n_purchases": 1, "last_cart_ms": 0, "last_purchase_ms": NOW - MIN},
            ],
            feedback_purchase_rows=[  # misma compra vista en feedback: NO debe contar dos veces
                {"user_id": "u2", "product_id": "x", "n_purchases": 1, "last_purchase_ms": NOW - MIN},
            ],
        )
        self.assertEqual([p for p, _ in carts["u1"]], ["a"])
        self.assertEqual(dict(rec["u1"]), {"r": 3})
        self.assertNotIn("u2", rec)  # 1 compra (events) + la misma en feedback != recurrente
        self.assertEqual(live_bought["u2"], frozenset({"x"}))

    def test_merge_recent_overrides_stale(self) -> None:
        full = [{"user_id": "u", "product_id": "p", "n_purchases": 0, "last_cart_ms": 10, "last_purchase_ms": 0}]
        recent = [{"user_id": "u", "product_id": "p", "n_purchases": 1, "last_cart_ms": 10, "last_purchase_ms": 20}]
        merged = _merge_live_cart_rows(full, recent)
        self.assertEqual(merged[0]["last_purchase_ms"], 20)

    def test_guarantee_items_replaces_tail(self) -> None:
        self.assertEqual(guarantee_items(["a", "b", "c"], ["z"]), ["a", "b", "z"])
        self.assertEqual(guarantee_items(["a", "z", "c"], ["z"]), ["a", "z", "c"])


class TestRecommendRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = generate_synthetic_dataset_v4(
            seed=3, n_warm_products=120, n_new_products=10, n_warm_users=80,
            n_history_sessions=400, n_feedback_sessions=60, n_test_sessions=40,
        )
        cls.snap = cls.data["snap_v4"]
        cls.req = {
            "request_id": "t1", "user_id": "u_test", "is_new_user": False,
            "context": {"region": "Madrid", "device": "web", "page": "home"}, "recent_views": [],
        }
        active = sorted(cls.snap.active_ids - cls.snap.new_product_ids)
        cls.p_active, cls.p_forgot, cls.p_once, cls.p_rec = active[:4]

    def _snap(self, carts=None, rec=None, bought=frozenset()):
        return replace(
            self.snap,
            user_carts=carts or {},
            user_recurrent=rec or {},
            live_user_bought={"u_test": frozenset(bought)} if bought else {},
        )

    def test_off_equals_core(self) -> None:
        rec = CalypsoRecommenderV4(replace(ConfigV4(), **OFF))
        snap = self._snap(carts={"u_test": ((self.p_active, NOW - MIN),)})
        out = rec.recommend(self.req, snap, now_ms=NOW)
        core = rec._recommend_core(
            ("u_test", False, "Madrid", "web", "home", []), snap, 10, frozenset()
        )
        self.assertEqual(out, core)

    def test_cart_active_then_forgotten_first(self) -> None:
        cfg = replace(ConfigV4(), v4_price_ascending=False)
        snap = self._snap(carts={"u_test": ((self.p_active, NOW - 5 * MIN), (self.p_forgot, NOW - 3 * DAY))})
        out = CalypsoRecommenderV4(cfg).recommend(self.req, snap, now_ms=NOW)
        self.assertEqual(out[:2], [self.p_active, self.p_forgot])
        self.assertTrue(validate_submission_record({"request_id": "t1", "items": out}, snap.active_ids))

    def test_too_old_cart_not_pinned(self) -> None:
        cfg = replace(ConfigV4(), v4_price_ascending=False)
        snap = self._snap(carts={"u_test": ((self.p_forgot, NOW - 90 * DAY),)})
        out = CalypsoRecommenderV4(cfg).recommend(self.req, snap, now_ms=NOW)
        self.assertNotEqual(out[0], self.p_forgot)

    def test_bought_once_excluded_recurrent_included(self) -> None:
        snap = self._snap(
            rec={"u_test": ((self.p_rec, 3),)},
            bought={self.p_once, self.p_rec},
        )
        out = CalypsoRecommenderV4(ConfigV4()).recommend(self.req, snap, now_ms=NOW)
        self.assertNotIn(self.p_once, out)
        self.assertIn(self.p_rec, out)

    def test_price_ascending_within_blocks(self) -> None:
        snap = self._snap(carts={"u_test": ((self.p_active, NOW - 5 * MIN),)})
        out = CalypsoRecommenderV4(ConfigV4()).recommend(self.req, snap, now_ms=NOW)
        self.assertEqual(out[0], self.p_active)
        prices = [snap.products[p].price_eur for p in out[1:]]
        self.assertEqual(prices, sorted(prices))

    def test_cart_cap(self) -> None:
        ids = sorted(self.snap.active_ids - self.snap.new_product_ids)[10:16]
        cart = tuple((p, NOW - (i + 1) * MIN) for i, p in enumerate(ids))
        out = CalypsoRecommenderV4(replace(ConfigV4(), v4_price_ascending=False)).recommend(
            self.req, self._snap(carts={"u_test": cart}), now_ms=NOW
        )
        self.assertEqual(out[:3], list(ids[:3]))
        self.assertEqual(len(set(out)), 10)


if __name__ == "__main__":
    unittest.main()
