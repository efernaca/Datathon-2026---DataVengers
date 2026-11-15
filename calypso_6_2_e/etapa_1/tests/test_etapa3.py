"""Tests de la Etapa 1 · versión especial:  python -m pytest etapa_1_version_especial/tests -q   (o  python -m unittest discover -s etapa_1_version_especial/tests)

Los que usan RawTree embebido (chdb) se saltan si `chdb` no está instalado.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ETAPA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ETAPA_DIR))

from src.bandit import ThompsonBandit  # noqa: E402
from src.catalog import CatalogManager, parse_ts  # noqa: E402
from src.config import Config, load_dotenv_files  # noqa: E402
from src.ease import fit_ease  # noqa: E402
from src.features import FEATURES, compute_features, parse_request  # noqa: E402
from src.guardrails import ensure_valid_submission, validate_submission_record  # noqa: E402
from src.live import DecayedCounter, LiveTracker  # noqa: E402
from src.model import LinearModel, load_model  # noqa: E402
from src.policy import Recommender  # noqa: E402
from src.sim import build_world, iso  # noqa: E402
from src.state import HistoryBundle, Snapshot, UserProfile  # noqa: E402

try:
    import chdb  # noqa: F401

    HAS_CHDB = True
except ImportError:
    HAS_CHDB = False

W = None
H = None


def setUpModule() -> None:
    global W, H
    W = build_world(seed=7, n_products=300, n_sessions_events=5000, n_sessions_log=3000)
    if HAS_CHDB:
        from src.history import build_history
        from src.localdb import LocalRawTree

        db = LocalRawTree()
        db.insert("shop", "catalog", W["catalog"])
        db.insert("shop", "history_events", W["events"])
        db.insert("shop", "history_sessions", W["sessions"])
        H = build_history(db, Config())
    else:   # histórico mínimo sin SQL
        H = HistoryBundle(catalog_rows=W["catalog"],
                          pop={p["product_id"]: float(v) for p, v in zip(W["catalog"], W["pop"])},
                          item_stats={}, ctx_prior={}, transitions={}, ease=None, users={})


def snap_of(cm: CatalogManager, **kw) -> Snapshot:
    cat = cm.build()
    lt = LiveTracker(cm.cfg)
    return Snapshot(history=cm.h, catalog=cat, live={a[0]: lt.view(cat, a[1]) for a in cm.cfg.bandit_arms}, **kw)


class TestGuardrails(unittest.TestCase):
    def test_ten_distinct_active(self) -> None:
        active = {f"p{i}" for i in range(15)}
        items = ensure_valid_submission(["p0", "p0", "oos", "p1"], active, sorted(active), {"p1", "p3"})
        self.assertEqual(len(set(items)), 10)
        self.assertNotIn("oos", items)
        self.assertNotIn("p1", items)
        self.assertTrue(validate_submission_record({"request_id": "r", "items": items}, active))

    def test_relax_excludes_when_tight(self) -> None:
        active = {f"p{i}" for i in range(10)}
        items = ensure_valid_submission([], active, sorted(active), {f"p{i}" for i in range(5)})
        self.assertEqual(len(set(items)), 10)


class TestConfig(unittest.TestCase):
    def test_loads_group_api_env_with_crlf(self) -> None:
        d = tempfile.mkdtemp()
        Path(d, "group_API.env").write_bytes(b"TEAM_ID=team_99\r\nRAWTREE_API_KEY=abc\r\n")
        old = os.getcwd()
        for k in ("TEAM_ID", "RAWTREE_API_KEY"):
            os.environ.pop(k, None)
        try:
            os.chdir(d)
            load_dotenv_files()
            self.assertEqual(os.environ.get("TEAM_ID"), "team_99")
            self.assertEqual(os.environ.get("RAWTREE_API_KEY"), "abc")
        finally:
            os.chdir(old)
            for k in ("TEAM_ID", "RAWTREE_API_KEY"):
                os.environ.pop(k, None)


class TestEase(unittest.TestCase):
    def test_closed_form(self) -> None:
        counts = {"a": 10, "b": 10, "c": 10}
        pairs = [{"i": "a", "j": "b", "c": 9}, {"i": "a", "j": "c", "c": 1}]
        m = fit_ease(counts, pairs, lam=1.0)
        B = m.B
        ia, ib, ic = m.index["a"], m.index["b"], m.index["c"]
        self.assertTrue(np.allclose(np.diag(B), 0))
        self.assertGreater(B[ia, ib], B[ia, ic])


class TestCatalogAndColdStart(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.cm = CatalogManager(H, self.cfg)

    def test_new_product_gets_neighbors_immediately(self) -> None:
        base = W["catalog"][0]
        upd = {**base, "product_id": "p_new", "ts": "2026-11-14T10:00:00Z", "change": "new_product",
               "description": base["description"]}
        self.cm.apply_updates([upd])
        cat = self.cm.build()
        self.assertIn("p_new", cat.active_ids)
        self.assertTrue(cat.is_new[cat.idx["p_new"]])
        nbs = cat.cold_neighbors["p_new"]
        self.assertTrue(nbs)
        # similitud absoluta: < 1 salvo copia exacta, y misma categoría
        cats = {r["product_id"]: r["category"] for r in W["catalog"]}
        self.assertTrue(all(cats[q] == base["category"] for q, _ in nbs))
        self.assertTrue(all(0 < s <= 1.0 for _, s in nbs))
        if cat.ease_proj is not None:
            self.assertGreater(cat.ease_proj[cat.idx["p_new"]].nnz, 0)

    def test_unrelated_product_not_forced_to_similarity_one(self) -> None:
        upd = {"product_id": "p_alien", "category": "otra", "subcategory": "nada", "brand": "zz",
               "price_eur": 9999.0, "description": "texto sin relación", "ts": "2026-11-14T10:00:00Z",
               "change": "new_product", "is_active": True}
        self.cm.apply_updates([upd])
        cat = self.cm.build()
        self.assertEqual(cat.cold_neighbors["p_alien"], ())

    def test_stock_and_price(self) -> None:
        p = W["catalog"][3]
        self.cm.apply_updates([{**p, "ts": "2026-11-14T10:00:00Z", "change": "out_of_stock"}])
        self.assertNotIn(p["product_id"], self.cm.build().active_ids)
        self.cm.apply_updates([{**p, "ts": "2026-11-14T10:00:05Z", "change": "back_in_stock"},
                               {**p, "ts": "2026-11-14T10:00:06Z", "change": "price_change",
                                "price_eur": p["price_eur"] * 0.8}])
        cat = self.cm.build()
        self.assertIn(p["product_id"], cat.active_ids)
        self.assertAlmostEqual(cat.price_change_rel[cat.idx[p["product_id"]]], 0.2, places=3)


class TestLive(unittest.TestCase):
    def test_decay_halflife(self) -> None:
        c = DecayedCounter(60.0)
        c.add("a", 1.0, 1000.0)
        self.assertAlmostEqual(c.array(["a"], 1060.0)[0], 0.5, places=6)

    def test_trend_and_smoothed_pop(self) -> None:
        cfg = Config()
        cm = CatalogManager(H, cfg)
        cat = cm.build()
        lt = LiveTracker(cfg)
        hot = cat.ids[-1]
        t0 = parse_ts("2026-11-14T10:00:00Z")
        evs = [{"ts": iso(t0 + i), "event_type": "view", "product_id": cat.ids[i % 50], "user_id": "u"}
               for i in range(2000)]
        evs += [{"ts": iso(t0 + 2000 + i), "event_type": "purchase", "product_id": hot, "user_id": "u9"}
                for i in range(30)]
        buys = lt.ingest(evs)
        self.assertIn(("u9", hot), buys)
        v = lt.view(cat)
        j = cat.idx[hot]
        self.assertGreater(v.trend[j], 0.5)
        self.assertGreater(v.log_pop[j], cat.log_pop_prior[j])


class TestFeaturesAndPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.cm = CatalogManager(H, self.cfg)
        self.snap = snap_of(self.cm)

    def test_subset_equals_full(self) -> None:
        for r in W["sessions"][:100]:
            q = parse_request(r)
            idx = np.array([self.snap.catalog.idx[p] for p in r["shown"]])
            Xs, _ = compute_features(q, self.snap, item_idx=idx)
            Xf, _ = compute_features(q, self.snap)
            np.testing.assert_allclose(Xs, Xf[idx], atol=1e-5)

    def test_policy_valid_and_excludes(self) -> None:
        uid = next(iter(H.users)) if H.users else "u_00001"
        snap = snap_of(self.cm, live_bought={uid: frozenset({self.snap.catalog.global_ranked[0]})})
        rec = Recommender(self.cfg, seed=0)
        for r in W["sessions"][:200]:
            r = {**r, "user_id": uid}
            d = rec.recommend(r, snap)
            self.assertTrue(validate_submission_record({"request_id": "x", "items": d.items}, snap.catalog.active_ids))
            self.assertNotIn(snap.catalog.global_ranked[0], d.items)
            prof = H.users.get(uid)
            if prof:
                self.assertFalse(set(d.items) & prof.bought)

    def test_diversify_without_session(self) -> None:
        r = {"request_id": "x", "user_id": "nuevo", "is_new_user": True,
             "context": {"page": "home", "region": "Madrid", "device": "web"}, "recent_views": []}
        ncat = {}
        for mode in ("never", "always"):
            cfg = Config(diversify=mode, bandit_enabled=False)
            d = Recommender(cfg, seed=0).recommend(r, self.snap, arm="hist")
            cats = {self.snap.catalog.category[self.snap.catalog.idx[p]] for p in d.items}
            ncat[mode] = len(cats)
        self.assertGreaterEqual(ncat["always"], ncat["never"])

    def test_exploration_inserts_new_product(self) -> None:
        base = W["catalog"][5]
        self.cm.apply_updates([{**base, "product_id": "p_explore", "ts": "2026-11-14T10:00:00Z",
                                "change": "new_product"}])
        snap = snap_of(self.cm)
        cfg = Config(bandit_arms=(("x", 1.0, 0.0, 1.0),), bandit_enabled=False)
        rec = Recommender(cfg, seed=1)
        r = {"request_id": "x", "user_id": "u", "is_new_user": True,
             "context": {"page": "product"}, "recent_views": [{"product_id": base["product_id"]}]}
        d = rec.recommend(r, snap, arm="x")
        self.assertIn("p_explore", d.items)
        for pid in d.explored:      # lo explorado solo ocupa los slots 8 y 10
            self.assertIn(d.items.index(pid) + 1, (8, 10))

    def test_model_roundtrip(self) -> None:
        m = LinearModel({"log_pop": 0.5}, -4.0, mean={"log_pop": 1.0}, scale={"log_pop": 2.0},
                        clip_lo={"log_pop": -1.0}, clip_hi={"log_pop": 2.0})
        p = Path(tempfile.mkdtemp()) / "model.json"
        m.save(p)
        m2 = load_model(p)
        X = np.zeros((3, len(FEATURES)), dtype=np.float32)
        X[:, FEATURES.index("log_pop")] = [-5, 0, 5]
        np.testing.assert_allclose(m.predict(X), m2.predict(X))
        self.assertAlmostEqual(m2.predict(X)[0], m2.predict(X[[0]] * 0 - 1)[0])  # recortado a -1


class TestBandit(unittest.TestCase):
    def test_learns_best_arm(self) -> None:
        b = ThompsonBandit(["a", "b"], seed=0)
        rng = np.random.default_rng(0)
        for _ in range(3000):
            arm = b.choose("s")
            b.update("s", arm, float(rng.random() < (0.2 if arm == "b" else 0.1)))
        s = b.summary()["s"]
        self.assertGreater(s["b"]["n"], s["a"]["n"])


@unittest.skipUnless(HAS_CHDB, "requiere chdb")
class TestSQLAndService(unittest.TestCase):
    def test_history_components_ok(self) -> None:
        self.assertTrue(all(H.ok_components.values()), H.ok_components)
        self.assertIsNotNone(H.ease)
        self.assertTrue(H.transitions)
        self.assertTrue(H.ctx_prior)
        self.assertTrue(any(u.subcat for u in H.users.values()))

    def test_service_end_to_end_with_flaky_network(self) -> None:
        from src.simulate import run_simulation

        res = run_simulation(seconds=20, rate=3, seed=4, n_new_products=5, fail_push_every=4, world_data=W,
                             history=H)
        self.assertGreater(res["n"], 30)
        self.assertGreaterEqual(res["reliability"], 0.9, res)
        self.assertLess(res["lat_max"], 3000)
        self.assertGreater(res["service_stats"]["push_fail"], 0)


if __name__ == "__main__":
    unittest.main()
