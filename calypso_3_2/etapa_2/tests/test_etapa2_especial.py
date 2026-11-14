"""Tests de la Etapa 2 · versión especial:  python -m pytest etapa_2_version_especial/tests -q

Sin la aserción "Etapa 2 > Etapa 1" sobre datos sintéticos (3.3 de la revisión): el mundo
sintético solo se usa para integración y latencia.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ETAPA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ETAPA_DIR))

from src.analyze import decide, lists_frame, switch_effects  # noqa: E402
from src.catalog import CatalogManager  # noqa: E402
from src.config import SIGNALS, SWITCHES, Config, load_dotenv_files  # noqa: E402
from src.guardrails import ensure_valid_submission, validate_submission_record  # noqa: E402
from src.learner import learn, to_table  # noqa: E402
from src.live import LiveTracker  # noqa: E402
from src.policy import Stage2Special, Weights, parse_tag  # noqa: E402
from src.signals import SIDX, compute_signals, parse_request  # noqa: E402
from src.sim import build_world  # noqa: E402
from src.state import HistoryBundle, Snapshot  # noqa: E402

try:
    import chdb  # noqa: F401

    HAS_CHDB = True
except ImportError:
    HAS_CHDB = False

W = H = DB = None


def setUpModule() -> None:
    global W, H, DB
    W = build_world(seed=7, n_products=300, n_sessions_events=5000, n_sessions_log=3000)
    if HAS_CHDB:
        from src.history import build_history
        from src.localdb import LocalRawTree

        DB = LocalRawTree(team_id="team_t")
        DB.insert("shop", "catalog", W["catalog"])
        DB.insert("shop", "history_events", W["events"])
        DB.insert("shop", "history_sessions", W["sessions"])
        H = build_history(DB, Config())
    else:
        H = HistoryBundle(catalog_rows=W["catalog"],
                          pop={p["product_id"]: float(v) for p, v in zip(W["catalog"], W["pop"])},
                          item_stats={}, ctx_prior={}, transitions={}, ease=None, users={})


def snap_of(cm: CatalogManager, **kw) -> Snapshot:
    cat = cm.build()
    return Snapshot(history=cm.h, catalog=cat, live={"default": LiveTracker(cm.cfg).view(cat)}, **kw)


class TestBasics(unittest.TestCase):
    def test_guardrails(self) -> None:
        active = {f"p{i}" for i in range(15)}
        items = ensure_valid_submission(["p0", "p0", "oos"], active, sorted(active), {"p1"})
        self.assertTrue(validate_submission_record({"request_id": "r", "items": items}, active))
        self.assertNotIn("p1", items)

    def test_group_api_env_crlf(self) -> None:
        d = tempfile.mkdtemp()
        Path(d, "group_API.env").write_bytes(b"TEAM_ID=team_42\r\n")
        old = os.getcwd()
        os.environ.pop("TEAM_ID", None)
        try:
            os.chdir(d)
            load_dotenv_files()
            self.assertEqual(os.environ.get("TEAM_ID"), "team_42")
        finally:
            os.chdir(old)
            os.environ.pop("TEAM_ID", None)


class TestColdStart(unittest.TestCase):
    def test_reverse_graft_absolute_and_same_subcat(self) -> None:
        """N7: un producto nuevo sin relación no queda enlazado a todos con similitud 1,0."""
        cm = CatalogManager(H, Config())
        base = W["catalog"][0]
        cm.apply_updates([
            {**base, "product_id": "p_twin", "ts": "2026-11-14T10:00:00Z", "change": "new_product"},
            {"product_id": "p_alien", "category": "otra", "subcategory": "nada", "brand": "x", "price_eur": 1e4,
             "description": "nada que ver", "ts": "2026-11-14T10:00:01Z", "change": "new_product", "is_active": True},
        ])
        cat = cm.build()
        targets = {q for lst in cat.reverse.values() for q, _ in lst}
        self.assertNotIn("p_alien", targets)
        self.assertIn("p_twin", targets)
        subs = {r["product_id"]: r["subcategory"] for r in W["catalog"]}
        for warm, lst in cat.reverse.items():
            for q, sim in lst:
                if q == "p_twin":
                    self.assertEqual(subs[warm], base["subcategory"])
                    self.assertGreaterEqual(sim, Config().reverse_min_sim)
                    self.assertLessEqual(sim, 1.0)


class TestSignalsAndPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.cm = CatalogManager(H, self.cfg)
        self.snap = snap_of(self.cm)

    def test_subset_equals_full(self) -> None:
        for r in W["sessions"][:80]:
            q = parse_request(r)
            idx = np.array([self.snap.catalog.idx[p] for p in r["shown"]])
            Ss, _ = compute_signals(q, self.snap, self.cfg, item_idx=idx)
            Sf, _ = compute_signals(q, self.snap, self.cfg)
            np.testing.assert_allclose(Ss, Sf[idx], atol=1e-6)

    def test_intent_decays_with_age(self) -> None:
        if not H.intent:
            self.skipTest("sin intenciones")
        uid, items = next(iter(H.intent.items()))
        q = parse_request({"request_id": "r", "user_id": uid, "context": {"page": "home"}, "recent_views": [],
                           "ts": "2026-11-14T10:00:00Z"})
        q_late = parse_request({"request_id": "r", "user_id": uid, "context": {"page": "home"},
                                "recent_views": [], "ts": "2026-12-14T10:00:00Z"})
        a, _ = compute_signals(q, self.snap, self.cfg)
        b, _ = compute_signals(q_late, self.snap, self.cfg)
        self.assertGreater(a[:, SIDX["intencion"]].sum(), b[:, SIDX["intencion"]].sum())

    def test_policy_valid_switches_deterministic(self) -> None:
        rec = Stage2Special(self.cfg)
        for r in W["sessions"][:150]:
            d1 = rec.recommend(r, self.snap)
            d2 = rec.recommend(r, self.snap)
            self.assertEqual(d1.items, d2.items)                 # reproducible (reintentos)
            self.assertTrue(validate_submission_record({"request_id": "x", "items": d1.items},
                                                       self.snap.catalog.active_ids))
            tag = parse_tag(d1.tag("e2esp", "manual"))
            for s in SWITCHES:
                self.assertEqual(tag[s], d1.switches[s])
            self.assertEqual(len(d1.signals), 10)
            self.assertEqual(len(d1.signals[0]), len(SIGNALS))

    def test_switch_shares_close_to_half(self) -> None:
        rec = Stage2Special(self.cfg)
        on = np.array([[rec.assign(f"r{i}")[0][s] for s in SWITCHES] for i in range(4000)])
        np.testing.assert_allclose(on.mean(axis=0), 0.5, atol=0.04)
        sh = np.mean([rec.assign(f"r{i}")[1] for i in range(4000)])
        self.assertAlmostEqual(sh, self.cfg.shuffle_rate, delta=0.02)

    def test_frozen_mode(self) -> None:
        cfg = Config(experiments=False)
        rec = Stage2Special(cfg, frozen_switches={"mmr": 0})
        sw, sh, fin = rec.assign("r1")
        self.assertEqual(sw["mmr"], 0)
        self.assertEqual(sw["rev"], 0)      # N9: apagado por defecto
        self.assertFalse(sh or fin)

    def test_revenue_tiebreak_only_within_eps(self) -> None:
        score = np.array([1.0, 0.999, 0.5])
        price = np.array([10.0, 50.0, 100.0])
        out = Stage2Special._revenue_tiebreak([0, 1, 2], score, price, 0.01)
        self.assertEqual(out, [1, 0, 2])    # 0 y 1 empatan casi: sube el caro; 2 no se mueve

    def test_excludes_bought(self) -> None:
        uid = next((u for u, p in H.users.items() if p.bought), None)
        if uid is None:
            self.skipTest("sin compras")
        rec = Stage2Special(self.cfg)
        for r in W["sessions"][:50]:
            d = rec.recommend({**r, "user_id": uid}, self.snap)
            self.assertFalse(set(d.items) & H.users[uid].bought)


class TestLearningAndAnalysis(unittest.TestCase):
    def _fake_joined(self, n: int = 6000, seed: int = 0) -> list[dict]:
        """Clics generados con un peso verdadero conocido sobre `lift` y `pop`."""
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            S = rng.random((10, len(SIGNALS))) * 0.5
            z = -3 + 3.0 * S[:, SIDX["lift"]] + 1.5 * S[:, SIDX["pop"]] - 0.15 * np.arange(10)
            p = 1 / (1 + np.exp(-z))
            items = [f"p{i}_{k}" for k in range(10)]
            clicked = next((items[k] for k in range(10) if rng.random() < p[k]), "")
            sw = {s: int(rng.random() < 0.5) for s in SWITCHES}
            out.append({"request_id": f"r{i}", "segment": "conocido_sesion", "switches": sw, "shuf": 0,
                        "fin": 0, "new_user": 0, "items": items, "signals": S.tolist(), "clicked": clicked,
                        "purchased": "", "revenue": 0.0, "new_pos": []})
        return out

    def test_learner_recovers_signal_weights(self) -> None:
        cfg = Config(learn_min_rows_segment=10_000)
        w, rep = learn(to_table(self._fake_joined()), Weights.default(), cfg)
        v = w.by_segment["conocido_sesion"]
        self.assertGreater(v["lift"], v["pop"])
        self.assertGreater(v["pop"], 0.5)
        self.assertTrue(any(r["decisión"] == "ACEPTADO" for r in rep))

    def test_decision_rule(self) -> None:
        df = lists_frame(self._fake_joined(3000))
        eff = switch_effects(df)
        sw = decide(eff)
        self.assertEqual(set(sw), set(SWITCHES))


@unittest.skipUnless(HAS_CHDB, "requiere chdb")
class TestSQLScratchService(unittest.TestCase):
    def test_history_components(self) -> None:
        self.assertTrue(all(H.ok_components.values()), H.ok_components)
        self.assertTrue(H.visto_stats)
        self.assertTrue(H.intent)

    def test_scratch_roundtrip(self) -> None:
        from src.history import load_aggregates, publish_aggregates

        publish_aggregates(DB, H)
        h2 = load_aggregates(DB)
        self.assertEqual(set(h2.pop), set(H.pop))
        self.assertEqual(set(h2.transitions), set(H.transitions))
        self.assertIsNotNone(h2.ease)

    def test_service_end_to_end(self) -> None:
        from src.simulate import run_simulation

        res = run_simulation(seconds=20, rate=4, seed=4, n_new_products=5, fail_push_every=5, world_data=W,
                             history=H, db=DB)
        self.assertGreater(res["n"], 40)
        self.assertGreaterEqual(res["reliability"], 0.9)
        self.assertLess(res["lat_max"], 3000)
        logs = list(Path(res["cfg"].log_dir).glob("signals_*.jsonl"))
        self.assertTrue(logs)                              # 4.2: registro de señales escrito


if __name__ == "__main__":
    unittest.main()
