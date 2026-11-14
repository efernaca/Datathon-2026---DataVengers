"""Ensayo general en local: el servicio completo contra un RawTree embebido y un "juez".

    python -m src.simulate --seconds 60 --rate 3

* Productor: inserta peticiones en `shop.requests` (~3/s), eventos orgánicos en `shop.events`
  y cambios de catálogo (productos nuevos, agotados, cambios de precio).
* Juez: a los 3 s de cada petición mira si hay respuesta válida y a tiempo (reloj de
  inserción), calcula la compra esperada con el modelo oculto y escribe `feedback`.
* Mide fiabilidad, edad de respuesta y score 0–100 frente a baseline y oráculo.

Sirve para probar F1–F8, C1, V2 y D de punta a punta. Los números de conversión
salen de un mundo sintético: no predicen el leaderboard.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
import logging
import tempfile
import threading
import time
from typing import Any

import numpy as np

from .config import Config
from .history import build_history
from .localdb import LocalRawTree
from .model import LinearModel, load_model
from .serve import Service
from .sim import build_world, iso

logger = logging.getLogger("etapa1esp.simulate")


def run_simulation(seconds: float = 45.0, rate: float = 3.0, seed: int = 3, model_path: str | None = None,
                   n_new_products: int = 15, new_user_rate: float = 0.5, fail_push_every: int = 0,
                   world_data: dict | None = None, history=None, cfg: Config | None = None) -> dict[str, Any]:
    W = world_data or build_world(seed=seed, n_products=400, n_sessions_events=8000, n_sessions_log=6000)
    world = W["world"]
    db = LocalRawTree(team_id="team_sim")
    db.insert("shop", "catalog", W["catalog"])
    db.insert("shop", "history_events", W["events"])
    db.insert("shop", "history_sessions", W["sessions"])
    tmp = tempfile.mkdtemp(prefix="sim_cache_")
    cfg = cfg or Config(team_id="team_sim", cache_dir=tmp, heartbeat_file=f"{tmp}/hb",
                        query_timeout_slow_sec=120)
    h = history or build_history(db, cfg)
    model = load_model(model_path) if model_path else LinearModel.default()

    if fail_push_every:      # inyección de fallos de red para probar F1/F2
        orig = db.push_submissions
        counter = {"n": 0}

        def flaky(answers, timeout=1.2):
            counter["n"] += 1
            if counter["n"] % fail_push_every == 0:
                time.sleep(min(timeout, 0.8))
                raise TimeoutError("fallo inyectado")
            return orig(answers, timeout)
        db.push_submissions = flaky

    stop = threading.Event()
    pending: dict[str, tuple[dict, Any, list[int], float]] = {}
    plock = threading.Lock()
    base_pop = W["pop"]
    active = np.ones(len(world.products), dtype=bool)
    baseline_list = [world.products[i]["product_id"] for i in np.argsort(-base_pop)]
    new_ids: list[int] = []
    oos_at: dict[int, float] = {}
    results: list[dict[str, Any]] = []

    def producer() -> None:
        rng = np.random.default_rng(seed + 1)
        k = 0
        t_new = time.time() + 3
        while not stop.is_set():
            now = time.time()
            if len(new_ids) < n_new_products and now > t_new:
                rows = world.add_products(1, new=True)
                active.resize(len(world.products), refcheck=False)
                active[-1] = True
                new_ids.append(len(world.products) - 1)
                db.insert("shop", "catalog_updates", [{**rows[0], "ts": iso(now), "change": "new_product"}])
                t_new = now + seconds / max(n_new_products, 1) * 0.6
            if rng.random() < 0.05:     # agotar un producto popular
                j = int(rng.choice(np.argsort(-base_pop)[:30]))
                if active[j]:
                    active[j] = False
                    oos_at[j] = now
                    db.insert("shop", "catalog_updates",
                              [{**world.products[j], "ts": iso(now), "change": "out_of_stock"}])
            if rng.random() < 0.03:
                j = int(rng.integers(len(world.products)))
                p = dict(world.products[j])
                p["price_eur"] = round(p["price_eur"] * 0.75, 2)
                db.insert("shop", "catalog_updates", [{**p, "ts": iso(now), "change": "price_change"}])
            req, u, rec = world.make_request(now, new_user_rate=new_user_rate,
                                             new_products=np.array(new_ids) if new_ids else None,
                                             rid=f"r_{seed}_{k:06d}")
            k += 1
            with plock:
                pending[req["request_id"]] = (req, u, rec, now)
            db.insert("shop", "requests", [req])
            evs = []
            for _ in range(int(rng.poisson(15))):
                j = int(rng.choice(new_ids)) if new_ids and rng.random() < 0.2 else int(rng.integers(len(world.products)))
                evs.append({"ts": iso(now), "event_type": "view" if rng.random() < 0.9 else "purchase",
                            "user_id": f"u_{int(rng.integers(world.n_users)):05d}", "session_id": "live",
                            "product_id": world.products[j]["product_id"], "price_eur": 1.0,
                            "device": "web", "region": "Madrid"})
            db.insert("shop", "events", evs)
            stop.wait(max(0.01, rng.exponential(1.0 / rate)))

    def judge() -> None:
        while not stop.is_set() or pending:
            now = time.time()
            with plock:
                due = [(rid, v) for rid, v in pending.items() if now - v[3] > 3.6]
                for rid, _ in due:
                    pending.pop(rid)
            if due:
                ids = ",".join(f"'{rid}'" for rid, _ in due)
                subs = db.query(f"SELECT request_id::String AS rid, items::Array(String) AS items, "
                                f"toUnixTimestamp64Milli(_ins) AS ins FROM team_sim.submissions "
                                f"WHERE request_id::String IN ({ids})")
                by = {}
                for s in subs:
                    by.setdefault(s["rid"], s)
                fb = []
                for rid, (req, u, rec, t_req) in due:
                    s = by.get(rid)
                    lat = (int(s["ins"]) / 1000 - t_req) * 1000 if s else None
                    items = list(s["items"]) if s else []
                    idx = {p["product_id"]: i for i, p in enumerate(world.products)}
                    ins_t = int(s["ins"]) / 1000 if s else 0.0
                    # válido = activo en el momento en que llegó la respuesta
                    valid = (len(items) == 10 and len(set(items)) == 10
                             and all(p in idx and oos_at.get(idx[p], 1e18) > ins_t for p in items))
                    on_time = lat is not None and lat <= 3000
                    served = "team" if (valid and on_time) else "fallback"
                    reason = "" if served == "team" else ("missing" if s is None else ("late" if not on_time else "invalid"))
                    base = [p for p in baseline_list if active[idx[p]]][:10]
                    shown = items if served == "team" else base
                    page = req["context"]["page"]
                    c, pu, rev, ep = world.outcome(u, rec, page, shown)
                    a, b = world.relevance(u, rec, page)
                    ep_base = world.expected_purchase([idx[p] for p in base], a, b)
                    ep_orc = world.expected_purchase(world.oracle(a, b, active), a, b)
                    cold = req["is_new_user"] or any(r in new_ids for r in rec)
                    results.append({"rid": rid, "served": served, "reason": reason, "lat": lat, "ep": ep,
                                    "ep_base": ep_base, "ep_oracle": ep_orc, "cold": cold})
                    fb.append({"request_id": rid, "ts": iso(time.time()), "shown": shown, "served_by": served,
                               "fallback_reason": reason, "latency_ms": lat or 0, "clicked": c,
                               "purchased": pu, "revenue_eur": rev})
                db.insert("team_sim", "feedback", fb)
            time.sleep(0.5)

    svc = Service(db, cfg, model, h, seed=seed, cold_boot=False)
    svc.bootstrap()
    th_p = threading.Thread(target=producer, daemon=True)
    th_j = threading.Thread(target=judge, daemon=True)
    th_p.start()
    th_j.start()
    svc.run(duration=seconds)
    stop.set()
    th_p.join(5)
    th_j.join(15)

    import pandas as pd

    df = pd.DataFrame(results)
    out: dict[str, Any] = {"n": len(df)}
    if len(df):
        out["reliability"] = float((df["served"] == "team").mean())
        out["fallback_reasons"] = df.loc[df["served"] != "team", "reason"].value_counts().to_dict()
        lat = df["lat"].dropna()
        out["lat_p50"] = float(lat.median()) if len(lat) else None
        out["lat_p95"] = float(lat.quantile(0.95)) if len(lat) else None
        out["lat_max"] = float(lat.max()) if len(lat) else None
        for seg, d in (("ALL", df), ("COLD", df[df["cold"]])):
            if len(d):
                team, base, orc = d["ep"].mean(), d["ep_base"].mean(), d["ep_oracle"].mean()
                out[f"score_{seg}"] = float(np.clip(100 * (team - base) / max(orc - base, 1e-9), 0, 100))
                out[f"ep100_{seg}"] = (float(100 * base), float(100 * team), float(100 * orc))
    out["service_stats"] = dict(svc.stats)
    out["bandit"] = svc.bandit.summary()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--model")
    ap.add_argument("--fail-push-every", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")
    res = run_simulation(args.seconds, args.rate, model_path=args.model, fail_push_every=args.fail_push_every)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
