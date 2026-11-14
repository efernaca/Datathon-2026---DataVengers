"""Mundo sintético con un modelo de respuesta oculto (solo para pruebas de integración).

**No es evidencia de calidad** (sección 3.4 de la revisión): el simulador de los
organizadores es otro. Sirve para comprobar que el pipeline completo funciona
(SQL → histórico → entrenamiento → servicio → feedback) y que el entrenamiento
recupera estructura que nadie le ha dado a mano. Para que no sea circular, el modelo
oculto usa variables latentes que el recomendador NO ve (calidad del producto,
gusto latente del usuario, subcategorías complementarias).

La métrica es la misma idea que el leaderboard: compras *esperadas* por petición según
el modelo oculto ("No luck"), más el score 0–100 entre baseline y oráculo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

REGIONS = ("Catalunya", "Madrid", "Andalucia", "Valencia")
DEVICES = ("web", "ios", "android")
PAGES = ("home", "product", "category")
ATTN = np.array([1.0, 0.8, 0.66, 0.56, 0.48, 0.42, 0.37, 0.33, 0.30, 0.27])


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class World:
    seed: int = 0
    n_products: int = 600
    n_users: int = 2500
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        rng = self.rng = np.random.default_rng(self.seed)
        self.cats = [f"cat{c}" for c in range(10)]
        self.subcats = [f"{c}_s{s}" for c in self.cats for s in range(3)]
        self.sub_cat = np.repeat(np.arange(10), 3)
        self.complement = {s: int(rng.integers(len(self.subcats))) for s in range(len(self.subcats))}
        self.words = {s: [f"w{s}_{k}" for k in range(6)] for s in range(len(self.subcats))}
        self.products: list[dict[str, Any]] = []
        self.p_sub = np.zeros(0, dtype=int)
        self.p_quality = np.zeros(0)
        self.p_price = np.zeros(0)
        self.p_brand = np.zeros(0, dtype=int)
        self.p_new = np.zeros(0, dtype=bool)
        self.t0 = datetime(2026, 9, 10, tzinfo=timezone.utc).timestamp()
        self.add_products(self.n_products, new=False)
        # usuarios: gusto latente concentrado en 1–2 subcategorías y precio preferido
        self.u_taste = rng.dirichlet(np.full(len(self.subcats), 0.08), size=self.n_users)
        self.u_price = rng.normal(3.6, 0.6, size=self.n_users)
        self.u_bought: dict[int, set[int]] = {}
        self.trend_boost = np.zeros(len(self.subcats))

    # ------------------------------------------------------------------ catálogo

    def add_products(self, n: int, new: bool) -> list[dict[str, Any]]:
        rng = self.rng
        start = len(self.products)
        sub = rng.integers(len(self.subcats), size=n)
        qual = rng.normal(0, 1, size=n) + (0.4 if new else 0.0)
        price = np.exp(rng.normal(3.2 + 0.08 * (sub % 7), 0.5))
        brand = rng.integers(5, size=n)
        rows = []
        for k in range(n):
            s = int(sub[k])
            w = rng.choice(self.words[s], size=3, replace=False)
            rows.append({
                "product_id": f"p_{start + k:05d}", "category": self.cats[self.sub_cat[s]],
                "subcategory": self.subcats[s], "brand": f"b{s}_{int(brand[k])}",
                "price_eur": round(float(price[k]), 2), "description": " ".join(w) + " product",
                "is_active": True, "created_at": iso(self.t0),
            })
        self.products.extend(rows)
        self.p_sub = np.concatenate([self.p_sub, sub])
        self.p_quality = np.concatenate([self.p_quality, qual])
        self.p_price = np.concatenate([self.p_price, price])
        self.p_brand = np.concatenate([self.p_brand, brand])
        self.p_new = np.concatenate([self.p_new, np.full(n, new)])
        return rows

    @property
    def pid(self) -> list[str]:
        return [p["product_id"] for p in self.products]

    # ------------------------------------------------------------------ modelo oculto

    def relevance(self, u: int | None, recent: list[int], page: str) -> tuple[np.ndarray, np.ndarray]:
        """Atractivo (prob. de clic si se mira el slot) y prob. de compra dado clic."""
        n = len(self.products)
        z = -4.6 + 0.5 * self.p_quality + 0.6 * self.trend_boost[self.p_sub]
        price_fit = np.zeros(n)
        if u is not None:
            z = z + 2.5 * self.u_taste[u][self.p_sub]
            price_fit = -0.5 * (np.log(self.p_price) - self.u_price[u]) ** 2
        if recent:
            last = recent[-1]
            ls = self.p_sub[last]
            z = z + 1.6 * (self.p_sub == ls) + 1.1 * (self.p_sub == self.complement[ls])
            rec = np.zeros(n, dtype=bool)
            rec[recent] = True
            z = z + (1.0 if page == "product" else 1.5) * rec
            z = z - 0.8 * ((np.arange(n) == last) & (page == "product"))
        attract = 1 / (1 + np.exp(-z))
        b = 1 / (1 + np.exp(-(-1.0 + 0.4 * self.p_quality + 0.6 * price_fit)))
        if u is not None and u in self.u_bought:
            idx = list(self.u_bought[u])
            b[idx] *= 0.1
        return attract, b

    def expected_purchase(self, items: list[int], attract: np.ndarray, b: np.ndarray) -> float:
        """Cascada: el comprador mira los slots en orden y clica como mucho uno."""
        p_none, ep = 1.0, 0.0
        for k, i in enumerate(items[:10]):
            a = ATTN[k] * attract[i]
            ep += p_none * a * b[i]
            p_none *= 1 - a
        return ep

    def oracle(self, attract: np.ndarray, b: np.ndarray, active: np.ndarray) -> list[int]:
        v = np.where(active, attract * b, -1)
        return list(np.argsort(-v)[:10])

    # ------------------------------------------------------------------ generación

    def _session(self, u: int | None, t: float, n_views: int) -> list[tuple[int, float]]:
        rng = self.rng
        n = len(self.products)
        taste = self.u_taste[u] if u is not None else np.full(len(self.subcats), 1 / len(self.subcats))
        s = int(rng.choice(len(self.subcats), p=taste))
        views = []
        for _ in range(n_views):
            if views and rng.random() < 0.25:
                s = self.complement[s]
            cand = np.where((self.p_sub == s))[0]
            w = np.exp(0.8 * self.p_quality[cand])
            i = int(rng.choice(cand, p=w / w.sum()))
            t += float(rng.exponential(25.0))
            views.append((i, t))
        return views

    def history_events(self, n_sessions: int, days: int = 60) -> list[dict[str, Any]]:
        rng = self.rng
        out = []
        for sid in range(n_sessions):
            u = int(rng.integers(self.n_users))
            t = self.t0 + float(rng.uniform(0, days * 86400))
            views = self._session(u, t, int(rng.integers(1, 7)))
            for i, tv in views:
                ev = {"ts": iso(tv), "event_type": "view", "user_id": f"u_{u:05d}",
                      "session_id": f"s_{sid:07d}", "product_id": self.products[i]["product_id"],
                      "price_eur": float(self.p_price[i]), "device": str(rng.choice(DEVICES)),
                      "region": str(rng.choice(REGIONS))}
                out.append(ev)
                pf = -0.5 * (math.log(self.p_price[i]) - self.u_price[u]) ** 2
                bb = 1 / (1 + math.exp(-(-1.0 + 0.4 * self.p_quality[i] + 0.6 * pf)))
                if rng.random() < 0.15:
                    out.append({**ev, "event_type": "add_to_cart", "ts": iso(tv + 5)})
                    if rng.random() < bb:
                        out.append({**ev, "event_type": "purchase", "ts": iso(tv + 30)})
                        self.u_bought.setdefault(u, set()).add(i)
        return out

    def old_system(self, pop: np.ndarray, active: np.ndarray) -> tuple[list[int], list[float]]:
        """Best sellers + aleatoriedad (Plackett-Luce) con propensities condicionadas."""
        logits = np.where(active, 2.0 * np.log(pop + 1), -np.inf)
        shown, props = [], []
        avail = logits.copy()
        for _ in range(10):
            w = np.exp(avail - np.nanmax(avail[np.isfinite(avail)]))
            w[~np.isfinite(avail)] = 0
            p = w / w.sum()
            i = int(self.rng.choice(len(p), p=p))
            shown.append(i)
            props.append(float(p[i]))
            avail[i] = -np.inf
        return shown, props

    def make_request(self, t: float, new_user_rate: float = 0.25, new_products: np.ndarray | None = None,
                     rid: str = "") -> tuple[dict[str, Any], int | None, list[int]]:
        rng = self.rng
        is_new = rng.random() < new_user_rate
        u = None if is_new else int(rng.integers(self.n_users))
        page = str(rng.choice(PAGES, p=[0.4, 0.45, 0.15]))
        n_views = 0 if page == "home" and rng.random() < 0.6 else int(rng.integers(1, 6))
        views = self._session(u, t - 300, n_views) if n_views else []
        if new_products is not None and len(new_products) and views and rng.random() < 0.3:
            j = int(rng.choice(new_products))
            views[-1] = (j, views[-1][1])
        uid = f"u_new_{rng.integers(1e9)}" if u is None else f"u_{u:05d}"
        req = {"request_id": rid or f"r_{rng.integers(1e12):012d}", "ts": iso(t), "user_id": uid,
               "is_new_user": bool(is_new),
               "context": {"device": str(rng.choice(DEVICES)), "region": str(rng.choice(REGIONS)), "page": page},
               "recent_views": [{"product_id": self.products[i]["product_id"], "ts": iso(min(tv, t - 1))}
                                for i, tv in views]}
        return req, u, [i for i, _ in views]

    def history_sessions(self, n: int, pop: np.ndarray, days: int = 60) -> list[dict[str, Any]]:
        rng = self.rng
        active = np.ones(len(self.products), dtype=bool)
        out = []
        for k in range(n):
            t = self.t0 + float(rng.uniform(0, days * 86400))
            req, u, rec = self.make_request(t, new_user_rate=0.15, rid=f"h_{k:08d}")
            shown, props = self.old_system(pop, active)
            a, b = self.relevance(u, rec, req["context"]["page"])
            clicked = purchased = None
            rev = 0.0
            for s, i in enumerate(shown):
                if rng.random() < ATTN[s] * a[i]:
                    clicked = self.products[i]["product_id"]
                    if rng.random() < b[i]:
                        purchased = clicked
                        rev = float(self.p_price[i])
                    break
            out.append({**req, "shown": [self.products[i]["product_id"] for i in shown],
                        "propensity": props, "clicked": clicked, "purchased": purchased, "revenue_eur": rev})
        return out

    def outcome(self, u: int | None, rec: list[int], page: str, items: list[str]) -> tuple[str, str, float, float]:
        """Clic/compra realizados + compra esperada para una lista servida."""
        idx = {p["product_id"]: i for i, p in enumerate(self.products)}
        ii = [idx[p] for p in items if p in idx]
        a, b = self.relevance(u, rec, page)
        ep = self.expected_purchase(ii, a, b)
        clicked = purchased = ""
        rev = 0.0
        for s, i in enumerate(ii):
            if self.rng.random() < ATTN[s] * a[i]:
                clicked = self.products[i]["product_id"]
                if self.rng.random() < b[i]:
                    purchased = clicked
                    rev = float(self.p_price[i])
                    if u is not None:
                        self.u_bought.setdefault(u, set()).add(i)
                break
        return clicked, purchased, rev, ep


def organic_popularity(world: World, events: list[dict[str, Any]]) -> np.ndarray:
    idx = {p["product_id"]: i for i, p in enumerate(world.products)}
    pop = np.zeros(len(world.products))
    w = {"view": 1.0, "add_to_cart": 3.0, "purchase": 8.0}
    for e in events:
        pop[idx[e["product_id"]]] += w[e["event_type"]]
    return pop


def build_world(seed: int = 0, n_products: int = 600, n_sessions_events: int = 20_000,
                n_sessions_log: int = 15_000) -> dict[str, Any]:
    w = World(seed=seed, n_products=n_products)
    events = w.history_events(n_sessions_events)
    pop = organic_popularity(w, events)
    sessions = w.history_sessions(n_sessions_log, pop)
    return {"world": w, "catalog": list(w.products), "events": events, "sessions": sessions, "pop": pop}


def ts_from_iso(s: str) -> float:
    return datetime.strptime(s.rstrip("Z"), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


_ = (math, timedelta)
