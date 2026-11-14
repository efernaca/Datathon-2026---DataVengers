"""Señales en vivo a partir de `shop.events` (D1, V5) y compras en vivo (V2).

* Contadores con decaimiento exponencial (semivida corta, larga y de popularidad).
  Se usa el `ts` de los eventos como reloj, no el del portátil.
* Popularidad suavizada hacia el prior histórico:
      pop(i) = (n_vivo(i) + κ · prior(i)) / (N_vivo + κ)
  Al empezar la final manda el prior; según llegan eventos, mandan los datos. Para
  un producto nuevo, el prior es el heredado de sus vecinos (ver `catalog.py`).
* Tendencia: log((n_corto + κ_t) / (n_esperado + κ_t)), con n_esperado sacado del
  contador largo reescalado. Sustituye al cociente con denominador 1 de la Etapa 1.
"""
from __future__ import annotations

import math
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .catalog import parse_ts
from .config import Config
from .state import CatalogState, LiveView

LN2 = math.log(2.0)


class DecayedCounter:
    """Σ w·exp(-λ (t_now - t_e)) por clave, guardado como Σ w·exp(λ (t_e - t0))."""

    def __init__(self, halflife_sec: float) -> None:
        self.lam = LN2 / halflife_sec
        self.t0: float | None = None
        self.v: dict[str, float] = defaultdict(float)
        self.total = 0.0

    def add(self, key: str, w: float, t: float) -> None:
        if self.t0 is None:
            self.t0 = t
        if (t - self.t0) * self.lam > 300:     # reescalar para evitar desbordes
            f = math.exp(-(t - self.t0) * self.lam)
            for k in self.v:
                self.v[k] *= f
            self.total *= f
            self.t0 = t
        x = w * math.exp((t - self.t0) * self.lam)
        self.v[key] += x
        self.total += x

    def factor(self, t_now: float) -> float:
        return 0.0 if self.t0 is None else math.exp(-(t_now - self.t0) * self.lam)

    def array(self, ids: Iterable[str], t_now: float) -> np.ndarray:
        f = self.factor(t_now)
        return np.array([self.v.get(p, 0.0) * f for p in ids])


class LiveTracker:
    """Lo alimenta el hilo de eventos. `view()` produce un `LiveView` alineado al catálogo."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.short = DecayedCounter(cfg.live_halflife_short_min * 60)
        self.long = DecayedCounter(cfg.live_halflife_long_min * 60)
        self.pop = DecayedCounter(cfg.live_pop_halflife_min * 60)
        self.now_ts = 0.0
        self.n_events = 0
        self._lock = threading.Lock()

    def ingest(self, events: Iterable[Mapping[str, Any]]) -> list[tuple[str, str]]:
        """Devuelve las compras (user_id, product_id) para excluirlas (V2)."""
        purchases: list[tuple[str, str]] = []
        with self._lock:
            for e in events:
                et = str(e.get("event_type", ""))
                w = self.cfg.event_weight(et)
                pid = str(e.get("product_id", ""))
                t = parse_ts(e.get("ts"))
                if not pid or w <= 0 or t <= 0:
                    continue
                self.short.add(pid, w, t)
                self.long.add(pid, w, t)
                self.pop.add(pid, w, t)
                self.now_ts = max(self.now_ts, t)
                self.n_events += 1
                if et == "purchase" and e.get("user_id"):
                    purchases.append((str(e["user_id"]), pid))
        return purchases

    def view(self, cat: CatalogState, kappa_mult: float = 1.0) -> LiveView:
        with self._lock:
            t = self.now_ts
            ids = cat.ids
            n = len(ids)
            live = self.pop.array(ids, t)
            total = float(live.sum())
            kappa = self.cfg.live_pop_kappa * kappa_mult
            share = (live + kappa * cat.pop_share_prior) / (total + kappa)
            log_pop = np.clip(np.log(np.maximum(share, 1e-12) * n), -6, 6)
            s = self.short.array(ids, t)
            l_ = self.long.array(ids, t)
            ratio = self.cfg.live_halflife_short_min / self.cfg.live_halflife_long_min
            kt = self.cfg.trend_kappa
            trend = np.log((s + kt) / (l_ * ratio + kt))
            return LiveView(n_items=n, log_pop=log_pop, trend=np.clip(trend, -2, 3),
                            total_weight=total, now_ts=t, kappa_mult=kappa_mult)
