"""Proceso SERVIDOR de la Etapa 2 · versión especial (4.5).

Solo lee peticiones, puntúa y envía, más lecturas incrementales pequeñas. Las consultas
pesadas las hace el proceso CONSTRUCTOR (`builder.py`), que escribe el histórico y los
pesos en disco de forma atómica; el servidor los recarga cuando cambian.

Hilos (F1, F4):
  * principal ...... lee `shop.requests` (S1, con edad), puntúa y encola. Nunca espera a un push.
  * pushers (pool) . insertan en `submissions`; reintentan solo si la petición más antigua
                     tiene < 2,2 s; marcan como hechas solo al confirmar (F2).
  * catálogo ....... cada 1 s: `catalog_updates` incremental (S2): stock, precios y
                     productos nuevos con vecinos en el momento (C1, V6).
  * eventos ........ cada 2 s: `shop.events` incremental → popularidad/tendencia con
                     decaimiento (D1, V5) y compras en vivo (V2).
  * feedback ....... cada 5 s: `feedback` incremental → Thompson (D2), compras propias (V2),
                     métricas de fiabilidad.
  * estado ......... cada 10 s: ¿hay histórico o pesos nuevos del constructor? → recarga.
  * registro ....... escribe las señales de cada lista servida (JSONL local + scratch) para
                     aprender los pesos con el feedback (4.2). Nunca desde el hilo que responde.
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
import os
import signal
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import gc
import json
import queue

from .catalog import CatalogManager, parse_ts
from .config import Config
from .guardrails import ensure_valid_submission, validate_submission_record
from .history import cache_path, get_or_build_history, load_aggregates, load_history
from .live import LiveTracker
from .policy import Stage2Special, Weights
from .state import Snapshot, StateStore

logger = logging.getLogger("e2esp.serve")


def fmt_ts(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


SQL_REQUESTS = """
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
       user_id::String AS user_id, is_new_user, context, recent_views,
       dateDiff('millisecond', ts::DateTime64(3), now64(3)) AS age_ms
FROM shop.requests
WHERE ts::DateTime64(3) > now64(3) - INTERVAL {window} SECOND
ORDER BY ts::DateTime64(3) ASC
"""

SQL_REQUESTS_LEGACY = """
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
       user_id::String AS user_id, is_new_user, context, recent_views
FROM shop.requests ORDER BY ts::DateTime64(3) DESC LIMIT 60
"""

SQL_CATALOG_UPDATES = """
SELECT toString(ts::DateTime64(3)) AS ts, change::String AS change,
       product_id::String AS product_id, ifNull(category::String, '') AS category,
       ifNull(subcategory::String, '') AS subcategory, ifNull(brand::String, '') AS brand,
       price_eur::Nullable(Float64) AS price_eur, ifNull(description::String, '') AS description,
       is_active::Nullable(Bool) AS is_active
FROM shop.catalog_updates
WHERE ts::DateTime64(3) >= '{since}'
ORDER BY ts::DateTime64(3) ASC
LIMIT 20000
"""

SQL_EVENTS = """
SELECT toString(ts::DateTime64(3)) AS ts, event_type::String AS event_type,
       user_id::String AS user_id, product_id::String AS product_id
FROM shop.events
WHERE ts::DateTime64(3) >= '{since}'
ORDER BY ts::DateTime64(3) ASC
LIMIT 50000
"""

SQL_FEEDBACK = """
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
       served_by::String AS served_by, ifNull(fallback_reason::String, '') AS fallback_reason,
       latency_ms::Float64 AS latency_ms, ifNull(clicked::String, '') AS clicked,
       ifNull(purchased::String, '') AS purchased, revenue_eur::Float64 AS revenue_eur
FROM {team}.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '{since}'
ORDER BY ts::DateTime64(3) ASC
LIMIT 20000
"""


class IncrementalReader:
    """Lee una tabla append-only por marca de agua `ts`, con solape de unos segundos y
    deduplicación (los inserts pueden llegar algo desordenados)."""

    def __init__(self, client: Any, sql: str, key: Callable[[dict], Any], since: float,
                 overlap_sec: float = 3.0, timeout: float = 3.0, **fmt: Any) -> None:
        self.client, self.sql, self.key = client, sql, key
        self.wm = since
        self.overlap = overlap_sec
        self.timeout = timeout
        self.fmt = fmt
        self.seen: dict[Any, float] = {}

    def fetch(self) -> list[dict[str, Any]] | None:
        since = fmt_ts(max(self.wm - self.overlap, 0))
        rows = self.client.query_safe(self.sql.format(since=since, **self.fmt), timeout=self.timeout)
        if rows is None:
            return None
        fresh = []
        for r in rows:
            k = self.key(r)
            if k in self.seen:
                continue
            t = parse_ts(r.get("ts"))
            self.seen[k] = t
            self.wm = max(self.wm, t)
            fresh.append(r)
        cutoff = self.wm - 4 * self.overlap
        if len(self.seen) > 5000:
            self.seen = {k: t for k, t in self.seen.items() if t >= cutoff}
        return fresh


class SignalLogger(threading.Thread):
    """4.2: guarda (señales, posición, interruptores) de cada lista servida. Hilo aparte."""

    def __init__(self, cfg: Config, client: Any = None, to_scratch: bool = False) -> None:
        super().__init__(name="signals", daemon=True)
        self.q: queue.Queue = queue.Queue(maxsize=100_000)
        self.dir = Path(cfg.log_dir)
        self.client = client if to_scratch else None
        self.stop_ev = threading.Event()

    def put(self, rec: dict) -> None:
        try:
            self.q.put_nowait(rec)
        except queue.Full:
            pass

    def run(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        buf: list[dict] = []
        last = time.monotonic()
        while not self.stop_ev.is_set() or not self.q.empty():
            try:
                buf.append(self.q.get(timeout=0.5))
            except queue.Empty:
                pass
            if buf and (len(buf) >= 200 or time.monotonic() - last > 10 or self.stop_ev.is_set()):
                path = self.dir / f"signals_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
                with path.open("a", encoding="utf-8") as fh:
                    for r in buf:
                        fh.write(json.dumps(r) + "\n")
                if self.client is not None:
                    try:
                        self.client.push_scratch([{**r, "kind": "signals"} for r in buf])
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("scratch signals: %s", exc)
                buf = []
                last = time.monotonic()


class Service:
    def __init__(self, client: Any, cfg: Config, history, *, weights: Weights | None = None,
                 frozen_switches: dict[str, int] | None = None, standby: bool = False,
                 cold_boot: bool = False, seed: int | None = None, write_heartbeat: bool = True,
                 signals_to_scratch: bool = False) -> None:
        self.client, self.cfg = client, cfg
        self.standby = standby
        self.write_heartbeat = write_heartbeat and not standby
        self.stop = threading.Event()
        self.history = history
        self.cm = CatalogManager(history, cfg)
        self.tracker = LiveTracker(cfg)
        self.weights_path = Path(cfg.state_dir) / "weights.json"
        self.weights_mtime = self.weights_path.stat().st_mtime if self.weights_path.is_file() else 0.0
        if weights is None:
            weights = Weights.load(self.weights_path)
        self.rec = Stage2Special(cfg, weights=weights, frozen_switches=frozen_switches)
        self.model_name = weights.version
        self.update_log: list[dict] = []
        self.siglog = SignalLogger(cfg, client, to_scratch=signals_to_scratch)
        self.n8 = {"lists": 0, "with_new": 0, "pos_sum": 0}
        self.scratch_cold_done: set[str] = set()
        self.cold_boot = cold_boot
        self.store = StateStore()
        self.new_impressions: dict[str, int] = {}
        self.live_bought: dict[str, set[str]] = {}
        self.meta: OrderedDict[str, tuple] = OrderedDict()
        self.meta_lock = threading.Lock()
        self.done: set[str] = set()
        self.done_q: deque[str] = deque()
        self.inflight: dict[str, dict] = {}
        self.answers: OrderedDict[str, dict] = OrderedDict()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=cfg.push_workers, thread_name_prefix="push")
        self.stats = {"answered": 0, "late_dropped": 0, "push_fail": 0, "retries": 0, "guard_fallback": 0,
                      "fb_team": 0, "fb_fallback": 0, "fb_clicks": 0, "fb_buys": 0}
        self.push_ages: deque[float] = deque(maxlen=2000)
        self.read_ages: deque[float] = deque(maxlen=2000)
        self.legacy_mode = False
        self._s1_fail = 0
        self.clock_skew_ms: deque[float] = deque(maxlen=500)

    # ------------------------------------------------------------------ arranque

    def bootstrap(self) -> None:
        """Estado inicial: catálogo + todas las actualizaciones + eventos recientes."""
        upd = self.client.query_safe(SQL_CATALOG_UPDATES.format(since="1970-01-01 00:00:00"),
                                     timeout=self.cfg.query_timeout_slow_sec) or []
        self.cm.apply_updates(upd)
        self.update_log.extend(upd)
        cat = self.cm.build()
        last_upd = max((parse_ts(r["ts"]) for r in upd), default=time.time() - 5)
        self.cat_reader = IncrementalReader(self.client, SQL_CATALOG_UPDATES,
                                            key=lambda r: (r["ts"], r["product_id"], r["change"]),
                                            since=last_upd, timeout=self.cfg.query_timeout_bg_sec)
        for r in upd:
            self.cat_reader.seen[(r["ts"], r["product_id"], r["change"])] = parse_ts(r["ts"])
        warm_since = time.time() - (0 if self.cold_boot else 3 * 3600)
        self.ev_reader = IncrementalReader(self.client, SQL_EVENTS,
                                           key=lambda r: (r["ts"], r["user_id"], r["product_id"], r["event_type"]),
                                           since=warm_since, overlap_sec=3.0, timeout=10.0)
        if not self.cold_boot:
            for _ in range(20):    # puede haber > 50.000 eventos en 3 h: se lee por tramos
                rows = self.ev_reader.fetch()
                if not rows:
                    break
                self._ingest_events(rows, publish=False)
                if len(rows) < 40000:
                    break
        self.ev_reader.timeout = self.cfg.query_timeout_bg_sec
        self.fb_reader = IncrementalReader(self.client, SQL_FEEDBACK,
                                           key=lambda r: r["request_id"], since=time.time() - 60,
                                           timeout=self.cfg.query_timeout_bg_sec, team=self.client.team_id)
        self.store.set(Snapshot(history=self.history, catalog=cat, live=self._live_views(cat),
                                live_bought={}, new_impressions=self.new_impressions))
        logger.info("Estado inicial: %d productos (%d activos, %d nuevos), %d eventos vivos",
                    cat.n, len(cat.active_ids), int(cat.is_new.sum()), self.tracker.n_events)

    def _live_views(self, cat) -> dict:
        return {"default": self.tracker.view(cat)}

    # ------------------------------------------------------------------ hilos de fondo

    def _loop(self, name: str, period: float, fn: Callable[[], None]) -> threading.Thread:
        def run() -> None:
            while not self.stop.is_set():
                t0 = time.monotonic()
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s: %s", name, exc)
                self.stop.wait(max(0.05, period - (time.monotonic() - t0)))
        th = threading.Thread(target=run, name=name, daemon=True)
        th.start()
        return th

    def catalog_tick(self) -> None:
        rows = self.cat_reader.fetch()
        if not rows:
            return
        n = self.cm.apply_updates(rows)
        if n:
            self.update_log.extend(rows)
            cat = self.cm.build()
            self.store.update(catalog=cat, live=self._live_views(cat))
            new = [p for p in cat.ids if cat.is_new[cat.idx[p]] and p not in self.scratch_cold_done]
            if new:   # N4: vecinos de cada producto nuevo, escritos UNA vez en scratch
                self.scratch_cold_done.update(new)
                recs = [{"kind": "cold_nb", "model_version": self.cfg.model_version, "product_id": p,
                         "neighbors": [q for q, _ in cat.cold_neighbors.get(p, ())],
                         "w": [round(x, 4) for _, x in cat.cold_neighbors.get(p, ())],
                         "ts": fmt_ts(time.time())} for p in new]
                self.pool.submit(self._push_scratch_safe, recs)

    def _ingest_events(self, rows: list[dict], publish: bool = True) -> None:
        buys = self.tracker.ingest(rows)
        for uid, pid in buys:
            self.live_bought.setdefault(uid, set()).add(pid)
        if publish:
            self._publish_live(bool(buys))

    def _publish_live(self, bought_changed: bool) -> None:
        def fn(snap: Snapshot) -> dict:
            ch: dict[str, Any] = {"live": self._live_views(snap.catalog)}
            if bought_changed:
                ch["live_bought"] = {u: frozenset(s) for u, s in self.live_bought.items()}
            return ch
        self.store.mutate(fn)

    def _push_scratch_safe(self, recs: list[dict]) -> None:
        try:
            self.client.push_scratch(recs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("scratch: %s", exc)

    def events_tick(self) -> None:
        rows = self.ev_reader.fetch()
        if rows is None:
            return
        self._ingest_events(rows)

    def feedback_tick(self) -> None:
        rows = self.fb_reader.fetch()
        if not rows:
            return
        bought = False
        lat = []
        for r in rows:
            rid = r["request_id"]
            with self.meta_lock:
                m = self.meta.pop(rid, None)
            team = r.get("served_by") == "team"
            self.stats["fb_team" if team else "fb_fallback"] += 1
            if r.get("clicked"):
                self.stats["fb_clicks"] += 1
            if m is not None:
                uid, segment, push_age = m
                if r.get("purchased") and uid:
                    self.live_bought.setdefault(uid, set()).add(str(r["purchased"]))
                    bought = True
                if push_age is not None and r.get("latency_ms") not in (None, ""):
                    lat.append(float(r["latency_ms"]) - push_age)
            if r.get("purchased"):
                self.stats["fb_buys"] += 1
            if not team:
                logger.warning("fallback en %s: %s", rid, r.get("fallback_reason"))
        if lat:
            self.clock_skew_ms.extend(lat)
        if bought:
            self._publish_live(True)

    def housekeeping_tick(self) -> None:
        pa = np.array(self.push_ages) if self.push_ages else np.zeros(1)
        ra = np.array(self.read_ages) if self.read_ages else np.zeros(1)
        sk = np.median(self.clock_skew_ms) if self.clock_skew_ms else float("nan")
        logger.info(
            "stats %s | edad lectura p50=%.0f p95=%.0f ms | edad push p50=%.0f p95=%.0f max=%.0f ms | "
            "latency_ms(feedback)-edad_push mediana=%.0f ms%s",
            self.stats, np.median(ra), np.percentile(ra, 95), np.median(pa), np.percentile(pa, 95), pa.max(),
            sk, " | MODO LEGACY" if self.legacy_mode else "")
        n8 = self.n8
        logger.info("N8: %.1f%% de listas con algún producto nuevo, posición media %.1f | pesos=%s",
                    100 * n8["with_new"] / max(n8["lists"], 1), n8["pos_sum"] / max(n8["with_new"], 1),
                    self.rec.weights.version)

    def state_tick(self) -> None:
        """4.5: recarga lo que escribe el constructor (pesos e histórico), sin consultas pesadas."""
        if self.weights_path.is_file():
            mt = self.weights_path.stat().st_mtime
            if mt != self.weights_mtime:
                self.weights_mtime = mt
                w = Weights.load(self.weights_path)
                self.rec.weights = w
                self.model_name = w.version
                logger.info("Pesos recargados: %s", w.version)
        vfile = Path(self.cfg.state_dir) / "history_version.txt"
        if vfile.is_file():
            v = vfile.read_text().strip()
            if v and v != self.history.version:
                path = cache_path(self.cfg, None)
                h = load_history(path)
                if h is not None and h.version == v:
                    cm = CatalogManager(h, self.cfg)
                    cm.apply_updates(self.update_log)
                    cat = cm.build()
                    self.cm, self.history = cm, h
                    self.store.update(history=h, catalog=cat, live=self._live_views(cat))
                    gc.collect()
                    gc.freeze()
                    logger.info("Histórico recargado: %s", v)

    # ------------------------------------------------------------------ bucle rápido

    def read_requests(self) -> list[dict]:
        t0 = time.monotonic()
        if not self.legacy_mode:
            rows = self.client.query_safe(SQL_REQUESTS.format(window=self.cfg.read_window_sec),
                                          timeout=self.cfg.query_timeout_fast_sec)
            if rows is None:
                self._s1_fail += 1
                if self._s1_fail >= 5:
                    logger.error("S1 falla 5 veces seguidas: paso a la lectura legacy")
                    self.legacy_mode = True
                return []
            self._s1_fail = 0
        else:
            rows = self.client.query_safe(SQL_REQUESTS_LEGACY, timeout=self.cfg.query_timeout_fast_sec) or []
        t_read = time.monotonic()
        now = time.time()
        for r in rows:
            age = r.get("age_ms")
            if age in (None, ""):
                age = (now - parse_ts(r.get("ts"))) * 1000.0
            age = float(age)
            # la edad la calcula el servidor al ejecutar la consulta: sumamos el viaje de vuelta
            r["_t_req"] = t_read - age / 1000.0 - (t_read - t0) / 2
        return rows

    def build_record(self, raw: dict, snap: Snapshot) -> tuple[dict, tuple]:
        try:
            d = self.rec.recommend(raw, snap)
            items, tag = d.items, d.tag(self.cfg.model_version, self.rec.weights.version)
            meta = (d.user_id, d.segment)
            self.n8["lists"] += 1
            if d.new_positions:
                self.n8["with_new"] += 1
                self.n8["pos_sum"] += float(np.mean(d.new_positions))
            self.siglog.put({"request_id": d.request_id, "ts": raw.get("ts"), "segment": d.segment,
                             "switches": d.switches, "shuf": int(d.shuffled), "fin": int(d.final_mode),
                             "new_user": int(d.is_new_user), "new_pos": d.new_positions,
                             "items": d.items, "signals": d.signals, "wver": self.rec.weights.version})
        except Exception as exc:  # noqa: BLE001
            logger.error("recommend falló en %s: %s", raw.get("request_id"), exc)
            items, tag, meta = [], f"{self.cfg.model_version}|error", (str(raw.get("user_id", "")), "")
        rec = {"request_id": str(raw["request_id"]), "items": items, "model_version": tag}
        if not validate_submission_record(rec, snap.catalog.active_ids):
            # F8: nunca descartar en silencio: mandar populares válidos y registrarlo
            self.stats["guard_fallback"] += 1
            rec["items"] = ensure_valid_submission([], snap.catalog.active_ids, snap.catalog.global_ranked)
            rec["model_version"] = f"{self.cfg.model_version}|guard"
            logger.warning("F8: respuesta inválida para %s, envío populares", rec["request_id"])
        return rec, meta

    def _mark_done(self, rid: str) -> None:
        self.done.add(rid)
        self.done_q.append(rid)
        if len(self.done_q) > self.cfg.seen_window_size:
            self.done.discard(self.done_q.popleft())

    def _push(self, batch: list[tuple[dict, float]], attempt: int = 0) -> None:
        cfg = self.cfg
        now = time.monotonic()
        oldest = max((now - t) * 1000 for _, t in batch)
        timeout = max(0.2, min(cfg.push_timeout_sec, (cfg.drop_age_ms - oldest) / 1000))
        try:
            self.client.push_submissions([r for r, _ in batch], timeout=timeout)
            done_t = time.monotonic()
            with self.lock:
                for r, t in batch:
                    self.inflight.pop(r["request_id"], None)
                    self._mark_done(r["request_id"])
                    age = (done_t - t) * 1000
                    self.push_ages.append(age)
                    with self.meta_lock:
                        m = self.meta.get(r["request_id"])
                        if m is not None:
                            self.meta[r["request_id"]] = (*m[:2], age)
                self.stats["answered"] += len(batch)
        except Exception as exc:  # noqa: BLE001
            self.stats["push_fail"] += 1
            now = time.monotonic()
            alive = [(r, t) for r, t in batch if (now - t) * 1000 < cfg.retry_max_age_ms]
            logger.warning("push falló (%s); %d/%d aún a tiempo", str(exc)[:120], len(alive), len(batch))
            alive_ids = {r["request_id"] for r, _ in alive}
            with self.lock:
                for r, _ in batch:
                    if r["request_id"] not in alive_ids:
                        self.inflight.pop(r["request_id"], None)
            if alive and attempt == 0:
                self.stats["retries"] += 1
                self._push(alive, attempt=1)
            else:
                with self.lock:   # F2: no se marcan como hechas: si siguen a tiempo se releerán
                    for r, _ in alive:
                        self.inflight.pop(r["request_id"], None)

    def heartbeat_alive(self) -> bool:
        try:
            return time.time() - os.path.getmtime(self.cfg.heartbeat_file) < self.cfg.standby_takeover_sec
        except OSError:
            return False

    def step(self) -> int:
        rows = self.read_requests()
        if self.write_heartbeat:
            Path(self.cfg.heartbeat_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self.cfg.heartbeat_file).touch()
        if not rows:
            return 0
        snap = self.store.get()
        if snap is None:
            return 0
        if self.standby and self.heartbeat_alive():
            return 0
        now = time.monotonic()
        batch: list[tuple[dict, float]] = []
        for raw in rows:
            rid = str(raw.get("request_id", ""))
            if not rid:
                continue
            with self.lock:
                if rid in self.done or rid in self.inflight:
                    continue
            age = (now - raw["_t_req"]) * 1000
            self.read_ages.append(age)
            if age > self.cfg.drop_age_ms:
                with self.lock:
                    self._mark_done(rid)
                self.stats["late_dropped"] += 1
                continue
            rec = self.answers.get(rid)
            if rec is None:
                rec, meta = self.build_record(raw, snap)
                self.answers[rid] = rec
                if len(self.answers) > 5000:
                    self.answers.popitem(last=False)
                with self.meta_lock:
                    self.meta[rid] = (*meta, None)
                    if len(self.meta) > 100_000:
                        self.meta.popitem(last=False)
            with self.lock:
                self.inflight[rid] = rec
            batch.append((rec, raw["_t_req"]))
        if batch:
            self.pool.submit(self._push, batch)
        return len(batch)

    # ------------------------------------------------------------------ ejecución

    def start_background(self) -> list[threading.Thread]:
        cfg = self.cfg
        ths = [
            self._loop("catalog", cfg.catalog_refresh_sec, self.catalog_tick),
            self._loop("events", cfg.events_refresh_sec, self.events_tick),
            self._loop("feedback", cfg.feedback_refresh_sec, self.feedback_tick),
            self._loop("housekeeping", 30.0, self.housekeeping_tick),
            self._loop("state", cfg.state_reload_check_sec, self.state_tick),
        ]
        self.siglog.start()
        return ths

    def run(self, duration: float | None = None) -> None:
        self.start_background()
        logger.info("Bucle rápido en marcha (team=%s, modelo=%s, standby=%s)",
                    getattr(self.client, "team_id", "?"), self.model_name, self.standby)
        t_end = None if duration is None else time.monotonic() + duration
        while not self.stop.is_set() and (t_end is None or time.monotonic() < t_end):
            t0 = time.monotonic()
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                logger.error("bucle rápido: %s", exc)
                time.sleep(0.5)
            self.stop.wait(max(0.0, self.cfg.poll_interval_sec - (time.monotonic() - t0)))
        self.stop.set()
        self.pool.shutdown(wait=True)
        self.siglog.stop_ev.set()
        self.siglog.join(timeout=15)


def main() -> None:
    ap = argparse.ArgumentParser(description="Etapa 2 · versión especial · servidor")
    ap.add_argument("--freeze", action="store_true",
                    help="domingo: sin experimentos; interruptores de state/switches.json (análisis 4.1)")
    ap.add_argument("--no-experiments", action="store_true", help="sin interruptores ni barajado ni modo final")
    ap.add_argument("--standby", action="store_true")
    ap.add_argument("--cold-boot", action="store_true", help="arrancar sin contadores vivos (ensayo 4.4)")
    ap.add_argument("--from-scratch", action="store_true", help="si no hay caché local, leer agregados de scratch")
    ap.add_argument("--signals-to-scratch", action="store_true", help="copiar el registro de señales a scratch")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
    cfg = Config.from_env(experiments=False if (args.freeze or args.no_experiments) else None)
    if not cfg.team_id or not cfg.api_key:
        raise SystemExit("Faltan TEAM_ID / RAWTREE_API_KEY (.env o group_API.env).")
    from .rawtree_client import RawTreeClient

    client = RawTreeClient(cfg.api_url, cfg.api_key, cfg.team_id)
    history = load_history(cache_path(cfg, None))
    if history is None and args.from_scratch:
        history = load_aggregates(client)
    if history is None:
        logger.warning("No hay histórico del constructor: lo calculo aquí una vez (mejor: python -m src.builder)")
        history = get_or_build_history(client, cfg)
    gc.collect()
    gc.freeze()            # N1: los objetos del histórico no vuelven a pasar por el recolector
    frozen = None
    if args.freeze:
        p = Path(cfg.state_dir) / "switches.json"
        frozen = json.loads(p.read_text()).get("switches") if p.is_file() else None
        logger.info("Modo congelado. Interruptores: %s", frozen or "todos encendidos salvo rev")
    svc = Service(client, cfg, history, frozen_switches=frozen, standby=args.standby, cold_boot=args.cold_boot,
                  signals_to_scratch=args.signals_to_scratch)

    def on_signal(*_: Any) -> None:
        svc.stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    for attempt in range(1, 1000):
        try:
            svc.bootstrap()
            break
        except Exception as exc:  # noqa: BLE001
            logger.error("bootstrap intento %d: %s", attempt, exc)
            time.sleep(min(10, attempt))
    svc.run()


if __name__ == "__main__":
    main()
