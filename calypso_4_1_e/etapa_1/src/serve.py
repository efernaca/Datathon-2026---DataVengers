"""Servicio en vivo (Etapa 1 · versión especial).

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
El histórico se carga de la caché de disco (F5); el arranque no depende de RawTree
más allá del catálogo inicial. Ver `run_forever.sh` / `run_forever.ps1` (F6) y
`monitor.py` (F7).
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

from .bandit import ThompsonBandit
from .catalog import CatalogManager, parse_ts
from .config import Config
from .guardrails import ensure_valid_submission, validate_submission_record
from .history import get_or_build_history, load_ease_from_scratch
from .live import LiveTracker
from .model import LinearModel, load_model
from .policy import Recommender
from .state import Snapshot, StateStore

logger = logging.getLogger("etapa1esp.serve")


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


class Service:
    def __init__(self, client: Any, cfg: Config, model: Any, history, *, standby: bool = False,
                 cold_boot: bool = False, seed: int | None = None, write_heartbeat: bool = True) -> None:
        self.client, self.cfg = client, cfg
        self.standby = standby
        self.write_heartbeat = write_heartbeat and not standby
        self.stop = threading.Event()
        self.history = history
        self.cm = CatalogManager(history, cfg)
        self.tracker = LiveTracker(cfg)
        self.bandit = ThompsonBandit([a[0] for a in cfg.bandit_arms], seed=seed)
        self.bandit_path = Path(cfg.cache_dir) / "bandit_state.json"
        if not cold_boot:
            self.bandit.load(self.bandit_path)
        self.rec = Recommender(cfg, model=model, control_model=LinearModel.default(),
                               bandit=self.bandit, seed=seed)
        self.model_name = getattr(model, "name", "model")
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
        return {a[0]: self.tracker.view(cat, a[1]) for a in self.cfg.bandit_arms}

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
            cat = self.cm.build()
            self.store.update(catalog=cat, live=self._live_views(cat))

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
                uid, segment, arm, variant, push_age = m
                if team and variant == "model":
                    self.bandit.update(segment, arm, 1.0 if r.get("clicked") else 0.0)
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
        self.bandit.save(self.bandit_path)
        pa = np.array(self.push_ages) if self.push_ages else np.zeros(1)
        ra = np.array(self.read_ages) if self.read_ages else np.zeros(1)
        sk = np.median(self.clock_skew_ms) if self.clock_skew_ms else float("nan")
        logger.info(
            "stats %s | edad lectura p50=%.0f p95=%.0f ms | edad push p50=%.0f p95=%.0f max=%.0f ms | "
            "latency_ms(feedback)-edad_push mediana=%.0f ms%s",
            self.stats, np.median(ra), np.percentile(ra, 95), np.median(pa), np.percentile(pa, 95), pa.max(),
            sk, " | MODO LEGACY" if self.legacy_mode else "")
        logger.info("bandit %s", {s: {a: round(v["mean"], 4) for a, v in arms.items()}
                                  for s, arms in self.bandit.summary().items()})

    def history_tick(self) -> None:
        cfg = self.cfg
        h = get_or_build_history(self.client, cfg, rebuild=True)
        self.history = h
        self.store.update(history=h)

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
            items, tag = d.items, f"{self.cfg.model_version}-{self.model_name if d.variant == 'model' else 'heur'}-{d.arm}"
            meta = (d.user_id, d.segment, d.arm, d.variant)
            for pid in items:
                j = snap.catalog.idx.get(pid)
                if j is not None and snap.catalog.is_new[j]:
                    self.new_impressions[pid] = self.new_impressions.get(pid, 0) + 1
        except Exception as exc:  # noqa: BLE001
            logger.error("recommend falló en %s: %s", raw.get("request_id"), exc)
            items, tag, meta = [], f"{self.cfg.model_version}-error", (str(raw.get("user_id", "")), "", "", "")
        rec = {"request_id": str(raw["request_id"]), "items": items, "model_version": tag}
        if not validate_submission_record(rec, snap.catalog.active_ids):
            # F8: nunca descartar en silencio: mandar populares válidos y registrarlo
            self.stats["guard_fallback"] += 1
            rec["items"] = ensure_valid_submission([], snap.catalog.active_ids, snap.catalog.global_ranked)
            rec["model_version"] = f"{self.cfg.model_version}-guard"
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
                            self.meta[r["request_id"]] = (*m[:4], age)
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
        ]
        if cfg.history_refresh_sec > 0:
            ths.append(self._loop("history", cfg.history_refresh_sec, self.history_tick))
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
        self.bandit.save(self.bandit_path)


def default_model_path(cfg: Config) -> Path | None:
    sel = Path(cfg.artifacts_dir) / "selected.txt"
    if sel.is_file() and (Path(cfg.artifacts_dir) / sel.read_text().strip()).is_file():
        return Path(cfg.artifacts_dir) / sel.read_text().strip()
    for name in ("model.json", "model_gbm.pkl"):
        p = Path(cfg.artifacts_dir) / name
        if p.is_file():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Etapa 1 · versión especial · servicio en vivo")
    ap.add_argument("--model", help="ruta al modelo (por defecto artifacts/model.json o heurística)")
    ap.add_argument("--heuristic", action="store_true", help="ignorar el modelo entrenado")
    ap.add_argument("--rebuild-history", action="store_true", help="recalcular el histórico (F5)")
    ap.add_argument("--standby", action="store_true", help="instancia de reserva: responde solo si el latido "
                                                          "de la principal lleva > 2 s parado")
    ap.add_argument("--cold-boot", action="store_true", help="arrancar sin contadores vivos ni estado del "
                                                            "bandit (ensayo del cambio de régimen, D)")
    ap.add_argument("--ab-share", type=float, help="fracción de tráfico con el modelo (resto: heurística)")
    ap.add_argument("--no-bandit", action="store_true")
    ap.add_argument("--diversify", choices=["never", "no_session", "always"])
    ap.add_argument("--ease-from-scratch", action="store_true",
                    help="si no hay caché local, leer los vecinos EASE de team.scratch")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
    cfg = Config.from_env(ab_treatment_share=args.ab_share, diversify=args.diversify,
                          bandit_enabled=False if args.no_bandit else None)
    if not cfg.team_id or not cfg.api_key:
        raise SystemExit("Faltan TEAM_ID / RAWTREE_API_KEY (.env o group_API.env).")
    from .rawtree_client import RawTreeClient

    client = RawTreeClient(cfg.api_url, cfg.api_key, cfg.team_id)
    history = get_or_build_history(client, cfg, rebuild=args.rebuild_history)
    if history.ease is None and args.ease_from_scratch:
        history.ease = load_ease_from_scratch(client, cfg.model_version)
    model = LinearModel.default() if args.heuristic else load_model(args.model or default_model_path(cfg))
    logger.info("Modelo: %s", getattr(model, "name", "?"))

    svc = Service(client, cfg, model, history, standby=args.standby, cold_boot=args.cold_boot)

    def on_signal(*_: Any) -> None:
        svc.stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    for attempt in range(1, 1000):           # F5: el arranque no puede morir por un fallo puntual
        try:
            svc.bootstrap()
            break
        except Exception as exc:  # noqa: BLE001
            logger.error("bootstrap intento %d: %s", attempt, exc)
            time.sleep(min(10, attempt))
    svc.run()


if __name__ == "__main__":
    main()
