"""Crash-hardened Fast/Live/Stock Stage 4 runner with DiskCache, watchdog, and post-push retries."""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
from collections import deque
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from .agent_scratch import CalypsoAgent
from .bandits import PositionAwareBandit
from .config import ConfigV4
from .disk_cache import DiskCache
from .eval import generate_synthetic_dataset_v4
from .guardrails import validate_submission_record
from .rawtree_client import RawTreeClient
from .recommender import (
    CalypsoRecommenderV4,
    build_product_static_table,
    build_snapshot_v4_from_records,
)
from .state import RecommenderSnapshotV4, StateStore, StaticHistoryBundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("calypso3_2.serve")


def write_heartbeat(heartbeat_path: Path, payload: dict[str, Any]) -> None:
    """Atomically write runtime heartbeat JSON so model_manager can probe readiness."""
    try:
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(heartbeat_path.parent),
            prefix=".heartbeat.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
            os.replace(tmp_name, heartbeat_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    except Exception:
        pass

SQL_CATALOG_BASE = """
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       price_eur::Float64 AS price_eur, description::String AS description,
       is_active::Bool AS is_active
FROM shop.catalog
"""

SQL_CATALOG_UPDATES = """
SELECT product_id::String AS product_id,
       argMax(category::String, ts::DateTime64(3)) AS category,
       argMax(subcategory::String, ts::DateTime64(3)) AS subcategory,
       argMax(brand::String, ts::DateTime64(3)) AS brand,
       argMax(price_eur::Float64, ts::DateTime64(3)) AS price_eur,
       argMax(description::String, ts::DateTime64(3)) AS description,
       argMax(is_active::Bool, ts::DateTime64(3)) AS is_active,
       max(if(change::String = 'new_product', 1, 0)) AS is_new_product
FROM shop.catalog_updates
GROUP BY product_id
"""

SQL_OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'
"""

SQL_POPULARITY = """
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'view') AS views,
       countIf(event_type::String = 'add_to_cart') AS carts,
       countIf(event_type::String = 'purchase') AS purchases,
       (purchases + 0.5) / (views + 25.0) AS self_cvr,
       (purchases * 5.0 + carts * 2.0 + views * 0.2) AS weighted_pop
FROM shop.history_events
GROUP BY product_id
ORDER BY weighted_pop DESC
"""

SQL_TRANSITIONS = """
SELECT a.product_id AS src_product,
       b.product_id AS dst_product,
       (countIf(b.event_type = 'purchase') * 6.0 +
        countIf(b.event_type = 'add_to_cart') * 2.5 +
        countIf(b.event_type = 'view') * 1.0) AS raw_transition_score
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id, ts::DateTime64(3) AS t
    FROM shop.history_events
    WHERE event_type::String = 'view'
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id, product_id::String AS product_id,
           event_type::String AS event_type, ts::DateTime64(3) AS t
    FROM shop.history_events
) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id AND b.t >= a.t
GROUP BY src_product, dst_product
ORDER BY raw_transition_score DESC
LIMIT 60 BY src_product
"""

SQL_USER_PROFILE = """
SELECT user_id::String AS user_id,
       groupUniqArrayIf(product_id::String, event_type::String = 'purchase') AS bought_products,
       avgIf(log1p(greatest(price_eur::Float64, 0.0)), event_type::String = 'purchase') AS log_price_mean,
       stddevPopIf(log1p(greatest(price_eur::Float64, 0.0)), event_type::String = 'purchase') AS log_price_std
FROM shop.history_events
GROUP BY user_id
"""

SQL_USER_INTENT = """
SELECT user_id::String AS user_id,
       product_id::String AS product_id,
       (countIf(event_type::String = 'add_to_cart') * 3.0 +
        countIf(event_type::String = 'view') * 0.5) *
       exp(-greatest(dateDiff('second', max(ts::DateTime64(3)),
           (SELECT max(ts::DateTime64(3)) FROM shop.history_events)) / 86400.0, 0.0) * 0.693 / 7.0) AS intent_score
FROM shop.history_events
GROUP BY user_id, product_id
HAVING countIf(event_type::String = 'purchase') = 0
   AND (countIf(event_type::String = 'add_to_cart') >= 1 OR countIf(event_type::String = 'view') >= 2)
ORDER BY intent_score DESC
LIMIT 10 BY user_id
"""

SQL_USER_AFFINITY = """
SELECT e.user_id::String AS user_id,
       c.subcategory::String AS subcategory,
       c.brand::String AS brand,
       (countIf(e.event_type::String = 'purchase') * 5.0 +
        countIf(e.event_type::String = 'add_to_cart') * 2.0 +
        countIf(e.event_type::String = 'view') * 0.3) AS affinity
FROM shop.history_events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
GROUP BY user_id, subcategory, brand
ORDER BY affinity DESC
LIMIT 8 BY user_id
"""

SQL_CONTEXT_POP = """
SELECT context.region::String AS region,
       context.device::String AS device,
       context.page::String AS page,
       purchased::String AS product_id,
       count() AS buys
FROM shop.history_sessions
WHERE purchased IS NOT NULL AND purchased::String != ''
GROUP BY region, device, page, product_id
ORDER BY buys DESC
LIMIT 35 BY region, device, page
"""

SQL_LIVE_TRENDS = """
SELECT product_id::String AS product_id,
       (countIf(event_type::String = 'purchase') * 5.0 +
        countIf(event_type::String = 'add_to_cart') * 2.0 +
        countIf(event_type::String = 'view') * 0.2) AS live_score
FROM shop.events
GROUP BY product_id
ORDER BY live_score DESC
LIMIT 200
"""

SQL_HISTORY_SESSIONS_TRAIN = """
SELECT request_id::String AS request_id,
       user_id::String AS user_id,
       false AS is_new_user,
       context.region::String AS region,
       context.device::String AS device,
       context.page::String AS page,
       recent_views,
       shown,
       ifNull(clicked::String, '') AS clicked,
       ifNull(purchased::String, '') AS purchased
FROM shop.history_sessions
WHERE length(shown) >= 10
LIMIT 3000
"""


def _sql_live_bought(team_id: str) -> str:
    return f"""
SELECT r.user_id::String AS user_id,
       groupUniqArray(f.purchased::String) AS live_bought_products
FROM {team_id}.feedback AS f
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
  AND f.purchased IS NOT NULL
  AND f.purchased::String != ''
GROUP BY user_id
"""


def _sql_feedback_with_shown(team_id: str) -> str:
    return f"""
SELECT f.request_id::String AS request_id,
       r.user_id::String AS user_id,
       r.is_new_user::Bool AS is_new_user,
       r.context.region::String AS region,
       r.context.device::String AS device,
       r.context.page::String AS page,
       r.recent_views AS recent_views,
       f.shown AS shown,
       f.clicked::String AS clicked,
       f.purchased::String AS purchased,
       f.revenue_eur::Float64 AS revenue_eur
FROM {team_id}.feedback AS f
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
ORDER BY f.ts DESC
LIMIT 3000
"""


def _sql_feedback_with_submissions(team_id: str) -> str:
    return f"""
SELECT f.request_id::String AS request_id,
       r.user_id::String AS user_id,
       r.is_new_user::Bool AS is_new_user,
       r.context.region::String AS region,
       r.context.device::String AS device,
       r.context.page::String AS page,
       r.recent_views AS recent_views,
       s.items AS shown,
       f.clicked::String AS clicked,
       f.purchased::String AS purchased,
       f.revenue_eur::Float64 AS revenue_eur
FROM {team_id}.feedback AS f
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
INNER JOIN {team_id}.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
ORDER BY f.ts DESC
LIMIT 3000
"""


def build_bootstrap_snapshot(
    client: RawTreeClient,
    config: ConfigV4,
    cache: DiskCache,
) -> RecommenderSnapshotV4:
    """Fastest possible initial snapshot: disk snapshot -> cached history -> 2s minimal SQL."""
    t0 = time.perf_counter()
    cached_snap = cache.load_snapshot()
    if cached_snap is not None and cached_snap.active_ids:
        logger.info(
            "[BOOT] Loaded cached snapshot v%d from disk in %.3fs (%d active items)",
            cached_snap.version,
            time.perf_counter() - t0,
            len(cached_snap.active_ids),
        )
        return cached_snap

    cached_hist = cache.load_static_history()
    if cached_hist is not None and cached_hist.catalog_rows:
        upd = client.query_safe(SQL_CATALOG_UPDATES, timeout=5.0) or []
        oos = client.query_safe(SQL_OUT_OF_STOCK, timeout=3.0) or []
        snap, _ = build_snapshot_v4_from_records(
            catalog_rows=cached_hist.catalog_rows,
            catalog_updates_rows=upd,
            popularity_rows=cached_hist.popularity_rows,
            context_rows=cached_hist.context_rows,
            transition_rows=cached_hist.transition_rows,
            user_profile_rows=cached_hist.user_profile_rows,
            user_intent_rows=cached_hist.user_intent_rows,
            user_affinity_rows=cached_hist.user_affinity_rows,
            out_of_stock_ids=[str(r["product_id"]) for r in oos],
            training_session_rows=cached_hist.training_session_rows,
            config=config,
            version=1,
        )
        cache.save_snapshot(snap)
        logger.info("[BOOT] Built snapshot from cached history in %.2fs", time.perf_counter() - t0)
        return snap

    base_cat = client.query(SQL_CATALOG_BASE, timeout=8.0)
    oos = client.query_safe(SQL_OUT_OF_STOCK, timeout=3.0) or []
    pop = client.query_safe(SQL_POPULARITY, timeout=8.0) or []
    snap, _ = build_snapshot_v4_from_records(
        catalog_rows=base_cat,
        popularity_rows=pop,
        out_of_stock_ids=[str(r["product_id"]) for r in oos],
        config=config,
        version=1,
    )
    logger.info(
        "[BOOT] Minimal 2s bootstrap ready in %.2fs (%d active items); background will load full history.",
        time.perf_counter() - t0,
        len(snap.active_ids),
    )
    return snap


class Stage4BackgroundUpdater(threading.Thread):
    """Separated 3-cadence updater: static history once/15m, live signals 15s, stock 1.5s."""

    def __init__(
        self,
        client: RawTreeClient,
        store: StateStore,
        config: ConfigV4,
        cache: DiskCache,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="Stage4BackgroundUpdater", daemon=True)
        self.client = client
        self.store = store
        self.config = config
        self.cache = cache
        self.stop_event = stop_event
        self.agent = CalypsoAgent()
        self.bandit = PositionAwareBandit(
            exam_power=config.bandit_exam_power,
            prior_strength=config.bandit_prior_strength,
            cold_ucb_multiplier=config.bandit_cold_ucb_multiplier,
        )
        self._static_bundle: StaticHistoryBundle | None = cache.load_static_history()
        cur = store.get()
        self._version = cur.version if cur is not None else 0

    def _fetch_static_history(self) -> StaticHistoryBundle:
        t0 = time.perf_counter()
        bundle = StaticHistoryBundle(
            fetched_at=time.time(),
            catalog_rows=self.client.query(SQL_CATALOG_BASE, timeout=18.0),
            popularity_rows=self.client.query(SQL_POPULARITY, timeout=18.0),
            transition_rows=self.client.query(SQL_TRANSITIONS, timeout=18.0),
            user_profile_rows=self.client.query(SQL_USER_PROFILE, timeout=18.0),
            user_intent_rows=self.client.query(SQL_USER_INTENT, timeout=18.0),
            user_affinity_rows=self.client.query(SQL_USER_AFFINITY, timeout=18.0),
            context_rows=self.client.query(SQL_CONTEXT_POP, timeout=18.0),
            training_session_rows=self.client.query_safe(SQL_HISTORY_SESSIONS_TRAIN, timeout=18.0) or [],
        )
        self._static_bundle = bundle
        self.cache.save_static_history(bundle)
        logger.info("Static 60-day history cached to disk in %.2fs", time.perf_counter() - t0)
        return bundle

    def _fetch_feedback_rows(self) -> list[dict[str, Any]]:
        fb_rows = self.client.query_safe(_sql_feedback_with_shown(self.config.team_id), timeout=8.0)
        if fb_rows is not None:
            return fb_rows
        fb_sub = self.client.query_safe(
            _sql_feedback_with_submissions(self.config.team_id), timeout=8.0
        )
        return fb_sub or []

    def refresh_live_and_model(self, force_static: bool = False) -> None:
        """Rebuild snapshot using cached static history + lightweight live SQL queries."""
        t0 = time.perf_counter()
        if force_static or self._static_bundle is None:
            bundle = self._fetch_static_history()
        else:
            bundle = self._static_bundle

        upd_cat = self.client.query_safe(SQL_CATALOG_UPDATES, timeout=8.0) or []
        oos_rows = self.client.query_safe(SQL_OUT_OF_STOCK, timeout=4.0) or []
        oos_ids = [str(r["product_id"]) for r in oos_rows]
        trend_rows = self.client.query_safe(SQL_LIVE_TRENDS, timeout=5.0) or []
        live_bought_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=5.0) or []
        feedback_rows = self._fetch_feedback_rows()

        self._version += 1
        snap, scratch_payload = build_snapshot_v4_from_records(
            catalog_rows=bundle.catalog_rows,
            catalog_updates_rows=upd_cat,
            popularity_rows=bundle.popularity_rows,
            context_rows=bundle.context_rows,
            transition_rows=bundle.transition_rows,
            user_profile_rows=bundle.user_profile_rows,
            user_intent_rows=bundle.user_intent_rows,
            user_affinity_rows=bundle.user_affinity_rows,
            out_of_stock_ids=oos_ids,
            live_trend_rows=trend_rows,
            live_bought_rows=live_bought_rows,
            feedback_rows=feedback_rows,
            training_session_rows=bundle.training_session_rows,
            config=self.config,
            version=self._version,
            agent=self.agent,
        )
        self.store.update(snap)
        self.cache.save_snapshot(snap)

        if scratch_payload:
            try:
                self.client.push_scratch(scratch_payload, timeout=4.0)
            except Exception as exc:
                logger.warning("Scratch telemetry push skipped: %s", exc)

        logger.info(
            "[%s] Live snapshot v%d ready & cached in %.2fs | active=%d | ml_alpha=%.3f | regime=%s | fb=%d",
            self.config.model_version,
            snap.version,
            time.perf_counter() - t0,
            len(snap.active_ids),
            snap.ranker.ml_blend_alpha if snap.ranker else 0.0,
            snap.agent_overrides.regime,
            len(feedback_rows),
        )

    def refresh_fast_stock_and_bought(self) -> None:
        """Lightweight 1.5s stock & live-bought refresh without disturbing heavy loops."""
        current = self.store.get()
        if current is None:
            return
        oos_rows = self.client.query_safe(SQL_OUT_OF_STOCK, timeout=2.5)
        live_bought_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=2.5)
        if oos_rows is None and live_bought_rows is None:
            return

        new_oos = (
            frozenset(str(r["product_id"]) for r in oos_rows)
            if oos_rows is not None
            else current.out_of_stock_ids
        )
        new_products = dict(current.products)
        stock_changed = new_oos != current.out_of_stock_ids
        if stock_changed:
            for pid, info in list(new_products.items()):
                active_now = info.is_active and (pid not in new_oos)
                if active_now != info.is_active:
                    new_products[pid] = replace(info, is_active=active_now)

        new_active = frozenset(pid for pid, info in new_products.items() if info.is_active)
        new_live_bought = (
            {
                str(r["user_id"]): frozenset(
                    str(p) for p in (r.get("live_bought_products") or ()) if p
                )
                for r in live_bought_rows
                if r.get("user_id")
            }
            if live_bought_rows is not None
            else current.live_user_bought
        )

        new_static = (
            build_product_static_table(
                new_products,
                new_active,
                current.global_pop_norm,
                current.live_trend_norm,
                current.max_log_price,
                self.bandit,
                current.bandit_state,
                current.agent_overrides.bandit_ucb_alpha,
            )
            if stock_changed
            else current.product_static
        )

        self._version += 1
        updated = RecommenderSnapshotV4(
            version=self._version,
            built_at=current.built_at,
            products=new_products,
            active_ids=new_active,
            out_of_stock_ids=new_oos,
            new_product_ids=frozenset(pid for pid in current.new_product_ids if pid in new_active),
            self_cvr=current.self_cvr,
            global_pop_norm=current.global_pop_norm,
            max_log_price=current.max_log_price,
            transitions=current.transitions,
            cold_neighbors=current.cold_neighbors,
            reverse_cold_neighbors=current.reverse_cold_neighbors,
            ctx_3=current.ctx_3,
            ctx_2=current.ctx_2,
            ctx_1=current.ctx_1,
            user_profiles=current.user_profiles,
            live_user_bought=new_live_bought,
            live_trend_norm=current.live_trend_norm,
            fallback_ranked=tuple(pid for pid in current.fallback_ranked if pid in new_active),
            revenue_fallback=tuple(pid for pid in current.revenue_fallback if pid in new_active),
            product_static=new_static,
            bandit_state=current.bandit_state,
            ranker=current.ranker,
            agent_overrides=current.agent_overrides,
        )
        self.store.update(updated)

    def run(self) -> None:
        last_static = self._static_bundle.fetched_at if self._static_bundle else 0.0
        last_live = 0.0
        last_stock = 0.0
        while not self.stop_event.is_set():
            now = time.time()
            try:
                need_static = (
                    self._static_bundle is None
                    or (now - last_static >= self.config.static_refresh_interval_s)
                )
                if need_static or (now - last_live >= self.config.live_refresh_interval_s):
                    self.refresh_live_and_model(force_static=need_static)
                    last_live = time.time()
                    if need_static:
                        last_static = last_live
                    last_stock = last_live
                elif now - last_stock >= self.config.stock_refresh_interval_s:
                    self.refresh_fast_stock_and_bought()
                    last_stock = time.time()
            except Exception as exc:
                logger.warning("Stage4BackgroundUpdater error: %s", exc)
            self.stop_event.wait(0.3)


class CalypsoFastLoopV4:
    """SLA-hardened FastLoop: pulls fresh requests only, confirms _seen only after push."""

    def __init__(
        self,
        client: RawTreeClient | None,
        store: StateStore,
        recommender: CalypsoRecommenderV4,
        config: ConfigV4,
    ) -> None:
        self.client = client
        self.store = store
        self.recommender = recommender
        self.config = config
        self._seen_queue: deque[str] = deque(maxlen=5000)
        self._seen_set: set[str] = set()
        self._sql_pull_fresh = (
            f"SELECT request_id, user_id, is_new_user, context, recent_views, ts "
            f"FROM shop.requests "
            f"WHERE ts::DateTime64(3) >= "
            f"(SELECT max(ts::DateTime64(3)) FROM shop.requests) "
            f"- INTERVAL {int(config.max_request_age_ms)} MILLISECOND "
            f"ORDER BY ts ASC LIMIT {int(config.max_requests_per_batch)}"
        )
        self._sql_pull_fallback = (
            f"SELECT request_id, user_id, is_new_user, context, recent_views, ts "
            f"FROM shop.requests ORDER BY ts DESC LIMIT {min(12, int(config.max_requests_per_batch))}"
        )

    def commit_seen(self, request_ids: list[str]) -> None:
        """Record request IDs as seen ONLY after submission succeeded."""
        for rid in request_ids:
            if rid in self._seen_set:
                continue
            if len(self._seen_queue) == self._seen_queue.maxlen:
                evicted = self._seen_queue.popleft()
                self._seen_set.discard(evicted)
            self._seen_queue.append(rid)
            self._seen_set.add(rid)

    def prepare_batch(self, request_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        """Score unseen requests and return (valid_answers, answered_request_ids)."""
        snap = self.store.get()
        if snap is None:
            return [], []

        fresh_rows = [
            req for req in request_rows if str(req.get("request_id") or "") not in self._seen_set
        ]
        if not fresh_rows:
            return [], []

        answers: list[dict[str, Any]] = []
        answered_ids: list[str] = []
        for req in fresh_rows:
            rid = str(req["request_id"])
            items = self.recommender.recommend(req, snap)
            record = {
                "request_id": rid,
                "items": items,
                "model_version": self.config.model_version,
            }
            if validate_submission_record(record, snap.active_ids):
                answers.append(record)
                answered_ids.append(rid)
        return answers, answered_ids

    def answer_batch(
        self,
        request_rows: list[dict[str, Any]],
        *,
        mark_seen: bool = True,
    ) -> list[dict[str, Any]]:
        """Score batch cleanly (optionally deferring commit_seen until push succeeds)."""
        answers, answered_ids = self.prepare_batch(request_rows)
        if mark_seen and answered_ids:
            self.commit_seen(answered_ids)
        return answers

    def step(self) -> int:
        assert self.client is not None
        rows = self.client.query_safe(self._sql_pull_fresh, timeout=0.9)
        if rows is None:
            rows = self.client.query_safe(self._sql_pull_fallback, timeout=0.9)
        if not rows:
            return 0
        answers, answered_ids = self.prepare_batch(rows)
        if not answers:
            return 0
        # Push FIRST, commit_seen SECOND so failed pushes retry automatically on next tick
        self.client.push_submissions(answers, timeout=0.9)
        self.commit_seen(answered_ids)
        return len(answers)


def run_dry_run(n_batches: int = 20, batch_size: int = 25) -> None:
    """Exercise DiskCache persistence, instant reload, Slot-Gated scoring, and post-push retry."""
    import tempfile

    syn = generate_synthetic_dataset_v4(seed=11, n_test_sessions=n_batches * batch_size)
    snap: RecommenderSnapshotV4 = syn["snap_v4"]
    bundle: StaticHistoryBundle = syn["static_bundle"]
    cfg = ConfigV4()

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = DiskCache(Path(tmp_dir))
        t_save = time.perf_counter()
        cache.save_static_history(bundle)
        cache.save_snapshot(snap)
        save_ms = (time.perf_counter() - t_save) * 1000.0

        t_load = time.perf_counter()
        reloaded_snap = cache.load_snapshot()
        load_ms = (time.perf_counter() - t_load) * 1000.0
        assert reloaded_snap is not None and len(reloaded_snap.active_ids) == len(snap.active_ids)

    store = StateStore(reloaded_snap)
    loop = CalypsoFastLoopV4(
        client=None,
        store=store,
        recommender=CalypsoRecommenderV4(cfg),
        config=cfg,
    )

    rows = syn["df_test"].to_dict(orient="records")
    total_answered = 0
    latencies_ms: list[float] = []

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        t0 = time.perf_counter()
        answers = loop.answer_batch(batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        total_answered += len(answers)

    avg_batch = sum(latencies_ms) / max(len(latencies_ms), 1)
    print(
        f"[DRY-RUN OK | Stage 4 · {cfg.model_version}] Answered {total_answered}/{len(rows)} reqs | "
        f"mean compute: {avg_batch:.2f} ms ({avg_batch / batch_size:.3f} ms/req) | "
        f"cache save={save_ms:.1f} ms / instant boot load={load_ms:.1f} ms"
    )


def run_self_check() -> None:
    """Fast pre-flight verification (<1.5s) used by model_manager before hot-swaps."""
    t0 = time.perf_counter()
    syn = generate_synthetic_dataset_v4(
        seed=7,
        n_warm_products=80,
        n_new_products=8,
        n_warm_users=60,
        n_history_sessions=160,
        n_feedback_sessions=40,
        n_test_sessions=10,
    )
    cfg = ConfigV4()
    snap: RecommenderSnapshotV4 = syn["snap_v4"]
    loop = CalypsoFastLoopV4(
        client=None,
        store=StateStore(snap),
        recommender=CalypsoRecommenderV4(cfg),
        config=cfg,
    )
    answers = loop.answer_batch(syn["df_test"].to_dict(orient="records"))
    if len(answers) != 10:
        raise RuntimeError(f"Expected 10 answers in self-check, got {len(answers)}")
    for ans in answers:
        if not validate_submission_record(ans, snap.active_ids):
            raise RuntimeError(f"Invalid submission record produced in self-check: {ans}")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[SELF-CHECK OK | {cfg.model_version}] Verified {len(answers)} recs in {elapsed_ms:.1f} ms")


def run_train_only(cfg: ConfigV4) -> None:
    """Train fresh snapshot (EASE, Cold-Start, Bandit, IPS ML Re-Ranker) to disk cache and exit."""
    if not cfg.team_id or not cfg.api_key:
        raise SystemExit("TEAM_ID i RAWTREE_API_KEY són obligatoris per entrenar des de RawTree.")
    t0 = time.perf_counter()
    client = RawTreeClient(api_url=cfg.api_url, api_key=cfg.api_key, team_id=cfg.team_id)
    cache = DiskCache(cfg.resolve_cache_dir())
    store = StateStore()
    updater = Stage4BackgroundUpdater(
        client=client,
        store=store,
        config=cfg,
        cache=cache,
        stop_event=threading.Event(),
    )
    updater.refresh_live_and_model(force_static=True)
    snap = store.get()
    if snap is None:
        raise RuntimeError("Training completed without producing a snapshot.")
    logger.info(
        "[TRAIN-ONLY OK | %s] Snapshot v%d trained in %.2fs | active=%d | ml_alpha=%.3f",
        cfg.model_version,
        snap.version,
        time.perf_counter() - t0,
        len(snap.active_ids),
        snap.ranker.ml_blend_alpha if snap.ranker else 0.0,
    )
    print(
        f"[TRAIN-ONLY OK | {cfg.model_version}] Trained & cached snapshot v{snap.version} "
        f"with {len(snap.active_ids)} active products"
    )


def run_watchdog(pass_through_args: list[str]) -> None:
    """Supervise child worker process and restart within 1s after any crash."""
    cmd = [sys.executable, str(Path(__file__).resolve()), *pass_through_args]
    logger.info("[WATCHDOG] Supervising worker: %s", " ".join(cmd))
    try:
        while True:
            proc = subprocess.run(cmd, check=False)
            logger.warning("[WATCHDOG] Worker exited with code %s. Restarting in 1s...", proc.returncode)
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("[WATCHDOG] Interrupted by user. Exiting cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calypso 3.2 Encapsulated Real-Time Recommender Service")
    parser.add_argument("--self-check", action="store_true", help="Run fast pre-flight verification and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run local dry-run with disk cache")
    parser.add_argument("--train-only", action="store_true", help="Train fresh snapshot + ML ranker + EASE to disk cache and exit")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore existing disk snapshot on boot and build fresh")
    parser.add_argument("--watchdog", action="store_true", help="Run auto-restarting supervisor loop")
    parser.add_argument("--model-version", type=str, default="", help="Override model_version label")
    parser.add_argument("--poll-interval", type=float, default=0.12, help="Fast-loop poll seconds")
    parser.add_argument("--stock-interval", type=float, default=1.5, help="Stock check seconds")
    parser.add_argument("--live-interval", type=float, default=15.0, help="Live feedback refit seconds")
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    if args.dry_run:
        run_dry_run()
        return

    if args.watchdog:
        child_args = [
            "--poll-interval",
            str(args.poll_interval),
            "--stock-interval",
            str(args.stock_interval),
            "--live-interval",
            str(args.live_interval),
        ]
        if args.model_version:
            child_args.extend(["--model-version", args.model_version])
        run_watchdog(child_args)
        return

    base_cfg = ConfigV4.from_env()
    cfg = replace(
        base_cfg,
        model_version=(args.model_version.strip() or base_cfg.model_version),
        poll_interval_s=args.poll_interval,
        stock_refresh_interval_s=args.stock_interval,
        live_refresh_interval_s=args.live_interval,
    )

    if args.train_only:
        run_train_only(cfg)
        return

    if not cfg.team_id or not cfg.api_key:
        raise SystemExit(
            "TEAM_ID i RAWTREE_API_KEY són obligatoris al .env o group_API.env. "
            "Fes servir '--dry-run' o '--self-check' per validar en local."
        )

    cache_dir = cfg.resolve_cache_dir()
    heartbeat_path = cache_dir / "heartbeat.json"
    started_at = time.time()
    pid = os.getpid()

    def _emit_hb(*, ready: bool, snap: RecommenderSnapshotV4 | None, submitted_total: int, last_batch: int) -> None:
        write_heartbeat(
            heartbeat_path,
            {
                "pid": pid,
                "model_name": "calypso 3.2",
                "model_version": cfg.model_version,
                "folder": "calypso_3_2",
                "started_at": started_at,
                "last_heartbeat_ts": time.time(),
                "ready": ready,
                "snapshot_version": snap.version if snap is not None else 0,
                "active_products": len(snap.active_ids) if snap is not None else 0,
                "ml_alpha": round(float(snap.ranker.ml_blend_alpha), 4) if (snap is not None and snap.ranker) else 0.0,
                "regime": snap.agent_overrides.regime if snap is not None else "booting",
                "submitted_total": submitted_total,
                "last_batch_size": last_batch,
            },
        )

    _emit_hb(ready=False, snap=None, submitted_total=0, last_batch=0)

    client = RawTreeClient(api_url=cfg.api_url, api_key=cfg.api_key, team_id=cfg.team_id)
    cache = DiskCache(cache_dir)
    if args.force_retrain and cache.snapshot_path.is_file():
        cache.snapshot_path.unlink(missing_ok=True)
    boot_snap = build_bootstrap_snapshot(client, cfg, cache)
    store = StateStore(boot_snap)
    stop_event = threading.Event()

    def _handle_sig(*_: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    updater = Stage4BackgroundUpdater(
        client=client,
        store=store,
        config=cfg,
        cache=cache,
        stop_event=stop_event,
    )
    updater.start()

    loop = CalypsoFastLoopV4(
        client=client,
        store=store,
        recommender=CalypsoRecommenderV4(cfg),
        config=cfg,
    )
    logger.info("Calypso 3.2 LIVE (team=%s, model=%s, pid=%d)", cfg.team_id, cfg.model_version, pid)

    submitted_total = 0
    last_batch_size = 0
    last_hb_written = 0.0

    while not stop_event.is_set():
        step_ok = False
        try:
            answered = loop.step()
            step_ok = True
            if answered:
                submitted_total += answered
                last_batch_size = answered
                logger.info("[%s] Submitted %d fresh answers", cfg.model_version, answered)
        except Exception as exc:
            logger.error("Calypso 3.2 FastLoop error: %s", exc)
            time.sleep(0.35)

        now_t = time.time()
        if step_ok and (last_hb_written == 0.0 or now_t - last_hb_written >= 1.0):
            _emit_hb(
                ready=True,
                snap=store.get(),
                submitted_total=submitted_total,
                last_batch=last_batch_size,
            )
            last_hb_written = now_t

        time.sleep(cfg.poll_interval_s)

    _emit_hb(ready=False, snap=store.get(), submitted_total=submitted_total, last_batch=last_batch_size)


if __name__ == "__main__":
    main()
