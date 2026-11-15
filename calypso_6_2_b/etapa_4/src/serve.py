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
from concurrent.futures import Future, ThreadPoolExecutor
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
from .business_rules import CartPurchaseIndex
from .config import ConfigV4
from .disk_cache import DiskCache
from .eval import generate_synthetic_dataset_v4
from .guardrails import validate_submission_record
from .rawtree_client import RawTreeClient
from .recommender import (
    CalypsoRecommenderV4,
    _extract_feature_matrix,
    build_product_static_table,
    build_snapshot_v4_from_records,
    compute_live_signals,
)
from .reranker import build_ips_training_dataset
from .state import RecommenderSnapshotV4, StateStore, StaticHistoryBundle
from .weights_profile import WeightsProfile, get_preset, load_active_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("calypso6_2b.serve")


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
            for attempt in range(5):
                try:
                    os.replace(tmp_name, heartbeat_path)
                    return
                except PermissionError:
                    if attempt < 4:
                        time.sleep(0.02 * (attempt + 1))
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
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

SQL_TRANSITIONS_DIRECTIONAL = """
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

# Version 2.1 (co-ocurrencia en sesion sin orden temporal, LIMIT 40)
SQL_TRANSITIONS_SYMMETRIC = """
SELECT a.product_id AS src_product,
       b.product_id AS dst_product,
       (countIf(b.event_type = 'purchase') * 6.0 +
        countIf(b.event_type = 'add_to_cart') * 2.5 +
        countIf(b.event_type = 'view') * 1.0) AS raw_transition_score
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type::String = 'view'
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id, product_id::String AS product_id, event_type::String AS event_type
    FROM shop.history_events
) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY src_product, dst_product
ORDER BY raw_transition_score DESC
LIMIT 40 BY src_product
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
        countIf(event_type::String = 'view') * 0.5) AS intent_score
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
       countIf(event_type::String = 'purchase') AS live_purchases,
       countIf(event_type::String = 'add_to_cart') AS live_carts,
       countIf(event_type::String = 'view') AS live_views,
       (countIf(event_type::String = 'purchase') * 5.0 +
        countIf(event_type::String = 'add_to_cart') * 2.0 +
        countIf(event_type::String = 'view') * 0.2) AS live_score
FROM shop.events
WHERE ts::DateTime64(3) >= (SELECT max(ts::DateTime64(3)) FROM shop.events) - INTERVAL 60 MINUTE
GROUP BY product_id
ORDER BY live_score DESC
LIMIT 400
"""

SQL_LIVE_TRENDS_FALLBACK = """
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'purchase') AS live_purchases,
       countIf(event_type::String = 'add_to_cart') AS live_carts,
       countIf(event_type::String = 'view') AS live_views,
       (countIf(event_type::String = 'purchase') * 5.0 +
        countIf(event_type::String = 'add_to_cart') * 2.0 +
        countIf(event_type::String = 'view') * 0.2) AS live_score
FROM shop.events
GROUP BY product_id
ORDER BY live_score DESC
LIMIT 400
"""

SQL_HISTORY_SESSIONS_TRAIN = """
SELECT request_id::String AS request_id,
       ts::String AS ts,
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
ORDER BY if(purchased IS NOT NULL AND purchased::String != '', 2, if(clicked IS NOT NULL AND clicked::String != '', 1, 0)) DESC
LIMIT 12000
"""


# ── 4.1 · carrito y compras repetidas ────────────────────────────────────────
# Carritos pendientes del histórico: último add_to_cart posterior a la última compra (máx. 6 por usuario)
SQL_CART_HIST = """
SELECT user_id::String AS user_id,
       product_id::String AS product_id,
       countIf(event_type::String = 'purchase') AS n_purchases,
       toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'add_to_cart')) AS last_cart_ms,
       toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'purchase')) AS last_purchase_ms
FROM shop.history_events
WHERE event_type::String IN ('add_to_cart', 'purchase')
GROUP BY user_id, product_id
HAVING last_cart_ms > last_purchase_ms
ORDER BY last_cart_ms DESC
LIMIT 6 BY user_id
"""

# Productos comprados >= 2 veces por el mismo usuario en el histórico (recurrentes)
SQL_RECURRENT_HIST = """
SELECT user_id::String AS user_id,
       product_id::String AS product_id,
       count() AS n_purchases,
       toUnixTimestamp64Milli(max(ts::DateTime64(3))) AS last_purchase_ms
FROM shop.history_events
WHERE event_type::String = 'purchase'
GROUP BY user_id, product_id
HAVING n_purchases >= 2
ORDER BY n_purchases DESC
LIMIT 10 BY user_id
"""

# Carrito y compras orgánicas en vivo (stream completo, cadencia 15 s)
SQL_LIVE_CART_EVENTS = """
SELECT user_id::String AS user_id,
       product_id::String AS product_id,
       countIf(event_type::String = 'purchase') AS n_purchases,
       toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'add_to_cart')) AS last_cart_ms,
       toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'purchase')) AS last_purchase_ms
FROM shop.events
WHERE event_type::String IN ('add_to_cart', 'purchase')
GROUP BY user_id, product_id
"""

# Lo mismo solo para los últimos 35 min (cadencia rápida 1,5 s: carrito recién añadido / recién comprado)
SQL_LIVE_CART_EVENTS_RECENT = """
SELECT user_id::String AS user_id,
       product_id::String AS product_id,
       countIf(event_type::String = 'purchase') AS n_purchases,
       toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'add_to_cart')) AS last_cart_ms,
       toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'purchase')) AS last_purchase_ms
FROM shop.events
WHERE event_type::String IN ('add_to_cart', 'purchase')
  AND ts::DateTime64(3) >= (SELECT max(ts::DateTime64(3)) FROM shop.events) - INTERVAL 35 MINUTE
GROUP BY user_id, product_id
"""


def _sql_feedback_purchases(team_id: str) -> str:
    """Compras en nuestro carrusel por (usuario, producto) con contador y última fecha."""
    return f"""
SELECT r.user_id::String AS user_id,
       f.purchased::String AS product_id,
       count() AS n_purchases,
       toUnixTimestamp64Milli(max(f.ts::DateTime64(3))) AS last_purchase_ms
FROM {team_id}.feedback AS f
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
  AND f.purchased IS NOT NULL
  AND f.purchased::String != ''
GROUP BY user_id, product_id
"""


def _merge_live_cart_rows(
    full_rows: list[dict[str, Any]],
    recent_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Superpone la ventana reciente (más fresca) sobre el stream completo (hasta 15 s de retraso)."""
    if not recent_rows:
        return list(full_rows)
    if not full_rows:
        return list(recent_rows)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in full_rows:
        merged[(str(row.get("user_id")), str(row.get("product_id")))] = dict(row)
    for row in recent_rows:
        key = (str(row.get("user_id")), str(row.get("product_id")))
        prev = merged.get(key)
        if prev is None:
            merged[key] = dict(row)
            continue
        for col in ("n_purchases", "last_cart_ms", "last_purchase_ms"):
            try:
                prev[col] = max(float(prev.get(col) or 0.0), float(row.get(col) or 0.0))
            except (TypeError, ValueError):
                pass
    return list(merged.values())


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
    *,
    weights_profile: WeightsProfile | None = None,
) -> RecommenderSnapshotV4:
    """Fastest possible initial snapshot: disk snapshot -> cached history -> 2s minimal SQL."""
    t0 = time.perf_counter()
    wp = weights_profile or config.load_weights_profile()
    cached_snap = cache.load_snapshot()
    if cached_snap is not None and cached_snap.active_ids:
        cached_snap.weights_profile = wp
        cached_snap.config_override = config
        logger.info(
            "[BOOT] Loaded cached snapshot v%d from disk in %.3fs (%d active items | profile=%s)",
            cached_snap.version,
            time.perf_counter() - t0,
            len(cached_snap.active_ids),
            wp.profile_name,
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
            cart_hist_rows=getattr(cached_hist, "cart_hist_rows", None) or [],
            recurrent_hist_rows=getattr(cached_hist, "recurrent_hist_rows", None) or [],
            weights_profile=wp,
        )
        snap.config_override = config
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
        weights_profile=wp,
    )
    snap.config_override = config
    logger.info(
        "[BOOT] Minimal 2s bootstrap ready in %.2fs (%d active items); background will load full history.",
        time.perf_counter() - t0,
        len(snap.active_ids),
    )
    return snap


class Stage4BackgroundUpdater(threading.Thread):
    """Separated 3-cadence updater with live hot-reload of weights_profile.json."""

    def __init__(
        self,
        client: RawTreeClient,
        store: StateStore,
        config: ConfigV4,
        cache: DiskCache,
        stop_event: threading.Event,
        *,
        weights_profile: WeightsProfile | None = None,
    ) -> None:
        super().__init__(name="Stage4BackgroundUpdater", daemon=True)
        self.client = client
        self.store = store
        self.config = config
        self.cache = cache
        self.stop_event = stop_event
        self.agent = CalypsoAgent()
        self._profile_path = config.resolve_weights_profile_path()
        try:
            self._profile_mtime: float | None = (
                self._profile_path.stat().st_mtime if self._profile_path.is_file() else None
            )
        except OSError:
            self._profile_mtime = None
        self._active_profile: WeightsProfile = (
            weights_profile if weights_profile is not None else config.load_weights_profile()
        )
        wp = self._active_profile
        self.bandit = PositionAwareBandit(
            exam_power=wp.bandit_exam_power,
            prior_strength=wp.bandit_prior_strength,
            cold_ucb_multiplier=wp.bandit_cold_ucb_multiplier,
            warm_ucb_multiplier=wp.bandit_warm_ucb_multiplier,
            click_reward=wp.bandit_click_reward,
            purchase_reward=wp.bandit_purchase_reward,
            use_empirical_exam_curve=wp.use_empirical_exam_curve,
        )
        self._static_bundle: StaticHistoryBundle | None = cache.load_static_history()
        # 4.1: índice de carrito/recurrentes (histórico precalculado + capa en vivo incremental)
        self._cart_index: CartPurchaseIndex | None = None
        self._live_cart_full: list[dict[str, Any]] = []
        self._latest_recent_cart: list[dict[str, Any]] = []
        self._fb_purchase_rows: list[dict[str, Any]] = []
        self._last_live_bought_extra: dict[str, frozenset[str]] = {}
        self._last_full_cart_ts: float = 0.0
        self._last_recent_cart_ts: float = 0.0
        self._last_cat_upd_ts: float = 0.0
        self._cached_upd_cat: list[dict[str, Any]] = []
        self._last_snapshot_disk_save: float = time.time()
        self._last_online_ft: float = 0.0
        cur = store.get()
        self._version = cur.version if cur is not None else 0

    def _fetch_static_history(self) -> StaticHistoryBundle:
        t0 = time.perf_counter()
        bundle = StaticHistoryBundle(
            fetched_at=time.time(),
            catalog_rows=self.client.query(SQL_CATALOG_BASE, timeout=18.0),
            popularity_rows=self.client.query(SQL_POPULARITY, timeout=18.0),
            transition_rows=self.client.query(
                SQL_TRANSITIONS_DIRECTIONAL if self.config.v3_directional_transitions else SQL_TRANSITIONS_SYMMETRIC,
                timeout=18.0,
            ),
            user_profile_rows=self.client.query(SQL_USER_PROFILE, timeout=18.0),
            user_intent_rows=self.client.query(SQL_USER_INTENT, timeout=18.0),
            user_affinity_rows=self.client.query(SQL_USER_AFFINITY, timeout=18.0),
            context_rows=self.client.query(SQL_CONTEXT_POP, timeout=18.0),
            training_session_rows=self.client.query_safe(SQL_HISTORY_SESSIONS_TRAIN, timeout=18.0) or [],
            cart_hist_rows=self.client.query_safe(SQL_CART_HIST, timeout=25.0) or [],
            recurrent_hist_rows=self.client.query_safe(SQL_RECURRENT_HIST, timeout=25.0) or [],
        )
        logger.info(
            "[4.1] Histórico carrito/recurrentes: %d carritos pendientes, %d pares recurrentes",
            len(bundle.cart_hist_rows),
            len(bundle.recurrent_hist_rows),
        )
        self._static_bundle = bundle
        self._cart_index = None  # se reconstruye con el nuevo histórico
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

    def _get_cart_index(self) -> CartPurchaseIndex:
        if self._cart_index is None:
            bundle = self._static_bundle
            self._cart_index = CartPurchaseIndex(
                getattr(bundle, "cart_hist_rows", None) or [] if bundle else [],
                getattr(bundle, "recurrent_hist_rows", None) or [] if bundle else [],
                min_recurrent=self.config.v4_recurrent_min_purchases,
            )
            if self._live_cart_full:
                self._cart_index.set_live_base(self._live_cart_full)
        return self._cart_index

    def _cart_state(self, recent_rows: list[dict[str, Any]] | None = None) -> tuple[Any, Any, Any]:
        idx = self._get_cart_index()
        delta = self._latest_recent_cart if recent_rows is None else recent_rows
        carts, rec, extra = idx.build_incremental(delta, self._fb_purchase_rows)
        self._last_live_bought_extra = extra
        return carts, rec, extra

    def _check_profile_reload(self) -> bool:
        """Reload weights_profile.json on disk modifications without stopping serving."""
        try:
            if not self._profile_path.is_file():
                return False
            cur_mtime = self._profile_path.stat().st_mtime
        except OSError:
            return False
        if self._profile_mtime is not None and abs(cur_mtime - self._profile_mtime) < 1e-6:
            return False
        try:
            new_wp = WeightsProfile.load_json(self._profile_path)
        except Exception as exc:
            logger.warning("[HOT-RELOAD] Failed reading %s: %s", self._profile_path, exc)
            return False

        self._profile_mtime = cur_mtime
        if new_wp == self._active_profile:
            return False
        old_name = self._active_profile.profile_name
        old_min_rec = self._active_profile.v4_recurrent_min_purchases
        self._active_profile = new_wp
        self.config = ConfigV4.from_profile(
            new_wp,
            api_url=self.config.api_url,
            team_id=self.config.team_id,
            api_key=self.config.api_key,
            model_version=self.config.model_version,
            poll_interval_s=self.config.poll_interval_s,
            stock_refresh_interval_s=self.config.stock_refresh_interval_s,
            live_refresh_interval_s=self.config.live_refresh_interval_s,
            weights_profile_path=self.config.weights_profile_path,
        )
        if new_wp.v4_recurrent_min_purchases != old_min_rec:
            self._cart_index = None
        self.bandit = PositionAwareBandit(
            exam_power=new_wp.bandit_exam_power,
            prior_strength=new_wp.bandit_prior_strength,
            cold_ucb_multiplier=new_wp.bandit_cold_ucb_multiplier,
            warm_ucb_multiplier=new_wp.bandit_warm_ucb_multiplier,
            click_reward=new_wp.bandit_click_reward,
            purchase_reward=new_wp.bandit_purchase_reward,
            use_empirical_exam_curve=new_wp.use_empirical_exam_curve,
        )
        logger.info(
            "[HOT-RELOAD] Reloaded profile '%s' -> '%s' from %s",
            old_name,
            new_wp.profile_name,
            self._profile_path.name,
        )
        return True

    def _save_snapshot_throttled(self, snap: RecommenderSnapshotV4, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_snapshot_disk_save) < 600.0:
            return
        try:
            if self.cache.save_snapshot(snap):
                self._last_snapshot_disk_save = now
        except Exception as exc:
            logger.warning("Snapshot disk cache skipped: %s", exc)

    def _refresh_live_incremental(
        self,
        current: RecommenderSnapshotV4,
        *,
        upd_cat: list[dict[str, Any]],
        oos_set: frozenset[str],
        trend_rows: list[dict[str, Any]],
        live_bought_rows: list[dict[str, Any]],
        feedback_rows: list[dict[str, Any]],
        cart_state: tuple[Any, Any, Any],
    ) -> RecommenderSnapshotV4:
        """Sub-second live update of trends, bandits, stock, carts & Ridge head without heavy EASE/SVD/MLP rebuild."""
        new_products = dict(current.products)
        new_product_ids = set(current.new_product_ids)
        for row in upd_cat:
            pid = str(row.get("product_id") or "").strip()
            if pid not in new_products:
                continue
            info = new_products[pid]
            raw_price = row.get("price_eur")
            price_eur = float(raw_price) if raw_price is not None else info.price_eur
            is_act = bool(row.get("is_active", info.is_active)) and (pid not in oos_set)
            is_new = info.is_new or bool(int(row.get("is_new_product") or 0))
            if is_new:
                new_product_ids.add(pid)
            new_products[pid] = replace(
                info,
                category=str(row.get("category") or info.category),
                subcategory=str(row.get("subcategory") or info.subcategory),
                brand=str(row.get("brand") or info.brand),
                price_eur=price_eur,
                description=str(row.get("description") or info.description),
                is_active=is_act,
                is_new=is_new,
            )
        for pid, info in list(new_products.items()):
            active_now = info.is_active and (pid not in oos_set)
            if active_now != info.is_active:
                new_products[pid] = replace(info, is_active=active_now)

        new_active = frozenset(pid for pid, info in new_products.items() if info.is_active)
        new_product_fset = frozenset(pid for pid in new_product_ids if pid in new_active)
        live_trend_norm, live_cvr_velocity, _ = compute_live_signals(trend_rows, new_active)

        new_live_bought: dict[str, frozenset[str]] = {
            str(r["user_id"]): frozenset(
                str(p) for p in (r.get("live_bought_products") or ()) if p
            )
            for r in live_bought_rows
            if r.get("user_id")
        }
        new_carts, new_recurrent, live_bought_extra = cart_state
        if self.config.v4_live_event_purchases and live_bought_extra:
            for uid, pids in live_bought_extra.items():
                new_live_bought[uid] = new_live_bought.get(uid, frozenset()) | pids

        if feedback_rows:
            bandit_state = self.bandit.fit_from_feedback(feedback_rows, fallback_cvr=current.self_cvr)
        else:
            bandit_state = current.bandit_state

        wp = self._active_profile
        self._version += 1
        updated = RecommenderSnapshotV4(
            version=self._version,
            built_at=time.time(),
            products=new_products,
            active_ids=new_active,
            out_of_stock_ids=oos_set,
            new_product_ids=new_product_fset,
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
            live_trend_norm=live_trend_norm,
            fallback_ranked=tuple(pid for pid in current.fallback_ranked if pid in new_active),
            revenue_fallback=tuple(pid for pid in current.revenue_fallback if pid in new_active),
            product_static=current.product_static,
            bandit_state=bandit_state,
            ranker=current.ranker,
            agent_overrides=current.agent_overrides,
            user_carts=new_carts,
            user_recurrent=new_recurrent,
            weights_profile=wp,
            config_override=self.config,
            ease_neighbors=current.ease_neighbors,
            hop2_transitions=current.hop2_transitions,
            covisit_neighbors=current.covisit_neighbors,
            svd_item_vectors=current.svd_item_vectors,
            svd_pid_to_idx=current.svd_pid_to_idx,
            subcat_top_items=current.subcat_top_items,
            cat_top_items=current.cat_top_items,
            subcat_price_stats=current.subcat_price_stats,
            item_subcat_pop=current.item_subcat_pop,
            item_cat_pop=current.item_cat_pop,
            subcat_transitions=current.subcat_transitions,
            live_cvr_velocity=live_cvr_velocity,
        )

        overrides, _, base_scratch = self.agent.diagnose_and_adapt(
            self.config,
            new_products,
            new_active,
            new_product_fset,
            feedback_rows,
            bandit_state,
            updated.ranker,
            weights_profile=wp,
        )
        updated.agent_overrides = overrides
        updated.product_static = build_product_static_table(
            new_products,
            new_active,
            updated.global_pop_norm,
            live_trend_norm,
            updated.max_log_price,
            self.bandit,
            bandit_state,
            overrides.bandit_ucb_alpha,
            item_subcat_pop=updated.item_subcat_pop,
            item_cat_pop=updated.item_cat_pop,
            self_cvr=updated.self_cvr,
            live_cvr_velocity=live_cvr_velocity,
        )

        now_ft = time.time()
        if (
            (now_ft - self._last_online_ft) >= 120.0
            and len(feedback_rows) >= 12
            and updated.ranker is not None
            and updated.ranker.nl_W1 is not None
        ):
            self.online_transfer_fine_tune(updated, feedback_rows[-60:])
            self._last_online_ft = now_ft

        self.store.update(updated)
        self._save_snapshot_throttled(updated)
        scratch_payload = self.agent.build_full_scratch_payload(
            self.config,
            self.bandit,
            bandit_state,
            updated.cold_neighbors,
            new_products,
            overrides,
            base_scratch,
        )
        if scratch_payload:
            try:
                self.client.push_scratch(scratch_payload, timeout=2.5)
            except Exception as exc:
                logger.warning("Scratch telemetry push skipped: %s", exc)
        return updated

    def refresh_live_and_model(self, force_static: bool = False, force_full: bool = False) -> None:
        """Rebuild snapshot using incremental live updates (<0.3s) or full static rebuild when needed."""
        t0 = time.perf_counter()
        if force_static or self._static_bundle is None:
            bundle = self._fetch_static_history()
        else:
            bundle = self._static_bundle

        now_t = time.time()
        if self._last_cat_upd_ts == 0.0 or now_t - self._last_cat_upd_ts >= 60.0:
            upd_cat_res = self.client.query_safe(SQL_CATALOG_UPDATES, timeout=5.0)
            if upd_cat_res is not None:
                self._cached_upd_cat = upd_cat_res
                self._last_cat_upd_ts = now_t
        upd_cat = self._cached_upd_cat
        oos_rows = self.client.query_safe(SQL_OUT_OF_STOCK, timeout=2.5) or []
        oos_ids = [str(r["product_id"]) for r in oos_rows]
        trend_rows = self.client.query_safe(SQL_LIVE_TRENDS, timeout=3.5)
        if trend_rows is None:
            trend_rows = self.client.query_safe(SQL_LIVE_TRENDS_FALLBACK, timeout=3.5) or []
        live_bought_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=3.5) or []
        feedback_rows = self._fetch_feedback_rows()
        if not self._live_cart_full or now_t - self._last_full_cart_ts >= 300.0:
            live_cart = self.client.query_safe(SQL_LIVE_CART_EVENTS, timeout=5.0)
            if live_cart is not None:
                self._live_cart_full = live_cart
                self._last_full_cart_ts = time.time()
                self._get_cart_index().set_live_base(live_cart)
        elif now_t - self._last_recent_cart_ts >= 5.0:
            recent_cart = self.client.query_safe(SQL_LIVE_CART_EVENTS_RECENT, timeout=2.5)
            if recent_cart is not None:
                self._latest_recent_cart = recent_cart
                self._last_recent_cart_ts = time.time()
        fb_purch = self.client.query_safe(_sql_feedback_purchases(self.config.team_id), timeout=3.5)
        if fb_purch is not None:
            self._fb_purchase_rows = fb_purch
        cart_state = self._cart_state()

        current = self.store.get()
        has_unseen_product = current is not None and any(
            str(r.get("product_id") or "").strip() not in current.products
            for r in upd_cat
            if r.get("product_id")
        )
        if (
            not force_static
            and not force_full
            and not has_unseen_product
            and current is not None
            and current.ranker is not None
            and current.ease_neighbors
        ):
            updated = self._refresh_live_incremental(
                current,
                upd_cat=upd_cat,
                oos_set=frozenset(oos_ids),
                trend_rows=trend_rows,
                live_bought_rows=live_bought_rows,
                feedback_rows=feedback_rows,
                cart_state=cart_state,
            )
            logger.info(
                "[%s | profile=%s] Live incremental v%d ready in %.2fs | active=%d | ml_alpha=%.3f | regime=%s | fb=%d",
                self.config.model_version,
                self._active_profile.profile_name,
                updated.version,
                time.perf_counter() - t0,
                len(updated.active_ids),
                updated.ranker.ml_blend_alpha if updated.ranker else 0.0,
                updated.agent_overrides.regime,
                len(feedback_rows),
            )
            return

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
            cart_hist_rows=bundle.cart_hist_rows,
            recurrent_hist_rows=bundle.recurrent_hist_rows,
            live_cart_event_rows=self._live_cart_full,
            feedback_purchase_rows=self._fb_purchase_rows,
            cart_state=cart_state,
            weights_profile=self._active_profile,
        )
        if len(feedback_rows) >= 12 and snap.ranker is not None and snap.ranker.nl_W1 is not None:
            self.online_transfer_fine_tune(snap, feedback_rows[-120:])

        snap.config_override = self.config
        self.store.update(snap)
        self._save_snapshot_throttled(snap, force=True)

        if scratch_payload:
            try:
                self.client.push_scratch(scratch_payload, timeout=4.0)
            except Exception as exc:
                logger.warning("Scratch telemetry push skipped: %s", exc)

        logger.info(
            "[%s | profile=%s] Live snapshot v%d ready & cached in %.2fs | active=%d | ml_alpha=%.3f | regime=%s | fb=%d"
            " | carts_users=%d | recurrent_users=%d",
            self.config.model_version,
            self._active_profile.profile_name,
            snap.version,
            time.perf_counter() - t0,
            len(snap.active_ids),
            snap.ranker.ml_blend_alpha if snap.ranker else 0.0,
            snap.agent_overrides.regime,
            len(feedback_rows),
            len(snap.user_carts),
            len(snap.user_recurrent),
        )

    def refresh_fast_stock_and_bought(self) -> None:
        """Lightweight 1.5s stock & live-bought refresh without disturbing heavy loops."""
        current = self.store.get()
        if current is None:
            return
        now_t = time.time()
        oos_rows = self.client.query_safe(SQL_OUT_OF_STOCK, timeout=1.8)
        live_bought_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=1.8)
        recent_cart_rows: list[dict[str, Any]] | None = None
        if self._last_recent_cart_ts == 0.0 or now_t - self._last_recent_cart_ts >= 5.0:
            recent_cart_rows = self.client.query_safe(SQL_LIVE_CART_EVENTS_RECENT, timeout=2.0)
            if recent_cart_rows is not None:
                self._latest_recent_cart = recent_cart_rows
                self._last_recent_cart_ts = time.time()
        if oos_rows is None and live_bought_rows is None and recent_cart_rows is None:
            return
        if recent_cart_rows is not None:
            new_carts, new_recurrent, live_bought_extra = self._cart_state(recent_cart_rows)
        else:
            new_carts, new_recurrent, live_bought_extra = (
                current.user_carts,
                current.user_recurrent,
                self._last_live_bought_extra,
            )

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
            else dict(current.live_user_bought)
        )
        if self.config.v4_live_event_purchases and live_bought_extra:
            for uid, pids in live_bought_extra.items():
                new_live_bought[uid] = new_live_bought.get(uid, frozenset()) | pids

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
                item_subcat_pop=current.item_subcat_pop,
                item_cat_pop=current.item_cat_pop,
                self_cvr=current.self_cvr,
                live_cvr_velocity=current.live_cvr_velocity,
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
            user_carts=new_carts,
            user_recurrent=new_recurrent,
            weights_profile=self._active_profile,
            config_override=self.config,
            ease_neighbors=current.ease_neighbors,
            hop2_transitions=current.hop2_transitions,
            covisit_neighbors=current.covisit_neighbors,
            svd_item_vectors=current.svd_item_vectors,
            svd_pid_to_idx=current.svd_pid_to_idx,
            subcat_top_items=current.subcat_top_items,
            cat_top_items=current.cat_top_items,
            subcat_price_stats=current.subcat_price_stats,
            item_subcat_pop=current.item_subcat_pop,
            item_cat_pop=current.item_cat_pop,
            subcat_transitions=current.subcat_transitions,
            live_cvr_velocity=current.live_cvr_velocity,
        )
        self.store.update(updated)

    def online_transfer_fine_tune(
        self,
        snap: RecommenderSnapshotV4,
        live_feedback_rows: list[dict[str, Any]],
    ) -> bool:
        """Fast (<20ms) online Ridge adaptation of the frozen-trunk MLP head from live feedback."""
        if snap.ranker is None or len(live_feedback_rows) < 12:
            return False
        wp = self._active_profile
        X_live, y_live, sw_live = build_ips_training_dataset(
            live_feedback_rows,
            lambda req, cands: _extract_feature_matrix(snap, req, cands, weights_profile=wp),
            exam_power=wp.bandit_exam_power,
            max_sessions=120,
            purchase_weight=wp.ips_purchase_weight,
            click_weight=wp.ips_click_weight,
            use_empirical_exam_curve=wp.use_empirical_exam_curve,
        )
        if X_live.shape[0] < 30 or int(y_live.sum()) < 2:
            return False
        snap.ranker.transfer_fine_tune(
            X_live,
            y_live,
            sw_live,
            transfer_prior_pull=wp.transfer_prior_pull,
            max_blend_alpha=wp.ml_max_blend_alpha,
            min_positives=2,
            nonlin_blend_alpha=wp.ml_nonlin_blend_alpha,
            nonlin_l2_alpha=max(wp.ml_nonlin_l2_alpha, 4.0),
            enable_nonlin=wp.enable_nonlin_mlp,
            ridge_only=True,
        )
        return True

    def run(self) -> None:
        last_static = self._static_bundle.fetched_at if self._static_bundle else 0.0
        last_live = 0.0
        last_stock = 0.0
        while not self.stop_event.is_set():
            now = time.time()
            try:
                profile_changed = self._check_profile_reload()
                need_static = (
                    self._static_bundle is None
                    or (now - last_static >= self.config.static_refresh_interval_s)
                )
                if profile_changed or need_static or (now - last_live >= self.config.live_refresh_interval_s):
                    self.refresh_live_and_model(force_static=need_static, force_full=profile_changed)
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
    """SLA-hardened FastLoop: pulls fresh requests only, async non-blocking push, confirms _seen after push."""

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
        self._state_lock = threading.Lock()
        self._seen_queue: deque[str] = deque(maxlen=5000)
        self._seen_set: set[str] = set()
        self._in_flight: set[str] = set()
        self._fresh_ok_once: bool = False
        self._push_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="calypso-push")
        self._pull_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="calypso-pull")
        self._pull_futures: deque[Future[list[dict[str, Any]]]] = deque()
        self._last_pull_dispatch: float = 0.0
        self._async_submitted: int = 0
        self._async_last_batch: int = 0
        self._sql_pull_fresh = (
            f"SELECT request_id, ts::String AS ts, user_id, is_new_user, context, recent_views "
            f"FROM shop.requests "
            f"WHERE ts::DateTime64(3) >= "
            f"(SELECT max(ts::DateTime64(3)) FROM shop.requests) "
            f"- INTERVAL {int(config.max_request_age_ms)} MILLISECOND "
            f"ORDER BY ts DESC LIMIT {int(config.max_requests_per_batch)}"
        )
        self._sql_pull_fallback = (
            f"SELECT request_id, ts::String AS ts, user_id, is_new_user, context, recent_views "
            f"FROM shop.requests ORDER BY ts DESC LIMIT {min(8, int(config.max_requests_per_batch))}"
        )

    def commit_seen(self, request_ids: list[str]) -> None:
        """Record request IDs as seen ONLY after submission succeeded."""
        with self._state_lock:
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

        with self._state_lock:
            skip_ids = self._seen_set | self._in_flight
        fresh_rows: list[dict[str, Any]] = []
        seen_in_batch: set[str] = set()
        for req in request_rows:
            rid = str(req.get("request_id") or "")
            if not rid or rid in skip_ids or rid in seen_in_batch:
                continue
            seen_in_batch.add(rid)
            fresh_rows.append(req)
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

    def _pull_requests(self) -> list[dict[str, Any]]:
        assert self.client is not None
        rows = self.client.query_safe(self._sql_pull_fresh, timeout=(0.9, 1.5))
        if rows is not None:
            self._fresh_ok_once = True
            return rows
        if not self._fresh_ok_once:
            fb_rows = self.client.query_safe(self._sql_pull_fallback, timeout=(0.9, 1.5))
            return fb_rows or []
        return []

    def step(self) -> int:
        rows = self._pull_requests()
        if not rows:
            return 0
        answers, answered_ids = self.prepare_batch(rows)
        if not answers:
            return 0
        # Push FIRST, commit_seen SECOND so failed pushes retry automatically on next tick
        self.client.push_submissions(answers, timeout=(0.9, 1.4))
        self.commit_seen(answered_ids)
        return len(answers)

    def _push_worker(self, answers: list[dict[str, Any]], answered_ids: list[str]) -> None:
        assert self.client is not None
        try:
            self.client.push_submissions(answers, timeout=(0.9, 1.4))
            with self._state_lock:
                for rid in answered_ids:
                    self._in_flight.discard(rid)
                    if rid not in self._seen_set:
                        if len(self._seen_queue) == self._seen_queue.maxlen:
                            evicted = self._seen_queue.popleft()
                            self._seen_set.discard(evicted)
                        self._seen_queue.append(rid)
                        self._seen_set.add(rid)
                n_sent = len(answers)
                self._async_submitted += n_sent
                self._async_last_batch = n_sent
            logger.info("[%s] Submitted %d fresh answers", self.config.model_version, n_sent)
        except Exception as exc:
            with self._state_lock:
                for rid in answered_ids:
                    self._in_flight.discard(rid)
            logger.warning("Calypso 6.2 async push failed (will retry): %s", exc)

    def step_async(self) -> tuple[int, int]:
        """Pull & score fresh requests without blocking on HTTP POST acknowledgements."""
        rows = self._pull_requests()
        if not rows:
            return 0, 0
        answers, answered_ids = self.prepare_batch(rows)
        if not answers:
            return len(rows), 0
        with self._state_lock:
            self._in_flight.update(answered_ids)
        self._push_pool.submit(self._push_worker, answers, answered_ids)
        return len(rows), len(answers)

    def step_pipelined(
        self,
        *,
        pull_interval_s: float = 0.20,
        max_in_flight_pulls: int = 3,
    ) -> tuple[int, int, bool]:
        """Overlap HTTP request pulls so ClickHouse RTT never stalls fresh arrivals."""
        now_t = time.monotonic()
        if len(self._pull_futures) < max_in_flight_pulls and (
            not self._pull_futures or (now_t - self._last_pull_dispatch) >= pull_interval_s
        ):
            self._pull_futures.append(self._pull_pool.submit(self._pull_requests))
            self._last_pull_dispatch = now_t

        completed_rows: list[dict[str, Any]] = []
        had_completed = False
        pending: deque[Future[list[dict[str, Any]]]] = deque()
        while self._pull_futures:
            fut = self._pull_futures.popleft()
            if fut.done():
                had_completed = True
                try:
                    rows = fut.result()
                    if rows:
                        completed_rows.extend(rows)
                except Exception as exc:
                    logger.warning("Calypso 6.2 pipelined pull error: %s", exc)
            else:
                pending.append(fut)
        self._pull_futures = pending

        if not completed_rows:
            return 0, 0, had_completed

        answers, answered_ids = self.prepare_batch(completed_rows)
        if not answers:
            return len(completed_rows), 0, True

        with self._state_lock:
            self._in_flight.update(answered_ids)
        self._push_pool.submit(self._push_worker, answers, answered_ids)
        return len(completed_rows), len(answers), True

    def drain_async_stats(self) -> tuple[int, int]:
        with self._state_lock:
            return self._async_submitted, self._async_last_batch

    def close(self) -> None:
        self._pull_pool.shutdown(wait=False)
        self._push_pool.shutdown(wait=False)


def _resolve_runtime_profile(preset: str = "", weights_profile: str = "") -> tuple[WeightsProfile, ConfigV4]:
    base_cfg = ConfigV4.from_env(weights_profile_path=weights_profile or None)
    if preset:
        wp = get_preset(preset)
        cfg = ConfigV4.from_profile(
            wp,
            api_url=base_cfg.api_url,
            team_id=base_cfg.team_id,
            api_key=base_cfg.api_key,
            model_version=base_cfg.model_version,
            weights_profile_path=weights_profile or base_cfg.weights_profile_path,
        )
        return wp, cfg
    wp = base_cfg.load_weights_profile()
    return wp, base_cfg


def run_dry_run(n_batches: int = 20, batch_size: int = 25, *, preset: str = "", weights_profile: str = "") -> None:
    """Exercise DiskCache persistence, instant reload, Slot-Gated scoring, and post-push retry."""
    import tempfile

    wp, cfg = _resolve_runtime_profile(preset=preset, weights_profile=weights_profile)
    syn = generate_synthetic_dataset_v4(
        seed=11,
        n_test_sessions=n_batches * batch_size,
        config_v4=cfg,
        weights_profile=wp,
    )
    snap: RecommenderSnapshotV4 = syn["snap_v4"]
    bundle: StaticHistoryBundle = syn["static_bundle"]

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
        f"[DRY-RUN OK | Stage 4 · {cfg.model_version} · profile={wp.profile_name}] Answered {total_answered}/{len(rows)} reqs | "
        f"mean compute: {avg_batch:.2f} ms ({avg_batch / batch_size:.3f} ms/req) | "
        f"cache save={save_ms:.1f} ms / instant boot load={load_ms:.1f} ms"
    )


def run_self_check(*, preset: str = "", weights_profile: str = "") -> None:
    """Fast pre-flight verification (<1.5s) used by model_manager before hot-swaps."""
    t0 = time.perf_counter()
    wp, cfg = _resolve_runtime_profile(preset=preset, weights_profile=weights_profile)
    syn = generate_synthetic_dataset_v4(
        seed=7,
        n_warm_products=80,
        n_new_products=8,
        n_warm_users=60,
        n_history_sessions=160,
        n_feedback_sessions=40,
        n_test_sessions=10,
        config_v4=cfg,
        weights_profile=wp,
    )
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
    print(f"[SELF-CHECK OK | {cfg.model_version} · profile={wp.profile_name}] Verified {len(answers)} recs in {elapsed_ms:.1f} ms")


def run_train_only(cfg: ConfigV4, *, weights_profile: WeightsProfile | None = None) -> None:
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
        weights_profile=weights_profile,
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
    parser = argparse.ArgumentParser(description="Calypso 5.1 High-Conversion Modular Real-Time Recommender Service")
    parser.add_argument("--self-check", action="store_true", help="Run fast pre-flight verification and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run local dry-run with disk cache")
    parser.add_argument("--train-only", action="store_true", help="Train fresh snapshot + ML ranker + EASE to disk cache and exit")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore existing disk snapshot on boot and build fresh")
    parser.add_argument("--watchdog", action="store_true", help="Run auto-restarting supervisor loop")
    parser.add_argument("--model-version", type=str, default="", help="Override model_version label")
    parser.add_argument("--preset", type=str, default="", help="Override WeightsProfile preset name")
    parser.add_argument("--weights-profile", type=str, default="", help="Path to custom weights_profile JSON")
    parser.add_argument("--poll-interval", type=float, default=0.06, help="Fast-loop poll seconds")
    parser.add_argument("--stock-interval", type=float, default=2.5, help="Stock check seconds")
    parser.add_argument("--live-interval", type=float, default=45.0, help="Live feedback refit seconds")
    args = parser.parse_args()

    if args.self_check:
        run_self_check(preset=args.preset, weights_profile=args.weights_profile)
        return

    if args.dry_run:
        run_dry_run(preset=args.preset, weights_profile=args.weights_profile)
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
        if args.preset:
            child_args.extend(["--preset", args.preset])
        if args.weights_profile:
            child_args.extend(["--weights-profile", args.weights_profile])
        run_watchdog(child_args)
        return

    active_wp, base_cfg = _resolve_runtime_profile(preset=args.preset, weights_profile=args.weights_profile)
    cfg = replace(
        base_cfg,
        model_version=(args.model_version.strip() or base_cfg.model_version),
        poll_interval_s=args.poll_interval,
        stock_refresh_interval_s=args.stock_interval,
        live_refresh_interval_s=args.live_interval,
    )

    if args.train_only:
        run_train_only(cfg, weights_profile=active_wp)
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
                "model_name": "calypso 6.2b",
                "model_version": cfg.model_version,
                "folder": "calypso_6_2_b",
                "profile": (
                    snap.weights_profile.profile_name
                    if (snap is not None and snap.weights_profile)
                    else active_wp.profile_name
                ),
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
    boot_snap = build_bootstrap_snapshot(client, cfg, cache, weights_profile=active_wp)
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
        weights_profile=active_wp,
    )
    updater.start()

    loop = CalypsoFastLoopV4(
        client=client,
        store=store,
        recommender=CalypsoRecommenderV4(cfg),
        config=cfg,
    )
    logger.info(
        "Calypso 6.2 LIVE (team=%s, model=%s, profile=%s, pid=%d)",
        cfg.team_id,
        cfg.model_version,
        active_wp.profile_name,
        pid,
    )

    last_hb_written = 0.0

    try:
        while not stop_event.is_set():
            step_ok = False
            dispatched = 0
            had_completed = False
            try:
                pulled, dispatched, had_completed = loop.step_pipelined()
                step_ok = True
            except Exception as exc:
                logger.error("Calypso 6.2 FastLoop error: %s", exc)
                time.sleep(0.25)

            submitted_total, last_batch_size = loop.drain_async_stats()
            now_t = time.time()
            if step_ok and (last_hb_written == 0.0 or now_t - last_hb_written >= 1.0):
                _emit_hb(
                    ready=True,
                    snap=store.get(),
                    submitted_total=submitted_total,
                    last_batch=last_batch_size,
                )
                last_hb_written = now_t

            time.sleep(0.015 if dispatched > 0 else (0.025 if not had_completed else cfg.poll_interval_s))
    finally:
        loop.close()

    submitted_total, last_batch_size = loop.drain_async_stats()
    _emit_hb(ready=False, snap=store.get(), submitted_total=submitted_total, last_batch=last_batch_size)


if __name__ == "__main__":
    main()
