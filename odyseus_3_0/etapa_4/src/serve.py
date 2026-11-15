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
import zlib

from .agent_scratch import CalypsoAgent
from .bandits import PositionAwareBandit
from .business_rules import CartPurchaseIndex
from .config import ConfigV4
from .disk_cache import DiskCache
from .eval import generate_synthetic_dataset_v4
from .guardrails import validate_submission_record
from .ml_features import ML_FEATURE_NAMES
from .ml_model import MODEL_FILENAME, ModelHolder
from .rawtree_client import RawTreeClient
from .recommender import (
    CalypsoRecommenderV4,
    build_product_static_table,
    build_snapshot_v4_from_records,
)
from .state import RecommenderSnapshotV4, StateStore, StaticHistoryBundle
from .weights_profile import WeightsProfile, get_preset, load_active_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("odyseus2.serve")


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
        # 4.1: índice de carrito/recurrentes (histórico precalculado + capa en vivo)
        self._cart_index: CartPurchaseIndex | None = None
        self._live_cart_full: list[dict[str, Any]] = []
        self._fb_purchase_rows: list[dict[str, Any]] = []
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
        return self._cart_index

    def _cart_state(self, recent_rows: list[dict[str, Any]] | None = None) -> tuple[Any, Any, Any]:
        live_rows = _merge_live_cart_rows(self._live_cart_full, recent_rows or [])
        return self._get_cart_index().build(live_rows, self._fb_purchase_rows)

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
        trend_rows = self.client.query_safe(SQL_LIVE_TRENDS, timeout=5.0)
        if trend_rows is None:
            trend_rows = self.client.query_safe(SQL_LIVE_TRENDS_FALLBACK, timeout=5.0) or []
        live_bought_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=5.0) or []
        feedback_rows = self._fetch_feedback_rows()
        live_cart = self.client.query_safe(SQL_LIVE_CART_EVENTS, timeout=6.0)
        if live_cart is not None:
            self._live_cart_full = live_cart
        fb_purch = self.client.query_safe(_sql_feedback_purchases(self.config.team_id), timeout=5.0)
        if fb_purch is not None:
            self._fb_purchase_rows = fb_purch
        cart_state = self._cart_state()

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
            cart_state=cart_state,
            weights_profile=self._active_profile,
        )
        snap.config_override = self.config
        self.store.update(snap)
        self.cache.save_snapshot(snap)

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
        oos_rows = self.client.query_safe(SQL_OUT_OF_STOCK, timeout=2.5)
        live_bought_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=2.5)
        recent_cart_rows = self.client.query_safe(SQL_LIVE_CART_EVENTS_RECENT, timeout=2.0)
        if oos_rows is None and live_bought_rows is None and recent_cart_rows is None:
            return
        if recent_cart_rows is not None:
            new_carts, new_recurrent, live_bought_extra = self._cart_state(recent_cart_rows)
        else:
            new_carts, new_recurrent, live_bought_extra = current.user_carts, current.user_recurrent, {}

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
        )
        self.store.update(updated)

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


def request_arm(request_id: str, ab_share: float) -> str:
    """Rama estable por petición: 'B' para una fracción `ab_share` de los request_id (crc32), 'A' el resto."""
    return "B" if (zlib.crc32(request_id.encode("utf-8")) % 10000) < int(round(ab_share * 10000)) else "A"


class CalypsoFastLoopV4:
    """SLA-hardened FastLoop: pulls fresh requests only, confirms _seen only after push."""

    def __init__(
        self,
        client: RawTreeClient | None,
        store: StateStore,
        recommender: CalypsoRecommenderV4,
        config: ConfigV4,
        *,
        ab_share: float = 0.0,
        rebuy_in_b: bool = False,
    ) -> None:
        self.client = client
        self.store = store
        self.recommender = recommender
        self.config = config
        # Odyseus 2.0 · A/B por hash(request_id): rama A = Calypso 5.1, rama B = Calypso 5.1 + residuo ML.
        # ab_share = fracción de peticiones en la rama B (0 = sin reparto, comportamiento original).
        self.ab_share = min(max(float(ab_share), 0.0), 1.0)
        self.rebuy_in_b = bool(rebuy_in_b)  # rama B: no excluir lo comprado en el histórico (recompra)
        self._ml_holder = ModelHolder(Path(__file__).resolve().parents[2] / MODEL_FILENAME, ML_FEATURE_NAMES)
        self._ml_checked_at = 0.0
        self._ab_control_path = Path(__file__).resolve().parents[2] / "ab_control.json"
        self._ab_control_mtime: float | None = None
        self._ml_published: Any = None
        self._ml_model = self._ml_holder.refresh()
        self._seen_queue: deque[str] = deque(maxlen=5000)
        self._seen_set: set[str] = set()
        self._sql_pull_fresh = (
            f"SELECT request_id, user_id, is_new_user, context, recent_views "
            f"FROM shop.requests "
            f"WHERE ts::DateTime64(3) >= "
            f"(SELECT max(ts::DateTime64(3)) FROM shop.requests) "
            f"- INTERVAL {int(config.max_request_age_ms)} MILLISECOND "
            f"ORDER BY ts ASC LIMIT {int(config.max_requests_per_batch)}"
        )
        self._sql_pull_fallback = (
            f"SELECT request_id, user_id, is_new_user, context, recent_views "
            f"FROM shop.requests ORDER BY ts DESC LIMIT {min(12, int(config.max_requests_per_batch))}"
        )

    def _publish_model_async(self, model: Any) -> None:
        """Publica los pesos del modelo residual en team_NN.scratch (en un hilo: nunca frena el bucle)."""
        self._ml_published = model
        if self.client is None:
            return

        def _push() -> None:
            try:
                raw = self._ml_holder.path.read_text(encoding="utf-8")
                self.client.push_scratch(
                    [{
                        "type": "ml_residual_model",
                        "model_version": f"{self.config.model_version}-B",
                        "name": model.name,
                        "n_nets": len(model.nets),
                        "n_features": model.n_features,
                        "feature_names": model.feature_names,
                        "prior_scale": model.prior_scale,
                        "alpha": model.alpha,
                        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "model_json": raw,
                    }],
                    timeout=6.0,
                )
                logger.info("[ML] Pesos del modelo %s publicados en %s.scratch", model.name, self.config.team_id)
            except Exception as exc:
                logger.warning("[ML] No se pudieron publicar los pesos en scratch: %s", str(exc)[:160])

        threading.Thread(target=_push, name="ml-publish", daemon=True).start()

    def _reload_ab_control(self) -> None:
        """ab_control.json ({"ab_share": 0.5, "rebuy_in_b": true}) cambia el reparto sin reiniciar.

        Lo escribe ml_autopilot.py (o cualquiera a mano). Si falta o está mal, no cambia nada.
        """
        try:
            if not self._ab_control_path.is_file():
                return
            mtime = self._ab_control_path.stat().st_mtime
            if mtime == self._ab_control_mtime:
                return
            self._ab_control_mtime = mtime
            data = json.loads(self._ab_control_path.read_text(encoding="utf-8"))
            share = min(max(float(data.get("ab_share", self.ab_share)), 0.0), 1.0)
            rebuy = bool(data.get("rebuy_in_b", self.rebuy_in_b))
            if share != self.ab_share or rebuy != self.rebuy_in_b:
                logger.info("[A/B] ab_control.json: reparto B %.0f %% -> %.0f %% | recompra en B: %s",
                            100.0 * self.ab_share, 100.0 * share, "SÍ" if rebuy else "no")
            self.ab_share, self.rebuy_in_b = share, rebuy
        except Exception as exc:
            logger.warning("[A/B] ab_control.json ilegible (%s): se mantiene el reparto actual", exc)

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
        now = time.time()
        if now - self._ml_checked_at > 5.0:  # recarga en caliente de ml_model.json y ab_control.json
            self._ml_checked_at = now
            new_model = self._ml_holder.refresh()
            if new_model is not None and new_model is not self._ml_published:
                self._publish_model_async(new_model)
            self._ml_model = new_model
            self._reload_ab_control()
        ml_model = self._ml_model
        for req in fresh_rows:
            rid = str(req["request_id"])
            if self.ab_share > 0.0:
                arm = request_arm(rid, self.ab_share)
                version = f"{self.config.model_version}-{arm}"
            else:
                arm, version = "A", self.config.model_version
            items = self.recommender.recommend(
                req, snap,
                ml_model=ml_model if arm == "B" else None,
                allow_rebuy=self.rebuy_in_b and arm == "B",
            )
            record = {
                "request_id": rid,
                "items": items,
                "model_version": version,
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
    parser.add_argument(
        "--ab-share", type=float, default=float(os.environ.get("ODYSEUS_AB_SHARE", "0") or 0),
        help="Fracció de peticions a la branca B (Calypso 5.1 + residu ML). 0 = sense repartiment A/B",
    )
    parser.add_argument(
        "--rebuy-in-b", action="store_true",
        help="Branca B: no excloure els productes comprats a l'historic (recompra); requereix ml_model.json",
    )
    parser.add_argument("--poll-interval", type=float, default=0.12, help="Fast-loop poll seconds")
    parser.add_argument("--stock-interval", type=float, default=1.5, help="Stock check seconds")
    parser.add_argument("--live-interval", type=float, default=15.0, help="Live feedback refit seconds")
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
        if args.ab_share > 0.0:
            child_args.extend(["--ab-share", str(args.ab_share)])
        if args.rebuy_in_b:
            child_args.append("--rebuy-in-b")
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
                "model_name": "odyseus_3_0",
                "model_version": cfg.model_version,
                "folder": "odyseus_3_0",
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
        ab_share=args.ab_share,
        rebuy_in_b=args.rebuy_in_b,
    )
    if args.ab_share > 0.0:
        logger.info(
            "[A/B] %.0f %% de peticiones a la rama B (%s-B) | recompra en B: %s | modelo residual %s",
            100.0 * loop.ab_share,
            cfg.model_version,
            "SÍ" if loop.rebuy_in_b else "no",
            "CARGADO" if loop._ml_model is not None else "NO encontrado: A y B son idénticas hasta que exista ml_model.json",
        )
    logger.info(
        "Odyseus 2.0 (Calypso 5.1 + ML) LIVE (team=%s, model=%s, profile=%s, pid=%d)",
        cfg.team_id,
        cfg.model_version,
        active_wp.profile_name,
        pid,
    )

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
            logger.error("Calypso 5.1 FastLoop error: %s", exc)
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
