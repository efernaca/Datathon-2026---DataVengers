"""Production Fast/Slow Loop runner and --dry-run verifier for calypso-2.1."""
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
import logging
import signal
import threading
import time
from typing import Any

from .agent_scratch import CalypsoAgent
from .config import ConfigV3
from .eval import generate_synthetic_dataset_v3
from .guardrails import validate_submission_record
from .rawtree_client import RawTreeClient
from .recommender import CalypsoRecommender, build_snapshot_v3_from_records
from .state import RecommenderSnapshotV3, StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("etapa3.calypso")

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
       (countIf(event_type::String = 'purchase') * 5.0 +
        countIf(event_type::String = 'add_to_cart') * 2.0 +
        countIf(event_type::String = 'view') * 0.2) AS live_score
FROM shop.events
GROUP BY product_id
ORDER BY live_score DESC
LIMIT 200
"""

SQL_HISTORY_SESSIONS_TRAIN = """
SELECT session_id::String AS request_id,
       user_id::String AS user_id,
       is_new_user::Bool AS is_new_user,
       context.region::String AS region,
       context.device::String AS device,
       context.page::String AS page,
       recent_views,
       shown,
       clicked::String AS clicked,
       purchased::String AS purchased
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


class CalypsoBackgroundUpdater(threading.Thread):
    """Slow Loop: rebuilds ML ranker, bandits, cold-start graph, and CalypsoAgent state."""

    def __init__(
        self,
        client: RawTreeClient,
        store: StateStore,
        config: ConfigV3,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="CalypsoBackgroundUpdater", daemon=True)
        self.client = client
        self.store = store
        self.config = config
        self.stop_event = stop_event
        self.agent = CalypsoAgent()
        self._version = 0

    def _fetch_feedback_rows(self) -> list[dict[str, Any]]:
        fb_rows = self.client.query_safe(_sql_feedback_with_shown(self.config.team_id), timeout=8.0)
        if fb_rows is not None:
            return fb_rows
        fb_sub = self.client.query_safe(
            _sql_feedback_with_submissions(self.config.team_id), timeout=8.0
        )
        return fb_sub or []

    def refresh_full(self) -> None:
        """Query full catalog/history/feedback, refit ML & bandits, and push scratch telemetry."""
        t0 = time.perf_counter()
        base_cat = self.client.query(SQL_CATALOG_BASE, timeout=15.0)
        upd_cat = self.client.query_safe(SQL_CATALOG_UPDATES, timeout=10.0) or []
        oos_rows = self.client.query_safe(SQL_OUT_OF_STOCK, timeout=5.0) or []
        oos_ids = [str(r["product_id"]) for r in oos_rows]

        pop_rows = self.client.query(SQL_POPULARITY, timeout=15.0)
        trans_rows = self.client.query(SQL_TRANSITIONS, timeout=15.0)
        profile_rows = self.client.query(SQL_USER_PROFILE, timeout=15.0)
        intent_rows = self.client.query(SQL_USER_INTENT, timeout=15.0)
        affinity_rows = self.client.query(SQL_USER_AFFINITY, timeout=15.0)
        context_rows = self.client.query(SQL_CONTEXT_POP, timeout=15.0)
        trend_rows = self.client.query_safe(SQL_LIVE_TRENDS, timeout=5.0) or []
        live_bought_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=5.0) or []
        feedback_rows = self._fetch_feedback_rows()
        train_hist_rows = self.client.query_safe(SQL_HISTORY_SESSIONS_TRAIN, timeout=15.0) or []

        self._version += 1
        snap, scratch_payload = build_snapshot_v3_from_records(
            catalog_rows=base_cat,
            catalog_updates_rows=upd_cat,
            popularity_rows=pop_rows,
            context_rows=context_rows,
            transition_rows=trans_rows,
            user_profile_rows=profile_rows,
            user_intent_rows=intent_rows,
            user_affinity_rows=affinity_rows,
            out_of_stock_ids=oos_ids,
            live_trend_rows=trend_rows,
            live_bought_rows=live_bought_rows,
            feedback_rows=feedback_rows,
            training_session_rows=train_hist_rows,
            config=self.config,
            version=self._version,
            agent=self.agent,
        )
        self.store.update(snap)

        if scratch_payload:
            try:
                self.client.push_scratch(scratch_payload, timeout=5.0)
            except Exception as exc:
                logger.warning("Scratch telemetry push skipped: %s", exc)

        logger.info(
            "[%s] Snapshot v%d ready in %.2fs | active=%d | ml_alpha=%.3f | regime=%s | fb=%d",
            self.config.model_version,
            snap.version,
            time.perf_counter() - t0,
            len(snap.active_ids),
            snap.ranker.ml_blend_alpha if snap.ranker else 0.0,
            snap.agent_overrides.regime,
            len(feedback_rows),
        )

    def refresh_fast_stock_and_bought(self) -> None:
        """Refresh OOS and live bought exclusions between full ML/bandit refits."""
        current = self.store.get()
        if current is None:
            return

        oos_rows = self.client.query_safe(SQL_OUT_OF_STOCK, timeout=3.0)
        live_bought_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=3.0)
        if oos_rows is None and live_bought_rows is None:
            return

        new_oos = (
            frozenset(str(r["product_id"]) for r in oos_rows)
            if oos_rows is not None
            else current.out_of_stock_ids
        )
        new_products = dict(current.products)
        for pid, info in list(new_products.items()):
            active_now = info.is_active and (pid not in new_oos)
            if active_now != info.is_active or (pid not in new_oos and pid in current.out_of_stock_ids):
                new_products[pid] = replace(info, is_active=active_now)

        new_active = frozenset(pid for pid, info in new_products.items() if info.is_active)
        new_live_bought = dict(current.live_user_bought)
        if live_bought_rows is not None:
            new_live_bought = {
                str(r["user_id"]): frozenset(str(p) for p in (r.get("live_bought_products") or ()) if p)
                for r in live_bought_rows
                if r.get("user_id")
            }

        self._version += 1
        updated = RecommenderSnapshotV3(
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
            bandit_state=current.bandit_state,
            ranker=current.ranker,
            agent_overrides=current.agent_overrides,
        )
        self.store.update(updated)

    def run(self) -> None:
        last_full = time.time()
        last_stock = last_full
        while not self.stop_event.is_set():
            now = time.time()
            try:
                if now - last_full >= self.config.slow_interval_s:
                    self.refresh_full()
                    last_full = time.time()
                    last_stock = last_full
                elif now - last_stock >= 2.0:
                    self.refresh_fast_stock_and_bought()
                    last_stock = time.time()
            except Exception as exc:
                logger.warning("BackgroundUpdater error: %s", exc)
            self.stop_event.wait(0.5)


class CalypsoFastLoop:
    """Fast Loop: polls shop.requests, scores with calypso-2.1, validates & pushes answers."""

    def __init__(
        self,
        client: RawTreeClient | None,
        store: StateStore,
        recommender: CalypsoRecommender,
        config: ConfigV3,
    ) -> None:
        self.client = client
        self.store = store
        self.recommender = recommender
        self.config = config
        self._seen_queue: deque[str] = deque(maxlen=5000)
        self._seen_set: set[str] = set()
        self._sql_pull = (
            f"SELECT request_id, user_id, is_new_user, context, recent_views "
            f"FROM shop.requests ORDER BY ts DESC LIMIT {config.max_requests_per_batch}"
        )

    def _mark_unseen(self, request_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fresh: list[dict[str, Any]] = []
        for req in request_rows:
            rid = str(req["request_id"])
            if rid in self._seen_set:
                continue
            if len(self._seen_queue) == self._seen_queue.maxlen:
                evicted = self._seen_queue.popleft()
                self._seen_set.discard(evicted)
            self._seen_queue.append(rid)
            self._seen_set.add(rid)
            fresh.append(req)
        return fresh

    def answer_batch(self, request_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score a batch of requests in memory and validate every submission."""
        snap = self.store.get()
        if snap is None:
            return []
        fresh = self._mark_unseen(request_rows)
        if not fresh:
            return []

        answers: list[dict[str, Any]] = []
        for req in fresh:
            items = self.recommender.recommend(req, snap)
            record = {
                "request_id": str(req["request_id"]),
                "items": items,
                "model_version": self.config.model_version,
            }
            if validate_submission_record(record, snap.active_ids):
                answers.append(record)
        return answers

    def step(self) -> int:
        assert self.client is not None
        rows = self.client.query_safe(self._sql_pull, timeout=1.2)
        if not rows:
            return 0
        answers = self.answer_batch(rows)
        if answers:
            self.client.push_submissions(answers, timeout=1.4)
        return len(answers)


def run_dry_run(n_batches: int = 20, batch_size: int = 25) -> None:
    """Exercise full calypso-2.1 serving pipeline locally without network access."""
    syn = generate_synthetic_dataset_v3(seed=11, n_test_sessions=n_batches * batch_size)
    snap: RecommenderSnapshotV3 = syn["snap_v3"]
    store = StateStore(snap)
    cfg = ConfigV3()
    loop = CalypsoFastLoop(
        client=None,
        store=store,
        recommender=CalypsoRecommender(cfg),
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
        f"[DRY-RUN OK | {cfg.model_version}] Answered {total_answered}/{len(rows)} requests "
        f"| mean batch compute: {avg_batch:.2f} ms ({avg_batch / batch_size:.3f} ms/req) "
        f"| scratch_records={len(syn['scratch_payload'])} | regime={snap.agent_overrides.regime} "
        f"| ml_alpha={snap.ranker.ml_blend_alpha:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 (calypso-2.1) Real-Time Recommender Service")
    parser.add_argument("--dry-run", action="store_true", help="Run local simulation dry-run")
    parser.add_argument("--poll-interval", type=float, default=0.20, help="Fast-loop poll seconds")
    parser.add_argument("--slow-interval", type=float, default=18.0, help="Slow-loop refit seconds")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    base_cfg = ConfigV3.from_env()
    cfg = replace(
        base_cfg,
        poll_interval_s=args.poll_interval,
        slow_interval_s=args.slow_interval,
    )
    if not cfg.team_id or not cfg.api_key:
        raise SystemExit(
            "TEAM_ID i RAWTREE_API_KEY són obligatoris al .env o group_API.env. "
            "Fes servir '--dry-run' per validar en local."
        )

    client = RawTreeClient(api_url=cfg.api_url, api_key=cfg.api_key, team_id=cfg.team_id)
    store = StateStore()
    stop_event = threading.Event()

    def _handle_sig(*_: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    updater = CalypsoBackgroundUpdater(client=client, store=store, config=cfg, stop_event=stop_event)
    updater.refresh_full()
    updater.start()

    loop = CalypsoFastLoop(
        client=client,
        store=store,
        recommender=CalypsoRecommender(cfg),
        config=cfg,
    )
    logger.info("CalypsoFastLoop active (team=%s, model=%s)", cfg.team_id, cfg.model_version)

    while not stop_event.is_set():
        try:
            answered = loop.step()
            if answered:
                logger.info("[%s] Submitted %d answers", cfg.model_version, answered)
        except Exception as exc:
            logger.error("CalypsoFastLoop error: %s", exc)
            time.sleep(0.8)
        time.sleep(cfg.poll_interval_s)


if __name__ == "__main__":
    main()
