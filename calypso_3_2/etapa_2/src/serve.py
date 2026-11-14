"""Production Fast/Slow Loop runner for Etapa 2 (Seq-Lift + UserProfile + Cold Start + MMR)."""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
from collections import deque
from datetime import datetime, timezone
import logging
import signal
import threading
import time
from typing import Any

from .cold_start import build_cold_start_scratch_records
from .config import ConfigV2
from .eval import generate_synthetic_dataset_v2
from .guardrails import validate_submission_record
from .rawtree_client import RawTreeClient
from .recommender import Stage2Recommender, build_snapshot_v2_from_records
from .state import StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("etapa2.serve")

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

SQL_POPULARITY_V2 = """
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

SQL_TRANSITIONS_V2 = """
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

SQL_USER_PROFILES = """
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

SQL_LIVE_TRENDS = """
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'purchase' AND ts::DateTime64(3) > now() - INTERVAL 15 MINUTE) AS buys_15m,
       countIf(event_type::String = 'view' AND ts::DateTime64(3) > now() - INTERVAL 15 MINUTE) AS views_15m,
       countIf(ts::DateTime64(3) <= now() - INTERVAL 15 MINUTE) / 12.0 AS avg_15m_prev,
       (buys_15m * 5.0 + views_15m) / greatest(avg_15m_prev, 1.0) AS live_momentum
FROM shop.events
WHERE ts::DateTime64(3) > now() - INTERVAL 195 MINUTE
GROUP BY product_id
HAVING (buys_15m + views_15m) >= 3
ORDER BY live_momentum DESC
LIMIT 100
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


def _merge_catalog_rows(
    base_rows: list[dict[str, Any]],
    update_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for r in base_rows:
        pid = str(r["product_id"])
        row_copy = dict(r)
        row_copy["initial_price_eur"] = float(r.get("price_eur", 0.0) or 0.0)
        row_copy["is_new_product"] = False
        merged[pid] = row_copy

    for u in update_rows or ():
        pid = str(u["product_id"])
        prev = merged.get(pid)
        row_copy = dict(u)
        row_copy["initial_price_eur"] = (
            prev["initial_price_eur"] if prev is not None else float(u.get("price_eur", 0.0) or 0.0)
        )
        merged[pid] = row_copy

    return list(merged.values())


class BackgroundUpdaterV2(threading.Thread):
    """Slow Loop: refreshes stock, live feedback purchases, trends, catalog & history matrices."""

    def __init__(
        self,
        client: RawTreeClient,
        store: StateStore,
        config: ConfigV2,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="BackgroundUpdaterV2", daemon=True)
        self.client = client
        self.store = store
        self.config = config
        self.stop_event = stop_event
        self._version = 0

    def refresh_full(self) -> None:
        t0 = time.perf_counter()
        timeout = self.config.query_timeout_slow_sec

        base_cat = self.client.query(SQL_CATALOG_BASE, timeout=timeout)
        upd_cat = self.client.query_safe(SQL_CATALOG_UPDATES, timeout=timeout)
        merged_cat = _merge_catalog_rows(base_cat, upd_cat)

        oos_rows = self.client.query_safe(SQL_OUT_OF_STOCK, timeout=5.0) or ()
        oos_ids = [str(r["product_id"]) for r in oos_rows]

        pop_rows = self.client.query(SQL_POPULARITY_V2, timeout=timeout)
        ctx_rows = self.client.query(SQL_CONTEXT_POP, timeout=timeout)
        trans_rows = self.client.query(SQL_TRANSITIONS_V2, timeout=timeout)
        prof_rows = self.client.query(SQL_USER_PROFILES, timeout=timeout)
        intent_rows = self.client.query(SQL_USER_INTENT, timeout=timeout)
        aff_rows = self.client.query(SQL_USER_AFFINITY, timeout=timeout)

        trend_rows = self.client.query_safe(SQL_LIVE_TRENDS, timeout=5.0) or ()
        live_bought_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=5.0) or ()

        self._version += 1
        snap = build_snapshot_v2_from_records(
            catalog_rows=merged_cat,
            popularity_rows=pop_rows,
            context_rows=ctx_rows,
            transition_rows=trans_rows,
            user_profile_rows=prof_rows,
            user_intent_rows=intent_rows,
            user_affinity_rows=aff_rows,
            out_of_stock_ids=oos_ids,
            live_trend_rows=trend_rows,
            live_bought_rows=live_bought_rows,
            lift_beta=self.config.lift_beta,
            lift_prior_pop=self.config.lift_prior_pop,
            version=self._version,
            max_cold_neighbors=self.config.max_cold_neighbors,
            max_reverse_cold_per_warm=self.config.max_reverse_cold_per_warm,
        )
        self.store.set(snap)

        # Persist cold-start neighbor mappings to team_NN.scratch for RawTree traceability
        try:
            now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            scratch_rows = build_cold_start_scratch_records(
                snap.cold_neighbors,
                snap.products,
                model_version=self.config.model_version,
                ts_iso=now_iso,
            )
            if scratch_rows:
                self.client.push_scratch(scratch_rows)
        except Exception as exc:
            logger.debug("Optional scratch write skipped: %s", exc)

        logger.info(
            "Snapshot v%d ready in %.2fs (%d active, %d transitions, %d user_profiles, %d reverse_cold)",
            snap.version,
            time.perf_counter() - t0,
            len(snap.active_ids),
            len(snap.transitions),
            len(snap.user_profiles),
            len(snap.warm_to_cold),
        )

    def refresh_fast_signals(self, include_trends: bool, include_feedback: bool) -> None:
        current = self.store.get()
        if current is None:
            return

        oos_rows = self.client.query_safe(SQL_OUT_OF_STOCK, timeout=3.0)
        new_oos = (
            frozenset(str(r["product_id"]) for r in oos_rows)
            if oos_rows is not None
            else current.out_of_stock
        )

        new_trends = current.live_trends
        if include_trends:
            trend_rows = self.client.query_safe(SQL_LIVE_TRENDS, timeout=3.0)
            if trend_rows is not None:
                raw_t = {str(r["product_id"]): float(r.get("live_momentum", 0.0) or 0.0) for r in trend_rows}
                max_t = max(raw_t.values(), default=1.0) or 1.0
                new_trends = {p: v / max_t for p, v in raw_t.items() if v > 0.0}

        new_live_bought = current.live_user_bought
        if include_feedback:
            lb_rows = self.client.query_safe(_sql_live_bought(self.config.team_id), timeout=3.0)
            if lb_rows is not None:
                new_live_bought = {
                    str(r["user_id"]): frozenset(str(p) for p in (r.get("live_bought_products") or ()))
                    for r in lb_rows
                }

        updated = current.with_fast_updates(
            out_of_stock=new_oos,
            live_trends=new_trends,
            live_user_bought=new_live_bought,
        )
        self._version = updated.version
        self.store.set(updated)

    def run(self) -> None:
        last_full = 0.0
        last_stock = 0.0
        last_trends = 0.0
        last_feedback = 0.0
        while not self.stop_event.is_set():
            now = time.time()
            try:
                if now - last_full >= self.config.history_refresh_sec or self.store.get() is None:
                    self.refresh_full()
                    last_full = last_stock = last_trends = last_feedback = time.time()
                elif now - last_stock >= self.config.stock_refresh_sec:
                    do_trends = (now - last_trends) >= self.config.trends_refresh_sec
                    do_fb = (now - last_feedback) >= self.config.feedback_refresh_sec
                    self.refresh_fast_signals(include_trends=do_trends, include_feedback=do_fb)
                    last_stock = time.time()
                    if do_trends:
                        last_trends = last_stock
                    if do_fb:
                        last_feedback = last_stock
            except Exception as exc:
                logger.warning("BackgroundUpdaterV2 error: %s", exc)
            self.stop_event.wait(1.0)


class FastServingLoopV2:
    """Fast Loop: pulls shop.requests, scores with Stage2Recommender, and batch-pushes."""

    def __init__(
        self,
        client: RawTreeClient | None,
        store: StateStore,
        recommender: Stage2Recommender,
        config: ConfigV2,
    ) -> None:
        self.client = client
        self.store = store
        self.recommender = recommender
        self.config = config
        self._seen_queue: deque[str] = deque(maxlen=config.seen_window_size)
        self._seen_set: set[str] = set()
        self._sql_pull = (
            f"SELECT request_id, ts, user_id, is_new_user, context, recent_views "
            f"FROM shop.requests ORDER BY ts DESC LIMIT {config.request_batch_limit}"
        )

    def _mark_unseen(self, requests_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fresh: list[dict[str, Any]] = []
        for req in requests_rows:
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

    def answer_batch(self, requests_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        snap = self.store.get()
        if snap is None:
            return []
        fresh = self._mark_unseen(requests_rows)
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
        rows = self.client.query_safe(self._sql_pull, timeout=self.config.query_timeout_fast_sec)
        if not rows:
            return 0
        answers = self.answer_batch(rows)
        if answers:
            self.client.push_submissions(answers, timeout=self.config.push_timeout_sec)
        return len(answers)


def run_dry_run(n_batches: int = 20, batch_size: int = 25) -> None:
    syn = generate_synthetic_dataset_v2(seed=7, n_test_sessions=n_batches * batch_size)
    store = StateStore(syn["snapshot_v2"])
    cfg = ConfigV2()
    loop = FastServingLoopV2(client=None, store=store, recommender=Stage2Recommender(cfg), config=cfg)

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
        f"[ETAPA 2 DRY-RUN OK] Answered {total_answered}/{len(rows)} requests across {len(latencies_ms)} batches "
        f"| mean batch compute: {avg_batch:.2f} ms ({avg_batch / batch_size:.3f} ms/req)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 2 Real-Time Recommender Service")
    parser.add_argument("--dry-run", action="store_true", help="Run local dry-run without API keys")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    cfg = ConfigV2.from_env()
    if not cfg.team_id or not cfg.api_key:
        raise SystemExit(
            "TEAM_ID i RAWTREE_API_KEY són obligatoris per executar en viu. "
            "Fes servir '--dry-run' per provar en local."
        )

    client = RawTreeClient(api_url=cfg.api_url, api_key=cfg.api_key, team_id=cfg.team_id)
    store = StateStore()
    stop_event = threading.Event()

    def _handle_sig(*_: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    updater = BackgroundUpdaterV2(client=client, store=store, config=cfg, stop_event=stop_event)
    updater.refresh_full()
    updater.start()

    loop = FastServingLoopV2(client=client, store=store, recommender=Stage2Recommender(cfg), config=cfg)
    logger.info("FastServingLoopV2 started (team=%s, model=%s)", cfg.team_id, cfg.model_version)

    while not stop_event.is_set():
        try:
            answered = loop.step()
            if answered:
                logger.info("Submitted %d answers", answered)
        except Exception as exc:
            logger.error("FastServingLoopV2 error: %s", exc)
            time.sleep(1.0)
        time.sleep(cfg.poll_interval_sec)


from .serve_e2esp import IncrementalReader, Service, SignalLogger, fmt_ts  # noqa: E402, F401


if __name__ == "__main__":
    main()
