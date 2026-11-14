"""Offline 5-way benchmark comparing Baseline, Stage 1, Stage 2, Stage 3, and Stage 4."""
from __future__ import annotations

import sys
from pathlib import Path

_ETAPA4_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _ETAPA4_ROOT.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

if __name__ == "__main__" and not __package__:
    if str(_ETAPA4_ROOT) not in sys.path:
        sys.path.insert(0, str(_ETAPA4_ROOT))
    __package__ = "src"
    __import__("src")

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from etapa_1.src.config import Config as ConfigV1
from etapa_1.src.recommender import Stage1Recommender, build_snapshot_from_records
from etapa_2.src.config import ConfigV2
from etapa_2.src.recommender import Stage2Recommender, build_snapshot_v2_from_records
from etapa_3.src.config import ConfigV3
from etapa_3.src.recommender import CalypsoRecommender, build_snapshot_v3_from_records

from .config import ConfigV4
from .guardrails import ensure_valid_submission
from .recommender import (
    CalypsoRecommenderV4,
    build_snapshot_v4_from_records,
    extract_recent_product_ids,
)
from .state import StaticHistoryBundle
from .weights_profile import WeightsProfile, get_preset

PolicyFn = Callable[[Any], list[str]]


def _is_nonempty(val: Any) -> bool:
    return val is not None and val != "" and not (isinstance(val, float) and np.isnan(val))


def evaluate_policy(
    policy_fn: PolicyFn,
    df: pd.DataFrame,
    cold_product_ids: frozenset[str] = frozenset(),
) -> dict[str, float | int]:
    """Compute HitRate@10, DCG@10, Cold-Item Recall@10, and IPS/SNIPS metrics."""
    n = len(df)
    if n == 0:
        return {
            "n_requests": 0,
            "hit_rate_click@10": 0.0,
            "hit_rate_buy@10": 0.0,
            "dcg_buy@10": 0.0,
            "cold_item_hit@10": 0.0,
            "slot1_matches": 0,
            "IPS_ctr_slot1": 0.0,
            "SNIPS_ctr_slot1": 0.0,
            "IPS_cvr_slot1": 0.0,
            "SNIPS_cvr_slot1": 0.0,
            "SNIPS_rev_slot1": 0.0,
            "SNIPS_cvr_se": 0.0,
            "latency_ms_mean": 0.0,
        }

    t0 = time.perf_counter()
    recs = [policy_fn(row) for row in df.itertuples(index=False)]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    clicked = np.array([str(x) if _is_nonempty(x) else "" for x in df["clicked"]], dtype=object)
    purchased = np.array([str(x) if _is_nonempty(x) else "" for x in df["purchased"]], dtype=object)
    revenue = np.array(
        [float(x) if _is_nonempty(x) else 0.0 for x in df.get("revenue_eur", np.zeros(n))],
        dtype=float,
    )

    has_click = clicked != ""
    has_buy = purchased != ""
    is_cold_buy = np.array([b in cold_product_ids for b in purchased], dtype=bool)

    hit_click_arr = np.array([c in r for c, r in zip(clicked, recs)], dtype=float)
    hit_buy_arr = np.array([b in r for b, r in zip(purchased, recs)], dtype=float)

    dcg_buy_arr = np.zeros(n, dtype=float)
    for i, (b, r) in enumerate(zip(purchased, recs)):
        if b and b in r:
            rank = r.index(b)
            dcg_buy_arr[i] = 1.0 / np.log2(rank + 2.0)

    first_rec = np.array([r[0] for r in recs], dtype=object)
    first_shown = np.array([s[0] if len(s) > 0 else "" for s in df["shown"]], dtype=object)
    prop1 = np.array(
        [max(float(p[0]), 1e-4) if len(p) > 0 else 1.0 for p in df["propensity"]],
        dtype=float,
    )

    match = (first_rec == first_shown).astype(float)
    w = match / prop1
    w_sum = float(w.sum())

    r_click = (clicked == first_shown).astype(float)
    r_buy = (purchased == first_shown).astype(float)
    r_rev = revenue * r_buy

    snips_cvr = float((w * r_buy).sum() / max(w_sum, 1e-9))
    snips_cvr_se = (
        float(np.sqrt(np.sum((w * (r_buy - snips_cvr)) ** 2)) / max(w_sum, 1e-9))
        if w_sum > 0
        else 0.0
    )

    return {
        "n_requests": int(n),
        "hit_rate_click@10": float(hit_click_arr[has_click].mean()) if has_click.any() else 0.0,
        "hit_rate_buy@10": float(hit_buy_arr[has_buy].mean()) if has_buy.any() else 0.0,
        "dcg_buy@10": float(dcg_buy_arr[has_buy].mean()) if has_buy.any() else 0.0,
        "cold_item_hit@10": float(hit_buy_arr[is_cold_buy].mean()) if is_cold_buy.any() else 0.0,
        "slot1_matches": int(match.sum()),
        "IPS_ctr_slot1": float((w * r_click).mean()),
        "SNIPS_ctr_slot1": float((w * r_click).sum() / max(w_sum, 1e-9)),
        "IPS_cvr_slot1": float((w * r_buy).mean()),
        "SNIPS_cvr_slot1": snips_cvr,
        "SNIPS_rev_slot1": float((w * r_rev).sum() / max(w_sum, 1e-9)),
        "SNIPS_cvr_se": snips_cvr_se,
        "latency_ms_mean": float(elapsed_ms / n),
    }


def is_cold_start_row(
    row: Any,
    cold_product_ids: frozenset[str],
    warm_user_ids: frozenset[str] | None = None,
) -> bool:
    """Match challenge Cold Start slicing (new user OR new product in recent_views / purchased)."""
    if isinstance(row, Mapping):
        is_new_user = bool(row.get("is_new_user", False))
        uid = str(row.get("user_id", "") or "")
        recent = row.get("recent_views")
        purchased = str(row.get("purchased") or "")
    else:
        is_new_user = bool(getattr(row, "is_new_user", False))
        uid = str(getattr(row, "user_id", "") or "")
        recent = getattr(row, "recent_views", None)
        purchased = str(getattr(row, "purchased", "") or "")

    if is_new_user:
        return True
    if warm_user_ids is not None and uid not in warm_user_ids:
        return True
    if purchased and purchased in cold_product_ids:
        return True
    for pid in extract_recent_product_ids(recent):
        if pid in cold_product_ids:
            return True
    return False


def evaluate_stratified(
    policies: Mapping[str, PolicyFn],
    df_test: pd.DataFrame,
    cold_product_ids: frozenset[str] = frozenset(),
    warm_user_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Evaluate policies across ALL, WARM, and COLD_START slices."""
    cold_mask = np.array(
        [
            is_cold_start_row(row, cold_product_ids, warm_user_ids)
            for row in df_test.itertuples(index=False)
        ],
        dtype=bool,
    )
    segments = {
        "ALL": df_test,
        "WARM": df_test[~cold_mask],
        "COLD_START": df_test[cold_mask],
    }

    rows: list[dict[str, Any]] = []
    for seg_name, seg_df in segments.items():
        for pol_name, pol_fn in policies.items():
            metrics = evaluate_policy(pol_fn, seg_df, cold_product_ids=cold_product_ids)
            rows.append({"segment": seg_name, "policy": pol_name, **metrics})

    return pd.DataFrame(rows).set_index(["segment", "policy"])


def generate_synthetic_dataset_v4(
    seed: int = 42,
    n_warm_products: int = 500,
    n_new_products: int = 45,
    n_warm_users: int = 1200,
    n_history_sessions: int = 4500,
    n_feedback_sessions: int = 1200,
    n_test_sessions: int = 2400,
    config_v4: ConfigV4 | None = None,
    weights_profile: WeightsProfile | None = None,
) -> dict[str, Any]:
    """Generate store records + IPS logs and build V1, V2, V3, V4 snapshots identically."""
    rng = np.random.default_rng(seed)

    subcat_specs = [
        ("Electronics", "Wireless Audio", ["Sonic", "Aura", "Pulse"], "wireless bluetooth headphones anc"),
        ("Electronics", "Smartwatches", ["Chrono", "FitPro", "Volt"], "smartwatch gps fitness tracker amoled"),
        ("Home", "Coffee Makers", ["Barista", "Crema", "Roast"], "espresso coffee maker grinder stainless"),
        ("Home", "Air Purifiers", ["PureAir", "Breeze", "Ozon"], "hepa air purifier quiet bedroom filter"),
        ("Sports", "Running Shoes", ["Stride", "Pace", "Aero"], "running trail shoes cushioned waterproof"),
        ("Sports", "Yoga Mats", ["ZenFlex", "Core", "Flow"], "non slip yoga mat eco cork fitness"),
        ("Books", "Data Science", ["O'Press", "TechPub", "Axiom"], "machine learning python algorithms textbook"),
        ("Fashion", "Backpacks", ["Urban", "Nomad", "Peak"], "laptop waterproof backpack travel commuter"),
    ]

    catalog_rows: list[dict[str, Any]] = []
    subcat_to_pids: dict[str, list[str]] = defaultdict(list)
    pid_to_subcat: dict[str, str] = {}
    pid_price: dict[str, float] = {}
    pid_brand: dict[str, str] = {}
    cold_product_ids: set[str] = set()

    total_products = n_warm_products + n_new_products
    for idx in range(total_products):
        pid = f"p_{idx:04d}"
        cat, subcat, brands, keywords = subcat_specs[idx % len(subcat_specs)]
        brand = brands[idx % len(brands)]
        price = float(np.round(rng.uniform(20.0, 240.0), 2))
        is_new = idx >= n_warm_products
        if is_new:
            cold_product_ids.add(pid)
        catalog_rows.append(
            {
                "product_id": pid,
                "category": cat,
                "subcategory": subcat,
                "brand": brand,
                "price_eur": price,
                "initial_price_eur": price,
                "description": f"{brand} {keywords} series {idx % 15} premium quality",
                "is_active": True,
                "is_new_product": is_new,
            }
        )
        subcat_to_pids[subcat].append(pid)
        pid_to_subcat[pid] = subcat
        pid_price[pid] = price
        pid_brand[pid] = brand

    warm_pids = [f"p_{i:04d}" for i in range(n_warm_products)]
    cold_pids = [f"p_{i:04d}" for i in range(n_warm_products, total_products)]
    warm_subcat_pids = {
        sc: [p for p in pids if p not in cold_product_ids] for sc, pids in subcat_to_pids.items()
    }
    cold_subcat_pids = {
        sc: [p for p in pids if p in cold_product_ids] for sc, pids in subcat_to_pids.items()
    }

    latent_next: dict[str, list[str]] = {}
    for sc, pids in subcat_to_pids.items():
        warm_sc = warm_subcat_pids[sc]
        for pid in pids:
            same_brand = [q for q in warm_sc if q != pid and pid_brand[q] == pid_brand[pid]]
            others = [q for q in warm_sc if q != pid]
            pool = same_brand if len(same_brand) >= 3 else others
            latent_next[pid] = list(rng.choice(pool, size=min(4, len(pool)), replace=False))

    cold_winners: set[str] = set(cold_pids[: max(1, len(cold_pids) // 2)])

    warm_users = [f"u_{i:04d}" for i in range(n_warm_users)]
    user_fav_subcat = {u: subcat_specs[i % len(subcat_specs)][1] for i, u in enumerate(warm_users)}

    user_bought_map: dict[str, set[str]] = defaultdict(set)
    user_cart_intent: dict[str, list[str]] = defaultdict(list)
    user_log_prices: dict[str, list[float]] = defaultdict(list)
    user_affinity_counts: dict[tuple[str, str, str], float] = defaultdict(float)

    pop_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"views": 0, "carts": 0, "purchases": 0}
    )
    trans_counts: dict[tuple[str, str], float] = defaultdict(float)
    context_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)

    regions = ["Catalunya", "Madrid", "Andalucia", "Valencia"]
    devices = ["web", "ios", "android"]
    pages = ["home", "product", "category"]

    ctx_heroes: dict[tuple[str, str, str], list[str]] = {}
    for r_i, reg in enumerate(regions):
        for d_i, dev in enumerate(devices):
            for p_i, pg in enumerate(pages):
                sc = subcat_specs[(r_i + d_i + p_i) % len(subcat_specs)][1]
                ctx_heroes[(reg, dev, pg)] = warm_subcat_pids[sc][:4]

    for _ in range(n_history_sessions):
        uid = warm_users[int(rng.integers(0, len(warm_users)))]
        fav_sc = user_fav_subcat[uid]
        reg = regions[int(rng.integers(0, len(regions)))]
        dev = devices[int(rng.integers(0, len(devices)))]
        pg = pages[int(rng.integers(0, len(pages)))]

        sc_pool = warm_subcat_pids[fav_sc]
        v1 = sc_pool[int(rng.integers(0, len(sc_pool)))]
        v2 = latent_next[v1][int(rng.integers(0, len(latent_next[v1])))]
        target = v2 if rng.random() < 0.72 else ctx_heroes[(reg, dev, pg)][0]

        pop_counts[v1]["views"] += 1
        pop_counts[v2]["views"] += 1
        trans_counts[(v1, v2)] += 1.0
        user_affinity_counts[(uid, pid_to_subcat[v1], pid_brand[v1])] += 0.6

        if rng.random() < 0.45:
            pop_counts[target]["carts"] += 1
            trans_counts[(v1, target)] += 2.5
            if target not in user_bought_map[uid] and len(user_cart_intent[uid]) < 3:
                user_cart_intent[uid].append(target)

        if rng.random() < 0.35:
            pop_counts[target]["purchases"] += 1
            trans_counts[(v1, target)] += 5.0
            context_counts[(reg, dev, pg, target)] += 1
            user_bought_map[uid].add(target)
            user_log_prices[uid].append(math.log1p(pid_price[target]))
            user_affinity_counts[(uid, pid_to_subcat[target], pid_brand[target])] += 5.0

    popularity_rows = []
    for pid in warm_pids:
        c = pop_counts[pid]
        wpop = c["purchases"] * 5.0 + c["carts"] * 2.0 + c["views"] * 0.2
        popularity_rows.append(
            {
                "product_id": pid,
                "views": c["views"],
                "carts": c["carts"],
                "purchases": c["purchases"],
                "self_cvr": (c["purchases"] + 0.5) / (c["views"] + 25.0),
                "weighted_pop": wpop,
            }
        )

    popularity_ranked = sorted(
        popularity_rows, key=lambda r: (r["weighted_pop"], r["product_id"]), reverse=True
    )
    baseline_top10 = [r["product_id"] for r in popularity_ranked[:10]]

    trans_rows_v1 = [
        {"src_product": s, "dst_product": d, "score": sc}
        for (s, d), sc in trans_counts.items()
    ]
    trans_rows_v2 = [
        {"src_product": s, "dst_product": d, "raw_transition_score": sc}
        for (s, d), sc in trans_counts.items()
    ]
    context_rows = [
        {"region": r, "device": d, "page": p, "product_id": pid, "buys": b}
        for (r, d, p, pid), b in context_counts.items()
    ]
    user_bought_rows = [
        {"user_id": uid, "bought_products": sorted(pids)}
        for uid, pids in user_bought_map.items()
        if pids
    ]
    user_profile_rows = []
    for uid in warm_users:
        buys = sorted(user_bought_map.get(uid, ()))
        lps = user_log_prices.get(uid, [])
        user_profile_rows.append(
            {
                "user_id": uid,
                "bought_products": buys,
                "log_price_mean": float(np.mean(lps)) if lps else 0.0,
                "log_price_std": float(np.std(lps)) if lps else 0.0,
            }
        )

    user_intent_rows = []
    for uid, pids in user_cart_intent.items():
        for rank, pid in enumerate(pids):
            if pid not in user_bought_map[uid]:
                user_intent_rows.append({"user_id": uid, "product_id": pid, "intent_score": 4.0 - rank})

    user_affinity_rows = [
        {"user_id": uid, "subcategory": sc, "brand": br, "affinity": aff}
        for (uid, sc, br), aff in user_affinity_counts.items()
    ]

    # 4.1: carritos pendientes con fecha. ~30 % "activos" (< 20 min, llegan por shop.events en vivo),
    # el resto "olvidados" (1 h – 20 días, del histórico). En este mundo nadie recompra -> sin recurrentes.
    now_ms_gen = time.time() * 1000.0
    cart_rng = np.random.default_rng(seed + 991)  # RNG propio: no altera el mundo de 3.1b
    cart_hist_rows: list[dict[str, Any]] = []
    live_cart_event_rows: list[dict[str, Any]] = []
    for uid, pids in user_cart_intent.items():
        for pid in pids:
            if pid in user_bought_map[uid]:
                continue
            if cart_rng.random() < 0.30:
                live_cart_event_rows.append(
                    {"user_id": uid, "product_id": pid, "n_purchases": 0,
                     "last_cart_ms": now_ms_gen - float(cart_rng.uniform(1.0, 20.0)) * 60_000.0,
                     "last_purchase_ms": 0.0}
                )
            else:
                cart_hist_rows.append(
                    {"user_id": uid, "product_id": pid, "n_purchases": 0,
                     "last_cart_ms": now_ms_gen - float(cart_rng.uniform(1.0, 480.0)) * 3_600_000.0,
                     "last_purchase_ms": 0.0}
                )

    trend_pids = warm_pids[:20]
    live_trend_rows = [
        {"product_id": pid, "live_score": float(rng.uniform(25.0, 95.0))}
        for pid in trend_pids
    ]

    exam_probs = [(k + 1.0) ** (-0.75) for k in range(10)]

    def _sample_request(idx: int, prefix: str) -> tuple[dict[str, Any], str, float]:
        is_new_user = rng.random() < 0.16
        uid = f"u_new_{idx:05d}" if is_new_user else warm_users[int(rng.integers(0, len(warm_users)))]
        reg = regions[int(rng.integers(0, len(regions)))]
        dev = devices[int(rng.integers(0, len(devices)))]
        pg = pages[int(rng.integers(0, len(pages)))]

        has_views = pg != "home" or rng.random() < 0.45
        recent_views: list[dict[str, str]] = []
        true_target = ""

        if has_views:
            use_cold_view = rng.random() < 0.18
            if use_cold_view:
                seed_p = cold_pids[int(rng.integers(0, len(cold_pids)))]
                sc = pid_to_subcat[seed_p]
                recent_views = [{"product_id": seed_p}]
                cand_pool = [q for q in latent_next[seed_p] if q not in user_bought_map.get(uid, ())]
                true_target = cand_pool[0] if cand_pool else warm_subcat_pids[sc][0]
            else:
                sc = (
                    user_fav_subcat[uid]
                    if (not is_new_user and rng.random() < 0.72)
                    else subcat_specs[int(rng.integers(0, len(subcat_specs)))][1]
                )
                w_sc = warm_subcat_pids[sc]
                v1 = w_sc[int(rng.integers(0, len(w_sc)))]
                recent_views = [{"product_id": v1}]
                if rng.random() < 0.24 and cold_subcat_pids[sc]:
                    matching_new = [
                        np_id
                        for np_id in cold_subcat_pids[sc]
                        if np_id in cold_winners and pid_brand[np_id] == pid_brand[v1]
                    ]
                    if not matching_new:
                        matching_new = [np_id for np_id in cold_subcat_pids[sc] if np_id in cold_winners]
                    true_target = matching_new[0] if matching_new else latent_next[v1][0]
                elif rng.random() < 0.25 and not is_new_user and user_cart_intent.get(uid):
                    avail = [x for x in user_cart_intent[uid] if x not in user_bought_map[uid]]
                    true_target = avail[0] if avail else latent_next[v1][0]
                else:
                    nxt = [x for x in latent_next[v1] if x not in user_bought_map.get(uid, ())]
                    true_target = nxt[0] if nxt else v1
        else:
            if not is_new_user and user_cart_intent.get(uid) and rng.random() < 0.45:
                avail = [x for x in user_cart_intent[uid] if x not in user_bought_map[uid]]
                true_target = avail[0] if avail else ctx_heroes[(reg, dev, pg)][0]
            else:
                heroes = [h for h in ctx_heroes[(reg, dev, pg)] if h not in user_bought_map.get(uid, ())]
                true_target = heroes[0] if heroes else baseline_top10[0]

        req = {
            "request_id": f"{prefix}_{idx:05d}",
            "user_id": uid,
            "is_new_user": is_new_user,
            "context": {"region": reg, "device": dev, "page": pg},
            "recent_views": recent_views,
        }
        return req, true_target, pid_price[true_target]

    feedback_rows: list[dict[str, Any]] = []
    for i in range(n_feedback_sessions):
        req, target_pid, target_price = _sample_request(i, "fb")
        recent_pids = extract_recent_product_ids(req["recent_views"])
        sc = pid_to_subcat.get(recent_pids[-1], "") if recent_pids else ""
        pool = (warm_subcat_pids.get(sc, warm_pids[:12]) + cold_subcat_pids.get(sc, []))[:18]
        if target_pid not in pool:
            pool.append(target_pid)
        shown = list(rng.choice(pool, size=min(10, len(pool)), replace=False))
        while len(shown) < 10:
            fill = warm_pids[int(rng.integers(0, len(warm_pids)))]
            if fill not in shown:
                shown.append(fill)

        clicked = ""
        purchased = ""
        rev = 0.0
        if target_pid in shown:
            slot = shown.index(target_pid)
            p_buy = 0.42 * exam_probs[slot]
            p_clk = 0.65 * exam_probs[slot]
            r = rng.random()
            if r < p_buy:
                clicked = target_pid
                purchased = target_pid
                rev = target_price
            elif r < p_clk:
                clicked = target_pid
        elif rng.random() < 0.06:
            clicked = shown[0]

        feedback_rows.append(
            {
                **req,
                "items": shown,
                "shown": shown,
                "clicked": clicked,
                "purchased": purchased,
                "revenue_eur": rev,
            }
        )

    test_records: list[dict[str, Any]] = []
    for i in range(n_test_sessions):
        req, target_pid, target_price = _sample_request(i, "test")
        slot0 = target_pid if rng.random() < 0.25 else baseline_top10[int(rng.integers(0, len(baseline_top10)))]
        prop0 = 0.25 if slot0 == target_pid else 0.08
        shown = [slot0]
        for b_pid in baseline_top10:
            if b_pid not in shown and len(shown) < 10:
                shown.append(b_pid)
        props = [prop0] + [0.05] * (len(shown) - 1)

        clicked = target_pid if rng.random() < 0.60 else ""
        purchased = target_pid if rng.random() < 0.38 else ""
        rev = target_price if purchased == target_pid else 0.0

        test_records.append(
            {
                **req,
                "shown": shown,
                "propensity": props,
                "clicked": clicked,
                "purchased": purchased,
                "revenue_eur": rev,
            }
        )

    df_test = pd.DataFrame(test_records)

    snap_v1 = build_snapshot_from_records(
        catalog_rows,
        popularity_rows=popularity_rows,
        context_rows=context_rows,
        transition_rows=trans_rows_v1,
        user_bought_rows=user_bought_rows,
        live_trend_rows=live_trend_rows,
        version=1,
    )
    snap_v2 = build_snapshot_v2_from_records(
        catalog_rows,
        popularity_rows=popularity_rows,
        context_rows=context_rows,
        transition_rows=trans_rows_v2,
        user_profile_rows=user_profile_rows,
        user_intent_rows=user_intent_rows,
        user_affinity_rows=user_affinity_rows,
        user_bought_rows=user_bought_rows,
        live_trend_rows=live_trend_rows,
        version=1,
    )
    snap_v3, _ = build_snapshot_v3_from_records(
        catalog_rows,
        popularity_rows=popularity_rows,
        context_rows=context_rows,
        transition_rows=trans_rows_v2,
        user_profile_rows=user_profile_rows,
        user_intent_rows=user_intent_rows,
        user_affinity_rows=user_affinity_rows,
        user_bought_rows=user_bought_rows,
        live_trend_rows=live_trend_rows,
        feedback_rows=feedback_rows,
        config=ConfigV3(),
        version=1,
    )
    v4_records = {
        "catalog_rows": catalog_rows,
        "popularity_rows": popularity_rows,
        "context_rows": context_rows,
        "transition_rows": trans_rows_v2,
        "user_profile_rows": user_profile_rows,
        "user_intent_rows": user_intent_rows,
        "user_affinity_rows": user_affinity_rows,
        "user_bought_rows": user_bought_rows,
        "live_trend_rows": live_trend_rows,
        "feedback_rows": feedback_rows,
        "cart_hist_rows": cart_hist_rows,
        "live_cart_event_rows": live_cart_event_rows,
    }
    snap_v4, scratch_v4 = build_snapshot_v4_from_records(
        **v4_records,
        config=config_v4 or ConfigV4(),
        version=1,
        weights_profile=weights_profile,
    )

    static_bundle = StaticHistoryBundle(
        fetched_at=time.time(),
        catalog_rows=catalog_rows,
        popularity_rows=popularity_rows,
        transition_rows=trans_rows_v2,
        user_profile_rows=user_profile_rows,
        user_intent_rows=user_intent_rows,
        user_affinity_rows=user_affinity_rows,
        context_rows=context_rows,
        training_session_rows=[],
        cart_hist_rows=cart_hist_rows,
    )

    return {
        "df_test": df_test,
        "baseline_top10": baseline_top10,
        "snap_v1": snap_v1,
        "snap_v2": snap_v2,
        "snap_v3": snap_v3,
        "snap_v4": snap_v4,
        "v4_records": v4_records,
        "scratch_payload": scratch_v4,
        "static_bundle": static_bundle,
        "cold_product_ids": frozenset(cold_product_ids),
        "warm_user_ids": frozenset(warm_users),
        "feedback_rows": feedback_rows,
        "catalog_rows": catalog_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 Offline 5-Way Benchmark (Calypso 4.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default="")
    parser.add_argument("--preset", type=str, default="", help="Evaluate a specific WeightsProfile preset")
    args = parser.parse_args()

    wp = get_preset(args.preset) if args.preset else ConfigV4().load_weights_profile()
    cfg_v4 = ConfigV4.from_profile(wp)
    data = generate_synthetic_dataset_v4(seed=args.seed, config_v4=cfg_v4, weights_profile=wp)
    df_test = data["df_test"]
    snap_v1 = data["snap_v1"]
    snap_v2 = data["snap_v2"]
    snap_v3 = data["snap_v3"]
    snap_v4 = data["snap_v4"]

    rec_v1 = Stage1Recommender(ConfigV1())
    rec_v2 = Stage2Recommender(ConfigV2())
    rec_v3 = CalypsoRecommender(ConfigV3())
    rec_v4 = CalypsoRecommenderV4(cfg_v4)
    baseline_top10 = data["baseline_top10"]

    def _baseline_policy(row: Any) -> list[str]:
        return ensure_valid_submission(baseline_top10, snap_v4.active_ids, snap_v4.fallback_ranked)

    policies: dict[str, PolicyFn] = {
        "baseline": _baseline_policy,
        "etapa1_hybrid": lambda row: rec_v1.recommend(row, snap_v1),
        "etapa2_seq_mmr": lambda row: rec_v2.recommend(row, snap_v2),
        "etapa3_calypso": lambda row: rec_v3.recommend(row, snap_v3),
        "etapa4_" + cfg_v4.model_version: lambda row: rec_v4.recommend(row, snap_v4),
    }

    report = evaluate_stratified(
        policies,
        df_test,
        cold_product_ids=data["cold_product_ids"],
        warm_user_ids=data["warm_user_ids"],
    )

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 190)
    print(
        f"=== Stage 4 Benchmark (Unified {cfg_v4.model_version} · profile={wp.profile_name}) | "
        f"ML alpha={snap_v4.ranker.ml_blend_alpha:.3f} | "
        f"Bandit eff_exp={snap_v4.bandit_state.total_effective_exposures:.1f} | "
        f"Agent regime={snap_v4.agent_overrides.regime} ==="
    )
    print(report.round(5))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path)
        print(f"Saved CSV report to {out_path}")


if __name__ == "__main__":
    main()
