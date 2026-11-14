"""Offline evaluation comparing Baseline vs Etapa 1 vs Etapa 2 across ALL, WARM & COLD_START."""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from .config import ConfigV2
from .guardrails import ensure_valid_submission
from .recommender import Stage2Recommender, _extract_recent_views, build_snapshot_v2_from_records
from .state import RecommenderSnapshotV2

# Ensure workspace root is on sys.path so we can benchmark cleanly against etapa_1
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from etapa_1.src.config import Config as ConfigV1  # noqa: E402
from etapa_1.src.recommender import (  # noqa: E402
    Stage1Recommender,
    build_snapshot_from_records as build_snapshot_v1_from_records,
)

PolicyFn = Callable[[Any], list[str]]


def _is_nonempty(val: Any) -> bool:
    return val is not None and val != "" and not (isinstance(val, float) and np.isnan(val))


def evaluate_policy(policy_fn: PolicyFn, df: pd.DataFrame) -> dict[str, float | int]:
    """Compute HitRate@10, DCG@10, and IPS/SNIPS (CTR, CVR, Revenue) on session logs."""
    n = len(df)
    if n == 0:
        return {
            "n_requests": 0,
            "hit_rate_click@10": 0.0,
            "hit_rate_buy@10": 0.0,
            "dcg_buy@10": 0.0,
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

    hit_click_arr = np.array([c in r for c, r in zip(clicked, recs)], dtype=float)
    hit_buy_arr = np.array([b in r for b, r in zip(purchased, recs)], dtype=float)

    dcg_buy_arr = np.zeros(n, dtype=float)
    for i, (b, r) in enumerate(zip(purchased, recs)):
        if b and b in r:
            rank = r.index(b)
            dcg_buy_arr[i] = 1.0 / np.log2(rank + 2.0)

    first_rec = np.array([r[0] for r in recs], dtype=object)
    first_shown = np.array([s[0] if s else "" for s in df["shown"]], dtype=object)
    prop1 = np.array(
        [max(float(p[0]), 1e-4) if p else 1.0 for p in df["propensity"]],
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
    row: Mapping[str, Any] | Any,
    cold_product_ids: frozenset[str],
    warm_user_ids: frozenset[str] | None = None,
) -> bool:
    is_new_user = (
        bool(row.get("is_new_user", False))
        if isinstance(row, Mapping)
        else bool(getattr(row, "is_new_user", False))
    )
    if is_new_user:
        return True
    if warm_user_ids is not None:
        uid = str(row.get("user_id", "") if isinstance(row, Mapping) else getattr(row, "user_id", ""))
        if uid not in warm_user_ids:
            return True
    recent = (
        row.get("recent_views", ())
        if isinstance(row, Mapping)
        else getattr(row, "recent_views", ())
    )
    for pid, _ in _extract_recent_views(recent):
        if pid in cold_product_ids:
            return True
    return False


def evaluate_stratified(
    policies: Mapping[str, PolicyFn],
    df_test: pd.DataFrame,
    cold_product_ids: frozenset[str] = frozenset(),
    warm_user_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
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
            metrics = evaluate_policy(pol_fn, seg_df)
            rows.append({"segment": seg_name, "policy": pol_name, **metrics})

    return pd.DataFrame(rows).set_index(["segment", "policy"])


def generate_synthetic_dataset_v2(
    seed: int = 42,
    n_warm_products: int = 500,
    n_new_products: int = 40,
    n_warm_users: int = 1200,
    n_history_events: int = 24_000,
    n_test_sessions: int = 5_000,
) -> dict[str, Any]:
    """Generate rich synthetic RawTree data modeling brand, budget, intent recovery, and cold start."""
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
    pid_to_brand: dict[str, str] = {}
    pid_price: dict[str, float] = {}

    total_products = n_warm_products + n_new_products
    cold_product_ids: set[str] = set()

    for idx in range(total_products):
        pid = f"p_{idx:04d}"
        cat, subcat, brands, keywords = subcat_specs[idx % len(subcat_specs)]
        brand = brands[idx % len(brands)]
        # Assign price tiers correlated with brand index so budget sensitivity matters
        tier_base = 35.0 + 55.0 * (idx % len(brands))
        price = float(np.round(tier_base * rng.uniform(0.8, 1.3), 2))
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
                "description": f"{brand} {keywords} edition {idx % 11} premium",
                "is_active": True,
                "is_new_product": is_new,
            }
        )
        subcat_to_pids[subcat].append(pid)
        pid_to_subcat[pid] = subcat
        pid_to_brand[pid] = brand
        pid_price[pid] = price

    warm_pids = [f"p_{i:04d}" for i in range(n_warm_products)]
    warm_subcat_pids = {
        sc: [p for p in pids if p not in cold_product_ids] for sc, pids in subcat_to_pids.items()
    }
    new_subcat_pids = {
        sc: [p for p in pids if p in cold_product_ids] for sc, pids in subcat_to_pids.items()
    }

    # Build niche co-purchase clusters (same subcategory + same brand)
    latent_next: dict[str, list[str]] = {}
    for sc, pids in subcat_to_pids.items():
        warm_sc = warm_subcat_pids[sc]
        for pid in pids:
            same_brand = [q for q in warm_sc if q != pid and pid_to_brand[q] == pid_to_brand[pid]]
            pool = same_brand if len(same_brand) >= 3 else [q for q in warm_sc if q != pid]
            latent_next[pid] = list(rng.choice(pool, size=min(4, len(pool)), replace=False))

    # User preferences: subcategory, brand, budget tier, and unbought cart intents
    warm_users = [f"u_{i:04d}" for i in range(n_warm_users)]
    user_fav_subcat = {u: subcat_specs[i % len(subcat_specs)][1] for i, u in enumerate(warm_users)}
    user_fav_brand: dict[str, str] = {}
    for i, u in enumerate(warm_users):
        sc = user_fav_subcat[u]
        brands = next(sp[2] for sp in subcat_specs if sp[1] == sc)
        user_fav_brand[u] = brands[i % len(brands)]

    user_bought_map: dict[str, set[str]] = defaultdict(set)
    user_log_prices: dict[str, list[float]] = defaultdict(list)
    user_intent_counts: dict[tuple[str, str], float] = defaultdict(float)
    user_aff_counts: dict[tuple[str, str, str], float] = defaultdict(float)

    pop_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"views": 0, "carts": 0, "purchases": 0})
    trans_counts: dict[tuple[str, str], float] = defaultdict(float)

    regions = ["Catalunya", "Madrid", "Andalucia", "Valencia"]
    devices = ["web", "ios", "android"]
    pages = ["home", "product", "category"]
    context_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)

    # Inject a few ultra-popular generic items to simulate popularity distortion in raw co-views
    generic_popular = warm_pids[:12]

    n_sessions_hist = n_history_events // 3
    for s_idx in range(n_sessions_hist):
        uid = warm_users[int(rng.integers(0, n_warm_users))]
        sc = user_fav_subcat[uid] if rng.random() < 0.78 else subcat_specs[int(rng.integers(0, len(subcat_specs)))][1]
        sc_warm = warm_subcat_pids[sc]
        brand_match = [p for p in sc_warm if pid_to_brand[p] == user_fav_brand[uid]]
        anchor = str(rng.choice(brand_match if brand_match and rng.random() < 0.65 else sc_warm))
        pop_counts[anchor]["views"] += 1
        user_aff_counts[(uid, sc, pid_to_brand[anchor])] += 0.3

        # Generic noisy co-views (simulates raw co-view pollution by global best-sellers)
        if rng.random() < 0.35:
            noisy = generic_popular[int(rng.integers(0, len(generic_popular)))]
            if noisy != anchor:
                pop_counts[noisy]["views"] += 1
                trans_counts[(anchor, noisy)] += 1.5

        mate = str(rng.choice(latent_next[anchor]))
        pop_counts[mate]["views"] += 1
        trans_counts[(anchor, mate)] += 1.0
        user_aff_counts[(uid, pid_to_subcat[mate], pid_to_brand[mate])] += 0.3

        # Abandoned cart / intent signal for mate or anchor
        if rng.random() < 0.42:
            pop_counts[mate]["carts"] += 1
            trans_counts[(anchor, mate)] += 2.5
            user_intent_counts[(uid, mate)] += 3.0
            user_aff_counts[(uid, pid_to_subcat[mate], pid_to_brand[mate])] += 2.0

        if rng.random() < 0.24:
            pop_counts[mate]["purchases"] += 1
            trans_counts[(anchor, mate)] += 6.0
            user_bought_map[uid].add(mate)
            user_log_prices[uid].append(math.log1p(pid_price[mate]))
            user_aff_counts[(uid, pid_to_subcat[mate], pid_to_brand[mate])] += 5.0
            context_counts[(regions[s_idx % len(regions)], devices[(s_idx // 2) % len(devices)], pages[(s_idx // 3) % len(pages)], mate)] += 1

    popularity_rows = [
        {
            "product_id": pid,
            "views": c["views"],
            "carts": c["carts"],
            "purchases": c["purchases"],
            "self_cvr": (c["purchases"] + 0.5) / (c["views"] + 25.0),
            "weighted_pop": c["purchases"] * 5.0 + c["carts"] * 2.0 + c["views"] * 0.2,
        }
        for pid, c in pop_counts.items()
    ]
    transition_rows = [
        {"src_product": src, "dst_product": dst, "raw_transition_score": score, "transition_score": score}
        for (src, dst), score in trans_counts.items()
    ]
    user_bought_rows = [{"user_id": uid, "bought_products": list(pids)} for uid, pids in user_bought_map.items()]
    user_profile_rows = [
        {
            "user_id": uid,
            "bought_products": list(user_bought_map.get(uid, ())),
            "log_price_mean": float(np.mean(user_log_prices[uid])) if user_log_prices[uid] else 0.0,
            "log_price_std": float(np.std(user_log_prices[uid])) if len(user_log_prices[uid]) > 1 else 0.4,
        }
        for uid in warm_users
    ]
    user_intent_rows = [
        {"user_id": uid, "product_id": pid, "intent_score": score}
        for (uid, pid), score in user_intent_counts.items()
        if pid not in user_bought_map.get(uid, set())
    ]
    user_affinity_rows = [
        {"user_id": uid, "subcategory": sc, "brand": br, "affinity": aff}
        for (uid, sc, br), aff in user_aff_counts.items()
    ]
    context_rows = [
        {"region": r, "device": d, "page": p, "product_id": pid, "buys": cnt}
        for (r, d, p, pid), cnt in context_counts.items()
    ]

    # Generate test sessions (>= CUT) including newly added cold products in logging policy exploration
    all_active_pids = [r["product_id"] for r in catalog_rows]
    global_best_warm = sorted(
        warm_pids,
        key=lambda p: -(pop_counts[p]["purchases"] * 5.0 + pop_counts[p]["views"] * 0.2),
    )

    test_rows: list[dict[str, Any]] = []
    slot_attention = np.array([1.0 / ((k + 1) ** 0.75) for k in range(10)], dtype=float)

    for t_idx in range(n_test_sessions):
        is_new_user = bool(rng.random() < 0.18)
        uid = f"u_new_{t_idx}" if is_new_user else warm_users[int(rng.integers(0, n_warm_users))]
        region = regions[int(rng.integers(0, len(regions)))]
        device = devices[int(rng.integers(0, len(devices)))]
        page = pages[int(rng.integers(0, len(pages)))]

        fav_sc = (
            subcat_specs[int(rng.integers(0, len(subcat_specs)))][1]
            if is_new_user
            else user_fav_subcat[uid]
        )
        fav_brand = "" if is_new_user else user_fav_brand[uid]
        unbought_intents = (
            set()
            if is_new_user
            else {
                pid
                for (u, pid), sc in user_intent_counts.items()
                if u == uid and pid not in user_bought_map.get(uid, set()) and sc >= 3.0
            }
        )

        recent_views: list[dict[str, str]] = []
        if page != "home" or rng.random() < 0.75:
            if rng.random() < 0.22 and new_subcat_pids[fav_sc]:
                anchor = str(rng.choice(new_subcat_pids[fav_sc]))
            else:
                brand_warm = [p for p in warm_subcat_pids[fav_sc] if pid_to_brand[p] == fav_brand]
                anchor = str(rng.choice(brand_warm if brand_warm else warm_subcat_pids[fav_sc]))
            recent_views.append({"product_id": anchor, "ts": "2026-11-12T09:58:00Z"})
            target_pool = (
                set(latent_next[anchor])
                | {
                    p
                    for p in subcat_to_pids[pid_to_subcat[anchor]]
                    if pid_to_brand[p] == pid_to_brand[anchor]
                }
                | unbought_intents
            )
        else:
            target_pool = {
                p for p in subcat_to_pids[fav_sc] if not fav_brand or pid_to_brand[p] == fav_brand
            } | unbought_intents

        # Logging policy: global best sellers + subcategory exploration + uniform so SNIPS has strong overlap
        shown: list[str] = []
        propensity: list[float] = []
        used: set[str] = set()
        sc_pool = subcat_to_pids[fav_sc]
        for _ in range(10):
            u_draw = rng.random()
            if u_draw < 0.25:
                cand_list = [p for p in global_best_warm[:20] if p not in used]
                choice = cand_list[0] if cand_list else warm_pids[0]
            elif u_draw < 0.55:
                avail_sc = [p for p in sc_pool if p not in used]
                choice = str(rng.choice(avail_sc)) if avail_sc else str(rng.choice(all_active_pids))
            else:
                avail = [p for p in all_active_pids if p not in used]
                choice = str(rng.choice(avail))

            p_sel = 0.45 / max(len(all_active_pids) - len(used), 1)
            if choice in global_best_warm[:20] and choice == next((p for p in global_best_warm[:20] if p not in used), None):
                p_sel += 0.25
            if choice in sc_pool:
                p_sel += 0.30 / max(len([p for p in sc_pool if p not in used]), 1)
            shown.append(choice)
            propensity.append(float(max(p_sel, 1e-4)))
            used.add(choice)

        already = user_bought_map.get(uid, set()) if not is_new_user else set()
        click_probs = np.zeros(10, dtype=float)
        for slot, pid in enumerate(shown):
            if pid in already:
                base_aff = 0.002
            elif pid in unbought_intents:
                base_aff = 0.65
            elif pid in target_pool:
                base_aff = 0.55 if pid not in cold_product_ids else 0.60
            elif pid_to_subcat[pid] == fav_sc:
                base_aff = 0.20
            else:
                base_aff = 0.02
            click_probs[slot] = base_aff * slot_attention[slot]

        clicked_pid = ""
        purchased_pid = ""
        rev = 0.0
        if rng.random() < min(0.78, float(click_probs.sum())):
            probs_norm = click_probs / click_probs.sum()
            chosen_slot = int(rng.choice(10, p=probs_norm))
            clicked_pid = shown[chosen_slot]
            buy_prob = 0.58 if (clicked_pid in target_pool or clicked_pid in unbought_intents) else 0.22
            if clicked_pid in already:
                buy_prob = 0.01
            if rng.random() < buy_prob:
                purchased_pid = clicked_pid
                rev = pid_price[purchased_pid]

        test_rows.append(
            {
                "request_id": f"r_syn_{t_idx:05d}",
                "ts": "2026-11-12T10:05:00Z",
                "user_id": uid,
                "is_new_user": is_new_user,
                "context": {"region": region, "device": device, "page": page},
                "recent_views": recent_views,
                "shown": shown,
                "propensity": propensity,
                "clicked": clicked_pid,
                "purchased": purchased_pid,
                "revenue_eur": rev,
            }
        )

    snapshot_v1 = build_snapshot_v1_from_records(
        catalog_rows=catalog_rows,
        popularity_rows=popularity_rows,
        context_rows=context_rows,
        transition_rows=transition_rows,
        user_bought_rows=user_bought_rows,
    )
    snapshot_v2 = build_snapshot_v2_from_records(
        catalog_rows=catalog_rows,
        popularity_rows=popularity_rows,
        context_rows=context_rows,
        transition_rows=transition_rows,
        user_profile_rows=user_profile_rows,
        user_intent_rows=user_intent_rows,
        user_affinity_rows=user_affinity_rows,
        user_bought_rows=user_bought_rows,
        lift_beta=ConfigV2().lift_beta,
        lift_prior_pop=ConfigV2().lift_prior_pop,
    )

    return {
        "snapshot_v1": snapshot_v1,
        "snapshot_v2": snapshot_v2,
        "df_test": pd.DataFrame(test_rows),
        "cold_product_ids": frozenset(cold_product_ids),
        "warm_user_ids": frozenset(warm_users),
    }


def run_benchmark_v2(seed: int = 42) -> pd.DataFrame:
    """Compare Baseline vs Etapa 1 Hybrid vs Etapa 2 Seq-MMR on the same dataset."""
    data = generate_synthetic_dataset_v2(seed=seed)
    snap_v1 = data["snapshot_v1"]
    snap_v2: RecommenderSnapshotV2 = data["snapshot_v2"]
    df_test: pd.DataFrame = data["df_test"]

    rec_v1 = Stage1Recommender(ConfigV1())
    rec_v2 = Stage2Recommender(ConfigV2())

    def baseline_policy(row: Any) -> list[str]:
        return ensure_valid_submission(
            candidates=snap_v2.global_ranked[:10],
            active_ids=snap_v2.active_ids,
            fallback_ranked=snap_v2.global_ranked,
        )

    def etapa1_policy(row: Any) -> list[str]:
        return rec_v1.recommend(row, snap_v1)

    def etapa2_policy(row: Any) -> list[str]:
        return rec_v2.recommend(row, snap_v2)

    return evaluate_stratified(
        policies={
            "baseline": baseline_policy,
            "etapa1_hybrid": etapa1_policy,
            "etapa2_seq_mmr": etapa2_policy,
        },
        df_test=df_test,
        cold_product_ids=data["cold_product_ids"],
        warm_user_ids=data["warm_user_ids"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 2 Offline Evaluator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = run_benchmark_v2(seed=args.seed)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print("\n=== Etapa 2 · Benchmark Comparatiu (Baseline vs Etapa 1 vs Etapa 2) ===")
    print(results.round(5))


if __name__ == "__main__":
    main()
