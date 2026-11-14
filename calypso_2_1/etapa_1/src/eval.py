"""Offline evaluation (HitRate, DCG, IPS/SNIPS for Clicks, Purchases & Revenue) + synthetic simulator."""
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
import time
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .guardrails import ensure_valid_submission
from .recommender import Stage1Recommender, _extract_recent_product_ids, build_snapshot_from_records
from .state import RecommenderSnapshot

PolicyFn = Callable[[Any], list[str]]


def _is_nonempty(val: Any) -> bool:
    return val is not None and val != "" and not (isinstance(val, float) and np.isnan(val))


def evaluate_policy(
    policy_fn: PolicyFn,
    df: pd.DataFrame,
) -> dict[str, float | int]:
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
    """Match the challenge Cold Start definition (new user or new product in recent_views)."""
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
    for pid in _extract_recent_product_ids(recent):
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
            metrics = evaluate_policy(pol_fn, seg_df)
            rows.append({"segment": seg_name, "policy": pol_name, **metrics})

    return pd.DataFrame(rows).set_index(["segment", "policy"])


def generate_synthetic_dataset(
    seed: int = 42,
    n_warm_products: int = 500,
    n_new_products: int = 40,
    n_warm_users: int = 1200,
    n_history_events: int = 20_000,
    n_test_sessions: int = 4_000,
) -> dict[str, Any]:
    """Generate a realistic synthetic store dataset matching RawTree's exact schema."""
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

    total_products = n_warm_products + n_new_products
    cold_product_ids: set[str] = set()

    for idx in range(total_products):
        pid = f"p_{idx:04d}"
        cat, subcat, brands, keywords = subcat_specs[idx % len(subcat_specs)]
        brand = brands[idx % len(brands)]
        price = float(np.round(rng.uniform(18.0, 220.0), 2))
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
                "description": f"{brand} {keywords} edition {idx % 17} quality design",
                "is_active": True,
                "is_new_product": is_new,
            }
        )
        subcat_to_pids[subcat].append(pid)
        pid_to_subcat[pid] = subcat
        pid_price[pid] = price

    warm_pids = [f"p_{i:04d}" for i in range(n_warm_products)]
    warm_subcat_pids = {
        sc: [p for p in pids if p not in cold_product_ids] for sc, pids in subcat_to_pids.items()
    }

    # Build latent co-purchase clusters within each subcategory + brand (semantic coherence)
    latent_next: dict[str, list[str]] = {}
    pid_brand = {r["product_id"]: r["brand"] for r in catalog_rows}
    for sc, pids in subcat_to_pids.items():
        warm_sc = warm_subcat_pids[sc]
        for pid in pids:
            same_brand = [q for q in warm_sc if q != pid and pid_brand[q] == pid_brand[pid]]
            others = [q for q in warm_sc if q != pid]
            pool = same_brand if len(same_brand) >= 3 else others
            latent_next[pid] = list(rng.choice(pool, size=min(4, len(pool)), replace=False))

    # Popularity Zipf weights among warm items
    warm_pop_weights = 1.0 / (np.arange(1, len(warm_pids) + 1) ** 0.85)
    warm_pop_weights /= warm_pop_weights.sum()

    # User affinity + organic history_events (< CUT)
    warm_users = [f"u_{i:04d}" for i in range(n_warm_users)]
    user_fav_subcat = {u: subcat_specs[i % len(subcat_specs)][1] for i, u in enumerate(warm_users)}
    user_bought_map: dict[str, set[str]] = defaultdict(set)

    pop_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"views": 0, "carts": 0, "purchases": 0})
    trans_counts: dict[tuple[str, str], float] = defaultdict(float)
    regions = ["Catalunya", "Madrid", "Andalucia", "Valencia"]
    devices = ["web", "ios", "android"]
    pages = ["home", "product", "category"]
    context_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)

    n_sessions_hist = n_history_events // 3
    for s_idx in range(n_sessions_hist):
        uid = warm_users[int(rng.integers(0, n_warm_users))]
        sc = user_fav_subcat[uid] if rng.random() < 0.75 else subcat_specs[int(rng.integers(0, len(subcat_specs)))][1]
        sc_warm = warm_subcat_pids[sc]
        anchor = str(rng.choice(sc_warm))
        pop_counts[anchor]["views"] += 1

        # Co-viewed / co-purchased item in same session
        mate = str(rng.choice(latent_next[anchor]))
        pop_counts[mate]["views"] += 1
        trans_counts[(anchor, mate)] += 1.0

        if rng.random() < 0.35:
            pop_counts[mate]["carts"] += 1
            trans_counts[(anchor, mate)] += 2.5
        if rng.random() < 0.22:
            pop_counts[mate]["purchases"] += 1
            trans_counts[(anchor, mate)] += 6.0
            user_bought_map[uid].add(mate)
            region = regions[s_idx % len(regions)]
            device = devices[(s_idx // 2) % len(devices)]
            page = pages[(s_idx // 3) % len(pages)]
            context_counts[(region, device, page, mate)] += 1

    popularity_rows = [
        {
            "product_id": pid,
            "views": c["views"],
            "carts": c["carts"],
            "purchases": c["purchases"],
            "weighted_pop": c["purchases"] * 5.0 + c["carts"] * 2.0 + c["views"] * 0.2,
        }
        for pid, c in pop_counts.items()
    ]
    transition_rows = [
        {"src_product": src, "dst_product": dst, "transition_score": score}
        for (src, dst), score in trans_counts.items()
    ]
    user_bought_rows = [
        {"user_id": uid, "bought_products": list(pids)}
        for uid, pids in user_bought_map.items()
    ]

    # Generate test sessions (>= CUT) produced by a popularity+epsilon logging policy
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

        # Cold-start or warm sessions viewing a recent product
        recent_views: list[dict[str, str]] = []
        if page != "home" or rng.random() < 0.75:
            if rng.random() < 0.22:
                sc_new = [p for p in subcat_to_pids[fav_sc] if p in cold_product_ids]
                anchor = str(rng.choice(sc_new)) if sc_new else str(rng.choice(warm_subcat_pids[fav_sc]))
            else:
                anchor = str(rng.choice(warm_subcat_pids[fav_sc]))
            recent_views.append({"product_id": anchor, "ts": "2026-11-12T10:00:00Z"})
            target_pool = set(latent_next[anchor]) | {
                p for p in warm_subcat_pids[pid_to_subcat[anchor]]
                if pid_brand[p] == pid_brand[anchor]
            }
        else:
            anchor = ""
            target_pool = set(warm_subcat_pids[fav_sc][:8])

        # Logging policy: top-sellers + uniform exploration so IPS has healthy overlap
        shown: list[str] = []
        propensity: list[float] = []
        used: set[str] = set()
        for slot in range(10):
            if rng.random() < 0.30:
                cand_list = [p for p in global_best_warm[:20] if p not in used]
                choice = cand_list[0] if cand_list else warm_pids[0]
                p_sel = 0.30 / max(len(cand_list[:3]), 1) + 0.70 / len(warm_pids)
            else:
                avail = [p for p in warm_pids if p not in used]
                choice = str(rng.choice(avail))
                p_sel = 0.70 / max(len(avail), 1)
            shown.append(choice)
            propensity.append(float(max(p_sel, 1e-4)))
            used.add(choice)

        already = user_bought_map.get(uid, set()) if not is_new_user else set()
        click_probs = np.zeros(10, dtype=float)
        for slot, pid in enumerate(shown):
            if pid in already:
                base_aff = 0.002
            elif pid in target_pool:
                base_aff = 0.55
            elif pid_to_subcat[pid] == fav_sc:
                base_aff = 0.22
            else:
                base_aff = 0.025
            click_probs[slot] = base_aff * slot_attention[slot]

        clicked_pid = ""
        purchased_pid = ""
        rev = 0.0
        if rng.random() < min(0.75, float(click_probs.sum())):
            probs_norm = click_probs / click_probs.sum()
            chosen_slot = int(rng.choice(10, p=probs_norm))
            clicked_pid = shown[chosen_slot]
            buy_prob = 0.55 if clicked_pid in target_pool else 0.24
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

    context_rows = [
        {"region": r, "device": d, "page": p, "product_id": pid, "buys": cnt}
        for (r, d, p, pid), cnt in context_counts.items()
    ]

    snapshot = build_snapshot_from_records(
        catalog_rows=catalog_rows,
        popularity_rows=popularity_rows,
        context_rows=context_rows,
        transition_rows=transition_rows,
        user_bought_rows=user_bought_rows,
    )

    return {
        "snapshot": snapshot,
        "df_test": pd.DataFrame(test_rows),
        "cold_product_ids": frozenset(cold_product_ids),
        "warm_user_ids": frozenset(warm_users),
    }


def run_synthetic_benchmark(seed: int = 42) -> pd.DataFrame:
    """Run offline comparison of Baseline vs Stage 1 Hybrid on the synthetic simulator."""
    data = generate_synthetic_dataset(seed=seed)
    snapshot: RecommenderSnapshot = data["snapshot"]
    df_test: pd.DataFrame = data["df_test"]

    recommender = Stage1Recommender(Config())

    def baseline_policy(row: Any) -> list[str]:
        return ensure_valid_submission(
            candidates=snapshot.global_ranked[:10],
            active_ids=snapshot.active_ids,
            fallback_ranked=snapshot.global_ranked,
        )

    def stage1_policy(row: Any) -> list[str]:
        return recommender.recommend(row, snapshot)

    return evaluate_stratified(
        policies={"baseline": baseline_policy, "stage1_hybrid": stage1_policy},
        df_test=df_test,
        cold_product_ids=data["cold_product_ids"],
        warm_user_ids=data["warm_user_ids"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 Offline Evaluator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic benchmark")
    args = parser.parse_args()

    results = run_synthetic_benchmark(seed=args.seed)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    print("\n=== Etapa 1 · Benchmark Offline (Baseline vs Stage 1 Hybrid) ===")
    print(results.round(5))


if __name__ == "__main__":
    main()
