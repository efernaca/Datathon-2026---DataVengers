"""Odyseus iterative offline parameter optimizer for Calypso 5.1f targeting max conversion."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

from .bandits import PositionAwareBandit
from .config import ConfigV4
from .disk_cache import DiskCache
from .eval import evaluate_policy, generate_synthetic_dataset_v4, is_cold_start_row
from .recommender import CalypsoRecommenderV4, build_product_static_table
from .reranker import PositionDebiasedRanker
from .state import RecommenderSnapshotV4
from .weights_profile import (
    SEGMENT_SIGNAL_NAMES,
    USER_SEGMENTS,
    WeightsProfile,
    get_preset,
    list_presets,
    load_active_profile,
)

MODEL_VERSION_LABEL = "calypso-5.1f"

# ---------------------------------------------------------------------------
# Full tunable parameter space on Calypso 5.1f WeightsProfile (~85 parameters)
# Spec types:
#   ("float", lo, hi, step, group)
#   ("int", lo, hi, step, group)
#   ("bool", group)
#   ("choice", options, group)
# ---------------------------------------------------------------------------
PARAM_SPECS: dict[str, tuple[Any, ...]] = {
    # 1. Session Core & 5.1 In-Session View / Category Coherency Signals
    "w_seq_lift": ("float", 0.30, 2.50, 0.22, "session_core"),
    "w_self_cvr_view": ("float", 0.50, 3.50, 0.28, "session_core"),
    "w_intent_score": ("float", 0.05, 1.60, 0.18, "session_core"),
    "w_same_top_subcat": ("float", 0.00, 0.70, 0.08, "session_core"),
    "w_same_top_brand": ("float", 0.00, 0.40, 0.05, "session_core"),
    "view_last_product_page_factor": ("float", 0.40, 1.35, 0.15, "session_core"),
    "view_base_bonus": ("float", 0.00, 1.10, 0.14, "session_core"),
    "recency_decay": ("float", 0.45, 0.90, 0.08, "session_core"),
    "slot1_category_match_bonus": ("float", 0.00, 0.55, 0.08, "session_core"),
    "same_cat_bonus_weight": ("float", 0.00, 0.25, 0.04, "session_core"),
    # 2. Calypso 5.1 Base-Rate Calibration, Pool Guarantees & Live Window
    "w_ctx_bayes": ("float", 0.05, 1.20, 0.14, "base_popularity_5_1"),
    "w_global_pop": ("float", 0.10, 1.40, 0.16, "base_popularity_5_1"),
    "w_live_trend": ("float", 0.02, 1.00, 0.12, "base_popularity_5_1"),
    "use_sales_rank_base_rate": ("bool", "base_popularity_5_1"),
    "sales_rank_blend_weight": ("float", 0.05, 0.95, 0.14, "base_popularity_5_1"),
    "live_pop_mix_weight": ("float", 0.05, 1.00, 0.15, "base_popularity_5_1"),
    "transition_dest_pop_gamma": ("float", 0.00, 0.85, 0.14, "base_popularity_5_1"),
    "candidate_pool_size": ("int", 50, 120, 12, "base_popularity_5_1"),
    "guaranteed_top_sellers_in_pool": ("int", 6, 40, 6, "base_popularity_5_1"),
    "v3_bg_penalty": ("bool", "base_popularity_5_1"),
    "bg_penalty_diff_subcat": ("float", 0.08, 1.00, 0.16, "base_popularity_5_1"),
    "bg_penalty_same_subcat": ("float", 0.15, 1.00, 0.16, "base_popularity_5_1"),
    # 3. Cold-Start, Neighbor Prior (X10) & Slot Gating
    "w_is_new_product": ("float", 0.00, 0.25, 0.04, "cold_start"),
    "w_reverse_graft": ("float", 0.05, 0.90, 0.12, "cold_start"),
    "w_forward_cold": ("float", 0.10, 1.10, 0.14, "cold_start"),
    "cold_neighbor_pop_prior_weight": ("float", 0.10, 1.20, 0.16, "cold_start"),
    "cold_sparse_cos_weight": ("float", 0.30, 0.85, 0.10, "cold_start"),
    "cold_subcat_match_bonus": ("float", 0.10, 0.70, 0.10, "cold_start"),
    "cold_cat_match_bonus": ("float", 0.00, 0.40, 0.06, "cold_start"),
    "cold_brand_match_bonus": ("float", 0.00, 0.40, 0.06, "cold_start"),
    "cold_price_prox_bonus": ("float", 0.00, 0.40, 0.06, "cold_start"),
    "cold_explore_start_slot_warm": ("int", 4, 9, 1, "cold_start"),
    "cold_explore_start_slot_cold": ("int", 0, 7, 2, "cold_start"),
    "max_cold_explore_in_warm": ("int", 0, 4, 1, "cold_start"),
    # 4. Personalization, Price Tie-Break & Discounts (X11)
    "w_user_affinity": ("float", 0.05, 1.00, 0.12, "personalization_price"),
    "w_price_compat": ("float", 0.00, 0.80, 0.10, "personalization_price"),
    "w_log_price_norm": ("float", -0.35, 0.15, 0.07, "personalization_price"),
    "price_std_floor": ("float", 0.20, 1.00, 0.12, "personalization_price"),
    "price_cheap_asymmetry": ("float", 0.00, 0.90, 0.16, "personalization_price"),
    "revenue_tie_weight": ("float", -0.15, 0.12, 0.04, "personalization_price"),
    "tie_break_mode": ("choice", ("conversion", "cheap", "revenue"), "personalization_price"),
    "tie_break_eps": ("float", 0.002, 0.040, 0.008, "personalization_price"),
    "discount_boost_weight": ("float", 0.00, 0.45, 0.08, "personalization_price"),
    # 5. Cart, Forgotten Cart & Recurrent Rebuy Rules (M6 / 4.1)
    "v4_cart_priority": ("bool", "cart_recurrent_rules"),
    "v4_cart_active_minutes": ("float", 5.0, 90.0, 15.0, "cart_recurrent_rules"),
    "v4_cart_max_slots": ("int", 1, 4, 1, "cart_recurrent_rules"),
    "cart_active_score_bonus": ("float", 0.25, 2.20, 0.28, "cart_recurrent_rules"),
    "v4_forgotten_cart": ("bool", "cart_recurrent_rules"),
    "v4_forgotten_max_slots": ("int", 0, 3, 1, "cart_recurrent_rules"),
    "cart_forgotten_score_bonus": ("float", 0.00, 1.00, 0.16, "cart_recurrent_rules"),
    "v4_recurrent": ("bool", "cart_recurrent_rules"),
    "v4_recurrent_max_slots": ("int", 1, 3, 1, "cart_recurrent_rules"),
    "recurrent_score_bonus": ("float", 0.00, 0.80, 0.12, "cart_recurrent_rules"),
    "allow_recurrent_rebuy": ("bool", "cart_recurrent_rules"),
    "v4_price_ascending": ("bool", "cart_recurrent_rules"),
    "v4_live_event_purchases": ("bool", "cart_recurrent_rules"),
    # 6. Diversity / MMR & Subcategory Floors
    "v3_no_subcat_floor": ("bool", "diversity_mmr"),
    "subcat_floor_min": ("float", 0.05, 0.55, 0.08, "diversity_mmr"),
    "brand_floor_min": ("float", 0.00, 0.30, 0.05, "diversity_mmr"),
    "mmr_start_slot_session": ("int", 3, 10, 2, "diversity_mmr"),
    "mmr_start_slot_home": ("int", 1, 8, 2, "diversity_mmr"),
    "mmr_subcat_lambda_session": ("float", 0.00, 0.35, 0.07, "diversity_mmr"),
    "mmr_subcat_lambda_home": ("float", 0.00, 0.30, 0.06, "diversity_mmr"),
    "mmr_brand_lambda": ("float", 0.00, 0.15, 0.03, "diversity_mmr"),
    "mmr_relative_penalty": ("bool", "diversity_mmr"),
    # 7. Bandit & Position Debiasing (M5)
    "bandit_ucb_alpha": ("float", 0.02, 0.30, 0.04, "bandit_ml"),
    "bandit_cold_ucb_multiplier": ("float", 0.80, 2.80, 0.30, "bandit_ml"),
    "bandit_warm_ucb_multiplier": ("float", 0.05, 0.60, 0.10, "bandit_ml"),
    "use_empirical_exam_curve": ("bool", "bandit_ml"),
    "bandit_exam_power": ("float", 0.45, 1.00, 0.10, "bandit_ml"),
}

_SEGMENT_ACTIVE_SIGNALS: dict[str, tuple[str, ...]] = {
    "new_with_views": ("seq", "ctx", "pop", "trend", "cold"),
    "new_no_views": ("ctx", "pop", "trend", "cold"),
    "known_with_views": ("seq", "intent", "aff", "ctx", "pop", "trend", "cold"),
    "known_with_history": ("seq", "intent", "aff", "ctx", "pop", "trend", "cold"),
    "known_cold": ("intent", "aff", "ctx", "pop", "trend", "cold"),
}

for _seg in USER_SEGMENTS:
    for _sig in _SEGMENT_ACTIVE_SIGNALS[_seg]:
        PARAM_SPECS[f"seg_{_seg}__{_sig}"] = ("float", 0.25, 2.00, 0.24, "segments")

PARAM_GROUPS: tuple[str, ...] = (
    "base_popularity_5_1",
    "session_core",
    "cart_recurrent_rules",
    "cold_start",
    "personalization_price",
    "diversity_mmr",
    "bandit_ml",
    "segments",
)


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        out = float(val)
        return default if math.isnan(out) else out
    except (TypeError, ValueError):
        return default


def get_param_value(wp: WeightsProfile, name: str) -> Any:
    """Read either a direct WeightsProfile field or a segment multiplier `seg_<seg>__<sig>`."""
    if name.startswith("seg_") and "__" in name:
        seg_part, sig_name = name[4:].split("__", 1)
        sig_idx = SEGMENT_SIGNAL_NAMES.index(sig_name)
        return float(getattr(wp, f"seg_{seg_part}")[sig_idx])
    return getattr(wp, name)


def apply_param_updates(wp: WeightsProfile, updates: dict[str, Any]) -> WeightsProfile:
    """Apply flat updates (including `seg_<seg>__<sig>`) cleanly onto a WeightsProfile."""
    direct: dict[str, Any] = {}
    seg_buffers: dict[str, list[float]] = {}
    for name, val in updates.items():
        if name.startswith("seg_") and "__" in name:
            seg_part, sig_name = name[4:].split("__", 1)
            field_name = f"seg_{seg_part}"
            if field_name not in seg_buffers:
                seg_buffers[field_name] = list(getattr(wp, field_name))
            sig_idx = SEGMENT_SIGNAL_NAMES.index(sig_name)
            seg_buffers[field_name][sig_idx] = round(float(val), 4)
        else:
            direct[name] = val
    for field_name, vals in seg_buffers.items():
        direct[field_name] = tuple(vals)

    # Keep coupled cold_dense_svd_weight synchronized if cold_sparse_cos_weight changed
    if "cold_sparse_cos_weight" in direct and "cold_dense_svd_weight" not in direct:
        direct["cold_dense_svd_weight"] = round(max(0.0, 1.0 - float(direct["cold_sparse_cos_weight"])), 4)
    return replace(wp, **direct)


def profile_signature(wp: WeightsProfile) -> str:
    """Return a deterministic hash of all tunable parameter values."""
    payload = {k: get_param_value(wp, k) for k in sorted(PARAM_SPECS)}
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _prepare_structural_tables(
    snap: RecommenderSnapshotV4,
    popularity_rows: list[dict[str, Any]] | None,
    transition_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Pre-index popularity and raw transitions so Stage 5.1 structural knobs update in <25ms."""
    raw_pop: dict[str, float] = {}
    hist_purchases: dict[str, float] = {}
    if popularity_rows:
        for row in popularity_rows:
            pid = str(row.get("product_id") or "").strip()
            if pid not in snap.active_ids:
                continue
            hist_purchases[pid] = max(_safe_float(row.get("purchases"), 0.0), 0.0)
            wpop = max(_safe_float(row.get("weighted_pop"), 0.0), 0.0)
            if wpop > 0.0:
                raw_pop[pid] = math.log1p(wpop)
        max_pop = max(raw_pop.values(), default=1.0) or 1.0
        log_pop_norm = {pid: v / max_pop for pid, v in raw_pop.items()}
    else:
        log_pop_norm = dict(snap.global_pop_norm)
        for pid, val in log_pop_norm.items():
            raw_pop[pid] = val * 6.0
            hist_purchases[pid] = max(0.0, math.expm1(val * 4.5))

    live_buys_equiv: dict[str, float] = {
        pid: 4.0 * max(val, 0.0) for pid, val in snap.live_trend_norm.items() if val > 0.0
    }

    trans_edges: list[tuple[str, str, float]] = []
    if transition_rows:
        for row in transition_rows:
            src = str(row.get("src_product") or "").strip()
            dst = str(row.get("dst_product") or "").strip()
            if not src or dst not in snap.active_ids or src == dst:
                continue
            sc = max(_safe_float(row.get("raw_transition_score", row.get("score")), 0.0), 0.0)
            if sc > 0.0:
                trans_edges.append((src, dst, sc))
    else:
        # Approximate raw score inverse from stored transitions when transition_rows aren't cached
        for src, edges in snap.transitions.items():
            for dst, sc in edges:
                if dst in snap.active_ids and sc > 0.0:
                    trans_edges.append((src, dst, float(sc)))

    return {
        "hist_purchases": hist_purchases,
        "raw_pop": raw_pop,
        "log_pop_norm": log_pop_norm,
        "live_buys_equiv": live_buys_equiv,
        "trans_edges": trans_edges,
        "has_raw_transitions": bool(transition_rows),
    }


def _recompute_5_1_structures(
    base_snap: RecommenderSnapshotV4,
    wp: WeightsProfile,
    cfg: ConfigV4,
    struct_prep: dict[str, Any],
) -> tuple[dict[str, float], dict[str, tuple[tuple[str, float], ...]], tuple[str, ...]]:
    """Fast exact recomputation of M3 base-rate, X10 neighbor prior, and D5 transition modulation."""
    active_ids = base_snap.active_ids
    raw_pop = struct_prep["raw_pop"]
    hist_purchases = struct_prep["hist_purchases"]
    log_pop_norm = struct_prep["log_pop_norm"]
    live_buys_equiv = struct_prep["live_buys_equiv"]

    mix_buys = {
        pid: hist_purchases.get(pid, 0.0) + wp.live_pop_mix_weight * live_buys_equiv.get(pid, 0.0)
        for pid in active_ids
    }
    if wp.use_sales_rank_base_rate and active_ids:
        ranked_by_sales = sorted(
            active_ids,
            key=lambda pid: (mix_buys[pid], raw_pop.get(pid, 0.0), pid),
            reverse=True,
        )
        denom_rank = math.log(max(len(ranked_by_sales) + 1, 100))
        blend_w = min(max(wp.sales_rank_blend_weight, 0.0), 1.0)
        global_pop_norm: dict[str, float] = {}
        for rank_idx, pid in enumerate(ranked_by_sales, start=1):
            has_signal = mix_buys[pid] > 0.0 or raw_pop.get(pid, 0.0) > 0.0
            sales_rank_norm = max(0.0, 1.0 - math.log(rank_idx) / denom_rank) if has_signal else 0.0
            global_pop_norm[pid] = (
                (1.0 - blend_w) * log_pop_norm.get(pid, 0.0) + blend_w * sales_rank_norm
            )
    else:
        global_pop_norm = dict(log_pop_norm)

    if wp.cold_neighbor_pop_prior_weight > 0.0 and base_snap.new_product_ids:
        for pid in base_snap.new_product_ids:
            if raw_pop.get(pid, 0.0) > 0.0:
                continue
            nbs = base_snap.cold_neighbors.get(pid)
            if not nbs:
                continue
            sim_sum = 0.0
            pop_sum = 0.0
            for warm_nb, sim in nbs:
                if sim > 0.0:
                    sim_sum += sim
                    pop_sum += sim * global_pop_norm.get(warm_nb, 0.0)
            if sim_sum > 0.0:
                prior_pop = (
                    wp.cold_neighbor_pop_prior_weight * (pop_sum / sim_sum)
                    + 0.30 * base_snap.live_trend_norm.get(pid, 0.0)
                )
                if prior_pop > global_pop_norm.get(pid, 0.0):
                    global_pop_norm[pid] = prior_pop

    gamma = min(max(wp.transition_dest_pop_gamma, 0.0), 1.0)
    raw_trans: dict[str, list[tuple[str, float]]] = defaultdict(list)
    use_raw = bool(struct_prep["has_raw_transitions"])
    for src, dst, base_sc in struct_prep["trans_edges"]:
        if use_raw:
            dst_pop = max(math.expm1(raw_pop.get(dst, 0.0)), 0.0)
            lift_score = base_sc / ((dst_pop + cfg.lift_prior_pop) ** cfg.lift_beta)
        else:
            lift_score = base_sc
        if gamma > 0.0:
            lift_score *= (1.0 - gamma) + gamma * global_pop_norm.get(dst, 0.25)
        raw_trans[src].append((dst, lift_score))

    transitions: dict[str, tuple[tuple[str, float], ...]] = {}
    for src, edges in raw_trans.items():
        edges.sort(key=lambda kv: kv[1], reverse=True)
        top_edges = edges[:35]
        max_s = top_edges[0][1] if top_edges else 1.0
        if max_s > 0.0:
            transitions[src] = tuple((dst, sc / max_s) for dst, sc in top_edges)

    fallback_scores: dict[str, float] = {}
    tie_mode = wp.tie_break_mode
    for pid in active_ids:
        p_obj = base_snap.products[pid]
        pop_s = global_pop_norm.get(pid, 0.0) + 0.6 * base_snap.live_trend_norm.get(pid, 0.0)
        lp_s = math.log1p(max(p_obj.price_eur, 0.0)) / base_snap.max_log_price
        if tie_mode == "cheap":
            tie_term = -abs(wp.revenue_tie_weight or -0.08) * lp_s
        elif tie_mode == "conversion":
            tie_term = 0.0
        else:
            tie_term = wp.revenue_tie_weight * lp_s
        fallback_scores[pid] = pop_s + tie_term + (0.02 if p_obj.is_new else 0.0)

    fallback_ranked = tuple(
        sorted(active_ids, key=lambda pid: (fallback_scores[pid], pid), reverse=True)
    )
    return global_pop_norm, transitions, fallback_ranked


def _update_snapshot_for_profile(
    base_snap: RecommenderSnapshotV4,
    wp: WeightsProfile,
    *,
    base_ucb_alpha: float,
    struct_prep: dict[str, Any] | None = None,
) -> tuple[RecommenderSnapshotV4, ConfigV4]:
    """Attach updated WeightsProfile, ConfigV4, ranker weights, 5.1 base-rate/transitions, and UCB cache."""
    cfg = ConfigV4.from_profile(wp, model_version=MODEL_VERSION_LABEL)
    base_ranker = base_snap.ranker
    pw = wp.prior_weights_array()
    if base_ranker is not None:
        alpha = base_ranker.ml_blend_alpha
        eff = (1.0 - alpha) * pw + alpha * base_ranker.ml_weights_raw
        cand_ranker = PositionDebiasedRanker(
            prior_weights=pw,
            effective_weights=eff,
            ml_weights_raw=base_ranker.ml_weights_raw.copy(),
            ml_blend_alpha=alpha,
            train_samples=base_ranker.train_samples,
            positive_samples=base_ranker.positive_samples,
            min_prior_ratios=wp.min_prior_ratios_array(),
            max_prior_multiplier=wp.max_prior_multiplier,
        )
    else:
        cand_ranker = PositionDebiasedRanker.from_priors(
            prior_weights=pw,
            min_prior_ratios=wp.min_prior_ratios_array(),
            max_prior_multiplier=wp.max_prior_multiplier,
        )

    base_wp = base_snap.weights_profile
    struct_changed = False
    global_pop_norm = base_snap.global_pop_norm
    transitions = base_snap.transitions
    fallback_ranked = base_snap.fallback_ranked

    if struct_prep is not None and (
        base_wp is None
        or base_wp.use_sales_rank_base_rate != wp.use_sales_rank_base_rate
        or abs(base_wp.sales_rank_blend_weight - wp.sales_rank_blend_weight) > 1e-6
        or abs(base_wp.live_pop_mix_weight - wp.live_pop_mix_weight) > 1e-6
        or abs(base_wp.cold_neighbor_pop_prior_weight - wp.cold_neighbor_pop_prior_weight) > 1e-6
        or abs(base_wp.transition_dest_pop_gamma - wp.transition_dest_pop_gamma) > 1e-6
        or base_wp.tie_break_mode != wp.tie_break_mode
        or abs(base_wp.revenue_tie_weight - wp.revenue_tie_weight) > 1e-6
    ):
        global_pop_norm, transitions, fallback_ranked = _recompute_5_1_structures(
            base_snap, wp, cfg, struct_prep
        )
        struct_changed = True

    product_static = base_snap.product_static
    overrides = base_snap.agent_overrides
    bandit_changed = (
        abs(wp.bandit_ucb_alpha - base_ucb_alpha) > 1e-9
        or (
            base_wp is not None
            and (
                abs(base_wp.bandit_cold_ucb_multiplier - wp.bandit_cold_ucb_multiplier) > 1e-6
                or abs(base_wp.bandit_warm_ucb_multiplier - wp.bandit_warm_ucb_multiplier) > 1e-6
                or base_wp.use_empirical_exam_curve != wp.use_empirical_exam_curve
                or abs(base_wp.bandit_exam_power - wp.bandit_exam_power) > 1e-6
            )
        )
    )
    if struct_changed or bandit_changed:
        bandit = PositionAwareBandit(
            exam_power=wp.bandit_exam_power,
            prior_strength=wp.bandit_prior_strength,
            cold_ucb_multiplier=wp.bandit_cold_ucb_multiplier,
            warm_ucb_multiplier=wp.bandit_warm_ucb_multiplier,
            click_reward=wp.bandit_click_reward,
            purchase_reward=wp.bandit_purchase_reward,
            use_empirical_exam_curve=wp.use_empirical_exam_curve,
        )
        product_static = build_product_static_table(
            base_snap.products,
            base_snap.active_ids,
            global_pop_norm,
            base_snap.live_trend_norm,
            base_snap.max_log_price,
            bandit,
            base_snap.bandit_state,
            wp.bandit_ucb_alpha,
        )
        if overrides is not None:
            overrides = replace(overrides, bandit_ucb_alpha=wp.bandit_ucb_alpha)

    snap = replace(
        base_snap,
        global_pop_norm=global_pop_norm,
        transitions=transitions,
        fallback_ranked=fallback_ranked,
        ranker=cand_ranker,
        weights_profile=wp,
        config_override=cfg,
        product_static=product_static,
        agent_overrides=overrides,
    )
    return snap, cfg


def _strip_future_test_purchases(
    snap: RecommenderSnapshotV4,
    df: pd.DataFrame,
) -> RecommenderSnapshotV4:
    """Remove ground-truth test purchases from bought history so offline real eval has no leakage."""
    buys = df[df["purchased"].notna() & (df["purchased"] != "")]
    if buys.empty:
        return snap
    leak_map: dict[str, set[str]] = {}
    for row in buys.itertuples(index=False):
        uid = str(getattr(row, "user_id", "") or "").strip()
        pid = str(getattr(row, "purchased", "") or "").strip()
        if uid and pid:
            leak_map.setdefault(uid, set()).add(pid)

    new_profiles = dict(snap.user_profiles)
    for uid, pids in leak_map.items():
        if uid in new_profiles:
            up = new_profiles[uid]
            new_profiles[uid] = replace(up, bought_products=up.bought_products - pids)

    new_live_bought = dict(snap.live_user_bought)
    for uid, pids in leak_map.items():
        if uid in new_live_bought:
            new_live_bought[uid] = new_live_bought[uid] - pids

    return replace(snap, user_profiles=new_profiles, live_user_bought=new_live_bought)


def _evaluate_real_weighted(
    policy_fn: Any,
    df_events: pd.DataFrame,
    df_bg: pd.DataFrame,
    bg_scale: float,
    cold_product_ids: frozenset[str],
) -> dict[str, float | int]:
    """Evaluate exact hit@10/dcg@10 on all real click/buy events + Horvitz-Thompson SNIPS."""
    df_eval = df_events if df_bg.empty else pd.concat([df_events, df_bg], ignore_index=True)
    n_ev = len(df_events)
    n = len(df_eval)
    if n == 0:
        return {}

    t0 = time.perf_counter()
    recs = [policy_fn(row) for row in df_eval.itertuples(index=False)]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    clicked = np.array(
        [str(x) if (x is not None and x != "" and not (isinstance(x, float) and np.isnan(x))) else "" for x in df_eval["clicked"]],
        dtype=object,
    )
    purchased = np.array(
        [str(x) if (x is not None and x != "" and not (isinstance(x, float) and np.isnan(x))) else "" for x in df_eval["purchased"]],
        dtype=object,
    )
    has_click = clicked != ""
    has_buy = purchased != ""

    hit_click_arr = np.array([c in r for c, r in zip(clicked, recs)], dtype=float)
    hit_buy_arr = np.array([b in r for b, r in zip(purchased, recs)], dtype=float)

    dcg_buy_arr = np.zeros(n, dtype=float)
    for i, (b, r) in enumerate(zip(purchased, recs)):
        if b and b in r:
            dcg_buy_arr[i] = 1.0 / np.log2(r.index(b) + 2.0)

    first_rec = np.array([r[0] for r in recs], dtype=object)
    first_shown = np.array([s[0] if len(s) > 0 else "" for s in df_eval["shown"]], dtype=object)
    prop1 = np.array(
        [max(float(p[0]), 1e-4) if len(p) > 0 else 1.0 for p in df_eval["propensity"]],
        dtype=float,
    )

    sample_mult = np.ones(n, dtype=float)
    if n > n_ev and bg_scale > 1.0:
        sample_mult[n_ev:] = bg_scale

    match = (first_rec == first_shown).astype(float)
    w = (match / prop1) * sample_mult
    w_sum = float(w.sum())

    r_click = (clicked == first_shown).astype(float)
    r_buy = (purchased == first_shown).astype(float)
    n_total_equiv = float(n_ev + len(df_bg) * bg_scale)

    snips_cvr = float((w * r_buy).sum() / max(w_sum, 1e-9))
    snips_ctr = float((w * r_click).sum() / max(w_sum, 1e-9))

    return {
        "n_evaluated": int(n),
        "n_equivalent": int(round(n_total_equiv)),
        "n_clicks": int(has_click.sum()),
        "n_buys": int(has_buy.sum()),
        "hit_rate_click@10": float(hit_click_arr[has_click].mean()) if has_click.any() else 0.0,
        "hit_rate_buy@10": float(hit_buy_arr[has_buy].mean()) if has_buy.any() else 0.0,
        "dcg_buy@10": float(dcg_buy_arr[has_buy].mean()) if has_buy.any() else 0.0,
        "slot1_matches": int(match.sum()),
        "IPS_ctr_slot1": float((w * r_click).sum() / max(n_total_equiv, 1.0)),
        "SNIPS_ctr_slot1": snips_ctr,
        "IPS_cvr_slot1": float((w * r_buy).sum() / max(n_total_equiv, 1.0)),
        "SNIPS_cvr_slot1": snips_cvr,
        "latency_ms_mean": float(elapsed_ms / n),
    }


def load_real_eval_bundle(
    model_root: Path,
    offline_cache_dir: Path,
    *,
    bg_sample_size: int = 2500,
    full_real: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Load real store snapshot (leakage-stripped) and official_test_data.parquet."""
    data_file = offline_cache_dir / "official_test_data.parquet"
    best_file = offline_cache_dir / "best_sellers.json"
    if not data_file.is_file():
        raise FileNotFoundError(f"Missing real offline dataset: {data_file}")

    dc = DiskCache(model_root / "cache")
    snap = dc.load_snapshot()
    if snap is None:
        raise FileNotFoundError(f"Missing snapshot cache in {model_root / 'cache'}")

    static_hist = dc.load_static_history()
    struct_prep = _prepare_structural_tables(
        snap,
        getattr(static_hist, "popularity_rows", None) if static_hist else None,
        getattr(static_hist, "transition_rows", None) if static_hist else None,
    )

    df = pd.read_parquet(data_file)
    if "is_new_user" not in df.columns:
        df["is_new_user"] = ~df["user_id"].isin(snap.user_profiles)

    snap_clean = _strip_future_test_purchases(snap, df)

    has_click = df["clicked"].notna() & (df["clicked"] != "")
    has_buy = df["purchased"].notna() & (df["purchased"] != "")
    ev_mask = has_click | has_buy
    df_events = df[ev_mask].reset_index(drop=True)
    df_bg_all = df[~ev_mask].reset_index(drop=True)

    if full_real or bg_sample_size <= 0 or bg_sample_size >= len(df_bg_all):
        df_bg = df_bg_all
        bg_scale = 1.0
    else:
        df_bg = df_bg_all.sample(n=bg_sample_size, random_state=seed).reset_index(drop=True)
        bg_scale = float(len(df_bg_all)) / float(len(df_bg))

    best_sellers = json.loads(best_file.read_text(encoding="utf-8")) if best_file.is_file() else []
    baseline_metrics = {}
    if best_sellers:
        top10 = best_sellers[:10]
        baseline_metrics = _evaluate_real_weighted(
            lambda _row: top10,
            df_events,
            df_bg,
            bg_scale,
            snap_clean.new_product_ids,
        )

    return {
        "snap": snap_clean,
        "base_ucb_alpha": float(snap_clean.agent_overrides.bandit_ucb_alpha),
        "struct_prep": struct_prep,
        "df_events": df_events,
        "df_bg": df_bg,
        "bg_scale": bg_scale,
        "baseline_metrics": baseline_metrics,
        "best_sellers": best_sellers,
    }


def load_synthetic_eval_bundle(
    *,
    seed: int = 42,
    n_test: int = 1200,
) -> dict[str, Any]:
    """Generate stratified synthetic dataset (ALL, WARM, COLD_START) once for fast tuning."""
    syn = generate_synthetic_dataset_v4(
        seed=seed,
        n_warm_products=400,
        n_new_products=35,
        n_warm_users=900,
        n_history_sessions=3000,
        n_feedback_sessions=800,
        n_test_sessions=n_test,
    )
    snap = syn["snap_v4"]
    v4_rec = syn.get("v4_records") or {}
    struct_prep = _prepare_structural_tables(
        snap,
        v4_rec.get("popularity_rows"),
        v4_rec.get("transition_rows"),
    )
    df_test = syn["df_test"]
    cold_ids = syn["cold_product_ids"]
    warm_users = syn["warm_user_ids"]
    cold_mask = np.array(
        [is_cold_start_row(row, cold_ids, warm_users) for row in df_test.itertuples(index=False)],
        dtype=bool,
    )
    return {
        "snap": snap,
        "base_ucb_alpha": float(snap.agent_overrides.bandit_ucb_alpha),
        "struct_prep": struct_prep,
        "df_all": df_test,
        "df_warm": df_test[~cold_mask],
        "df_cold": df_test[cold_mask],
        "cold_product_ids": cold_ids,
    }


def compute_conversion_objective(
    real_metrics: dict[str, Any] | None,
    syn_metrics: dict[str, Any] | None,
    eval_mode: str,
) -> float:
    """Compute primary purchase-conversion objective across active evaluation slices."""
    real_score = 0.0
    if real_metrics:
        r_hit = float(real_metrics.get("hit_rate_buy@10", 0.0))
        r_dcg = float(real_metrics.get("dcg_buy@10", 0.0))
        r_clk = float(real_metrics.get("hit_rate_click@10", 0.0))
        r_snips = min(float(real_metrics.get("SNIPS_cvr_slot1", 0.0)) * 5.0, 1.0)
        real_score = 0.45 * r_hit + 0.35 * r_dcg + 0.12 * r_clk + 0.08 * r_snips

    syn_score = 0.0
    if syn_metrics:
        a = syn_metrics["ALL"]
        c = syn_metrics["COLD_START"]
        s_hit = float(a.get("hit_rate_buy@10", 0.0))
        s_dcg = float(a.get("dcg_buy@10", 0.0))
        s_cold = float(c.get("hit_rate_buy@10", 0.0))
        s_snips = float(a.get("SNIPS_cvr_slot1", 0.0))
        syn_score = 0.45 * s_hit + 0.35 * s_dcg + 0.12 * s_cold + 0.08 * s_snips

    if eval_mode == "real":
        return real_score
    if eval_mode == "synthetic":
        return syn_score
    return 0.65 * real_score + 0.35 * syn_score


def evaluate_candidate(
    wp: WeightsProfile,
    *,
    eval_mode: str,
    real_bundle: dict[str, Any] | None,
    syn_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate a WeightsProfile across real and/or synthetic offline datasets."""
    t0 = time.perf_counter()
    real_metrics: dict[str, Any] | None = None
    syn_metrics: dict[str, Any] | None = None

    if real_bundle is not None and eval_mode in {"real", "hybrid"}:
        snap_r, cfg_r = _update_snapshot_for_profile(
            real_bundle["snap"],
            wp,
            base_ucb_alpha=real_bundle["base_ucb_alpha"],
            struct_prep=real_bundle.get("struct_prep"),
        )
        rec_r = CalypsoRecommenderV4(cfg_r)
        real_metrics = _evaluate_real_weighted(
            lambda row: rec_r.recommend(row, snap_r),
            real_bundle["df_events"],
            real_bundle["df_bg"],
            real_bundle["bg_scale"],
            snap_r.new_product_ids,
        )
        base_m = real_bundle.get("baseline_metrics") or {}
        if base_m.get("hit_rate_buy@10", 0.0) > 0:
            real_metrics["vs_baseline_hit_ratio"] = round(
                float(real_metrics["hit_rate_buy@10"]) / float(base_m["hit_rate_buy@10"]), 5
            )
        if base_m.get("dcg_buy@10", 0.0) > 0:
            real_metrics["vs_baseline_dcg_ratio"] = round(
                float(real_metrics["dcg_buy@10"]) / float(base_m["dcg_buy@10"]), 5
            )

    if syn_bundle is not None and eval_mode in {"synthetic", "hybrid"}:
        snap_s, cfg_s = _update_snapshot_for_profile(
            syn_bundle["snap"],
            wp,
            base_ucb_alpha=syn_bundle["base_ucb_alpha"],
            struct_prep=syn_bundle.get("struct_prep"),
        )
        rec_s = CalypsoRecommenderV4(cfg_s)
        cold_ids = syn_bundle["cold_product_ids"]
        syn_metrics = {
            "ALL": evaluate_policy(lambda row: rec_s.recommend(row, snap_s), syn_bundle["df_all"], cold_product_ids=cold_ids),
            "WARM": evaluate_policy(lambda row: rec_s.recommend(row, snap_s), syn_bundle["df_warm"], cold_product_ids=cold_ids),
            "COLD_START": evaluate_policy(lambda row: rec_s.recommend(row, snap_s), syn_bundle["df_cold"], cold_product_ids=cold_ids),
        }

    obj = compute_conversion_objective(real_metrics, syn_metrics, eval_mode)
    return {
        "objective": round(float(obj), 6),
        "eval_time_s": round(time.perf_counter() - t0, 3),
        "real": real_metrics,
        "synthetic": syn_metrics,
    }


def compute_adaptive_intelligence(
    iterations: list[dict[str, Any]],
    active_groups: tuple[str, ...] = PARAM_GROUPS,
) -> dict[str, Any]:
    """Learn per-parameter and per-group UCB priorities from prior iteration outcomes."""
    p_trials: dict[str, int] = defaultdict(int)
    p_wins: dict[str, int] = defaultdict(int)
    p_pos_gain: dict[str, float] = defaultdict(float)
    p_abs_impact: dict[str, float] = defaultdict(float)

    g_trials: dict[str, int] = defaultdict(int)
    g_wins: dict[str, int] = defaultdict(int)
    g_pos_gain: dict[str, float] = defaultdict(float)
    g_abs_impact: dict[str, float] = defaultdict(float)

    prev_best = float(iterations[0]["objective"]) if iterations else 0.0
    for it in iterations[1:]:
        changed = it.get("changed_params") or {}
        if not changed:
            continue
        obj = float(it.get("objective", prev_best))
        accepted = bool(it.get("accepted", False))
        gain = max(0.0, obj - prev_best) if accepted else 0.0
        sensitivity = abs(obj - prev_best)

        touched_groups: set[str] = set()
        k_n = max(len(changed), 1)
        for pname in changed:
            if pname not in PARAM_SPECS:
                continue
            grp = str(PARAM_SPECS[pname][-1])
            touched_groups.add(grp)
            p_trials[pname] += 1
            p_abs_impact[pname] += sensitivity / k_n
            if accepted:
                p_wins[pname] += 1
                p_pos_gain[pname] += gain / k_n

        g_n = max(len(touched_groups), 1)
        for grp in touched_groups:
            g_trials[grp] += 1
            g_abs_impact[grp] += sensitivity / g_n
            if accepted:
                g_wins[grp] += 1
                g_pos_gain[grp] += gain / g_n

        if accepted and obj > prev_best:
            prev_best = obj

    total_p_trials = sum(p_trials.values()) + 1
    total_g_trials = sum(g_trials.values()) + 1

    group_scores: dict[str, float] = {}
    for grp in active_groups:
        n_t = g_trials[grp]
        win_rate = g_wins[grp] / max(n_t, 1)
        mean_gain = (g_pos_gain[grp] * 120.0) / max(n_t, 1)
        mean_sens = min((g_abs_impact[grp] * 18.0) / max(n_t, 1), 0.65)
        explore = 0.85 * math.sqrt(math.log(total_g_trials + 1.0) / (n_t + 1.0))
        group_scores[grp] = max(0.15, 0.35 + win_rate * 1.4 + mean_gain + mean_sens + explore)

    param_scores: dict[str, float] = {}
    for pname, spec in PARAM_SPECS.items():
        if spec[-1] not in active_groups:
            continue
        n_t = p_trials[pname]
        win_rate = p_wins[pname] / max(n_t, 1)
        mean_gain = (p_pos_gain[pname] * 140.0) / max(n_t, 1)
        mean_sens = min((p_abs_impact[pname] * 22.0) / max(n_t, 1), 0.60)
        explore = 0.90 * math.sqrt(math.log(total_p_trials + 1.0) / (n_t + 1.0))
        param_scores[pname] = max(0.15, 0.30 + win_rate * 1.6 + mean_gain + mean_sens + explore)

    top_learned = sorted(
        (
            {
                "param": k,
                "group": PARAM_SPECS[k][-1],
                "trials": p_trials[k],
                "wins": p_wins[k],
                "ucb_score": round(v, 4),
            }
            for k, v in param_scores.items()
            if p_trials[k] > 0
        ),
        key=lambda r: (r["wins"], r["ucb_score"]),
        reverse=True,
    )[:12]

    return {
        "group_scores": group_scores,
        "param_scores": param_scores,
        "group_trials": dict(g_trials),
        "group_wins": dict(g_wins),
        "top_learned_params": top_learned,
    }


def _weighted_sample(
    rng: np.random.Generator,
    candidates: list[str],
    scores: dict[str, float],
    k: int,
) -> list[str]:
    if k >= len(candidates):
        return list(candidates)
    weights = np.array([max(float(scores.get(c, 0.25)), 1e-4) for c in candidates], dtype=float)
    probs = weights / weights.sum()
    return [str(x) for x in rng.choice(candidates, size=k, replace=False, p=probs)]


def _propose_candidate(
    best_wp: WeightsProfile,
    rng: np.random.Generator,
    step_idx: int,
    total_iters: int,
    seen_sigs: set[str],
    last_accepted_delta: dict[str, Any] | None,
    intel: dict[str, Any],
    active_groups: tuple[str, ...],
) -> tuple[WeightsProfile, str, dict[str, dict[str, Any]]]:
    """Generate a non-duplicate candidate WeightsProfile guided by adaptive UCB parameter priorities."""
    frac = step_idx / max(total_iters, 1)
    group_scores = intel["group_scores"]
    param_scores = intel["param_scores"]
    active_pool = [k for k, v in PARAM_SPECS.items() if v[-1] in active_groups]

    for attempt in range(25):
        updates: dict[str, Any] = {}

        if attempt == 0 and last_accepted_delta and rng.random() < 0.30:
            strategy = "momentum"
            for pname, dval in last_accepted_delta.items():
                spec = PARAM_SPECS[pname]
                if spec[0] == "float":
                    _, lo, hi, _, _ = spec
                    cur = float(get_param_value(best_wp, pname))
                    updates[pname] = round(min(max(cur + float(dval) * 0.65, lo), hi), 4)
        else:
            picked_grp = _weighted_sample(rng, list(active_groups), group_scores, 1)[0]
            if frac <= 0.30:
                mode = "explore_broad" if rng.random() < 0.45 else f"group:{picked_grp}"
            elif frac <= 0.75:
                mode = f"group:{picked_grp}"
            else:
                mode = "fine_tune"

            strategy = mode
            if mode == "explore_broad":
                pool = active_pool
                n_pick = int(rng.integers(3, 6))
                temp = 1.10
            elif mode.startswith("group:"):
                grp = mode.split(":", 1)[1]
                pool = [k for k in active_pool if PARAM_SPECS[k][-1] == grp]
                n_pick = int(rng.integers(2, min(4, len(pool)) + 1))
                temp = max(0.45, 1.0 - 0.5 * frac)
            else:
                pool = active_pool
                n_pick = int(rng.integers(1, 3))
                temp = 0.40

            chosen = _weighted_sample(rng, pool, param_scores, min(n_pick, len(pool)))
            for pname in chosen:
                spec = PARAM_SPECS[pname]
                cur = get_param_value(best_wp, pname)
                if spec[0] == "float":
                    _, lo, hi, scale, _ = spec
                    new_v = round(min(max(float(cur) + float(rng.normal(0.0, scale * temp)), lo), hi), 4)
                    if abs(new_v - float(cur)) > 1e-6:
                        updates[pname] = new_v
                elif spec[0] == "int":
                    _, lo, hi, scale, _ = spec
                    delta = int(round(float(rng.normal(0.0, max(1.0, scale * temp)))))
                    if delta == 0:
                        delta = int(rng.choice([-1, 1]))
                    new_v = int(min(max(int(cur) + delta, lo), hi))
                    if new_v != int(cur):
                        updates[pname] = new_v
                elif spec[0] == "bool":
                    updates[pname] = not bool(cur)
                elif spec[0] == "choice":
                    opts = [o for o in spec[1] if o != cur]
                    if opts:
                        updates[pname] = str(rng.choice(opts))

        if not updates:
            continue

        cand_wp = apply_param_updates(best_wp, updates)
        sig = profile_signature(cand_wp)
        if sig not in seen_sigs:
            changed_diff = {
                k: {"old": get_param_value(best_wp, k), "new": get_param_value(cand_wp, k)}
                for k in updates
                if get_param_value(best_wp, k) != get_param_value(cand_wp, k)
            }
            if changed_diff:
                return cand_wp, strategy, changed_diff

    pname = str(rng.choice(active_pool))
    spec = PARAM_SPECS[pname]
    cur = get_param_value(best_wp, pname)
    if spec[0] == "float":
        _, lo, hi, _, _ = spec
        cand_val: Any = round(float(rng.uniform(lo, hi)), 4)
    elif spec[0] == "int":
        _, lo, hi, _, _ = spec
        cand_val = int(rng.integers(lo, hi + 1))
    elif spec[0] == "bool":
        cand_val = not bool(cur)
    else:
        cand_val = str(rng.choice(spec[1]))
    cand_wp = apply_param_updates(best_wp, {pname: cand_val})
    return cand_wp, "fallback_jump", {pname: {"old": cur, "new": cand_val}}


def _compact_metrics_line(res: dict[str, Any]) -> str:
    parts = [f"obj={res['objective']:.5f}"]
    r = res.get("real")
    if r:
        parts.append(
            f"real_buy@10={r['hit_rate_buy@10']:.4f} real_dcg={r['dcg_buy@10']:.4f} "
            f"real_clk@10={r['hit_rate_click@10']:.4f} snips_cvr={r['SNIPS_cvr_slot1']:.4f}"
        )
    s = res.get("synthetic")
    if s:
        parts.append(
            f"syn_buy@10={s['ALL']['hit_rate_buy@10']:.4f} syn_dcg={s['ALL']['dcg_buy@10']:.4f} "
            f"cold_buy@10={s['COLD_START']['hit_rate_buy@10']:.4f}"
        )
    parts.append(f"({res['eval_time_s']:.2f}s)")
    return " | ".join(parts)


def run_odyseus_optimization(
    model_root: Path,
    *,
    base_profile: WeightsProfile,
    history_path: Path,
    eval_mode: str = "hybrid",
    iters: int = 25,
    seed: int = 42,
    n_syn: int = 1200,
    real_bg_sample: int = 2500,
    full_real: bool = False,
    eval_presets_first: bool = True,
    resume: bool = False,
    save_best: bool = False,
    target_profile_path: Path | None = None,
    offline_cache_dir: Path | None = None,
    focus_groups: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run Odyseus adaptive parameter search and persist every iteration to a single archive file."""
    repo_root = model_root.parent
    off_dir = offline_cache_dir or (repo_root / "offline_eval_cache")
    active_groups = tuple(g for g in (focus_groups or PARAM_GROUPS) if g in PARAM_GROUPS) or PARAM_GROUPS

    real_bundle = (
        load_real_eval_bundle(
            model_root,
            off_dir,
            bg_sample_size=real_bg_sample,
            full_real=full_real,
            seed=seed,
        )
        if eval_mode in {"real", "hybrid"}
        else None
    )
    syn_bundle = (
        load_synthetic_eval_bundle(seed=seed, n_test=n_syn)
        if eval_mode in {"synthetic", "hybrid"}
        else None
    )

    rng = np.random.default_rng(seed)
    seen_sigs: set[str] = set()
    iterations: list[dict[str, Any]] = []

    if resume and history_path.is_file():
        existing = json.loads(history_path.read_text(encoding="utf-8"))
        iterations = list(existing.get("iterations") or [])
        for it in iterations:
            wp_prev = WeightsProfile.from_dict(it["params"], base=base_profile)
            seen_sigs.add(profile_signature(wp_prev))
        best_entry = max(iterations, key=lambda x: float(x["objective"]))
        best_wp = WeightsProfile.from_dict(best_entry["params"], base=base_profile)
        best_obj = float(best_entry["objective"])
        best_iter = int(best_entry["iter"])
        best_metrics = best_entry["metrics"]
        baseline_obj = float(existing.get("baseline_objective", iterations[0]["objective"]))
        created_at = str(existing.get("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds")))
        start_iter = max(int(it["iter"]) for it in iterations) + 1
        print(f"[ODYSEUS] Resuming from {history_path} ({len(iterations)} prior iters, best obj={best_obj:.5f})")
    else:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        best_wp = base_profile
        best_obj = -1.0
        best_iter = 0
        best_metrics = {}
        baseline_obj = 0.0
        start_iter = 0

    def _record_iteration(
        iter_num: int,
        wp: WeightsProfile,
        strategy: str,
        changed_params: dict[str, Any],
        res: dict[str, Any],
    ) -> bool:
        nonlocal best_wp, best_obj, best_iter, best_metrics, baseline_obj
        obj = float(res["objective"])
        if iter_num == 0 and not iterations:
            baseline_obj = obj
        accepted = obj > best_obj + 1e-6
        if accepted:
            best_wp = wp
            best_obj = obj
            best_iter = iter_num
            best_metrics = {"real": res["real"], "synthetic": res["synthetic"]}

        entry = {
            "iter": iter_num,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy": strategy,
            "accepted": accepted,
            "objective": obj,
            "best_objective_so_far": best_obj,
            "eval_time_s": res["eval_time_s"],
            "changed_params": changed_params,
            "metrics": {"real": res["real"], "synthetic": res["synthetic"]},
            "params": wp.to_dict(),
        }
        iterations.append(entry)
        seen_sigs.add(profile_signature(wp))

        intel_now = compute_adaptive_intelligence(iterations, active_groups=active_groups)
        archive = {
            "model": MODEL_VERSION_LABEL,
            "eval_mode": eval_mode,
            "selector_mode": "adaptive_ucb_bandit",
            "objective_formula": (
                "0.65*real_conv + 0.35*syn_conv"
                if eval_mode == "hybrid"
                else ("real_conv" if eval_mode == "real" else "syn_conv")
            ),
            "created_at": created_at,
            "updated_at": entry["timestamp"],
            "total_iterations": len(iterations),
            "baseline_objective": baseline_obj,
            "real_baseline_reference": (real_bundle["baseline_metrics"] if real_bundle else None),
            "best_iter": best_iter,
            "best_objective": best_obj,
            "best_metrics": best_metrics,
            "best_params": best_wp.to_dict(),
            "selector_summary": {
                "active_groups": list(active_groups),
                "group_trials": intel_now["group_trials"],
                "group_wins": intel_now["group_wins"],
                "top_learned_params": intel_now["top_learned_params"],
            },
            "iterations": iterations,
        }
        _atomic_write_json(history_path, archive)

        tag = "BEST*" if accepted else "     "
        print(f"[iter {iter_num:03d}] {tag} [{strategy:<25}] {_compact_metrics_line(res)}")
        return accepted

    cur_iter = start_iter
    if not iterations:
        res0 = evaluate_candidate(base_profile, eval_mode=eval_mode, real_bundle=real_bundle, syn_bundle=syn_bundle)
        _record_iteration(cur_iter, base_profile, f"baseline:{base_profile.profile_name}", {}, res0)
        cur_iter += 1

        if eval_presets_first:
            for pname in list_presets():
                p_wp = get_preset(pname)
                if profile_signature(p_wp) in seen_sigs:
                    continue
                res_p = evaluate_candidate(p_wp, eval_mode=eval_mode, real_bundle=real_bundle, syn_bundle=syn_bundle)
                diff = {
                    k: {"old": get_param_value(best_wp, k), "new": get_param_value(p_wp, k)}
                    for k in PARAM_SPECS
                    if get_param_value(best_wp, k) != get_param_value(p_wp, k)
                }
                _record_iteration(cur_iter, p_wp, f"preset:{pname}", diff, res_p)
                cur_iter += 1

    last_accepted_delta: dict[str, Any] | None = None
    for step in range(1, max(iters, 0) + 1):
        intel = compute_adaptive_intelligence(iterations, active_groups=active_groups)
        cand_wp, strategy, changed_diff = _propose_candidate(
            best_wp,
            rng,
            step,
            iters,
            seen_sigs,
            last_accepted_delta,
            intel,
            active_groups,
        )
        res_c = evaluate_candidate(cand_wp, eval_mode=eval_mode, real_bundle=real_bundle, syn_bundle=syn_bundle)
        accepted = _record_iteration(cur_iter, cand_wp, strategy, changed_diff, res_c)
        if accepted:
            delta_map: dict[str, Any] = {}
            for pname, d in changed_diff.items():
                if PARAM_SPECS[pname][0] == "float":
                    delta_map[pname] = float(d["new"]) - float(d["old"])
            last_accepted_delta = delta_map or None
        else:
            last_accepted_delta = None
        cur_iter += 1

    final_name = (
        best_wp.profile_name
        if best_wp.profile_name.endswith("_odyseus_opt")
        else f"{best_wp.profile_name}_odyseus_opt"
    )
    final_wp = replace(
        best_wp,
        profile_name=final_name,
        description=(
            f"Odyseus ({MODEL_VERSION_LABEL}) best profile "
            f"(iter={best_iter}, mode={eval_mode}, obj={best_obj:.5f} vs baseline={baseline_obj:.5f})."
        ),
    )

    if save_best and target_profile_path is not None:
        final_wp.save_json(target_profile_path)
        print(f"[ODYSEUS] Saved optimal profile '{final_wp.profile_name}' -> {target_profile_path}")

    print(
        f"\n=== Odyseus Optimization Summary ({MODEL_VERSION_LABEL}) ===\n"
        f"History archive : {history_path}\n"
        f"Total iterations: {len(iterations)} (best at iter {best_iter})\n"
        f"Baseline obj    : {baseline_obj:.6f}\n"
        f"Best obj        : {best_obj:.6f} ({best_obj - baseline_obj:+.6f})"
    )
    return {
        "history_path": str(history_path),
        "best_iter": best_iter,
        "best_objective": best_obj,
        "baseline_objective": baseline_obj,
        "best_profile": final_wp,
        "total_iterations": len(iterations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Odyseus Offline Iterative Conversion Optimizer (calypso-5.1f)")
    parser.add_argument(
        "--eval-mode",
        choices=("hybrid", "real", "synthetic"),
        default="hybrid",
        help="Offline dataset mode: hybrid (real+synthetic), real (official_test_data), or synthetic",
    )
    parser.add_argument("--iters", type=int, default=20, help="Number of optimization iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-syn", type=int, default=1200, help="Synthetic test requests")
    parser.add_argument(
        "--real-bg-sample",
        type=int,
        default=2500,
        help="Background non-event rows sampled alongside ALL real click/buy rows",
    )
    parser.add_argument("--full-real", action="store_true", help="Evaluate all 25,000 real rows every iteration")
    parser.add_argument("--preset", type=str, default="", help="Start from a specific preset name")
    parser.add_argument(
        "--no-preset-warmup",
        action="store_true",
        help="Skip initial built-in preset sweep before adaptive search",
    )
    parser.add_argument("--resume", action="store_true", help="Resume appending to existing history archive")
    parser.add_argument(
        "--history-file",
        type=str,
        default="",
        help="Path to single JSON archive file (default: odyseus/results/odyseus_history.json)",
    )
    parser.add_argument(
        "--focus-groups",
        nargs="+",
        choices=PARAM_GROUPS,
        default=None,
        help="Optional subset of parameter groups to optimize",
    )
    parser.add_argument("--save-best", action="store_true", help="Write best profile to weights_profile.json")
    args = parser.parse_args()

    model_root = Path(__file__).resolve().parents[2]
    cfg = ConfigV4()
    target_profile = cfg.resolve_weights_profile_path()
    base_wp = get_preset(args.preset) if args.preset else load_active_profile(target_profile)
    history_path = Path(args.history_file) if args.history_file else (model_root / "results" / "odyseus_history.json")
    if not history_path.is_absolute():
        history_path = model_root / history_path

    run_odyseus_optimization(
        model_root,
        base_profile=base_wp,
        history_path=history_path,
        eval_mode=args.eval_mode,
        iters=args.iters,
        seed=args.seed,
        n_syn=args.n_syn,
        real_bg_sample=args.real_bg_sample,
        full_real=args.full_real,
        eval_presets_first=not args.no_preset_warmup,
        resume=args.resume,
        save_best=args.save_best,
        target_profile_path=target_profile,
        focus_groups=tuple(args.focus_groups) if args.focus_groups else None,
    )


if __name__ == "__main__":
    main()
