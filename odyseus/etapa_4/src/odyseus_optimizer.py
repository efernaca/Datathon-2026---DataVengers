"""Odyseus iterative offline parameter optimizer for Calypso 5.1f targeting max conversion."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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

from .bandits import EMPIRICAL_SLOT_WEIGHTS, PositionAwareBandit
from .config import ConfigV4
from .disk_cache import DiskCache
from .ease import fit_ease_from_transitions
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
    "w_seq_lift": ("float", 0.20, 4.20, 0.28, "session_core"),
    "w_self_cvr_view": ("float", 0.40, 5.50, 0.35, "session_core"),
    "w_intent_score": ("float", 0.05, 2.20, 0.22, "session_core"),
    "w_same_top_subcat": ("float", 0.00, 0.80, 0.08, "session_core"),
    "w_same_top_brand": ("float", 0.00, 0.50, 0.05, "session_core"),
    "view_last_product_page_factor": ("float", 0.35, 1.40, 0.14, "session_core"),
    "view_base_bonus": ("float", 0.00, 1.10, 0.14, "session_core"),
    "legacy_self_view_linear": ("bool", "session_core"),
    "recency_decay": ("float", 0.40, 0.92, 0.08, "session_core"),
    "slot1_category_match_bonus": ("float", 0.00, 0.65, 0.08, "session_core"),
    "same_cat_bonus_weight": ("float", 0.00, 0.30, 0.04, "session_core"),
    "lift_beta": ("float", 0.15, 0.85, 0.08, "session_core"),
    "lift_prior_pop": ("float", 5.0, 80.0, 10.0, "session_core"),
    # 2. Calypso 5.1 Base-Rate Calibration, Pool Guarantees & Live Window
    "w_ctx_bayes": ("float", 0.05, 1.40, 0.14, "base_popularity_5_1"),
    "w_global_pop": ("float", 0.10, 6.50, 0.40, "base_popularity_5_1"),
    "w_live_trend": ("float", 0.02, 1.20, 0.14, "base_popularity_5_1"),
    "use_sales_rank_base_rate": ("bool", "base_popularity_5_1"),
    "sales_rank_blend_weight": ("float", 0.05, 1.00, 0.12, "base_popularity_5_1"),
    "live_pop_mix_weight": ("float", 0.00, 1.00, 0.14, "base_popularity_5_1"),
    "fallback_live_trend_weight": ("float", 0.00, 0.85, 0.12, "base_popularity_5_1"),
    "transition_dest_pop_gamma": ("float", 0.00, 0.90, 0.14, "base_popularity_5_1"),
    "candidate_pool_size": ("int", 50, 160, 14, "base_popularity_5_1"),
    "guaranteed_top_sellers_in_pool": ("int", 8, 65, 8, "base_popularity_5_1"),
    "v3_bg_penalty": ("bool", "base_popularity_5_1"),
    "bg_penalty_diff_subcat": ("float", 0.08, 1.00, 0.16, "base_popularity_5_1"),
    "bg_penalty_same_subcat": ("float", 0.15, 1.00, 0.16, "base_popularity_5_1"),
    # 3. Calibrated Logit & Hybrid Scoring (MEDIDAS_CONVERSION + 5.1f)
    "scoring_mode": ("choice", ("legacy", "calibrated", "hybrid"), "calibrated_logit"),
    "cal_w_rank": ("float", 0.25, 1.40, 0.09, "calibrated_logit"),
    "cal_w_seen": ("float", 0.70, 4.00, 0.24, "calibrated_logit"),
    "cal_w_rel": ("float", 0.10, 2.00, 0.15, "calibrated_logit"),
    "cal_rel_cap": ("float", 0.40, 2.50, 0.25, "calibrated_logit"),
    "cal_w_cart_active": ("float", 0.60, 4.20, 0.28, "calibrated_logit"),
    "cal_w_cart_forgotten": ("float", 0.00, 2.00, 0.20, "calibrated_logit"),
    "cal_w_recurrent": ("float", 0.00, 1.50, 0.16, "calibrated_logit"),
    "cal_w_intent": ("float", 0.00, 1.60, 0.16, "calibrated_logit"),
    "cal_w_aff": ("float", 0.00, 1.20, 0.14, "calibrated_logit"),
    "cal_w_pcompat": ("float", 0.00, 0.80, 0.10, "calibrated_logit"),
    "cal_w_ctx": ("float", 0.00, 1.20, 0.14, "calibrated_logit"),
    "cal_w_cold": ("float", 0.00, 1.50, 0.16, "calibrated_logit"),
    "cal_w_subcat": ("float", 0.00, 0.65, 0.08, "calibrated_logit"),
    "cal_w_brand": ("float", 0.00, 0.45, 0.06, "calibrated_logit"),
    "cal_w_live_trend": ("float", 0.00, 1.20, 0.14, "calibrated_logit"),
    "cal_w_discount": ("float", 0.00, 0.65, 0.08, "calibrated_logit"),
    "cal_home_rank_mult": ("float", 0.75, 2.20, 0.16, "calibrated_logit"),
    "cal_hybrid_ml_weight": ("float", 0.00, 0.80, 0.10, "calibrated_logit"),
    # 4. Cold-Start, Neighbor Prior (X10) & Slot Gating
    "w_is_new_product": ("float", 0.00, 0.25, 0.04, "cold_start"),
    "w_reverse_graft": ("float", 0.05, 0.90, 0.12, "cold_start"),
    "w_forward_cold": ("float", 0.10, 1.10, 0.14, "cold_start"),
    "cold_neighbor_pop_prior_weight": ("float", 0.10, 1.20, 0.16, "cold_start"),
    "cold_sparse_cos_weight": ("float", 0.30, 0.85, 0.10, "cold_start"),
    "cold_subcat_match_bonus": ("float", 0.10, 0.70, 0.10, "cold_start"),
    "cold_cat_match_bonus": ("float", 0.00, 0.40, 0.06, "cold_start"),
    "cold_brand_match_bonus": ("float", 0.00, 0.40, 0.06, "cold_start"),
    "cold_price_prox_bonus": ("float", 0.00, 0.40, 0.06, "cold_start"),
    "cold_explore_start_slot_warm": ("int", 4, 10, 1, "cold_start"),
    "cold_explore_start_slot_cold": ("int", 0, 8, 2, "cold_start"),
    "max_cold_explore_in_warm": ("int", 0, 4, 1, "cold_start"),
    # 5. Personalization, Price Tie-Break & Discounts (X11)
    "w_user_affinity": ("float", 0.05, 1.20, 0.12, "personalization_price"),
    "w_price_compat": ("float", 0.00, 0.80, 0.10, "personalization_price"),
    "w_log_price_norm": ("float", -0.35, 0.15, 0.07, "personalization_price"),
    "price_std_floor": ("float", 0.20, 1.00, 0.12, "personalization_price"),
    "price_cheap_asymmetry": ("float", 0.00, 0.90, 0.16, "personalization_price"),
    "revenue_tie_weight": ("float", -0.15, 0.12, 0.04, "personalization_price"),
    "tie_break_mode": ("choice", ("conversion", "cheap", "revenue"), "personalization_price"),
    "tie_break_eps": ("float", 0.002, 0.040, 0.008, "personalization_price"),
    "discount_boost_weight": ("float", 0.00, 0.55, 0.08, "personalization_price"),
    # 6. Cart, Forgotten Cart & Recurrent Rebuy Rules (M6 / 4.1)
    "v4_cart_priority": ("bool", "cart_recurrent_rules"),
    "v4_cart_active_minutes": ("float", 5.0, 90.0, 15.0, "cart_recurrent_rules"),
    "v4_cart_max_slots": ("int", 1, 4, 1, "cart_recurrent_rules"),
    "cart_active_score_bonus": ("float", 0.25, 4.50, 0.32, "cart_recurrent_rules"),
    "v4_forgotten_cart": ("bool", "cart_recurrent_rules"),
    "v4_forgotten_max_slots": ("int", 0, 3, 1, "cart_recurrent_rules"),
    "cart_forgotten_score_bonus": ("float", 0.00, 1.40, 0.18, "cart_recurrent_rules"),
    "v4_recurrent": ("bool", "cart_recurrent_rules"),
    "v4_recurrent_max_slots": ("int", 1, 3, 1, "cart_recurrent_rules"),
    "recurrent_score_bonus": ("float", 0.00, 1.00, 0.14, "cart_recurrent_rules"),
    "allow_recurrent_rebuy": ("bool", "cart_recurrent_rules"),
    "v4_price_ascending": ("bool", "cart_recurrent_rules"),
    "v4_live_event_purchases": ("bool", "cart_recurrent_rules"),
    # 7. Diversity / MMR & Subcategory Floors
    "v3_no_subcat_floor": ("bool", "diversity_mmr"),
    "subcat_floor_min": ("float", 0.05, 0.55, 0.08, "diversity_mmr"),
    "brand_floor_min": ("float", 0.00, 0.30, 0.05, "diversity_mmr"),
    "mmr_start_slot_session": ("int", 3, 10, 2, "diversity_mmr"),
    "mmr_start_slot_home": ("int", 1, 10, 2, "diversity_mmr"),
    "mmr_subcat_lambda_session": ("float", 0.00, 0.35, 0.07, "diversity_mmr"),
    "mmr_subcat_lambda_home": ("float", 0.00, 0.30, 0.06, "diversity_mmr"),
    "mmr_brand_lambda": ("float", 0.00, 0.15, 0.03, "diversity_mmr"),
    "mmr_relative_penalty": ("bool", "diversity_mmr"),
    # 8. Bandit, Position Debiasing, EASE & ML Blend
    "bandit_ucb_alpha": ("float", 0.02, 0.30, 0.04, "bandit_ml"),
    "bandit_cold_ucb_multiplier": ("float", 0.80, 2.80, 0.30, "bandit_ml"),
    "bandit_warm_ucb_multiplier": ("float", 0.05, 0.60, 0.10, "bandit_ml"),
    "use_empirical_exam_curve": ("bool", "bandit_ml"),
    "bandit_exam_power": ("float", 0.45, 1.00, 0.10, "bandit_ml"),
    "ml_max_blend_alpha": ("float", 0.00, 0.65, 0.10, "bandit_ml"),
    "v3_ease_weight": ("float", 0.00, 0.50, 0.08, "bandit_ml"),
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
        PARAM_SPECS[f"seg_{_seg}__{_sig}"] = ("float", 0.25, 2.20, 0.24, "segments")

PARAM_GROUPS: tuple[str, ...] = (
    "calibrated_logit",
    "base_popularity_5_1",
    "session_core",
    "cart_recurrent_rules",
    "cold_start",
    "personalization_price",
    "diversity_mmr",
    "bandit_ml",
    "segments",
)


def _is_param_relevant(pname: str, wp: WeightsProfile) -> bool:
    """Filter out dead parameters for the parent's current scoring_mode."""
    mode = getattr(wp, "scoring_mode", "legacy")
    if mode == "legacy":
        return pname == "scoring_mode" or not pname.startswith("cal_")
    if mode == "calibrated":
        if pname.startswith(("seg_", "w_")) or pname in {
            "legacy_self_view_linear",
            "view_base_bonus",
            "slot1_category_match_bonus",
            "same_cat_bonus_weight",
            "v3_bg_penalty",
            "bg_penalty_diff_subcat",
            "bg_penalty_same_subcat",
            "subcat_floor_min",
            "brand_floor_min",
            "v3_no_subcat_floor",
            "cart_active_score_bonus",
            "cart_forgotten_score_bonus",
            "recurrent_score_bonus",
            "discount_boost_weight",
            "cal_hybrid_ml_weight",
            "ml_max_blend_alpha",
        }:
            return False
        return True
    # hybrid uses cal_* + segments + bandit + structural
    if pname.startswith("w_") or pname in {
        "legacy_self_view_linear",
        "view_base_bonus",
        "slot1_category_match_bonus",
        "same_cat_bonus_weight",
        "v3_bg_penalty",
        "bg_penalty_diff_subcat",
        "bg_penalty_same_subcat",
        "subcat_floor_min",
        "brand_floor_min",
        "v3_no_subcat_floor",
        "cart_active_score_bonus",
        "cart_forgotten_score_bonus",
        "recurrent_score_bonus",
        "discount_boost_weight",
    }:
        return False
    return True


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
        "pop_rows": list(popularity_rows) if popularity_rows else [],
        "trans_rows": list(transition_rows) if transition_rows else [],
        "_ease_cache": None,
    }


def _recompute_5_1_structures(
    base_snap: RecommenderSnapshotV4,
    wp: WeightsProfile,
    cfg: ConfigV4,
    struct_prep: dict[str, Any],
) -> tuple[dict[str, float], dict[str, tuple[tuple[str, float], ...]], tuple[str, ...], dict[str, int]]:
    """Fast exact recomputation of M3 base-rate, X10 neighbor prior, sales_rank, EASE, and D5 transitions."""
    active_ids = base_snap.active_ids
    raw_pop = struct_prep["raw_pop"]
    hist_purchases = struct_prep["hist_purchases"]
    log_pop_norm = struct_prep["log_pop_norm"]
    live_buys_equiv = struct_prep["live_buys_equiv"]

    mix_buys = {
        pid: hist_purchases.get(pid, 0.0) + wp.live_pop_mix_weight * live_buys_equiv.get(pid, 0.0)
        for pid in active_ids
    }
    ranked_by_sales = sorted(
        active_ids,
        key=lambda pid: (mix_buys[pid], raw_pop.get(pid, 0.0), pid),
        reverse=True,
    )
    max_cal_rank = max(int(getattr(wp, "cal_max_rank", 3000)), 1)
    sales_rank: dict[str, int] = {
        pid: min(idx, max_cal_rank) if (mix_buys[pid] > 0.0 or raw_pop.get(pid, 0.0) > 0.0) else max_cal_rank
        for idx, pid in enumerate(ranked_by_sales)
    }
    if wp.use_sales_rank_base_rate and active_ids:
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
    eff_beta = float(getattr(wp, "lift_beta", cfg.lift_beta))
    eff_prior = max(float(getattr(wp, "lift_prior_pop", cfg.lift_prior_pop)), 1.0)
    raw_trans: dict[str, list[tuple[str, float]]] = defaultdict(list)
    use_raw = bool(struct_prep["has_raw_transitions"])
    for src, dst, base_sc in struct_prep["trans_edges"]:
        if use_raw:
            dst_pop = max(math.expm1(raw_pop.get(dst, 0.0)), 0.0)
            lift_score = base_sc / ((dst_pop + eff_prior) ** eff_beta)
        else:
            lift_score = base_sc
        if gamma > 0.0:
            lift_score *= (1.0 - gamma) + gamma * global_pop_norm.get(dst, 0.25)
        raw_trans[src].append((dst, lift_score))

    ease_w = min(max(float(wp.v3_ease_weight), 0.0), 1.0)
    ease_neighbors: dict[str, tuple[tuple[str, float], ...]] = {}
    if ease_w > 0.0 and struct_prep.get("trans_rows"):
        if struct_prep.get("_ease_cache") is None:
            struct_prep["_ease_cache"] = fit_ease_from_transitions(
                struct_prep.get("pop_rows") or [],
                struct_prep["trans_rows"],
                active_ids,
                lam=250.0,
                top_k=35,
            )
        ease_neighbors = struct_prep["_ease_cache"] or {}

    transitions: dict[str, tuple[tuple[str, float], ...]] = {}
    if ease_w <= 0.0 or not ease_neighbors:
        for src, edges in raw_trans.items():
            edges.sort(key=lambda kv: kv[1], reverse=True)
            top_edges = edges[:35]
            max_s = top_edges[0][1] if top_edges else 1.0
            if max_s > 0.0:
                transitions[src] = tuple((dst, sc / max_s) for dst, sc in top_edges)
    else:
        for src in (set(raw_trans) | set(ease_neighbors)):
            edges = raw_trans.get(src, [])
            edges.sort(key=lambda kv: kv[1], reverse=True)
            top_edges = edges[:35]
            max_l = top_edges[0][1] if top_edges else 0.0
            lift_norm = {dst: (sc / max_l) for dst, sc in top_edges} if max_l > 0.0 else {}
            ease_norm = dict(ease_neighbors.get(src, ()))
            combined = [
                (dst, (1.0 - ease_w) * lift_norm.get(dst, 0.0) + ease_w * ease_norm.get(dst, 0.0))
                for dst in (set(lift_norm) | set(ease_norm))
            ]
            combined.sort(key=lambda kv: kv[1], reverse=True)
            top_c = combined[:35]
            max_c = top_c[0][1] if top_c else 0.0
            if max_c > 0.0:
                transitions[src] = tuple((dst, sc / max_c) for dst, sc in top_c if sc > 0.0)

    fallback_scores: dict[str, float] = {}
    tie_mode = wp.tie_break_mode
    fb_live_w = float(getattr(wp, "fallback_live_trend_weight", 0.6))
    for pid in active_ids:
        p_obj = base_snap.products[pid]
        pop_s = global_pop_norm.get(pid, 0.0) + fb_live_w * base_snap.live_trend_norm.get(pid, 0.0)
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
    return global_pop_norm, transitions, fallback_ranked, sales_rank


def _update_snapshot_for_profile(
    base_snap: RecommenderSnapshotV4,
    wp: WeightsProfile,
    *,
    base_ucb_alpha: float,
    struct_prep: dict[str, Any] | None = None,
) -> tuple[RecommenderSnapshotV4, ConfigV4]:
    """Attach updated WeightsProfile, ConfigV4, ranker weights, 5.1f base-rate/transitions, and UCB cache."""
    cfg = ConfigV4.from_profile(wp, model_version=MODEL_VERSION_LABEL)
    base_ranker = base_snap.ranker
    pw = wp.prior_weights_array()
    if base_ranker is not None:
        base_alpha_limit = max(float(getattr(base_snap.weights_profile, "ml_max_blend_alpha", 0.45)), 1e-6)
        alpha = min(max(base_ranker.ml_blend_alpha * (wp.ml_max_blend_alpha / base_alpha_limit), 0.0), wp.ml_max_blend_alpha)
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
    sales_rank = getattr(base_snap, "sales_rank", None) or {}

    if struct_prep is not None and (
        base_wp is None
        or not sales_rank
        or base_wp.use_sales_rank_base_rate != wp.use_sales_rank_base_rate
        or abs(base_wp.sales_rank_blend_weight - wp.sales_rank_blend_weight) > 1e-6
        or abs(base_wp.live_pop_mix_weight - wp.live_pop_mix_weight) > 1e-6
        or abs(base_wp.cold_neighbor_pop_prior_weight - wp.cold_neighbor_pop_prior_weight) > 1e-6
        or abs(base_wp.transition_dest_pop_gamma - wp.transition_dest_pop_gamma) > 1e-6
        or abs(getattr(base_wp, "lift_beta", 0.45) - getattr(wp, "lift_beta", 0.45)) > 1e-6
        or abs(getattr(base_wp, "lift_prior_pop", 30.0) - getattr(wp, "lift_prior_pop", 30.0)) > 1e-6
        or abs(getattr(base_wp, "fallback_live_trend_weight", 0.6) - getattr(wp, "fallback_live_trend_weight", 0.6)) > 1e-6
        or abs(base_wp.v3_ease_weight - wp.v3_ease_weight) > 1e-6
        or base_wp.tie_break_mode != wp.tie_break_mode
        or abs(base_wp.revenue_tie_weight - wp.revenue_tie_weight) > 1e-6
    ):
        global_pop_norm, transitions, fallback_ranked, sales_rank = _recompute_5_1_structures(
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
        sales_rank=sales_rank,
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
    hit_buy1_arr = np.zeros(n, dtype=float)
    hit_buy3_arr = np.zeros(n, dtype=float)
    attn_buy_arr = np.zeros(n, dtype=float)
    dcg_buy_arr = np.zeros(n, dtype=float)
    for i, (b, r) in enumerate(zip(purchased, recs)):
        if b and b in r:
            pos = r.index(b)
            dcg_buy_arr[i] = 1.0 / np.log2(pos + 2.0)
            if pos == 0:
                hit_buy1_arr[i] = 1.0
            if pos < 3:
                hit_buy3_arr[i] = 1.0
            attn_buy_arr[i] = EMPIRICAL_SLOT_WEIGHTS[pos] if pos < len(EMPIRICAL_SLOT_WEIGHTS) else 0.50

    pages_ev = [
        (ctx.get("page") if isinstance(ctx, dict) else "")
        for ctx in df_eval["context"]
    ]
    prod_mask = np.array([p == "product" for p in pages_ev], dtype=bool) & has_buy
    cat_mask = np.array([p == "category" for p in pages_ev], dtype=bool) & has_buy
    home_mask = np.array([p == "home" for p in pages_ev], dtype=bool) & has_buy

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
    attn_mean = float(attn_buy_arr[has_buy].mean()) if has_buy.any() else 0.0
    expected_conv_100 = float(100.0 * attn_buy_arr[has_buy].sum() / max(n_total_equiv, 1.0))

    return {
        "n_evaluated": int(n),
        "n_equivalent": int(round(n_total_equiv)),
        "n_clicks": int(has_click.sum()),
        "n_buys": int(has_buy.sum()),
        "hit_rate_click@10": float(hit_click_arr[has_click].mean()) if has_click.any() else 0.0,
        "hit_rate_buy@1": float(hit_buy1_arr[has_buy].mean()) if has_buy.any() else 0.0,
        "hit_rate_buy@3": float(hit_buy3_arr[has_buy].mean()) if has_buy.any() else 0.0,
        "hit_rate_buy@10": float(hit_buy_arr[has_buy].mean()) if has_buy.any() else 0.0,
        "dcg_buy@10": float(dcg_buy_arr[has_buy].mean()) if has_buy.any() else 0.0,
        "attention_buy@10": attn_mean,
        "expected_conv_100": expected_conv_100,
        "page_product_buy@10": float(hit_buy_arr[prod_mask].mean()) if prod_mask.any() else 0.0,
        "page_category_buy@10": float(hit_buy_arr[cat_mask].mean()) if cat_mask.any() else 0.0,
        "page_home_buy@10": float(hit_buy_arr[home_mask].mean()) if home_mask.any() else 0.0,
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
    """Compute primary purchase-conversion objective prioritizing hit@10, top-slot DCG, attention CVR & SLA."""
    real_score = 0.0
    if real_metrics:
        r_hit = float(real_metrics.get("hit_rate_buy@10", 0.0))
        r_hit3 = float(real_metrics.get("hit_rate_buy@3", 0.0))
        r_dcg = float(real_metrics.get("dcg_buy@10", 0.0))
        r_attn = float(real_metrics.get("attention_buy@10", r_dcg))
        r_clk = float(real_metrics.get("hit_rate_click@10", 0.0))
        r_snips = min(float(real_metrics.get("SNIPS_cvr_slot1", 0.0)) * 5.0, 1.0)
        lat_ms = float(real_metrics.get("latency_ms_mean", 0.0))
        lat_pen = max(0.0, (lat_ms - 4.0) * 0.025)
        real_score = (
            0.30 * r_hit
            + 0.30 * r_dcg
            + 0.20 * r_attn
            + 0.10 * r_hit3
            + 0.05 * r_clk
            + 0.05 * r_snips
            - lat_pen
        )

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
    *,
    time_frac: float | None = None,
) -> tuple[WeightsProfile, str, dict[str, dict[str, Any]]]:
    """Generate a non-duplicate candidate WeightsProfile guided by mode-aware adaptive UCB priorities."""
    frac = float(time_frac) if time_frac is not None else (step_idx / max(total_iters, 1))
    frac = min(max(frac, 0.0), 1.0)
    group_scores = intel["group_scores"]
    param_scores = intel["param_scores"]
    active_pool = [
        k
        for k, v in PARAM_SPECS.items()
        if v[-1] in active_groups and _is_param_relevant(k, best_wp)
    ]
    if not active_pool:
        active_pool = [k for k, v in PARAM_SPECS.items() if v[-1] in active_groups]

    def _mutate_param(pname: str, temp: float) -> Any | None:
        spec = PARAM_SPECS[pname]
        cur = get_param_value(best_wp, pname)
        if spec[0] == "float":
            _, lo, hi, scale, _ = spec
            new_v = round(min(max(float(cur) + float(rng.normal(0.0, scale * temp)), lo), hi), 4)
            if abs(new_v - float(cur)) > 1e-6:
                return new_v
            return None
        if spec[0] == "int":
            _, lo, hi, scale, _ = spec
            delta = int(round(float(rng.normal(0.0, max(1.0, scale * temp)))))
            if delta == 0:
                delta = int(rng.choice([-1, 1]))
            new_v = int(min(max(int(cur) + delta, lo), hi))
            return new_v if new_v != int(cur) else None
        if spec[0] == "bool":
            return not bool(cur)
        opts = [o for o in spec[1] if o != cur]
        return str(rng.choice(opts)) if opts else None

    for attempt in range(30):
        updates: dict[str, Any] = {}

        if attempt == 0 and last_accepted_delta and rng.random() < 0.28:
            strategy = "momentum"
            for pname, dval in last_accepted_delta.items():
                if pname not in PARAM_SPECS or not _is_param_relevant(pname, best_wp):
                    continue
                spec = PARAM_SPECS[pname]
                if spec[0] == "float":
                    _, lo, hi, _, _ = spec
                    cur = float(get_param_value(best_wp, pname))
                    updates[pname] = round(min(max(cur + float(dval) * 0.70, lo), hi), 4)
        else:
            relevant_groups = [
                g
                for g in active_groups
                if any(PARAM_SPECS[k][-1] == g for k in active_pool)
            ] or list(active_groups)
            picked_grp = _weighted_sample(rng, relevant_groups, group_scores, 1)[0]
            roll = float(rng.random())
            if frac <= 0.25:
                mode = "explore_broad" if roll < 0.45 else f"group:{picked_grp}"
            elif frac <= 0.72:
                if roll < 0.18:
                    mode = "coord_sweep"
                elif roll < 0.34:
                    mode = "explore_broad"
                else:
                    mode = f"group:{picked_grp}"
            else:
                if roll < 0.35:
                    mode = "coord_sweep"
                elif roll < 0.65:
                    mode = "fine_tune"
                else:
                    mode = f"group:{picked_grp}"

            strategy = mode
            if mode == "explore_broad":
                pool = active_pool
                n_pick = int(rng.integers(3, 6))
                temp = 1.15
            elif mode == "coord_sweep":
                pool = active_pool
                n_pick = 1
                temp = max(0.35, 0.90 - 0.55 * frac)
            elif mode.startswith("group:"):
                grp = mode.split(":", 1)[1]
                pool = [k for k in active_pool if PARAM_SPECS[k][-1] == grp]
                if not pool:
                    pool = active_pool
                n_pick = int(rng.integers(1, min(4, len(pool)) + 1))
                temp = max(0.35, 1.0 - 0.55 * frac)
            else:
                pool = active_pool
                n_pick = int(rng.integers(1, 3))
                temp = 0.35

            chosen = _weighted_sample(rng, pool, param_scores, min(n_pick, len(pool)))
            for pname in chosen:
                val = _mutate_param(pname, temp)
                if val is not None:
                    updates[pname] = val

        if not updates:
            continue

        cand_wp = apply_param_updates(best_wp, updates)
        if cand_wp.candidate_pool_size < cand_wp.guaranteed_top_sellers_in_pool + 6:
            cand_wp = replace(
                cand_wp,
                candidate_pool_size=min(160, cand_wp.guaranteed_top_sellers_in_pool + 12),
            )
        sig = profile_signature(cand_wp)
        if sig not in seen_sigs:
            changed_diff = {
                k: {"old": get_param_value(best_wp, k), "new": get_param_value(cand_wp, k)}
                for k in PARAM_SPECS
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
            f"buy@10={r['hit_rate_buy@10']:.4f} @3={r.get('hit_rate_buy@3', 0.0):.4f} "
            f"dcg={r['dcg_buy@10']:.4f} attn={r.get('attention_buy@10', 0.0):.4f} "
            f"clk={r['hit_rate_click@10']:.4f} snips={r['SNIPS_cvr_slot1']:.4f} "
            f"lat={r.get('latency_ms_mean', 0.0):.2f}ms"
        )
    s = res.get("synthetic")
    if s:
        parts.append(
            f"syn_buy@10={s['ALL']['hit_rate_buy@10']:.4f} syn_dcg={s['ALL']['dcg_buy@10']:.4f} "
            f"cold_buy@10={s['COLD_START']['hit_rate_buy@10']:.4f}"
        )
    parts.append(f"({res['eval_time_s']:.2f}s)")
    return " | ".join(parts)


def _extract_param_from_dict(params: dict[str, Any], pname: str) -> Any:
    if pname.startswith("seg_") and "__" in pname:
        seg_part, sig_name = pname[4:].split("__", 1)
        sig_idx = SEGMENT_SIGNAL_NAMES.index(sig_name)
        vals = params.get(f"seg_{seg_part}", ())
        if isinstance(vals, (list, tuple)) and sig_idx < len(vals):
            return float(vals[sig_idx])
        return None
    return params.get(pname)


def export_odyseus_report(
    archive: dict[str, Any],
    out_dir: Path,
    verified_top: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Export complete per-iteration CSV log and executive Markdown parameter sensitivity report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "odyseus_iterations.csv"
    md_path = out_dir / "INFORME_ODYSEUS_5_1F.md"

    iterations: list[dict[str, Any]] = list(archive.get("iterations") or [])
    fieldnames = [
        "iter",
        "timestamp",
        "strategy",
        "scoring_mode",
        "accepted",
        "objective",
        "best_objective_so_far",
        "eval_time_s",
        "hit_rate_buy@10",
        "hit_rate_buy@3",
        "hit_rate_buy@1",
        "dcg_buy@10",
        "attention_buy@10",
        "hit_rate_click@10",
        "SNIPS_cvr_slot1",
        "expected_conv_100",
        "page_product_buy@10",
        "page_category_buy@10",
        "page_home_buy@10",
        "cold_buy@10",
        "latency_ms_mean",
        "changed_params",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for it in iterations:
            metrics = it.get("metrics") or {}
            rm = metrics.get("real") or {}
            sm = metrics.get("synthetic") or {}
            params = it.get("params") or {}
            cold_buy = ((sm.get("COLD_START") or {}).get("hit_rate_buy@10")) if sm else ""
            changed = "; ".join(
                f"{k}:{v.get('old')}->{v.get('new')}"
                for k, v in (it.get("changed_params") or {}).items()
            )
            writer.writerow(
                {
                    "iter": it.get("iter", ""),
                    "timestamp": it.get("timestamp", ""),
                    "strategy": it.get("strategy", ""),
                    "scoring_mode": params.get("scoring_mode", "legacy"),
                    "accepted": it.get("accepted", False),
                    "objective": f"{float(it.get('objective', 0.0)):.6f}",
                    "best_objective_so_far": f"{float(it.get('best_objective_so_far', 0.0)):.6f}",
                    "eval_time_s": f"{float(it.get('eval_time_s', 0.0)):.3f}",
                    "hit_rate_buy@10": f"{float(rm.get('hit_rate_buy@10', 0.0)):.5f}" if rm else "",
                    "hit_rate_buy@3": f"{float(rm.get('hit_rate_buy@3', 0.0)):.5f}" if rm else "",
                    "hit_rate_buy@1": f"{float(rm.get('hit_rate_buy@1', 0.0)):.5f}" if rm else "",
                    "dcg_buy@10": f"{float(rm.get('dcg_buy@10', 0.0)):.5f}" if rm else "",
                    "attention_buy@10": f"{float(rm.get('attention_buy@10', 0.0)):.5f}" if rm else "",
                    "hit_rate_click@10": f"{float(rm.get('hit_rate_click@10', 0.0)):.5f}" if rm else "",
                    "SNIPS_cvr_slot1": f"{float(rm.get('SNIPS_cvr_slot1', 0.0)):.5f}" if rm else "",
                    "expected_conv_100": f"{float(rm.get('expected_conv_100', 0.0)):.5f}" if rm else "",
                    "page_product_buy@10": f"{float(rm.get('page_product_buy@10', 0.0)):.5f}" if rm else "",
                    "page_category_buy@10": f"{float(rm.get('page_category_buy@10', 0.0)):.5f}" if rm else "",
                    "page_home_buy@10": f"{float(rm.get('page_home_buy@10', 0.0)):.5f}" if rm else "",
                    "cold_buy@10": f"{float(cold_buy):.5f}" if cold_buy != "" else "",
                    "latency_ms_mean": f"{float(rm.get('latency_ms_mean', 0.0)):.3f}" if rm else "",
                    "changed_params": changed,
                }
            )

    p_trials: dict[str, int] = defaultdict(int)
    p_wins: dict[str, int] = defaultdict(int)
    p_abs_impact: dict[str, float] = defaultdict(float)
    p_pos_gain: dict[str, float] = defaultdict(float)
    prev_best = float(iterations[0]["objective"]) if iterations else 0.0
    for it in iterations:
        obj = float(it.get("objective", 0.0))
        accepted = bool(it.get("accepted"))
        changed = it.get("changed_params") or {}
        gain = max(0.0, obj - prev_best)
        sens = abs(obj - prev_best)
        k_n = max(len(changed), 1)
        for pname in changed:
            if pname not in PARAM_SPECS:
                continue
            p_trials[pname] += 1
            p_abs_impact[pname] += sens / k_n
            if accepted:
                p_wins[pname] += 1
                p_pos_gain[pname] += gain / k_n
        if accepted and obj > prev_best:
            prev_best = obj

    best_params = archive.get("best_params") or {}
    default_params = WeightsProfile().to_dict()
    verified_rows = verified_top if verified_top is not None else (archive.get("verified_full_real") or [])
    sel_sum = archive.get("selector_summary") or {}
    best_by_mode = archive.get("best_by_mode") or {}

    lines: list[str] = [
        f"# Informe d'Optimització Odyseus — {archive.get('model', MODEL_VERSION_LABEL)}",
        "",
        f"- **Data / Hora actualització:** `{archive.get('updated_at', '')}`",
        f"- **Mode d'avaluació:** `{archive.get('eval_mode', '')}` | **Durada execució:** `{archive.get('elapsed_seconds', 0.0):.1f}s`",
        f"- **Iteracions totals:** `{archive.get('total_iterations', len(iterations))}` | **Millor iteració (fast):** `#{archive.get('best_iter', 0)}`",
        f"- **Objectiu Baseline:** `{float(archive.get('baseline_objective', 0.0)):.6f}`",
        f"- **Objectiu Millor:** `{float(archive.get('best_objective', 0.0)):.6f}`",
        "",
    ]

    if best_by_mode:
        lines.extend(
            [
                "## Resum per `scoring_mode` (cerca ràpida)",
                "",
                "| Scoring Mode | Millor Iter | Objective | Hit@10 Buy | Hit@3 Buy | DCG@10 Buy | Attn@10 | Latència (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mname, info in sorted(best_by_mode.items(), key=lambda kv: float(kv[1].get("objective", 0.0)), reverse=True):
            rm = (info.get("metrics") or {}).get("real") or {}
            lines.append(
                f"| `{mname}` | {info.get('iter', '')} | {float(info.get('objective', 0.0)):.5f} | "
                f"{float(rm.get('hit_rate_buy@10', 0.0)):.4f} | {float(rm.get('hit_rate_buy@3', 0.0)):.4f} | "
                f"{float(rm.get('dcg_buy@10', 0.0)):.4f} | {float(rm.get('attention_buy@10', 0.0)):.4f} | "
                f"{float(rm.get('latency_ms_mean', 0.0)):.2f} |"
            )
        lines.append("")

    if verified_rows:
        lines.extend(
            [
                "## Verificació Exacta Full-Real (totes les 25.000 peticions reals)",
                "",
                "| Rank | Candidat | Scoring Mode | Real Obj | Hit@10 Buy | Hit@3 Buy | Hit@1 Buy | DCG@10 | Attn@10 | Product Buy@10 | Category Buy@10 | Home Buy@10 | SNIPS CVR1 | Lat (ms) |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for idx, row in enumerate(verified_rows, start=1):
            rm = row.get("metrics") or {}
            lines.append(
                f"| {idx} | `{row.get('label', '')}` | `{row.get('scoring_mode', '')}` | "
                f"{float(row.get('objective', 0.0)):.5f} | "
                f"{float(rm.get('hit_rate_buy@10', 0.0)):.4f} | {float(rm.get('hit_rate_buy@3', 0.0)):.4f} | "
                f"{float(rm.get('hit_rate_buy@1', 0.0)):.4f} | {float(rm.get('dcg_buy@10', 0.0)):.4f} | "
                f"{float(rm.get('attention_buy@10', 0.0)):.4f} | "
                f"{float(rm.get('page_product_buy@10', 0.0)):.4f} | "
                f"{float(rm.get('page_category_buy@10', 0.0)):.4f} | "
                f"{float(rm.get('page_home_buy@10', 0.0)):.4f} | "
                f"{float(rm.get('SNIPS_cvr_slot1', 0.0)):.5f} | "
                f"{float(rm.get('latency_ms_mean', 0.0)):.2f} |"
            )
        lines.append("")

    g_trials = sel_sum.get("group_trials") or {}
    g_wins = sel_sum.get("group_wins") or {}
    if g_trials:
        lines.extend(
            [
                "## Rendiment per Grup de Paràmetres (Adaptive UCB)",
                "",
                "| Grup | Proves | Victòries | Win Rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for grp in sorted(g_trials, key=lambda g: g_wins.get(g, 0), reverse=True):
            t = max(int(g_trials.get(grp, 0)), 1)
            w = int(g_wins.get(grp, 0))
            lines.append(f"| `{grp}` | {t} | {w} | {(100.0 * w / t):.1f}% |")
        lines.append("")

    ranked_params = sorted(
        p_trials.keys(),
        key=lambda k: (p_wins[k], p_pos_gain[k], p_abs_impact[k] / max(p_trials[k], 1)),
        reverse=True,
    )
    if ranked_params:
        lines.extend(
            [
                "## Top 35 Paràmetres per Impacte i Taxa de Victòria",
                "",
                "| Paràmetre | Grup | Proves | Victòries | Win% | Impacte Mitjà | Guany Positiu Total | Valor Campió |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for pname in ranked_params[:35]:
            t = max(p_trials[pname], 1)
            w = p_wins[pname]
            val = _extract_param_from_dict(best_params, pname)
            lines.append(
                f"| `{pname}` | `{PARAM_SPECS[pname][-1]}` | {t} | {w} | "
                f"{(100.0 * w / t):.1f}% | {(p_abs_impact[pname] / t):.5f} | "
                f"{p_pos_gain[pname]:.5f} | `{val}` |"
            )
        lines.append("")

    changed_vs_default = [
        pname
        for pname in PARAM_SPECS
        if _extract_param_from_dict(best_params, pname) != _extract_param_from_dict(default_params, pname)
    ]
    if changed_vs_default:
        lines.extend(
            [
                "## Diferències Principals del Campió vs `calypso_5_1_default`",
                "",
                "| Paràmetre | Default 5.1 | Campió Odyseus |",
                "|---|---|---|",
            ]
        )
        for pname in changed_vs_default[:60]:
            lines.append(
                f"| `{pname}` | `{_extract_param_from_dict(default_params, pname)}` | "
                f"`{_extract_param_from_dict(best_params, pname)}` |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def run_odyseus_optimization(
    model_root: Path,
    *,
    base_profile: WeightsProfile,
    history_path: Path,
    eval_mode: str = "hybrid",
    iters: int = 25,
    duration_seconds: float | None = None,
    seed: int = 42,
    n_syn: int = 1200,
    real_bg_sample: int = 2500,
    full_real: bool = False,
    eval_presets_first: bool = True,
    resume: bool = False,
    save_best: bool = False,
    verify_top_k: int = 0,
    export_report: bool = True,
    target_profile_path: Path | None = None,
    offline_cache_dir: Path | None = None,
    focus_groups: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run Odyseus adaptive parameter search and persist every iteration to a single archive file."""
    run_start_ts = time.monotonic()
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
    best_by_mode: dict[str, dict[str, Any]] = {}

    if resume and history_path.is_file():
        existing = json.loads(history_path.read_text(encoding="utf-8"))
        iterations = list(existing.get("iterations") or [])
        for it in iterations:
            wp_prev = WeightsProfile.from_dict(it["params"], base=base_profile)
            seen_sigs.add(profile_signature(wp_prev))
            mname = getattr(wp_prev, "scoring_mode", "legacy")
            it_obj = float(it["objective"])
            if mname not in best_by_mode or it_obj > float(best_by_mode[mname]["obj"]):
                best_by_mode[mname] = {
                    "wp": wp_prev,
                    "obj": it_obj,
                    "iter": int(it["iter"]),
                    "metrics": it["metrics"],
                }
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

    latest_archive: dict[str, Any] = {}

    def _build_archive(ts_now: str, intel_now: dict[str, Any], verified_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        mode_summary = {
            m: {
                "iter": int(info["iter"]),
                "objective": float(info["obj"]),
                "metrics": info["metrics"],
            }
            for m, info in best_by_mode.items()
        }
        return {
            "model": MODEL_VERSION_LABEL,
            "eval_mode": eval_mode,
            "selector_mode": "adaptive_ucb_bandit",
            "objective_formula": (
                "0.65*real_conv + 0.35*syn_conv"
                if eval_mode == "hybrid"
                else ("real_conv" if eval_mode == "real" else "syn_conv")
            ),
            "created_at": created_at,
            "updated_at": ts_now,
            "elapsed_seconds": round(time.monotonic() - run_start_ts, 2),
            "total_iterations": len(iterations),
            "baseline_objective": baseline_obj,
            "real_baseline_reference": (real_bundle["baseline_metrics"] if real_bundle else None),
            "best_iter": best_iter,
            "best_objective": best_obj,
            "best_metrics": best_metrics,
            "best_params": best_wp.to_dict(),
            "best_by_mode": mode_summary,
            "selector_summary": {
                "active_groups": list(active_groups),
                "group_trials": intel_now["group_trials"],
                "group_wins": intel_now["group_wins"],
                "top_learned_params": intel_now["top_learned_params"],
            },
            "verified_full_real": verified_rows,
            "iterations": iterations,
        }

    def _record_iteration(
        iter_num: int,
        wp: WeightsProfile,
        strategy: str,
        changed_params: dict[str, Any],
        res: dict[str, Any],
    ) -> bool:
        nonlocal best_wp, best_obj, best_iter, best_metrics, baseline_obj, latest_archive
        obj = float(res["objective"])
        if iter_num == 0 and not iterations:
            baseline_obj = obj
        accepted = obj > best_obj + 1e-6
        if accepted:
            best_wp = wp
            best_obj = obj
            best_iter = iter_num
            best_metrics = {"real": res["real"], "synthetic": res["synthetic"]}

        mname = getattr(wp, "scoring_mode", "legacy")
        prev_mode_best = float(best_by_mode[mname]["obj"]) if mname in best_by_mode else -1.0
        if obj > prev_mode_best + 1e-6:
            best_by_mode[mname] = {
                "wp": wp,
                "obj": obj,
                "iter": iter_num,
                "metrics": {"real": res["real"], "synthetic": res["synthetic"]},
            }

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
        latest_archive = _build_archive(entry["timestamp"], intel_now)
        _atomic_write_json(history_path, latest_archive)

        tag = "BEST*" if accepted else ("MODE+" if obj > prev_mode_best + 1e-6 else "     ")
        print(f"[iter {iter_num:04d}] {tag} [{strategy:<25}] ({mname:<10}) {_compact_metrics_line(res)}")
        return accepted

    use_time_budget = duration_seconds is not None and float(duration_seconds) > 0.0
    cur_iter = start_iter
    if not iterations:
        res0 = evaluate_candidate(base_profile, eval_mode=eval_mode, real_bundle=real_bundle, syn_bundle=syn_bundle)
        _record_iteration(cur_iter, base_profile, f"baseline:{base_profile.profile_name}", {}, res0)
        cur_iter += 1

        if eval_presets_first:
            for pname in list_presets():
                if use_time_budget and (time.monotonic() - run_start_ts) >= float(duration_seconds):
                    break
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
    max_steps = 100000 if use_time_budget else max(int(iters), 0)
    step = 0
    while step < max_steps:
        if use_time_budget and step >= 1 and (time.monotonic() - run_start_ts) >= float(duration_seconds):
            break
        step += 1
        time_frac = (
            min(1.0, (time.monotonic() - run_start_ts) / float(duration_seconds))
            if use_time_budget
            else None
        )
        intel = compute_adaptive_intelligence(iterations, active_groups=active_groups)

        parent_wp = best_wp
        using_global_best = True
        if use_time_budget and len(best_by_mode) > 1 and rng.random() < 0.30:
            mode_keys = list(best_by_mode.keys())
            chosen_mode = str(rng.choice(mode_keys))
            parent_wp = best_by_mode[chosen_mode]["wp"]
            using_global_best = chosen_mode == getattr(best_wp, "scoring_mode", "legacy")

        cand_wp, strategy, changed_diff = _propose_candidate(
            parent_wp,
            rng,
            step,
            max(int(iters), step),
            seen_sigs,
            last_accepted_delta if using_global_best else None,
            intel,
            active_groups,
            time_frac=time_frac,
        )
        res_c = evaluate_candidate(cand_wp, eval_mode=eval_mode, real_bundle=real_bundle, syn_bundle=syn_bundle)
        accepted = _record_iteration(cur_iter, cand_wp, strategy, changed_diff, res_c)
        if using_global_best:
            if accepted:
                delta_map: dict[str, Any] = {}
                for pname, d in changed_diff.items():
                    if PARAM_SPECS[pname][0] == "float":
                        delta_map[pname] = float(d["new"]) - float(d["old"])
                last_accepted_delta = delta_map or None
            else:
                last_accepted_delta = None
        cur_iter += 1

    verified_top: list[dict[str, Any]] | None = None
    best_legacy_wp: WeightsProfile | None = None
    if verify_top_k > 0 and eval_mode in {"real", "hybrid"}:
        print(f"\n[ODYSEUS] Verifying top {verify_top_k} candidates + reference baselines on ALL 25,000 real rows...")
        full_bundle = load_real_eval_bundle(
            model_root,
            off_dir,
            full_real=True,
            seed=seed,
        )
        candidates_to_verify: list[tuple[str, WeightsProfile, int]] = []
        seen_v: set[str] = set()

        def _queue_verify(label: str, wp_v: WeightsProfile, it_num: int) -> None:
            sig_v = profile_signature(wp_v)
            if sig_v not in seen_v:
                seen_v.add(sig_v)
                candidates_to_verify.append((label, wp_v, it_num))

        _queue_verify(f"fast_best_iter{best_iter}", best_wp, best_iter)
        for mname, info in best_by_mode.items():
            _queue_verify(f"fast_mode_{mname}_iter{info['iter']}", info["wp"], int(info["iter"]))

        ranked_iters = sorted(iterations, key=lambda x: float(x["objective"]), reverse=True)
        for it in ranked_iters:
            if len(candidates_to_verify) >= verify_top_k:
                break
            wp_it = WeightsProfile.from_dict(it["params"], base=base_profile)
            _queue_verify(f"iter_{it['iter']}", wp_it, int(it["iter"]))

        _queue_verify(f"base:{base_profile.profile_name}", base_profile, 0)
        for pname in ("calibrated_m3", "conversion_first", "calibrated_logit", "calibrated_hybrid"):
            try:
                _queue_verify(f"preset:{pname}", get_preset(pname), -1)
            except KeyError:
                pass
        prev_opt_path = history_path.parent / "prev_odyseus_49_opt.json"
        if prev_opt_path.is_file():
            try:
                prev_raw = json.loads(prev_opt_path.read_text(encoding="utf-8"))
                _queue_verify("prev_odyseus_49_opt", WeightsProfile.from_dict(prev_raw), -2)
            except Exception:
                pass

        verified_rows: list[dict[str, Any]] = []
        if full_bundle.get("baseline_metrics"):
            bm = full_bundle["baseline_metrics"]
            verified_rows.append(
                {
                    "label": "baseline:global_bestsellers",
                    "scoring_mode": "baseline",
                    "iter": -99,
                    "objective": compute_conversion_objective(bm, None, "real"),
                    "metrics": bm,
                    "wp": None,
                }
            )

        for label, wp_v, it_num in candidates_to_verify:
            res_full = evaluate_candidate(
                wp_v,
                eval_mode="real",
                real_bundle=full_bundle,
                syn_bundle=None,
            )
            verified_rows.append(
                {
                    "label": label,
                    "scoring_mode": getattr(wp_v, "scoring_mode", "legacy"),
                    "iter": it_num,
                    "objective": float(res_full["objective"]),
                    "metrics": res_full["real"],
                    "wp": wp_v,
                }
            )

        verified_rows.sort(key=lambda r: float(r["objective"]), reverse=True)
        for rank_i, vr in enumerate(verified_rows, start=1):
            rm = vr["metrics"] or {}
            print(
                f"  [25k #{rank_i:02d}] {vr['label']:<28} ({vr['scoring_mode']:<10}) "
                f"obj={vr['objective']:.5f} | buy@10={rm.get('hit_rate_buy@10', 0):.4f} "
                f"@3={rm.get('hit_rate_buy@3', 0):.4f} dcg={rm.get('dcg_buy@10', 0):.4f} "
                f"snips={rm.get('SNIPS_cvr_slot1', 0):.5f} lat={rm.get('latency_ms_mean', 0):.2f}ms"
            )

        verified_top = [
            {
                "label": vr["label"],
                "scoring_mode": vr["scoring_mode"],
                "iter": vr["iter"],
                "objective": vr["objective"],
                "metrics": vr["metrics"],
            }
            for vr in verified_rows
        ]

        search_winners = [vr for vr in verified_rows if vr["wp"] is not None]
        if search_winners:
            top_v = search_winners[0]
            best_wp = top_v["wp"]
            if top_v["iter"] >= 0:
                best_iter = int(top_v["iter"])
            best_obj = float(top_v["objective"])
            best_metrics = {"real": top_v["metrics"], "synthetic": None}
            replace(
                best_wp,
                profile_name="odyseus_5_1f_verified_champion",
                description=(
                    f"25k-verified Odyseus 5.1f champion ({top_v['label']}, mode={top_v['scoring_mode']}, "
                    f"real_obj={best_obj:.5f})."
                ),
            ).save_json(history_path.parent / "best_5_1f_verified.json")

        legacy_winners = [vr for vr in search_winners if vr["scoring_mode"] == "legacy"]
        if legacy_winners:
            best_legacy_wp = legacy_winners[0]["wp"]
            replace(
                best_legacy_wp,
                profile_name="odyseus_5_1f_legacy_hotreload",
                description=(
                    f"Best 25k-verified legacy hot-reload profile ({legacy_winners[0]['label']}, "
                    f"real_obj={legacy_winners[0]['objective']:.5f})."
                ),
            ).save_json(history_path.parent / "best_legacy_hotreload.json")

        intel_final = compute_adaptive_intelligence(iterations, active_groups=active_groups)
        latest_archive = _build_archive(
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            intel_final,
            verified_rows=verified_top,
        )
        _atomic_write_json(history_path, latest_archive)

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

    csv_path: Path | None = None
    md_path: Path | None = None
    if export_report and latest_archive:
        csv_path, md_path = export_odyseus_report(latest_archive, history_path.parent, verified_top=verified_top)
        print(f"[ODYSEUS] Exported report -> {csv_path} & {md_path}")

    print(
        f"\n=== Odyseus Optimization Summary ({MODEL_VERSION_LABEL}) ===\n"
        f"History archive : {history_path}\n"
        f"Total iterations: {len(iterations)} (best at iter {best_iter})\n"
        f"Elapsed time    : {time.monotonic() - run_start_ts:.1f}s\n"
        f"Baseline obj    : {baseline_obj:.6f}\n"
        f"Best obj        : {best_obj:.6f} ({best_obj - baseline_obj:+.6f})"
    )
    return {
        "history_path": str(history_path),
        "csv_path": str(csv_path) if csv_path else None,
        "report_path": str(md_path) if md_path else None,
        "best_iter": best_iter,
        "best_objective": best_obj,
        "baseline_objective": baseline_obj,
        "best_profile": final_wp,
        "best_legacy_profile": best_legacy_wp,
        "verified_full_real": verified_top,
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
    parser.add_argument("--iters", type=int, default=20, help="Number of optimization iterations (ignored when --duration-minutes > 0)")
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=0.0,
        help="Optional wall-clock optimization budget in minutes (e.g. 54 for a 1h run with verification)",
    )
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
    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=0,
        help="Verify top-K candidates + baselines on all 25,000 real rows at the end",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable exporting odyseus_iterations.csv and INFORME_ODYSEUS_5_1F.md",
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
    duration_seconds = float(args.duration_minutes) * 60.0 if args.duration_minutes > 0.0 else None

    run_odyseus_optimization(
        model_root,
        base_profile=base_wp,
        history_path=history_path,
        eval_mode=args.eval_mode,
        iters=args.iters,
        duration_seconds=duration_seconds,
        seed=args.seed,
        n_syn=args.n_syn,
        real_bg_sample=args.real_bg_sample,
        full_real=args.full_real,
        eval_presets_first=not args.no_preset_warmup,
        resume=args.resume,
        save_best=args.save_best,
        verify_top_k=args.verify_top_k,
        export_report=not args.no_report,
        target_profile_path=target_profile,
        focus_groups=tuple(args.focus_groups) if args.focus_groups else None,
    )


if __name__ == "__main__":
    main()
