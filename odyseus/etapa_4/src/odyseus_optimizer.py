"""Odyseus iterative offline parameter optimizer targeting maximum purchase conversion."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
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
    PRESETS,
    SEGMENT_SIGNAL_NAMES,
    USER_SEGMENTS,
    WeightsProfile,
    get_preset,
    list_presets,
    load_active_profile,
)

# ---------------------------------------------------------------------------
# Full tunable parameter space on WeightsProfile
# Spec types:
#   ("float", lo, hi, step, group)
#   ("int", lo, hi, step, group)
#   ("bool", group)
#   ("choice", options, group)
# ---------------------------------------------------------------------------
PARAM_SPECS: dict[str, tuple[Any, ...]] = {
    # 14 Prior Weights
    "w_seq_lift": ("float", 0.20, 2.50, 0.25, "session_core"),
    "w_self_cvr_view": ("float", 0.40, 3.50, 0.30, "session_core"),
    "w_intent_score": ("float", 0.00, 1.60, 0.20, "session_core"),
    "w_same_top_subcat": ("float", 0.00, 0.80, 0.10, "session_core"),
    "w_same_top_brand": ("float", 0.00, 0.40, 0.06, "session_core"),
    "w_user_affinity": ("float", 0.00, 1.00, 0.12, "personalization_price"),
    "w_price_compat": ("float", 0.00, 0.80, 0.10, "personalization_price"),
    "w_log_price_norm": ("float", -0.40, 0.20, 0.08, "personalization_price"),
    "price_std_floor": ("float", 0.20, 1.00, 0.12, "personalization_price"),
    "price_cheap_asymmetry": ("float", 0.00, 1.00, 0.18, "personalization_price"),
    "revenue_tie_weight": ("float", -0.15, 0.15, 0.05, "personalization_price"),
    "tie_break_mode": ("choice", ("conversion", "cheap", "revenue"), "personalization_price"),
    "w_ctx_bayes": ("float", 0.00, 1.20, 0.15, "base_popularity"),
    "w_global_pop": ("float", 0.00, 1.60, 0.22, "base_popularity"),
    "w_live_trend": ("float", 0.00, 1.20, 0.15, "base_popularity"),
    "w_is_new_product": ("float", 0.00, 0.25, 0.04, "cold_start"),
    "w_reverse_graft": ("float", 0.00, 0.90, 0.12, "cold_start"),
    "w_forward_cold": ("float", 0.00, 1.10, 0.15, "cold_start"),
    "bandit_ucb_alpha": ("float", 0.00, 0.35, 0.05, "cold_start"),
    "cold_explore_start_slot_warm": ("int", 3, 9, 2, "cold_start"),
    "max_cold_explore_in_warm": ("int", 0, 4, 1, "cold_start"),
    # Background penalties, floors & MMR diversity
    "v3_no_subcat_floor": ("bool", "diversity_mmr"),
    "subcat_floor_min": ("float", 0.00, 0.60, 0.10, "diversity_mmr"),
    "brand_floor_min": ("float", 0.00, 0.30, 0.06, "diversity_mmr"),
    "v3_bg_penalty": ("bool", "base_popularity"),
    "bg_penalty_diff_subcat": ("float", 0.05, 1.00, 0.18, "base_popularity"),
    "bg_penalty_same_subcat": ("float", 0.10, 1.00, 0.18, "base_popularity"),
    "mmr_start_slot_session": ("int", 2, 10, 2, "diversity_mmr"),
    "mmr_start_slot_home": ("int", 1, 10, 2, "diversity_mmr"),
    "mmr_subcat_lambda_session": ("float", 0.00, 0.40, 0.08, "diversity_mmr"),
    "mmr_subcat_lambda_home": ("float", 0.00, 0.40, 0.08, "diversity_mmr"),
    "mmr_brand_lambda": ("float", 0.00, 0.20, 0.04, "diversity_mmr"),
    "mmr_relative_penalty": ("bool", "diversity_mmr"),
    # Stage 4.1 Business rules
    "v4_cart_priority": ("bool", "business_rules"),
    "v4_cart_active_minutes": ("float", 5.0, 120.0, 20.0, "business_rules"),
    "v4_cart_max_slots": ("int", 1, 5, 1, "business_rules"),
    "v4_forgotten_cart": ("bool", "business_rules"),
    "v4_forgotten_max_slots": ("int", 0, 4, 1, "business_rules"),
    "v4_price_ascending": ("bool", "business_rules"),
    "v4_recurrent": ("bool", "business_rules"),
    "v4_recurrent_max_slots": ("int", 1, 4, 1, "business_rules"),
    "v4_live_event_purchases": ("bool", "business_rules"),
}

# Add active user-segment multipliers as flat virtual keys: seg_<segment>__<signal>
_SEGMENT_ACTIVE_SIGNALS: dict[str, tuple[str, ...]] = {
    "new_with_views": ("seq", "ctx", "pop", "trend", "cold"),
    "new_no_views": ("ctx", "pop", "trend", "cold"),
    "known_with_views": ("seq", "intent", "aff", "ctx", "pop", "trend", "cold"),
    "known_with_history": ("seq", "intent", "aff", "ctx", "pop", "trend", "cold"),
    "known_cold": ("intent", "aff", "ctx", "pop", "trend", "cold"),
}

for _seg in USER_SEGMENTS:
    for _sig in _SEGMENT_ACTIVE_SIGNALS[_seg]:
        PARAM_SPECS[f"seg_{_seg}__{_sig}"] = ("float", 0.20, 2.00, 0.25, "segments")

PARAM_GROUPS: tuple[str, ...] = (
    "business_rules",
    "base_popularity",
    "session_core",
    "personalization_price",
    "cold_start",
    "diversity_mmr",
    "segments",
)


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


def _update_snapshot_for_profile(
    base_snap: RecommenderSnapshotV4,
    wp: WeightsProfile,
    *,
    base_ucb_alpha: float,
) -> tuple[RecommenderSnapshotV4, ConfigV4]:
    """Attach updated WeightsProfile, ConfigV4, ranker weights, and UCB cache in <20ms."""
    cfg = ConfigV4.from_profile(wp, model_version="odyseus-4.1e")
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

    product_static = base_snap.product_static
    overrides = base_snap.agent_overrides
    if abs(wp.bandit_ucb_alpha - base_ucb_alpha) > 1e-9:
        bandit = PositionAwareBandit(
            exam_power=wp.bandit_exam_power,
            prior_strength=wp.bandit_prior_strength,
            cold_ucb_multiplier=wp.bandit_cold_ucb_multiplier,
            warm_ucb_multiplier=wp.bandit_warm_ucb_multiplier,
            click_reward=wp.bandit_click_reward,
            purchase_reward=wp.bandit_purchase_reward,
        )
        product_static = build_product_static_table(
            base_snap.products,
            base_snap.active_ids,
            base_snap.global_pop_norm,
            base_snap.live_trend_norm,
            base_snap.max_log_price,
            bandit,
            base_snap.bandit_state,
            wp.bandit_ucb_alpha,
        )
        overrides = replace(overrides, bandit_ucb_alpha=wp.bandit_ucb_alpha)

    snap = replace(
        base_snap,
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
    df_test = syn["df_test"]
    cold_ids = syn["cold_product_ids"]
    warm_users = syn["warm_user_ids"]
    cold_mask = np.array(
        [is_cold_start_row(row, cold_ids, warm_users) for row in df_test.itertuples(index=False)],
        dtype=bool,
    )
    return {
        "snap": syn["snap_v4"],
        "base_ucb_alpha": float(syn["snap_v4"].agent_overrides.bandit_ucb_alpha),
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


def _propose_candidate(
    best_wp: WeightsProfile,
    rng: np.random.Generator,
    step_idx: int,
    total_iters: int,
    seen_sigs: set[str],
    last_accepted_delta: dict[str, Any] | None,
) -> tuple[WeightsProfile, str, dict[str, dict[str, Any]]]:
    """Generate a diverse, non-duplicate candidate WeightsProfile using multi-phase proposals."""
    frac = step_idx / max(total_iters, 1)

    for attempt in range(25):
        updates: dict[str, Any] = {}

        # Momentum move right after an accepted improvement
        if attempt == 0 and last_accepted_delta and rng.random() < 0.30:
            strategy = "momentum"
            for pname, dval in last_accepted_delta.items():
                spec = PARAM_SPECS[pname]
                if spec[0] == "float":
                    _, lo, hi, _, _ = spec
                    cur = float(get_param_value(best_wp, pname))
                    updates[pname] = round(min(max(cur + float(dval) * 0.65, lo), hi), 4)
        else:
            if frac <= 0.35:
                # Early exploration: cross-group or structural/business-rule moves
                mode = "explore_broad" if rng.random() < 0.55 else f"group:{rng.choice(PARAM_GROUPS)}"
            elif frac <= 0.75:
                # Mid phase: focused group optimization
                mode = f"group:{rng.choice(PARAM_GROUPS)}"
            else:
                # Late phase: fine 1-2 parameter local polish
                mode = "fine_tune"

            strategy = mode
            if mode == "explore_broad":
                pool = list(PARAM_SPECS.keys())
                n_pick = int(rng.integers(3, 7))
                temp = 1.15
            elif mode.startswith("group:"):
                grp = mode.split(":", 1)[1]
                pool = [k for k, v in PARAM_SPECS.items() if v[-1] == grp]
                n_pick = int(rng.integers(2, min(5, len(pool)) + 1))
                temp = max(0.45, 1.0 - 0.5 * frac)
            else:
                pool = list(PARAM_SPECS.keys())
                n_pick = int(rng.integers(1, 3))
                temp = 0.40

            chosen = list(rng.choice(pool, size=min(n_pick, len(pool)), replace=False))
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

    # Fallback single-parameter perturbation if crowded
    pname = str(rng.choice(list(PARAM_SPECS.keys())))
    spec = PARAM_SPECS[pname]
    cur = get_param_value(best_wp, pname)
    if spec[0] == "float":
        _, lo, hi, _, _ = spec
        cand_val = round(float(rng.uniform(lo, hi)), 4)
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
) -> dict[str, Any]:
    """Run Odyseus iterative parameter search and persist every iteration to a single archive file."""
    repo_root = model_root.parent
    off_dir = offline_cache_dir or (repo_root / "offline_eval_cache")

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

        archive = {
            "model": "odyseus-4.1e",
            "eval_mode": eval_mode,
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
            "iterations": iterations,
        }
        _atomic_write_json(history_path, archive)

        tag = "BEST*" if accepted else "     "
        print(f"[iter {iter_num:03d}] {tag} [{strategy:<22}] {_compact_metrics_line(res)}")
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
        cand_wp, strategy, changed_diff = _propose_candidate(
            best_wp,
            rng,
            step,
            iters,
            seen_sigs,
            last_accepted_delta,
        )
        res_c = evaluate_candidate(cand_wp, eval_mode=eval_mode, real_bundle=real_bundle, syn_bundle=syn_bundle)
        prev_best = best_wp
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
            f"Odyseus best profile (iter={best_iter}, mode={eval_mode}, "
            f"obj={best_obj:.5f} vs baseline={baseline_obj:.5f})."
        ),
    )

    if save_best and target_profile_path is not None:
        final_wp.save_json(target_profile_path)
        print(f"[ODYSEUS] Saved optimal profile '{final_wp.profile_name}' -> {target_profile_path}")

    print(
        f"\n=== Odyseus Optimization Summary ===\n"
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
    parser = argparse.ArgumentParser(description="Odyseus Offline Iterative Conversion Optimizer")
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
        help="Skip initial built-in preset sweep before hill-climbing",
    )
    parser.add_argument("--resume", action="store_true", help="Resume appending to existing history archive")
    parser.add_argument(
        "--history-file",
        type=str,
        default="",
        help="Path to single JSON archive file (default: odyseus/results/odyseus_history.json)",
    )
    parser.add_argument("--save-best", action="store_true", help="Write best profile to weights_profile.json")
    args = parser.parse_args()

    model_root = Path(__file__).resolve().parents[2]
    cfg = ConfigV4()
    target_profile = cfg.resolve_weights_profile_path()
    base_wp = get_preset(args.preset) if args.preset else load_active_profile(target_profile)
    history_path = (
        Path(args.history_file)
        if args.history_file
        else (model_root / "results" / "odyseus_history.json")
    )

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
    )


if __name__ == "__main__":
    main()
